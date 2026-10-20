"""Guarded vertical extraction using full carried geometry and normal controls."""

def extraction_height(bounds, obstacles, margin=.02, neighborhood=.10):
    """Clear the support and nearby low clutter before free-space planning.

    Tall barriers are handled by the collision checker, not by trying to lift
    above a wall. This is an extraction waypoint, not a replacement path plan.
    """
    low, high = bounds
    target = margin
    for other_low, other_high in obstacles:
        near = all(other_high[i] >= low[i]-neighborhood and other_low[i] <= high[i]+neighborhood
                   for i in (0, 1))
        low_clutter = other_low[2] <= high[2]+margin and other_high[2] <= low[2]+.18
        if near and low_clutter:
            target = max(target, other_high[2]-low[2]+margin)
    return target


def guarded_lift(controller, obj, record=lambda **fields: None):
    import torch as th
    from omnigibson.utils import transform_utils as T
    from omnigibson.action_primitives.curobo import create_world_mesh_collision
    from curobo.geom.types import WorldConfig
    from task_undo.simulation.contact_retreat import CollisionView, bounded_joint_step, admissible
    from task_undo.simulation.rotation import rotation_vector

    c, robot = controller, controller.robot
    if c._get_obj_in_hand() is not obj:
        raise RuntimeError('Extraction requires the intended object in hand')
    view = CollisionView(c)
    q0 = robot.get_joint_positions()
    js = view.JointState(position=view.planner.tensor_args.to_device(q0[None]),
                         joint_names=view.planner.robot_joint_names)
    view.planner.update_locked_joints(js, view.emb)
    info = view.planner._attach_objects_to_robot(
        {robot.eef_link_names[c.arm]: obj.root_link},
        {robot.eef_link_names[c.arm]: 1.0},
        js.get_ordered_joint_state(view.mg.kinematics.joint_names), view.emb)
    try:
        initial, self_cost = view.measure(q0)
        if self_cost > 0:
            raise RuntimeError('Extraction cannot resolve self collision')
        # Only an observed support below the object may carry initial contact.
        # A separate checker attributes contact without deleting any obstacles.
        bottom, top = obj.aabb
        supports = []
        obstacles = []
        for other in robot.scene.objects:
            if other is obj or other is robot:
                continue
            lo, hi = other.aabb
            obstacles.append((lo.tolist(), hi.tolist()))
            if (abs(float(hi[2]-bottom[2])) <= .015
                    and bool(th.all(hi[:2] >= bottom[:2])) and bool(th.all(lo[:2] <= top[:2]))):
                supports.append(other)
        target_rise = extraction_height((bottom.tolist(), top.tolist()), obstacles)
        if bool((initial > 0).any()):
            world = view.checker.world_model
            if isinstance(world, list):
                world = world[0]
            separate = create_world_mesh_collision(view.planner.tensor_args)
            meshes = [m for m in world.mesh if any(m.name.startswith(o.prim_path+'/') for o in supports)]
            if not meshes:
                raise RuntimeError('Initial extraction collision has no observed support')
            # Query forbidden obstacles directly. Subtracting an aggregate
            # world cost from a support-only cost is not an attribution test:
            # aggregation and numeric roundoff can hide or invent differences.
            support_names = {m.name for m in meshes}
            attached_names = {name for item in info for name in item['obj_paths']}
            forbidden_meshes = [m for m in world.mesh if m.name not in support_names | attached_names]
            separate.load_collision_model(WorldConfig(mesh=forbidden_meshes))
            link = robot.curobo_attached_object_link_names[robot.eef_link_names[c.arm]]
            ids = set(view.mg.kinematics.kinematics_config.get_sphere_index_from_link_name(link).tolist())
            if any(i not in ids for i in th.where(initial > 0)[0].tolist()):
                raise RuntimeError('Initial contact is not confined to held object and its support')
            view.forbidden_checker, view.allowed_spheres = separate, ids
            view.measure(q0)  # Refuse any non-support collision before motion.
        start, orientation = robot.eef_links[c.arm].get_position_orientation()
        start, orientation = start.clone(), orientation.clone()
        if not bool(th.isfinite(start).all() and th.isfinite(orientation).all()):
            raise RuntimeError('Extraction requires a finite measured grasp pose')
        # Native acquisition may stop an approach before the requested wrist
        # attitude is reached. Preserve that measured attitude while lifting;
        # wrist verticality is not a certificate of payload clearance. The
        # full held geometry, support attribution and per-step corridor/cost
        # checks below decide whether upward extraction is feasible.
        root_orientation = robot.root_link.get_position_orientation()[1]
        world_to_root = T.quat2mat(root_orientation).T
        upward = world_to_root @ th.tensor([0., 0., 1.])
        # The regular manipulation solver also uses the lift trunk. Freezing
        # it needlessly forces a near-extended arm to supply all vertical motion.
        indices = th.cat([robot.trunk_control_idx, robot.arm_control_idx[c.arm]])
        record(event='lift_started', costs=initial.tolist(), supports=[o.name for o in supports],
               active_joint_indices=indices.tolist(), target_rise_m=target_rise,
               measured_orientation=orientation.tolist(),
               wrist_vertical_alignment=abs(float(T.quat2mat(orientation)[2, 2])))
        for step in range(400):
            if c._get_obj_in_hand() is not obj:
                raise RuntimeError('Grasp lost during extraction')
            position, quaternion = robot.eef_links[c.arm].get_position_orientation()
            rise = float(position[2]-start[2])
            lateral_error = float(th.norm((position-start)[:2]))
            angular_error = float(T.get_orientation_diff_in_radian(quaternion,orientation))
            record(event='lift_pose', step=step, rise_m=rise, lateral_error_m=lateral_error,
                   angular_error_rad=angular_error, position=position.tolist(), quaternion=quaternion.tolist())
            if lateral_error > .01 or angular_error > .035:
                raise RuntimeError('Extraction left its vertical corridor')
            q = robot.get_joint_positions()
            costs, self_cost = view.measure(q)
            record(event='lift_measurement', step=step, joint_positions=q.tolist(),
                   costs=costs.tolist(), self_cost=self_cost)
            if (self_cost > 0 or float(costs.max()) > float(initial.max())+.0005
                    or float(costs.sum()) > float(initial.sum())+.0005):
                raise RuntimeError('Observed extraction collision exceeded initial contact')
            record(event='lift_observed', step=step, rise_m=rise, costs=costs.tolist())
            if rise >= target_rise and float(costs.max()) <= 1e-7:
                record(event='lift_completed', steps=step, rise_m=rise)
                return
            if rise > target_rise+.01:
                raise RuntimeError('Extraction exceeded its travel bound')
            state = view.state(q)
            pos = state.ee_pos_seq.reshape(-1,3)[0].cpu()
            quat = state.ee_quat_seq.reshape(-1,4)[0][[1,2,3,0]].cpu()
            jac = th.zeros((6,len(indices)))
            for j,index in enumerate(indices):
                sample = q.clone(); sample[index] += .0001
                perturbed = view.state(sample)
                pq = perturbed.ee_quat_seq.reshape(-1,4)[0][[1,2,3,0]].cpu()
                jac[:3,j] = (perturbed.ee_pos_seq.reshape(-1,3)[0].cpu()-pos)/.0001
                jac[3:,j] = rotation_vector(T.quat_multiply(pq,T.quat_conjugate(quat)))/.0001
            # Close the loop on measured link pose, not only forward kinematics:
            # articulation tracking and grasp constraints need not follow FK exactly.
            lateral = start-position
            lateral[2] = 0.
            rotation = rotation_vector(T.quat_multiply(orientation,T.quat_conjugate(quaternion)))
            # One millimetre can be smaller than the passive base's settling
            # motion during a control step. Command a bounded 3 mm rise and
            # taper at the waypoint. The measured contact/corridor guards and
            # per-sphere candidate check still decide whether motion is safe.
            advance = min(.003, max(.001, target_rise-rise))
            delta = th.cat([upward*advance + (world_to_root @ lateral)*.5,
                           (world_to_root @ rotation)*.5])
            lower, upper = robot.joint_lower_limits[indices], robot.joint_upper_limits[indices]
            dq = bounded_joint_step(jac, delta,
                th.maximum(lower-q[indices],th.full_like(lower,-.01)),
                th.minimum(upper-q[indices],th.full_like(upper,.01)))
            candidate = q.clone(); candidate[indices] = th.clamp(q[indices]+dq, lower, upper)
            proposed, self_cost = view.measure(candidate)
            record(event='lift_command', step=step, joint_positions=candidate.tolist(),
                   delta=delta.tolist(), joint_delta=dq.tolist(),
                   costs=proposed.tolist(), self_cost=self_cost,
                   jacobian=jac.tolist(), active_joint_indices=indices.tolist())
            if self_cost > 0 or not admissible(costs.tolist(), proposed.tolist()):
                raise RuntimeError('No non-worsening collision-checked extraction step')
            yield c._postprocess_action(robot.q_to_action(candidate))
        raise RuntimeError('Extraction exceeded 400 control steps')
    finally:
        view.planner._detach_objects_from_robot(info, view.emb)


class GuardedGraspLift:
    def _record_grasp_lift(self, **fields):
        pass

    def _grasp(self, obj):
        yield from super()._grasp(obj)
        yield from guarded_lift(self, obj, self._record_grasp_lift)

    def _settle_loaded_joints(self):
        from task_undo.simulation.joint_settling import settle_joint_interior
        yield from settle_joint_interior(self, self._record_grasp_lift)

    def _reset_robot(self):
        try:
            if self._get_obj_in_hand() is not None:
                yield from self._settle_loaded_joints()
            yield from super()._reset_robot()
        except Exception as exc:
            # Upstream cleanup suppresses primitive errors. A failed loaded
            # reset must propagate and cannot become a successful grasp.
            if self._get_obj_in_hand() is not None:
                raise RuntimeError('Loaded-arm retraction did not complete') from exc
            raise
