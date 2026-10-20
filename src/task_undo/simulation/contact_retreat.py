"""Bounded vertical exit for a released upright mug, before free-space planning.

All physical motion uses normal joint commands. Existing finger/mug contact is
allowed only during escape; other collision pairs remain disallowed. No object
is removed from the planner world, and no physical state is assigned here.
"""


def admissible(previous, candidate, allowance=0.0002):
    """Per-sphere penetration cannot grow; a clear sphere must remain clear."""
    import math
    if len(previous) != len(candidate):
        return False
    return all(math.isfinite(a) and math.isfinite(b) and b <= (a + allowance if a > 0 else 1e-7)
               for a, b in zip(previous, candidate))


def within_selected_limits(q, lower, upper, indices):
    import math
    return all(math.isfinite(q[i]) and lower[i] <= q[i] <= upper[i] for i in indices)


class ContactViolation(RuntimeError):
    """A geometric candidate touches something outside its contact permission."""


def inside_contact_envelope(initial, observed, allowance=.0005):
    import math
    return (bool(initial) and len(initial) == len(observed)
            and all(math.isfinite(x) and x >= 0 for x in (*initial, *observed))
            and max(observed) <= max(initial) + allowance
            and sum(observed) <= sum(initial) + allowance)


def release_samples(q, arm_target, effective_target, fingers, speed_limits, dt):
    """Sample a control interval with independent, velocity-bounded finger opening.

    The open *command* stays at its limit to preserve release intent. The
    predicted finger positions are bounded by distance achievable in one
    control interval, rather than pretending that the goal is reached at once.
    These are finite geometric samples, not a dynamics or swept-volume proof.
    """
    import itertools
    import math
    import torch as th
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Positive finite control interval required')
    values = th.cat((q, arm_target, effective_target, speed_limits[fingers]))
    if not bool(th.isfinite(values).all()) or bool((speed_limits[fingers] < 0).any()):
        raise ValueError('Invalid joint state or finger velocity limits')
    gap = effective_target[fingers] - q[fingers]
    if bool((gap < -1e-6).any()):
        raise ValueError('Contact exit cannot command closing fingers')
    travel = th.minimum(gap.clamp(min=0), speed_limits[fingers] * dt)
    # The fingers need not open at equal rates: one may still touch the rim.
    for fraction in (.5, 1.):
        for weights in itertools.product((0., .5, 1.), repeat=len(fingers)):
            sample = q + (arm_target-q) * fraction
            sample[fingers] = q[fingers] + travel * fraction * q.new_tensor(weights)
            yield sample


def checked_samples(view, samples, costs, initial):
    """Check hypothetical geometry; backend faults must propagate."""
    import math
    count, worst = 0, costs.clone()
    try:
        for sample in samples:
            observed, self_cost = view.measure(sample)
            count += 1
            if (not math.isfinite(self_cost) or self_cost > 0
                    or not admissible(costs.tolist(), observed.tolist())
                    or not inside_contact_envelope(initial.tolist(), observed.tolist())):
                return None, count
            worst = worst.maximum(observed)
    except ContactViolation:
        return None, count
    return worst, count


def checked_release_candidate(view, q, candidate, effective, fingers, speeds, dt, costs, initial):
    return checked_samples(view, release_samples(q, candidate, effective, fingers, speeds, dt),
                           costs, initial)


def hold_step_budget(window, control_dt):
    """Leave two complete control intervals before the native grasp window."""
    import math
    if not all(math.isfinite(v) and v > 0 for v in (window, control_dt)):
        raise ValueError('Positive finite timing required')
    return max(0, math.floor(window / control_dt + 1e-9) - 2)


def hold_aperture_action(controller, q):
    """Apply a measured aperture using ordinary gripper position controls.

    The simulator treats a partial aperture command as grasp intent. Callers
    therefore bound this phase below its native acquisition window and check
    observed grasp state. No grasp counters or constraints are assigned here.
    """
    command = controller.robot.q_to_action(q)
    action = controller._postprocess_action(command.clone())
    indices = controller.robot.controller_action_idx[f'gripper_{controller.arm}']
    action[indices] = command[indices]
    return action


def bounded_joint_step(jac, delta, lower, upper):
    """Damped least squares with saturated joints fixed at their step bounds."""
    import torch as th
    step = th.zeros(jac.shape[1], dtype=jac.dtype, device=jac.device)
    free = th.ones(jac.shape[1], dtype=th.bool, device=jac.device)
    for _ in range(jac.shape[1] + 1):
        if not bool(free.any()):
            break
        indices = th.where(free)[0]
        matrix = jac[:, indices]
        residual = delta - jac @ step
        proposal = matrix.T @ th.linalg.solve(
            matrix @ matrix.T + .02**2 * th.eye(jac.shape[0],device=jac.device,dtype=jac.dtype), residual)
        outside = (proposal < lower[indices]) | (proposal > upper[indices])
        if not bool(outside.any()):
            step[indices] = proposal
            break
        fixed = indices[outside]
        step[fixed] = th.clamp(proposal[outside],lower[fixed],upper[fixed])
        free[fixed] = False
    return step


class CollisionView:
    """Same ARM geometry and non-ESDF query used by the planner constraint."""
    def __init__(self, controller):
        from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection as Emb
        from curobo.geom.sdf.world import CollisionQueryBuffer
        from curobo.types.state import JointState
        self.JointState, self.Buffer, self.emb = JointState, CollisionQueryBuffer, Emb.ARM
        self.planner = controller._motion_generator
        self.mg = self.planner.mg[self.emb]
        self.robot = controller.robot
        self.planner.update_obstacles()
        self.checker = self.mg.rollout_fn.primitive_collision_constraint.world_coll_checker
        self.forbidden_checker = None
        self.last_forbidden_costs = None
        self.allowed_spheres = None
        self.allowed_meshes = None

    def restrict_contacts_to(self, mesh_names, sphere_ids):
        """Check forbidden obstacles directly, without subtracting world costs.

        The main planner world stays intact. A second checker contains every
        mesh except the explicitly permitted contact partner; this also catches
        a small forbidden collision hidden by a larger permitted one.
        """
        from omnigibson.action_primitives.curobo import create_world_mesh_collision
        world = self.checker.world_model
        if isinstance(world, list):
            world = world[0]
        names = set(mesh_names)
        if not names or not names <= {mesh.name for mesh in world.mesh}:
            raise ValueError('Contact permission must identify existing obstacle meshes')
        separate = create_world_mesh_collision(self.planner.tensor_args)
        self.forbidden_checker = separate
        self.allowed_spheres = set(sphere_ids)
        self.allowed_meshes = names
        self._sync_forbidden_world()

    def _sync_forbidden_world(self):
        from curobo.geom.types import WorldConfig
        world = self.checker.world_model
        if isinstance(world, list):
            world = world[0]
        if not self.allowed_meshes <= {mesh.name for mesh in world.mesh}:
            raise ValueError('Permitted contact meshes disappeared during exit')
        self.forbidden_checker.load_collision_model(WorldConfig(
            mesh=[mesh for mesh in world.mesh if mesh.name not in self.allowed_meshes]))

    def refresh(self):
        """Synchronize both worlds at a control boundary, before any candidates.

        The planner converts current mesh poses into its current root frame.
        Candidate queries then share this fixed observation; they must not
        refresh the world while differentiating hypothetical joint positions.
        """
        self.planner.update_obstacles()
        if self.forbidden_checker is not None:
            self._sync_forbidden_world()

    def state(self, q):
        p = self.planner
        js = self.JointState(position=p.tensor_args.to_device(q[None]), joint_names=p.robot_joint_names)
        p.update_locked_joints(js, self.emb)
        return self.mg.compute_kinematics(js.get_ordered_joint_state(self.mg.kinematics.joint_names))

    def measure(self, q):
        import math
        state = self.state(q)
        spheres = state.robot_spheres.unsqueeze(1)
        p = self.planner
        buf = self.Buffer.initialize_from_shape(spheres.shape, p.tensor_args, self.checker.collision_types)
        costs = self.checker.get_sphere_collision(spheres, buf, p.tensor_args.to_device([1.]),
                    activation_distance=p.tensor_args.to_device([0.])).flatten().detach().cpu()
        if not bool(costs.isfinite().all()) or bool((costs < 0).any()):
            raise RuntimeError('Invalid world collision measurement')
        if self.forbidden_checker is not None:
            checker = self.forbidden_checker
            buf = self.Buffer.initialize_from_shape(spheres.shape, p.tensor_args, checker.collision_types)
            forbidden = checker.get_sphere_collision(spheres, buf, p.tensor_args.to_device([1.]),
                activation_distance=p.tensor_args.to_device([0.])).flatten().detach().cpu()
            self.last_forbidden_costs = forbidden
            if not bool(forbidden.isfinite().all()) or bool((forbidden < 0).any()):
                raise RuntimeError('Invalid forbidden collision measurement')
            if bool((forbidden > 1e-7).any()):
                raise ContactViolation(f'Extraction would contact a forbidden obstacle: {float(forbidden.max())}')
        if self.allowed_spheres is not None and any(i not in self.allowed_spheres for i in (costs>0).nonzero().flatten().tolist()):
            raise ContactViolation('Contact exit collision reached a non-finger link')
        self_cost = float(self.mg.rollout_fn.robot_self_collision_constraint.forward(spheres).max())
        if not math.isfinite(self_cost) or self_cost < 0:
            raise RuntimeError('Invalid self collision measurement')
        return costs, self_cost


def vertical_mug_exit(controller, obj, record=lambda **fields: None):
    """Try at most 5 cm / 150 control steps of a measured, guarded upward exit."""
    import torch as th
    import omnigibson as og
    from omnigibson.utils import transform_utils as T
    from task_undo.simulation.retraction_checkpoint import released
    from task_undo.simulation.rotation import rotation_vector

    c, robot = controller, controller.robot
    if not c._keep_released or not released(robot):
        raise ValueError('Contact exit requires release intent and empty hands')
    if obj.category != 'mug':
        raise ValueError('Vertical contact exit is validated only for upright mugs')
    obj_pos, obj_ori = obj.get_position_orientation()
    obj_pos, obj_ori = obj_pos.clone(), obj_ori.clone()
    view = CollisionView(c)
    if view.planner.ee_link[view.emb] != robot.eef_link_names[c.arm]:
        raise ValueError('Contact exit requires the primary planner end effector')
    q0 = robot.get_joint_positions()
    initial, self_cost = view.measure(q0)
    if self_cost > 0:
        raise ValueError('Contact exit cannot resolve self collision')
    if not bool((initial > 0).any()):
        record(event='exit_not_needed')
        return
    if float(T.quat2mat(obj_ori)[2, 2]) < .98:
        raise ValueError('Mug is not upright')
    # Refuse other links or obstacles: exact per-object attribution with a
    # separate checker leaves every obstacle enabled in the motion planner.
    world = view.checker.world_model
    if isinstance(world, list):
        world = world[0]
    mug_meshes = {mesh.name for mesh in world.mesh if mesh.name.startswith(obj.prim_path+'/')}
    kc = view.mg.kinematics.kinematics_config
    finger_ids = {int(i) for link in robot.finger_links[c.arm]
                  for i in kc.get_sphere_index_from_link_name(link.body_name).tolist()}
    view.restrict_contacts_to(mug_meshes, finger_ids)
    initial, self_cost = view.measure(q0)  # Refuse forbidden or non-finger contact before motion.
    if self_cost > 0:
        raise ValueError('Contact exit cannot resolve self collision')
    start_pos, start_ori = robot.eef_links[c.arm].get_position_orientation()
    start_pos, start_ori = start_pos.clone(), start_ori.clone()
    local = view.state(q0)
    target_quat = local.ee_quat_seq.reshape(-1,4)[0][[1,2,3,0]].cpu().clone()
    root_ori = robot.root_link.get_position_orientation()[1]
    up_local = T.quat2mat(root_ori).T @ th.tensor([0.,0.,1.])
    arm_idx = robot.arm_control_idx[c.arm]
    record(event='exit_started', collision_costs=initial.tolist(), start_position=start_pos.tolist(),
           joint_positions=q0.tolist(), lower_limits=robot.joint_lower_limits.tolist(),
           upper_limits=robot.joint_upper_limits.tolist(), active_arm_indices=arm_idx.tolist(),
           permitted_contact_meshes=sorted(mug_meshes),
           forbidden_collision_max=float(view.last_forbidden_costs.max()))
    from omnigibson.robots.robot import m as robot_macros
    dt = og.sim.get_sim_step_dt()
    hold_budget = hold_step_budget(robot_macros.GRASP_WINDOW, dt)
    if any(robot._ag_grasp_counter[a] is not None for a in robot.arm_names):
        raise RuntimeError('Aperture hold requires no pending grasp window')
    fingers = robot.gripper_control_idx[c.arm]
    held_aperture = q0[fingers].clone()
    holding, hold_steps, open_steps = True, 0, 0
    record(event='aperture_hold_started', max_control_steps=hold_budget,
           finger_targets=held_aperture.tolist(), native_grasp_window_s=robot_macros.GRASP_WINDOW)
    clear_steps = 0
    for step in range(150):
        if not released(robot):
            raise RuntimeError('Unexpected grasp during contact exit')
        p, ori = obj.get_position_orientation()
        displacement = float(th.norm(p-obj_pos))
        rotation = float(T.get_orientation_diff_in_radian(ori,obj_ori))
        if displacement > .005 or rotation > .0872664626:
            raise RuntimeError('Released mug moved beyond 5 mm / 5 degree exit guard')
        eef, eef_ori = robot.eef_links[c.arm].get_position_orientation()
        rise = float(eef[2]-start_pos[2])
        if float(th.norm((eef-start_pos)[:2])) > .005 or float(T.get_orientation_diff_in_radian(eef_ori,start_ori)) > .035:
            raise RuntimeError('Contact exit deviated from its vertical corridor')
        q = robot.get_joint_positions()
        if step:
            view.refresh()
        costs, self_cost = view.measure(q)
        record(event='exit_observed', step=step, rise_m=rise, mug_displacement_m=displacement,
               collision_costs=costs.tolist(), self_collision_cost=self_cost,
               forbidden_collision_max=float(view.last_forbidden_costs.max()),
               joint_positions=q.tolist(), joint_velocities=robot.get_joint_velocities().tolist(),
               object_position=p.tolist(), object_orientation=ori.tolist(),
               aperture_mode='hold' if holding else 'open',
               native_grasp_counter=robot._ag_grasp_counter[c.arm])
        if self_cost > 0 or not inside_contact_envelope(initial.tolist(), costs.tolist()):
            raise RuntimeError('Measured collision exceeded the initial contact envelope')
        from omnigibson.utils.usd_utils import RigidContactAPI
        finger_paths = {link.prim_path for link in robot.finger_links[c.arm]}
        covered = all(path in RigidContactAPI._PATH_TO_ROW_IDX.get(robot.scene.idx,{}) for path in finger_paths)
        pairs = RigidContactAPI.get_contact_pairs(scene_idx=robot.scene.idx,query_set=finger_paths,
                                                  with_set=None,current_only=True)
        touching_mug = any(path == obj.prim_path or path.startswith(obj.prim_path+'/')
                           for pair in pairs for path in pair)
        clear_steps = clear_steps+1 if rise >= .005 and float(costs.max()) <= 1e-7 and covered and not touching_mug else 0
        if holding and clear_steps >= 2:
            holding = False
            record(event='aperture_open_started', held_control_steps=hold_steps)
        if clear_steps >= 2 and open_steps >= 2:
            record(event='exit_completed', steps=step, rise_m=rise, mug_displacement_m=displacement,
                   mug_rotation_rad=rotation, collision_costs=costs.tolist())
            return
        if rise > .05:
            raise RuntimeError('Contact exit reached its 5 cm travel limit')
        if holding:
            counter = robot._ag_grasp_counter[c.arm]
            elapsed = (counter or 0) * og.sim.get_physics_dt()
            if hold_steps >= hold_budget or elapsed + dt >= robot_macros.GRASP_WINDOW:
                raise RuntimeError('Aperture hold exhausted its bounded acquisition-free interval')
        # Finite-difference FK uses hypothetical joints only; no simulator pose
        # edits. Only the active arm is commanded, trunk and other arm stay put.
        state = view.state(q)
        pos = state.ee_pos_seq.reshape(-1,3)[0].cpu()
        quat = state.ee_quat_seq.reshape(-1,4)[0][[1,2,3,0]].cpu()
        jac = th.zeros((6,len(arm_idx)))
        eps = .0001
        for j,index in enumerate(arm_idx):
            sample = q.clone(); sample[index] += eps
            perturbed = view.state(sample)
            pq = perturbed.ee_quat_seq.reshape(-1,4)[0][[1,2,3,0]].cpu()
            jac[:3,j]=(perturbed.ee_pos_seq.reshape(-1,3)[0].cpu()-pos)/eps
            jac[3:,j]=rotation_vector(T.quat_multiply(pq,T.quat_conjugate(quat)))/eps
        lower = robot.joint_lower_limits[arm_idx]
        upper = robot.joint_upper_limits[arm_idx]
        fingers = robot.gripper_control_idx[c.arm]
        speeds = robot.max_joint_velocities
        dt = og.sim.get_sim_step_dt()
        proposed = None
        # A 1 mm arm move can be overtaken by the 0.05 m/s finger opening.
        # Compare bounded alternatives against the same contact observation;
        # no new action is issued for a rejected candidate.
        for advance in ((.003, .002, .001) if holding else (.001, .002, .003)):
            delta = th.cat([up_local*advance,
                           rotation_vector(T.quat_multiply(target_quat,T.quat_conjugate(quat)))])
            dq = bounded_joint_step(jac, delta, th.maximum(lower-q[arm_idx],th.full_like(lower,-.01)),
                                   th.minimum(upper-q[arm_idx],th.full_like(upper,.01)))
            candidate = q.clone()
            candidate[arm_idx] = th.clamp(q[arm_idx]+dq,lower,upper)
            if not within_selected_limits(candidate.tolist(), robot.joint_lower_limits.tolist(),
                                          robot.joint_upper_limits.tolist(), arm_idx.tolist()):
                raise RuntimeError('Contact exit would exceed a joint limit')
            effective = candidate.clone()
            effective[fingers] = (held_aperture if holding else
                                  c._get_joint_position_with_fingers_at_limit('upper')[fingers])
            if holding:
                # Bound the geometric hypothesis between observed fingers and
                # their measured hold target, with the same arm interpolation.
                samples_q = []
                for fraction in (.5, 1.):
                    for finger_fraction in (0., 1.):
                        sample = q + (candidate-q)*fraction
                        sample[fingers] = q[fingers] + (held_aperture-q[fingers])*finger_fraction
                        samples_q.append(sample)
                proposed, samples = checked_samples(view, samples_q, costs, initial)
            else:
                proposed, samples = checked_release_candidate(
                    view, q, candidate, effective, fingers, speeds, dt, costs, initial)
            record(event='exit_candidate', step=step, advance_m=advance, samples=samples,
                   accepted=proposed is not None, control_dt=dt,
                   finger_speed_limits=speeds[fingers].tolist())
            if proposed is not None:
                break
        if proposed is None:
            raise RuntimeError('No non-worsening coordinated release step')
        record(event='exit_step', step=step, rise_m=rise, mug_displacement_m=displacement,
               collision_costs=costs.tolist(), proposed_collision_costs=proposed.tolist(),
               proposed_joint_positions=candidate.tolist(), effective_joint_targets=effective.tolist())
        if holding:
            hold_steps += 1
            yield hold_aperture_action(c, effective)
        else:
            open_steps += 1
            yield c._postprocess_action(robot.q_to_action(effective))
    raise RuntimeError('Contact exit exceeded 150 control steps')


class ContactReleaseRetreat:
    """Run the guarded exit only for a mug just released by this PLACE."""
    _released_object = None

    def apply_ref(self, primitive, *args, **kwargs):
        previous = self._released_object
        self._released_object = None
        try:
            return (yield from super().apply_ref(primitive, *args, **kwargs))
        finally:
            self._released_object = previous

    def _execute_release(self):
        obj = self._get_obj_in_hand()
        value = yield from super()._execute_release()
        if self._placing and obj is not None:
            self._released_object = obj
        return value

    def _record_contact_exit(self, **fields):
        pass  # Optional diagnostic hook; normal execution has no log dependency.

    def _reset_robot(self):
        obj = self._released_object
        if not (self._placing and self._keep_released and obj is not None and obj.category == 'mug'):
            return (yield from super()._reset_robot())
        yield from vertical_mug_exit(self, obj, self._record_contact_exit)
        try:
            return (yield from super()._reset_robot())
        except Exception as exc:
            # Upstream apply_ref suppresses ActionPrimitiveError during cleanup.
            # A mug that was not retracted must not silently pass this path.
            raise RuntimeError('Post-release mug retraction did not complete') from exc
