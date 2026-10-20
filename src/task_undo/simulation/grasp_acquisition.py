"""Maintain target contact until native sticky acquisition completes.

The native contact window can outlast a waypoint's tracking budget or a completed
approach. Holding
the last commanded configuration avoids advancing to later waypoints while
that window runs. Only ordinary controls are issued; grasp state is never assigned.
"""


def acquire_after_contact(actions, is_stall, touching, acquire):
    """Hold on observed contact after approach completion or a tracking stall."""
    try:
        result = yield from actions
    except Exception as error:
        if not is_stall(error) or not touching():
            raise
        actions.close()
        yield from acquire(error)
    else:
        if touching():
            actions.close()
            yield from acquire(None)
        return result
    finally:
        actions.close()


class ContactGraspAcquisition:
    _acquisition_target = None

    def _record_grasp_acquisition(self, **fields):
        pass

    def _move_hand(self, *args, **kwargs):
        previous = self._acquisition_target
        objects = kwargs.get('ignore_objects') or []
        self._acquisition_target = (objects[0] if kwargs.get('stop_on_ag')
                                    and len(objects) == 1
                                    and self.robot.grasping_mode == 'sticky' else None)
        try:
            return (yield from super()._move_hand(*args, **kwargs))
        finally:
            self._acquisition_target = previous

    def _acquisition_contacts(self):
        from omnigibson.utils.usd_utils import RigidContactAPI
        return {frozenset(pair) for pair in RigidContactAPI.get_contact_pairs(
            scene_idx=self.robot.scene.idx, query_set={self.robot},
            with_set=None, current_only=True)}

    def _execute_motion_plan(self, *args, **kwargs):
        obj = self._acquisition_target
        if obj is None or not kwargs.get('stop_on_ag'):
            return (yield from super()._execute_motion_plan(*args, **kwargs))
        from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
        trajectory = args[0] if args else kwargs['q_traj']
        last = {}
        def remembered():
            for q in trajectory:
                last['q'] = q.clone()
                yield q
        if args:
            args = (remembered(), *args[1:])
        else:
            kwargs = {**kwargs, 'q_traj': remembered()}
        actions = super()._execute_motion_plan(*args, **kwargs)
        fingers = {link.prim_path for link in self.robot.finger_links[self.arm]}
        target = set(obj.link_prim_paths)
        allowed = {frozenset((finger, link)) for finger in fingers for link in target}
        baseline = self._acquisition_contacts()
        touching = lambda: (self._get_obj_in_hand() is None and 'q' in last
                            and bool(self._acquisition_contacts() & allowed))
        def is_stall(error):
            return (isinstance(error, ActionPrimitiveError)
                    and error.reason == ActionPrimitiveError.Reason.EXECUTION_ERROR
                    and 'Could not reach the target articulation joint positions' in str(error))
        def acquire(error):
            self._record_grasp_acquisition(
                event='approach_contact_stall' if error else 'approach_contact_pending',
                error=str(error) if error else None)
            yield from self._wait_for_acquisition(obj, baseline, allowed, last['q'])
        yield from acquire_after_contact(actions, is_stall, touching, acquire)

    def _wait_for_acquisition(self, obj, baseline, allowed, commanded_q):
        import math
        import torch as th
        import omnigibson as og
        from omnigibson.robots.robot import m as robot_macros
        from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
        from task_undo.simulation.rotation import rotation_vector
        from omnigibson.utils import transform_utils as T

        def fail(message):
            raise ActionPrimitiveError(ActionPrimitiveError.Reason.EXECUTION_ERROR, message)

        robot = self.robot
        # Keep the already issued arm target. Replacing it by a measured pose
        # removes the contact preload and caused a book to lose contact in v1.
        # Base stays at its measured location; no later waypoint is requested.
        q = commanded_q.clone()
        q[robot.base_control_idx] = robot.get_joint_positions()[robot.base_control_idx]
        closed = self._get_joint_position_with_fingers_at_limit('lower')
        q[robot.gripper_control_idx[self.arm]] = closed[robot.gripper_control_idx[self.arm]]
        position, orientation = robot.eef_links[self.arm].get_position_orientation()
        position, orientation = position.clone(), orientation.clone()
        # Original 0.3 s window plus two control ticks for observation ordering.
        budget = math.ceil(robot_macros.GRASP_WINDOW / og.sim.get_sim_step_dt()) + 2
        self._record_grasp_acquisition(event='contact_hold_started', object=obj.name,
                                       max_control_steps=budget, target_q=q.tolist())
        for step in range(budget + 1):
            held = self._get_obj_in_hand()
            contacts = self._acquisition_contacts()
            # Existing floor/support contacts may remain. New contacts must be
            # confined to the intended object's finger contact.
            if contacts - baseline - allowed:
                fail('Unexpected robot contact while waiting for grasp acquisition')
            if any(obj.prim_path + '/' in path for pair in contacts - allowed for path in pair):
                fail('Target contacted a non-finger link during grasp acquisition')
            p, o = robot.eef_links[self.arm].get_position_orientation()
            translation = float(th.linalg.norm(p-position))
            rotation = float(th.linalg.norm(rotation_vector(T.quat_multiply(o, T.quat_conjugate(orientation)))))
            if translation > .01 or rotation > .035:
                fail('Contact acquisition left its measured-pose corridor')
            if held is not None:
                if held is not obj:
                    fail('Unexpected object acquired during contact hold')
                self._record_grasp_acquisition(event='contact_hold_acquired', object=obj.name,
                                               control_steps=step)
                return
            if step == budget:
                fail('Native grasp window elapsed without observed acquisition')
            if not contacts & allowed:
                fail('Target contact lost during grasp acquisition')
            yield self._postprocess_action(robot.q_to_action(q))
