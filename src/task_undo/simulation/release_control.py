"""Preserve release intent after PLACE, without changing intentional GRASP.

This mixin uses the upstream normal action conversion. It does not touch
assisted-grasp state, constraints, object poses, or the physics clock.
"""


def finish_release(actions, is_holding, max_steps):
    """Keep issuing the open command until observed release, within its budget.

    Finger convergence can precede the simulator's release notification. Keep
    the original command during that delay; never edit grasp state or counters.
    """
    last, count = None, 0
    for last in actions:
        count += 1
        yield last
    while last is not None and count < max_steps and is_holding():
        count += 1
        yield last.clone()


class PlaceReleaseControl:
    _placing = False
    _keep_released = False

    def apply_ref(self, primitive, *args, **kwargs):
        previous = self._placing, self._keep_released
        self._placing = primitive.name == "PLACE_ON_TOP"
        self._keep_released = False
        try:
            return (yield from super().apply_ref(primitive, *args, **kwargs))
        finally:
            self._placing, self._keep_released = previous

    def _execute_release(self):
        # The upstream method verifies release before returning. A failed or
        # interrupted release must not activate the post-release override.
        value = yield from super()._execute_release()
        if self._placing:
            self._keep_released = True
        return value

    def _move_fingers_to_limit(self, limit_type):
        actions = super()._move_fingers_to_limit(limit_type)
        if limit_type != "upper":
            return (yield from actions)
        from omnigibson.action_primitives.starter_semantic_action_primitives import m
        yield from finish_release(actions, lambda: self._get_obj_in_hand() is not None,
                                  m.MAX_STEPS_FOR_GRASP_OR_RELEASE)

    def _postprocess_action(self, action):
        action = super()._postprocess_action(action)
        if not self._keep_released:
            return action
        # Retract and settle may otherwise hold measured finger positions below
        # the open limit, which the robot interprets as applying a grasp.
        open_q = self._get_joint_position_with_fingers_at_limit("upper")
        open_action = self.robot.q_to_action(open_q)
        indices = self.robot.controller_action_idx[f"gripper_{self.arm}"]
        action = action.clone()
        action[indices] = open_action[indices]
        return action
