"""Backtrack a navigation candidate before issuing any physical command.

Endpoint IK alone cannot certify a reachable interaction. Try at most three
base paths for the same object/EEF target. Once execution starts, any failure
propagates normally; this module never replays or rolls back physical state.
"""


def pre_motion_planning_failure(error, controls_issued):
    return not controls_issued and getattr(getattr(error, 'reason', None), 'name', None) == 'PLANNING_ERROR'


class InteractionSearch:
    _navigation_candidate = None

    def apply_ref(self, *args, **kwargs):
        previous = self._navigation_candidate
        self._navigation_candidate = None
        try:
            return (yield from super().apply_ref(*args, **kwargs))
        finally:
            self._navigation_candidate = previous

    def _remember_navigation_candidate(self, pose, obj, eef_pose, plan_with_open_gripper=False):
        self._navigation_candidate = pose, obj, eef_pose, plan_with_open_gripper

    def _sample_pose_near_object(self, obj, eef_pose=None, plan_with_open_gripper=False, **kwargs):
        # PLACE supplies an explicit endpoint budget. Split it between the
        # upstream bearing-tied heading and independent base headings: changing
        # the support pose alone does not explore this missing degree of freedom.
        # GRASP and callers without an explicit budget keep their original path.
        budget = kwargs.get('sampling_attempts', 0)
        free_budget = budget // 2 if plan_with_open_gripper and eef_pose is not None else 0
        if free_budget:
            kwargs = {**kwargs, 'sampling_attempts': budget-free_budget}
        pose = super()._sample_pose_near_object(obj, eef_pose=eef_pose,
            plan_with_open_gripper=plan_with_open_gripper, **kwargs)
        if pose is None and free_budget:
            pose = self._sample_yaw_free_base(obj, eef_pose, sampling_attempts=free_budget,
                skip_obstacle_update=True, plan_with_open_gripper=True)
        if pose is not None:
            self._remember_navigation_candidate(pose, obj, eef_pose, plan_with_open_gripper)
        return pose

    def _record_interaction_search(self, **fields):
        pass

    def _navigate_to_pose(self, pose, skip_obstacle_update=False):
        context, self._navigation_candidate = self._navigation_candidate, None
        if context is None or context[0] is not pose or context[2] is None:
            return (yield from super()._navigate_to_pose(pose, skip_obstacle_update))
        _, obj, eef_pose, open_gripper = context
        try:
            for attempt in range(3):
                issued = False
                try:
                    for control in super()._navigate_to_pose(pose, skip_obstacle_update):
                        issued = True
                        yield control
                    self._record_interaction_search(event='path_executed', path_attempt=attempt+1)
                    return
                except Exception as exc:
                    if attempt == 2 or not pre_motion_planning_failure(exc, issued):
                        raise
                    self._record_interaction_search(event='path_rejected_before_motion',
                        path_attempt=attempt+1, error=str(exc), next_base_budget=50)
                    alternative = self._sample_yaw_free_base(obj, eef_pose, sampling_attempts=50,
                        skip_obstacle_update=True, plan_with_open_gripper=open_gripper)
                    if alternative is None:
                        raise
                    pose = alternative
        finally:
            self._navigation_candidate = None
