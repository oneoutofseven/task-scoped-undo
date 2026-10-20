"""Search top-grasp yaw and base placement before executing either motion.

The original candidate keeps its original 200-base budget. If exhausted,
search at most four object-relative yaw alternatives with 50 bases each.
All candidates use the normal full-world collision and IK checks. This is a
development fallback, not a claim of complete manipulation feasibility.
"""
import math


def rotate_top_pose(position, quaternion, center, yaw):
    """Rotate both the offset contact point and wrist about the object's center."""
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = position[0]-center[0], position[1]-center[1]
    p = [center[0]+c*x-s*y, center[1]+s*x+c*y, position[2]]
    c, s = math.cos(yaw/2), math.sin(yaw/2)
    x, y, z, w = quaternion
    return p, [c*x-s*y, s*x+c*y, c*z+s*w, c*w-s*z]


def alternative_yaws(reference_yaw):
    """Four directions tied to the object; omit the already tried world yaw."""
    values = [math.atan2(math.sin(reference_yaw+d), math.cos(reference_yaw+d))
              for d in (0., math.pi/2, math.pi, -math.pi/2)]
    return [v for v in values if abs(v) > 1e-5]


def radial_base_pose(target, distance, azimuth, facing_offset):
    """Independent base location and heading, expressed relative to a target."""
    yaw = azimuth + math.pi + facing_offset
    return [target[0]+distance*math.cos(azimuth), target[1]+distance*math.sin(azimuth),
            math.atan2(math.sin(yaw), math.cos(yaw))]


class JointGraspSearch:
    _searching_grasp = False
    _chosen_grasp_base = None
    _grasp_attempt = 0
    _grasp_clearance_pair = None

    def apply_ref(self, primitive, *args, **kwargs):
        previous = self._grasp_attempt
        self._grasp_attempt = 0
        try:
            return (yield from super().apply_ref(primitive, *args, **kwargs))
        finally:
            self._grasp_attempt = previous

    def _record_grasp_search(self, **fields):
        pass

    def _validate_poses(self, candidate_poses, *args, **kwargs):
        valid = super()._validate_poses(candidate_poses, *args, **kwargs)
        pair = self._grasp_clearance_pair
        if pair is not None and kwargs.get('eef_pose') is pair[0]:
            indices = valid.nonzero().flatten().tolist()
            if indices:
                extra = super()._validate_poses([candidate_poses[i] for i in indices], *args,
                    **{**kwargs, 'eef_pose': pair[1]})
                valid = valid.clone()
                for index, feasible in zip(indices, extra):
                    valid[index] = feasible
                self._record_grasp_search(event='extraction_endpoint_filter',
                    candidates=len(indices), accepted=int(extra.sum()))
        return valid

    def _grasp(self, obj):
        previous = self._searching_grasp, self._chosen_grasp_base
        self._searching_grasp, self._chosen_grasp_base = True, None
        self._grasp_attempt += 1
        try:
            return (yield from super()._grasp(obj))
        finally:
            self._searching_grasp, self._chosen_grasp_base = previous

    def _grasp_alternatives(self, obj, pair):
        import torch as th
        from omnigibson.utils import transform_utils as T
        center = obj.aabb_center.tolist()
        yaw = float(T.quat2euler(obj.get_position_orientation()[1])[2])
        for angle in alternative_yaws(yaw):
            rotated = tuple(tuple(th.tensor(v, dtype=source[0].dtype, device=source[0].device)
                                  for v in rotate_top_pose(source[0].tolist(), source[1].tolist(), center, angle))
                            for source in pair)
            yield angle, rotated

    def _sample_yaw_free_base(self, obj, eef_pose, sampling_attempts, skip_obstacle_update=True,
                              plan_with_open_gripper=False):
        import torch as th
        target = eef_pose[0]
        # The upstream sampler fixes base yaw to the target bearing minus the
        # mean arm workspace angle. Vary this independent degree of freedom;
        # retain its 1.5 m radius and the same full collision / IK validation.
        offsets = (-math.pi/4, 0., math.pi/4, -math.pi/2, math.pi/2, math.pi)
        for attempt in range(sampling_attempts):
            for _ in range(20):
                radius, angle = th.rand(2).tolist()
                pose = th.tensor(radial_base_pose(target, radius*1.5, angle*2*math.pi,
                                                 offsets[attempt % len(offsets)]))
                # A semantic room boundary is not a physical barrier. An arm
                # may reach across it from a collision-free neighboring room.
                # Keep a mapped-floor check, then let collision / IK / actual
                # navigation determine whether this base pose is feasible.
                if self.robot.scene._seg_map.get_room_instance_by_point(pose[:2]) is not None:
                    break
            else:
                continue
            valid = self._validate_poses([pose], eef_pose=eef_pose,
                plan_with_open_gripper=plan_with_open_gripper, skip_obstacle_update=skip_obstacle_update)
            if bool(valid[0]):
                return pose
        return None

    def _select_grasp(self, obj, pair):
        if not self._searching_grasp:
            return pair
        self._motion_generator.update_obstacles()

        from task_undo.simulation.grasp_lift import extraction_height
        bottom, top = obj.aabb
        obstacles = [(lo.tolist(), hi.tolist()) for other in self.robot.scene.objects
                     if other is not obj and other is not self.robot for lo, hi in [other.aabb]]
        rise = extraction_height((bottom.tolist(), top.tolist()), obstacles)

        def choose(candidate, yaw, budget, free_base_yaw=False):
            pregrasp, _ = candidate
            raised = pregrasp[0].clone(), pregrasp[1].clone()
            raised[0][2] += rise
            self._record_grasp_search(event='candidate', yaw_rad=yaw, base_budget=budget,
                                     free_base_yaw=free_base_yaw,
                                     extraction_rise_m=rise,
                                     pregrasp=[v.tolist() for v in pregrasp])
            # A feasible endpoint IK is not a feasible approach trajectory.
            # After an execution/planning failure, use the next existing
            # primitive attempt to sample another base instead of selecting
            # the same current-base IK again. Total attempts remain upstream.
            if (self._grasp_attempt <= 1
                    and self._target_in_reach_of_robot(pregrasp, skip_obstacle_update=True)
                    and self._target_in_reach_of_robot(raised, skip_obstacle_update=True)):
                self._chosen_grasp_base = obj, pregrasp, None, raised
                self._record_grasp_search(event='selected', yaw_rad=yaw, navigate=False)
                return True
            sample = self._sample_yaw_free_base if free_base_yaw else self._sample_pose_near_object
            previous, self._grasp_clearance_pair = self._grasp_clearance_pair, (pregrasp, raised)
            try:
                base = sample(obj, eef_pose=pregrasp, sampling_attempts=budget, skip_obstacle_update=True)
            finally:
                self._grasp_clearance_pair = previous
            if base is None:
                return False
            self._chosen_grasp_base = obj, pregrasp, base, raised
            self._record_grasp_search(event='selected', yaw_rad=yaw, navigate=True, base=base.tolist())
            return True

        if choose(pair, 0., 200):
            return pair
        for yaw, candidate in self._grasp_alternatives(obj, pair):
            if choose(candidate, yaw, 50, free_base_yaw=True):
                return candidate
        from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
        raise ActionPrimitiveError(ActionPrimitiveError.Reason.PLANNING_ERROR,
                                   'No feasible top-grasp/base pair within search budget', {'object': obj.name})

    def _navigate_if_needed(self, obj, eef_pose=None):
        selected = self._chosen_grasp_base
        self._chosen_grasp_base = None
        if selected is None or selected[0] is not obj or selected[1] is not eef_pose:
            return (yield from super()._navigate_if_needed(obj, eef_pose=eef_pose))
        if selected[2] is not None:
            if hasattr(self, '_remember_navigation_candidate'):
                self._remember_navigation_candidate(selected[2], obj, eef_pose)
            previous, self._grasp_clearance_pair = self._grasp_clearance_pair, (eef_pose, selected[3])
            try:
                yield from self._navigate_to_pose(selected[2], skip_obstacle_update=True)
            finally:
                self._grasp_clearance_pair = previous
