"""Opt-in smaller loaded-arm setpoint increments on an existing planned path.

This changes sampling of the same joint-space segments, not collision margins,
joint convergence thresholds, motor effort limits or the outer control budget.
Contact approach, release, base travel and empty-arm motion stay upstream.
"""
import math


def densify_joint_path(path, indices, max_step=.005, max_waypoints=20000):
    import torch as th
    if path.ndim != 2 or len(path) < 1 or not indices:
        raise ValueError("A nonempty joint path and active joint indices are required")
    if not math.isfinite(max_step) or max_step <= 0 or max_waypoints < 1 or not bool(th.isfinite(path).all()):
        raise ValueError("Invalid loaded trajectory or resampling limits")
    if len(path) == 1:
        return path
    differences = (path[1:, indices] - path[:-1, indices]).abs().amax(dim=1)
    counts = th.ceil(differences / max_step).clamp_min(1).to(th.int64).tolist()
    count = 1 + sum(counts)
    if count > max_waypoints:
        raise ValueError("Loaded trajectory exceeds the bounded waypoint count")
    if count == len(path):
        return path
    rows = [path[:1]]
    for start, end, n in zip(path[:-1], path[1:], counts):
        # Preserve original endpoints exactly; avoid accumulating roundoff.
        if n > 1:
            fraction = th.arange(1, n, device=path.device, dtype=path.dtype)[:, None] / n
            rows.append(start[None] + fraction * (end - start)[None])
        rows.append(end[None])
    return th.cat(rows)


class LoadedArmMotion:
    motion_profile = "legacy"

    def _execute_motion_plan(self, q_traj, stop_on_contact=False, stop_on_ag=False,
                             ignore_failure=False, low_precision=False, ignore_physics=False):
        if (self.motion_profile == "loaded_waypoints_v1" and self._get_obj_in_hand() is not None
                and not any((stop_on_contact, stop_on_ag, ignore_failure, low_precision, ignore_physics))):
            base = self.robot.base_control_idx
            # Base trajectories already have a separate, validated retimer.
            stationary_base = len(q_traj) <= 1 or float((q_traj[:, base] - q_traj[0, base]).abs().max()) < 1e-6
            if stationary_base:
                indices = self.robot.trunk_control_idx.tolist() + self.robot.arm_control_idx[self.arm].tolist()
                dense = densify_joint_path(q_traj, indices)
                self._record_interaction_search(event="loaded_path_resampled",
                    original_waypoints=len(q_traj), executed_waypoints=len(dense),
                    maximum_joint_increment=.005, motion_profile=self.motion_profile,
                    joint_threshold_unchanged=True, control_budget_unchanged=True)
                q_traj = dense
        return (yield from super()._execute_motion_plan(
            q_traj, stop_on_contact=stop_on_contact, stop_on_ag=stop_on_ag,
            ignore_failure=ignore_failure, low_precision=low_precision, ignore_physics=ignore_physics))
