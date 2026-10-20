"""Move measured joints inside planner limits using bounded normal controls.

PhysX tracking can leave a resting joint microradians outside its URDF limit.
Do not clamp the planner's start state or relax its strict feasibility check.
"""
import math


def interior_targets(position, lower, upper, trigger=.0005, margin=.002, max_violation=.0001):
    if not len(position) == len(lower) == len(upper) or not position:
        raise ValueError('Complete joint-limit vectors required')
    if not 0 < trigger < margin or max_violation <= 0:
        raise ValueError('Invalid interior settling bounds')
    target, needed = [], False
    for q, lo, hi in zip(position, lower, upper):
        if not all(math.isfinite(v) for v in (q, lo, hi)) or hi-lo <= 2*margin:
            raise ValueError('Invalid joint limit')
        if max(lo-q, q-hi) > max_violation:
            raise ValueError('Joint violation exceeds numerical settling scope')
        close = min(q-lo, hi-q) < trigger
        needed |= close
        target.append(min(max(q, lo+margin), hi-margin) if close else q)
    return target if needed else None


def settle_joint_interior(controller, record=lambda **fields: None):
    """At most 20 controls, full payload geometry, unchanged planner bounds."""
    import torch as th
    from omnigibson.utils import transform_utils as T
    from task_undo.simulation.contact_retreat import CollisionView

    c, robot = controller, controller.robot
    obj = c._get_obj_in_hand()
    if obj is None:
        return
    view = CollisionView(c)
    limits = view.mg.kinematics.kinematics_config.joint_limits
    names = limits.joint_names
    indices = [view.planner.robot_joint_names.index(name) for name in names]
    q0 = robot.get_joint_positions()
    lower, upper = limits.position.detach().cpu()
    lower = th.maximum(lower, robot.joint_lower_limits[indices])
    upper = th.minimum(upper, robot.joint_upper_limits[indices])
    targets = interior_targets(q0[indices].tolist(), lower.tolist(), upper.tolist())
    if targets is None:
        return
    target = q0.clone()
    target[indices] = th.tensor(targets, dtype=q0.dtype, device=q0.device)
    js = view.JointState(position=view.planner.tensor_args.to_device(q0[None]),
                         joint_names=view.planner.robot_joint_names)
    view.planner.update_locked_joints(js, view.emb)
    info = view.planner._attach_objects_to_robot(
        {robot.eef_link_names[c.arm]: obj.root_link},
        {robot.eef_link_names[c.arm]: 1.0},
        js.get_ordered_joint_state(view.mg.kinematics.joint_names), view.emb)
    start, orientation = robot.eef_links[c.arm].get_position_orientation()
    start, orientation = start.clone(), orientation.clone()
    try:
        record(event='joint_interior_started', joint_names=names,
               initial=q0[indices].tolist(), target=targets,
               lower=lower.tolist(), upper=upper.tolist(), max_controls=20)
        for step in range(21):
            if c._get_obj_in_hand() is not obj:
                raise RuntimeError('Grasp lost during joint settling')
            q = robot.get_joint_positions()
            position, quaternion = robot.eef_links[c.arm].get_position_orientation()
            if (float(th.norm(position-start)) > .01
                    or float(T.get_orientation_diff_in_radian(quaternion,orientation)) > .035
                    or float((q[indices]-q0[indices]).abs().max()) > .005):
                raise RuntimeError('Joint settling left its motion bounds')
            costs, self_cost = view.measure(q)
            if self_cost > 0 or float(costs.max()) > 1e-7:
                raise RuntimeError('Joint settling requires a collision-free held pose')
            margin = float(th.minimum(q[indices]-lower, upper-q[indices]).min())
            if margin >= .00025:
                record(event='joint_interior_completed', steps=step,
                       minimum_margin=margin, position=q[indices].tolist())
                return
            if step == 20:
                break
            candidate = q.clone()
            candidate[indices] += (target[indices]-q[indices]).clamp(-.001, .001)
            # Check both the midpoint and command with the full attached object.
            for sample in ((q+candidate)*.5, candidate):
                sampled, self_cost = view.measure(sample)
                if self_cost > 0 or float(sampled.max()) > 1e-7:
                    raise RuntimeError('Joint settling candidate is in collision')
            yield c._postprocess_action(robot.q_to_action(candidate))
        raise RuntimeError('Joint settling exceeded 20 control steps')
    finally:
        view.planner._detach_objects_from_robot(info, view.emb)
