"""Local surface targeting for thin objects in top-down sticky grasps.

Ray queries read privileged simulator collision geometry. They do not replace
native contact acquisition or certify feasibility of the subsequent trajectory.
"""
import math

CONTACT_PRELOAD_M = .005


def ray_evidence(hit):
    """Convert native tensor ray results without retaining simulator objects."""
    return {key: value.tolist() if hasattr(value, "tolist") else value
            for key in ('hit', 'rigidBody', 'position', 'normal')
            for value in [hit.get(key)]}


def is_thin_object(lower, upper):
    extents = [hi-lo for lo, hi in zip(lower, upper)]
    return 0 < extents[2] <= .04 and extents[2] <= .25 * min(extents[:2])


def corrected_contact_height(current_height, finger_length, lower, upper, hit, target_paths):
    """Lower an insufficient top approach using the first visible target surface.

    Keep already deeper targets unchanged. Never skip an occluder or use the
    support surface as if it belonged to the object. A 5 mm preload accommodates
    tracking error; the ordinary planner still checks all non-target geometry.
    """
    if not is_thin_object(lower, upper):
        return current_height
    if (not hit.get('hit') or hit.get('rigidBody') not in target_paths
            or not all(math.isfinite(float(v)) for v in hit.get('position', []))
            or len(hit.get('position', [])) != 3
            or len(hit.get('normal', [])) != 3
            or not math.isfinite(float(hit['normal'][2])) or hit['normal'][2] < .9):
        raise ValueError('Thin-object approach requires an unobstructed upward-facing target surface')
    surface = float(hit['position'][2])
    if not lower[2] <= surface <= upper[2]:
        raise ValueError('Contact surface lies outside target bounds')
    # Do not command the nominal fingertip below the object's bottom. Thin
    # geometry cannot justify penetrating the supporting furniture.
    fingertip = max(surface-CONTACT_PRELOAD_M, lower[2]+.001)
    return min(current_height, fingertip+finger_length)


def refine_thin_contact(controller, obj, grasp, finger_length):
    lower, upper = [v.tolist() for v in obj.aabb]
    if not is_thin_object(lower, upper):
        return grasp
    from omnigibson.utils.sampling_utils import raytest
    from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
    position, orientation = grasp
    x, y = position[:2].tolist()
    hit = raytest([x, y, upper[2]+.04], [x, y, lower[2]-.001])
    try:
        height = corrected_contact_height(float(position[2]), finger_length, lower, upper,
                                          hit, set(obj.link_prim_paths))
    except ValueError as error:
        controller._record_grasp_search(event='surface_contact_rejected', object=obj.name,
            ray_start=[x, y, upper[2]+.04], ray_end=[x, y, lower[2]-.001],
            hit=ray_evidence(hit),
            bounds=[lower, upper], target_paths=sorted(obj.link_prim_paths),
            reason=str(error), observation='privileged_collision_raycast')
        raise ActionPrimitiveError(ActionPrimitiveError.Reason.PLANNING_ERROR, str(error),
                                   {'object': obj.name}) from error
    refined = position.clone()
    refined[2] = height
    controller._record_grasp_search(event='surface_contact_target', object=obj.name,
        previous_height_m=float(position[2]), height_m=height,
        surface_m=float(hit['position'][2]), preload_budget_m=CONTACT_PRELOAD_M,
        observation='privileged_collision_raycast')
    return refined, orientation
