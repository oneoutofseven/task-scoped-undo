"""Fixed high-level execution for a public-history goal, without a model.

This is not a motion planner: each GRASP / PLACE still uses the same physical
primitives as the Agent. Object order is fixed by ID, not by private labels.
"""


def fixed_actions(goal):
    restore, protect = goal["restore"], goal["protect"]
    if not isinstance(restore, dict) or not isinstance(protect, list):
        raise ValueError("Invalid rule goal")
    if not all(isinstance(name, str) and name for name in [*restore, *protect]):
        raise ValueError("Object IDs must be nonempty strings")
    if len(set(protect)) != len(protect) or set(restore) & set(protect):
        raise ValueError("Restore and protect must be disjoint")
    if not all(isinstance(target, str) and target for target in restore.values()):
        raise ValueError("Restore targets must be nonempty strings")
    actions = []
    for name in sorted(restore):
        actions.extend([
            {"action": "GRASP", "object_id": name, "target_surface": None},
            {"action": "PLACE_ON_TOP", "object_id": name, "target_surface": restore[name]},
        ])
    return actions
