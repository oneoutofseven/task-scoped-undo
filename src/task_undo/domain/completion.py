"""Public predicted-goal completion checks; no author labels or simulator state writes."""

import math
from task_undo.domain.protection import pose_error

SETTLE_STEPS = 30
EXECUTOR = "observe_completion_v1"


def goal_satisfied(goal, reference, current):
    """Fail closed on incomplete observations; protection history is checked separately."""
    try:
        restore, protect = goal["restore"], goal["protect"]
        names = set(reference["objects"])
        if (set(restore) & set(protect) or set(restore) | set(protect) != names
                or len(protect) != len(set(protect)) or set(current["objects"]) != names
                or current["hand_empty"] is not True):
            return False
        for state in (reference, current):
            for obj in state["objects"].values():
                if type(obj["held"]) is not bool or not isinstance(obj["on"], list):
                    return False
                for key, size in (("position", 3), ("orientation", 4)):
                    values = obj[key]
                    if len(values) != size or not all(type(x) in (int, float) and math.isfinite(x) for x in values):
                        return False
                if not any(obj["orientation"]):
                    return False
        if any(obj["held"] for obj in current["objects"].values()):
            return False
        for name, target in restore.items():
            if current["objects"][name]["on"] != [target]:
                return False
        for name in protect:
            old, new = reference["objects"][name], current["objects"][name]
            distance, angle = pose_error(old, new)
            if old["on"] != new["on"] or distance > 0.02 or angle > 5:
                return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
