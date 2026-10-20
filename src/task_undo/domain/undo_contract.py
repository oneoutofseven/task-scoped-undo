"""Public goal constraints and observed next actions, without history or physics.

This module cannot decide task ownership or repair an incorrect model goal.
It checks and executes that supplied hypothesis under the sequential v1 scope.
"""
from copy import deepcopy
import math

from task_undo.domain.protection import pose_error

EXECUTOR = "guarded_goal_v1"
SETTLE_STEPS = 30


class ContractError(ValueError):
    def __init__(self, reason, detail):
        self.reason = reason
        super().__init__(detail)


def validate_observation(state, objects, surfaces, *, at_start=False):
    """Reject missing, ambiguous or inconsistent observations before dispatch."""
    def require(ok, message):
        if not ok:
            raise ContractError("invalid_observation", message)

    try:
        require(isinstance(state, dict) and isinstance(state["objects"], dict), "Missing objects")
        require(set(state["objects"]) == set(objects), "Observed object set changed")
        require(type(state["hand_empty"]) is bool, "Missing boolean hand state")
        held = []
        for name, obj in state["objects"].items():
            require(type(obj["held"]) is bool, "Invalid held state: " + name)
            supports = obj["on"]
            require(isinstance(supports, list) and all(isinstance(s, str) for s in supports),
                    "Invalid supports: " + name)
            require(len(supports) <= 1 and set(supports) <= set(surfaces),
                    "Ambiguous or unknown support: " + name)
            require(obj["held"] or len(supports) == 1, "Unheld object has no known support: " + name)
            for key, size in (("position", 3), ("orientation", 4)):
                values = obj[key]
                require(isinstance(values, (list, tuple)) and len(values) == size,
                        "Invalid pose dimensions: " + name)
                require(all(type(x) in (int, float) and math.isfinite(x) for x in values),
                        "Nonfinite pose: " + name)
            require(any(obj["orientation"]), "Zero orientation: " + name)
            if obj["held"]:
                held.append(name)
        require(len(held) <= 1 and state["hand_empty"] == (not held), "Inconsistent grasp/hand state")
        require(not at_start or not held, "Undo must start empty-handed")
    except (KeyError, TypeError, OverflowError) as error:
        raise ContractError("invalid_observation", "Incomplete observation") from error


class UndoContract:
    """Copy the public goal and reference so caller mutations cannot change them."""

    def __init__(self, goal, reference, surfaces, target_task_id):
        if not isinstance(target_task_id, str) or not target_task_id:
            raise ContractError("invalid_goal", "A target task ID is required")
        self.target_task_id = target_task_id
        self.surfaces = tuple(sorted(surfaces))
        try:
            names = set(reference["objects"])
            restore, protect = goal["restore"], goal["protect"]
            if (not isinstance(restore, dict) or not isinstance(protect, list)
                    or not all(isinstance(n, str) and n for n in [*restore, *protect])
                    or len(protect) != len(set(protect)) or set(restore) & set(protect)
                    or set(restore) | set(protect) != names
                    or not all(isinstance(s, str) and s in self.surfaces for s in restore.values())):
                raise ContractError("invalid_goal", "Goal must partition every object and use known supports")
            validate_observation(reference, names, self.surfaces, at_start=True)
        except (KeyError, TypeError) as error:
            raise ContractError("invalid_goal", "Incomplete goal or reference") from error
        self._restore = tuple(sorted(restore.items()))
        self.protect = tuple(sorted(protect))
        self._reference = deepcopy(reference)
        self.objects = tuple(sorted(names))
        self.max_actions = 2 * len(self._restore)

    @property
    def goal(self):
        return {"restore": dict(self._restore), "protect": list(self.protect)}

    @property
    def reference(self):
        return deepcopy(self._reference)

    def record(self):
        return {"schema_version": 1, "executor": EXECUTOR, "target_task_id": self.target_task_id,
                "goal": self.goal, "reference": self.reference, "surfaces": list(self.surfaces),
                "position_tolerance_m": .02, "orientation_tolerance_deg": 5,
                "max_actions": self.max_actions, "settle_steps": SETTLE_STEPS,
                "primitive_failures_reconciled": False, "private_labels_used": False}

    def next_action(self, state):
        validate_observation(state, self.objects, self.surfaces)
        for name in self.protect:
            before, now = self._reference["objects"][name], state["objects"][name]
            distance, angle = pose_error(before, now)
            if now["held"] or now["on"] != before["on"] or distance > .02 or angle > 5:
                raise ContractError("protection_violation", "Protected state changed: " + name)
        held = next((n for n in self.objects if state["objects"][n]["held"]), None)
        if held is not None:
            return {"action": "PLACE_ON_TOP", "object_id": held,
                    "target_surface": dict(self._restore)[held]}
        for name, target in self._restore:
            if state["objects"][name]["on"] != [target]:
                return {"action": "GRASP", "object_id": name, "target_surface": None}
        return None

    def check_action_result(self, action, state):
        # Also checks every protected object; a native success is not sufficient.
        self.next_action(state)
        obj = state["objects"][action["object_id"]]
        valid = (obj["held"] if action["action"] == "GRASP" else
                 state["hand_empty"] and obj["on"] == [action["target_surface"]])
        if not valid:
            raise ContractError("postcondition_failed", "Observed primitive result disagrees with its request")
