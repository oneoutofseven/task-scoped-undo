"""Reference-run protection checks; independent of the simulator and model."""

import math


def pose_error(reference, current):
    distance = math.dist(reference["position"], current["position"])
    a, b = reference["orientation"], current["orientation"]
    norm = math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
    if not math.isfinite(distance) or not math.isfinite(norm) or norm <= 0:
        raise ValueError("Invalid pose")
    dot = abs(sum(x*y for x, y in zip(a, b))) / norm
    angle = math.degrees(2 * math.acos(min(1.0, max(0.0, dot))))
    return distance, angle


class ProtectionMonitor:
    """Latch violations, so putting an object back cannot erase them."""

    def __init__(self, reference, objects, position_m=0.02, angle_deg=5.0):
        self.reference = reference
        self.objects = tuple(objects)
        self.position_m = position_m
        self.angle_deg = angle_deg
        self.samples = 0
        self.physics_samples = 0
        self.violations = {}
        self.maxima = {name: {"position_m": 0.0, "angle_deg": 0.0} for name in self.objects}

    def violation(self, name, kind, step):
        self.violations.setdefault((name, kind), {"object_id": name, "kind": kind, "first_step": step})

    def check_grasp(self, held_objects):
        self.physics_samples += 1
        for name in self.objects:
            if name in held_objects:
                self.violation(name, "grasped", self.physics_samples)

    def check(self, state, step):
        self.samples += 1
        for name in self.objects:
            original, current = self.reference["objects"][name], state["objects"][name]
            distance, angle = pose_error(original, current)
            self.maxima[name]["position_m"] = max(self.maxima[name]["position_m"], distance)
            self.maxima[name]["angle_deg"] = max(self.maxima[name]["angle_deg"], angle)
            if current["held"]:
                self.violation(name, "grasped", step)
            if set(original["on"]) != set(current["on"]):
                self.violation(name, "support_changed", step)
            if distance > self.position_m:
                self.violation(name, "position_changed", step)
            if angle > self.angle_deg:
                self.violation(name, "orientation_changed", step)

    def report(self):
        return {"passed": not self.violations, "state_samples": self.samples,
                "physics_grasp_samples": self.physics_samples,
                "limits": {"position_m": self.position_m, "angle_deg": self.angle_deg},
                "maxima": self.maxima, "violations": list(self.violations.values())}

    def require_valid(self):
        if self.violations:
            raise RuntimeError(f"Protected state changed: {list(self.violations.values())}")
