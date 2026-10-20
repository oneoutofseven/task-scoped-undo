"""Capture post-release starts and run separately labelled retraction probes.

Scene restoration is initialization of a NEW diagnostic process only. A probe
is not an undo episode, and a scene file is not a full PhysX/planner checkpoint.
"""

import hashlib
import math
import traceback

from task_undo.simulation.retraction_diagnostics import RetractionDiagnostics
from task_undo.common.storage import read_json, write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def released(robot):
    return all(robot._ag_obj_in_hand[a] is None and robot._ag_obj_constraints[a] is None
               for a in robot.arm_names)


def state_sample(session):
    def body(obj):
        p, q = obj.get_position_orientation()
        return {"position": p.tolist(), "orientation": q.tolist(),
                "linear_velocity": obj.get_linear_velocity().tolist(),
                "angular_velocity": obj.get_angular_velocity().tolist()}

    return {"robot": body(session.robot),
            "joint_positions": session.robot.get_joint_positions().tolist(),
            "joint_velocities": session.robot.get_joint_velocities().tolist(),
            "objects": {name: body(obj) for name, obj in session.objects.items()}}


def compare_states(expected, actual):
    """Check restored start before executing; quaternion signs are equivalent."""
    def vector(values, size):
        if (not isinstance(values, (list, tuple)) or len(values) != size or size == 0
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
            raise ValueError("Checkpoint state requires complete finite vectors")

    for state in (expected, actual):
        if not isinstance(state.get("objects"), dict):
            raise ValueError("Checkpoint state requires an object mapping")
        for key in ("joint_positions", "joint_velocities"):
            vector(state[key], len(expected["joint_positions"]))
        for body in (state["robot"], *state["objects"].values()):
            for key in ("position", "linear_velocity", "angular_velocity"):
                vector(body[key], 3)
            vector(body["orientation"], 4)
            if not any(body["orientation"]):
                raise ValueError("Checkpoint quaternion cannot be zero")
    if set(expected["objects"]) != set(actual["objects"]):
        raise ValueError("Restored checkpoint object set changed")
    errors = {}
    for name in ("joint_positions", "joint_velocities"):
        errors[name] = max(abs(a - b) for a, b in zip(expected[name], actual[name]))
    bodies = [("robot", expected["robot"], actual["robot"]),
              *[(name, body, actual["objects"][name]) for name, body in expected["objects"].items()]]
    for name, a, b in bodies:
        for field in ("position", "linear_velocity", "angular_velocity"):
            errors[f"{name}.{field}"] = max(abs(x - y) for x, y in zip(a[field], b[field]))
        errors[f"{name}.orientation"] = min(
            max(abs(x - sign * y) for x, y in zip(a["orientation"], b["orientation"])) for sign in (1, -1))
    # Units: metres, radians, m/s, rad/s; orientation compares quaternion components.
    return {"passed": all(math.isfinite(v) and v <= 1e-4 for v in errors.values()),
            "component_tolerance": 1e-4, "max_abs_errors": errors}


def compare_contacts(expected, actual):
    """Reject missing contact state even when restored poses match exactly.

    Matching contact pairs is necessary, not sufficient: scene serialization
    still does not preserve the PhysX solver caches or contact impulses.
    """
    def signature(geometry):
        arms = geometry.get('arms', {})
        covered = bool(arms) and geometry.get('contact_matrix_available') is True
        pairs, per_arm = set(), {}
        for name, arm in arms.items():
            rows = arm.get('contact_rows_present', {})
            covered = covered and bool(rows) and all(v is True for v in rows.values())
            covered = covered and 'current_contact_pairs' in arm
            contacts = {tuple(sorted(pair)) for pair in arm.get('current_contact_pairs', [])}
            pairs.update(contacts)
            per_arm[name] = (set(rows), contacts)
        return covered, per_arm, pairs

    covered_before, per_arm_before, before = signature(expected)
    covered_after, per_arm_after, after = signature(actual)
    return {'passed': covered_before and covered_after and per_arm_before == per_arm_after,
            'captured_coverage_complete': covered_before, 'restored_coverage_complete': covered_after,
            'same_per_arm_coverage_and_contacts': per_arm_before == per_arm_after,
            'missing_pairs': sorted(before-after), 'unexpected_pairs': sorted(after-before),
            'solver_cache_equivalence_verified': False}


class CheckpointDiagnostics(RetractionDiagnostics):
    def __init__(self, session):
        super().__init__(session)
        self.controller = None
        self.checkpoints = []

    def install(self, controller):
        self.controller = controller
        super().install(controller)

    def record(self, kind, **fields):
        c = self.controller
        if (kind == "phase_started" and self.stack == ["_reset_robot"] and c is not None
                and c._placing and c._keep_released):
            try:
                self.capture_checkpoint()
            except Exception:
                self.error = traceback.format_exc()
                fields["checkpoint_error"] = self.error
        super().record(kind, **fields)

    def capture_checkpoint(self):
        import omnigibson as og

        s, c = self.session, self.controller
        if not released(s.robot):
            raise ValueError("Post-release checkpoint still has an assisted grasp")
        folder = s.output / "retraction_checkpoints" / f"{len(self.checkpoints) + 1:03d}"
        folder.mkdir(parents=True, exist_ok=False)
        before = state_sample(s)
        clock = (s.physics_steps, float(og.sim.current_time))
        # Scene.save only dumps state/metadata in the pinned OG implementation.
        # Never call restore, step, or a pose setter during capture.
        s.env.scene.save(json_path=str(folder / "scene.json"))
        after = state_sample(s)
        if before != after or clock != (s.physics_steps, float(og.sim.current_time)):
            raise RuntimeError("Checkpoint capture changed physical state or clock")
        write_json(folder / "case.json", s.case)
        write_json(folder / "checkpoint.json", {
            "schema_version": 1, "kind": "post_release_retraction_start",
            "scope": "diagnostic_initialization_only", "phase": s.result["phase"],
            "control_step": s.result["control_steps"], "physics_steps": clock[0], "sim_time": clock[1],
            "capture_unchanged": True, "state": before, "geometry": self.geometry(),
            "controller": {"arm": c.arm, "keep_released": c._keep_released,
                "reset_eef_pose": {a: [x.tolist() for x in pose] for a, pose in c._reset_eef_pose.items()},
                "arm_targets": {a: x.tolist() for a, x in c._arm_targets.items()},
                "tracking_object": c._tracking_object.name if c._tracking_object is not None else None},
            "sha256": {name: digest(folder / name) for name in ("scene.json", "case.json")},
            "limitations": ["PhysX contact/solver caches and planner caches are not serialized",
                            "Standalone probe uses its own seed, not the original planner RNG continuation"],
        })
        self.checkpoints.append(str(folder.relative_to(s.output)))

    def close(self):
        self.session.result["retraction_checkpoints"] = list(self.checkpoints)
        super().close()


def validate_checkpoint(folder):
    info = read_json(folder / "checkpoint.json")
    if (info.get("schema_version") != 1 or info.get("kind") != "post_release_retraction_start"
            or info.get("scope") != "diagnostic_initialization_only" or not info.get("capture_unchanged")):
        raise ValueError("Not a verified post-release checkpoint")
    for name in ("scene.json", "case.json"):
        if digest(folder / name) != info["sha256"][name]:
            raise ValueError(f"Checkpoint file changed: {name}")
    if not info["controller"]["keep_released"]:
        raise ValueError("Checkpoint lacks release intent")
    return info


def released_retraction(controller):
    """Same normal reset/settle controls, with the captured release intent.

    Unlike upstream apply_ref cleanup, propagate reset failures to the probe.
    No grasp bookkeeping or physical state is modified here.
    """
    if not released(controller.robot):
        raise ValueError("Retraction probe requires both hands released")
    previous = controller._placing, controller._keep_released
    controller._placing, controller._keep_released = True, True
    try:
        yield from controller._reset_robot()
        yield from controller._settle_robot()
    finally:
        controller._placing, controller._keep_released = previous


def initialize_probe(session, info):
    import torch

    s = session
    s.result.update(diagnostic_only=True, undo_evaluation_eligible=False)
    s.start_recording()
    before_init = state_sample(s)
    s.initialize_controller()
    c = s.controller
    if ":PLACE_ON_TOP:" in info["phase"]:
        c._released_object = s.objects[info["phase"].rsplit(":",1)[-1]]
    saved = info["controller"]
    if c.arm != saved["arm"]:
        raise ValueError("Active arm changed")
    # These are primitive planner targets, not physical pose assignments. A new
    # primitive would otherwise mistake the release pose for its reset target.
    c._reset_eef_pose = {a: tuple(torch.tensor(x) for x in pose)
                         for a, pose in saved["reset_eef_pose"].items()}
    c._arm_targets = {a: torch.tensor(x) for a, x in saved["arm_targets"].items()}
    name = saved["tracking_object"]
    c._tracking_object = s.env.scene.object_registry("name", name) if name else None
    if name and c._tracking_object is None:
        raise ValueError("Captured head tracking object missing")
    restored = state_sample(s)
    check = compare_states(info["state"], restored)
    geometry = s.gripper_diagnostics.geometry()
    contacts = compare_contacts(info.get('geometry', {}), geometry)
    check.update(kinematic_state_passed=check['passed'], contact_comparison=contacts,
                 passed=check['passed'] and contacts['passed'])
    check.update(controller_init_unchanged=before_init == restored, restored_state=restored,
                 original_phase=info["phase"], source_control_step=info["control_step"],
                 released=released(s.robot), geometry=geometry)
    write_json(s.output / "initialization_check.json", check)
    if not contacts['passed']:
        raise ValueError('Restored probe contact state differs from capture; use a full episode regression')
    if not check["passed"] or not check["controller_init_unchanged"] or not check["released"]:
        raise ValueError("Restored probe start does not match the captured physical state")
    return c


def run_probe(session, info):
    s = session
    c = initialize_probe(s, info)
    s.result["phase"] = "isolated_retraction"
    s.capture("before_retraction")
    generator = released_retraction(c)
    try:
        for action in generator:
            if s.result["control_steps"] >= s.max_steps:
                raise RuntimeError("Retraction probe step budget exhausted")
            s.env.step(action)
            s.result["control_steps"] += 1
            s.observe_execution()
            if s.result["control_steps"] % 3 == 0:
                s.capture()
    finally:
        generator.close()
    s.capture("after_retraction")
    s.result.update(passed=True, phase="finished", retraction_completed=True, final=s.observe())
