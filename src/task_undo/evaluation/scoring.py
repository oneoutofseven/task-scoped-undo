"""Offline scoring from private labels and raw observations, without simulator imports.

Never use executor 'passed' or online protection reports as task-success labels.
"""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from task_undo.common.storage import read_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def check_state(state, names):
    require(isinstance(state, dict) and isinstance(state.get("objects"), dict), "Invalid observation structure")
    require(set(state["objects"]) == names, "Observation object set changed")
    require(type(state["hand_empty"]) is bool, "Invalid hand state")
    for obj in state["objects"].values():
        require(type(obj["held"]) is bool, "Invalid grasp state")
        require(isinstance(obj["on"], list) and all(isinstance(s, str) for s in obj["on"]), "Invalid support relation")
        for key, size in (("position", 3), ("orientation", 4)):
            values = obj[key]
            require(len(values) == size and all(type(x) in (int, float) and math.isfinite(x) for x in values),
                    "Invalid pose values")
        norm_squared = sum(x*x for x in obj["orientation"])
        require(math.isfinite(norm_squared) and norm_squared > 0, "Invalid quaternion norm")
        require(not (state["hand_empty"] and obj["held"]), "Empty hand contradicts observed grasp")


def pose_difference(a, b):
    distance = math.sqrt(sum((x-y)**2 for x, y in zip(a["position"], b["position"])))
    qa, qb = a["orientation"], b["orientation"]
    dot = sum(x*y for x, y in zip(qa, qb))
    norm = math.sqrt(sum(x*x for x in qa) * sum(x*x for x in qb))
    angle = math.degrees(2 * math.acos(min(1.0, abs(dot) / norm)))
    return distance, angle


def score_trace(labels, reference, initial, trace, physics, final, events, control_steps, execution_plan=None):
    """Score one complete recording; malformed/incomplete evidence raises ValueError."""
    version = labels["protocol_version"]
    require(type(version) is int and version in (1, 2), "Unknown evaluation protocol")
    restore, protected = labels["restore"], labels["protect"]
    require(isinstance(restore, dict) and isinstance(protected, dict), "Invalid object label sets")
    names = set(reference["objects"])
    require(not set(restore) & set(protected) and set(restore) | set(protected) == names,
            "Labels must partition observed objects into restore and protect sets")
    position_limit, angle_limit = labels["position_tolerance_m"], labels["orientation_tolerance_deg"]
    require(0 < position_limit <= 0.02 and 0 < angle_limit <= 5, "Invalid protocol tolerances")
    for state in (reference, initial, final):
        check_state(state, names)
    require(initial["hand_empty"], "Undo must start empty-handed")
    for name in names:
        distance, angle = pose_difference(reference["objects"][name], initial["objects"][name])
        require(distance <= position_limit and angle <= angle_limit
                and reference["objects"][name]["on"] == initial["objects"][name]["on"],
                "Restored starting state differs from source")
    for name, support in protected.items():
        require(reference["objects"][name]["on"] == [support], "Protection label contradicts source support")

    require(type(control_steps) is int and control_steps > 0, "No recorded execution steps")
    require(bool(trace) and bool(physics), "Raw state or physics trace missing")
    require(trace[0]["control_step"] == 0 and trace[0]["observation"] == initial, "Initial trace state mismatch")
    require(trace[-1]["control_step"] == control_steps and trace[-1]["observation"] == final, "Final trace state mismatch")
    previous_step, previous_time, seen_steps = -1, -1, set()
    for row in trace:
        step, timestamp = row["control_step"], row["sim_seconds"]
        require(type(step) is int and previous_step <= step <= control_steps, "State trace steps run backwards")
        require(math.isfinite(timestamp) and timestamp >= previous_time, "State trace time runs backwards")
        check_state(row["observation"], names)
        seen_steps.add(step)
        previous_step, previous_time = step, timestamp
    require(seen_steps == set(range(control_steps + 1)), "State trace has missing control steps")

    counts, previous_step = Counter(), 0
    for number, row in enumerate(physics, 1):
        require(row["physics_step"] == number, "Physics trace sequence has gaps")
        step = row["control_step"]
        require(type(step) is int and previous_step <= step <= control_steps, "Invalid physics control step")
        require(isinstance(row["held_objects"], list) and set(row["held_objects"]) <= names, "Unknown grasped object")
        counts[step] += 1
        previous_step = step
    # Fixed 120 Hz physics / 30 Hz control. Extra settling steps may share a
    # control counter, but every executed control step must have raw evidence.
    require(all(counts[step] >= 4 for step in range(control_steps)), "Missing physics evidence for control step")
    require(all(count % 4 == 0 for count in counts.values()), "Incomplete physics tick")
    require(not (final["hand_empty"] and physics[-1]["held_objects"]), "Final hand state contradicts physics record")

    require(bool(events) and events[0]["kind"] == "undo_started", "Undo start event missing")
    require(events[0]["target_task_id"] == labels["target_task_id"], "Wrong undo target")
    require(events[0]["observation"] == initial, "Start event and trace disagree")
    require(events[-1]["kind"] in ("undo_completed", "rollout_completed")
            and events[-1]["observation"] == final, "Rollout completion event missing or inconsistent")
    failures = 0
    if version == 2:
        from task_undo.evaluation.completion import validate_completion
        require(isinstance(execution_plan, dict), "V2 requires the saved execution plan")
        failures = validate_completion(execution_plan, initial, final, trace, events, control_steps)
    pending = {}
    for number, event in enumerate(events, 1):
        require(event["event_id"] == number and event["episode_id"] == labels["episode_id"], "Wrong event identity")
        kind = event["kind"]
        require(kind in ("undo_started", "action_requested", "action_succeeded", "undo_completed", "rollout_completed")
                or version == 2 and kind in ("action_failed", "task_completion_verified"),
                "Incomplete or failed execution history")
        if kind == "undo_started":
            require(number == 1, "Repeated undo start")
        if kind in ("undo_completed", "rollout_completed"):
            require(number == len(events), "Events after rollout completion")
        if kind == "action_requested":
            require(not pending, "Overlapping actions")
            pending[number] = event
        elif kind == "action_failed" and version == 2:
            request = pending.pop(event["request_event_id"], None)
            require(request is not None, "Action failure lacks matching request")
        elif kind == "action_succeeded":
            request = pending.pop(event["request_event_id"], None)
            require(request is not None and all(request[k] == event[k] for k in ("action", "object_id", "target_surface")),
                    "Action result lacks matching request")
    require(not pending, "Unfinished action request")

    violations = {}
    maxima = {name: {"position_m": 0.0, "angle_deg": 0.0} for name in protected}

    def violate(name, reason, sample):
        violations.setdefault(name, {}).setdefault(reason, sample)

    for row in trace:
        for name, support in protected.items():
            obj = row["observation"]["objects"][name]
            distance, angle = pose_difference(reference["objects"][name], obj)
            maxima[name]["position_m"] = max(maxima[name]["position_m"], distance)
            maxima[name]["angle_deg"] = max(maxima[name]["angle_deg"], angle)
            sample = {"control_step": row["control_step"]}
            if obj["held"]:
                violate(name, "grasped", sample)
            if obj["on"] != [support]:
                violate(name, "support_changed", sample)
            if distance > position_limit:
                violate(name, "position_changed", sample)
            if angle > angle_limit:
                violate(name, "orientation_changed", sample)
    for row in physics:
        for name in set(row["held_objects"]) & set(protected):
            violate(name, "grasped", {"physics_step": row["physics_step"]})

    restored = [name for name, target in restore.items()
                if target in final["objects"][name]["on"] and not final["objects"][name]["held"]]
    completion = len(restored) / len(restore) if restore else 1.0
    violation_rate = len(violations) / len(protected) if protected else 0.0
    hand_ok = not labels["require_empty_hand"] or final["hand_empty"]
    return {"valid": True, "protocol_version": version, "undo_completion_rate": completion,
            **({"primitive_execution_clean": failures == 0, "reconciled_failures": failures,
                "strict_joint_success": completion == 1 and violation_rate == 0 and hand_ok and failures == 0}
               if version == 2 else {}),
            "protection_violation_rate": violation_rate,
            "joint_success": completion == 1 and violation_rate == 0 and hand_ok,
            "counts": {"restored": len(restored), "restore_required": len(restore),
                       "protected_violated": len(violations), "protected_total": len(protected)},
            "restored_objects": restored, "violations": violations, "protection_maxima": maxima,
            "empty_hand_satisfied": hand_ok,
            "evidence": {"state_samples": len(trace), "physics_samples": len(physics), "control_steps": control_steps}}


def evaluate_directory(folder, labels_path):
    """Inputs are trusted recorder artifacts, not adversarially attested logs."""
    try:
        labels = read_json(labels_path)
        require(set(labels["source_sha256"]) == {"before_undo.json", "before_undo_scene.json", "public_history.jsonl"},
                "Source binding must include history, observation and scene")
        for name, expected in labels["source_sha256"].items():
            require(hashlib.sha256((folder / "input" / name).read_bytes()).hexdigest() == expected,
                    "Source episode differs from labels")
        result = read_json(folder / "result.json")
        require(result["worker_exit_code"] == 0 and not result.get("timed_out") and not result.get("interrupted")
                and result["phase"] == "finished" and "error" not in result, "Execution did not finish normally")
        require(result.get("trace_schema_version") == 2, "Raw physics recording requires trace schema 2")
        require(result["episode_id"] == labels["episode_id"], "Wrong episode")
        final = read_json(folder / "after_undo.json")
        require(result["final"] == final, "Execution report and final observation disagree")
        trace, physics = read_jsonl(folder / "execution_trace.jsonl"), read_jsonl(folder / "physics_trace.jsonl")
        require(result["physics_steps"] == len(physics), "Physics trace truncated")
        events = read_jsonl(folder / "public_history.jsonl")
        require(result["event_count"] == len(events), "Event history truncated")
        plan = None
        if labels["protocol_version"] == 2:
            from task_undo.application.predictions import load_prediction
            from task_undo.domain.completion import EXECUTOR
            require(result.get("execution_mode") == EXECUTOR, "V2 executor not enabled")
            manifest = read_json(folder / "manifest.json")
            require(manifest["options"].get("execution_mode") == EXECUTOR, "V2 was not declared before execution")
            plan = read_json(folder / "execution_plan.json")
            prediction = load_prediction(folder / "prediction", read_json(folder / "case.json"),
                                         read_json(folder / "initial_state.json"),
                                         manifest["options"].get("prediction_readout", "staged_rules"))
            require(plan["goal"] == prediction, "Execution goal differs from saved prediction")
        score = score_trace(labels, read_json(folder / "input/before_undo.json"),
                            read_json(folder / "initial_state.json"), trace, physics, final, events, result["control_steps"], execution_plan=plan)
        score["labels_sha256"] = hashlib.sha256(labels_path.read_bytes()).hexdigest()
        score["evaluator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return score
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        return {"valid": False, "error": str(exc), "undo_completion_rate": None,
                "protection_violation_rate": None, "joint_success": False}
