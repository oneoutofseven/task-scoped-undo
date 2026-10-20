"""Offline admission and error attribution for short physical diagnostics."""
import json

from task_undo.application.transfer_probe import transfer_plan
from task_undo.common.storage import read_json
from task_undo.evaluation.splits import verify_hashes


def audit_transfer(folder):
    result = read_json(folder / "result.json")
    manifest = read_json(folder / "manifest.json")
    verify_hashes(folder, manifest["sha256"])
    index = manifest["options"]["transfer_index"]
    plan = transfer_plan(read_json(folder / "case.json"))
    events = [json.loads(line) for line in (folder / "public_history.jsonl").read_text().splitlines()]
    expected = [(m["task_id"], action, m["object_id"],
                 m["target_surface"] if action == "PLACE_ON_TOP" else None)
                for m in plan[:index + 1] for action in ("GRASP", "PLACE_ON_TOP")]
    successful = [(e["task_id"], e["action"], e["object_id"], e.get("target_surface"))
                  for e in events if e["kind"] == "action_succeeded"]
    errors = []
    path = folder / "gripper_trace.jsonl"
    if path.exists():
        inside_body = False
        with path.open() as stream:
            for line in stream:
                event = json.loads(line)
                if event["kind"] == "primitive_body_started":
                    inside_body = True
                if event["kind"] in ("primitive_body_error", "phase_error", "motion_error"):
                    errors.append({**{k: event.get(k) for k in
                                      ("sequence", "kind", "control_step", "action_phase",
                                       "subphases", "primitive", "error_type", "error")},
                                   "inside_primitive_body": inside_body})
                if event["kind"] in ("primitive_body_error", "primitive_body_completed"):
                    inside_body = False
    probe = read_json(folder / "transfer_probe.json")
    final = result.get("final", {})
    target = plan[index]
    completed_events = [e for e in events if e["kind"] == "transfer_probe_completed"]
    terminal_ok = (len(completed_events) == 1 and completed_events[0]["observation"] == final
                   and completed_events[0]["index"] == index)
    diagnostic_ok = result.get("gripper_diagnostics", {}).get("passed") is True
    passed = bool(
        result.get("passed") and result.get("worker_exit_code") == 0
        and not result.get("timed_out") and not result.get("interrupted")
        and result.get("diagnostic_only") and result.get("history_eligible") is False
        and diagnostic_ok and successful == expected and terminal_ok
        and probe["target"] == target and probe["target_completed"]
        and probe["prefix_completed"] and probe["completed_transfers"] == list(range(index + 1))
        and final.get("hand_empty")
        and target["target_surface"] in final.get("objects", {}).get(target["object_id"], {}).get("on", [])
    )
    body_errors = [e for e in errors if e["kind"] == "primitive_body_error"]
    cleanup_errors = [e for e in errors if e["kind"] == "phase_error" and not e["inside_primitive_body"]]
    return {"passed": passed, "diagnostic_only": True, "history_eligible": False,
            "target": target, "prefix_completed": probe["prefix_completed"],
            "failure_scope": probe["failure_scope"], "phase": result.get("phase"),
            "control_steps": result.get("control_steps"), "diagnostic_trace_valid": diagnostic_ok,
            "successful_primitives": len(successful), "expected_primitives": len(expected),
            "first_primitive_error": body_errors[0] if body_errors else None,
            "first_cleanup_error": cleanup_errors[0] if cleanup_errors else None,
            "errors": errors, "terminal_error": result.get("error")}


def diagnostic_gate(planned, outcomes):
    """Missing, duplicate or failed probes block promotion; keep planned N."""
    expected = [row["probe_id"] for row in planned]
    actual = [row["probe_id"] for row in outcomes]
    identities_ok = bool(expected) and len(expected) == len(set(expected)) and sorted(actual) == sorted(expected)
    return {"passed": identities_ok and all(row.get("audit", {}).get("passed") is True for row in outcomes),
            "planned": len(expected), "completed": len(actual),
            "successful": sum(row.get("audit", {}).get("passed") is True for row in outcomes),
            "identity_check": identities_ok,
            "policy": "Every preregistered diagnostic must pass before full development regression."}
