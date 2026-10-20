"""Check stage-3 history consistency, not the future undo agent's score."""

import json

from task_undo.domain.case_layout import claim_timing, object_ids, task_map


def check_events(events, case):
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    tasks = task_map(case)
    timings = {key: claim_timing(task) for key, task in tasks.items()}
    completed, claims, pending = set(), set(), {}
    successful = {key: [] for key in tasks}
    active = None
    previous_time = previous_step = -1
    cancelled = False
    require(bool(events) and events[0]["kind"] == "episode_started", "Initial event missing")
    names = (set(object_ids(case)) if "objects" in case
             else set(events[0]["observation"]["objects"]))
    for number, e in enumerate(events, 1):
        require(e["event_id"] == number, "Event IDs must be contiguous")
        require(e["episode_id"] == case["episode_id"], "Wrong episode ID")
        require(e["sim_seconds"] >= previous_time and e["control_step"] >= previous_step, "History runs backwards")
        previous_time, previous_step = e["sim_seconds"], e["control_step"]
        kind = e["kind"]
        require(not cancelled, "Events occurred after final undo request")
        require(kind not in ("action_failed", "episode_failed"), "Failed execution cannot be accepted as a complete case")
        require(kind in {"episode_started", "task_started", "use_claimed", "action_requested",
                         "action_succeeded", "task_completed", "undo_requested"}, "Unknown event kind")
        if "observation" in e:
            require(set(e["observation"]["objects"]) == names, "History observation object set changed")
        if kind == "episode_started":
            require(number == 1, "Episode restarted inside history")
            initial = e["observation"]
            require(initial["hand_empty"] and all(
                "storage_surface" in obj["on"] and not obj["held"]
                for obj in initial["objects"].values()), "Invalid starting relations")
        if kind == "task_started":
            require(active is None and len(completed) < len(tasks), "Another task is still active")
            require(e["task_id"] == case["tasks"][len(completed)]["task_id"], "Wrong task order")
            active = e["task_id"]
            require(active in tasks and active not in completed and not pending, "Invalid task start")
            require(e["initiator"] == tasks[active]["initiator"] and e["executor"] == tasks[active]["executor"], "Wrong task actors")
            require(e["instruction"] == tasks[active]["instruction"], "Wrong task instruction")
        elif kind == "use_claimed":
            require(active == e["task_id"], "Claim outside its task")
            claim = tasks[active].get("use_claim")
            require(claim is not None and all(e.get(k) == v for k, v in claim.items()), "Wrong claim evidence")
            require(e["evidence_source"] == "scenario_authored_declaration", "Claim source missing")
            require(not pending, "Use declaration interrupts a pending action")
            if timings[active] == "before_moves":
                require(not successful[active], "Use declaration must precede this case's later actions")
            else:
                require(len(successful[active]) == 2 * len(tasks[active]["moves"]),
                        "Use declaration must follow all task moves")
            require(active not in claims, "Duplicate use declaration")
            claims.add(active)
        elif kind == "action_requested":
            require(e["task_id"] == active and not pending, "Overlapping action or wrong task")
            require(e["initiator"] == tasks[active]["initiator"] and e["executor"] == tasks[active]["executor"], "Wrong action actors")
            pending[number] = e
        elif kind == "action_succeeded":
            request = pending.pop(e["request_event_id"], None)
            require(request is not None, "Success without matching request")
            require(e["control_step"] > request["control_step"], "No control steps executed for successful action")
            require(all(request[k] == e[k] for k in ("task_id", "action", "object_id", "target_surface")), "Success changes requested action")
            obj = e["observation"]["objects"][e["object_id"]]
            if e["action"] == "GRASP":
                require(obj["held"] and not e["observation"]["hand_empty"], "Grasp success contradicts observed state")
            elif e["action"] == "PLACE_ON_TOP":
                require(e["target_surface"] in obj["on"] and not obj["held"] and e["observation"]["hand_empty"], "Place success contradicts observed state")
            else:
                raise ValueError("Unexpected primitive")
            successful[active].append((e["action"], e["object_id"], e["target_surface"]))
        elif kind == "task_completed":
            require(e["task_id"] == active and not pending, "Invalid completion")
            expected = [(action, obj, target if action == "PLACE_ON_TOP" else None)
                        for obj, target in tasks[active]["moves"] for action in ("GRASP", "PLACE_ON_TOP")]
            require(successful[active] == expected, "Task completion missing successful actions")
            # Earlier successful placements remain verified above; only the last
            # destination per object must still hold at the task boundary.
            require(all(target in e["observation"]["objects"][obj]["on"] for obj, target in dict(tasks[active]["moves"]).items()), "Task completion contradicts observed relations")
            require("use_claim" not in tasks[active] or active in claims, "Required takeover evidence missing")
            completed.add(active)
            active = None
        elif kind == "undo_requested":
            require(completed == set(tasks) and not pending and active is None, "Undo requested before episode completion")
            require(all(e.get(k) == v for k, v in case["undo_request"].items()), "Wrong undo instruction")
            cancelled = True
    require(cancelled, "Final undo request missing")
    return {"events": len(events), "completed_tasks": sorted(completed),
            "successful_primitives": sum(len(v) for v in successful.values()), "claim_tasks": sorted(claims)}


def check_directory(folder, write_report=True):
    case = json.loads((folder / "case.json").read_text())
    events = [json.loads(line) for line in (folder / "public_history.jsonl").read_text().splitlines()]
    result = json.loads((folder / "result.json").read_text())
    if result.get("diagnostic_only") or result.get("history_eligible") is False:
        raise ValueError("Diagnostic prefixes cannot be admitted as complete histories")
    if not result["passed"] or result["worker_exit_code"] != 0:
        raise ValueError("Physical episode did not pass")
    report = check_events(events, case)
    if events[-1]["observation"] != result["final"]:
        raise ValueError("Final history and execution report disagree")
    if result["event_count"] != len(events):
        raise ValueError("Reported event count does not match history")
    if not (folder / "before_undo_scene.json").is_file():
        raise ValueError("Starting scene for next stage missing")
    report["passed"] = True
    if write_report:
        (folder / "history_check.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
