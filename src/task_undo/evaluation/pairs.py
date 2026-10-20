"""Dataset construction: append declared, zero-motion activities to a real prefix.

This module may read author labels; policies and model prompts never call it.
The branches share physical execution. They are not two new simulator rollouts.
"""

import copy
import hashlib
import json
from pathlib import Path
import shutil

from task_undo.domain.history import check_directory, check_events
from task_undo.common.storage import read_json, write_json
from task_undo.common.paths import freeze_package


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def branch_history(events, case, followup):
    """Preserve every pre-cancellation event, then append a scripted declaration."""
    if followup["moves"] or "use_claim" not in followup:
        raise ValueError("Only a use declaration with zero movements can share this checkpoint")
    if followup["task_id"] in {task["task_id"] for task in case["tasks"]}:
        raise ValueError("Follow-up task ID already exists")
    check_events(events, case)
    final = events[-1]
    if followup["use_claim"]["object_id"] not in final["observation"]["objects"]:
        raise ValueError("Claimed object does not exist")
    branched_case = copy.deepcopy(case)
    branched_case["tasks"].append(copy.deepcopy(followup))
    branched = copy.deepcopy(events[:-1])

    def append(kind, **fields):
        branched.append({"event_id": len(branched) + 1, "episode_id": case["episode_id"],
                         "kind": kind, "control_step": final["control_step"],
                         "sim_seconds": final["sim_seconds"], **fields})

    append("task_started", task_id=followup["task_id"], initiator=followup["initiator"],
           executor=followup["executor"], instruction=followup["instruction"])
    append("use_claimed", task_id=followup["task_id"], evidence_source="scenario_authored_declaration",
           **followup["use_claim"])
    append("task_completed", task_id=followup["task_id"], observation=copy.deepcopy(final["observation"]))
    append("undo_requested", **case["undo_request"], observation=copy.deepcopy(final["observation"]))
    check_events(branched, branched_case)
    return branched_case, branched


def branch_declarations(events, case, variant):
    """Append one or several declarations without rewriting the physical prefix."""
    if ("followup_task" in variant) == ("followup_tasks" in variant):
        raise ValueError("Specify either followup_task or followup_tasks")
    followups = variant.get("followup_tasks", [variant.get("followup_task")])
    if not isinstance(followups, list) or not followups:
        raise ValueError("At least one follow-up declaration is required")
    for followup in followups:
        case, events = branch_history(events, case, followup)
    return case, events


def build_pair(source, output, specification):
    check_directory(source, write_report=False)
    spec = read_json(specification)
    case = read_json(source / "case.json")
    if case["episode_id"] != spec["source_episode_id"]:
        raise ValueError("Wrong source episode")
    if set(spec["source_sha256"]) != {"before_undo.json", "before_undo_scene.json", "public_history.jsonl"}:
        raise ValueError("Pair specification must bind the complete source")
    for name, expected in spec["source_sha256"].items():
        if digest(source / name) != expected:
            raise ValueError(f"Source differs from the fixed pair specification: {name}")
    state = read_json(source / "before_undo.json")
    if state != read_json(source / "result.json")["final"]:
        raise ValueError("Source observation disagrees with physical result")
    history = [json.loads(line) for line in (source / "public_history.jsonl").read_text().splitlines()]
    prepared = {}
    for name, variant in spec["variants"].items():
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("Invalid variant directory name")
        branch_case, events = branch_declarations(history, case, variant)
        restore = variant["restore"]
        if not isinstance(restore, dict) or not set(restore) <= set(state["objects"]):
            raise ValueError("Invalid restoration label")
        if not all(target in case["surfaces"] for target in restore.values()):
            raise ValueError("Unknown restoration surface")
        prepared[name] = (branch_case, events)
    if len(prepared) != 2:
        raise ValueError("This experiment requires exactly two variants")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(specification, output / "specification.json")
    freeze_package(output)
    (output / "private").mkdir()
    entries = {}
    for name, (branch_case, events) in prepared.items():
        folder = output / name
        folder.mkdir()
        write_json(folder / "case.json", branch_case)
        for filename in ("before_undo.json", "before_undo_scene.json"):
            shutil.copyfile(source / filename, folder / filename)
        (folder / "public_history.jsonl").write_text(
            "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events))
        provenance = {"kind": "shared_physical_prefix_with_scripted_declaration",
                      "physical_source": str(source.resolve()), "new_physical_actions": 0,
                      "prefix_events_preserved": len(history) - 1,
                      "source_sha256": {filename: digest(source / filename) for filename in
                                        ("case.json", "result.json", *spec["source_sha256"])}}
        write_json(folder / "provenance.json", provenance)
        result = read_json(source / "result.json")
        result.update(event_count=len(events), completed_tasks=[t["task_id"] for t in branch_case["tasks"]],
                      source_kind=provenance["kind"], new_physical_actions=0,
                      worker_exit_code_source="shared physical prefix; no new worker was launched")
        write_json(folder / "result.json", result)
        check_directory(folder)
        restore = spec["variants"][name]["restore"]
        protected = {obj: value["on"][0] for obj, value in state["objects"].items() if obj not in restore}
        labels = {"protocol_version": 1, "episode_id": case["episode_id"],
                  "target_task_id": case["undo_request"]["target_task_id"],
                  "source_sha256": {filename: digest(folder / filename) for filename in spec["source_sha256"]},
                  "restore": restore, "protect": protected, "position_tolerance_m": 0.02,
                  "orientation_tolerance_deg": 5.0, "require_empty_hand": True}
        write_json(output / "private" / f"{name}.labels.json", labels)
        write_json(output / "private" / f"{name}.goal.json", {
            "episode_id": case["episode_id"], "target_task_id": labels["target_task_id"],
            "goal_source": "human_reference", "moves": list(map(list, restore.items())),
            "protected_objects": list(protected), "position_tolerance_m": 0.02, "orientation_tolerance_deg": 5.0})
        entries[name] = {"episode": name, "labels": f"private/{name}.labels.json",
                         "reference_goal": f"private/{name}.goal.json"}
    write_json(output / "pair.json", {"pair_id": spec["pair_id"], "physical_source": str(source.resolve()),
                                      "variants": entries, "physical_rollouts_completed": False})
    write_json(output / "manifest.json", {"sha256": {
        str(path.relative_to(output)): digest(path) for path in output.rglob("*") if path.is_file()}})
    report = check_pair(output)
    write_json(output / "pair_check.json", report)
    return report


def check_pair(folder):
    for name, expected in read_json(folder / "manifest.json")["sha256"].items():
        if digest(folder / name) != expected:
            raise ValueError(f"Pair artifact changed: {name}")
    pair = read_json(folder / "pair.json")
    spec = read_json(folder / "specification.json")
    branches = [folder / entry["episode"] for entry in pair["variants"].values()]
    if len(branches) != 2:
        raise ValueError("Expected two branches")
    histories = []
    for branch in branches:
        check_directory(branch, write_report=False)
        provenance = read_json(branch / "provenance.json")
        source = Path(provenance["physical_source"])
        for name, expected in provenance["source_sha256"].items():
            if digest(source / name) != expected:
                raise ValueError("Physical provenance changed")
        events = [json.loads(line) for line in (branch / "public_history.jsonl").read_text().splitlines()]
        source_events = [json.loads(line) for line in (source / "public_history.jsonl").read_text().splitlines()]
        n = provenance["prefix_events_preserved"]
        if n != len(source_events) - 1 or events[:n] != source_events[:-1]:
            raise ValueError("A past physical event was changed")
        branch_case = read_json(branch / "case.json")
        expected_case, expected_events = branch_declarations(
            source_events, read_json(source / "case.json"), spec["variants"][branch.name])
        if branch_case != expected_case or events != expected_events:
            raise ValueError("Branch is not the original prefix plus its zero-motion declarations")
        for filename in ("before_undo.json", "before_undo_scene.json"):
            if (branch / filename).read_bytes() != (source / filename).read_bytes():
                raise ValueError("Branch changed the source physical state")
        histories.append(events)
    a, b = branches
    for filename in ("before_undo_scene.json", "before_undo.json"):
        if (a / filename).read_bytes() != (b / filename).read_bytes():
            raise ValueError("Paired current scenes differ")
    if read_json(a / "case.json")["undo_request"] != read_json(b / "case.json")["undo_request"]:
        raise ValueError("Paired undo instructions differ")
    action_events = lambda events: [e for e in events if e["kind"] in ("action_requested", "action_succeeded")]
    if action_events(histories[0]) != action_events(histories[1]):
        raise ValueError("Paired physical action histories differ")
    if histories[0] == histories[1]:
        raise ValueError("Paired histories must differ")
    return {"passed": True, "same_scene_bytes": True, "same_observation_bytes": True,
            "same_undo_instruction": True, "same_physical_actions": True,
            "different_history": True, "new_physical_actions_in_branching": 0,
            "physical_undo_runs_verified": False}


def compare_rule_goals(folder):
    from task_undo.domain.policies import POLICIES, infer_goal

    check_pair(folder)
    pair = read_json(folder / "pair.json")
    rows = []
    for name, entry in pair["variants"].items():
        branch = folder / entry["episode"]
        labels = read_json(folder / entry["labels"])
        for filename, expected in labels["source_sha256"].items():
            if digest(branch / filename) != expected:
                raise ValueError("Labels do not match branch inputs")
        public = {"history": [json.loads(line) for line in (branch / "public_history.jsonl").read_text().splitlines()],
                  "current_state": read_json(branch / "before_undo.json"),
                  "undo_request": read_json(branch / "case.json")["undo_request"]}
        for method in POLICIES:
            goal = infer_goal(method, public)
            correct = goal["restore"] == labels["restore"] and set(goal["protect"]) == set(labels["protect"])
            rows.append({"variant": name, "method": method, "goal": goal, "goal_correct": correct})
    report = {"scope": "goal inference only; no physical rollout scores", "rows": rows}
    write_json(folder / "rule_goals.json", report)
    return report
