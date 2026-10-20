"""English views of recorded histories, with event bindings and independent labels.

这里只准备数据，不运行策略。译文共享原始物理来源，不能作为新留出样本。
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil

from task_undo.agents.narrative import public_input
from task_undo.application.predictions import validate_physical_start
from task_undo.common.storage import read_json, write_json
from task_undo.common.task_language import require_english_task
from task_undo.domain.history import check_directory
from task_undo.evaluation.compositions import compile_history
from task_undo.evaluation.splits import SOURCE_FILES, digest


def structured_episode_input(folder, observation=None):
    folder = Path(folder)
    return {"history": [json.loads(line) for line in (folder / "public_history.jsonl").read_text().splitlines()],
            "current_state": read_json(folder / "before_undo.json") if observation is None else observation,
            "undo_request": read_json(folder / "case.json")["undo_request"]}


def without_authored_text(value):
    """The only fields permitted to change during the English migration."""
    if isinstance(value, dict):
        return {k: without_authored_text(v) for k, v in value.items()
                if k not in ("instruction", "utterance", "description")}
    if isinstance(value, list):
        return [without_authored_text(v) for v in value]
    return value


def english_view(source, output, instructions, undo_request):
    source, output = Path(source), Path(output)
    check_directory(source, write_report=False)
    original = read_json(source / "case.json")
    case = deepcopy(original)
    if set(instructions) != {t["task_id"] for t in case["tasks"]}:
        raise ValueError("Supply exactly one English instruction per recorded task")
    case["description"] = "English text view of a completed physical history. No physical state or action is changed."
    for task in case["tasks"]:
        task["instruction"] = instructions[task["task_id"]]
        if "use_claim" in task:
            claim = task["use_claim"]
            claim["utterance"] = f"I am reserving the {claim['object_id']} for my own use. Leave it in place."
    case["undo_request"]["utterance"] = undo_request
    old_events = [json.loads(line) for line in (source / "public_history.jsonl").read_text().splitlines()]
    events = deepcopy(old_events)
    tasks = {t["task_id"]: t for t in case["tasks"]}
    for event in events:
        if event["kind"] == "task_started":
            event["instruction"] = tasks[event["task_id"]]["instruction"]
        elif event["kind"] == "use_claimed":
            event["utterance"] = tasks[event["task_id"]]["use_claim"]["utterance"]
        elif event["kind"] == "undo_requested":
            event["utterance"] = undo_request
    if (without_authored_text(original) != without_authored_text(case)
            or without_authored_text(old_events) != without_authored_text(events)):
        raise ValueError("English view changed a physical event or task boundary")
    require_english_task(case, "English physical view")
    require_english_task(events, "English recorded history")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "case.json", case)
    (output / "public_history.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    for filename in ("before_undo.json", "before_undo_scene.json", "result.json"):
        shutil.copyfile(source / filename, output / filename)
    check_directory(output)
    write_json(output / "translation.json", {
        "physical_ancestor": str(source.resolve()), "new_physical_actions": 0,
        "source_sha256": {f: digest(source / f) for f in SOURCE_FILES},
        "nontext_events_identical": True, "physical_checkpoints_identical": True,
        "split_rule": "This view and every descendant remain with the original physical ancestor."})


def render_episode(folder, task_names, objects, surfaces):
    """Render successful placements/claims, never intended but failed actions."""
    folder = Path(folder)
    check_directory(folder, write_report=False)
    case = read_json(folder / "case.json")
    events = [json.loads(line) for line in (folder / "public_history.jsonl").read_text().splitlines()]
    initial = {o: value["on"][0] for o, value in events[0]["observation"]["objects"].items()
               if len(value["on"]) == 1}
    if set(initial) != set(objects) or set(surfaces) != set(case["surfaces"]):
        raise ValueError("Narrative catalogs must match the physical observations")
    order = [t["task_id"] for t in case["tasks"]]
    if set(task_names) != set(order) or len(set(task_names.values())) != len(order):
        raise ValueError("Task names must be complete and unambiguous")
    sequences, lines = [], {"0": [events[0]["event_id"]]}
    state = dict(initial)
    log = ["Initially, " + "; ".join(f"{objects[o]} is on {surfaces[s]}" for o, s in initial.items()) + "."]
    for i, task_id in enumerate(order, 1):
        relevant = [e for e in events if e.get("task_id") == task_id and
                    (e["kind"] == "use_claimed" or
                     (e["kind"] == "action_succeeded" and e["action"] == "PLACE_ON_TOP"))]
        sequence, sentences = [], []
        for event in relevant:
            obj = event["object_id"]
            if event["kind"] == "use_claimed":
                sequence.append(("claim", obj, None))
                sentences.append(f"a family member explicitly reserved {objects[obj]} for this independent activity, leaving it in place")
            else:
                target = event["target_surface"]
                if event["observation"]["objects"][obj]["on"] != [target]:
                    raise ValueError("A reported placement lacks its observed destination")
                sequence.append(("place", obj, target))
                sentences.append(f"moved {objects[obj]} from {surfaces[state[obj]]} to {surfaces[target]}")
                state[obj] = target
        sequences.append(sequence)
        lines[str(i)] = [e["event_id"] for e in relevant]
        log.append(task_names[task_id] + ": " + "; ".join(sentences) + ". " + task_names[task_id] + " completed.")
    target = order.index(case["undo_request"]["target_task_id"])
    public, _, gold = compile_history(initial, sequences, target, objects, surfaces)
    public.update(activity_log=log, undo_request=case["undo_request"]["utterance"])
    for task_id, facts in zip(order, gold["facts"]["tasks"]):
        facts["name"] = task_names[task_id]
    public = public_input({"input": public})
    validate_physical_start(public, case, read_json(folder / "before_undo.json"))
    require_english_task(public, "Physical narrative")
    return public, gold, {"line_event_ids": lines, "task_ids_in_order": order,
                          "history_sha256": digest(folder / "public_history.jsonl"),
                          "labels_source": "Independent forward obligation ledger over observed successful events"}
