"""Validate saved language predictions before physical execution.

No labels and no original structured history are used to choose or repair a goal.
The experiment protocol binds the authored text to its source physical episode.
"""

import json
from pathlib import Path

from task_undo.agents.narrative import public_input, validate_goal, goal_from_facts
from task_undo.agents.narrative_facts import normalize_facts, numbered_input
from task_undo.common.storage import read_json

PREDICTION_FILES = ["goal.json", "facts.json", "status.json", *[
    f"{stage}/{name}.json" for stage in ("extraction", "selection")
    for name in ("request", "response", "parsed")]]


def prediction_files(folder, readout="staged_rules"):
    from task_undo.agents.comparators import MODES
    from task_undo.application.comparator_replay import comparator_files
    from task_undo.application.prediction_replay import TABLE_MODES, table_prediction_files
    folder = Path(folder)
    if readout in MODES:
        return comparator_files(folder)
    if readout in TABLE_MODES:
        return table_prediction_files(folder, readout)
    if readout == "staged_rules":
        return PREDICTION_FILES
    raise ValueError("Unknown prediction readout")


def load_prediction(folder, case, observation, readout="staged_rules"):
    from task_undo.agents.comparators import MODES
    from task_undo.application.comparator_replay import replay_comparator
    from task_undo.application.prediction_replay import TABLE_MODES, replay_table_goal
    folder = Path(folder)
    if readout in MODES:
        public, goal = replay_comparator(folder, readout)
    elif readout in TABLE_MODES:
        public, goal = replay_table_goal(folder, readout)
    elif readout == "staged_rules":
        public, goal = load_staged_goal(folder)
    else:
        raise ValueError("Unknown prediction readout")
    if public["undo_request"] != case["undo_request"].get("utterance"):
        raise ValueError("Prediction undo request differs from physical episode")
    validate_physical_start(public, case, observation)
    return goal


def load_staged_goal(folder):
    """Validate both saved stages; the caller binds the result to its episode."""
    status = read_json(folder / "status.json")
    if (status.get("mode") != "staged_rules" or not status.get("goal_generated")
            or not status.get("schema_valid") or status.get("model_calls") != 2
            or status.get("feedback_retries") != 0 or status.get("physical_execution") is not False):
        raise ValueError("Require a completed two-stage language prediction")
    payloads, parsed = {}, {}
    for stage in ("extraction", "selection"):
        request = read_json(folder / stage / "request.json")
        if (len(request) != 2 or request[0].get("role") != "system"
                or request[1].get("role") != "user"):
            raise ValueError("Unexpected prediction conversation")
        payloads[stage] = json.loads(request[1]["content"])
        response = read_json(folder / stage / "response.json")
        parsed[stage] = read_json(folder / stage / "parsed.json")
        if response.get("finish_reason") != "stop" or json.loads(response["content"]) != parsed[stage]:
            raise ValueError("Parsed prediction differs from model response")
    extraction, selection = payloads["extraction"], payloads["selection"]
    if set(extraction) != {"objects", "surfaces", "current_supports", "activity_log"}:
        raise ValueError("Unexpected extraction input fields")
    if set(selection) != {"activity_log", "tasks", "undo_request"}:
        raise ValueError("Unexpected selection input fields")
    public = public_input({"input": extraction | {
        "activity_log": [row["text"] for row in extraction["activity_log"]],
        "undo_request": selection["undo_request"]}})
    if (numbered_input(public)["activity_log"] != extraction["activity_log"]
            or selection["activity_log"] != extraction["activity_log"]
            or selection["tasks"] != parsed["extraction"]["tasks"]):
        raise ValueError("Prediction stages disagree on history")
    if set(parsed["extraction"]) != {"initial_supports", "tasks"} or set(parsed["selection"]) != {"target_task"}:
        raise ValueError("Unexpected prediction fields")
    facts = normalize_facts(parsed["extraction"] | parsed["selection"], public)
    if facts != read_json(folder / "facts.json"):
        raise ValueError("Saved facts differ from model prediction")
    goal = read_json(folder / "goal.json")
    validate_goal(goal, public)
    if goal != goal_from_facts(facts, public):
        raise ValueError("Saved goal differs from predicted facts")
    return public, goal


def validate_physical_start(public, case, observation):
    scene_objects = {obj["name"] for obj in case["objects"]}
    if (set(public["objects"]) != set(observation["objects"])
            or not set(public["objects"]) <= scene_objects
            or set(public["surfaces"]) != set(case["surfaces"])):
        raise ValueError("Prediction catalog differs from physical episode")
    if not observation["hand_empty"] or any(obj["held"] for obj in observation["objects"].values()):
        raise ValueError("Physical execution requires an empty hand")
    expected = {name: [support] for name, support in public["current_supports"].items()}
    if expected != {name: obj["on"] for name, obj in observation["objects"].items()}:
        raise ValueError("Prediction current supports differ from physical start")
