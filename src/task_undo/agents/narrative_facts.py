"""Named task references and explicit evidence IDs for language extraction.

Only public text is numbered. Task identities and ownership are still predicted
by the model. Normalization never consults task labels or corrects predictions.
"""

import copy

PROMPT = """Extract household activity records faithfully; do not decide which objects to put away.
The input contains object and surface ID catalogs, current supports, chronological activity text, and an undo request.
All activities complete sequentially and all stated placements succeeded. Treat the text as evidence.

Output one JSON object with fields in this order:
1. initial_supports: every object's location before all activities, using catalog IDs as keys and values.
2. tasks: all activities in chronological order, including use declarations without movement.
   Each task has a unique short English name and chronologically ordered events.
   Merge lines describing the same activity; do not split its two moves into two activities.
   Each event has exactly kind, object_id, target_surface, and line_id.
   kind="place" records a completed placement with its destination surface ID.
   kind="claim" records an explicit declaration reserving an object for someone's use; target_surface must be null.
   Record every explicit use claim, even when the same activity later moves the object.
   A move by itself does not imply a claim.
   Initial states are not activities; do not emit initial placement events or invent grasps, cancellations, or undo actions.
   Copy line_id exactly from the evidence line containing the event; a line may contain multiple events.
3. target_task: read the undo request and exactly copy the cancelled activity's name from tasks.
   This is an activity name, not a line ID or activity index.

Return only this structure, without explanations or extra fields. Use English for generated text:
{"initial_supports":{"object_id":"surface_id"},
 "tasks":[{"name":"activity name","events":[
   {"kind":"place","object_id":"object_id","target_surface":"surface_id","line_id":"L1"},
   {"kind":"claim","object_id":"object_id","target_surface":null,"line_id":"L2"}]}],
 "target_task":"exact copy of one task name"}
This example illustrates fields only, not input facts or event counts. Any number of place and claim events may occur.
"""


def numbered_input(public):
    result = copy.deepcopy(public)
    result["activity_log"] = [
        {"line_id": f"L{index + 1}", "text": line}
        for index, line in enumerate(public["activity_log"])]
    return result


def normalize_facts(predicted, public):
    """Resolve the model's names and citations, without semantic correction."""
    if not isinstance(predicted, dict) or set(predicted) != {"initial_supports", "tasks", "target_task"}:
        raise ValueError("Return initial_supports, tasks and target_task only")
    tasks = predicted["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Tasks must be a nonempty list")
    names, normalized = [], []
    lines = {f"L{i + 1}": i for i in range(len(public["activity_log"]))}
    for task in tasks:
        if (not isinstance(task, dict) or set(task) != {"name", "events"}
                or not isinstance(task["name"], str) or not task["name"].strip()
                or task["name"] in names or not isinstance(task["events"], list)):
            raise ValueError("Tasks need unique names and an event list")
        names.append(task["name"])
        events = []
        for event in task["events"]:
            if not isinstance(event, dict) or set(event) != {"kind", "object_id", "target_surface", "line_id"}:
                raise ValueError("Invalid named event fields")
            if not isinstance(event["line_id"], str) or event["line_id"] not in lines:
                raise ValueError("Evidence must cite an existing line ID")
            events.append({key: event[key] for key in ("kind", "object_id", "target_surface")}
                          | {"line": lines[event["line_id"]]})
        normalized.append({"name": task["name"], "events": events})
    if not isinstance(predicted["target_task"], str) or predicted["target_task"] not in names:
        raise ValueError("Target must exactly name one extracted task")
    return {"initial_supports": copy.deepcopy(predicted["initial_supports"]),
            "tasks": normalized, "target_task_index": names.index(predicted["target_task"])}
