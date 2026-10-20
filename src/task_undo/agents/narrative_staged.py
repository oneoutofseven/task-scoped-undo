"""Extract the full history independently of the requested cancellation.

Two fixed model calls, no answer feedback. Only the second call sees the request.
Keeping factual extraction request-independent also supports history reuse.
"""

import json

from task_undo.agents.narrative import public_input, validate_facts, goal_from_facts
from task_undo.agents.narrative_facts import numbered_input, normalize_facts
from task_undo.common.storage import write_json

FACT_PROMPT = """You are an activity-history recorder. Read the evidence line by line in chronological order and extract all completed activities.
The input contains object and surface IDs, current supports, and activity text with line IDs. Return JSON only, using English for generated names.
This step has no undo request: record every activity without selecting objects to restore or protect.

Output fields in this order:
initial_supports: each object's location before all activities, using input catalog IDs as keys and values.
tasks: all activities ordered by start time, each with a unique English name and chronological events.
An activity can span multiple lines. List later independent activities separately; a use declaration without movement is also an activity.
Each event has exactly kind, object_id, target_surface, and line_id:
- For kind=place, record a completed placement with the destination surface ID.
- For kind=claim, record an explicit declaration reserving an object for someone's use; target_surface is null.
- Copy line_id exactly from the evidence line containing the event; one line may contain multiple events.

Attribution: an event belongs to the activity during which it occurred. A family member's later declaration of personal use belongs to that later activity, not the activity that first brought out the object.
Completeness: check each line for placements and use declarations. Statements such as "keep this for me" produce a claim even without movement.
If an activity first claims use and then moves an object, record both claim and place; the move does not replace the claim.
Do not invent claims without explicit evidence. Initial locations are not activities or events.
Do not output grasps, cancellations, undo actions, plans, or actions that did not occur. Do not omit earlier moves based on current locations.

Output only this structure (the example shows fields, not event counts or actual facts):
{"initial_supports":{"object_id":"surface_id"},
 "tasks":[{"name":"unique activity name","events":[
 {"kind":"place","object_id":"object_id","target_surface":"surface_id","line_id":"L1"},
 {"kind":"claim","object_id":"object_id","target_surface":null,"line_id":"L2"}]}]}
"""

TARGET_PROMPT = """Select the cancelled activity from the given activity list using the user's undo request.
Activities are chronological and their names were extracted by the recorder. The original text is provided to resolve the request's reference.
Output only {"target_task":"activity name"}, copying one name from the list exactly.
Do not use line IDs or indices, modify the history, or decide which objects to restore or protect. Use English for generated text.
"""


def extraction_input(public):
    result = numbered_input(public)
    del result["undo_request"]
    return result


def run_staged(model, public, output):
    public = public_input({"input": public})
    status = {"mode": "staged_rules", "model_calls": 0, "goal_generated": False,
              "schema_valid": False, "physical_execution": False, "feedback_retries": 0}

    def call(stage, prompt, payload):
        (output / stage).mkdir(parents=True, exist_ok=True)
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        write_json(output / stage / "request.json", messages)
        status["model_calls"] += 1
        response = model.complete(messages)
        write_json(output / stage / "response.json", response)
        parsed = json.loads(response["content"])
        write_json(output / stage / "parsed.json", parsed)
        return parsed

    try:
        extracted = call("extraction", FACT_PROMPT, extraction_input(public))
        if not isinstance(extracted, dict) or set(extracted) != {"initial_supports", "tasks"}:
            raise ValueError("Extraction must return initial_supports and tasks only")
        # The temporary first-task reference is only for structural validation.
        # It is neither saved as a prediction nor passed to goal inference.
        tasks = extracted["tasks"]
        if not isinstance(tasks, list) or not tasks or not isinstance(tasks[0], dict):
            raise ValueError("Extraction requires a nonempty task list")
        structural = normalize_facts(extracted | {"target_task": tasks[0].get("name")}, public)
        validate_facts(structural, public)
        selected = call("selection", TARGET_PROMPT, {
            "activity_log": numbered_input(public)["activity_log"],
            "tasks": tasks, "undo_request": public["undo_request"]})
        if not isinstance(selected, dict) or set(selected) != {"target_task"}:
            raise ValueError("Selection must return target_task only")
        facts = normalize_facts(extracted | selected, public)
        validate_facts(facts, public)
        status["schema_valid"] = True
        write_json(output / "facts.json", facts)
        goal = goal_from_facts(facts, public)
        write_json(output / "goal.json", goal)
        status["goal_generated"] = True
    except (ValueError, KeyError, TypeError, RuntimeError) as exc:
        status["error"] = str(exc)
    write_json(output / "status.json", status)
    return status
