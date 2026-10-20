"""Public-history goal inference with one model decision per object.

The compiler checks structure only. It never decides task ownership, reads labels,
or rewrites a model decision using a rule baseline.
"""

import json

from task_undo.agents.direct import SYSTEM_PROMPT as DIRECT_PROMPT, validate_action, validate_plan
from task_undo.common.storage import write_json


SYSTEM_PROMPT = DIRECT_PROMPT.split("\nReturn one JSON object per response", 1)[0] + """

Return one JSON object, with no Markdown. Decide separately for EVERY observed
object using the temporal procedure above. In each row, write evidence and reason
first, then the explicit decision and target. The decision must agree with the reason.
{"kind":"decision_table","items":[
 {"object_id":"actual_object_id","event_ids":[1],"reason":"brief evidence-based explanation",
  "decision":"restore","target_surface":"actual_earlier_surface_id"},
 {"object_id":"another_actual_object_id","event_ids":[2],"reason":"brief evidence-based explanation",
  "decision":"protect","target_surface":null}
]}
The example illustrates the schema only; choose restore or protect independently
for each real object. Include exactly one row per observed object. Cite actual event
IDs supporting the decision, including later use claims when relevant. For restore,
target_surface is the object's support before the cancelled task. For protect it is null.
Use exactly the fields shown. Do not output a separate restore mapping or protect list:
code will compile those directly from your rows without changing your decisions.
This request only infers a goal; do not request physical actions. If evidence is
insufficient, return {"kind":"stop","reason":"what is missing"}.
Invalid structure receives feedback, not an answer or a correctness judgment.
"""


# Used only after a valid table. The original goal-only request stays unchanged.
# Qwen renders the first system message only, so replace it at the phase boundary.
ACTION_PROMPT = DIRECT_PROMPT.split("\nReturn one JSON object per response", 1)[0] + """

The goal-inference phase is complete. Now execute physical undo.
The compiled plan below is your own hypothesis, not a supplied or graded answer.
Do not output another table or plan. Choose one primitive, wait for actual feedback,
and use the updated observation before choosing the next primitive.
{"kind":"act","action":"GRASP","object_id":"...","target_surface":null,"reason":"..."}
{"kind":"act","action":"PLACE_ON_TOP","object_id":"...","target_surface":"...","reason":"..."}
GRASP requires an empty hand; PLACE_ON_TOP requires holding the named object.
Preserve protected objects throughout execution, including never grasping them.
You cannot edit world state, access files, or call other tools. Finish empty-handed:
{"kind":"finish","reason":"brief explanation of observed completion"}.
Before choosing an action, compare the latest observation with your recorded plan.
If every restore target is already satisfied and your hand is empty, use finish.
An empty restore mapping requires no manipulation: with an empty hand, finish
successfully while leaving all protected objects untouched. Zero actions can be
a complete undo; do not invent an action just to have something to execute.
A rejected request was not executed. Reassess the same goal: finish if it is
already satisfied, otherwise choose a permitted action for remaining work.
Rejection alone does not mean the task is impossible.
Use {"kind":"stop","reason":"what is missing or prevents completion"} only when
the goal remains unmet and you cannot proceed because of missing information or
an unrecoverable obstacle. Stop reports an incomplete task, not successful completion.
Invalid requests receive format or mechanical feedback only. Physical failures stop
this prototype. Neither the plan nor completion is graded during this loop.
"""


def compile_plan(table, public):
    """Convert a complete table to the existing plan format, without semantic repair."""
    if (not isinstance(table, dict) or set(table) != {"kind", "items"}
            or table["kind"] != "decision_table" or not isinstance(table["items"], list)):
        raise ValueError("Return kind=decision_table and items only")
    names = set(public["current_state"]["objects"])
    seen, restore, protect, evidence = set(), {}, [], []
    for row in table["items"]:
        if not isinstance(row, dict) or set(row) != {
                "object_id", "event_ids", "reason", "decision", "target_surface"}:
            raise ValueError("Each row needs object_id, event_ids, reason, decision, target_surface")
        name = row["object_id"]
        if not isinstance(name, str) or name not in names or name in seen:
            raise ValueError("Each observed object must occur exactly once")
        seen.add(name)
        if row["decision"] == "restore":
            restore[name] = row["target_surface"]
        elif row["decision"] == "protect" and row["target_surface"] is None:
            protect.append(name)
        else:
            raise ValueError("Use restore with a surface, or protect with null target_surface")
        evidence.append({key: row[key] for key in ("object_id", "event_ids", "reason")})
    if seen != names:
        raise ValueError("The table must cover every observed object")
    plan = {"kind": "plan", "evidence": evidence, "restore": restore, "protect": protect}
    validate_plan(plan, public)
    return plan


def validate_plan_action(command, plan):
    """Check an otherwise valid primitive against the model's own recorded plan.

    This is not an ownership oracle: an incorrect restore decision still permits
    actions on that object. Physical side effects are checked by offline scoring.
    """
    name = command["object_id"]
    if name not in plan["restore"]:
        raise ValueError(f"Your recorded plan protects {name}; do not manipulate it")
    if command["action"] == "PLACE_ON_TOP" and command["target_surface"] != plan["restore"][name]:
        raise ValueError(f"Your recorded plan restores {name} to {plan['restore'][name]}; "
                         "the requested destination differs")


def run_decision_table(model, public, output, config, *, observe=None, execute=None, plan_only=True,
                       plan_guard=False, evidence_guard=False):
    """Infer a table, then optionally act within one shared call/action budget."""
    if not plan_only and (observe is None or execute is None):
        raise ValueError("Physical execution requires observation and action callbacks")
    if plan_guard and plan_only:
        raise ValueError("Plan guard requires physical execution")
    plan = None
    surfaces = set(public["history"][0]["surfaces"])
    prompt, payload, compiler = SYSTEM_PROMPT, public, compile_plan
    if evidence_guard:
        from task_undo.agents.evidence import SYSTEM_PROMPT as evidence_prompt, model_input, compile_evidence
        prompt, payload, compiler = evidence_prompt, model_input(public), compile_evidence
        write_json(output / "evidence_ledger.json", payload["evidence_ledger"])
    messages = [{"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    status = {"execution_completed": False, "stop_reason": "turn_budget", "model_calls": 0, "actions": 0}
    if plan_guard:
        status["rejected_actions"] = 0
    if evidence_guard:
        status.update(goal_assistance="public_history_rules", rejected_tables=0)
    with (output / "agent_trace.jsonl").open("w") as trace:
        def record(kind, **fields):
            trace.write(json.dumps({"kind": kind, **fields}, ensure_ascii=False) + "\n")
            trace.flush()

        def feedback(value):
            record("feedback", **value)
            messages.append({"role": "user", "content": json.dumps(value, ensure_ascii=False)})

        try:
            for turn in range(1, config["max_turns"] + 1):
                status["model_calls"] = turn
                record("model_request", turn=turn, messages=messages)
                response = model.complete(messages)
                record("model_response", turn=turn, **response)
                messages.append({"role": "assistant", "content": response["content"]})
                try:
                    command = json.loads(response["content"])
                    if not isinstance(command, dict):
                        raise ValueError("Response must be a JSON object")
                    kind = command.get("kind")
                    if kind == "stop":
                        status.update(stop_reason="agent_stopped", explanation=str(command.get("reason", "")))
                        break
                    if plan is None:
                        plan = compiler(command, public)
                        write_json(output / ("evidence_table.json" if evidence_guard else "decision_table.json"), command)
                        write_json(output / "agent_plan.json", plan)
                        record("plan_compiled", plan=plan)
                        status["plan_generated"] = True
                        if plan_only:
                            status["stop_reason"] = "plan_only"
                            break
                        messages[0] = {"role": "system", "content": ACTION_PROMPT}
                        feedback({"status": "plan_recorded", "plan": plan, "observation": observe()})
                        continue
                    state = observe()
                    if kind == "finish":
                        if not state["hand_empty"]:
                            raise ValueError("Cannot finish while holding an object")
                        if plan_guard:
                            remaining = {name: target for name, target in plan['restore'].items()
                                         if target not in state['objects'][name]['on']}
                            if remaining:
                                raise ValueError(f"Recorded restore targets are not yet observed: {remaining}")
                        status.update(execution_completed=True, stop_reason="agent_finished",
                                      explanation=str(command.get("reason", "")))
                        break
                    if kind != "act":
                        raise ValueError("Expected act, finish, or stop")
                    validate_action(command, state, surfaces)
                except (ValueError, TypeError) as exc:
                    if evidence_guard and plan is None:
                        status["rejected_tables"] += 1
                    feedback({"status": "invalid_request", "error": str(exc)})
                    continue
                if plan_guard:
                    try:
                        validate_plan_action(command, plan)
                    except ValueError as exc:
                        status["rejected_actions"] += 1
                        record("action_rejected", command=command, error=str(exc))
                        feedback({"status": "plan_violation", "error": str(exc)})
                        continue
                if status["actions"] >= config["max_actions"]:
                    status["stop_reason"] = "action_budget"
                    break
                status["actions"] += 1
                record("action_dispatched", command=command)
                try:
                    execute(command["action"], command["object_id"], command.get("target_surface"))
                except Exception as exc:
                    status["stop_reason"] = "physical_action_failed"
                    record("action_failed", error=str(exc))
                    raise
                feedback({"status": "action_succeeded", "observation": observe()})
        except Exception:
            if status["stop_reason"] != "physical_action_failed":
                status["stop_reason"] = "model_or_runtime_error"
            raise
        finally:
            record("stopped", **status)
            write_json(output / "agent_status.json", status)
    return status
