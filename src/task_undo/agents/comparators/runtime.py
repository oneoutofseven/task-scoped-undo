"""English goal-track ReAct and CaP adapters with deterministic offline replay."""

import copy
import json
from pathlib import Path

from task_undo.agents.narrative import public_input, validate_goal
from task_undo.agents.narrative_controlled import RULES_EN
from task_undo.agents.comparators.program import Program, MissingFunction, LIMITS
from task_undo.common.storage import write_json
from task_undo.common.task_language import require_english_task


# Preserve the same semantics as the controlled baselines; only their output
# instructions differ. No dataset-specific examples or privileged event ledger.
SEMANTICS = RULES_EN.replace(
    "Return JSON only, without actions or explanations. Use English for any generated text.",
    "Use English for any generated text. Return one JSON object per response.")

REACT = SEMANTICS + """
This is a ReAct adaptation for goal inference, not physical action selection.
Alternate a brief evidence-based rationale, a tool action, and its observation.
The complete public input is supplied. Available read-only actions:
{"rationale":"brief evidence","action":"read_history","start":0,"stop":3}
  Returns activity_log[start:stop], with zero-based line indices.
{"rationale":"brief evidence","action":"inspect_object","object_id":"catalog ID"}
  Returns its catalog description/current support and literal matching history
  lines. This is string search, NOT inferred ownership; inspect other lines too.
{"rationale":"brief evidence","action":"finish","goal":{"restore":{},"protect":[]}}
No tool decides the correct goal, validates semantic correctness, or changes the
world. A finish goal is checked for schema only. Tool/schema errors consume a
call. Finish within the stated maximum number of model calls.
"""

CAP = SEMANTICS + """
This is a Code as Policies adaptation for goal inference. Generate a program
that returns the undo goal; the robot executor will use that result unchanged.
Return {"program":"Python source defining build_goal(public)"}.
public is the complete input dictionary. It is the ONLY input available to code.
Define helper functions as needed; if a helper is undefined the interpreter
reports its name. In the next response include the entire program with that
helper defined. Execution errors may be corrected within the same call budget.
There are NO supplied history extractors, task selectors, ownership functions,
or undo-rule helpers. Implement all decisions in your generated program.
The interpreter supports plain functions with positional arguments, return,
assignment, if/else, for/while, break/continue, lists/dicts/sets/tuples, indexing,
slicing, comprehensions, comparisons, booleans, +, -, integer *, // and %.
Builtins: len, list, dict, set, tuple, sorted, str, int, bool, min, max, any, all,
abs, range, enumerate, zip, reversed. No keyword arguments or starred arguments.
String methods: lower, upper, strip, split, rsplit, startswith, endswith,
replace, find, join. Dict: get, keys, values, items, copy, update, pop, setdefault.
List: append, extend, pop, index, count, copy, remove. Set: add, discard, union,
intersection, difference. Top level contains function definitions only.
No imports, external libraries, files, network, exceptions, arbitrary attributes,
decorators, annotations, closures, globals, or access to the Python runtime.
The return value is {"restore":{"object_id":"surface_id"},"protect":["object_id"]}.
Only program/schema errors are reported. A valid but wrong goal is accepted;
there is no correctness feedback or human repair.
"""

CAP_NATIVE = SEMANTICS + """
This is a Code as Policies adaptation for goal inference. Generate a program
that returns the undo goal; the robot executor will use that result unchanged.
Return {"program":"Python source defining build_goal(public)"}.
public is the complete input dictionary. It is the ONLY task input available.
Define helper functions as needed; if a name is undefined execution reports it.
In the next response include the entire replacement program with helpers defined.
Execution errors may be corrected within the same model-call budget.
There are NO supplied history extractors, task selectors, ownership functions,
or undo-rule helpers. Implement all decisions in your generated program.
Code runs in CPython 3.10 with normal Python syntax and the standard library
(including re, json, collections, itertools and math). Nested functions,
closures, default/keyword arguments, f-strings, comprehensions, annotations and
exceptions work normally. Top-level definitions and assignments are allowed.
After executing the source, the runtime calls build_goal(public).
No third-party libraries, host files, network, subprocesses, GPU or simulator
are available. Use deterministic computations over public. Execution is limited
to 200000 policy opcodes, 3 CPU seconds, 10 wall seconds, 256 MiB of memory,
32000 source characters and 65536 output bytes. Printed text is discarded.
The return value is {"restore":{"object_id":"surface_id"},"protect":["object_id"]}.
Only program/schema errors are reported. A valid but wrong goal is accepted;
there is no correctness feedback or human repair.
"""

PROMPTS = {"react_goal_v1": REACT, "cap_goal_v1": CAP, "cap_goal_v2": CAP_NATIVE}


def program_settings(mode, max_calls):
    settings = {"mode": mode, "max_calls": max_calls, "program_limits": LIMITS}
    if mode == "cap_goal_v2":
        from task_undo.agents.comparators.native_program import LIMITS as NATIVE_LIMITS, runtime_identity
        settings.update(program_limits=NATIVE_LIMITS, runtime=runtime_identity())
    return settings


def inspect_history(public, command):
    action = command.get("action")
    if not isinstance(command.get("rationale"), str) or not command["rationale"].strip():
        raise ValueError("Provide a brief rationale")
    if action == "finish":
        if set(command) != {"rationale", "action", "goal"}:
            raise ValueError("finish requires rationale, action and goal")
        validate_goal(command["goal"], public)
        return command["goal"], {"finished": True}
    if action == "read_history":
        if set(command) != {"rationale", "action", "start", "stop"}:
            raise ValueError("read_history requires start and stop")
        start, stop = command["start"], command["stop"]
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= len(public["activity_log"]):
            raise ValueError("Use a nonempty valid zero-based history slice")
        lines = list(enumerate(public["activity_log"]))[start:stop]
        return None, {"lines": [{"line": i, "text": text} for i, text in lines]}
    if action == "inspect_object":
        if set(command) != {"rationale", "action", "object_id"}:
            raise ValueError("inspect_object requires object_id")
        name = command["object_id"]
        if not isinstance(name, str) or name not in public["objects"]:
            raise ValueError("Unknown object ID")
        description = public["objects"][name]
        lines = [{"line": i, "text": text} for i, text in enumerate(public["activity_log"])
                 if name.lower() in text.lower() or description.lower() in text.lower()]
        return None, {"object_id": name, "description": description,
                      "current_support": public["current_supports"][name], "literal_matches": lines}
    raise ValueError("Unknown read-only action")


def run(model, public, output, mode, *, max_calls=4):
    if mode not in PROMPTS or type(max_calls) is not int or not 1 <= max_calls <= 8:
        raise ValueError("Require a supported adapter and 1..8 model calls")
    public = public_input({"input": public})
    require_english_task(public, "Comparator public input")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Comparator output must be empty")
    settings = program_settings(mode, max_calls)
    write_json(output / "input.json", public)
    write_json(output / "settings.json", settings)
    messages = [{"role": "system", "content": PROMPTS[mode]},
                {"role": "user", "content": json.dumps(
                    {"public": public, "max_model_calls": max_calls}, ensure_ascii=False)}]
    status = {"mode": mode, "model_calls": 0, "feedback_retries": 0,
              "goal_generated": False, "schema_valid": False, "physical_execution": False,
              "track": "goal_inference", "termination": "call_budget_exhausted"}
    for index in range(max_calls):
        folder = output / f"turn_{index:02d}"
        folder.mkdir()
        write_json(folder / "request.json", messages)
        status["model_calls"] += 1
        try:
            response = model.complete(copy.deepcopy(messages))
        except RuntimeError as exc:
            status.update(termination="model_error", error=str(exc))
            write_json(folder / "model_error.json", {"error": str(exc)})
            break
        write_json(folder / "response.json", response)
        goal = None
        try:
            if response.get("finish_reason") != "stop":
                raise ValueError("Incomplete model response")
            require_english_task(response["content"], "Comparator response")
            command = json.loads(response["content"])
            if not isinstance(command, dict):
                raise ValueError("Return a JSON object")
            if mode == "react_goal_v1":
                goal, feedback = inspect_history(public, command)
            else:
                if set(command) != {"program"}:
                    raise ValueError("Return program only")
                backend = Program
                if mode == "cap_goal_v2":
                    from task_undo.agents.comparators.native_program import NativeProgram
                    backend = NativeProgram
                program = backend(command["program"])
                goal = program.run(public)
                validate_goal(goal, public)
                write_json(folder / "program_result.json", {"goal": goal, "steps": program.steps})
                feedback = {"finished": True}
        except (ValueError, KeyError, TypeError) as exc:
            goal = None
            feedback = {"error": str(exc), "semantic_feedback": False}
            if isinstance(exc, MissingFunction):
                feedback["undefined_helper"] = exc.name
            if mode == "cap_goal_v2" and getattr(exc, "name", None):
                feedback["undefined_helper"] = exc.name
            status["feedback_retries"] += 1
        write_json(folder / "observation.json", feedback)
        if goal is not None:
            write_json(output / "goal.json", goal)
            status.update(goal_generated=True, schema_valid=True, termination="goal_returned")
            break
        messages.extend([{"role": "assistant", "content": response["content"]},
                         {"role": "user", "content": json.dumps(
                             {"observation": feedback, "remaining_model_calls": max_calls - index - 1},
                             ensure_ascii=False)}])
    write_json(output / "status.json", status)
    return status
