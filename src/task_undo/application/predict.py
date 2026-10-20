"""Portable goal-only entry point for the existing two-call language pipeline."""

import argparse
import json
from pathlib import Path

from task_undo.agents.model import ChatModel, validate_config
from task_undo.agents.narrative import public_input
from task_undo.agents.narrative_controlled import run_controlled
from task_undo.common.paths import freeze_package
from task_undo.common.storage import read_json, write_json
from task_undo.common.task_language import require_english_task

METHODS = {
    "direct": ("controlled_table_direct", "goal_generated", "goal.json"),
    "facts-rule": ("controlled_table_shared", "rule_goal_generated", "rule_goal.json"),
    "no-history": ("controlled_table_no_history", "goal_generated", "goal.json"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        public = public_input({"input": read_json(args.input)})
        require_english_task(public, "Public narrative")
        config = read_json(args.config)
        validate_config(config)
        args.output.mkdir(parents=True, exist_ok=False)
        freeze_package(args.output)
        write_json(args.output / "public_input.json", public)
        write_json(args.output / "model_config.json", config)
        mode, success_key, goal_file = METHODS[args.method]
        write_json(args.output / "method.json", {
            "method": args.method, "mode": mode, "selected_goal_file": goal_file,
            "physical_execution": False,
        })
        status = run_controlled(ChatModel(config), public, args.output, mode)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"method": args.method, "selected_goal_file": goal_file, **status}, indent=2))
    return 0 if status.get(success_key) else 1


if __name__ == "__main__":
    raise SystemExit(main())
