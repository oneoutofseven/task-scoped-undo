"""Run the real rule on authored facts; no model or simulator is involved."""

import json
from pathlib import Path

from task_undo.agents.narrative import goal_from_facts, public_input
from task_undo.agents.narrative_facts import normalize_facts


def authored_facts(adopted):
    """Explicit synthetic annotations, not facts extracted from text by a model."""
    return {
        "initial_supports": dict.fromkeys(("cup", "book", "plate"), "storage_surface"),
        "target_task": "setup",
        "tasks": [
            {"name": "setup", "events": [
                {"kind": "place", "object_id": name, "target_surface": "dining_table",
                 "line_id": "L2"} for name in ("cup", "book")]},
            {"name": "snack", "events": [
                {"kind": "place", "object_id": "plate", "target_surface": "dining_table",
                 "line_id": "L3"}]},
            {"name": "personal use", "events": [
                {"kind": "claim", "object_id": adopted, "target_surface": None,
                 "line_id": "L4"}]},
        ],
    }


def contrast():
    examples = Path(__file__).resolve().parent
    public, goals = {}, {}
    for adopted in ("book", "cup"):
        record = json.loads((examples / f"{adopted}_adopted.json").read_text())
        public[adopted] = public_input({"input": record})
        facts = normalize_facts(authored_facts(adopted), public[adopted])
        goals[f"{adopted}_adopted"] = goal_from_facts(facts, public[adopted])
    return {
        "synthetic_example": True,
        "model_calls": 0,
        "physical_execution": False,
        "same_current_supports": public["book"]["current_supports"] == public["cup"]["current_supports"],
        "same_request": public["book"]["undo_request"] == public["cup"]["undo_request"],
        "goals": goals,
    }


if __name__ == "__main__":
    print(json.dumps(contrast(), indent=2))
