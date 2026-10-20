"""Bind an authored reference to one accepted physical history for offline scoring."""

import argparse
import hashlib
import json
from pathlib import Path

from task_undo.application.cli import validate_undo
from task_undo.common.storage import read_json


def bind_reference(episode, reference, output):
    goal = read_json(reference)
    case = validate_undo(episode, goal)
    state = read_json(episode / "before_undo.json")
    restore = dict(goal["moves"])
    if len(restore) != len(goal["moves"]):
        raise ValueError("Reference must not assign multiple destinations to an object")
    protected = goal["protected_objects"]
    if any(len(state["objects"][name]["on"]) != 1 or state["objects"][name]["held"]
           for name in protected):
        raise ValueError("Each protected object must start unheld on one known support")
    labels = {
        "protocol_version": 1, "episode_id": case["episode_id"],
        "target_task_id": goal["target_task_id"],
        "source_sha256": {
            name: hashlib.sha256((episode / name).read_bytes()).hexdigest()
            for name in ("before_undo.json", "before_undo_scene.json", "public_history.jsonl")
        },
        "restore": restore,
        "protect": {name: state["objects"][name]["on"][0] for name in protected},
        "position_tolerance_m": goal["position_tolerance_m"],
        "orientation_tolerance_deg": goal["orientation_tolerance_deg"],
        "require_empty_hand": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(labels, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        labels = bind_reference(args.episode, args.reference, args.output)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"labels": str(args.output), "episode_id": labels["episode_id"],
                      "reference_source": "authored", "physical_execution": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
