"""Audit fixed controlled arms, including invalid predictions, without gold."""

from pathlib import Path
import tempfile

from task_undo.agents.narrative import public_input
from task_undo.agents.narrative_controlled import run_controlled
from task_undo.common.storage import read_json, write_json


def record_controlled(model, public, folder, mode):
    """Keep model-service errors replayable without modifying the frozen method."""
    write_json(folder / "input.json", public)

    class Recorder:
        index = 0

        def complete(self, messages, *, response_schema=None):
            stage = ("first", "second")[self.index]
            self.index += 1
            try:
                return model.complete(messages, response_schema=response_schema)
            except RuntimeError as exc:
                write_json(folder / stage / "model_error.json", {"error": str(exc)})
                raise

    return run_controlled(Recorder(), public, folder, mode)


def replay_controlled(folder, mode, readout):
    if (mode, readout) not in (
        ("controlled_table_direct", "object_table_direct_v1"),
        ("controlled_table_shared", "object_table_rule_v1"),
    ):
        raise ValueError("Unknown controlled comparison arm")
    folder = Path(folder)
    public = public_input({"input": read_json(folder / "input.json")})

    class Replay:
        index = 0

        def complete(self, messages, *, response_schema=None):
            stage = ("first", "second")[self.index]
            self.index += 1
            if read_json(folder / stage / "request.json") != messages:
                raise ValueError("Controlled request differs from method")
            schema = folder / stage / "response_schema.json"
            if (schema.exists() != (response_schema is not None)
                    or schema.exists() and read_json(schema) != response_schema):
                raise ValueError("Controlled response schema differs")
            if (folder / stage / "model_error.json").exists():
                if (folder / stage / "response.json").exists():
                    raise ValueError("Both response and error were recorded")
                raise RuntimeError(read_json(folder / stage / "model_error.json")["error"])
            return read_json(folder / stage / "response.json")

    replay = Replay()
    with tempfile.TemporaryDirectory(prefix="undo_controlled_audit_") as temp:
        generated = Path(temp)
        status = record_controlled(replay, public, generated, mode)
        names = [str(p.relative_to(generated)) for p in sorted(generated.rglob("*.json"))]
        original = [str(p.relative_to(folder)) for p in sorted(folder.rglob("*.json"))]
        if names != original or status != read_json(folder / "status.json"):
            raise ValueError("Controlled artifact set/status differs from replay")
        for name in names:
            if read_json(generated / name) != read_json(folder / name):
                raise ValueError("Controlled artifact differs: " + name)
    goal_file = "rule_goal.json" if readout == "object_table_rule_v1" else "goal.json"
    return public, read_json(folder / goal_file) if (folder / goal_file).exists() else None
