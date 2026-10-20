"""Replay saved comparator calls and tool observations without inference or gold."""

from pathlib import Path
import tempfile

from task_undo.agents.comparators import MODES, run_comparator
from task_undo.agents.comparators.runtime import program_settings
from task_undo.agents.narrative import public_input
from task_undo.common.storage import read_json


def comparator_files(folder):
    folder = Path(folder)
    names = ["input.json", "settings.json", "status.json"]
    if (folder / "goal.json").exists():
        names.append("goal.json")
    names.extend(str(p.relative_to(folder)) for p in sorted(folder.glob("turn_*/*.json")))
    return names


def replay_comparator(folder, mode, *, require_goal=True):
    folder = Path(folder)
    settings = read_json(folder / "settings.json")
    if (mode not in MODES or settings != program_settings(mode, settings.get("max_calls"))):
        raise ValueError("Comparator settings differ from the declared method")
    public = public_input({"input": read_json(folder / "input.json")})

    class Replay:
        index = 0

        def complete(self, messages):
            turn = folder / f"turn_{self.index:02d}"
            if messages != read_json(turn / "request.json"):
                raise ValueError("Comparator request differs from the declared method")
            self.index += 1
            if (turn / "model_error.json").exists():
                if (turn / "response.json").exists():
                    raise ValueError("A call cannot have both a response and model error")
                raise RuntimeError(read_json(turn / "model_error.json")["error"])
            return read_json(turn / "response.json")

    replay = Replay()
    with tempfile.TemporaryDirectory(prefix="undo_comparator_replay_") as temp:
        generated = Path(temp)
        status = run_comparator(replay, public, generated, mode, max_calls=settings["max_calls"])
        if require_goal and not status["goal_generated"]:
            raise ValueError("Comparator did not return a valid goal")
        expected, actual = comparator_files(folder), comparator_files(generated)
        if expected != actual or status["model_calls"] != replay.index:
            raise ValueError("Comparator artifact set differs from replay")
        for name in actual:
            if read_json(generated / name) != read_json(folder / name):
                raise ValueError(f"Comparator artifact differs from replay: {name}")
    return public, read_json(folder / "goal.json") if status["goal_generated"] else None
