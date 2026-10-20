"""Published-method adaptations for the goal inference track.

These are ReAct/CaP-inspired task adapters, not original benchmark reproductions.
They never consume private labels or control the simulator directly.
"""

MODES = ("react_goal_v1", "cap_goal_v1", "cap_goal_v2")


def run_comparator(model, public, output, mode, *, max_calls=4):
    from task_undo.agents.comparators.runtime import run
    return run(model, public, output, mode, max_calls=max_calls)
