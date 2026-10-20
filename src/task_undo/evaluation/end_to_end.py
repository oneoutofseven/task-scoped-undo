"""Goal, executor and task metrics have distinct meanings and fixed denominators."""
from collections import Counter

from task_undo.evaluation.goals import score_goal


def trial_metrics(goal, gold, *, attempted, result=None, score=None, execution_audit=None, exit_code=None):
    goal_score = score_goal(goal, gold)
    completed = bool(attempted and exit_code == 0 and result and result.get("passed")
                     and execution_audit and execution_audit.get("audit_passed")
                     and execution_audit.get("completed"))
    physical = bool(completed and score and score.get("valid") and score.get("joint_success"))
    return {"goal_correct": goal_score["goal_correct"], "goal_score": goal_score,
            "physical_attempted": bool(attempted), "predicted_goal_completed": completed,
            "physical_task_success": physical,
            "joint_success": bool(goal_score["goal_correct"] and physical),
            "score_available": bool(score and score.get("valid")),
            "undo_completion_rate": score.get("undo_completion_rate") if score and score.get("valid") else None,
            "protection_violation_rate": score.get("protection_violation_rate") if score and score.get("valid") else None}


def summarize_trials(conditions, methods, rows):
    """An unavailable prediction still occupies its planned method/condition cell."""
    if (not conditions or not methods or len({c["id"] for c in conditions}) != len(conditions)
            or len(set(methods)) != len(methods)):
        raise ValueError("Unique, nonempty conditions and methods are required")
    expected = {(c["id"], method) for c in conditions for method in methods}
    lookup = {(r["condition"], r["method"]): r for r in rows}
    if len(lookup) != len(rows) or not set(lookup) <= expected:
        raise ValueError("Duplicate or unplanned end-to-end outcome")
    summary = {}
    groups = {}
    for condition in conditions:
        groups.setdefault(condition["group"], []).append(condition["id"])
    for method in methods:
        members = [lookup.get((c["id"], method)) for c in conditions]
        counts = {"planned": len(conditions), "recorded": sum(r is not None for r in members),
                  "completed_trials": sum(bool(r and r["status"] == "complete") for r in members)}
        for key in ("goal_correct", "physical_attempted", "predicted_goal_completed", "physical_task_success",
                    "joint_success", "score_available"):
            counts[key] = sum(bool(r and r["metrics"][key]) for r in members)
        counts["planned_source_groups"] = len(groups)
        counts["both_branches_joint_success"] = sum(all(
            lookup.get((name, method), {}).get("metrics", {}).get("joint_success", False) for name in names)
            for names in groups.values())
        counts["failure_stages"] = dict(Counter(r["failure_stage"] for r in members if r and r.get("failure_stage")))
        summary[method] = counts
    return {"status": "complete" if all(v["completed_trials"] == v["planned"] for v in summary.values()) else "incomplete",
            "methods": summary, "planned_trials": len(expected), "recorded_trials": len(rows),
            "scope": "All planned method/condition cells are retained. Branches and translations share source groups; no IID sample claim."}
