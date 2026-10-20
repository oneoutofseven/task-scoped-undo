"""Offline goal accuracy against author labels; never used by the Agent."""


def score_goal(plan, labels):
    """A missing plan is a failure, never removed from the denominator."""
    return {"goal_correct": bool(plan is not None and plan["restore"] == labels["restore"]
                                 and set(plan["protect"]) == set(labels["protect"])),
            "missing_plan": plan is None,
            "wrongly_restored": sorted(set(plan["restore"]) & set(labels["protect"])) if plan else [],
            "missed_restorations": sorted(set(labels["restore"]) - set(plan["restore"])) if plan else sorted(labels["restore"]),
            "wrong_destinations": sorted(k for k in plan["restore"] if k in labels["restore"]
                                         and plan["restore"][k] != labels["restore"][k]) if plan else []}
