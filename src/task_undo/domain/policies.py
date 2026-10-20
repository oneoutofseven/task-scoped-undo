"""Small public-history baselines; no labels, scene resets or private goals."""


POLICIES = ("noop", "restore_initial", "last_writer", "claim_aware")


def _objects_before_task(history, target):
    """Read the task-start boundary in the current sequential-task histories.

    task_started usually carries no observation, so use the latest observation
    at or before that marker. Never fall back to the episode's initial state
    when the target boundary is missing or ambiguous.
    """
    starts = [i for i, event in enumerate(history)
              if event["kind"] == "task_started" and event["task_id"] == target]
    if len(starts) != 1:
        raise ValueError("Rule baseline requires exactly one target task start")
    for event in reversed(history[:starts[0] + 1]):
        if "observation" in event:
            return event["observation"]["objects"]
    raise ValueError("Rule baseline requires an observation before the target task")


def infer_goal(policy, public):
    history, current = public["history"], public["current_state"]
    target = public["undo_request"]["target_task_id"]
    initial = history[0]["observation"]["objects"]
    names = set(current["objects"])
    if set(initial) != names:
        raise ValueError("Object set changed between initial and current state")
    if policy not in POLICIES:
        raise ValueError(f"Unknown history baseline: {policy}")
    last_write, last_target_write = {}, {}
    for event in history:
        if event["kind"] == "action_succeeded" and event["action"] == "PLACE_ON_TOP":
            name = event["object_id"]
            last_write[name] = event["task_id"]
            if event["task_id"] == target:
                last_target_write[name] = event["event_id"]
    if policy == "noop":
        selected = set()
    elif policy == "restore_initial":
        selected = names
    else:
        selected = {name for name in names if last_write.get(name) == target}
        if policy == "claim_aware":
            for event in history:
                if (event["kind"] == "use_claimed" and event["task_id"] != target
                        and event["event_id"] > last_target_write.get(event["object_id"], float("inf"))):
                    selected.discard(event["object_id"])
    # Restore-all deliberately remains an episode-initial baseline. Selective
    # policies must preserve valid work that preceded the cancelled task.
    reference = (_objects_before_task(history, target)
                 if policy in ("last_writer", "claim_aware") else initial)
    if set(reference) != names:
        raise ValueError("Object set changed at the target task boundary")
    restore = {}
    for name in sorted(selected):
        supports = reference[name]["on"]
        if len(supports) != 1 or reference[name]["held"]:
            raise ValueError("Rule baseline requires one known unheld reference support")
        restore[name] = supports[0]
    return {"policy": policy, "restore": restore, "protect": sorted(names - selected)}
