"""Run an observed history prefix through one selected transfer, without rewind.

A successful diagnostic is deliberately not an accepted episode. The original
case, task boundaries, action order, seed and action checks remain in use.
"""

from task_undo.common.storage import write_json


def transfer_plan(case):
    return [
        {"index": index, "task_id": task["task_id"], "object_id": obj, "target_surface": target}
        for index, (task, obj, target) in enumerate(
            (task, obj, target) for task in case["tasks"] for obj, target in task["moves"]
        )
    ]


def validate_transfer(case, index):
    plan = transfer_plan(case)
    if type(index) is not int or not 0 <= index < len(plan):
        raise ValueError(f"Transfer index must be in [0, {len(plan) - 1}]")
    return plan[index]


class TransferProbe:
    def __init__(self, session, index):
        self.session = session
        self.target = validate_transfer(session.case, index)
        self.index = -1
        self.state = {
            "schema_version": 1, "target": self.target,
            "diagnostic_only": True, "history_eligible": False,
            "prefix_completed": False, "target_completed": False,
            "completed_transfers": [], "failure_scope": "initialization",
        }
        session.result.update(diagnostic_only=True, history_eligible=False, transfer_probe=self.state)
        self.persist()

    def persist(self):
        write_json(self.session.output / "transfer_probe.json", self.state)

    def before(self):
        self.index += 1
        selected = self.index == self.target["index"]
        self.state["failure_scope"] = "target" if selected else "prefix"
        self.state["current_transfer"] = self.index
        if selected:
            self.state["prefix_completed"] = True
            self.state["target_start_step"] = self.session.result["control_steps"]
            observation = self.session.observe()
            write_json(self.session.output / "transfer_start.json", observation)
            self.session.capture("transfer_start")
            self.session.event("transfer_probe_target_started", **self.target, observation=observation)
        self.persist()

    def after(self):
        self.state["completed_transfers"].append(self.index)
        selected = self.index == self.target["index"]
        if selected:
            final = self.session.observe()
            if not final["hand_empty"] or self.target["target_surface"] not in final["objects"][self.target["object_id"]]["on"]:
                raise RuntimeError("Transfer diagnostic final conditions failed")
            self.state.update(target_completed=True, failure_scope=None,
                              target_control_steps=self.session.result["control_steps"] - self.state["target_start_step"])
            self.session.event("transfer_probe_completed", **self.target, observation=final)
            self.session.result.update(passed=True, phase="transfer_probe_finished", final=final)
        self.persist()
        return selected
