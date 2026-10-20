"""One guarded execution loop for public rule goals and saved language goals."""
import json

from task_undo.common.storage import write_json
from task_undo.domain.protection import ProtectionMonitor
from task_undo.domain.undo_contract import ContractError, EXECUTOR, SETTLE_STEPS, UndoContract


def execute_goal(session, goal, reference, goal_source, *, preview=False):
    """Keep physical failures terminal; labels and ownership inference stay outside."""
    status = {"executor": EXECUTOR, "completed": False, "stop_reason": "not_started",
              "actions": 0, "settle_steps": 0, "additional_model_calls": 0}
    stage = "contract"
    try:
        contract = UndoContract(goal, reference, session.case["surfaces"],
                                session.case["undo_request"]["target_task_id"])
        write_json(session.output / "undo_contract.json", contract.record())
        write_json(session.output / "execution_plan.json", {
            "executor": EXECUTOR, "goal_source": goal_source, "goal": contract.goal,
            "selection": "held_object_then_unsatisfied_object_id", "max_actions": contract.max_actions,
            "additional_model_calls": 0})
        if preview:
            status.update(stop_reason="preview_ready")
            session.result.update(phase="preview_ready", preview_passed=True, final=reference)
            return
        session.event("undo_started", target_task_id=contract.target_task_id,
                      policy=goal_source, observation=reference)
        stage = "protection"
        session.start_protection(ProtectionMonitor(contract.reference, contract.protect))
        task = {"task_id": "guarded_undo", "initiator": goal_source, "executor": "robot"}
        with (session.output / "execution_decisions.jsonl").open("w") as log:
            def record(kind, **fields):
                log.write(json.dumps({"kind": kind, "control_step": session.result["control_steps"],
                                      **fields}, ensure_ascii=False) + "\n")
                log.flush()

            while True:
                stage = "observation"
                session.observe_execution()
                state = session.observe()
                action = contract.next_action(state)
                if action is None:
                    record("settle_started", observation=state, steps=SETTLE_STEPS)
                    for _ in range(SETTLE_STEPS):
                        stage = "settling"
                        session.idle(1)
                        status["settle_steps"] += 1
                        if contract.next_action(session.observe()) is not None:
                            raise ContractError("unstable_goal", "Goal changed during final observation window")
                    session.monitor.require_valid()
                    record("completed", observation=session.observe())
                    status.update(completed=True, stop_reason="predicted_goal_verified")
                    session.result.update(execution_mode=EXECUTOR, predicted_goal_verified=True)
                    return
                if status["actions"] >= contract.max_actions:
                    raise ContractError("action_budget", "Goal action budget exhausted")
                if session.controller is None:
                    stage = "initialization"
                    session.initialize_controller()
                    if session.controller is None:
                        raise ContractError("initialization_failed", "Controller setup returned without a controller")
                    # Controller setup may advance the world. Reobserve before dispatch.
                    continue
                status["actions"] += 1
                record("dispatch", index=status["actions"], action=action, observation=state)
                stage = "physical_action"
                session.action(task, action["action"], action["object_id"], action["target_surface"])
                stage = "postcondition"
                current = session.observe()
                contract.check_action_result(action, current)
                session.monitor.require_valid()
                record("verified", index=status["actions"], observation=current)
    except Exception as error:
        reason = error.reason if isinstance(error, ContractError) else stage + "_failed"
        if session.monitor is not None and not session.monitor.report()["passed"]:
            reason = "protection_violation"
        status.update(stop_reason=reason, failed_stage=stage, error=str(error))
        raise
    finally:
        session.result["goal_execution"] = status
        write_json(session.output / "goal_execution.json", status)
