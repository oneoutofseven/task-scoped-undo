"""Task-level procedures: create history, then execute a human reference undo."""

import sys
import time
import traceback

from task_undo.domain.protection import ProtectionMonitor, pose_error
from task_undo.domain.history import claim_timing
from task_undo.common.storage import read_json, write_json


def expected_final_supports(case, initial):
    """Expected supports from the task specification, used only to check execution."""
    expected = {}
    for name, obj in initial["objects"].items():
        if len(obj["on"]) != 1:
            raise ValueError("Episode requires one initial support per object")
        expected[name] = obj["on"][0]
    for task in case["tasks"]:
        claim_timing(task)
        for name, target in task["moves"]:
            if name not in expected or target not in case["surfaces"]:
                raise ValueError("Unknown object or support in episode moves")
            expected[name] = target
    return expected


def create_episode(session, preview, probe=None):
    initial = session.observe()
    expected = expected_final_supports(session.case, initial)
    write_json(session.output / "initial_state.json", initial)
    session.capture("initial")
    session.event("episode_started", scenario_origin="scripted", surfaces=session.case["surfaces"], observation=initial)
    if not initial["hand_empty"] or not all("storage_surface" in x["on"] for x in initial["objects"].values()):
        raise RuntimeError("Invalid initial layout")
    if preview:
        session.result.update(phase="preview_ready", preview_passed=True, final=initial)
        return
    session.start_recording()
    session.initialize_controller()
    for task in session.case["tasks"]:
        session.event("task_started", task_id=task["task_id"], initiator=task["initiator"],
                      executor=task["executor"], instruction=task["instruction"])
        if "use_claim" in task and claim_timing(task) == "before_moves":
            session.event("use_claimed", task_id=task["task_id"],
                          evidence_source="scenario_authored_declaration", **task["use_claim"])
        for object_id, target in task["moves"]:
            if probe is not None:
                probe.before()
            session.action(task, "GRASP", object_id)
            session.action(task, "PLACE_ON_TOP", object_id, target)
            if probe is not None and probe.after():
                return
        if "use_claim" in task and claim_timing(task) == "after_moves":
            session.event("use_claimed", task_id=task["task_id"],
                          evidence_source="scenario_authored_declaration", **task["use_claim"])
        state = session.observe()
        # Intermediate destinations may be superseded within the same task.
        # Each primitive already checks its own result at execution time.
        if not all(target in state["objects"][obj]["on"] for obj, target in dict(task["moves"]).items()):
            raise RuntimeError(f"Task {task['task_id']} final relations do not hold")
        session.event("task_completed", task_id=task["task_id"], observation=state)
        write_json(session.output / f"after_task_{task['task_id']}.json", state)
        session.capture(f"after_task_{task['task_id']}")

    final = session.observe()
    if not final["hand_empty"] or not all(target in final["objects"][obj]["on"] for obj, target in expected.items()):
        raise RuntimeError("Episode final relations failed")
    import omnigibson as og
    checkpoint = session.output / "before_undo_scene.json"
    og.sim.save(json_paths=[str(checkpoint)])
    if not checkpoint.is_file():
        raise RuntimeError("Scene checkpoint missing")
    session.event("undo_requested", **session.case["undo_request"], observation=final)
    write_json(session.output / "before_undo.json", final)
    session.capture("before_undo")
    session.result.update(passed=True, phase="finished", final=final,
                          completed_tasks=[task["task_id"] for task in session.case["tasks"]])


def check_restored_scene(session, source, position_m=0.02, angle_deg=5.0):
    reference = read_json(source / "before_undo.json")
    restored = session.observe()
    differences = {}
    for name, original in reference["objects"].items():
        current = restored["objects"][name]
        distance, angle = pose_error(original, current)
        differences[name] = {"position_m": distance, "angle_deg": angle,
                             "relations_match": original["on"] == current["on"], "held": current["held"]}
    valid = restored["hand_empty"] and all(
        d["position_m"] <= position_m and d["angle_deg"] <= angle_deg
        and d["relations_match"] and not d["held"] for d in differences.values())
    write_json(session.output / "restore_check.json", {"passed": valid, "differences": differences})
    write_json(session.output / "initial_state.json", restored)
    session.capture("initial")
    if not valid:
        raise RuntimeError("Restored scene does not match the recorded undo starting state")
    return reference, restored


def reference_undo(session, goal, source, preview):
    reference, restored = check_restored_scene(session, source, goal["position_tolerance_m"],
                                               goal["orientation_tolerance_deg"])
    if preview:
        session.result.update(phase="preview_ready", preview_passed=True, final=restored)
        return

    session.event("undo_started", target_task_id=goal["target_task_id"], goal_source="human_reference",
                  observation=restored)
    monitor = ProtectionMonitor(reference, goal["protected_objects"], goal["position_tolerance_m"],
                                goal["orientation_tolerance_deg"])
    session.start_protection(monitor)
    if goal["moves"]:
        session.initialize_controller()
    else:
        session.idle(30)
    task = {"task_id": f"undo_{goal['target_task_id']}", "initiator": "human_reference", "executor": "robot"}
    for object_id, target in goal["moves"]:
        session.action(task, "GRASP", object_id)
        session.action(task, "PLACE_ON_TOP", object_id, target)
    final = session.observe()
    session.observe_execution()
    if not final["hand_empty"] or not all(target in final["objects"][obj]["on"] for obj, target in goal["moves"]):
        raise RuntimeError("Reference undo final conditions failed")
    session.event("undo_completed", target_task_id=goal["target_task_id"], observation=final)
    session.capture("after_undo")
    write_json(session.output / "after_undo.json", final)
    session.result.update(passed=True, phase="finished", final=final, goal_source="human_reference")


def _complete_rollout(session, policy):
    """Save execution completion; task success is determined by offline scoring."""
    session.observe_execution()
    final = session.observe()
    session.event("rollout_completed", policy=policy, observation=final)
    session.capture("after_undo")
    write_json(session.output / "after_undo.json", final)
    session.result.update(passed=True, execution_completed=True, phase="finished", final=final,
                          policy=policy)


def baseline_undo(session, policy, source, preview, execution_mode="fixed"):
    """Deliberately simple policies; no access to evaluator labels or guards."""
    from task_undo.domain.undo_contract import EXECUTOR as GUARDED_EXECUTOR
    if execution_mode not in ("fixed", GUARDED_EXECUTOR):
        raise ValueError("Unknown baseline executor")
    if execution_mode == GUARDED_EXECUTOR and policy == "clear_table":
        raise ValueError("Guarded execution requires a public goal policy")
    _, restored = check_restored_scene(session, source)
    if execution_mode == GUARDED_EXECUTOR:
        from task_undo.agents.direct import load_public_input
        from task_undo.domain.policies import infer_goal
        from task_undo.application.goal_execution import execute_goal
        public = load_public_input(session.output / "public_input")
        public["current_state"] = restored
        goal = infer_goal(policy, public)
        write_json(session.output / "baseline_goal.json", goal)
        execute_goal(session, goal, restored, policy, preview=preview)
        if not preview:
            _complete_rollout(session, policy)
        return
    if preview:
        session.result.update(phase="preview_ready", preview_passed=True, final=restored)
        return
    session.event("undo_started", target_task_id=session.case["undo_request"]["target_task_id"],
                  policy=policy, observation=restored)
    session.start_recording()
    if policy == "noop":
        session.idle(30)
    elif policy == "clear_table":
        # Uses the visible support relation only; intentionally ignores history.
        objects = [name for name, obj in restored["objects"].items() if "dining_table" in obj["on"]]
        session.initialize_controller()
        task = {"task_id": "baseline", "initiator": policy, "executor": "robot"}
        for name in objects:
            session.action(task, "GRASP", name)
            session.action(task, "PLACE_ON_TOP", name, "storage_surface")
    else:
        from task_undo.agents.direct import load_public_input
        from task_undo.application.baselines import fixed_actions
        from task_undo.domain.policies import infer_goal
        public = load_public_input(session.output / "public_input")
        public["current_state"] = restored
        goal = infer_goal(policy, public)
        write_json(session.output / "baseline_goal.json", goal)
        actions = fixed_actions(goal)
        write_json(session.output / "baseline_plan.json", {
            "policy": policy, "executor": "fixed_object_id_order", "model_calls": 0,
            "actions": actions,
        })
        if actions:
            session.initialize_controller()
            task = {"task_id": "baseline", "initiator": policy, "executor": "robot"}
            for action in actions:
                session.action(task, action["action"], action["object_id"], action["target_surface"])
        else:
            session.idle(30)
    _complete_rollout(session, policy)


def language_undo(session, source, preview, execution_mode="fixed", prediction_readout="staged_rules"):
    """Execute exactly the saved model-derived goal; score success offline."""
    from task_undo.application.predictions import load_prediction
    from task_undo.application.baselines import fixed_actions

    from task_undo.domain.completion import EXECUTOR, SETTLE_STEPS, goal_satisfied
    from task_undo.domain.undo_contract import EXECUTOR as GUARDED_EXECUTOR

    if execution_mode not in ("fixed", EXECUTOR, GUARDED_EXECUTOR):
        raise ValueError("Unknown saved-prediction executor")
    observed_completion = execution_mode == EXECUTOR
    _, restored = check_restored_scene(session, source)
    goal = load_prediction(session.output / "prediction", session.case, restored, prediction_readout)
    actions = fixed_actions(goal)
    goal_source = "object_table_rule_prediction" if prediction_readout == "object_table_rule_v1" else "staged_language_prediction"
    from task_undo.application.prediction_replay import TABLE_MODEL_MODES
    if prediction_readout in TABLE_MODEL_MODES:
        goal_source = TABLE_MODEL_MODES[prediction_readout] + '_prediction'
    if prediction_readout in ("react_goal_v1", "cap_goal_v1", "cap_goal_v2"):
        goal_source = prediction_readout + "_prediction"
    if execution_mode == GUARDED_EXECUTOR:
        from task_undo.application.goal_execution import execute_goal
        execute_goal(session, goal, restored, goal_source, preview=preview)
        if not preview:
            _complete_rollout(session, goal_source)
        return
    write_json(session.output / "execution_plan.json", {
        "goal_source": goal_source, "goal": goal,
        "prediction_readout": prediction_readout,
        "executor": EXECUTOR if observed_completion else "fixed_object_id_order", "actions": actions,
        "additional_model_calls": 0})
    if preview:
        session.result.update(phase="preview_ready", preview_passed=True, final=restored)
        return
    session.event("undo_started", target_task_id=session.case["undo_request"]["target_task_id"],
                  policy=goal_source, observation=restored)
    if observed_completion:
        # This monitor uses the model's prediction, never the private scoring labels.
        session.start_protection(ProtectionMonitor(restored, goal["protect"]))
    else:
        session.start_recording()
    recovered = False
    if actions:
        session.initialize_controller()
        task = {"task_id": "language_undo", "initiator": goal_source, "executor": "robot"}
        for index, action in enumerate(actions):
            try:
                session.action(task, action["action"], action["object_id"], action["target_surface"])
            except Exception:
                # A deliberately narrow policy: only the final placement can be
                # resolved by observing the entire predicted goal already achieved.
                if (not observed_completion or index != len(actions) - 1
                        or action["action"] != "PLACE_ON_TOP"
                        or not goal_satisfied(goal, restored, session.observe())):
                    raise
                session.monitor.require_valid()
                failure_event_id = session.event_count
                start_step = session.result["control_steps"]
                session.observe_execution()
                for _ in range(SETTLE_STEPS):
                    session.idle(1)
                    if not goal_satisfied(goal, restored, session.observe()):
                        raise RuntimeError("Predicted goal did not remain satisfied after action failure")
                session.event("task_completion_verified", failure_event_id=failure_event_id,
                              start_control_step=start_step, settle_steps=SETTLE_STEPS,
                              observation=session.observe())
                recovered = True
    else:
        session.idle(30)
    if observed_completion:
        session.monitor.require_valid()
        if not goal_satisfied(goal, restored, session.observe()):
            raise RuntimeError("Execution ended without satisfying the predicted goal")
        session.result.update(execution_mode=EXECUTOR, completion_reconciled=recovered,
                              primitive_execution_clean=not recovered)
    _complete_rollout(session, goal_source)


def agent_undo(session, source, preview, planner="direct", plan_guard=False):
    from task_undo.agents.direct import load_public_input, run_agent
    from task_undo.agents.model import ChatModel

    _, restored = check_restored_scene(session, source)
    if preview:
        session.result.update(phase="preview_ready", preview_passed=True, final=restored)
        return
    config = read_json(session.output / "agent_config.json")
    public = load_public_input(session.output / "public_input")
    public["current_state"] = restored
    session.event("undo_started", target_task_id=public["undo_request"]["target_task_id"],
                  policy="full_history_agent", observation=restored)
    session.start_recording()
    task = {"task_id": "agent_undo", "initiator": "full_history_agent", "executor": "robot"}

    def execute(action, name, target):
        if session.controller is None:
            session.initialize_controller()
        session.action(task, action, name, target)

    if planner in ("decision_table", "evidence_table"):
        from task_undo.agents.decision_table import run_decision_table
        status = run_decision_table(ChatModel(config), public, session.output, config,
                                    observe=session.observe, execute=execute, plan_only=False,
                                    plan_guard=plan_guard, evidence_guard=planner == "evidence_table")
    else:
        status = run_agent(ChatModel(config), public, session.observe, execute, session.output, config)
    session.result["agent"] = status
    if not status["execution_completed"]:
        raise RuntimeError(f"Agent stopped: {status['stop_reason']}")
    # A model may immediately finish. Record actual physics even for a NoOp.
    if session.result["control_steps"] == 0:
        session.idle(30)
    _complete_rollout(session, "full_history_agent")


def run_plan_only(args):
    from task_undo.agents.direct import load_public_input, run_agent
    from task_undo.agents.model import ChatModel

    started = time.monotonic()
    result = {"passed": False, "phase": "plan_only", "physical_execution": False}
    try:
        config = read_json(args.output / "agent_config.json")
        public = load_public_input(args.output / "public_input")
        if args.planner in ("decision_table", "evidence_table"):
            from task_undo.agents.decision_table import run_decision_table
            status = run_decision_table(ChatModel(config), public, args.output, config,
                                        evidence_guard=args.planner == "evidence_table")
        else:
            status = run_agent(ChatModel(config), public, None, None, args.output, config, plan_only=True)
        result.update(passed=bool(status.get("plan_generated")), agent=status)
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        result["wall_seconds"] = round(time.monotonic() - started, 2)
        write_json(args.output / "worker.json", result)


def run_worker(args):
    if args.command == "agent" and args.plan_only:
        run_plan_only(args)
        return
    started = time.monotonic()
    result = {"passed": False, "phase": "imports", "control_steps": 0, "seed": args.seed,
              "motion_profile": getattr(args, "motion_profile", "legacy")}
    session = None
    try:
        from task_undo.simulation.session import Session, create_environment
        from task_undo.simulation.primitives import CONTACT_DEPTH_M, MUG_RIM_FRACTION
        case = read_json(args.output / "case.json")
        result.update(phase="scene", episode_id=case["episode_id"], grasping_mode="sticky", curobo_batch_size=1,
                      grasp_contact_depth_m=CONTACT_DEPTH_M, mug_rim_fraction=MUG_RIM_FRACTION)
        source = args.output / "input"
        checkpoint = source / "before_undo_scene.json" if args.command in ("undo", "baseline", "agent", "execute-language") else None
        if args.command == "probe-retraction":
            from task_undo.simulation.retraction_checkpoint import validate_checkpoint
            probe_info = validate_checkpoint(source)
            env = create_environment(case, args.output, args.seed, source / "scene.json", checkpoint_settle_steps=0)
        else:
            env = create_environment(case, args.output, args.seed, checkpoint,
                                     restore_mode=getattr(args, "restore_mode", "default"))
        session = Session(env, case, args.output, result, args.max_steps)
        if getattr(args, "capture_retraction", False):
            from task_undo.simulation.retraction_checkpoint import CheckpointDiagnostics
            session.gripper_diagnostics = CheckpointDiagnostics(session)
        elif getattr(args, "trace_retraction", False) or args.command in ("probe-retraction", "probe-transfer"):
            from task_undo.simulation.retraction_diagnostics import RetractionDiagnostics
            session.gripper_diagnostics = RetractionDiagnostics(session)
        elif getattr(args, "trace_gripper", False):
            from task_undo.simulation.diagnostics import GripperDiagnostics
            session.gripper_diagnostics = GripperDiagnostics(session)
        if args.command == "probe-retraction":
            from task_undo.simulation.retraction_checkpoint import run_probe
            run_probe(session, probe_info)
        elif args.command == "probe-transfer":
            from task_undo.application.transfer_probe import TransferProbe
            create_episode(session, False, probe=TransferProbe(session, args.transfer_index))
        elif args.command == "episode":
            create_episode(session, args.preview)
        elif args.command == "undo":
            reference_undo(session, read_json(args.output / "goal.json"), source, args.preview)
        elif args.command == "agent":
            agent_undo(session, source, args.preview, args.planner, args.plan_guard)
        elif args.command == "execute-language":
            language_undo(session, source, args.preview, args.execution_mode, args.prediction_readout)
        else:
            baseline_undo(session, read_json(args.output / "policy.json")["name"], source, args.preview,
                          args.execution_mode)
    except Exception:
        result["error"] = traceback.format_exc()
        print(result["error"], flush=True)
        if session:
            try:
                result["final"] = session.observe()
                session.event("episode_failed", phase=result["phase"], error=result["error"], observation=result["final"])
                session.capture("failed")
                write_json(session.output / "support_diagnostics.json", session.support_diagnostics())
            except Exception:
                result["evidence_error"] = traceback.format_exc()
    finally:
        if session:
            result["event_count"] = session.event_count
            result["physics_steps"] = session.physics_steps
            if session.monitor:
                result["protection"] = session.monitor.report()
                write_json(args.output / "protection.json", result["protection"])
            session.close()
        result["wall_seconds"] = round(time.monotonic() - started, 2)
        write_json(args.output / "worker.json", result)
        og = sys.modules.get("omnigibson")
        if og is not None and og.app is not None:
            og.shutdown()
