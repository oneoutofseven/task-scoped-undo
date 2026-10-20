"""Unified entry point; keep simulator imports inside the isolated worker."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from task_undo.domain.history import check_directory
from task_undo.common.storage import read_json, write_json
from task_undo.common.paths import freeze_package, project_root
from task_undo.common.task_language import require_english_task

ROOT = project_root()


def validate_undo(source, goal):
    check_directory(source, write_report=False)
    case = read_json(source / "case.json")
    if goal["episode_id"] != case["episode_id"] or goal["target_task_id"] != case["undo_request"]["target_task_id"]:
        raise ValueError("Reference goal does not match the cancelled task")
    if goal["goal_source"] != "human_reference":
        raise ValueError("This stage only supports a human reference goal")
    final = read_json(source / "result.json")["final"]
    if read_json(source / "before_undo.json") != final:
        raise ValueError("Starting observation does not match the accepted episode")
    objects = set(final["objects"])
    moved = {name for name, _ in goal["moves"]}
    protected = set(goal["protected_objects"])
    if moved & protected or moved | protected != objects:
        raise ValueError("Every case object must be moved or protected, without overlap")
    if any(target not in case["surfaces"] for _, target in goal["moves"]):
        raise ValueError("Unknown target surface")
    if not 0 < goal["position_tolerance_m"] <= 0.02 or not 0 < goal["orientation_tolerance_deg"] <= 5:
        raise ValueError("Reference tolerances exceed the fixed case protocol")
    return case


def prepare_run(args):
    if (args.command == "baseline" and getattr(args, "execution_mode", "fixed") == "guarded_goal_v1"
            and args.policy == "clear_table"):
        raise ValueError("Guarded execution requires a public goal policy; clear_table has no goal contract")
    if args.command == "probe-retraction":
        from task_undo.simulation.retraction_checkpoint import validate_checkpoint
        source = args.checkpoint.resolve()
        validate_checkpoint(source)
        case = read_json(source / "case.json")
    elif args.command == "undo":
        if args.episode is None:
            raise ValueError("undo requires --episode pointing to a successful episode directory")
        source = args.episode.resolve()
        goal = read_json(args.goal)
        case = validate_undo(source, goal)
    elif args.command in ("baseline", "agent", "execute-language"):
        if args.episode is None:
            raise ValueError(f"{args.command} requires --episode")
        source = args.episode.resolve()
        check_directory(source, write_report=False)
        case = read_json(source / "case.json")
        if read_json(source / "before_undo.json") != read_json(source / "result.json")["final"]:
            raise ValueError("Starting observation disagrees with accepted episode")
    else:
        case = read_json(args.case)
    if args.command == "probe-transfer":
        from task_undo.application.transfer_probe import validate_transfer
        validate_transfer(case, args.transfer_index)
    require_english_task(case, "Task case")
    if args.command in ("undo", "baseline", "agent", "execute-language"):
        for line in (source / "public_history.jsonl").read_text().splitlines():
            require_english_task(json.loads(line), "Public history")
    if "objects" in case:
        from task_undo.domain.case_layout import validate_layout
        validate_layout(case)
    if args.command == "agent":
        from task_undo.agents.model import validate_config
        agent_config = read_json(args.agent_config)
        validate_config(agent_config)
    if args.command == "execute-language":
        from task_undo.application.predictions import load_prediction
        if args.prediction is None:
            raise ValueError("execute-language requires --prediction")
        load_prediction(args.prediction, case, read_json(source / "before_undo.json"), args.prediction_readout)
    args.output.mkdir(parents=True, exist_ok=False)
    code = freeze_package(args.output)
    write_json(args.output / "case.json", case)
    if args.command == "undo":
        write_json(args.output / "goal.json", goal)
    if args.command == "baseline":
        write_json(args.output / "policy.json", {"name": args.policy})
    if args.command == "agent":
        write_json(args.output / "agent_config.json", agent_config)
    if args.command == "execute-language":
        from task_undo.application.predictions import prediction_files
        for name in prediction_files(args.prediction, args.prediction_readout):
            destination = args.output / "prediction" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.prediction / name, destination)
    if args.command in ("undo", "baseline", "agent", "execute-language"):
        inputs = args.output / "input"
        inputs.mkdir()
        for name in ("before_undo_scene.json", "before_undo.json", "public_history.jsonl", "result.json"):
            shutil.copyfile(source / name, inputs / name)
        public = args.output / "public_input"
        public.mkdir()
        shutil.copyfile(source / "public_history.jsonl", public / "history.jsonl")
        shutil.copyfile(source / "before_undo.json", public / "current_state.json")
        write_json(public / "undo_request.json", case["undo_request"])
    if args.command == "probe-retraction":
        inputs = args.output / "input"
        inputs.mkdir()
        for name in ("scene.json", "case.json", "checkpoint.json"):
            shutil.copyfile(source / name, inputs / name)
    files = [p for p in args.output.rglob("*") if p.is_file()]
    write_json(args.output / "manifest.json", {
        "command": args.command, "source_episode": str(source) if args.command not in ("episode", "probe-transfer") else None,
        "options": {"seed": args.seed, "max_steps": args.max_steps, "timeout": args.timeout,
                    "preview": args.preview, "plan_only": getattr(args, "plan_only", False),
                    **({"transfer_index": args.transfer_index} if args.command == "probe-transfer" else {}),
                    **({"motion_profile": args.motion_profile} if getattr(args, "motion_profile", "legacy") != "legacy" else {}),
                    **({"planner": args.planner} if getattr(args, "planner", "direct") != "direct" else {}),
                    **({"plan_guard": True} if getattr(args, "plan_guard", False) else {}),
                    **({"restore_mode": args.restore_mode} if getattr(args, "restore_mode", "default") != "default" else {}),
                    **({"prediction_readout": args.prediction_readout} if args.command == "execute-language" else {}),
                    **({"execution_mode": args.execution_mode} if getattr(args, "execution_mode", "fixed") != "fixed" else {}),
                    **({"trace_gripper": True} if getattr(args, "trace_gripper", False) else {}),
                    **({"capture_retraction": True} if getattr(args, "capture_retraction", False) else {}),
                    **({"trace_retraction": True} if getattr(args, "trace_retraction", False) else {})},
        "sha256": {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    })
    return code


def stop_worker(process):
    # The worker and its video encoder form their own process group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_isolated(args):
    code_dir = prepare_run(args)
    command = [sys.executable, "-u", "-m", "task_undo", args.command, "--worker",
               "--output", str(args.output), "--seed", str(args.seed), "--max-steps", str(args.max_steps)]
    if args.command == "probe-transfer":
        command.extend(["--transfer-index", str(args.transfer_index)])
    if getattr(args, "motion_profile", "legacy") != "legacy":
        command.extend(["--motion-profile", args.motion_profile])
    if getattr(args, "execution_mode", "fixed") != "fixed":
        command.extend(["--execution-mode", args.execution_mode])
    if getattr(args, "restore_mode", "default") != "default":
        command.extend(["--restore-mode", args.restore_mode])
    if args.command == "execute-language":
        command.extend(["--prediction-readout", args.prediction_readout])
    if args.preview:
        command.append("--preview")
    if getattr(args, "plan_only", False):
        command.append("--plan-only")
    if getattr(args, "planner", "direct") != "direct":
        command.extend(["--planner", args.planner])
    if getattr(args, "plan_guard", False):
        command.append("--plan-guard")
    if getattr(args, "trace_gripper", False):
        command.append("--trace-gripper")
    if getattr(args, "trace_retraction", False):
        command.append("--trace-retraction")
    if getattr(args, "capture_retraction", False):
        command.append("--capture-retraction")
    # Keep the frozen package first even when PYTHONSAFEPATH disables cwd imports
    # or the caller has a development checkout in PYTHONPATH.
    worker_env = os.environ.copy()
    worker_env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(code_dir), worker_env.get("PYTHONPATH"))))
    timed_out = interrupted = False
    with (args.output / "run.log").open("w") as log:
        process = subprocess.Popen(command, cwd=code_dir, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=worker_env)
        try:
            code = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop_worker(process)
            code = process.returncode
        except KeyboardInterrupt:
            interrupted = True
            stop_worker(process)
            code = process.returncode
    report = args.output / "worker.json"
    result = read_json(report) if report.is_file() else {"passed": False, "error": "Worker report missing; see run.log"}
    normal = code == 0 and not timed_out and not interrupted
    result.update(worker_exit_code=code, timed_out=timed_out, interrupted=interrupted)
    result["passed"] = bool(result["passed"] and normal)
    write_json(args.output / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if interrupted:
        return 130
    return 0 if normal and (result["passed"] or (args.preview and result.get("preview_passed"))) else 1


def main():
    parser = argparse.ArgumentParser(description="Create histories, execute task-scoped undo, and score recorded rollouts.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("episode", "undo", "baseline", "agent", "execute-language", "probe-retraction", "probe-transfer"):
        sub = commands.add_parser(name)
        sub.add_argument("--output", type=Path)
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument("--max-steps", type=int, default=30000 if name == "episode" else 15000)
        sub.add_argument("--timeout", type=int, default=3600)
        sub.add_argument("--motion-profile", choices=("legacy", "loaded_waypoints_v1"), default="legacy",
                         help="Opt-in bounded loaded-arm waypoint resampling; retains physical tolerances")
        sub.add_argument("--preview", action="store_true", help="Validate starting layout only; no task execution")
        sub.add_argument("--trace-gripper", action="store_true",
                         help="Record read-only primitive cleanup and assisted-grasp diagnostics")
        sub.add_argument("--trace-retraction", action="store_true",
                         help="Include gripper diagnostics, phase-boundary contacts and raw planner status")
        sub.add_argument("--capture-retraction", action="store_true",
                         help="Save read-only post-release starts for separate diagnostic probes")
        sub.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
        if name == "probe-retraction":
            sub.add_argument("--checkpoint", type=Path, help="Captured retraction checkpoint directory")
        elif name in ("episode", "probe-transfer"):
            sub.add_argument("--case", type=Path, default=ROOT / "cases/dinner_001.json")
            if name == "probe-transfer":
                sub.add_argument("--transfer-index", type=int, required=True,
                                 help="Zero-based transfer index; execute the original prefix before this transfer")
        else:
            sub.add_argument("--restore-mode", choices=("default", "contact_refresh_v1"), default="default",
                             help="Opt-in contact query refresh during checkpoint initialization")
            sub.add_argument("--episode", type=Path, help="Successful stage-3 output directory")
            if name == "undo":
                sub.add_argument("--goal", type=Path, default=ROOT / "cases/dinner_001_undo.json")
            elif name == "baseline":
                sub.add_argument("--policy", choices=("noop", "clear_table", "restore_initial", "last_writer", "claim_aware"), default="noop")
                sub.add_argument("--execution-mode", choices=("fixed", "guarded_goal_v1"), default="fixed",
                                 help="Optional observed-goal execution with public-goal protection")
            elif name == "execute-language":
                sub.add_argument("--execution-mode", choices=("fixed", "observe_completion_v1", "guarded_goal_v1"), default="fixed")
                sub.add_argument("--prediction", type=Path, help="Completed saved language prediction directory")
                sub.add_argument("--prediction-readout", choices=("staged_rules", "object_table_rule_v1",
                                 "object_table_direct_v1", "object_table_model_v1", "object_table_no_history_v1",
                                 "react_goal_v1", "cap_goal_v1", "cap_goal_v2"),
                                 default="staged_rules", help="Readout selected before physical evaluation")
            else:
                sub.add_argument("--agent-config", type=Path, default=ROOT / "configs/qwen3vl.json")
                sub.add_argument("--plan-only", action="store_true", help="Infer a goal without loading the simulator")
                sub.add_argument("--planner", choices=("direct", "decision_table", "evidence_table"), default="direct",
                                 help="Goal output format; all planners support planning and physical execution")
                sub.add_argument("--plan-guard", action="store_true",
                                 help="Reject physical actions that contradict the decision table plan")
    check = commands.add_parser("check-history")
    check.add_argument("folder", type=Path)
    evaluate = commands.add_parser("evaluate", help="Score a recorded rollout without loading the simulator")
    evaluate.add_argument("folder", type=Path)
    evaluate.add_argument("--labels", type=Path, required=True,
                          help="Private reference bound to this exact history")
    pair = commands.add_parser("make-pair", help="Branch a verified physical prefix with zero-motion declarations")
    pair.add_argument("--episode", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    pair.add_argument("--specification", type=Path, required=True,
                      help="Pair specification bound to a verified physical source")
    pair_check = commands.add_parser("check-pair", help="Check paired inputs and compare rule goals without simulation")
    pair_check.add_argument("folder", type=Path)
    args = parser.parse_args()
    if args.command in ("make-pair", "check-pair"):
        from task_undo.evaluation.pairs import build_pair, compare_rule_goals
        try:
            report = (build_pair(args.episode.resolve(), args.output.resolve(), args.specification)
                      if args.command == "make-pair" else compare_rule_goals(args.folder.resolve()))
        except (ValueError, FileNotFoundError, FileExistsError) as exc:
            parser.error(str(exc))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "check-history":
        print(json.dumps(check_directory(args.folder), indent=2))
        return 0
    if args.command == "evaluate":
        from task_undo.evaluation.scoring import evaluate_directory
        score = evaluate_directory(args.folder, args.labels)
        write_json(args.folder / "score.json", score)
        print(json.dumps(score, ensure_ascii=False, indent=2))
        return 0 if score["valid"] else 2
    if min(args.max_steps, args.timeout) < 1:
        parser.error("Budgets must be positive")
    if args.command == "agent" and args.plan_only and args.preview:
        parser.error("--plan-only and --preview cannot be combined")
    if args.command == "agent" and args.plan_guard and (args.planner not in ("decision_table", "evidence_table") or args.plan_only):
        parser.error("--plan-guard requires physical --planner decision_table or evidence_table")
    if args.command == "probe-retraction" and not args.worker and args.checkpoint is None:
        parser.error("probe-retraction requires --checkpoint")
    if args.command == "probe-retraction" and (args.preview or args.capture_retraction):
        parser.error("probe-retraction cannot preview or capture new starts")
    if args.command == "probe-transfer" and args.preview:
        parser.error("probe-transfer requires physical execution")
    args.output = (args.output or ROOT / "outputs" / args.command / time.strftime("%Y%m%d_%H%M%S")).resolve()
    if args.worker:
        os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
        os.environ.setdefault("OMNIGIBSON_GPU_ID", "0")
        from task_undo.application.workflows import run_worker
        run_worker(args)
        return 0
    try:
        return run_isolated(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
