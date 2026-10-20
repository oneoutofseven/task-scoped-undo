"""Complete-candidate development regression with paired, independently scored undo."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import threading
import traceback

from task_undo.common.storage import read_json, write_json
from task_undo.domain.history import check_directory
from task_undo.evaluation.pairs import build_pair
from task_undo.evaluation.prospective import summarize
from task_undo.evaluation.scoring import evaluate_directory
from task_undo.evaluation.splits import digest


def run_regression(plan, verify, environment, now):
    root, output = Path(plan["root"]), Path(plan["output"])
    lock, rows = threading.Lock(), []

    def launch(folder, command, options, candidate, gpu, stage):
        verify()
        budget = plan["budget"]
        cmd = [sys.executable, "-u", "-m", "task_undo", command, "--output", str(folder),
               "--seed", str(candidate["seed"]), "--max-steps", str(budget[stage + "_steps"]),
               "--timeout", str(budget[stage + "_timeout_s"]),
               "--motion-profile", plan["motion_profile"], "--capture-retraction", *options]
        record = {"attempted": True, "command": cmd, "gpu": gpu, "started_at_utc": now()}
        with folder.with_suffix(".log").open("x") as log:
            process = subprocess.Popen(cmd, cwd=root, env=environment(gpu), stdout=log, stderr=subprocess.STDOUT)
            record["pid"] = process.pid
            write_json(folder.with_suffix(".launch.json"), record)
            record["exit_code"] = process.wait()
        if (folder / "result.json").exists():
            raw = read_json(folder / "result.json")
            record.update({key: raw.get(key) for key in
                           ("passed", "preview_passed", "phase", "control_steps", "error", "timed_out")})
        record["finished_at_utc"] = now()
        write_json(folder.with_suffix(".launch.json"), record)
        return record

    def execute(candidate, gpu):
        folder = output / candidate["id"]
        folder.mkdir()
        row = {"id": candidate["id"], "status": "running", "construction_passed": True,
               "preview_passed": False, "source_success": False, "pipeline_success": False,
               "branches": [], "failure_stage": "preview",
               "ancestor": candidate["ancestor"], "ancestor_recipe_group": candidate["ancestor_recipe_group"]}
        try:
            row["preview"] = launch(folder / "preview", "episode",
                                    ["--case", candidate["case"], "--preview"], candidate, gpu, "preview")
            row["preview_passed"] = row["preview"]["exit_code"] == 0 and row["preview"].get("preview_passed") is True
            if not row["preview_passed"]:
                return row
            row["failure_stage"] = "history_generation"
            source = folder / "episode"
            row["source"] = launch(source, "episode", ["--case", candidate["case"]], candidate, gpu, "source")
            if row["source"]["exit_code"] != 0 or row["source"].get("passed") is not True:
                return row
            row["failure_stage"] = "history_validation"
            checked = check_directory(source, write_report=False)
            if checked["successful_primitives"] != candidate["source_required_primitives"]:
                raise ValueError("History action count disagrees with the frozen recipe")
            row["source_success"] = True
            row["failure_stage"] = "branch_preparation"
            spec = {"pair_id": candidate["id"], "source_episode_id": read_json(source / "case.json")["episode_id"],
                    "source_sha256": {name: digest(source / name) for name in
                                      ("before_undo.json", "before_undo_scene.json", "public_history.jsonl")},
                    "variants": plan["branch_specs"][str(candidate["factors"]["relation"])]}
            write_json(folder / "pair_specification.json", spec)
            build_pair(source, folder / "pair", folder / "pair_specification.json")
            row["failure_stage"] = "undo_execution"
            for name in candidate["variants"]:
                branch = {"condition": name, "attempted": False, "joint_success": False}
                try:
                    trial = folder / ("undo_" + name)
                    branch.update(launch(trial, "baseline",
                        ["--episode", str(folder / "pair" / name), "--policy", plan["policy"],
                         "--execution-mode", plan["execution_mode"], "--restore-mode", plan["restore_mode"]],
                        candidate, gpu, "undo"))
                    branch["score"] = evaluate_directory(trial, folder / "pair/private" / (name + ".labels.json"))
                    branch["joint_success"] = bool(branch["exit_code"] == 0 and branch.get("passed")
                                                   and branch["score"]["joint_success"])
                    write_json(trial / "score.json", branch["score"])
                except Exception:
                    branch["error"] = traceback.format_exc()
                row["branches"].append(branch)
                write_json(folder / "summary.json", row)
            row["pipeline_success"] = all(b["joint_success"] for b in row["branches"])
            if row["pipeline_success"]:
                row["failure_stage"] = None
        except Exception:
            row["error"] = traceback.format_exc()
        finally:
            if row["source_success"]:
                for name in candidate["variants"]:
                    if not any(b["condition"] == name for b in row["branches"]):
                        row["branches"].append({"condition": name, "attempted": False, "joint_success": False,
                                                "error": "Branch preparation or infrastructure failed"})
            row.update(status="complete", finished_at_utc=now())
            write_json(folder / "summary.json", row)
        return row

    def lane(gpu):
        for candidate in (c for c in plan["candidates"] if c["gpu"] == gpu):
            row = execute(candidate, gpu)
            with lock:
                rows.append(row)
                report = summarize(plan["candidates"], rows)
                report.update(outcomes=rows, unseen_generalization_claim=False,
                              independent_recipe_groups=len({c["ancestor_recipe_group"] for c in plan["candidates"]}),
                              motion_profile=plan["motion_profile"], model_calls=0)
                temporary = output / "summary.pending.json"
                write_json(temporary, report)
                temporary.replace(output / "summary.json")

    with ThreadPoolExecutor(max_workers=len(plan["gpu_lanes"])) as pool:
        list(pool.map(lane, sorted(map(int, plan["gpu_lanes"]))))
    verify()
    return read_json(output / "summary.json")
