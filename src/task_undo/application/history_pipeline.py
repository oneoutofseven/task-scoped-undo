"""Freeze and run diagnostics -> full histories -> paired physical undo.

The pipeline stops at a failed diagnostic gate. Repairs require a new version;
it never selects easy histories or edits the frozen runtime during execution.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

from task_undo.application.transfer_probe import validate_transfer
from task_undo.common.paths import freeze_package, project_root
from task_undo.common.storage import read_json, write_json
from task_undo.evaluation.splits import digest, verify_hashes
from task_undo.evaluation.transfer_probe import audit_transfer, diagnostic_gate


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(parent_path, recipe_path, probes_path, output, motion_profile="legacy"):
    root = project_root()
    if motion_profile not in ("legacy", "loaded_waypoints_v1"):
        raise ValueError("Unknown motion profile")
    parent, recipe, probe_spec = map(read_json, (parent_path, recipe_path, probes_path))
    for source in (parent, recipe):
        verify_hashes(root, source["sha256"])
        verify_hashes(root / source.get("implementation", source.get("runtime")), source["runtime_sha256"])
    parents = {row["id"]: row for row in parent["candidates"]}
    recipes = {row["id"]: row for row in recipe["candidates"]}
    if len(parents) != len(parent["candidates"]) or parents.keys() != recipes.keys():
        raise ValueError("Parent and recipe must retain the same complete candidate set")
    probes = deepcopy(probe_spec["probes"])
    if not probes or len({p["probe_id"] for p in probes}) != len(probes):
        raise ValueError("Nonempty, unique diagnostic IDs required")
    for probe in probes:
        if Path(probe["probe_id"]).name != probe["probe_id"] or probe["probe_id"] in (".", ".."):
            raise ValueError("Invalid diagnostic ID")
        row = parents[probe["candidate_id"]]
        validate_transfer(read_json(root / row["case"]), probe["transfer_index"])
        if type(probe["gpu"]) is not int or probe["gpu"] < 0:
            raise ValueError("Invalid diagnostic GPU")
    output.mkdir(parents=True, exist_ok=False)
    runtime = freeze_package(output).parent
    runtime_hashes = {str(p.relative_to(runtime)): digest(p) for p in sorted((runtime / "src").rglob("*.py"))}
    inputs = output / "inputs"
    inputs.mkdir()
    cases = []
    for row in parent["candidates"]:
        copied = inputs / f"{row['id']}.json"
        shutil.copyfile(root / row["case"], copied)
        # Do not silently use a different historical recipe after construction.
        actual = read_json(copied)
        if row["required_primitives"] != 2 * sum(len(t["moves"]) for t in actual["tasks"]):
            raise ValueError("Parent primitive count disagrees with case")
        cases.append({**deepcopy(recipes[row["id"]]), **deepcopy(row),
                      "case": str(copied.relative_to(root)), "construction_passed": True,
                      "source_required_primitives": row["required_primitives"]})
    regression_output = output / "regression"
    regression_output.mkdir()
    regression = {
        "schema_version": 1, "id": output.name + "_regression", "frozen_at_utc": now(),
        "scope": "Exposed development regression; zero unseen-generalization claims.",
        "runtime": str(runtime.relative_to(root)), "runtime_sha256": runtime_hashes,
        "root": str(root), "output": str(regression_output),
        "motion_profile": motion_profile,
        "sha256": {str(p.relative_to(root)): digest(p) for p in inputs.glob("*.json")},
        "candidates": cases, "branch_specs": recipe["branch_specs"],
        "budget": {**recipe["budget"], "source_steps": parent["budget"]["history"]["steps"],
                   "source_timeout_s": parent["budget"]["history"]["timeout_s"],
                   "preview_steps": parent["budget"]["preview"]["steps"],
                   "preview_timeout_s": parent["budget"]["preview"]["timeout_s"]},
        "policy": recipe["policy"], "execution_mode": recipe["execution_mode"],
        "restore_mode": recipe["restore_mode"],
        "gpu_lanes": {str(g): [c["id"] for c in cases if c["gpu"] == g] for g in sorted({c["gpu"] for c in cases})},
        "claim_limit": "Paired rule undo is a physical feasibility witness, not a new algorithm or language result.",
    }
    write_json(output / "regression_protocol.json", regression)
    write_json(regression_output / "protocol.json", regression)
    plan = {
        "schema_version": 1, "id": output.name, "frozen_at_utc": now(),
        "root": str(root), "output": str(output), "runtime": str(runtime),
        "runtime_sha256": runtime_hashes, "probes": probes, "candidates": cases,
        "motion_profile": motion_profile,
        "budget": parent["budget"]["history"],
        "provenance": {str(p.resolve().relative_to(root)): digest(p) for p in (parent_path, recipe_path, probes_path)},
        "sha256": {str(p.relative_to(root)): digest(p) for p in
                   (output / "regression_protocol.json", regression_output / "protocol.json", *inputs.glob("*.json"))},
        "claim_limit": "Development only. Prefixes, repeats and branches inherit the parent source group.",
        "promotion_rule": "All frozen diagnostics pass; otherwise stop before full regression.",
    }
    write_json(output / "protocol.json", plan)
    return output / "protocol.json"


def clean_env(runtime, gpu=None):
    env = {k: v for k, v in os.environ.items()
           if k.lower() not in ("http_proxy", "https_proxy", "all_proxy", "cuda_visible_devices")}
    env.update(PYTHONPATH=str(Path(runtime) / "src"), OMNIGIBSON_HEADLESS="1", NO_PROXY="*", no_proxy="*")
    if gpu is not None:
        env["OMNIGIBSON_GPU_ID"] = str(gpu)
    return env


def wait_for_gpu(gpu):
    while True:
        read = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
                               "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True)
        if int(read.stdout.strip()) < 1000:
            return
        time.sleep(10)


def run(protocol):
    plan, protocol_hash = read_json(protocol), digest(protocol)
    root, output, runtime = map(Path, (plan["root"], plan["output"], plan["runtime"]))
    # The orchestrator, auditor and subprocess workers must use the same freeze.
    if Path(__file__).resolve() != runtime / "src/task_undo/application/history_pipeline.py":
        raise ValueError("Run this protocol using its frozen PYTHONPATH, as printed by prepare")

    def verify():
        if digest(protocol) != protocol_hash:
            raise ValueError("Pipeline protocol changed")
        verify_hashes(runtime, plan["runtime_sha256"])
        verify_hashes(root, {**plan["sha256"], **plan["provenance"]})

    verify()
    with (output / "started.json").open("x") as stream:
        json.dump({"pid": os.getpid(), "at_utc": now(), "protocol_sha256": protocol_hash}, stream)
    lock = threading.Lock()
    state = {"status": "diagnostics", "started_at_utc": now(), "outcomes": [], "lanes": {},
             "protocol_sha256": protocol_hash, "full_regression_started": False}
    (output / "diagnostics").mkdir()
    candidates = {c["id"]: c for c in plan["candidates"]}

    def persist():
        state["gate"] = diagnostic_gate(plan["probes"], state["outcomes"])
        # Atomic replacement prevents status readers observing partial JSON.
        temporary = output / "status.pending.json"
        write_json(temporary, state)
        temporary.replace(output / "status.json")

    persist()

    def lane(gpu):
        with lock:
            state["lanes"][str(gpu)] = "waiting_for_free_gpu"
            persist()
        wait_for_gpu(gpu)
        for probe in (p for p in plan["probes"] if p["gpu"] == gpu):
            verify()
            candidate = candidates[probe["candidate_id"]]
            folder = output / "diagnostics" / probe["probe_id"]
            command = [sys.executable, "-u", "-m", "task_undo", "probe-transfer",
                       "--case", candidate["case"], "--transfer-index", str(probe["transfer_index"]),
                       "--seed", str(candidate["seed"]), "--max-steps", str(plan["budget"]["steps"]),
                       "--timeout", str(plan["budget"]["timeout_s"]), "--output", str(folder),
                       "--motion-profile", plan["motion_profile"], "--capture-retraction"]
            row = {**probe, "ancestor": candidate["ancestor"],
                   "ancestor_recipe_group": candidate["ancestor_recipe_group"],
                   "command": command, "started_at_utc": now()}
            with lock:
                state["lanes"][str(gpu)] = probe["probe_id"]
                persist()
            try:
                with folder.with_suffix(".log").open("x") as log:
                    process = subprocess.Popen(command, cwd=root, env=clean_env(runtime, gpu),
                                               stdout=log, stderr=subprocess.STDOUT)
                    row["pid"] = process.pid
                    write_json(folder.with_suffix(".launch.json"), row)
                    row["exit_code"] = process.wait()
                row["audit"] = audit_transfer(folder)
                if row["exit_code"] != 0:
                    row["audit"]["passed"] = False
            except Exception:
                row["error"] = traceback.format_exc()
            row["finished_at_utc"] = now()
            write_json(folder.with_suffix(".outcome.json"), row)
            with lock:
                state["outcomes"].append(row)
                persist()
        with lock:
            state["lanes"][str(gpu)] = "complete"
            persist()

    try:
        with ThreadPoolExecutor(max_workers=len({p["gpu"] for p in plan["probes"]})) as pool:
            list(pool.map(lane, sorted({p["gpu"] for p in plan["probes"]})))
        verify()
        if not state["gate"]["passed"]:
            state.update(status="diagnostics_failed", finished_at_utc=now(),
                         next_action="Inspect primary and cleanup evidence; freeze a new repair version.")
            persist()
            return 1
        # Every candidate and both branches after an admitted source remain in
        # the denominator. All phases use the same frozen motion profile.
        state.update(status="full_regression", full_regression_started=True)
        persist()
        for gpu in read_json(output / "regression_protocol.json")["gpu_lanes"]:
            wait_for_gpu(gpu)
        from task_undo.application.history_regression import run_regression
        regression = run_regression(read_json(output / "regression_protocol.json"), verify,
                                    lambda gpu: clean_env(runtime, gpu), now)
        verify()
        state.update(status="complete", finished_at_utc=now(), regression_counts=regression["counts"])
        persist()
        return 0
    except Exception:
        state.update(status="error", error=traceback.format_exc(), finished_at_utc=now())
        persist()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("prepare")
    for name in ("parent", "recipe", "probes", "output"):
        freeze.add_argument("--" + name, type=Path, required=True)
    freeze.add_argument("--motion-profile", choices=("legacy", "loaded_waypoints_v1"), default="legacy")
    execute = sub.add_parser("run")
    execute.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        protocol = prepare(args.parent.resolve(), args.recipe.resolve(), args.probes.resolve(), args.output.resolve(),
                           args.motion_profile)
        print(json.dumps({"protocol": str(protocol), "PYTHONPATH": str(protocol.parent / "source/src"),
                          "command": [sys.executable, "-m", "task_undo.application.history_pipeline",
                                      "run", "--protocol", str(protocol)]}, indent=2))
        return 0
    return run(args.protocol.resolve())


if __name__ == "__main__":
    sys.exit(main())
