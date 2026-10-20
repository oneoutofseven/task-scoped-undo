"""Exercise installed entry points and real subprocess source isolation without a GPU."""

from contextlib import redirect_stdout
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from task_undo.application import cli
from task_undo.common.paths import PACKAGE_ROOT, freeze_package, project_root
from task_undo.common.storage import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


class PackagingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)

    def test_installed_entry_is_lightweight_outside_checkout(self):
        program = """
import json, sys
from task_undo.application import cli, workflows
from task_undo.evaluation import scoring
assert 'omnigibson' not in sys.modules
assert 'torch' not in sys.modules
print(json.dumps(str(cli.ROOT)))
"""
        result = subprocess.run([sys.executable, "-c", program], cwd=self.folder,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(json.loads(result.stdout)), ROOT)
        self.assertEqual(project_root(), ROOT)

    def test_frozen_package_runs_without_site_packages(self):
        source = freeze_package(self.folder)
        result = subprocess.run([sys.executable, "-S", "-m", "task_undo", "--help"],
                                cwd=source, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("check-history", result.stdout)
        self.assertEqual(
            {p.relative_to(PACKAGE_ROOT) for p in PACKAGE_ROOT.rglob("*.py")},
            {p.relative_to(source / "task_undo") for p in (source / "task_undo").rglob("*.py")})

    def test_worker_imports_frozen_code_even_with_safe_path(self):
        args = SimpleNamespace(command="episode", case=ROOT / "cases/dinner_001.json",
                               output=self.folder / "run", seed=0, max_steps=1,
                               timeout=20, preview=False)
        prepare = cli.prepare_run

        def prepare_with_sentinel(options):
            source = prepare(options)
            # A deliberate failure in ONLY the copy proves which code executes.
            (source / "task_undo/__init__.py").write_text(
                'raise RuntimeError("FROZEN_SOURCE_SENTINEL")\n')
            return source

        with patch.object(cli, "prepare_run", side_effect=prepare_with_sentinel), \
                patch.dict(os.environ, {"PYTHONSAFEPATH": "1", "PYTHONPATH": str(ROOT / "src")}), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(cli.run_isolated(args), 1)
        self.assertIn("FROZEN_SOURCE_SENTINEL", (args.output / "run.log").read_text())
        self.assertFalse(read_json(args.output / "result.json")["passed"])

    def test_cli_plan_only_round_trip_with_synthetic_model(self):
        fixture = read_json(ROOT / "tests/fixtures/reading_history.json")
        source = self.folder / "episode"
        source.mkdir()
        state = fixture["events"][-1]["observation"]
        write_json(source / "case.json", fixture["case"])
        write_json(source / "before_undo.json", state)
        write_json(source / "before_undo_scene.json", {"synthetic": True})
        write_json(source / "result.json", {"passed": True, "worker_exit_code": 0,
                   "event_count": len(fixture["events"]), "final": state})
        (source / "public_history.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in fixture["events"]))
        table = {"kind": "evidence_table", "items": [
            {"object_id": obj, "before_event_id": 1,
             "target_change_event_ids": changes, "later_activity_event_ids": later,
             "reason": "Synthetic packaging integration response",
             "decision": "restore" if target else "protect", "target_surface": target}
            for obj, changes, later, target in [
                ("book", [6], [], "storage_surface"),
                ("cup", [10], [13, 17], None), ("plate", [], [21], None)]]}
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                payload = json.dumps({"choices": [{"finish_reason": "stop",
                    "message": {"content": json.dumps(table)}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        config = read_json(ROOT / "configs/qwen3vl.json")
        config["base_url"] = f"http://127.0.0.1:{server.server_port}/v1"
        write_json(self.folder / "config.json", config)
        output = self.folder / "run"
        try:
            result = subprocess.run([
                sys.executable, "-m", "task_undo", "agent", "--planner", "evidence_table",
                "--plan-only", "--episode", str(source), "--output", str(output),
                "--agent-config", str(self.folder / "config.json"), "--timeout", "20"],
                cwd=self.folder, capture_output=True, text=True, timeout=30)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(requests), 1)
        self.assertTrue(read_json(output / "result.json")["passed"])
        self.assertFalse(read_json(output / "result.json")["physical_execution"])
        plan = read_json(output / "agent_plan.json")
        self.assertEqual(plan["restore"], {"book": "storage_surface"})
        self.assertEqual(set(plan["protect"]), {"cup", "plate"})
        manifest = read_json(output / "manifest.json")["sha256"]
        self.assertIn("source/src/task_undo/agents/evidence.py", manifest)
        for name, expected in manifest.items():
            self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), expected, name)
