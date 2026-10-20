"""Loop/transport contract tests with synthetic responses, not model-quality evidence."""

import copy
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from task_undo.agents.direct import load_public_input, run_agent, validate_plan
from task_undo.agents.model import ChatModel, validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/qwen3vl.json").read_text())
HISTORY = json.loads((ROOT / "tests/fixtures/dinner_history.json").read_text())["events"]


class FakeModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def complete(self, messages):
        self.requests.append(copy.deepcopy(messages))
        reply = next(self.replies)
        return {"content": reply if isinstance(reply, str) else json.dumps(reply), "model": "test_only"}


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.public = {"history": HISTORY, "current_state": HISTORY[-1]["observation"],
                       "undo_request": {"target_task_id": "A"}}
        self.state = copy.deepcopy(self.public["current_state"])
        self.plan = {"kind": "plan", "restore": {"plate": "storage_surface"}, "protect": ["cup", "book"],
                     "evidence": [{"object_id": name, "event_ids": [1], "reason": "test fixture"}
                                  for name in self.state["objects"]]}
        self.grasp = {"kind": "act", "action": "GRASP", "object_id": "plate", "target_surface": None}
        self.place = {"kind": "act", "action": "PLACE_ON_TOP", "object_id": "plate", "target_surface": "storage_surface"}
        self.finish = {"kind": "finish", "reason": "test fixture"}
        self.dispatched = []

    def execute(self, action, name, target):
        self.dispatched.append((action, name, target))
        self.state["objects"][name]["held"] = action == "GRASP"
        self.state["hand_empty"] = action != "GRASP"
        self.state["objects"][name]["on"] = [] if action == "GRASP" else [target]

    def run_loop(self, replies, config=None, execute=None):
        self.model = FakeModel(replies)
        return run_agent(self.model, self.public, lambda: copy.deepcopy(self.state),
                         execute or self.execute, self.output, config or CONFIG)

    def test_observed_feedback_drives_next_action(self):
        status = self.run_loop([self.plan, self.grasp, self.place, self.finish])
        self.assertTrue(status["execution_completed"])
        self.assertEqual(status["actions"], 2)
        feedback = json.loads(self.model.requests[2][-1]["content"])
        self.assertTrue(feedback["observation"]["objects"]["plate"]["held"])
        self.assertFalse(feedback["observation"]["hand_empty"])

    def test_public_loader_only_reads_allowlisted_files(self):
        (self.output / "history.jsonl").write_text("\n".join(json.dumps(e) for e in HISTORY))
        (self.output / "current_state.json").write_text(json.dumps(self.state))
        (self.output / "undo_request.json").write_text('{"target_task_id":"A"}')
        (self.output / "goal.json").write_text("PRIVATE_SENTINEL")
        (self.output / "labels.json").write_text("PRIVATE_SENTINEL")
        self.assertEqual(load_public_input(self.output), self.public)

    def test_wrong_semantic_plan_is_not_corrected_by_hidden_answer(self):
        self.plan["restore"] = {"plate": "storage_surface", "book": "storage_surface"}
        self.plan["protect"] = ["cup"]
        validate_plan(self.plan, self.public)
        status = self.run_loop([self.plan, self.finish])
        self.assertTrue(status["execution_completed"])
        self.assertEqual(status["actions"], 0)

    def test_invalid_requests_can_be_corrected_without_execution(self):
        invalid = dict(self.grasp, object_id="unknown")
        status = self.run_loop(["not JSON", self.grasp, self.plan, self.place, invalid, self.finish])
        self.assertEqual(status["actions"], 0)
        self.assertTrue(status["execution_completed"])

    def test_no_false_finish_while_holding_object(self):
        config = dict(CONFIG, max_turns=3)
        status = self.run_loop([self.plan, self.grasp, self.finish], config)
        self.assertFalse(status["execution_completed"])
        self.assertEqual(status["stop_reason"], "turn_budget")

    def test_action_budget_is_checked_before_dispatch(self):
        status = self.run_loop([self.plan, self.grasp, self.place], dict(CONFIG, max_actions=1))
        self.assertEqual(status["stop_reason"], "action_budget")
        self.assertEqual(len(self.dispatched), 1)

    def test_physical_failure_stops_and_is_recorded(self):
        def fail(*args):
            raise RuntimeError("synthetic execution failure")
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            self.run_loop([self.plan, self.grasp], execute=fail)
        status = json.loads((self.output / "agent_status.json").read_text())
        self.assertEqual(status["stop_reason"], "physical_action_failed")

    def test_nonexistent_evidence_id_rejected(self):
        self.plan["evidence"][0]["event_ids"] = [99999]
        with self.assertRaises(ValueError):
            validate_plan(self.plan, self.public)

    def test_plan_only_does_not_call_physical_callbacks(self):
        status = run_agent(FakeModel([self.plan]), self.public, None, None,
                           self.output, CONFIG, plan_only=True)
        self.assertTrue(status["plan_generated"])
        self.assertFalse(status["execution_completed"])


class ModelTests(unittest.TestCase):
    def test_credentials_must_not_be_stored_in_config(self):
        for config in (dict(CONFIG, api_key="secret"), dict(CONFIG, base_url="http://user:pass@localhost/v1")):
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_http_contract_and_proxy_bypass(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))) )
                result = {"model": "test_only", "choices": [{"finish_reason": "stop", "message": {"content": '{"ok":true}'}}]}
                body = json.dumps(result).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            config = dict(CONFIG, base_url=f"http://127.0.0.1:{server.server_port}/v1")
            with patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:1", "no_proxy": ""}):
                response = ChatModel(config).complete([{"role": "user", "content": "JSON please"}])
            self.assertEqual(json.loads(response["content"]), {"ok": True})
            self.assertEqual(received[0][0], "/v1/chat/completions")
            self.assertEqual(received[0][1]["model"], CONFIG["model"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
