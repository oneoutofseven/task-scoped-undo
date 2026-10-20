"""Portable example and release entry points; no GPU or real model calls."""

from contextlib import redirect_stdout, redirect_stderr
import copy
import hashlib
import io
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

from task_undo.application import predict
from task_undo.application.reference import bind_reference
from task_undo.common.storage import read_json, write_json
from task_undo.domain.decisions import goal_to_table

ROOT = Path(__file__).resolve().parents[1]
DEMO = runpy.run_path(str(ROOT / "examples/history_contrast.py"))


class Model:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def complete(self, messages, **kwargs):
        self.requests.append(copy.deepcopy(messages))
        reply = next(self.replies)
        return {"content": reply if isinstance(reply, str) else json.dumps(reply),
                "finish_reason": "stop", "model": "synthetic_test"}


class PublicEntrypointTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.public = read_json(ROOT / "examples/book_adopted.json")
        self.correct = {"restore": {"cup": "storage_surface"}, "protect": ["book", "plate"]}

    def arguments(self, method="facts-rule", source=None):
        return ["--method", method, "--input", str(source or ROOT / "examples/book_adopted.json"),
                "--config", str(ROOT / "configs/qwen3vl.json"), "--output", str(self.folder / "prediction")]

    def test_same_observation_and_request_require_different_goals(self):
        result = DEMO["contrast"]()
        self.assertTrue(result["same_current_supports"])
        self.assertTrue(result["same_request"])
        self.assertEqual(result["goals"]["book_adopted"], self.correct)
        self.assertEqual(result["goals"]["cup_adopted"],
                         {"restore": {"book": "storage_surface"}, "protect": ["cup", "plate"]})
        self.assertFalse(result["physical_execution"])
        self.assertEqual(result["model_calls"], 0)

    def test_shared_readouts_keep_wrong_model_goal_and_correct_rule_goal(self):
        facts = DEMO["authored_facts"]("book")
        target = facts.pop("target_task")
        wrong = {"restore": {}, "protect": ["book", "cup", "plate"]}
        table = goal_to_table(wrong, self.public["objects"], self.public["surfaces"])
        model = Model([facts, {"target_task": target, "decisions": table}])
        with patch.object(predict, "ChatModel", return_value=model), redirect_stdout(io.StringIO()):
            status = predict.main(self.arguments())
        self.assertEqual(status, 0)
        self.assertEqual(len(model.requests), 2)
        folder = self.folder / "prediction"
        self.assertEqual(read_json(folder / "goal.json"), wrong)
        self.assertEqual(read_json(folder / "rule_goal.json"), self.correct)
        self.assertNotIn("undo_request", json.loads(model.requests[0][1]["content"]))
        self.assertEqual(read_json(folder / "status.json")["feedback_retries"], 0)

    def test_failed_extraction_does_not_fall_back_to_model_goal(self):
        table = goal_to_table(self.correct, self.public["objects"], self.public["surfaces"])
        model = Model(["not json", {"target_task": "setup", "decisions": table}])
        with patch.object(predict, "ChatModel", return_value=model), redirect_stdout(io.StringIO()):
            status = predict.main(self.arguments())
        self.assertEqual(status, 1)
        self.assertEqual(len(model.requests), 2)
        self.assertTrue((self.folder / "prediction/goal.json").is_file())
        self.assertFalse((self.folder / "prediction/rule_goal.json").exists())

    def test_direct_review_keeps_invalid_first_response(self):
        table = goal_to_table(self.correct, self.public["objects"], self.public["surfaces"])
        model = Model(["invalid first response", {"decisions": table}])
        with patch.object(predict, "ChatModel", return_value=model), redirect_stdout(io.StringIO()):
            status = predict.main(self.arguments("direct"))
        self.assertEqual(status, 0)
        self.assertEqual(len(model.requests), 2)
        self.assertEqual(json.loads(model.requests[1][1]["content"])["candidate_raw"],
                         "invalid first response")
        self.assertEqual(read_json(self.folder / "prediction/goal.json"), self.correct)

    def test_existing_output_is_not_overwritten(self):
        output = self.folder / "prediction"
        output.mkdir()
        (output / "sentinel").write_text("keep")
        with patch.object(predict, "ChatModel") as client, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                predict.main(self.arguments())
        self.assertEqual(error.exception.code, 2)
        client.assert_not_called()
        self.assertEqual((output / "sentinel").read_text(), "keep")

    def test_public_input_rejects_extra_private_answer(self):
        source = self.folder / "private_input.json"
        write_json(source, self.public | {"gold_goal": self.correct})
        with patch.object(predict, "ChatModel") as client, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                predict.main(self.arguments(source=source))
        client.assert_not_called()
        self.assertFalse((self.folder / "prediction").exists())

    def test_task_input_language_is_checked_before_model_call(self):
        source = self.folder / "non_english.json"
        write_json(source, self.public | {"undo_request": "撤销布置"})
        with patch.object(predict, "ChatModel") as client, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                predict.main(self.arguments(source=source))
        client.assert_not_called()


class ReferenceBindingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        fixture = read_json(ROOT / "tests/fixtures/reading_history.json")
        self.episode = self.folder / "episode"
        self.episode.mkdir()
        write_json(self.episode / "case.json", fixture["case"])
        state = fixture["events"][-1]["observation"]
        write_json(self.episode / "before_undo.json", state)
        write_json(self.episode / "before_undo_scene.json", {"synthetic_test_checkpoint": True})
        write_json(self.episode / "result.json", {
            "passed": True, "worker_exit_code": 0,
            "event_count": len(fixture["events"]), "final": state})
        (self.episode / "public_history.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in fixture["events"]))
        self.reference = ROOT / "cases/reading_001_undo.json"
        self.output = self.folder / "private/labels.json"

    def test_reference_is_bound_to_exact_source_bytes(self):
        labels = bind_reference(self.episode, self.reference, self.output)
        self.assertEqual(labels["restore"], {"book": "storage_surface"})
        self.assertEqual(labels["protect"], {"cup": "desk", "plate": "dining_table"})
        for name, digest in labels["source_sha256"].items():
            self.assertEqual(digest, hashlib.sha256((self.episode / name).read_bytes()).hexdigest())
        with self.assertRaises(FileExistsError):
            bind_reference(self.episode, self.reference, self.output)

    def test_wrong_target_is_rejected_before_labels_are_written(self):
        goal = read_json(self.reference)
        goal["target_task_id"] = "unrelated_task"
        alternate = self.folder / "wrong_reference.json"
        write_json(alternate, goal)
        with self.assertRaises(ValueError):
            bind_reference(self.episode, alternate, self.output)
        self.assertFalse(self.output.exists())

    def test_mismatched_start_is_rejected(self):
        state = read_json(self.episode / "before_undo.json")
        state["hand_empty"] = False
        write_json(self.episode / "before_undo.json", state)
        with self.assertRaises(ValueError):
            bind_reference(self.episode, self.reference, self.output)
        self.assertFalse(self.output.exists())
