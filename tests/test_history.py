import copy
import json
from pathlib import Path
import unittest

from task_undo.domain.history import check_events
from task_undo.domain.policies import infer_goal

FIXTURE = json.loads((Path(__file__).parent / "fixtures/dinner_history.json").read_text())


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.events = copy.deepcopy(FIXTURE["events"])
        self.case = FIXTURE["case"]

    def test_actual_accepted_history(self):
        self.assertEqual(check_events(self.events, self.case)["successful_primitives"], 8)

    def test_reading_history_changes_which_object_is_restored(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/reading_history.json").read_text())
        events, case = fixture["events"], fixture["case"]
        self.assertEqual(check_events(events, case)["successful_primitives"], 8)
        goal = infer_goal("claim_aware", {
            "history": events, "current_state": events[-1]["observation"],
            "undo_request": case["undo_request"]})
        self.assertEqual(goal["restore"], {"book": "storage_surface"})
        self.assertEqual(goal["protect"], ["cup", "plate"])

    def test_missing_claim_rejected(self):
        events = [e for e in self.events if e["kind"] != "use_claimed"]
        ids = {e["event_id"]: i for i, e in enumerate(events, 1)}
        for e in events:
            e["event_id"] = ids[e["event_id"]]
            if "request_event_id" in e:
                e["request_event_id"] = ids[e["request_event_id"]]
        with self.assertRaisesRegex(ValueError, "takeover evidence"):
            check_events(events, self.case)

    def test_fake_grasp_rejected(self):
        e = next(e for e in self.events if e["kind"] == "action_succeeded")
        e["observation"]["objects"][e["object_id"]]["held"] = False
        with self.assertRaisesRegex(ValueError, "Grasp success"):
            check_events(self.events, self.case)

    def test_incomplete_history_rejected(self):
        with self.assertRaisesRegex(ValueError, "undo request missing"):
            check_events(self.events[:-1], self.case)

    def test_failed_execution_rejected(self):
        self.events[3]["kind"] = "action_failed"
        with self.assertRaisesRegex(ValueError, "Failed execution"):
            check_events(self.events, self.case)

    def test_unknown_cancelled_task_rejected_even_when_request_matches_case(self):
        case = copy.deepcopy(self.case)
        case["undo_request"]["target_task_id"] = "missing"
        self.events[-1]["target_task_id"] = "missing"
        with self.assertRaisesRegex(ValueError, "undo target"):
            check_events(self.events, case)

    def test_every_observation_must_cover_all_configured_objects(self):
        for index, event in enumerate(self.events):
            if "observation" not in event:
                continue
            with self.subTest(kind=event["kind"], index=index):
                events = copy.deepcopy(self.events)
                del events[index]["observation"]["objects"]["book"]
                with self.assertRaisesRegex(ValueError, "object set"):
                    check_events(events, self.case)

    def test_history_cannot_omit_an_untouched_configured_object(self):
        case = copy.deepcopy(self.case)
        case["objects"].append({"name": "untouched"})
        case["task_object_ids"] = ["plate", "cup", "book", "untouched"]
        with self.assertRaisesRegex(ValueError, "object set"):
            check_events(self.events, case)
