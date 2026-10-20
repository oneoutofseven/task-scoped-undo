"""Wrong semantic goals must not become successful tasks through executor status."""
import unittest

from task_undo.evaluation.end_to_end import summarize_trials, trial_metrics


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.gold = {"restore": {"cup": "storage"}, "protect": ["book"]}
        self.completed = {"attempted": True, "result": {"passed": True}, "exit_code": 0,
                          "execution_audit": {"audit_passed": True, "completed": True}}

    def test_wrong_noop_can_complete_its_prediction_and_fail_the_task(self):
        row = trial_metrics({"restore": {}, "protect": ["cup", "book"]}, self.gold,
            score={"valid": True, "joint_success": False, "undo_completion_rate": 0,
                   "protection_violation_rate": 0}, **self.completed)
        self.assertTrue(row["predicted_goal_completed"])
        self.assertFalse(row["goal_correct"])
        self.assertFalse(row["physical_task_success"])
        self.assertFalse(row["joint_success"])

    def test_incorrect_goal_with_incidental_physical_success_is_not_joint_success(self):
        row = trial_metrics({"restore": {"cup": "storage", "book": "desk"}, "protect": []}, self.gold,
            score={"valid": True, "joint_success": True, "undo_completion_rate": 1,
                   "protection_violation_rate": 0}, **self.completed)
        self.assertTrue(row["physical_task_success"])
        self.assertFalse(row["joint_success"])

    def test_correct_goal_without_physical_evidence_is_failure_with_missing_rates(self):
        row = trial_metrics(self.gold, self.gold, attempted=False)
        self.assertTrue(row["goal_correct"])
        self.assertFalse(row["joint_success"])
        self.assertIsNone(row["protection_violation_rate"])

    def test_missing_prediction_and_missing_cell_do_not_shrink_denominators(self):
        conditions = [{"id": "a", "group": "same_source"}, {"id": "b", "group": "same_source"}]
        rows = [{"condition": "a", "method": "model", "status": "complete",
                 "failure_stage": "prediction_unavailable",
                 "metrics": trial_metrics(None, self.gold, attempted=False)}]
        report = summarize_trials(conditions, ["model"], rows)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["methods"]["model"]["planned"], 2)
        self.assertEqual(report["methods"]["model"]["planned_source_groups"], 1)
        self.assertEqual(report["methods"]["model"]["joint_success"], 0)

    def test_duplicate_or_unplanned_trials_are_rejected(self):
        conditions = [{"id": "a", "group": "g"}]
        with self.assertRaises(ValueError):
            summarize_trials(conditions, ["model"], [{"condition": "b", "method": "model"}])
        with self.assertRaises(ValueError):
            summarize_trials(conditions, ["model"], [{"condition": "a", "method": "model"}] * 2)
