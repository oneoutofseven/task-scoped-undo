import copy
import math
import unittest

from task_undo.domain.protection import ProtectionMonitor, pose_error


class ProtectionTests(unittest.TestCase):
    def setUp(self):
        self.state = {"objects": {"cup": {"position": [1, 0, 0], "orientation": [0, 0, 0, 1],
                                         "on": ["desk"], "held": False}}, "hand_empty": True}
        self.monitor = ProtectionMonitor(self.state, ["cup"])

    def test_small_drift_allowed(self):
        moved = copy.deepcopy(self.state)
        moved["objects"]["cup"]["position"][0] += 0.001
        self.monitor.check(moved, 1)
        self.assertTrue(self.monitor.report()["passed"])

    def test_generator_of_protected_objects_is_consumed_once(self):
        monitor = ProtectionMonitor(self.state, iter(["cup"]))
        moved = copy.deepcopy(self.state)
        moved["objects"]["cup"]["position"][0] += 0.03
        monitor.check(moved, 1)
        self.assertFalse(monitor.report()["passed"])
        self.assertGreater(monitor.report()["maxima"]["cup"]["position_m"], 0.02)

    def test_move_then_return_still_fails(self):
        moved = copy.deepcopy(self.state)
        moved["objects"]["cup"]["position"][0] += 0.03
        self.monitor.check(moved, 1)
        self.monitor.check(self.state, 2)
        self.assertFalse(self.monitor.report()["passed"])
        self.assertEqual(self.monitor.report()["violations"][0]["first_step"], 1)

    def test_brief_grasp_between_control_steps_is_latched(self):
        self.monitor.check_grasp(["cup"])
        self.monitor.check_grasp([])
        self.monitor.check(self.state, 1)
        with self.assertRaises(RuntimeError):
            self.monitor.require_valid()

    def test_support_change_fails_even_without_pose_change(self):
        state = copy.deepcopy(self.state)
        state["objects"]["cup"]["on"] = []
        self.monitor.check(state, 1)
        self.assertFalse(self.monitor.report()["passed"])

    def test_rotation_limit_and_quaternion_sign(self):
        original = self.state["objects"]["cup"]
        rotated = copy.deepcopy(original)
        rotated["orientation"] = [0, 0, 0, -1]
        self.assertAlmostEqual(pose_error(original, rotated)[1], 0)
        state = copy.deepcopy(self.state)
        half = math.radians(6) / 2
        state["objects"]["cup"]["orientation"] = [0, 0, math.sin(half), math.cos(half)]
        self.monitor.check(state, 1)
        self.assertFalse(self.monitor.report()["passed"])

    def test_invalid_pose_rejected(self):
        invalid = copy.deepcopy(self.state["objects"]["cup"])
        invalid["position"][0] = float("nan")
        with self.assertRaises(ValueError):
            pose_error(self.state["objects"]["cup"], invalid)
