"""Opt-in, read-only tracing of primitive cleanup and assisted grasping.

Hooks delegate to the original methods exactly once. They never change actions,
advance physics, compute contact candidates, or clear grasp constraints.
"""

import json
import traceback


class GripperDiagnostics:
    def __init__(self, session):
        self.session = session
        self.stream = (session.output / "gripper_trace.jsonl").open("w", buffering=1)
        self.stack = []
        self.hooks = []
        self.error = None
        self.sequence = 0

    def record(self, kind, **fields):
        # A diagnostic failure must not interrupt or change the physical action.
        # It is reported separately and invalidates the diagnostic evidence.
        try:
            self.sequence += 1
            self.stream.write(json.dumps({
                "sequence": self.sequence, "kind": kind,
                "control_step": self.session.result["control_steps"],
                "physics_steps_observed": self.session.physics_steps,
                "action_phase": self.session.result["phase"],
                "subphases": list(self.stack), **fields,
            }) + "\n")
        except Exception:
            self.error = traceback.format_exc()

    def _replace(self, obj, name, method):
        own = name in vars(obj)
        old = vars(obj).get(name)
        self.hooks.append((obj, name, own, old))
        setattr(obj, name, method)

    def trace_generator(self, obj, name):
        original = getattr(obj, name)

        def traced(*args, **kwargs):
            self.stack.append(name)
            self.record("phase_started")
            try:
                value = yield from original(*args, **kwargs)
            except Exception as exc:
                self.record("phase_error", error_type=type(exc).__name__, error=str(exc),
                            traceback=traceback.format_exc())
                raise
            else:
                self.record("phase_completed")
                return value
            finally:
                self.stack.pop()

        self._replace(obj, name, traced)

    def trace_primitives(self, controller):
        # apply_ref caches a callable before running cleanup. Hook the actual
        # dispatch table, so the first failure survives a later cleanup error.
        wrapped = {}
        for primitive, original in controller.controller_functions.items():
            name = getattr(primitive, "name", str(primitive))

            def traced(*args, _original=original, _name=name, **kwargs):
                self.record("primitive_body_started", primitive=_name)
                try:
                    returned = yield from _original(*args, **kwargs)
                except Exception as exc:
                    self.record("primitive_body_error", primitive=_name,
                                error_type=type(exc).__name__, error=str(exc),
                                traceback=traceback.format_exc())
                    raise
                else:
                    self.record("primitive_body_completed", primitive=_name)
                    return returned

            wrapped[primitive] = traced
        self._replace(controller, "controller_functions", wrapped)

    def _read_controls(self):
        # Lazy import keeps tests and non-simulator commands lightweight.
        from omnigibson.controllers import ControllerView, ControlType

        robot = self.session.robot
        q = robot.get_joint_positions()
        arms = {}
        for arm in robot.arm_names:
            group, index = robot._controllers[f"gripper_{arm}"]
            joints = ControllerView.get_dof_idx(group)
            control = ControllerView.get_control(group, index)
            position = ControllerView.get_control_type(group) == ControlType.POSITION
            lower = robot.joint_lower_limits[joints]
            upper = robot.joint_upper_limits[joints]
            if control is None:
                applying = False
            elif robot._grasping_direction == "lower":
                applying = bool((control < (upper if position else 0)).any())
            else:
                applying = bool((control > (lower if position else 0)).any())
            arms[arm] = {
                "joint_positions": q[joints].tolist(),
                "control": control.tolist() if control is not None else None,
                "position_control": position, "direction": robot._grasping_direction,
                "lower_limits": lower.tolist(), "upper_limits": upper.tolist(),
                "applying_grasp_recomputed": applying,
            }
        return arms

    def _read_grasps(self):
        robot = self.session.robot
        return {arm: {
            "held_object": robot._ag_obj_in_hand[arm].name if robot._ag_obj_in_hand[arm] else None,
            "constraint_present": robot._ag_obj_constraints[arm] is not None,
            "grasp_counter": robot._ag_grasp_counter[arm],
            "release_counter": robot._ag_release_counter[arm],
        } for arm in robot.arm_names}

    def install(self, controller):
        if hasattr(controller, "controller_functions"):
            self.trace_primitives(controller)
        for name in ("_execute_release", "_reset_robot", "_settle_robot"):
            self.trace_generator(controller, name)
        robot = self.session.robot
        original = robot._handle_assisted_grasping

        def traced_grasp():
            sample = None
            try:
                sample = {"controls": self._read_controls(), "before": self._read_grasps()}
            except Exception:
                self.error = traceback.format_exc()
            # Outside the diagnostic try: preserve original errors and returns.
            value = original()
            if sample is not None:
                try:
                    self.record("assisted_grasp", **sample, after=self._read_grasps())
                except Exception:
                    self.error = traceback.format_exc()
            return value

        self._replace(robot, "_handle_assisted_grasping", traced_grasp)

    def close(self):
        for obj, name, own, old in reversed(self.hooks):
            if own:
                setattr(obj, name, old)
            else:
                delattr(obj, name)
        self.hooks.clear()
        self.session.result["gripper_diagnostics"] = {
            "file": "gripper_trace.jsonl", "records": self.sequence,
            "passed": self.error is None, "error": self.error,
        }
        self.stream.close()
