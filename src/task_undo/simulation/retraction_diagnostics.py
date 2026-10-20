"""Read existing contacts and raw planner results; never run extra planning."""

import traceback
import os
from collections import deque

from task_undo.simulation.diagnostics import GripperDiagnostics


def plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    return str(value)


def head_arm_contacts(pairs):
    result = set()
    for pair in pairs:
        names = [path.rsplit('/', 1)[-1] for path in pair]
        if any(name.startswith('head_') for name in names) and any(
                name.startswith(('arm_', 'gripper_')) for name in names):
            result.add(tuple(sorted(pair)))
    return sorted(result)


class RetractionDiagnostics(GripperDiagnostics):
    def read_head_arm_contacts(self):
        from omnigibson.utils.usd_utils import RigidContactAPI
        robot = self.session.robot
        return head_arm_contacts(RigidContactAPI.get_contact_pairs(
            scene_idx=robot.scene.idx, query_set={robot}, with_set=None, current_only=False))

    def trace_head_actions(self, controller):
        original = controller._postprocess_action
        robot = self.session.robot

        def traced(action):
            before = None
            try:
                before = plain(action[robot.controller_action_idx['camera']])
            except Exception:
                self.error = traceback.format_exc()
            returned = original(action)
            if before is not None:
                try:
                    self.record('head_control', enabled=controller._enable_head_tracking,
                                planned_command=before,
                                applied_command=plain(returned[robot.controller_action_idx['camera']]),
                                actual_head_q=plain(robot.get_joint_positions()[robot.camera_control_idx]),
                                head_arm_contacts=self.read_head_arm_contacts())
                except Exception:
                    self.error = traceback.format_exc()
            return returned

        self._replace(controller, '_postprocess_action', traced)

    def geometry(self):
        from omnigibson.utils.usd_utils import RigidContactAPI

        robot = self.session.robot

        def pose(body):
            position, orientation = body.get_position_orientation()
            return {"prim_path": body.prim_path, "position": position.tolist(),
                    "orientation": orientation.tolist(), "aabb": [x.tolist() for x in body.aabb]}

        arms = {}
        for arm in robot.arm_names:
            fingers = robot.finger_links[arm]
            paths = {link.prim_path for link in fingers}
            current = RigidContactAPI.get_contact_pairs(
                scene_idx=robot.scene.idx, query_set=paths, with_set=None, current_only=True)
            recent = RigidContactAPI.get_contact_pairs(
                scene_idx=robot.scene.idx, query_set=paths, with_set=None, current_only=False)
            arms[arm] = {"eef": pose(robot.eef_links[arm]), "fingers": [pose(link) for link in fingers],
                         "current_contact_pairs": sorted(current), "recent_contact_pairs": sorted(recent),
                         "contact_rows_present": {path: path in RigidContactAPI._PATH_TO_ROW_IDX.get(robot.scene.idx, {})
                                                  for path in paths}}
        return {"arms": arms, "objects": {name: pose(obj) for name, obj in self.session.objects.items()},
                "surfaces": {name: pose(obj) for name, obj in self.session.surfaces.items()},
                "robot_root": pose(robot.root_link),
                "robot_base": pose(robot.links[robot.base_footprint_link_name]),
                "joint_positions": plain(robot.get_joint_positions()),
                "joint_names": list(robot.joints),
                "controls": self._read_controls(), "grasps": self._read_grasps(),
                "contact_matrix_available": robot.scene.idx in RigidContactAPI._CONTACT_MATRIX,
                "robot_current_contacts": sorted(RigidContactAPI.get_contact_pairs(
                    scene_idx=robot.scene.idx, query_set={robot}, with_set=None, current_only=True)),
                "robot_recent_contacts": sorted(RigidContactAPI.get_contact_pairs(
                    scene_idx=robot.scene.idx, query_set={robot}, with_set=None, current_only=False))}

    def record(self, kind, **fields):
        if (kind.startswith("phase_") or kind in ("planner_result", "retraction_final", "motion_error")
                or (kind == "contact_exit" and fields.get("event") in
                    ("exit_started", "exit_completed", "exit_not_needed"))):
            try:
                fields["geometry"] = self.geometry()
            except Exception:
                self.error = traceback.format_exc()
                fields["geometry_error"] = self.error
        super().record(kind, **fields)

    def trace_planner(self, planner):
        original = planner.plan_batch

        def traced(*args, **kwargs):
            # Snapshot the query before the planner can modify its tensors.
            # These are reads only: no extra IK, collision queries or physics.
            query = None
            try:
                def argument(index, name):
                    return args[index] if len(args) > index else kwargs.get(name)

                start, goal, config = [argument(i, name) for i, name in enumerate(
                    ('start_state', 'goal_pose', 'plan_config'))]
                query = {
                    'start': {name: plain(getattr(start, name, None)) for name in
                              ('position', 'velocity', 'joint_names')},
                    'goal': {name: plain(getattr(goal, name, None)) for name in
                             ('position', 'quaternion', 'name')},
                    'config': {name: plain(getattr(config, name, None)) for name in
                               ('enable_graph', 'enable_graph_attempt', 'max_attempts', 'timeout',
                                'enable_opt', 'enable_finetune_trajopt')},
                    'goal_frame': 'cuRobo base link; quaternion wxyz',
                    'cuda_graph_reset': os.environ.get('CUROBO_TORCH_CUDA_GRAPH_RESET'),
                }
            except Exception:
                self.error = traceback.format_exc()
            returned = original(*args, **kwargs)
            # The original result, success mask, and trajectories are returned
            # unchanged. Only copy already-computed metadata into the log.
            try:
                result = returned[0]
                fields = {name: plain(getattr(result, name, None)) for name in
                          ("success", "status", "valid_query", "attempts", "total_time", "solve_time",
                           "position_error", "rotation_error", "cspace_error", "used_graph", "graph_time",
                           "optimized_dt")}
                self.record("planner_result", planner=fields, query=query,
                            embodiment=str(kwargs.get("emb_sel")))
            except Exception:
                self.error = traceback.format_exc()
            return returned

        self._replace(planner, "plan_batch", traced)

    def trace_motion(self, controller):
        """Copy the last commanded joint target if motion execution raises.

        Both methods delegate exactly once. No extra controls or physics steps
        are emitted; generator send/return/close behavior is preserved.
        """
        robot = self.session.robot
        convert = robot.q_to_action
        execute = controller._execute_motion_plan
        contexts = []

        def traced_convert(q, *args, **kwargs):
            action = convert(q, *args, **kwargs)
            if contexts:
                try:
                    contexts[-1]["last_commanded_q"] = plain(q)
                    sample = {"control_step": self.session.result["control_steps"],
                              "target": plain(q),
                              "position": plain(robot.get_joint_positions()),
                              "velocity": plain(robot.get_joint_velocities())}
                    if hasattr(robot, 'get_joint_efforts'):
                        sample['effort'] = plain(robot.get_joint_efforts())
                    contexts[-1]['recent_tracking'].append(sample)
                except Exception:
                    self.error = traceback.format_exc()
            return action

        def traced_execute(*args, **kwargs):
            context = {'recent_tracking': deque(maxlen=32)}
            contexts.append(context)
            try:
                return (yield from execute(*args, **kwargs))
            except Exception as exc:
                try:
                    self.record("motion_error", error_type=type(exc).__name__, error=str(exc),
                                last_commanded_q=context.get('last_commanded_q'),
                                recent_tracking=list(context['recent_tracking']),
                                actual_q=plain(robot.get_joint_positions()),
                                actual_dq=plain(robot.get_joint_velocities()),
                                joint_names=list(robot.joints),
                                trunk_indices=plain(robot.trunk_control_idx),
                                arm_indices={a: plain(robot.arm_control_idx[a]) for a in robot.arm_names},
                                base_indices=plain(robot.base_control_idx))
                except Exception:
                    self.error = traceback.format_exc()
                raise
            finally:
                contexts.pop()

        self._replace(robot, "q_to_action", traced_convert)
        self._replace(controller, "_execute_motion_plan", traced_execute)

    def trace_planner_attempts(self, planner):
        """Read before MotionGenResult.copy_idx drops the used_graph flag."""
        for embodiment, generator in planner.mg.items():
            for method in ('_plan_from_solve_state_batch', '_plan_from_solve_state'):
                if not hasattr(generator, method):
                    continue
                original = getattr(generator, method)

                def traced(*args, _original=original, _embodiment=embodiment,
                           _method=method, _generator=generator, **kwargs):
                    returned = _original(*args, **kwargs)
                    try:
                        self.record('planner_attempt', embodiment=str(_embodiment), solver_method=_method,
                                    planner={name: plain(getattr(returned, name, None)) for name in
                                             ('success', 'status', 'valid_query', 'used_graph', 'graph_time',
                                              'optimized_dt')},
                                    timing={name: {
                                        'max_dt': plain(solver.traj_evaluator_config.max_dt),
                                        'horizon': solver.action_horizon,
                                    } for name in ('trajopt_solver', 'finetune_trajopt_solver')
                                        if (solver := getattr(_generator, name, None)) is not None},
                                    grasps=self._read_grasps())
                    except Exception:
                        self.error = traceback.format_exc()
                    return returned

                self._replace(generator, method, traced)

    def trace_trajectory_stages(self, planner):
        """Distinguish initial optimization from finetuning before status collapse."""
        name = '_solve_trajopt_from_solve_state'
        for embodiment, generator in planner.mg.items():
            if not hasattr(generator, name):
                continue
            original = getattr(generator, name)

            def traced(*args, _original=original, _embodiment=embodiment,
                       _generator=generator, **kwargs):
                returned = _original(*args, **kwargs)
                try:
                    solver = kwargs.get('trajopt_instance', args[7] if len(args) > 7 else None)
                    stage = ('finetune' if solver is not None and
                             solver is getattr(_generator, 'finetune_trajopt_solver', None)
                             else 'initial')
                    metrics = getattr(returned, 'metrics', None)
                    self.record('trajectory_stage', embodiment=str(_embodiment), stage=stage,
                                result={key: plain(getattr(returned, key, None)) for key in
                                        ('success', 'position_error', 'rotation_error',
                                         'optimized_dt', 'smooth_label', 'smooth_error')},
                                dt_exception=plain((getattr(returned, 'debug_info', None) or {})
                                                   .get('dt_exception', False)),
                                metrics={key: plain(getattr(metrics, key, None)) for key in
                                         ('feasible', 'constraint')})
                except Exception:
                    self.error = traceback.format_exc()
                return returned

            self._replace(generator, name, traced)

    def install(self, controller):
        super().install(controller)
        from task_undo.simulation.reachability_diagnostics import install_reachability_trace
        install_reachability_trace(self, controller, plain)
        self.trace_planner(controller._motion_generator)
        self.trace_planner_attempts(controller._motion_generator)
        self.trace_trajectory_stages(controller._motion_generator)
        if hasattr(controller, '_record_navigation_route'):
            self._replace(controller, '_record_navigation_route',
                          lambda **fields: self.record('navigation_route', **fields))
        self.trace_motion(controller)
        self.trace_head_actions(controller)
        if hasattr(controller, "_record_contact_exit"):
            self._replace(controller, "_record_contact_exit", lambda **fields: self.record("contact_exit", **fields))
        if hasattr(controller, "_record_grasp_lift"):
            self._replace(controller, "_record_grasp_lift", lambda **fields: self.record("grasp_lift", **fields))
        if hasattr(controller, "_record_grasp_search"):
            self._replace(controller, "_record_grasp_search", lambda **fields: self.record("grasp_search", **fields))
        if hasattr(controller, "_record_grasp_acquisition"):
            self._replace(controller, "_record_grasp_acquisition", lambda **fields: self.record("grasp_acquisition", **fields))
        if hasattr(controller, "_record_base_clearance"):
            self._replace(controller, "_record_base_clearance", lambda **fields: self.record("base_clearance", **fields))
        if hasattr(controller, "_record_interaction_search"):
            self._replace(controller, "_record_interaction_search", lambda **fields: self.record("interaction_search", **fields))

    def close(self):
        if self.stream.closed:
            return
        self.record("retraction_final")
        super().close()
