"""OmniGibson setup and the shared, observed GRASP/PLACE execution loop."""

import json
import random
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch as th
import yaml
# Kit can otherwise inject an incompatible, older warp.torch module.
import warp.context
import warp.torch
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.controllers import IsGraspingState
from omnigibson.object_states import OnTop
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitiveSet as Primitive,
)

from task_undo.simulation.primitives import DinnerPrimitives
from task_undo.common.storage import write_json
from task_undo.domain.case_layout import object_ids, scene_settings, validate_layout


def create_environment(case, output, seed, checkpoint=None, *, checkpoint_settle_steps=30, restore_mode="default"):
    if restore_mode != "default" and (checkpoint is None or checkpoint_settle_steps == 0):
        raise ValueError("Contact refresh requires a normal checkpoint initialization")
    validate_layout(case)
    layout = scene_settings(case)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = False
    with open(Path(og.example_config_path) / "tiago_primitives.yaml") as handle:
        config = yaml.safe_load(handle)
    config["render"].update(viewer_width=640, viewer_height=480)
    config["scene"].update(scene_model=layout["scene_model"],
                           load_object_categories=layout["load_object_categories"])
    if "robot_position" in layout:
        config["robots"][0].update(position=layout["robot_position"],
                                    orientation=layout["robot_orientation"], pose_frame="scene")
    config["objects"] = case["objects"]
    write_json(output / "config.json", config)
    env = og.Environment(configs=config)
    if checkpoint is not None:
        # Initialization only. No workflow may call restore after undo_started.
        env.scene.restore(scene_file=str(checkpoint))
    for _ in range(checkpoint_settle_steps if checkpoint else 60):
        og.sim.step()
    if checkpoint is not None:
        from task_undo.simulation.checkpoint_contacts import initialize_contacts
        initialize_contacts(env.scene, og.sim.step, restore_mode)
        if restore_mode != "default":
            write_json(output / "checkpoint_initialization.json", {
                "mode": restore_mode, "settle_steps": checkpoint_settle_steps, "contact_steps": 5,
                "phase": "before_undo", "pose_setters": 0, "predicate_setters": 0})
    og.sim.viewer_camera.set_position_orientation(
        th.tensor(layout["camera_position"]), th.tensor(layout["camera_orientation"]))
    for _ in range(5):
        og.sim.render()
    return env


class Session:
    """Execute actions, record observed results, and check optional protection."""

    def __init__(self, env, case, output, result, max_steps):
        self.env, self.case, self.output = env, case, output
        self.result, self.max_steps = result, max_steps
        self.result['support_observation_policy'] = 'kinematic_sleep_cache_v2'
        self.robot = env.scene.robots[0]
        self.objects = {name: env.scene.object_registry("name", name) for name in object_ids(case)}
        self.surfaces = {role: env.scene.object_registry("name", name) for role, name in case["surfaces"].items()}
        self.history = (output / "public_history.jsonl").open("w")
        self.writer = imageio.get_writer(str(output / "video.mp4"), fps=10)
        self.event_count = 0
        self.controller = self.monitor = self.trace = self.subscription = None
        self.physics_trace = None
        self.physics_steps = 0
        self.monitor_error = None
        self.gripper_diagnostics = None

    def observe(self):
        from task_undo.simulation.support_query import current_supports
        return {"objects": {
            name: {"position": obj.get_position_orientation()[0].tolist(),
                   "orientation": obj.get_position_orientation()[1].tolist(),
                   "on": current_supports(obj.states[OnTop], self.surfaces),
                   "held": any(self.robot.is_grasping(arm=arm, candidate_obj=obj) == IsGraspingState.TRUE
                               for arm in self.robot.arm_names)}
            for name, obj in self.objects.items()},
            "hand_empty": all(self.robot.is_grasping(arm=arm) == IsGraspingState.FALSE
                              for arm in self.robot.arm_names)}

    def event(self, kind, **fields):
        self.event_count += 1
        entry = dict(event_id=self.event_count, episode_id=self.case["episode_id"], kind=kind,
                     control_step=self.result["control_steps"], sim_seconds=float(og.sim.current_time), **fields)
        self.history.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.history.flush()
        return self.event_count

    def support_diagnostics(self):
        """Read failure evidence without advancing physics or changing predicates.

        The pinned OG tensor states expose a CPU mirror via _get_value. Compare
        it with the public cached read to help diagnose support-check failures.
        These readings are diagnostic only and never replace success checks.
        """
        from omnigibson.object_states import Adjacency, Touching

        def value(state, surface, raw=False):
            result = state._get_value(surface) if raw else state.get_value(surface)
            return result.tolist() if hasattr(result, "tolist") else result

        return {name: {
            "aabb": [corner.tolist() for corner in obj.aabb],
            "is_asleep": bool(obj.is_asleep),
            "supports": {role: {
                "aabb": [corner.tolist() for corner in surface.aabb],
                "is_asleep": bool(surface.is_asleep),
                "predicates": {state.__name__: {
                    "public": value(obj.states[state], surface),
                    "raw": value(obj.states[state], surface, raw=True)}
                    for state in (OnTop, Touching, Adjacency)}}
                for role, surface in self.surfaces.items()}}
            for name, obj in self.objects.items()}

    def capture(self, name=None):
        frame = og.sim.viewer_camera.get_obs()[0]["rgb"].cpu().numpy()[..., :3].astype(np.uint8)
        self.writer.append_data(frame)
        if name:
            imageio.imwrite(self.output / f"{name}.png", frame)

    def start_recording(self):
        """Record observations for every policy, without consulting any labels."""
        if self.trace is not None:
            raise RuntimeError("Execution recording already started")
        self.trace = (self.output / "execution_trace.jsonl").open("w")
        self.physics_trace = (self.output / "physics_trace.jsonl").open("w")
        self.result["trace_schema_version"] = 2

        def check_grasps(_):
            try:
                # Read the pinned OG assisted-grasp bookkeeping after each physics
                # substep, including steps executed inside upstream primitives.
                held = [obj.name for obj in self.robot._ag_obj_in_hand.values() if obj is not None]
                self.physics_steps += 1
                self.physics_trace.write(json.dumps({"physics_step": self.physics_steps,
                    "control_step": self.result["control_steps"], "held_objects": sorted(set(held))}) + "\n")
                if self.monitor:
                    self.monitor.check_grasp(held)
            except Exception as exc:
                self.monitor_error = repr(exc)

        self.subscription = og.sim._physics_context._physx_interface.subscribe_physics_on_step_events(
            check_grasps, pre_step=False, order=1)
        self.observe_execution()

    def start_protection(self, monitor):
        self.monitor = monitor
        self.start_recording()

    def observe_execution(self):
        if self.trace is None:
            return
        if self.monitor_error:
            raise RuntimeError(f"Recording callback failed: {self.monitor_error}")
        state = self.observe()
        self.trace.write(json.dumps({"control_step": self.result["control_steps"],
                                     "sim_seconds": float(og.sim.current_time), "observation": state}) + "\n")
        if self.monitor:
            self.monitor.check(state, self.result["control_steps"])
            self.monitor.require_valid()

    def idle(self, steps=30):
        """Let physics advance with no new robot command; still record evidence."""
        for _ in range(steps):
            if self.result["control_steps"] >= self.max_steps:
                raise RuntimeError("Control step budget exhausted")
            og.sim.step()
            self.result["control_steps"] += 1
            self.observe_execution()
            if self.result["control_steps"] % 3 == 0:
                self.capture()

    def initialize_controller(self):
        self.result["phase"] = "planner"
        print("Initializing planner", flush=True)
        # cuRobo locks the head at planning time. Automatic tracking overwrites
        # those commands and can collide the head with the auxiliary arm.
        self.controller = DinnerPrimitives(self.env, self.robot, enable_head_tracking=False, curobo_batch_size=1,
                                          motion_profile=self.result.get("motion_profile", "legacy"))
        write_json(self.output / 'planner_geometry.json', self.controller.collision_geometry)
        if self.gripper_diagnostics:
            self.gripper_diagnostics.install(self.controller)
        self.observe_execution()

    def action(self, task, action_name, object_id, target_surface=None):
        primitive = Primitive[action_name]
        argument = self.objects[object_id] if primitive == Primitive.GRASP else self.surfaces[target_surface]
        if self.monitor and object_id in self.monitor.objects:
            raise ValueError(f"Reference executor cannot manipulate protected object {object_id}")
        self.result["phase"] = f"{task['task_id']}:{action_name}:{object_id}"
        print(self.result["phase"], flush=True)
        request = self.event("action_requested", task_id=task["task_id"], initiator=task["initiator"],
                             executor=task["executor"], action=action_name, object_id=object_id,
                             target_surface=target_surface, observation=self.observe())
        try:
            for control in self.controller.apply_ref(primitive, argument, attempts=3):
                if self.result["control_steps"] >= self.max_steps:
                    raise RuntimeError("Control step budget exhausted")
                self.env.step(control)
                self.result["control_steps"] += 1
                self.observe_execution()
                if self.result["control_steps"] % 3 == 0:
                    self.capture()
            state = self.observe()
            valid = state["objects"][object_id]["held"] if primitive == Primitive.GRASP else True
            if primitive == Primitive.PLACE_ON_TOP:
                for step in range(30):
                    og.sim.step()
                    self.observe_execution()
                    state = self.observe()
                    valid = valid and target_surface in state["objects"][object_id]["on"] and state["hand_empty"]
                    if step % 3 == 0:
                        self.capture()
            self.observe_execution()
            if not valid:
                raise RuntimeError("Observed action postcondition failed")
        except Exception as exc:
            self.event("action_failed", request_event_id=request, task_id=task["task_id"],
                       error=str(exc), observation=self.observe())
            raise
        self.event("action_succeeded", request_event_id=request, task_id=task["task_id"],
                   action=action_name, object_id=object_id, target_surface=target_surface, observation=state)
        self.capture(f"{task['task_id']}_{object_id}_{action_name.lower()}")

    def close(self):
        if self.gripper_diagnostics:
            self.gripper_diagnostics.close()
        if self.subscription is not None:
            self.subscription.unsubscribe()
        if self.trace:
            self.trace.close()
        if self.physics_trace:
            self.physics_trace.close()
        self.history.close()
        self.writer.close()
