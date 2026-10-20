"""Compose the Tiago controller and category-specific contact targets.

Import only after session.py has loaded Warp's compatibility modules.
"""

import torch as th
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from task_undo.simulation.attachment_frames import install_attachment_frame
from task_undo.simulation.release_control import PlaceReleaseControl
from task_undo.simulation.contact_retreat import ContactReleaseRetreat
from task_undo.simulation.navigation import enable_cuda_graph_reset
from task_undo.simulation.feasible_navigation import FeasibleMotionPlanning
from task_undo.simulation.retimed_navigation import RetimedBaseNavigation
from task_undo.simulation.grasp_acquisition import ContactGraspAcquisition
from task_undo.simulation.contact_target import refine_thin_contact

from task_undo.simulation.grasp_lift import GuardedGraspLift
from task_undo.simulation.grasp_search import JointGraspSearch
from task_undo.simulation.placement import StablePlacement
from task_undo.simulation.base_geometry import BaseGeometryValidation, base_planner_configs
from task_undo.simulation.interaction_search import InteractionSearch
from task_undo.simulation.loaded_motion import LoadedArmMotion
from task_undo.simulation.planner_device import planner_device_defaults, check_evaluator_devices

CONTACT_DEPTH_M = {"plate": 0.015, "hardback": 0.01, "mug": 0.008}
MUG_RIM_FRACTION = 0.3


class DinnerPrimitives(LoadedArmMotion, ContactGraspAcquisition, InteractionSearch, BaseGeometryValidation, StablePlacement, GuardedGraspLift,
                       JointGraspSearch, RetimedBaseNavigation, FeasibleMotionPlanning,
                       ContactReleaseRetreat, PlaceReleaseControl, StarterSemanticActionPrimitives):
    def __init__(self, *args, motion_profile="legacy", **kwargs):
        if motion_profile not in ("legacy", "loaded_waypoints_v1"):
            raise ValueError("Unknown motion profile")
        self.motion_profile = motion_profile
        enable_cuda_graph_reset(th.version.cuda)
        from omnigibson.action_primitives.curobo import CuRoboMotionGenerator
        from omnigibson.action_primitives.starter_semantic_action_primitives import m
        from omnigibson.macros import gm
        super().__init__(*args, **kwargs, skip_curobo_initilization=True)
        device = f"cuda:{0 if gm.GPU_ID is None else int(gm.GPU_ID)}"
        with base_planner_configs(self.robot, clearance=.01) as paths, planner_device_defaults(device):
            self._motion_generator = CuRoboMotionGenerator(
                robot=self.robot, robot_cfg_path=paths, batch_size=self._curobo_batch_size,
                device=device,
                collision_activation_distance=m.DEFAULT_COLLISION_ACTIVATION_DISTANCE)
        self.install_base_geometry()
        self.collision_geometry['base_optimizer_uses_full_cover'] = True
        self.collision_geometry['base_optimizer_added_clearance_m'] = .01
        self.collision_geometry['policy'] = 'full chassis BASE optimization and independent path validation; original ARM and DEFAULT models'
        self.collision_geometry['planner_device'] = str(self._motion_generator.tensor_args.device)
        self.collision_geometry['trajectory_evaluator_devices'] = check_evaluator_devices(self._motion_generator)
        self.collision_geometry['device_forwarding_adapter'] = 'curobo_559_scoped'
        install_attachment_frame(self._motion_generator, scale=1.0)

    def _sample_grasp_pose(self, obj):
        pregrasp, grasp = super()._sample_grasp_pose(obj)
        if obj.category not in CONTACT_DEPTH_M:
            return self._select_grasp(obj, (pregrasp, grasp))
        lengths = self.robot.eef_to_fingertip_lengths[self.arm].values()
        finger_length = sum(lengths) / len(lengths)
        # The upstream EEF target is at the object's top. Account for finger
        # length so thin objects do not require a path through the table.
        grasp = (grasp[0] + th.tensor([0., 0., finger_length - CONTACT_DEPTH_M[obj.category]]), grasp[1])
        if obj.category == "mug":
            # Shallow rim contact allows the gripper to release outside the cup.
            lower, upper = obj.aabb
            offset = th.tensor([float(upper[0] - lower[0]) * MUG_RIM_FRACTION, 0., 0.])
            pregrasp = (pregrasp[0] + offset, pregrasp[1])
            grasp = (grasp[0] + offset, grasp[1])
        grasp = refine_thin_contact(self, obj, grasp, finger_length)
        return self._select_grasp(obj, (pregrasp, grasp))
