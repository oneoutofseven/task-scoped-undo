"""Separate geometric BASE search, conservative timing, and collision validation."""
import math

from task_undo.simulation.navigation import BaseGraphNavigation
from task_undo.simulation.retiming import retime_polyline
from task_undo.simulation.navigation import capture_navigation_routes


class RetimedBaseNavigation(BaseGraphNavigation):
    """Only a fully checked reference can replace failed BASE optimization.

    No raw solver path is executed. Preserve successful initial optimization
    geometry even when later time finetuning fails. If neither optimization
    nor graph search supplies a path, propose a direct joint-space base segment.
    That proposal is subject to exactly the same limits and collision checks.
    Timing stops at every vertex and uses the native
    velocity, acceleration and jerk limits. Check the full robot, held object,
    self-collision and independent chassis clearance before issuing a control.
    Native feedback execution / task budgets / protection checks remain active.
    Derivative bounds describe the nominal reference, not measured dynamics of
    the feedback controller, which may repeat a reference until it is reached.
    """

    def _record_navigation_route(self, **fields):
        pass

    def _navigate_to_pose(self, pose_2d, skip_obstacle_update=False):
        import torch as th
        import omnigibson as og
        from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection as Emb
        from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError

        planner = self._motion_generator
        generator = planner.mg[Emb.BASE]
        issued = False
        with capture_navigation_routes(generator) as captured:
            stream = super()._navigate_to_pose(pose_2d, skip_obstacle_update)
            try:
                for action in stream:
                    issued = True
                    yield action
                return
            except ActionPrimitiveError as error:
                if issued or error.reason != ActionPrimitiveError.Reason.PLANNING_ERROR:
                    raise
                original_error = error
            finally:
                stream.close()

        limits = generator.kinematics.get_joint_limits()
        path = captured.get('path')
        route_source = captured.get('source','native_graph') if path is not None else 'direct_segment'
        names = path.joint_names if path is not None else limits.joint_names
        columns = [planner.robot_joint_names.index(name) for name in names]
        if names != limits.joint_names:
            raise RuntimeError('Graph and native timing-limit joint order differ')
        current = self.robot.get_joint_positions().clone()
        points = [current[columns].tolist()]
        if path is not None:
            points.extend(path.position.detach().cpu().tolist())
        goal = list(points[-1])
        root_position = self.robot.root_link.get_position_orientation()[0]
        for axis, value in zip(('x','y'), pose_2d[:2]-root_position[:2]):
            goal[names.index('base_footprint_'+axis+'_joint')] = float(value)
        yaw_index = names.index('base_footprint_rz_joint')
        requested_yaw = float(pose_2d[2])
        goal[yaw_index] = requested_yaw+2*math.pi*round((goal[yaw_index]-requested_yaw)/(2*math.pi))
        points.append(goal)
        low, high = limits.position.detach().cpu().tolist()
        if any(not math.isfinite(q) or q < lo-1e-6 or q > hi+1e-6
               for point in points for q,lo,hi in zip(point,low,high)):
            raise original_error
        bounds = [th.minimum(-getattr(limits,key)[0],getattr(limits,key)[1]).detach().cpu().tolist()
                  for key in ('velocity','acceleration','jerk')]
        try:
            timed = retime_polyline(points,*bounds,dt=og.sim.get_sim_step_dt(),max_steps=3000)
        except ValueError as error:
            self._record_navigation_route(event='retiming_rejected',error=str(error),route_source=route_source)
            raise original_error
        trajectory = current.repeat(len(timed['positions']),1)
        trajectory[:,columns] = th.tensor(timed['positions'],dtype=current.dtype,device=current.device)
        if len(trajectory)>1 and float((trajectory[1:]-trajectory[:-1]).abs().max())>.01+1e-6:
            raise RuntimeError('Retimed reference exceeds native waypoint increment')
        held = self._get_obj_in_hand()
        attached = {self.robot.eef_link_names[self.arm]:held.root_link} if held is not None else None
        planner.update_obstacles()
        for chunk in trajectory.split(128):
            collision = planner.check_collisions(chunk,initial_joint_pos=current,
                self_collision_check=True,skip_obstacle_update=True,attached_obj=attached,attached_obj_scale=None)
            if bool(collision.any()):
                self._record_navigation_route(event='retiming_rejected',error='full_robot_collision',route_source=route_source)
                raise original_error
        poses = trajectory[:,self.robot.base_control_idx].clone()
        poses[:,:2] += root_position[:2]
        costs = self._base_collision_costs(poses)
        if bool((costs > 1e-7).any()):
            self._record_navigation_route(event='retiming_rejected',error='full_chassis_clearance',
                                           route_source=route_source,
                                           rejected=int((costs > 1e-7).sum()),max_cost=float(costs.max()))
            raise original_error
        self._record_navigation_route(event='retimed_route_validated',joint_names=names,
            route_source=route_source,
            geometric_positions=points,velocity_limits=bounds[0],acceleration_limits=bounds[1],
            jerk_limits=bounds[2],reference_steps=len(trajectory),dt=timed['dt'],
            max_joint_increment=timed['max_joint_increment'],
            duration_s=timed['duration_s'],segments=timed['segments'],
            original_error=str(original_error),held_object=held.name if held is not None else None)
        yield from self._execute_motion_plan(trajectory)
        self._record_navigation_route(event='retimed_route_executed',reference_steps=len(trajectory))
