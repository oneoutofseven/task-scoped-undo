"""Validate base candidates and paths against the measured chassis envelope.

The upstream sparse spheres can miss parts of the physical chassis. A grid of
circumscribed cells covers its entire link-frame AABB, including face interiors.
This is conservative geometry, not a relaxation of collision checking.
"""
import itertools
import math
import copy
from contextlib import contextmanager
from pathlib import Path
import tempfile


def cover_box(lower, upper, cell_limits=(.06, .06, .15)):
    if any(not math.isfinite(v) for v in [*lower, *upper, *cell_limits]):
        raise ValueError('Finite geometry required')
    if any(upper[i] <= lower[i] or cell_limits[i] <= 0 for i in range(3)):
        raise ValueError('Positive box and cell dimensions required')
    counts = [math.ceil((upper[i]-lower[i])/cell_limits[i]) for i in range(3)]
    widths = [(upper[i]-lower[i])/counts[i] for i in range(3)]
    radius = math.sqrt(sum(w*w for w in widths))/2
    return [{'center': [lower[i]+(index[i]+.5)*widths[i] for i in range(3)],
             'radius': radius}
            for index in itertools.product(*(range(n) for n in counts))]


def measured_base_cover(robot, link_name='base_link'):
    """Read the physical meshes; return a complete cover in the link frame."""
    import torch as th
    from omnigibson.utils import transform_utils as T

    link = robot.links[link_name]
    inverse = th.linalg.inv(T.pose2mat(link.get_position_orientation()))
    vertices = th.cat([T.transform_points(mesh.points, inverse @ mesh.scaled_transform)
                       for mesh in link.collision_meshes.values()])
    lower, upper = vertices.amin(0).tolist(), vertices.amax(0).tolist()
    spheres = cover_box(lower, upper)
    report = {'link': link_name, 'local_aabb': [lower, upper],
              'cell_limits_m': [.06, .06, .15], 'sphere_count': len(spheres),
              'sphere_radius_m': spheres[0]['radius'],
              'policy': 'original planner plus full chassis candidate/path validation',
              'extra_world_clearance_m': .01}
    return th.tensor([s['center']+[s['radius']] for s in spheres]), report


def with_base_spheres(config, spheres, clearance=0.):
    """Replace only the chassis geometry; preserve all native constraints."""
    result = copy.deepcopy(config)
    result['robot_cfg']['kinematics']['collision_spheres']['base_link'] = copy.deepcopy(spheres)
    if not math.isfinite(clearance) or clearance < 0:
        raise ValueError('Nonnegative finite chassis clearance required')
    for sphere in result['robot_cfg']['kinematics']['collision_spheres']['base_link']:
        sphere['radius'] += clearance
    return result


@contextmanager
def base_planner_configs(robot, clearance=0.):
    """Give BASE optimization the same chassis cover used by final validation.

    ARM and DEFAULT keep their upstream geometry and solver dimensions. Unlike
    the abandoned all-embodiment dense model, no arm optimizer is expanded.
    Source configuration files and self-collision rules remain unchanged.
    """
    import yaml
    from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection as Emb
    cover, _ = measured_base_cover(robot)
    spheres = [{'center': row[:3], 'radius': row[3]} for row in cover.tolist()]
    paths = dict(robot.curobo_path)
    config = yaml.safe_load(Path(paths[Emb.BASE]).read_text())
    with tempfile.TemporaryDirectory(prefix='task_undo_base_') as folder:
        path = Path(folder)/'base.yaml'
        path.write_text(yaml.safe_dump(with_base_spheres(config, spheres, clearance)))
        paths[Emb.BASE] = str(path)
        yield paths


class BaseGeometryValidation:
    """Retain the original optimizer and reject insufficiently clear solutions.

    The full envelope is checked in candidate filtering and on every interpolated
    base waypoint, using the same world collision meshes. It does not enter the
    trajectory optimizer's summed cost or change any self-collision settings.
    """
    def install_base_geometry(self):
        import torch as th
        from omnigibson.utils import transform_utils as T
        self._base_cover, self.collision_geometry = measured_base_cover(self.robot)
        footprint = self.robot.links[self.robot.base_footprint_link_name]
        self._chassis_in_footprint = T.relative_pose_transform(
            *self.robot.links['base_link'].get_position_orientation(),
            *footprint.get_position_orientation())
        pose = self.robot.get_joint_positions()[self.robot.base_control_idx].clone()
        pose[:2] += self.robot.root_link.get_position_orientation()[0][:2]
        predicted = self._get_robot_pose_from_2d_pose(pose)
        actual = footprint.get_position_orientation()
        distance = float(th.linalg.norm(predicted[0]-actual[0]))
        angle = float(T.get_orientation_diff_in_radian(predicted[1], actual[1]))
        self.collision_geometry['initial_frame_error'] = {'position_m': distance, 'angle_rad': angle}
        if distance > .001 or angle > .001:
            raise RuntimeError('Base envelope frame does not match measured robot kinematics')

    def _record_base_clearance(self, **fields):
        pass

    def _base_collision_costs(self, poses):
        import torch as th
        from omnigibson.utils import transform_utils as T
        from omnigibson.action_primitives.curobo import CuRoboEmbodimentSelection as Emb
        from curobo.geom.sdf.world import CollisionQueryBuffer

        planner = self._motion_generator
        checker = planner.mg[Emb.DEFAULT].rollout_fn.primitive_collision_constraint.world_coll_checker
        inverse_root = th.linalg.inv(T.pose2mat(self.robot.root_link.get_position_orientation()))
        spheres = []
        for pose in poses:
            footprint = self._get_robot_pose_from_2d_pose(pose)
            chassis = T.pose_transform(*footprint, *self._chassis_in_footprint)
            transform = inverse_root @ T.pose2mat(chassis)
            sample = self._base_cover.clone()
            sample[:, :3] = T.transform_points(sample[:, :3], transform)
            spheres.append(sample)
        spheres = planner.tensor_args.to_device(th.stack(spheres).unsqueeze(1))
        costs = []
        for chunk in spheres.split(128):
            buffer = CollisionQueryBuffer.initialize_from_shape(
                chunk.shape, planner.tensor_args, checker.collision_types)
            values = checker.get_sphere_collision(chunk, buffer,
                planner.tensor_args.to_device([1.]),
                activation_distance=planner.tensor_args.to_device([.01]))
            costs.append(values.reshape(len(chunk), -1).amax(1).detach().cpu())
        costs = th.cat(costs)
        if not bool(th.isfinite(costs).all()):
            raise RuntimeError('Nonfinite chassis collision result')
        return costs

    def _validate_poses(self, candidate_poses, *args, **kwargs):
        valid = super()._validate_poses(candidate_poses, *args, **kwargs)
        indices = valid.nonzero().flatten().tolist()
        if indices:
            costs = self._base_collision_costs([candidate_poses[i] for i in indices])
            valid = valid.clone()
            for i, cost in zip(indices, costs):
                if float(cost) > 1e-7:
                    valid[i] = False
            self._record_base_clearance(event='candidates', count=len(indices),
                                       rejected=int((costs > 1e-7).sum()), max_cost=float(costs.max()))
        return valid

    def _plan_joint_motion(self, *args, **kwargs):
        trajectory = super()._plan_joint_motion(*args, **kwargs)
        embodiment = kwargs.get('embodiment_selection', args[2] if len(args) > 2 else None)
        if getattr(embodiment, 'value', None) != 'base':
            return trajectory
        from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
        poses = trajectory[:, self.robot.base_control_idx].clone()
        poses[:, :2] += self.robot.root_link.get_position_orientation()[0][:2]
        costs = self._base_collision_costs(poses)
        self._record_base_clearance(event='trajectory', waypoints=len(poses),
                                   rejected=int((costs > 1e-7).sum()), max_cost=float(costs.max()))
        if bool((costs > 1e-7).any()):
            raise ActionPrimitiveError(ActionPrimitiveError.Reason.PLANNING_ERROR,
                'Proposed base path intersects the full chassis envelope clearance',
                {'rejected_waypoints': int((costs > 1e-7).sum()), 'max_cost': float(costs.max())})
        return trajectory
