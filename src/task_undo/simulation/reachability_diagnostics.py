"""Observe existing candidate and IK calls without sampling or solving again."""
import traceback


def install_reachability_trace(trace, controller, plain):
    planner = controller._motion_generator
    context = {'batch': None, 'next_batch': 0}

    def emit(kind, **values):
        try:
            trace.record(kind, candidate_batch=context['batch'], **plain(values))
        except Exception:
            trace.error = traceback.format_exc()

    def arg(args, kwargs, index, name):
        return args[index] if len(args) > index else kwargs.get(name)

    original_validate = controller._validate_poses

    def validate(*args, **kwargs):
        previous = context['batch']
        context['next_batch'] += 1
        context['batch'] = context['next_batch']
        try:
            emit('candidate_batch_started', poses=arg(args, kwargs, 0, 'candidate_poses'),
                 eef_pose=arg(args, kwargs, 1, 'eef_pose'))
            result = original_validate(*args, **kwargs)
            emit('candidate_batch_result', valid=result)
            return result
        except Exception as exc:
            emit('candidate_batch_error', error=repr(exc))
            raise
        finally:
            context['batch'] = previous

    trace._replace(controller, '_validate_poses', validate)
    original_collisions = planner.check_collisions

    def collisions(*args, **kwargs):
        # Input is copied before the original can modify its tensors.
        snapshot = None
        if context['batch'] is not None:
            try:
                snapshot = plain(arg(args, kwargs, 0, 'q'))
            except Exception:
                trace.error = traceback.format_exc()
        result = original_collisions(*args, **kwargs)
        if context['batch'] is not None:
            emit('candidate_collisions', joint_positions=snapshot, invalid=result)
        return result

    trace._replace(planner, 'check_collisions', collisions)
    original_ik = planner.solve_ik_batch

    def ik(*args, **kwargs):
        try:
            start = arg(args, kwargs, 0, 'start_state')
            goal = arg(args, kwargs, 1, 'goal_pose')
            emit('reachability_ik_started',
                 joints=getattr(start, 'position', None),
                 joint_names=getattr(start, 'joint_names', None),
                 target_position=getattr(goal, 'position', None),
                 target_quaternion=getattr(goal, 'quaternion', None),
                 frame='cuRobo base link; quaternion wxyz')
        except Exception:
            trace.error = traceback.format_exc()
        try:
            returned = original_ik(*args, **kwargs)
        except Exception as exc:
            emit('reachability_ik_error', error=repr(exc))
            raise
        try:
            raw, success, _ = returned
            emit('reachability_ik_result', success=success,
                 details={k: getattr(raw, k, None) for k in
                          ('success', 'position_error', 'rotation_error', 'error', 'solve_time')})
        except Exception:
            trace.error = traceback.format_exc()
        return returned

    trace._replace(planner, 'solve_ik_batch', ik)

    for embodiment, generator in getattr(planner, 'mg', {}).items():
        constraint = generator.rollout_fn.primitive_collision_constraint
        original_query = constraint.coll_check_fn
        worlds = set()

        def query(*args, _original=original_query, _constraint=constraint,
                  _generator=generator, _emb=str(embodiment), _worlds=worlds, **kwargs):
            result = _original(*args, **kwargs)
            if context['batch'] is not None:
                try:
                    emit('candidate_collision_detail', embodiment=_emb,
                         spheres=args[0], costs=result,
                         activation_distance=kwargs.get('activation_distance'))
                    phase = trace.session.result['phase']
                    if phase not in _worlds:
                        world = _constraint.world_coll_checker.world_model
                        if isinstance(world, list): world = world[0]
                        kc = _generator.kinematics.kinematics_config
                        emit('candidate_world', embodiment=_emb,
                             sphere_link_indices=kc.link_sphere_idx_map,
                             link_indices=kc.link_name_to_idx_map,
                             meshes=[{k:getattr(mesh,k,None) for k in ('name','pose','vertices','faces')}
                                     for mesh in world.mesh])
                        _worlds.add(phase)
                except Exception:
                    trace.error = traceback.format_exc()
            return result

        trace._replace(constraint, 'coll_check_fn', query)
        solver = generator.ik_solver
        original_result = solver._get_result

        def ik_metrics(num_seeds, result, goal_pose, return_seeds,
                       _original=original_result, _emb=str(embodiment)):
            if context['batch'] is not None:
                try:
                    metrics = result.metrics
                    emit('candidate_ik_constraints', embodiment=_emb,
                         metrics={k:getattr(metrics,k,None) for k in
                                  ('feasible','constraint','position_error','rotation_error')})
                except Exception:
                    trace.error = traceback.format_exc()
            return _original(num_seeds, result, goal_pose, return_seeds)

        trace._replace(solver, '_get_result', ik_metrics)
