"""Bounded graph search and route capture for empty-hand and loaded navigation."""

from contextlib import contextmanager
import os


def enable_cuda_graph_reset(cuda_version):
    """Allow cuRobo's supported cache rebuild when local/graph seeds change.

    This worker-wide setting affects CUDA computation caches, not simulation
    state. The pinned cuRobo requires explicit opt-in and CUDA >= 12.
    """
    if cuda_version is None or int(cuda_version.split('.')[0]) < 12:
        raise RuntimeError('Held-object graph planning requires CUDA >= 12 for graph cache reset')
    os.environ['CUROBO_TORCH_CUDA_GRAPH_RESET'] = '1'


@contextmanager
def graph_fallback(planner):
    """Adapt this synchronous query only; preserve upstream planning budgets.

    OmniGibson disables cuRobo's graph fallback in compute_trajectories. Adapt
    its plan_batch call rather than copy its attachment/collision bookkeeping.
    The original method is restored even if planning or diagnostics raises.
    """
    original = planner.plan_batch

    def plan_batch(start_state, goal_pose, plan_config, *args, **kwargs):
        config = plan_config.clone()
        config.enable_graph_attempt = 3
        return original(start_state, goal_pose, config, *args, **kwargs)

    planner.plan_batch = plan_batch
    try:
        yield
    finally:
        planner.plan_batch = original


class BaseGraphNavigation:
    """Enable geometric seeds after three failed local planning attempts.

    ARM/DEFAULT queries keep upstream behavior. Empty-hand and loaded base
    motion use the same search policy: clutter can block either trajectory.
    Motion execution, attached geometry, collision checks and scoring are
    unchanged. This adapter assumes the session's synchronous controller.
    """

    def _plan_joint_motion(self, *args, **kwargs):
        embodiment = kwargs.get('embodiment_selection', args[2] if len(args) > 2 else None)
        if getattr(embodiment, 'value', None) != 'base':
            return super()._plan_joint_motion(*args, **kwargs)
        with graph_fallback(self._motion_generator):
            return super()._plan_joint_motion(*args, **kwargs)


@contextmanager
def capture_navigation_routes(generator):
    """Copy singleton geometric proposals from existing graph/initial solves.

    A copied optimization path is not an executable trajectory. The caller
    must retime it and independently check every reference configuration.
    Clone before finetuning can overwrite the solver's reusable buffers.
    """
    original = generator.graph_search
    owned, previous = 'graph_search' in vars(generator), vars(generator).get('graph_search')
    captured = {}

    def search(*args, **kwargs):
        result = original(*args, **kwargs)
        success = result.success.reshape(-1)
        path = getattr(result, 'interpolated_plan', None)
        if (result.valid_query and len(success) == 1 and bool(success[0])
                and path is not None and path.position.shape[0] == 1):
            end = int(result.path_buffer_last_tstep[0])
            captured['path'] = path[0].trim_trajectory(0, end).clone()
            captured['source'] = 'native_graph'
        return result

    solve_name = '_solve_trajopt_from_solve_state'
    solve = getattr(generator, solve_name, None)
    solve_owned = solve_name in vars(generator)
    def initial(*args, **kwargs):
        result = solve(*args, **kwargs)
        solver = kwargs.get('trajopt_instance', args[7] if len(args)>7 else None)
        finetuning = solver is not None and solver is getattr(generator,'finetune_trajopt_solver',None)
        success = result.success.reshape(-1)
        path = getattr(result,'solution',None)
        if not finetuning and len(success)==1 and bool(success[0]) and path is not None:
            shape = path.position.shape
            if len(shape)==2 or (len(shape)==3 and shape[0]==1):
                proposal = (path[0] if len(shape)==3 else path).clone()
                # cuRobo's active dynamics model defines this coordinate order.
                names = list(generator.kinematics.get_joint_limits().joint_names)
                if proposal.joint_names is None and proposal.position.shape[-1]==len(names):
                    proposal.joint_names = names
                if proposal.joint_names == names:
                    captured['path'] = proposal
                    captured['source'] = 'initial_optimization'
        return result

    generator.graph_search = search
    if solve is not None:
        setattr(generator,solve_name,initial)
    try:
        yield captured
    finally:
        if owned:
            generator.graph_search = previous
        else:
            delattr(generator, 'graph_search')
        if solve is not None:
            if solve_owned:
                setattr(generator,solve_name,solve)
            else:
                delattr(generator,solve_name)
