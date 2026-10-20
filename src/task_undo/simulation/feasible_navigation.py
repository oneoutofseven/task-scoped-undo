"""Use evaluated initial trajectories after repeated BASE or ARM failures."""
from contextlib import contextmanager, nullcontext
from task_undo.simulation.batch_paths import singleton_batch_paths

from task_undo.simulation.navigation import BaseGraphNavigation


@contextmanager
def evaluated_initial_trajectory(planner):
    """Disable optional finetuning, retaining optimization and result validation.

    With this option the pinned cuRobo selects the initial optimization result
    through its normal interpolated collision, convergence and smoothness
    evaluation. This does not return or execute the unevaluated intermediate
    result exposed by return_all_solutions=True during ordinary finetuning.
    """
    original = planner.plan_batch
    owned, previous = 'plan_batch' in vars(planner), vars(planner).get('plan_batch')

    def plan(start_state, goal_pose, plan_config, *args, **kwargs):
        config = plan_config.clone()
        config.enable_finetune_trajopt = False
        # The pinned batch solver can return [time, joints] when optional
        # finetuning is disabled for one goal. Adapt only its result accessor.
        generators = getattr(planner, 'mg', {})
        generator = generators.get(kwargs.get('emb_sel'))
        context = singleton_batch_paths(generator) if generator is not None else nullcontext()
        with context:
            return original(start_state, goal_pose, config, *args, **kwargs)

    planner.plan_batch = plan
    try:
        yield
    finally:
        if owned:
            planner.plan_batch = previous
        else:
            delattr(planner, 'plan_batch')


class FeasibleMotionPlanning(BaseGraphNavigation):
    """Change optional finetuning after two failures within one primitive.

    Successful BASE queries reset the BASE counter. ARM cleanup queries may
    succeed between approach failures, so ARM successes retain their counter.
    Counters are independent and reset for each primitive. No extra query is
    issued; collision, convergence, smoothness checks and budgets are unchanged.
    """
    _base_query_failures = 0
    _arm_query_failures = 0

    def apply_ref(self, *args, **kwargs):
        previous = self._base_query_failures, self._arm_query_failures
        self._base_query_failures = self._arm_query_failures = 0
        try:
            return (yield from super().apply_ref(*args, **kwargs))
        finally:
            self._base_query_failures, self._arm_query_failures = previous

    def _plan_joint_motion(self, *args, **kwargs):
        embodiment = kwargs.get('embodiment_selection', args[2] if len(args) > 2 else None)
        kind = getattr(embodiment, 'value', None)
        if kind not in ('base', 'arm'):
            return super()._plan_joint_motion(*args, **kwargs)
        counter = f'_{kind}_query_failures'
        alternative = (evaluated_initial_trajectory(self._motion_generator)
                       if getattr(self, counter) >= 2 else nullcontext())
        try:
            with alternative:
                result = super()._plan_joint_motion(*args, **kwargs)
        except Exception:
            setattr(self, counter, getattr(self, counter) + 1)
            raise
        if kind == 'base':
            self._base_query_failures = 0
        return result
