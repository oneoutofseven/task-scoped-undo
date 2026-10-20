"""Read a singleton batch result without changing its validated trajectory."""
from contextlib import contextmanager


@contextmanager
def singleton_batch_paths(generator):
    original = generator.plan_batch
    owned, previous = 'plan_batch' in vars(generator), vars(generator).get('plan_batch')

    def plan(start_state, goal_pose, config, *args, **kwargs):
        result = original(start_state, goal_pose, config, *args, **kwargs)
        trajectory = getattr(result, 'interpolated_plan', None)
        if (goal_pose.batch == 1 and start_state.position.shape[0] == 1 and
                trajectory is not None and len(trajectory.position.shape) == 2):
            # Use the pinned solver's endpoint-aware accessor; retain the result,
            # all trajectory derivatives, success status and planning metadata.
            result.get_paths = lambda: [result.get_interpolated_plan()]
        return result

    generator.plan_batch = plan
    try:
        yield
    finally:
        if owned:
            generator.plan_batch = previous
        else:
            delattr(generator, 'plan_batch')
