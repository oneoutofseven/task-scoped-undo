"""Prospective v2 evidence contract for the opt-in observed-completion executor."""

from task_undo.application.baselines import fixed_actions
from task_undo.domain.completion import EXECUTOR, SETTLE_STEPS, goal_satisfied


def validate_completion(plan, initial, final, trace, events, control_steps):
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(plan['executor'] == EXECUTOR and plan['additional_model_calls'] == 0,
            'Wrong completion executor')
    actions = fixed_actions(plan['goal'])
    require(plan['actions'] == actions, 'Execution actions differ from predicted goal')
    requests = [e for e in events if e['kind'] == 'action_requested']
    require([{k: e[k] for k in ('action', 'object_id', 'target_surface')} for e in requests] == actions,
            'Execution did not follow the declared action sequence')
    require(goal_satisfied(plan['goal'], initial, final), 'Predicted goal not satisfied at completion')
    failures = [e for e in events if e['kind'] == 'action_failed']
    verifications = [e for e in events if e['kind'] == 'task_completion_verified']
    require(len(failures) == len(verifications) <= 1, 'Unresolved or repeated action failures')
    if not failures:
        return 0
    failure, verified = failures[0], verifications[0]
    request = requests[-1]
    require(request['action'] == 'PLACE_ON_TOP'
            and failure['request_event_id'] == request['event_id']
            and failure['task_id'] == request['task_id']
            and isinstance(failure['error'], str) and bool(failure['error']), 'Invalid final placement failure')
    require(verified['failure_event_id'] == failure['event_id']
            and verified['event_id'] == failure['event_id'] + 1
            and verified == events[-2], 'Failure verification must immediately precede completion')
    start = verified['start_control_step']
    require(type(start) is int and failure['control_step'] == start
            and verified['settle_steps'] == SETTLE_STEPS
            and verified['control_step'] == control_steps == start + SETTLE_STEPS,
            'Invalid observation window')
    starting = [row for row in trace if row['control_step'] == start]
    window = [row for row in trace if start < row['control_step'] <= control_steps]
    require(bool(starting) and starting[-1]['observation'] == failure['observation'],
            'Failure state lacks raw evidence')
    window = [starting[-1], *window]
    require({row['control_step'] for row in window} == set(range(start, control_steps + 1)),
            'Incomplete observation window')
    require(all(goal_satisfied(plan['goal'], initial, row['observation']) for row in window),
            'Predicted goal unstable during observation window')
    require(verified['observation'] == final, 'Verification final state mismatch')
    return 1
