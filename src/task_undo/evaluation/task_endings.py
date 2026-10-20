"""Author zero-motion declaration endings for a shared physical action prefix.

This is dataset construction, never policy inference. It preserves every source
event except the final completion/request markers, explicitly recording that
the alternative declaration was authored rather than physically perceived.
"""
import copy

from task_undo.domain.history import check_events


def declare_at_task_end(events, case, claim, undo_request):
    check_events(events, case)
    final_task = case['tasks'][-1]
    completion, request = events[-2:]
    if (completion['kind'] != 'task_completed' or request['kind'] != 'undo_requested'
            or completion['task_id'] != final_task['task_id']):
        raise ValueError('Expected final task completion followed by undo request')
    if 'use_claim' in final_task:
        raise ValueError('Cannot overwrite an existing declaration')
    if set(claim) != {'object_id', 'actor', 'purpose', 'utterance'}:
        raise ValueError('Declaration requires object, actor, purpose and utterance')
    if claim['object_id'] not in request['observation']['objects']:
        raise ValueError('Unknown claimed object')
    if undo_request['target_task_id'] not in {task['task_id'] for task in case['tasks']}:
        raise ValueError('Unknown cancellation target')
    if completion['observation'] != request['observation'] or not request['observation']['hand_empty']:
        raise ValueError('Ending requires the same empty-handed physical state')
    result_case = copy.deepcopy(case)
    result_case['tasks'][-1].update(use_claim=copy.deepcopy(claim), use_claim_timing='after_moves')
    result_case['undo_request'] = copy.deepcopy(undo_request)
    result_events = copy.deepcopy(events[:-2])
    result_events.append({key: completion[key] for key in
                          ('event_id', 'episode_id', 'control_step', 'sim_seconds')} | {
        'kind': 'use_claimed', 'task_id': final_task['task_id'],
        'evidence_source': 'scenario_authored_declaration', **copy.deepcopy(claim)})
    result_events.append(copy.deepcopy(completion) | {'event_id': len(result_events) + 1})
    result_events.append({key: request[key] for key in
                          ('episode_id', 'control_step', 'sim_seconds')} | {
        'event_id': len(result_events) + 1, 'kind': 'undo_requested',
        **copy.deepcopy(undo_request), 'observation': copy.deepcopy(request['observation'])})
    check_events(result_events, result_case)
    return result_case, result_events
