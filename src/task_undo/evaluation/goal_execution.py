"""Audit observed-goal execution against original requests and raw observations."""
from task_undo.common.storage import read_json
from task_undo.domain.undo_contract import EXECUTOR, SETTLE_STEPS, UndoContract
from task_undo.domain.protection import ProtectionMonitor
from task_undo.evaluation.scoring import read_jsonl


def audit_goal_execution(folder, expected_goal):
    """The caller supplies a replayed PUBLIC prediction, never the gold answer."""
    def require(value, message):
        if not value:
            raise ValueError(message)

    initial = read_json(folder / 'initial_state.json')
    case = read_json(folder / 'case.json')
    contract = UndoContract(expected_goal, initial, case['surfaces'], case['undo_request']['target_task_id'])
    require(read_json(folder / 'undo_contract.json') == contract.record(), 'Contract differs from public goal')
    plan = read_json(folder / 'execution_plan.json')
    require(plan['executor'] == EXECUTOR and plan['goal'] == contract.goal
            and plan['additional_model_calls'] == 0 and plan['max_actions'] == contract.max_actions,
            'Wrong declared executor or goal')
    manifest = read_json(folder / 'manifest.json')
    require(manifest['options'].get('execution_mode') == EXECUTOR, 'Executor was not declared before execution')
    result = read_json(folder / 'result.json')
    status = read_json(folder / 'goal_execution.json')
    require(result['goal_execution'] == status and status['executor'] == EXECUTOR, 'Status mismatch')
    records = read_jsonl(folder / 'execution_decisions.jsonl')
    events = read_jsonl(folder / 'public_history.jsonl')
    trace = read_jsonl(folder / 'execution_trace.jsonl')
    require(bool(trace), 'Missing raw execution trace')
    monitor = ProtectionMonitor(initial, contract.protect)
    for row in trace:
        monitor.check(row['observation'], row['control_step'])
    dispatches = [r for r in records if r['kind'] == 'dispatch']
    requests = [e for e in events if e['kind'] == 'action_requested']
    require(len(dispatches) == len(requests) == status['actions'] <= contract.max_actions,
            'Missing or undeclared action request')
    require([r['index'] for r in dispatches] == list(range(1, len(dispatches) + 1)), 'Action order mismatch')
    for row, request in zip(dispatches, requests):
        require(row['action'] == {k: request[k] for k in ('action', 'object_id', 'target_surface')}
                and row['control_step'] == request['control_step']
                and row['observation'] == request['observation'], 'Decision not bound to actual request')
        state = row['observation']; action = row['action']; obj = action['object_id']
        require(contract.next_action(state) == action, 'Decision violates the public contract')
        require(any(t['control_step'] == row['control_step'] and t['observation'] == state for t in trace),
                'Decision lacks raw observation evidence')
        require(obj in contract.goal['restore'], 'Protected or unknown manipulation')
        held = [n for n, value in state['objects'].items() if value['held']]
        if action['action'] == 'GRASP':
            pending = sorted(n for n, target in contract.goal['restore'].items()
                             if state['objects'][n]['on'] != [target])
            require(not held and state['hand_empty'] and pending and obj == pending[0],
                    'Grasp ignored observed completion or occupied hand')
        else:
            require(action['action'] == 'PLACE_ON_TOP' and held == [obj]
                    and action['target_surface'] == contract.goal['restore'][obj], 'Invalid held-object placement')
    verified = [r for r in records if r['kind'] == 'verified']
    successes = [e for e in events if e['kind'] == 'action_succeeded']
    # An executor-level postcondition rejection can follow a native success.
    require(len(verified) <= len(successes), 'Invented successful action')
    for index, (row, event) in enumerate(zip(verified, successes), 1):
        require(row['index'] == index and row['observation'] == event['observation']
                and row['control_step'] == event['control_step'], 'Postcondition observation mismatch')
        require(any(t['control_step'] == row['control_step'] and t['observation'] == row['observation'] for t in trace),
                'Postcondition lacks raw observation evidence')
    if status['completed']:
        require(result['passed'] and result['phase'] == 'finished' and not result.get('error'), 'Failed run marked complete')
        require(monitor.report()['passed'], 'Protected object changed in the raw trace')
        require(len(verified) == len(dispatches) == len(successes)
                and not any(e['kind'] == 'action_failed' for e in events), 'Primitive failure was hidden')
        require(status['settle_steps'] == SETTLE_STEPS and records[-1]['kind'] == 'completed', 'Missing completion window')
        windows = [r for r in records if r['kind'] == 'settle_started']
        require(len(windows) == 1, 'Repeated or missing settlement')
        start, end = windows[0]['control_step'], records[-1]['control_step']
        require(end - start == SETTLE_STEPS and end == result['control_steps'], 'Wrong settlement duration')
        samples = [r for r in trace if start <= r['control_step'] <= end]
        require({r['control_step'] for r in samples} == set(range(start, end + 1)), 'Incomplete settlement trace')
        for sample in samples:
            state = sample['observation']
            require(state['hand_empty'] and not any(o['held'] for o in state['objects'].values())
                    and all(state['objects'][n]['on'] == [s] for n, s in contract.goal['restore'].items()),
                    'Unstable predicted goal')
        require(records[-1]['observation'] == result['final'], 'Completion state mismatch')
    else:
        require(not result['passed'] and (not records or records[-1]['kind'] != 'completed'), 'Unfinished run marked successful')
    return {'audit_passed': True, 'executor': EXECUTOR, 'completed': status['completed'],
            'actions': len(dispatches), 'stop_reason': status['stop_reason'],
            'settle_steps': status['settle_steps'], 'additional_model_calls': 0,
            'semantic_correctness': 'not asserted; score the supplied goal independently'}
