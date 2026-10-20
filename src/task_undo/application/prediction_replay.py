"""Read a frozen table goal by replaying its saved responses offline.

No private labels, goal correction, or additional model calls are permitted.
Model goals remain usable even if the shared rule readout failed to parse facts.
"""
import json
from pathlib import Path
import tempfile

from task_undo.agents.narrative import public_input
from task_undo.agents.narrative_controlled import run_controlled
from task_undo.common.storage import read_json

TABLE_MODEL_MODES = {
    'object_table_direct_v1': 'controlled_table_direct',
    'object_table_model_v1': 'controlled_table_shared',
    'object_table_no_history_v1': 'controlled_table_no_history',
}
TABLE_MODES = {**TABLE_MODEL_MODES, 'object_table_rule_v1': 'controlled_table_shared'}
ARTIFACTS = ('facts.json', 'goal.json', 'rule_goal.json',
             'decision_table.json', 'rule_decision_table.json')


def table_prediction_files(folder, readout):
    goal_file = 'rule_goal.json' if readout == 'object_table_rule_v1' else 'goal.json'
    required = ['status.json', goal_file] + [f'{stage}/{name}.json'
        for stage in ('first', 'second') for name in ('request', 'response')]
    optional = [name for name in ARTIFACTS if name != goal_file] + [
        f'{stage}/response_schema.json' for stage in ('first', 'second')]
    return required + [name for name in optional if (folder / name).is_file()]


def replay_table_goal(folder, readout):
    mode = TABLE_MODES[readout]
    rule_readout = readout == 'object_table_rule_v1'
    stages = ('first', 'second')
    requests = [read_json(folder / s / 'request.json') for s in stages]
    responses = [read_json(folder / s / 'response.json') for s in stages]
    for request, response in zip(requests, responses):
        if (not isinstance(request, list) or len(request) != 2 or request[0].get('role') != 'system'
                or request[1].get('role') != 'user' or response.get('finish_reason') != 'stop'):
            raise ValueError('Require two completed saved calls')
    first, second = [json.loads(request[1]['content']) for request in requests]
    # Preserve public-field insertion order: the saved request contains a JSON
    # string, and direct/no-history replay must reproduce that exact string.
    payload = {k: v for k, v in first.items() if k in
               ('objects', 'surfaces', 'current_supports', 'activity_log', 'undo_request')}
    payload['activity_log'] = [line['text'] for line in first['activity_log']]
    payload['undo_request'] = second['undo_request'] if mode == 'controlled_table_shared' else first['undo_request']
    public = public_input({'input': payload})

    class Replay:
        index = 0

        def complete(self, messages, *, response_schema=None):
            if self.index >= 2 or messages != requests[self.index]:
                raise ValueError('Request differs from frozen method')
            schema = folder / stages[self.index] / 'response_schema.json'
            if ((response_schema is not None) != schema.exists()
                    or (schema.exists() and read_json(schema) != response_schema)):
                raise ValueError('Response schema differs from frozen method')
            response = responses[self.index]
            self.index += 1
            return response

    replay = Replay()
    with tempfile.TemporaryDirectory(prefix='undo_model_readout_') as tmp:
        generated = Path(tmp)
        status = run_controlled(replay, public, generated, mode)
        generated_key = 'rule_goal_generated' if rule_readout else 'goal_generated'
        if (replay.index != 2 or not status[generated_key]
                or status != read_json(folder / 'status.json')):
            raise ValueError('Saved table prediction provenance replay failed')
        for name in ARTIFACTS:
            if ((generated / name).exists() != (folder / name).exists()
                    or ((generated / name).exists() and
                        read_json(generated / name) != read_json(folder / name))):
                raise ValueError(f'Saved artifact differs from replay: {name}')
    return public, read_json(folder / ('rule_goal.json' if rule_readout else 'goal.json'))
