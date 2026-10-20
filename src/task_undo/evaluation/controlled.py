"""Grouped summaries for controlled development probes; no inference dependencies."""
from task_undo.agents.narrative_controlled import visible_input
from task_undo.evaluation.goals import score_goal

READOUTS = {'direct_2call': ('controlled_direct', 'goal_score'),
            'shared_model': ('controlled_shared', 'goal_score'),
            'shared_rule': ('controlled_shared', 'rule_goal_score'),
            'no_history_2call': ('controlled_no_history', 'goal_score')}


def validate_design(design, cases, labels):
    members = [name for group in design['groups'].values() for name in group]
    if len(members) != len(set(members)) or set(members) != set(cases) or set(cases) != set(labels):
        raise ValueError('Every condition must belong to exactly one design group')
    for pair in design['pairs']:
        left, right = pair['cases']
        if left == right or not {left, right} <= set(design['groups'][pair['group']]):
            raise ValueError('Pair members must share their declared design group')
        a, b = cases[left], cases[right]
        if visible_input(a, 'controlled_no_history') != visible_input(b, 'controlled_no_history'):
            raise ValueError('Pairs must have identical visible observation and request')
        if a['activity_log'] == b['activity_log']:
            raise ValueError('Pair histories must differ')
        same_goal = score_goal(labels[left], labels[right])['goal_correct']
        if pair['kind'] not in ('contrast', 'invariance') or same_goal != (pair['kind'] == 'invariance'):
            raise ValueError('Pair kind contradicts author goals')


def summarize(design, rows, readouts=None):
    readouts = READOUTS if readouts is None else readouts
    indexed = {(r['case'], r['mode']): r for r in rows}
    members = [name for group in design['groups'].values() for name in group]
    if len(indexed) != len(rows) or set(indexed) != {
            (name, mode) for name in members for mode in {v[0] for v in readouts.values()}}:
        raise ValueError('Missing or duplicate planned inference arms')
    report = {'readouts': {}, 'actual_model_calls': sum(r['status']['model_calls'] for r in rows),
              'cost_by_arm': {}, 'shared_readouts_are_independent_runs': False}
    for name, (mode, metric) in readouts.items():
        correct = {case: indexed[case, mode][metric]['goal_correct'] for case in members}
        groups = {g: {'correct': sum(correct[c] for c in cases), 'total': len(cases),
                      'accuracy': sum(correct[c] for c in cases) / len(cases)}
                  for g, cases in design['groups'].items()}
        pairs = [dict(pair, both_correct=all(correct[c] for c in pair['cases'])) for pair in design['pairs']]
        report['readouts'][name] = {'correct': sum(correct.values()), 'total': len(correct), 'groups': groups,
            'group_macro_accuracy': sum(g['accuracy'] for g in groups.values()) / len(groups),
            'pairs': pairs, 'contrast_both_correct': sum(p['both_correct'] for p in pairs if p['kind'] == 'contrast'),
            'contrast_total': sum(p['kind'] == 'contrast' for p in pairs),
            'invariance_both_correct': sum(p['both_correct'] for p in pairs if p['kind'] == 'invariance'),
            'invariance_total': sum(p['kind'] == 'invariance' for p in pairs)}
    for mode in sorted({v[0] for v in readouts.values()}):
        arm = [r for r in rows if r['mode'] == mode]
        report['cost_by_arm'][mode] = {'model_calls': sum(r['status']['model_calls'] for r in arm),
            'wall_seconds': round(sum(r['wall_seconds'] for r in arm), 3),
            **{key: sum(r['usage'][key] for r in arm)
               for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}}
    paired = [r for r in rows if r['mode'] == readouts['shared_model'][0]]
    report['shared_computation'] = {
        key: sum((r['goal_score']['goal_correct'], r['rule_goal_score']['goal_correct']) == values for r in paired)
        for key, values in [('both_correct', (True, True)), ('model_only', (True, False)),
                            ('rule_only', (False, True)), ('both_wrong', (False, False))]}
    report['shared_facts_exact'] = sum(r['fact_score']['facts_exact'] for r in paired)
    report['shared_citations_exact'] = sum(r['fact_score']['cited_event_sequence_correct'] for r in paired)
    report['case_results'] = [{
        'case': c, **{name: indexed[c, mode][metric]['goal_correct'] for name, (mode, metric) in readouts.items()}}
        for c in members]
    return report
