"""Descriptive repeated-pair results; failed trials remain in the denominator."""
from collections import defaultdict


def summarize(trials, rows):
    planned = {trial['id']: trial for trial in trials}
    observed = {row['id']: row for row in rows}
    if (not trials or len(planned) != len(trials) or len(observed) != len(rows)
            or set(planned) != set(observed)):
        raise ValueError('Every predeclared trial must appear exactly once')
    pairs, groups = defaultdict(list), defaultdict(list)
    conditions, seeds = defaultdict(set), defaultdict(set)
    for trial in trials:
        row = observed[trial['id']]
        for key in ('group', 'condition', 'seed'):
            if row.get(key) != trial[key]:
                raise ValueError('Trial grouping differs from the frozen design')
        score = row.get('score')
        if score is not None and not isinstance(score, dict):
            raise ValueError('Expected a score mapping or missing evidence')
        if score and score.get('joint_success') and score.get('valid') is not True:
            raise ValueError('Invalid evidence cannot count as success')
        group = trial['group']
        pairs[group, trial['seed']].append(row)
        groups[group].append(row)
        conditions[group].add(trial['condition'])
        seeds[group].add(trial['seed'])
    if any(len(values) != 2 for values in conditions.values()):
        raise ValueError('Each history contrast requires exactly two conditions')
    if len({tuple(sorted(values)) for values in seeds.values()}) != 1:
        raise ValueError('All source groups must use the same predeclared seeds')

    def successful(row):
        return bool(row.get('score') and row['score'].get('valid') is True
                    and row['score'].get('joint_success') is True)

    def counts(members):
        scored = [r['score'] for r in members if r.get('score') and r['score'].get('valid') is True]
        return {'trials': len(members), 'scorable': len(scored),
                'unscorable_failures': len(members) - len(scored),
                'joint_successes': sum(successful(row) for row in members),
                'joint_success_rate': sum(successful(row) for row in members) / len(members),
                'undo_completion_mean_scored': sum(s['undo_completion_rate'] for s in scored) / len(scored) if scored else None,
                'protection_violation_mean_scored': sum(s['protection_violation_rate'] for s in scored) / len(scored) if scored else None}

    paired = []
    for (group, seed), members in sorted(pairs.items()):
        if len(members) != 2 or {r['condition'] for r in members} != conditions[group]:
            raise ValueError('Missing or duplicate condition within a paired seed')
        paired.append({'group': group, 'seed': seed, 'trials': [r['id'] for r in members],
                       'joint_success': all(successful(row) for row in members)})
    return {'overall': counts(rows),
            'groups': {name: counts(members) for name, members in sorted(groups.items())},
            'pairs': paired, 'paired_successes': sum(p['joint_success'] for p in paired),
            'paired_trials': len(paired), 'physical_ancestor_groups': len(groups),
            'statistics_scope': 'Conditional execution repeats on development ancestors; seeds and history branches are not independent physical ancestors.'}
