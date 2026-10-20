"""Blind review packets and descriptive agreement; never infer gold labels."""

from copy import deepcopy
from datetime import datetime
import hashlib
import json

PUBLIC_FIELDS = {'objects', 'surfaces', 'current_supports', 'activity_log', 'undo_request'}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def check_public(public):
    require(isinstance(public, dict) and set(public) == PUBLIC_FIELDS,
            'input must contain only the five public fields')
    for key in ('objects', 'surfaces'):
        catalog = public[key]
        require(isinstance(catalog, dict) and bool(catalog) and
                all(nonempty(k) and nonempty(v) for k, v in catalog.items()),
                f'invalid {key} catalog')
    supports = public['current_supports']
    require(isinstance(supports, dict) and set(supports) == set(public['objects']),
            'current_supports must cover exactly the objects')
    require(all(isinstance(v, str) and v in public['surfaces'] for v in supports.values()),
            'unknown current surface')
    log = public['activity_log']
    require(isinstance(log, list) and bool(log) and all(nonempty(x) for x in log),
            'activity_log must contain nonempty lines')
    require(nonempty(public['undo_request']), 'missing undo request')


def make_packet(cases, spec_sha256):
    require(isinstance(cases, list) and bool(cases), 'empty cases')
    require(isinstance(spec_sha256, str) and len(spec_sha256) == 64 and
            all(c in '0123456789abcdef' for c in spec_sha256), 'invalid spec hash')
    conditions, mapping = [], {}
    for case in cases:
        require(isinstance(case, dict) and nonempty(case.get('id')), 'missing case id')
        require(case['id'] not in mapping.values(), 'duplicate case id')
        public = case.get('input')
        check_public(public)
        review_id = f'condition_{len(conditions) + 1:02d}'
        mapping[review_id] = case['id']
        conditions.append({'review_id': review_id, 'input': deepcopy(public),
                           'cancelled_activity': None,
                           'objects': {obj: {'decision': None, 'target_surface': None,
                                             'evidence_lines': [], 'reason': None}
                                       for obj in public['objects']}, 'issues': []})
    packet = {'schema_version': 1, 'spec_sha256': spec_sha256, 'status': 'unreviewed',
              'reviewer_id': None, 'reviewed_at': None, 'independent_review': None,
              'conditions': conditions}
    return packet, mapping


def validate_review(packet, review):
    """Check submission completeness and binding, not semantic correctness."""
    require(isinstance(packet, dict) and isinstance(packet.get('conditions'), list),
            'invalid reference packet')
    require(all(isinstance(row, dict) and 'input' in row for row in packet['conditions']),
            'invalid reference conditions')
    expected, _ = make_packet([
        {'id': str(i), 'input': row['input']} for i, row in enumerate(packet['conditions'])
    ], packet.get('spec_sha256'))
    require(packet == expected, 'reference must be the original blank packet')
    require(isinstance(review, dict) and set(review) == set(packet), 'review fields changed')
    for key in ('schema_version', 'spec_sha256'):
        require(review[key] == packet[key], f'{key} changed')
    require(review['status'] == 'submitted', 'status must be submitted')
    require(nonempty(review['reviewer_id']), 'missing reviewer_id')
    require(review['independent_review'] is True, 'independent review attestation required')
    try:
        when = datetime.fromisoformat(review['reviewed_at'])
    except (TypeError, ValueError):
        raise ValueError('reviewed_at must be ISO 8601 with timezone') from None
    require(when.utcoffset() is not None, 'reviewed_at timezone required')
    rows = review['conditions']
    require(isinstance(rows, list) and len(rows) == len(packet['conditions']),
            'conditions missing or added')
    uncertain, issues_count, total = [], 0, 0
    for expected, row in zip(packet['conditions'], rows):
        require(isinstance(row, dict) and set(row) == set(expected), 'condition fields changed')
        prefix = expected['review_id']
        require(row['review_id'] == prefix and row['input'] == expected['input'],
                f'{prefix}: public input or order changed')
        issues = row['issues']
        require(isinstance(issues, list) and all(nonempty(x) for x in issues),
                f'{prefix}: issues must be nonempty strings')
        issues_count += len(issues)
        require(nonempty(row['cancelled_activity']) or
                (row['cancelled_activity'] is None and bool(issues)),
                f'{prefix}: missing cancelled_activity or explanation')
        require(isinstance(row['objects'], dict) and set(row['objects']) == set(expected['objects']),
                f'{prefix}: objects missing or added')
        for obj, answer in row['objects'].items():
            total += 1
            key = f'{prefix}/{obj}'
            require(isinstance(answer, dict) and set(answer) == set(expected['objects'][obj]),
                    f'{key}: answer fields changed')
            decision = answer['decision']
            require(isinstance(decision, str) and decision in ('restore', 'protect', 'uncertain'),
                    f'{key}: invalid decision')
            target = answer['target_surface']
            if decision == 'restore':
                require(isinstance(target, str) and target in row['input']['surfaces'],
                        f'{key}: unknown restore surface')
            else:
                require(target is None, f'{key}: non-restore target must be null')
            evidence = answer['evidence_lines']
            require(isinstance(evidence, list) and all(type(n) is int and
                    1 <= n <= len(row['input']['activity_log']) for n in evidence),
                    f'{key}: evidence lines out of range (one-based)')
            require(len(evidence) == len(set(evidence)), f'{key}: duplicate evidence lines')
            require(nonempty(answer['reason']), f'{key}: missing reason')
            if decision == 'uncertain':
                require(bool(issues), f'{key}: uncertainty requires an issue')
                uncertain.append(key)
            else:
                require(bool(evidence), f'{key}: evidence required')
    return {'valid_submission': True, 'total_objects': total, 'uncertain_objects': uncertain,
            'issue_count': issues_count,
            'ready_for_adjudication': not uncertain and issues_count == 0}


def compare_reviews(packet, left, right):
    """All objects stay in the denominator; uncertain rows never count as agreements."""
    checks = [validate_review(packet, review) for review in (left, right)]
    require(left['reviewer_id'].strip() != right['reviewer_id'].strip(),
            'two distinct reviewer IDs required')
    agreements, decision_agreements, resolved, condition_agreements = 0, 0, 0, 0
    rows, tasks = [], []
    for a, b in zip(left['conditions'], right['conditions']):
        all_agree = True
        task_equal = nonempty(a['cancelled_activity']) and a['cancelled_activity'] == b['cancelled_activity']
        tasks.append({'review_id': a['review_id'], 'left': a['cancelled_activity'],
                      'right': b['cancelled_activity'], 'exact_text_match': task_equal})
        for obj, x in a['objects'].items():
            y = b['objects'][obj]
            known = x['decision'] != 'uncertain' and y['decision'] != 'uncertain'
            same_decision = known and x['decision'] == y['decision']
            same_goal = same_decision and x['target_surface'] == y['target_surface']
            resolved += known
            decision_agreements += same_decision
            agreements += same_goal
            all_agree = all_agree and same_goal
            rows.append({'review_id': a['review_id'], 'object': obj,
                         'left': deepcopy(x), 'right': deepcopy(y),
                         'resolved_pair': known, 'goal_agreement': same_goal,
                         'issues_left': a['issues'], 'issues_right': b['issues'],
                         'adjudication': None})
        condition_agreements += all_agree
    total = len(rows)
    return {'schema_version': 1, 'status': 'awaiting_human_adjudication',
            'packet_sha256': digest(packet), 'review_sha256': [digest(left), digest(right)],
            'reviewer_ids': [left['reviewer_id'], right['reviewer_id']],
            'independence': 'self_attested_not_verified', 'submission_checks': checks,
            'counts': {'conditions': len(tasks), 'objects': total, 'resolved_pairs': resolved,
                       'decision_agreements': decision_agreements, 'goal_agreements': agreements,
                       'all_object_goal_agreement_conditions': condition_agreements},
            'goal_agreement_over_all_objects': agreements / total,
            'goal_agreement_over_resolved_pairs': agreements / resolved if resolved else None,
            'task_name_comparison': tasks, 'objects': rows,
            'limitations': ['Agreement is not label correctness or independent-review verification.',
                           'Conditions may share histories; no population confidence interval.',
                           'Task names are free text; text mismatch requires human interpretation.',
                           'No gold labels are changed and adjudication remains manual.']}
