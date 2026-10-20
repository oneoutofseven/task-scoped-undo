"""Declaration timing and branch construction preserve actual movement evidence."""
import copy
import json
from pathlib import Path
import unittest

from task_undo.domain.history import check_events, claim_timing
from task_undo.evaluation.task_endings import declare_at_task_end

FIXTURE = json.loads((Path(__file__).parent / 'fixtures/dinner_history.json').read_text())


def without_last_claim():
    case = copy.deepcopy(FIXTURE['case'])
    claim = case['tasks'][-1].pop('use_claim')
    events = [copy.deepcopy(e) for e in FIXTURE['events'] if e['kind'] != 'use_claimed']
    old_ids = {e['event_id']: i for i, e in enumerate(events, 1)}
    for event in events:
        event['event_id'] = old_ids[event['event_id']]
        if 'request_event_id' in event:
            event['request_event_id'] = old_ids[event['request_event_id']]
    check_events(events, case)
    return case, events, claim


class TaskEndingTests(unittest.TestCase):
    def test_valid_ending_preserves_all_physical_events_and_source(self):
        case, events, claim = without_last_claim()
        original = copy.deepcopy((case, events))
        for target in ('A', 'B'):
            request = case['undo_request'] | {'target_task_id': target}
            new_case, new_events = declare_at_task_end(events, case, claim, request)
            self.assertEqual(new_events[:-3], events[:-2])
            self.assertEqual(new_events[-1]['observation'], events[-1]['observation'])
            self.assertEqual(check_events(new_events, new_case)['successful_primitives'], 8)
            self.assertEqual([e for e in events if e['kind'].startswith('action_')],
                             [e for e in new_events if e['kind'].startswith('action_')])
        self.assertEqual((case, events), original)

    def test_cannot_replace_a_real_claim_or_invent_an_object(self):
        with self.assertRaisesRegex(ValueError, 'existing declaration'):
            declare_at_task_end(FIXTURE['events'], FIXTURE['case'],
                                FIXTURE['case']['tasks'][-1]['use_claim'], FIXTURE['case']['undo_request'])
        case, events, claim = without_last_claim()
        with self.assertRaisesRegex(ValueError, 'Unknown claimed object'):
            declare_at_task_end(events, case, claim | {'object_id': 'missing'}, case['undo_request'])

    def test_after_moves_claim_cannot_be_recorded_before_actions(self):
        case = copy.deepcopy(FIXTURE['case'])
        case['tasks'][-1]['use_claim_timing'] = 'after_moves'
        with self.assertRaisesRegex(ValueError, 'follow all task moves'):
            check_events(FIXTURE['events'], case)

    def test_default_timing_cannot_accept_a_late_claim(self):
        case, events, claim = without_last_claim()
        case, events = declare_at_task_end(events, case, claim, case['undo_request'])
        case['tasks'][-1].pop('use_claim_timing')
        with self.assertRaisesRegex(ValueError, 'precede'):
            check_events(events, case)

    def test_invalid_timing_is_not_silently_accepted(self):
        for task in [{'use_claim_timing': 'after_moves'},
                     {'use_claim': {}, 'use_claim_timing': 'anytime'}]:
            with self.assertRaises(ValueError):
                claim_timing(task)
