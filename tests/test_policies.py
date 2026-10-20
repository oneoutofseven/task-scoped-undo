"""Task selection and restoration destinations are separate decisions."""

import copy
import json
from pathlib import Path
import unittest

from task_undo.domain.policies import infer_goal


FIXTURE = json.loads((Path(__file__).parent / 'fixtures/reading_history.json').read_text())


class TaskBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.public = {'history': copy.deepcopy(FIXTURE['events']),
                       'current_state': copy.deepcopy(FIXTURE['events'][-1]['observation']),
                       'undo_request': {'target_task_id': 'B'}}

    def test_cancel_b_restores_prior_a_placement_and_protects_book(self):
        # B's own use claim is cancelled with B, so it does not protect the cup.
        for policy in ('last_writer', 'claim_aware'):
            with self.subTest(policy=policy):
                goal = infer_goal(policy, self.public)
                self.assertEqual(goal['restore'], {'cup': 'dining_table', 'plate': 'storage_surface'})
                self.assertEqual(goal['protect'], ['book'])

    def test_cancel_a_protects_b_takeover_on_the_same_current_state(self):
        self.public['undo_request']['target_task_id'] = 'A'
        goal = infer_goal('claim_aware', self.public)
        self.assertEqual(goal['restore'], {'book': 'storage_surface'})
        self.assertEqual(goal['protect'], ['cup', 'plate'])

    def test_restore_all_remains_deliberately_episode_initial(self):
        goal = infer_goal('restore_initial', self.public)
        self.assertEqual(goal['restore'], {k: 'storage_surface' for k in ('book', 'cup', 'plate')})
        self.assertEqual(infer_goal('noop', self.public)['restore'], {})

    def test_later_no_motion_claim_protects_cup_when_cancelling_b(self):
        self.public['history'].insert(-1, {'kind': 'use_claimed', 'task_id': 'C',
            'object_id': 'cup', 'event_id': 23})
        goal = infer_goal('claim_aware', self.public)
        self.assertEqual(goal['restore'], {'plate': 'storage_surface'})
        self.assertEqual(goal['protect'], ['book', 'cup'])
        self.assertIn('cup', infer_goal('last_writer', self.public)['restore'])

    def test_multiple_target_placements_do_not_change_the_restoration_boundary(self):
        extra = copy.deepcopy(next(e for e in self.public['history']
            if e['kind'] == 'action_succeeded' and e.get('task_id') == 'B'
            and e.get('action') == 'PLACE_ON_TOP' and e['object_id'] == 'cup'))
        extra.update(event_id=23, target_surface='storage_surface')
        extra['observation']['objects']['cup']['on'] = ['storage_surface']
        self.public['history'].insert(-1, extra)
        self.public['current_state'] = copy.deepcopy(extra['observation'])
        self.assertEqual(infer_goal('claim_aware', self.public)['restore']['cup'], 'dining_table')

    def test_missing_or_duplicate_task_start_does_not_fall_back_to_initial(self):
        start = next(e for e in self.public['history'] if e['kind'] == 'task_started' and e['task_id'] == 'B')
        for count in (0, 2):
            public = copy.deepcopy(self.public)
            public['history'] = [e for e in public['history']
                if not (e['kind'] == 'task_started' and e['task_id'] == 'B')]
            public['history'].extend([copy.deepcopy(start) for _ in range(count)])
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'exactly one target task start'):
                infer_goal('claim_aware', public)

    def test_ambiguous_or_held_boundary_state_is_not_guessed(self):
        for supports, held in (([], False), (['desk', 'dining_table'], False), (['dining_table'], True)):
            public = copy.deepcopy(self.public)
            boundary = next(e for e in public['history'] if e['kind'] == 'task_completed' and e['task_id'] == 'A')
            boundary['observation']['objects']['cup'].update(on=supports, held=held)
            with self.subTest(supports=supports, held=held), self.assertRaisesRegex(ValueError, 'known unheld reference support'):
                infer_goal('claim_aware', public)

    def test_renamed_objects_and_tasks_have_the_same_semantics(self):
        # Rename the whole public record, so no cup / B identifier can be special.
        encoded = json.dumps(self.public)
        for old, new in [('cup', 'object_17'), ('B', 'task_9'), ('A', 'task_4')]:
            encoded = encoded.replace('"' + old + '"', '"' + new + '"')
        public = json.loads(encoded)
        before = copy.deepcopy(public)
        goal = infer_goal('claim_aware', public)
        self.assertEqual(goal['restore'], {'object_17': 'dining_table', 'plate': 'storage_surface'})
        self.assertEqual(public, before)
