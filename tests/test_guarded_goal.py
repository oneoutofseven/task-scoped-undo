"""Adversarial synthetic sessions; none of these tests constitute physical evidence."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from task_undo.application.goal_execution import execute_goal
from task_undo.application.workflows import language_undo, baseline_undo
from task_undo.common.storage import read_json
from task_undo.domain.undo_contract import ContractError, EXECUTOR, UndoContract


def observation():
    return {'hand_empty': True, 'objects': {
        name: {'position': [i, 0, .5], 'orientation': [0, 0, 0, 1],
               'on': [support], 'held': False}
        for i, (name, support) in enumerate([('a', 'table'), ('b', 'desk'), ('p', 'table')])}}


def goal():
    return {'restore': {'a': 'storage', 'b': 'desk'}, 'protect': ['p']}


class Session:
    def __init__(self, output):
        self.output = Path(output)
        self.state = observation()
        self.result = {'control_steps': 0}
        self.case = {'surfaces': dict.fromkeys(['storage', 'table', 'desk']),
                     'undo_request': {'target_task_id': 'A'}}
        self.monitor = self.controller = None
        self.actions = []
        self.events = []
        self.initializations = 0
        self.on_action = self.on_idle = self.on_initialize = None

    def observe(self): return copy.deepcopy(self.state)
    def event(self, kind, **fields): self.events.append({'kind': kind, **fields})
    def capture(self, name): pass
    def observe_execution(self):
        if self.monitor:
            self.monitor.check_grasp([n for n, s in self.state['objects'].items() if s['held']])
            self.monitor.check(self.observe(), self.result['control_steps'])
            self.monitor.require_valid()
    def start_protection(self, monitor):
        self.monitor = monitor
        self.observe_execution()
    def initialize_controller(self):
        self.initializations += 1
        self.controller = object()
        if self.on_initialize: self.on_initialize(self)
    def idle(self, steps):
        for _ in range(steps):
            self.result['control_steps'] += 1
            if self.on_idle: self.on_idle(self)
            self.observe_execution()
    def action(self, task, primitive, obj, support):
        self.actions.append((primitive, obj, support))
        if self.on_action and self.on_action(self, primitive, obj, support): return
        self.state['objects'][obj].update(held=primitive == 'GRASP', on=[] if primitive == 'GRASP' else [support])
        self.state['hand_empty'] = primitive != 'GRASP'
        self.result['control_steps'] += 1
        self.observe_execution()


class ContractTests(unittest.TestCase):
    def contract(self): return UndoContract(goal(), observation(), ['storage', 'table', 'desk'], 'A')

    def test_skips_satisfied_object_and_prioritizes_held_object(self):
        c = self.contract(); state = observation()
        self.assertEqual(c.next_action(state)['object_id'], 'a')
        state['objects']['b'].update(held=True, on=[]); state['hand_empty'] = False
        self.assertEqual(c.next_action(state), {'action': 'PLACE_ON_TOP', 'object_id': 'b', 'target_surface': 'desk'})

    def test_goal_and_reference_cannot_be_mutated_through_caller(self):
        g, ref = goal(), observation(); c = UndoContract(g, ref, ['storage', 'table', 'desk'], 'A')
        g['restore']['a'] = 'table'; ref['objects']['p']['position'][0] = 99
        c.reference['objects']['p']['position'][0] = 20
        self.assertEqual(c.goal, goal()); self.assertEqual(c.reference, observation())

    def test_incomplete_overlapping_or_unknown_goal_is_rejected(self):
        for bad in [dict(restore={'a': 'storage'}, protect=['p']),
                    dict(restore={'a': 'missing', 'b': 'desk'}, protect=['p']),
                    dict(restore={'a': 'storage', 'b': 'desk'}, protect=['a', 'p'])]:
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                UndoContract(bad, observation(), ['storage', 'table', 'desk'], 'A')

    def test_invalid_observations_fail_closed(self):
        mutations = [lambda s: s['objects'].pop('a'),
                     lambda s: s.update(hand_empty=False),
                     lambda s: s['objects']['a'].update(on=[]),
                     lambda s: s['objects']['a'].update(on=['table', 'desk']),
                     lambda s: s['objects']['a'].update(position=[0, 1]),
                     lambda s: s['objects']['a'].update(orientation=[0, 0, 0, 0]),
                     lambda s: s['objects']['a'].update(position=[float('nan'), 0, 0])]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                state = observation(); mutation(state); self.contract().next_action(state)

    def test_protected_motion_is_not_accepted_as_goal_completion(self):
        state = observation(); state['objects']['p']['position'][0] += .03
        with self.assertRaisesRegex(ContractError, 'Protected state'):
            self.contract().next_action(state)

    def test_native_success_requires_an_observed_postcondition(self):
        c = self.contract(); state = observation()
        with self.assertRaisesRegex(ContractError, 'primitive result'):
            c.check_action_result(c.next_action(state), state)


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.session = Session(self.tmp.name)
    def run_goal(self, g=None, preview=False):
        execute_goal(self.session, g or goal(), observation(), 'test_public_goal', preview=preview)
    def status(self): return read_json(self.session.output / 'goal_execution.json')

    def test_partial_completion_runs_only_remaining_pair(self):
        self.run_goal()
        self.assertEqual(self.session.actions, [('GRASP', 'a', None), ('PLACE_ON_TOP', 'a', 'storage')])
        self.assertEqual(self.status()['settle_steps'], 30)
        self.assertTrue(self.status()['completed'])
        self.assertNotIn('joint_success', self.session.result)

    def test_noop_has_physics_window_without_controller(self):
        self.run_goal({'restore': {}, 'protect': ['a', 'b', 'p']})
        self.assertEqual(self.session.initializations, 0)
        self.assertEqual(self.session.actions, [])
        self.assertEqual(self.session.result['control_steps'], 30)

    def test_controller_setup_is_followed_by_reobservation(self):
        self.session.on_initialize = lambda s: s.state['objects']['a'].update(on=['storage'])
        self.run_goal()
        self.assertEqual(self.session.initializations, 1)
        self.assertEqual(self.session.actions, [])

    def test_primitive_failure_is_terminal_even_if_goal_looks_met(self):
        def fail(s, primitive, obj, support):
            if primitive == 'PLACE_ON_TOP':
                s.state['objects'][obj].update(on=[support], held=False); s.state['hand_empty'] = True
                raise RuntimeError('release still failed')
        self.session.on_action = fail
        with self.assertRaisesRegex(RuntimeError, 'release still failed'): self.run_goal()
        self.assertFalse(self.status()['completed'])
        self.assertEqual(self.status()['stop_reason'], 'physical_action_failed')
        self.assertEqual(self.status()['settle_steps'], 0)

    def test_false_success_is_detected_before_next_action(self):
        self.session.on_action = lambda *args: True
        with self.assertRaises(ContractError): self.run_goal()
        self.assertEqual(self.status()['stop_reason'], 'postcondition_failed')
        self.assertEqual(len(self.session.actions), 1)

    def test_transient_protected_change_latches_after_object_returns(self):
        def violate(s, *_):
            s.state['objects']['p']['position'][0] += .1
            s.monitor.check(s.observe(), 0)
            s.state['objects']['p']['position'][0] -= .1
        self.session.on_action = violate
        with self.assertRaises(RuntimeError): self.run_goal()
        self.assertEqual(self.status()['stop_reason'], 'protection_violation')
        self.assertFalse(self.status()['completed'])

    def test_completion_window_rejects_drift_without_restarting(self):
        self.session.on_idle = lambda s: s.state['objects']['a'].update(on=['table'])
        with self.assertRaises(ContractError): self.run_goal()
        self.assertEqual(self.status()['stop_reason'], 'unstable_goal')
        self.assertEqual(len(self.session.actions), 2)

    def test_unknown_state_after_setup_prevents_dispatch(self):
        self.session.on_initialize = lambda s: s.state['objects'].pop('a')
        with self.assertRaises(ContractError): self.run_goal()
        self.assertEqual(self.status()['stop_reason'], 'invalid_observation')
        self.assertEqual(self.session.actions, [])

    def test_action_budget_prevents_repeated_repair_of_disturbed_targets(self):
        g = {'restore': {'a': 'storage', 'b': 'storage'}, 'protect': ['p']}
        def disturb(s, primitive, obj, target):
            if primitive == 'PLACE_ON_TOP' and obj == 'b': s.state['objects']['a']['on'] = ['table']
        self.session.on_action = disturb
        with self.assertRaises(ContractError): self.run_goal(g)
        self.assertEqual(self.status()['stop_reason'], 'action_budget')
        self.assertEqual(len(self.session.actions), 4)

    def test_preview_validates_without_claiming_execution(self):
        self.run_goal(preview=True)
        self.assertEqual(self.status()['stop_reason'], 'preview_ready')
        self.assertFalse(self.status()['completed'])
        self.assertEqual(self.session.actions, [])

    def test_language_adapter_preserves_supplied_goal_without_semantic_correction(self):
        wrong = {'restore': {'p': 'storage'}, 'protect': ['a', 'b']}
        with patch('task_undo.application.workflows.check_restored_scene', return_value=(observation(), observation())), \
             patch('task_undo.application.predictions.load_prediction', return_value=wrong):
            language_undo(self.session, Path(self.tmp.name), False, EXECUTOR)
        self.assertEqual(read_json(self.session.output / 'undo_contract.json')['goal'], wrong)
        self.assertEqual(self.session.actions[0][1], 'p')
        self.assertTrue(self.session.result['predicted_goal_verified'])
        self.assertNotIn('joint_success', self.session.result)

    def test_rule_adapter_uses_same_executor(self):
        public = {'history': [], 'current_state': observation()}
        with patch('task_undo.application.workflows.check_restored_scene', return_value=(observation(), observation())), \
             patch('task_undo.agents.direct.load_public_input', return_value=public), \
             patch('task_undo.domain.policies.infer_goal', return_value=goal()):
            baseline_undo(self.session, 'claim_aware', Path(self.tmp.name), False, EXECUTOR)
        self.assertEqual(read_json(self.session.output / 'execution_plan.json')['executor'], EXECUTOR)
        self.assertEqual(self.session.events[-1]['kind'], 'rollout_completed')


if __name__ == '__main__': unittest.main()
