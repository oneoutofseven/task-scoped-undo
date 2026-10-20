"""Rule assistance is explicit, rejects conflicts, and never repairs a plan silently."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from task_undo.agents.evidence import build_ledger, compile_evidence
from task_undo.agents.decision_table import run_decision_table
from test_agent import FakeModel

FIXTURE = json.loads((Path(__file__).parent/'fixtures/reading_history.json').read_text())


def public(target='A'):
    return {'history': copy.deepcopy(FIXTURE['events']),
            'current_state': copy.deepcopy(FIXTURE['events'][-1]['observation']),
            'undo_request': {'target_task_id': target}}


def table(target='A'):
    # Independently authored expectations, not derived from the production ledger.
    facts = ({'book': ([6], [], 'storage_surface'), 'cup': ([10], [13,17], None),
              'plate': ([], [21], None)} if target == 'A' else
             {'book': ([], [], None), 'cup': ([17], [], 'dining_table'),
              'plate': ([21], [], 'storage_surface')})
    return {'kind': 'evidence_table', 'items': [{'object_id': obj,
        'before_event_id': 1 if target == 'A' else 11,
        'target_change_event_ids': changes, 'later_activity_event_ids': later,
        'reason': 'Synthetic evidence contract test', 'decision': 'restore' if support else 'protect',
        'target_surface': support} for obj,(changes,later,support) in facts.items()]}


class EvidenceTests(unittest.TestCase):
    def test_same_scene_different_cancelled_task(self):
        self.assertEqual(compile_evidence(table(),public())['restore'], {'book':'storage_surface'})
        self.assertEqual(compile_evidence(table('B'),public('B'))['restore'],
                         {'cup':'dining_table','plate':'storage_surface'})
        ledger=build_ledger(public('B'))
        cup=next(r for r in ledger['objects'] if r['object_id']=='cup')
        self.assertEqual(cup['target_task_claim_event_ids'],[13])
        self.assertEqual(cup['later_external_activities'],[])

    def test_wrong_owner_and_omitted_adoption_cannot_pass(self):
        for obj,field,value in [('plate','target_change_event_ids',[21]),
                                 ('cup','later_activity_event_ids',[])]:
            bad=table();next(r for r in bad['items'] if r['object_id']==obj)[field]=value
            with self.subTest(obj=obj),self.assertRaisesRegex(ValueError,'public task/object/chronology'):
                compile_evidence(bad,public())

    def test_own_claim_cannot_protect_and_global_initial_is_wrong_destination(self):
        for mutation in [{'later_activity_event_ids':[13]},
                         {'decision':'protect','target_surface':None},
                         {'target_surface':'storage_surface'},{'before_event_id':1}]:
            bad=table('B');next(r for r in bad['items'] if r['object_id']=='cup').update(mutation)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                compile_evidence(bad,public('B'))

    def test_later_independent_claim_changes_b_boundary(self):
        p=public('B');p['history'][-1]['event_id']=24
        p['history'].insert(-1,{'event_id':23,'kind':'use_claimed','task_id':'C','object_id':'cup'})
        t=table('B');cup=next(r for r in t['items'] if r['object_id']=='cup')
        cup.update(later_activity_event_ids=[23],decision='protect',target_surface=None)
        self.assertEqual(compile_evidence(t,p)['restore'],{'plate':'storage_surface'})

    def test_boolean_duplicate_unknown_and_partial_evidence_rejected(self):
        mutations=[lambda t:t['items'].pop(),lambda t:t['items'].append(t['items'][0]),
            lambda t:t['items'][0].update(before_event_id=True),
            lambda t:t['items'][0].update(target_change_event_ids=[6,6]),
            lambda t:t['items'][0].update(target_change_event_ids=[999]),
            lambda t:t['items'][0].update(reason='')]
        for mutate in mutations:
            t=table();mutate(t)
            with self.assertRaises(ValueError):compile_evidence(t,public())

    def test_rejection_is_logged_and_model_must_resubmit(self):
        bad=table();bad['items'][0].update(decision='protect',target_surface=None)
        model=FakeModel([bad,table()])
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            status=run_decision_table(model,public(),out,{'max_turns':2},evidence_guard=True)
            self.assertTrue(status['plan_generated']);self.assertEqual(status['rejected_tables'],1)
            self.assertEqual(status['goal_assistance'],'public_history_rules')
            self.assertIn('evidence_ledger',json.loads(model.requests[0][1]['content']))
            self.assertEqual(json.loads((out/'evidence_table.json').read_text()),table())

    def test_rejected_table_never_executes_or_saves_goal(self):
        bad=table();bad['items'][0].update(decision='protect',target_surface=None)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            result=run_decision_table(FakeModel([bad]),public(),out,{'max_turns':1,'max_actions':8},
                observe=lambda:public()['current_state'],execute=lambda *a:self.fail('No dispatch'),
                plan_only=False,plan_guard=True,evidence_guard=True)
            self.assertFalse(result['execution_completed']);self.assertFalse((out/'agent_plan.json').exists())

    def test_guarded_finish_requires_observed_targets(self):
        state=public()['current_state'];actions=[]
        def execute(action,name,target):
            actions.append(action);state['hand_empty']=action!='GRASP'
            state['objects'][name].update(held=action=='GRASP',on=[target] if target else [])
        replies=[table(),{'kind':'finish'},
            {'kind':'act','action':'GRASP','object_id':'book','target_surface':None},
            {'kind':'act','action':'PLACE_ON_TOP','object_id':'book','target_surface':'storage_surface'},
            {'kind':'finish'}]
        with tempfile.TemporaryDirectory() as tmp:
            model=FakeModel(replies)
            result=run_decision_table(model,public(),Path(tmp),{'max_turns':5,'max_actions':2},
                observe=lambda:state,execute=execute,plan_only=False,plan_guard=True,evidence_guard=True)
            self.assertTrue(result['execution_completed']);self.assertEqual(actions,['GRASP','PLACE_ON_TOP'])
            self.assertIn('not yet observed',model.requests[2][-1]['content'])
