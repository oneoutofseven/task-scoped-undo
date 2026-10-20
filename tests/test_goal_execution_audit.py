"""Adversarial artifact checks; synthetic traces are never physical evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from task_undo.common.storage import write_json
from task_undo.domain.undo_contract import UndoContract, EXECUTOR
from task_undo.evaluation.goal_execution import audit_goal_execution
from test_guarded_goal import observation


class GoalExecutionAuditTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name); self.state = observation()
        self.goal = {'restore':{}, 'protect':list(self.state['objects'])}
        self.contract = UndoContract(self.goal,self.state,['storage','table','desk'],'A')
        status = {'executor':EXECUTOR,'actions':0,'completed':True,'stop_reason':'predicted_goal_verified','settle_steps':30}
        for name, data in {
            'initial_state':self.state,'case':{'surfaces':self.contract.surfaces,'undo_request':{'target_task_id':'A'}},
            'undo_contract':self.contract.record(),
            'execution_plan':{'executor':EXECUTOR,'goal':self.goal,'additional_model_calls':0,'max_actions':0},
            'manifest':{'options':{'execution_mode':EXECUTOR}},'goal_execution':status,
            'result':{'goal_execution':status,'passed':True,'phase':'finished','final':self.state,'control_steps':30},
        }.items():write_json(self.path/(name+'.json'),data)
        self.trace = [{'control_step':i,'observation':copy.deepcopy(self.state)} for i in range(31)]
        self.rows = [{'kind':'settle_started','control_step':0,'observation':self.state},
                     {'kind':'completed','control_step':30,'observation':self.state}]
        self.save('execution_trace',self.trace); self.save('execution_decisions',self.rows)
        self.save('public_history',[])
    def save(self,name,rows):
        (self.path/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    def test_complete_noop_requires_full_raw_window(self):
        self.assertTrue(audit_goal_execution(self.path,self.goal)['audit_passed'])
        self.save('execution_trace',self.trace[:10]+self.trace[11:])
        with self.assertRaisesRegex(ValueError,'Incomplete settlement'):audit_goal_execution(self.path,self.goal)
    def test_transient_protected_change_is_not_hidden_by_final_state(self):
        self.trace[10]['observation']['objects']['p']['position'][0] += .1
        self.save('execution_trace',self.trace)
        with self.assertRaisesRegex(ValueError,'Protected object'):audit_goal_execution(self.path,self.goal)
    def test_public_goal_mismatch_is_not_replaced_with_labels(self):
        wrong={'restore':{'a':'storage'},'protect':['b','p']}
        with self.assertRaisesRegex(ValueError,'Contract differs'):audit_goal_execution(self.path,wrong)
    def test_failed_action_cannot_be_marked_completed(self):
        self.save('public_history',[{'kind':'action_failed'}])
        with self.assertRaisesRegex(ValueError,'failure was hidden'):audit_goal_execution(self.path,self.goal)


if __name__=='__main__':unittest.main()
