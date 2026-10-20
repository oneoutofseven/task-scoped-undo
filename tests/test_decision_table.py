"""The table is the single source of goals, not a semantic answer checker."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from task_undo.agents.decision_table import ACTION_PROMPT, SYSTEM_PROMPT, compile_plan, run_decision_table

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests/fixtures/reading_history.json").read_text())
PUBLIC = {"history": FIXTURE["events"], "current_state": FIXTURE["events"][-1]["observation"],
          "undo_request": FIXTURE["case"]["undo_request"]}


def example():
    return {"kind": "decision_table", "items": [
        {"object_id": name, "event_ids": [1], "reason": "Synthetic contract test",
         "decision": "restore" if name == "book" else "protect",
         "target_surface": "storage_surface" if name == "book" else None}
        for name in PUBLIC["current_state"]["objects"]]}


class FakeModel:
    def __init__(self, replies):
        self.replies, self.requests = iter(replies), []

    def complete(self, messages):
        self.requests.append(copy.deepcopy(messages))
        reply = next(self.replies)
        return {"content": reply if isinstance(reply, str) else json.dumps(reply), "model": "test_only"}


class DecisionTableTests(unittest.TestCase):
    def test_compile_and_empty_restore_partition(self):
        table = example()
        plan = compile_plan(table, PUBLIC)
        self.assertEqual(plan["restore"], {"book": "storage_surface"})
        self.assertEqual(set(plan["protect"]), {"cup", "plate"})
        for row in table["items"]:
            row.update(decision="protect", target_surface=None)
        self.assertEqual(compile_plan(table, PUBLIC)["restore"], {})

    def test_wrong_semantics_and_conflicting_reason_are_not_secretly_repaired(self):
        table = example()
        cup = next(r for r in table["items"] if r["object_id"] == "cup")
        cup.update(decision="restore", target_surface="storage_surface", event_ids=[13],
                   reason="A later valid use claim protects this cup.")
        plan = compile_plan(table, PUBLIC)
        self.assertIn("cup", plan["restore"])
        self.assertNotIn("cup", plan["protect"])

    def test_no_second_goal_or_ambiguous_rows(self):
        mutations = [lambda t: t.update(restore={}),
                     lambda t: t["items"].pop(),
                     lambda t: t["items"].append(copy.deepcopy(t["items"][0])),
                     lambda t: t["items"][0].update(object_id="unknown"),
                     lambda t: t["items"][0].update(decision="maybe"),
                     lambda t: t["items"][0].update(decision="protect", target_surface="desk")]
        for mutate in mutations:
            table = example()
            mutate(table)
            with self.assertRaises(ValueError):
                compile_plan(table, PUBLIC)

    def test_destinations_and_evidence_use_existing_ids(self):
        for fields in ({"event_ids": [9999]}, {"event_ids": [True]}, {"reason": ""},
                       {"decision": "restore", "target_surface": "unknown"}):
            table = example()
            table["items"][0].update(fields)
            with self.assertRaises(ValueError):
                compile_plan(table, PUBLIC)

    def test_raw_table_and_compiled_plan_are_both_recorded(self):
        table = example()
        model = FakeModel(["not JSON", table])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            status = run_decision_table(model, PUBLIC, out, {"max_turns": 2})
            self.assertTrue(status["plan_generated"])
            self.assertEqual(status["actions"], 0)
            self.assertEqual(json.loads(model.requests[0][1]["content"]), PUBLIC)
            self.assertEqual(json.loads((out / "decision_table.json").read_text()), table)
            self.assertEqual(json.loads((out / "agent_plan.json").read_text()), compile_plan(table, PUBLIC))
            feedback = json.loads(model.requests[1][-1]["content"])
            self.assertEqual(set(feedback), {"status", "error"})

    def test_stop_and_budget_exhaustion_do_not_create_a_goal(self):
        for reply, reason in (({"kind": "stop", "reason": "insufficient evidence"}, "agent_stopped"),
                              ({"kind": "decision_table", "items": []}, "turn_budget")):
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                status = run_decision_table(FakeModel([reply]), PUBLIC, out, {"max_turns": 1})
                self.assertEqual(status["stop_reason"], reason)
                self.assertFalse((out / "agent_plan.json").exists())

    def test_cli_physical_mode_reaches_episode_validation(self):
        result = subprocess.run([sys.executable, "-m", "task_undo", "agent", "--planner", "decision_table"],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("agent requires --episode", result.stderr)

    def test_physical_loop_uses_observed_feedback_and_one_shared_budget(self):
        state = copy.deepcopy(PUBLIC["current_state"])
        actions = []

        def execute(action, name, target):
            actions.append((action, name, target))
            state["objects"][name]["held"] = action == "GRASP"
            state["hand_empty"] = action != "GRASP"
            if target:
                state["objects"][name]["on"] = [target]

        grasp = {"kind": "act", "action": "GRASP", "object_id": "book", "target_surface": None}
        place = {"kind": "act", "action": "PLACE_ON_TOP", "object_id": "book",
                 "target_surface": "storage_surface"}
        # Premature finish must receive mechanical feedback, not silently complete.
        model = FakeModel([example(), grasp, {"kind": "finish"}, place, {"kind": "finish"}])
        with tempfile.TemporaryDirectory() as tmp:
            status = run_decision_table(model, PUBLIC, Path(tmp), {"max_turns": 5, "max_actions": 2},
                                        observe=lambda: state, execute=execute, plan_only=False)
            self.assertTrue(status["execution_completed"])
            self.assertEqual((status["model_calls"], status["actions"]), (5, 2))
            self.assertEqual(actions, [("GRASP", "book", None), ("PLACE_ON_TOP", "book", "storage_surface")])
            feedback = json.loads(model.requests[2][-1]["content"])
            self.assertTrue(feedback["observation"]["objects"]["book"]["held"])
            self.assertEqual(json.loads(model.requests[3][-1]["content"])["status"], "invalid_request")
            final_feedback = json.loads(model.requests[-1][-1]["content"])
            self.assertEqual(final_feedback["observation"]["objects"]["book"]["on"], ["storage_surface"])
            self.assertEqual(json.loads((Path(tmp) / "agent_plan.json").read_text()), compile_plan(example(), PUBLIC))

    def test_empty_goal_can_finish_without_actions(self):
        table = example()
        for row in table["items"]:
            row.update(decision="protect", target_surface=None)
        with tempfile.TemporaryDirectory() as tmp:
            status = run_decision_table(FakeModel([table, {"kind": "finish"}]), PUBLIC, Path(tmp),
                                        {"max_turns": 2, "max_actions": 8},
                                        observe=lambda: PUBLIC["current_state"],
                                        execute=lambda *args: self.fail("No action expected"), plan_only=False)
            self.assertTrue(status["execution_completed"])
            self.assertEqual(status["actions"], 0)

    def test_budgets_include_planning_and_prevent_extra_dispatch(self):
        grasp = {"kind": "act", "action": "GRASP", "object_id": "book", "target_surface": None}
        for config, reason in (({"max_turns": 1, "max_actions": 8}, "turn_budget"),
                               ({"max_turns": 2, "max_actions": 0}, "action_budget")):
            with tempfile.TemporaryDirectory() as tmp:
                status = run_decision_table(FakeModel([example(), grasp]), PUBLIC, Path(tmp), config,
                                            observe=lambda: PUBLIC["current_state"],
                                            execute=lambda *args: self.fail("Budget must prevent action"),
                                            plan_only=False)
                self.assertFalse(status["execution_completed"])
                self.assertEqual(status["stop_reason"], reason)
                self.assertEqual(status["actions"], 0)

    def test_protection_is_scored_offline_and_physical_failure_is_preserved(self):
        # No semantic guard: an erroneous attempt on a protected cup reaches execution.
        grasp = {"kind": "act", "action": "GRASP", "object_id": "cup", "target_surface": None}
        dispatched = []

        def fail(action, name, target):
            dispatched.append(name)
            raise RuntimeError("synthetic actuator failure")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "synthetic actuator failure"):
                run_decision_table(FakeModel([example(), grasp]), PUBLIC, out,
                                   {"max_turns": 4, "max_actions": 8},
                                   observe=lambda: PUBLIC["current_state"], execute=fail, plan_only=False)
            status = json.loads((out / "agent_status.json").read_text())
            self.assertEqual(dispatched, ["cup"])
            self.assertEqual(status["stop_reason"], "physical_action_failed")
            self.assertFalse(status["execution_completed"])
            self.assertEqual(status["actions"], 1)

    def test_phase_switch_replaces_first_system_message_without_losing_history(self):
        model = FakeModel([example(), {"kind": "finish"}])
        with tempfile.TemporaryDirectory() as tmp:
            run_decision_table(model, PUBLIC, Path(tmp), {"max_turns": 2, "max_actions": 8},
                               observe=lambda: PUBLIC["current_state"], execute=lambda *args: None,
                               plan_only=False)
        planning, execution = model.requests
        self.assertEqual(planning[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(execution[0], {"role": "system", "content": ACTION_PROMPT})
        self.assertEqual(sum(m["role"] == "system" for m in execution), 1)
        self.assertEqual(execution[1], planning[1])
        self.assertEqual(json.loads(execution[2]["content"]), example())
        self.assertNotIn("do not request physical actions", execution[0]["content"])
        self.assertIn('"kind":"act"', execution[0]["content"])

    def test_guard_rejects_protected_grasp_then_allows_finish(self):
        table = example()
        for row in table['items']:
            row.update(decision='protect', target_surface=None)
        grasp = {'kind': 'act', 'action': 'GRASP', 'object_id': 'plate', 'target_surface': None}
        model = FakeModel([table, grasp, {'kind': 'finish'}])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            status = run_decision_table(model, PUBLIC, out, {'max_turns': 3, 'max_actions': 1},
                                       observe=lambda: PUBLIC['current_state'],
                                       execute=lambda *args: self.fail('Rejected action reached robot'),
                                       plan_only=False, plan_guard=True)
            self.assertTrue(status['execution_completed'])
            self.assertEqual((status['model_calls'], status['actions'], status['rejected_actions']), (3, 0, 1))
            feedback = json.loads(model.requests[2][-1]['content'])
            self.assertEqual(set(feedback), {'status', 'error'})
            self.assertEqual(feedback['status'], 'plan_violation')
            self.assertIn('protects plate', feedback['error'])
            trace = list(map(json.loads, (out / 'agent_trace.jsonl').read_text().splitlines()))
            self.assertEqual([r['command'] for r in trace if r['kind'] == 'action_rejected'], [grasp])
            self.assertFalse(any(r['kind'] == 'action_dispatched' for r in trace))
            self.assertEqual(json.loads((out / 'agent_plan.json').read_text()), compile_plan(table, PUBLIC))

    def test_guard_rejects_wrong_destination_without_losing_held_object(self):
        state = copy.deepcopy(PUBLIC['current_state'])
        actions = []

        def execute(action, name, target):
            actions.append((action, name, target))
            state['objects'][name]['held'] = action == 'GRASP'
            state['hand_empty'] = action != 'GRASP'
            if target:
                state['objects'][name]['on'] = [target]

        grasp = {'kind': 'act', 'action': 'GRASP', 'object_id': 'book', 'target_surface': None}
        wrong = {'kind': 'act', 'action': 'PLACE_ON_TOP', 'object_id': 'book', 'target_surface': 'dining_table'}
        correct = {**wrong, 'target_surface': 'storage_surface'}
        model = FakeModel([example(), grasp, wrong, correct, {'kind': 'finish'}])
        with tempfile.TemporaryDirectory() as tmp:
            status = run_decision_table(model, PUBLIC, Path(tmp), {'max_turns': 5, 'max_actions': 2},
                                       observe=lambda: state, execute=execute, plan_only=False, plan_guard=True)
        self.assertTrue(status['execution_completed'])
        self.assertEqual(actions, [('GRASP', 'book', None), ('PLACE_ON_TOP', 'book', 'storage_surface')])
        self.assertEqual(status['rejected_actions'], 1)
        self.assertIn('storage_surface', json.loads(model.requests[3][-1]['content'])['error'])
        self.assertEqual(state['objects']['book']['on'], ['storage_surface'])

    def test_repeated_guard_rejections_exhaust_call_budget_without_dispatch(self):
        grasp = {'kind': 'act', 'action': 'GRASP', 'object_id': 'cup', 'target_surface': None}
        with tempfile.TemporaryDirectory() as tmp:
            status = run_decision_table(FakeModel([example(), grasp, grasp]), PUBLIC, Path(tmp),
                                       {'max_turns': 3, 'max_actions': 1},
                                       observe=lambda: PUBLIC['current_state'],
                                       execute=lambda *args: self.fail('Rejected action reached robot'),
                                       plan_only=False, plan_guard=True)
        self.assertEqual(status['stop_reason'], 'turn_budget')
        self.assertFalse(status['execution_completed'])
        self.assertEqual((status['actions'], status['rejected_actions']), (0, 2))

    def test_guard_does_not_fix_an_incorrect_ownership_decision(self):
        table = example()
        cup = next(row for row in table['items'] if row['object_id'] == 'cup')
        cup.update(decision='restore', target_surface='storage_surface')
        grasp = {'kind': 'act', 'action': 'GRASP', 'object_id': 'cup', 'target_surface': None}
        executed = []
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            status = run_decision_table(FakeModel([table, grasp, {'kind': 'stop'}]), PUBLIC, out,
                                       {'max_turns': 3, 'max_actions': 1}, observe=lambda: PUBLIC['current_state'],
                                       execute=lambda *args: executed.append(args), plan_only=False, plan_guard=True)
            self.assertEqual(json.loads((out / 'agent_plan.json').read_text()), compile_plan(table, PUBLIC))
        self.assertEqual(executed, [('GRASP', 'cup', None)])
        self.assertEqual(status['rejected_actions'], 0)
        self.assertFalse(status['execution_completed'])

    def test_guard_cli_requires_physical_decision_table_mode(self):
        for options in (['--plan-guard'], ['--planner', 'decision_table', '--plan-guard', '--plan-only']):
            result = subprocess.run([sys.executable, '-m', 'task_undo', 'agent', *options],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('--plan-guard requires physical --planner decision_table', result.stderr)
        result = subprocess.run([sys.executable, '-m', 'task_undo', 'agent', '--planner',
                                 'decision_table', '--plan-guard'], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('agent requires --episode', result.stderr)
