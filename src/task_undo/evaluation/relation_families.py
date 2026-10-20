"""Three disjoint structural probes under the existing sequential v1 semantics.

No policy is used for labels. Each constructor has its own event topology;
compile_history supplies the common text format and forward obligation ledger.
These are author-defined families, not independent human annotations.
"""
from itertools import product

from task_undo.evaluation.compositions import compile_history

OBJECTS = {'a':'cup', 'b':'plate', 'c':'book', 'd':'bowl', 'u':'tray'}
SURFACES = {'s0':'storage surface', 's1':'dining table', 's2':'desk', 's3':'side table'}


def _within_target_revisits(depth, reverse):
    # Target repeatedly rewrites its own objects. Restore the task-start support,
    # not the source of its last move. Subsequent ownership remains selective.
    order = ['b', 'a'] if reverse else ['a', 'b']
    tasks = [[('place','a','s1'), ('place','b','s2')]]
    actions = []
    for i in range(depth):
        for obj in order:
            actions.append(('place',obj,'s3' if i % 2 == 0 else 's0'))
    actions += [('place','c','s1'), ('place','d','s2')]
    tasks += [actions, [('place','u','s1')]]
    return tasks, 1


def _distributed_takeovers(depth, reverse):
    # Each later task takes over one of two target objects. Returning to the
    # target's placement never revives the cancelled task's ownership.
    order = ['d','c'] if reverse else ['c','d']
    tasks = [[('place','a','s1'), ('place','b','s2')],
             [('place','a','s3'), ('place','b','s3'),
              ('place','c','s1'), ('place','d','s1')]]
    for i in range(depth):
        for obj in order:
            tasks.append([('place',obj,'s2' if i % 2 == 0 else 's1')])
    return tasks, 1


def _prior_task_cycles(depth, reverse):
    # Several completed tasks change the restoration boundary itself. Target
    # writes each object once; no later placement of any target object occurs.
    order = ['b','a'] if reverse else ['a','b']
    tasks = []
    for i in range(depth):
        tasks.append([('place',o,'s1' if i % 2 == 0 else 's2') for o in order])
    target = len(tasks)
    tasks.append([('place','a','s3'), ('place','b','s3'),
                  ('place','c','s1'), ('place','d','s2')])
    tasks.append([('claim','c',None), ('claim','d',None)])
    return tasks, target


CONSTRUCTORS = {'within_target_revisits': _within_target_revisits,
                'distributed_takeovers': _distributed_takeovers,
                'prior_task_cycles': _prior_task_cycles}


def identify_family(facts):
    """Check topology without trusting the author's family string."""
    target = facts['target_task_index']; tasks = facts['tasks']
    writes = [e['object_id'] for e in tasks[target]['events'] if e['kind']=='place']
    active = set(writes)
    prior = sum(any(e['kind']=='place' and e['object_id'] in active for e in t['events'])
                for t in tasks[:target])
    later = sum(any(e['kind']=='place' and e['object_id'] in active for e in t['events'])
                for t in tasks[target+1:])
    if len(writes) > len(active) and prior == 1 and later == 0:
        return 'within_target_revisits'
    if len(writes) == len(active) and prior == 1 and later >= 4:
        return 'distributed_takeovers'
    if len(writes) == len(active) and prior >= 2 and later == 0:
        return 'prior_task_cycles'
    raise ValueError('History is outside the three disjoint declared topologies')


def generate_relation_families():
    records, labels, histories, groups = [], {}, {}, []
    for family, construct in CONSTRUCTORS.items():
        for depth, reverse in product((2,4),(False,True)):
            group = f'{family}_d{depth}_r{int(reverse)}'
            members = []
            for branch in ('a','b'):
                tasks, target = construct(depth, reverse)
                tasks.append([('claim',branch,None)])
                public, structured, gold = compile_history(
                    {o:'s0' for o in OBJECTS}, tasks, target, OBJECTS, SURFACES)
                if identify_family(gold['facts']) != family:
                    raise ValueError('Constructor violates its structural family')
                name = group+'_'+branch
                records.append({'id':name,'input':public})
                labels[name], histories[name] = gold, structured
                members.append(name)
            groups.append({'id':group,'family':family,'depth':depth,'reverse':reverse,'members':members})
    return records, labels, histories, groups
