"""Authored history compositions, separate from simulator evidence and policies.

The forward obligation ledger creates labels without calling a tested policy.
All family members remain one design group; changed names are never new groups.
"""
from itertools import permutations, product
import hashlib
import json


def structure_signature(facts, objects, surfaces):
    """Exact ordered event structure modulo bijective object/support renaming.

    Event positions distinguish active objects. Inactive objects are ordered by
    initial support; ties are interchangeable. Only supports need permutation.
    """
    occurrences = {obj: [] for obj in objects}
    for i, task in enumerate(facts['tasks']):
        for j, event in enumerate(task['events']):
            occurrences[event['object_id']].append((i,j))
    candidates = []
    for order in permutations(surfaces):
        support = {name:i for i,name in enumerate(order)}
        ordered = sorted(objects, key=lambda obj:(occurrences[obj], support[facts['initial_supports'][obj]]))
        object_ids = {name:i for i,name in enumerate(ordered)}
        payload = {'initial':[support[facts['initial_supports'][obj]] for obj in ordered],
            'target':facts['target_task_index'], 'surfaces':len(order),
            'tasks':[[(e['kind'],object_ids[e['object_id']],
                       support[e['target_surface']] if e['target_surface'] is not None else None)
                      for e in task['events']] for task in facts['tasks']]}
        candidates.append(json.dumps(payload,sort_keys=True,separators=(',',':')))
    return hashlib.sha256(min(candidates).encode()).hexdigest()


def compile_history(initial, tasks, target, objects, surfaces):
    """Compile event tuples into public text/structured input and private labels."""
    if not 0 <= target < len(tasks) or set(initial) != set(objects):
        raise ValueError('Invalid initial state or target task')
    state, obligations, boundary = dict(initial), set(), None
    history, facts, log = [], [], ['Initially, '+ '; '.join(f'{objects[o]} is on {surfaces[s]}' for o,s in initial.items())+'.']
    def observation():
        return {'hand_empty':True,'objects':{o:{'on':[s],'held':False} for o,s in state.items()}}
    def event(kind, **fields):
        history.append({'event_id':len(history)+1,'kind':kind,**fields})
    event('episode_started',observation=observation())
    for i, actions in enumerate(tasks):
        name = 'Shared setup' if i == target else f'Activity {i+1}'
        if i == target:
            boundary = dict(state)
        event('task_started',task_id=str(i),observation=observation())
        sentences, extracted = [], []
        if not actions:
            raise ValueError('Do not pad histories with empty tasks')
        for kind, obj, surface in actions:
            if obj not in objects or kind not in ('place','claim'):
                raise ValueError('Unknown object or action')
            if kind == 'place':
                if surface not in surfaces or state[obj] == surface:
                    raise ValueError('Unknown support or vacuous placement')
                sentences.append(f'moved {objects[obj]} from {surfaces[state[obj]]} to {surfaces[surface]}')
                state[obj] = surface
                if i == target:
                    obligations.add(obj)
                event('action_succeeded',task_id=str(i),action='PLACE_ON_TOP',object_id=obj,
                      target_surface=surface,observation=observation())
            else:
                if surface is not None:
                    raise ValueError('Claims have no placement destination')
                sentences.append(f'a family member explicitly reserved {objects[obj]} for use in this activity, leaving it in place')
                event('use_claimed',task_id=str(i),object_id=obj)
            if i > target:
                obligations.discard(obj)
            extracted.append({'kind':kind,'object_id':obj,'target_surface':surface,'line':i+1})
        event('task_completed',task_id=str(i),observation=observation())
        facts.append({'name':name,'events':extracted})
        log.append(f'{name}: '+'; '.join(sentences)+f'. {name} completed.')
    labels = {'restore':{o:boundary[o] for o in sorted(obligations)},
              'protect':sorted(set(objects)-obligations),
              'facts':{'initial_supports':dict(initial),'target_task_index':target,'tasks':facts}}
    public = {'objects':dict(objects),'surfaces':dict(surfaces),'current_supports':dict(state),
              'activity_log':log,'undo_request':'Shared setup is cancelled. Undo Shared setup, including its own use reservations.'}
    structured = {'history':history,'current_state':observation(),
                  'undo_request':{'target_task_id':str(target)}}
    return public, structured, labels


def composition(restore_count, task_count, target, mechanism, branch):
    """Cross residual count, target boundary, history length and later takeover.

    Both branches share every physical placement and the cancellation request.
    They differ in one later use declaration. Extra tasks perform actual moves
    of an unrelated object; they are still authored symbolic events, not physics.
    """
    if (restore_count not in (2,3) or task_count not in (5,9,13)
            or not 1 <= target <= task_count-3 or mechanism not in ('claim','move','move_back')
            or branch not in ('a','b')):
        raise ValueError('Unsupported composition factors')
    names = ('cup','plate','book','bowl','spoon','tray')
    objects = {f'o{i}':names[i] for i in range(restore_count+3)}
    surfaces = {'s0':'storage surface','s1':'dining table','s2':'desk'}
    state = {o:'s0' for o in objects}; initial = dict(state)
    tasks = [[] for _ in range(task_count)]
    fixed = f'o{restore_count+1}'
    unrelated = f'o{restore_count+2}'
    choice = f'o{restore_count}' if branch == 'a' else f'o{restore_count-1}'
    def place(task, obj, surface):
        tasks[task].append(('place',obj,surface));state[obj]=surface
    # Two target objects have prior valid work, with distinct starting supports.
    place(0,'o0','s1')
    place(0,f'o{restore_count-1}','s2')
    for i in range(1,target):
        place(i,unrelated,'s1' if state[unrelated]!='s1' else 's2')
    for obj in list(objects)[:-1]:
        place(target,obj,'s2' if state[obj]!='s2' else 's1')
    original = state[fixed]
    for i in range(target+1,task_count):
        place(i,unrelated,'s1' if state[unrelated]!='s1' else 's2')
        if i == target+1:
            if mechanism == 'claim':
                tasks[i].append(('claim',fixed,None))
            else:
                place(i,fixed,'s0')
        if i == target+2 and mechanism == 'move_back':
            place(i,fixed,original)
        if i == task_count-1:
            tasks[i].append(('claim',choice,None))
    return compile_history(initial,tasks,target,objects,surfaces)


def generate_compositions():
    records, labels, histories, groups = [], {}, {}, []
    for count, length, position, mechanism in product((2,3),(5,9,13),('early','middle'),('claim','move','move_back')):
        target = 1 if position == 'early' else length//2
        group = f'r{count}_t{length}_{position}_{mechanism}'
        members = []
        for branch in ('a','b'):
            name = group+'_'+branch
            public, structured, gold = composition(count,length,target,mechanism,branch)
            records.append({'id':name,'input':public})
            labels[name], histories[name] = gold, structured
            members.append(name)
        groups.append({'id':group,'members':members,'restore_count':count,'task_count':length,
                       'target_index':target,'mechanism':mechanism})
    return records, labels, histories, groups
