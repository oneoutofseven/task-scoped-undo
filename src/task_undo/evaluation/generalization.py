"""Explicit generalization boundaries, independent of policy and simulator code."""
from copy import deepcopy

from task_undo.domain.case_layout import object_ids, scene_settings
from task_undo.evaluation.compositions import structure_signature


def check_physical_axis(case, registry, axis):
    """Model holdout concerns manipulated objects; shared furniture is permitted.

    Compare against ALL registered assets, including failed inspected cases.
    Scene families use the conservative prefix convention of the existing gate.
    """
    if axis not in ('scene', 'manipulated_object_models',
                    'scene_and_manipulated_object_models'):
        raise ValueError('Unknown physical generalization axis')
    scene = scene_settings(case)['scene_model']
    seen_scenes = {s.split('_')[0] for row in registry['records'] for s in row['scenes']}
    seen_assets = {a for row in registry['records'] for a in row['assets']}
    tracked = set(object_ids(case))
    assets = sorted({f"{o['category']}/{o['model']}" for o in case['objects']
                     if o['name'] in tracked})
    if 'scene' in axis and scene.split('_')[0] in seen_scenes:
        raise ValueError('Previously exposed scene family: ' + scene)
    if 'models' in axis and set(assets) & seen_assets:
        raise ValueError('Previously exposed manipulated model')
    return {'axis': axis, 'scene': scene, 'manipulated_models': assets,
            'shared_furniture_allowed': True, 'passed': True}


def target_projection(facts, objects, surfaces):
    """Remove unrelated objects/empty tasks before exact renaming comparison.

    This rejects apparent novelty created only by padding unrelated activities.
    Claims are retained on target-written objects, including prior declarations.
    """
    target = facts['target_task_index']
    active = {e['object_id'] for e in facts['tasks'][target]['events'] if e['kind'] == 'place'}
    if not active:
        raise ValueError('Structural probe requires a nonempty target placement task')
    reduced = deepcopy(facts)
    reduced['initial_supports'] = {o:s for o,s in facts['initial_supports'].items() if o in active}
    reduced['tasks'] = []
    for index, task in enumerate(facts['tasks']):
        events = [e for e in task['events'] if e['object_id'] in active]
        if events:
            if index == target:
                reduced['target_task_index'] = len(reduced['tasks'])
            reduced['tasks'].append({**task, 'events': events})
    used = set(reduced['initial_supports'].values()) | {
        e['target_surface'] for t in reduced['tasks'] for e in t['events'] if e['kind'] == 'place'}
    return structure_signature(reduced, {o:objects[o] for o in active}, {s:surfaces[s] for s in used})
