"""Case-level scene, tracked-object and task configuration, without simulator imports."""
import copy
import math

LEGACY_OBJECTS = ['plate', 'cup', 'book']
LEGACY_SCENE = {'scene_model':'Rs_int',
                'load_object_categories':['floors','ceilings','walls','coffee_table'],
                'camera_position':[1.8294,-3.2502,1.6885],
                'camera_orientation':[0.5770,0.1719,0.2280,0.7652]}


def object_ids(case):
    names = case.get('task_object_ids', LEGACY_OBJECTS)
    if (not isinstance(names,list) or not names or not all(isinstance(n,str) and n for n in names)
            or len(names)!=len(set(names))):
        raise ValueError('task_object_ids must be unique nonempty names')
    configured = [obj['name'] for obj in case['objects']]
    if len(configured)!=len(set(configured)):
        raise ValueError('Duplicate configured object name')
    surfaces = set(case['surfaces'].values())
    if not set(names)<=set(configured) or set(names)&surfaces:
        raise ValueError('Tracked objects must exist and cannot be support furniture')
    if set(configured)-surfaces != set(names):
        raise ValueError('All configured movable objects, including untouched objects, must be tracked')
    return list(names)


def scene_settings(case):
    supplied = case.get('scene', {})
    if not isinstance(supplied,dict) or set(supplied)-(set(LEGACY_SCENE) | {'robot_position', 'robot_orientation'}):
        raise ValueError('Unknown scene configuration field')
    scene = copy.deepcopy(LEGACY_SCENE | supplied)
    if not isinstance(scene['scene_model'],str) or not scene['scene_model'].strip():
        raise ValueError('Scene model must be a nonempty string')
    categories = scene['load_object_categories']
    if not isinstance(categories,list) or not categories or not all(isinstance(x,str) and x for x in categories):
        raise ValueError('Scene categories must be nonempty strings')
    if ('robot_position' in scene) != ('robot_orientation' in scene):
        raise ValueError('Robot position and orientation must be supplied together')
    poses = [('camera_position',3),('camera_orientation',4)]
    if 'robot_position' in scene:
        poses += [('robot_position',3),('robot_orientation',4)]
    for key,length in poses:
        values=scene[key]
        if (not isinstance(values,list) or len(values)!=length
                or not all(type(x) in (int,float) and math.isfinite(x) for x in values)):
            raise ValueError(f'Invalid {key}')
        if length == 4 and not any(values):
            raise ValueError(f'Invalid {key} quaternion')
    # The pinned holonomic pose sampler subtracts root translation but does not
    # rotate candidate base joints into the root frame. Until that path supports
    # rotated roots, start with identity and let normal base actions turn.
    if 'robot_orientation' in scene:
        x,y,z,w = scene['robot_orientation']
        if any(abs(v) > 1e-8 for v in (x,y,z)) or abs(abs(w)-1) > 1e-8:
            raise ValueError('Pinned holonomic planner requires identity robot root orientation')
    return scene


def task_map(case):
    """Resolve the cancellation against unique declared tasks before execution."""
    tasks = case.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Case requires a nonempty task list")
    by_id = {}
    for task in tasks:
        name = task.get("task_id") if isinstance(task, dict) else None
        if not isinstance(name, str) or not name.strip() or name in by_id:
            raise ValueError("Task IDs must be nonempty and unique")
        by_id[name] = task
    request = case.get("undo_request")
    target = request.get("target_task_id") if isinstance(request, dict) else None
    if not isinstance(target, str) or target not in by_id:
        raise ValueError("Unknown undo target task")
    return by_id


def claim_timing(task):
    """Existing cases declare before moves; new cases may declare after them."""
    timing = task.get("use_claim_timing", "before_moves")
    if timing not in ("before_moves", "after_moves"):
        raise ValueError("Unknown use declaration timing")
    if "use_claim_timing" in task and "use_claim" not in task:
        raise ValueError("Declaration timing requires use_claim")
    return timing


def validate_layout(case):
    names=set(object_ids(case));scene_settings(case)
    for task in task_map(case).values():
        claim_timing(task)
        for name,support in task['moves']:
            if name not in names or support not in case['surfaces']:
                raise ValueError('Task move references an untracked object or unknown support')
        if 'use_claim' in task and task['use_claim']['object_id'] not in names:
            raise ValueError('Task claim references an untracked object')
