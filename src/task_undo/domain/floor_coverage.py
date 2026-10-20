"""Conservative construction filter using the loaded scene's floor envelopes.

Floor AABBs overapproximate nonrectangular meshes. This filter removes known
unsupported map regions; subsequent physical initialization remains mandatory.
"""
from task_undo.domain.workspace import collision_aabb


def floor_boxes(scene, metadata):
    boxes=[]
    for name,record in scene['objects_info']['init_info'].items():
        args=record['args']
        if args.get('category')!='floors':continue
        pose=scene['state']['registry']['object_registry'][name]['root_link']
        box=collision_aabb({**args,'position':pose['pos'],'orientation':pose['ori']},metadata[name])
        # Current episode initializer uses z=0 as the floor datum. Reject
        # elevated floors rather than silently constructing below their surface.
        if abs(box[1][2])>.05:continue
        boxes.append(box)
    if not boxes:raise ValueError('No floor envelope at the declared zero-height datum')
    return boxes


def intersect_floor_coverage(free, boxes, resolution=.01):
    """Intersect map-free cells with floor XY envelopes, using pixel centers."""
    import numpy as np
    if free.ndim!=2 or free.shape[0]!=free.shape[1] or resolution<=0:
        raise ValueError('Square map and positive resolution required')
    size=free.shape[0];coordinate=(np.arange(size)-size/2)*resolution
    covered=np.zeros_like(free,dtype=bool)
    for lower,upper in boxes:
        columns=(coordinate>=lower[0]) & (coordinate<=upper[0])
        rows=(coordinate>=lower[1]) & (coordinate<=upper[1])
        covered |= rows[:,None] & columns[None,:]
    return free & covered
