"""Construct new scene layouts from geometry, with no controller or labels."""
from copy import deepcopy
import math

from task_undo.domain.floor_coverage import intersect_floor_coverage
from task_undo.domain.workspace import workspace_envelope


def transfer_layout(case, object_metadata, free_map, floor_envelopes, scene_model,
                    *, resolution=.01, approach_margin=.6, robot_radius=.5):
    """Translate a fixed relative layout into an actually supported map region.

    Choose maximum map clearance deterministically. No outcome-based replacement
    or physical feasibility claim; the caller must still run episode preview.
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    if not math.isfinite(resolution) or resolution<=0:
        raise ValueError('Finite positive map resolution required')
    envelope=workspace_envelope(case,object_metadata,approach_margin=approach_margin,
                                robot_radius=robot_radius)
    supported=intersect_floor_coverage(free_map,floor_envelopes,resolution)
    width,height=[math.ceil(size/resolution) for size in envelope['size_xy']]
    if height>supported.shape[0] or width>supported.shape[1]:
        raise ValueError('Workspace exceeds map dimensions')
    integral=np.pad(supported.astype(np.int64).cumsum(0).cumsum(1),((1,0),(1,0)))
    totals=(integral[height:,width:]-integral[:-height,width:]
            -integral[height:,:-width]+integral[:-height,:-width])
    positions=np.argwhere(totals==height*width)
    if not len(positions):raise ValueError('No supported free workspace fits the layout')
    centers=positions+np.array([height/2,width/2]);pixels=centers.astype(int)
    clearance=distance_transform_edt(supported)
    y,x=centers[int(clearance[pixels[:,0],pixels[:,1]].argmax())]
    center=(np.array([x,y])-supported.shape[0]/2)*resolution
    delta=center-np.array(envelope['center_xy']);result=deepcopy(case)
    def translate(pos):return [float(pos[0]+delta[0]),float(pos[1]+delta[1]),pos[2]]
    for obj in result['objects']:obj['position']=translate(obj['position'])
    for key in ('robot_position','camera_position'):
        result['scene'][key]=translate(result['scene'][key])
    result['scene']['scene_model']=scene_model
    report={'workspace':envelope,'translation_xy':delta.tolist(),'world_center_xy':center.tolist(),
            'valid_rectangles':len(positions),'free_pixels':int(free_map.sum()),
            'supported_free_pixels':int(supported.sum()),'resolution_m':resolution,
            'physical_actions':0,'feasibility_proven':False}
    return result,report
