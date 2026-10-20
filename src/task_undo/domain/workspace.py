"""Static footprint envelopes for constructing new episodes, never undo states."""
import math


def collision_aabb(entry, metadata):
    """Transform the rigid root collision box, including offset and orientation."""
    links=metadata['link_bounding_boxes']
    if set(links)!={'base_link'}:raise ValueError('Rigid single-link geometry required')
    box=links['base_link']['collision']['axis_aligned']
    transform=box['transform'];extent=box['extent'];scale=entry.get('scale',[1.,1.,1.])
    pos=entry['position'];q=entry['orientation']
    if any(abs(transform[i][j]-(1. if i==j else 0.))>1e-7 for i in range(3) for j in range(3)):
        raise ValueError('Root-aligned metadata box required')
    if (any(not math.isfinite(v) for v in [*extent,*scale,*pos,*q]) or
            any(v<=0 for v in [*extent,*scale])):raise ValueError('Finite positive box required')
    norm=math.sqrt(sum(v*v for v in q))
    if norm<1e-8:raise ValueError('Invalid quaternion')
    x,y,z,w=[v/norm for v in q]
    rotation=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
              [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
              [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    center=[pos[i]+sum(rotation[i][j]*scale[j]*transform[j][3] for j in range(3)) for i in range(3)]
    radius=[sum(abs(rotation[i][j])*scale[j]*extent[j]/2 for j in range(3)) for i in range(3)]
    return [[center[i]-radius[i] for i in range(3)],[center[i]+radius[i] for i in range(3)]]


def workspace_envelope(case, metadata, *, approach_margin=.6, robot_radius=.5):
    """Union of furniture approach bands and the initial robot footprint.

    This is a layout heuristic. Full robot/payload path validation remains in
    the executor, so map fit is never reported as physical feasibility.
    """
    if approach_margin<=0 or robot_radius<=0:raise ValueError('Positive margins required')
    boxes=[collision_aabb(obj,metadata[obj['name']]) for obj in case['objects']]
    robot=case['scene']['robot_position']
    lower=[min(min(b[0][i]-approach_margin for b in boxes),robot[i]-robot_radius) for i in range(2)]
    upper=[max(max(b[1][i]+approach_margin for b in boxes),robot[i]+robot_radius) for i in range(2)]
    return {'lower_xy':lower,'upper_xy':upper,'center_xy':[(a+b)/2 for a,b in zip(lower,upper)],
            'size_xy':[b-a for a,b in zip(lower,upper)],'approach_margin_m':approach_margin,
            'robot_radius_m':robot_radius,'feasibility_proven':False}
