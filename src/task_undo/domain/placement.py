"""Initial placement candidates from measured support geometry, without physics.

Packing near an approach edge is a data-generation heuristic. Only subsequent
normal-action history and reference-undo witnesses establish feasibility.
"""
import math
import random


def approach_band(support, objects, observer, *, seed=0, margin=.05, gap=.10):
    """Place AABB centers in one separated band at the edge nearest the robot.

Inputs are world-frame AABBs. Output coordinates locate AABB centers in XY;
callers must compensate the object's root-to-center offset. Z is unchanged.
The edge tangent follows a right-handed convention, including after 90° turns.
"""
    if not objects or margin < 0 or gap < 0:
        raise ValueError('Nonempty objects and nonnegative clearances required')
    for box in [support, *objects.values()]:
        if (len(box) != 2 or any(len(v) != 3 for v in box)
                or not all(math.isfinite(v) for corner in box for v in corner)
                or any(box[0][i] >= box[1][i] for i in range(3))):
            raise ValueError('Finite, positive AABB dimensions required')
    if len(observer) < 2 or not all(math.isfinite(v) for v in observer[:2]):
        raise ValueError('Finite observer XY required')
    lo, hi = support
    edges = []
    for axis in (0, 1):
        other = 1-axis
        for sign in (-1, 1):
            edge = lo[axis] if sign < 0 else hi[axis]
            if sign*(observer[axis]-edge) <= 0:
                continue
            nearest = max(lo[other], min(hi[other], observer[other]))
            distance = math.hypot(observer[axis]-edge, observer[other]-nearest)
            edges.append((distance, axis, sign))
    if not edges:
        raise ValueError('Observer must stand outside the support footprint')
    _, axis, sign = min(edges)
    tangent_axis = 1-axis
    tangent_sign = sign if axis == 0 else -sign
    tangent_low, tangent_high = sorted([tangent_sign*lo[tangent_axis], tangent_sign*hi[tangent_axis]])
    order = sorted(objects)
    random.Random(seed).shuffle(order)
    widths = {n:objects[n][1][tangent_axis]-objects[n][0][tangent_axis] for n in order}
    length = sum(widths.values()) + gap*(len(order)-1)
    if length + 2*margin > tangent_high-tangent_low:
        raise ValueError('Objects do not fit the approach band with requested clearance')
    cursor = (tangent_low+tangent_high-length)/2
    edge_projection = max(sign*lo[axis], sign*hi[axis])
    centers = {}
    for name in order:
        depth = objects[name][1][axis]-objects[name][0][axis]
        if depth + 2*margin > hi[axis]-lo[axis]:
            raise ValueError('Object depth exceeds usable support')
        point = [0., 0.]
        point[axis] = sign*(edge_projection-margin-depth/2)
        point[tangent_axis] = tangent_sign*(cursor+widths[name]/2)
        centers[name] = point
        cursor += widths[name]+gap
    return {'centers_xy':centers, 'outward_axis':axis, 'outward_sign':sign,
            'order':order, 'margin_m':margin, 'gap_m':gap, 'seed':seed,
            'feasibility_proven':False}
