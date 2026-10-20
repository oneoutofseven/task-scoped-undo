"""Conservative timing of a joint-space polyline, with a stop at each vertex.

This is not a time-optimal solver. A quintic phase keeps each segment on its
original straight line, with zero velocity and acceleration at both ends.
Analytic derivative bounds constrain the nominal reference; tracking and
collision validation remain separate requirements before physical execution.
"""
import math


def retime_polyline(positions, velocity, acceleration, jerk, *, dt, max_steps=3000,
                    max_increment=.01):
    """Return sampled reference positions and independently checkable bounds."""
    limits = (velocity, acceleration, jerk)
    dof = len(velocity)
    if (not positions or not dof or any(len(v) != dof for v in limits)
            or any(len(row) != dof for row in positions)
            or not math.isfinite(dt) or dt <= 0 or type(max_steps) is not int or max_steps < 1
            or not math.isfinite(max_increment) or max_increment <= 0
            or any(not math.isfinite(x) for row in positions for x in row)
            or any(not math.isfinite(x) or x <= 0 for v in limits for x in v)):
        raise ValueError('Finite path, positive limits, time step and budget required')
    result, segments = [list(positions[0])], []
    for start, end in zip(positions, positions[1:]):
        delta = [b-a for a,b in zip(start,end)]
        if not any(delta):
            continue
        # Exact maxima of s', |s''| and |s'''| for s=10u^3-15u^4+6u^5.
        duration = max(max(1.875*abs(d)/v, math.sqrt((10/math.sqrt(3))*abs(d)/a),
                           (60*abs(d)/j)**(1/3), 1.875*abs(d)*dt/max_increment)
                       for d,v,a,j in zip(delta,*limits))
        steps = max(1, math.ceil(duration/dt))
        if len(result)+steps > max_steps:
            raise ValueError('Retimed route exceeds reference-step budget')
        duration = steps*dt
        ratios = [max(coefficient*abs(d)/duration**power/bound
                      for d,bound in zip(delta,bounds))
                  for coefficient,power,bounds in
                  [(1.875,1,velocity),(10/math.sqrt(3),2,acceleration),(60,3,jerk)]]
        if any(r > 1+1e-10 for r in ratios):
            raise ValueError('Nominal derivative limit exceeded')
        first = len(result)-1
        for step in range(1, steps+1):
            u = step/steps
            phase = u**3*(10+u*(-15+6*u))
            result.append([a+phase*d for a,d in zip(start,delta)])
        result[-1] = list(end)
        segments.append({'first':first,'last':len(result)-1,'duration_s':duration,
                         'peak_limit_ratios':ratios})
    return {'positions':result,'dt':dt,'duration_s':(len(result)-1)*dt,'segments':segments,
            'max_joint_increment':max_increment}
