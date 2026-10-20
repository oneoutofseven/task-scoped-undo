"""Stable shortest rotation vectors for small-angle feedback and FK derivatives."""


def rotation_vector(quaternion_xyzw):
    """Quaternion logarithm using atan2, including float32 angles near zero.

    For a 1e-4 rad rotation, float32 can round cos(angle/2) to exactly one.
    sqrt(1-w*w)/acos(w) then loses the rotation even though its vector part
    remains nonzero. atan2(norm(vector), w) preserves that information.
    """
    import torch as th
    norm = th.linalg.vector_norm(quaternion_xyzw, dim=-1, keepdim=True)
    if not bool(th.isfinite(norm).all()) or bool((norm <= 0).any()):
        raise ValueError('Finite nonzero quaternion required')
    q = quaternion_xyzw/norm
    q = th.where(q[..., 3:] < 0, -q, q)
    vector = q[..., :3]
    sine = th.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2*th.atan2(sine, q[..., 3:])
    scale = th.where(sine > 1e-8, angle/sine.clamp_min(1e-12), th.full_like(sine, 2.))
    return vector*scale
