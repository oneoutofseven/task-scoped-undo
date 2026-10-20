"""Refresh support queries without discarding the native sleeping-body cache.

The pinned contact tensor can lose resting contacts when bodies sleep. Native
KinematicsMixin intentionally preserves their last observation. Conversely, a
cached negative can survive newly established contact. Keep a native positive
only when both bodies sleep; otherwise refresh that pair through update_cache.
This changes query caching only, never a tensor, body, contact or physics step.
"""


def current_supports(state, surfaces):
    supported = []
    for role, surface in surfaces.items():
        cached = state.get_value(surface)
        if not (cached and state.obj.is_asleep and surface.is_asleep):
            state.update_cache((surface,))
        if state.get_value(surface):
            supported.append(role)
    return supported
