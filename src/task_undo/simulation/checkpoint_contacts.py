"""Opt-in contact-query initialization after loading a physical checkpoint.

Only use before the undo phase. No poses, predicates or expected labels are set.
"""
CONTACT_REFRESH = 'contact_refresh_v1'
CONTACT_STEPS = 5


def initialize_contacts(scene, step, mode='default'):
    if mode not in ('default', CONTACT_REFRESH):
        raise ValueError('Unknown checkpoint restore mode')
    if mode == 'default':
        return
    # Sleeping bodies may not emit contact reports immediately after restore.
    scene.wake_scene_objects()
    for _ in range(CONTACT_STEPS):
        step()
    # Kinematics-based query caches can still retain the pre-refresh false value
    # despite fresh positive contact reports. Invalidate; never assign an answer.
    for obj in scene.objects:
        for state in obj.states.values():
            state.clear_cache()
