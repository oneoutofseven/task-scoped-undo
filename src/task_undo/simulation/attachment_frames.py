"""Match held-object attachment poses to the pinned OG collision-world frame.

OG builds its cuRobo obstacle meshes in robot.root_link coordinates, but its
attachment adapter passes a world-frame EEF pose. Convert only that explicit
pose, inside the adapter call; retain upstream fitting, scaling and detachment.
No simulator state is changed. This is for the pinned synchronous OG adapter.
"""
from contextlib import contextmanager


def attachment_pose_in_root(world_eef, root_position, root_quaternion_xyzw):
    """Return root_T_eef = inverse(world_T_root) @ world_T_eef.

    cuRobo Pose uses wxyz; the simulator root quaternion uses xyzw. Pose tensor
    operations preserve the query device/dtype and do not modify the inputs.
    """
    world_root = type(world_eef)(
        position=root_position.to(world_eef.position),
        quaternion=root_quaternion_xyzw[[3, 0, 1, 2]].to(world_eef.quaternion),
    )
    return world_root.inverse().multiply(world_eef)


@contextmanager
def root_frame_attachment(generator, root, scale=None):
    """Scope the conversion to OG's explicit attachment calls, never FK poses."""
    original = generator.attach_objects_to_robot
    owned = 'attach_objects_to_robot' in vars(generator)
    previous = vars(generator).get('attach_objects_to_robot')

    def attach(*args, **kwargs):
        # OG supplies ee_pose by keyword. None selects cuRobo FK, already local.
        pose = kwargs.get('ee_pose')
        if pose is not None:
            kwargs['ee_pose'] = attachment_pose_in_root(pose, *root.get_position_orientation())
        if scale is not None:
            kwargs['scale'] = scale
        return original(*args, **kwargs)

    generator.attach_objects_to_robot = attach
    try:
        yield
    finally:
        if owned:
            generator.attach_objects_to_robot = previous
        else:
            delattr(generator, 'attach_objects_to_robot')


def install_attachment_frame(planner, scale=None):
    """Install once on a session-local planner; no global monkey patching."""
    if getattr(planner, '_task_undo_root_attachment', False):
        if getattr(planner, '_task_undo_attachment_scale', None) != scale:
            raise ValueError('Cannot change geometry policy on an installed planner')
        return
    original = planner._attach_objects_to_robot

    def attach(attached_obj, attached_obj_scale, cu_js_batch, emb_sel):
        if attached_obj is None:
            return original(attached_obj, attached_obj_scale, cu_js_batch, emb_sel)
        with root_frame_attachment(planner.mg[emb_sel], planner.robot.root_link, scale=scale):
            return original(attached_obj, attached_obj_scale, cu_js_batch, emb_sel)

    planner._attach_objects_to_robot = attach
    planner._task_undo_root_attachment = True
    planner._task_undo_attachment_scale = scale
