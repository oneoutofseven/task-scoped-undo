"""Scoped compatibility for missing device forwarding in pinned cuRobo.

Upstream issue: https://github.com/NVlabs/curobo/issues/559
The shared third_party checkout remains unchanged, preserving old runtimes.
"""
from contextlib import contextmanager, ExitStack
import functools
import inspect


@contextmanager
def forwarded_default(owner, name, parameter, value):
    """Supply only a missing argument; preserve static/classmethod dispatch."""
    descriptor = vars(owner)[name]
    if not isinstance(descriptor, (staticmethod, classmethod)):
        raise TypeError("Expected a static or class factory")
    function = descriptor.__func__
    signature = inspect.signature(function)
    if parameter not in signature.parameters:
        raise ValueError("Factory signature no longer matches the pinned compatibility adapter")

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        if parameter not in bound.arguments:
            kwargs[parameter] = value
        return function(*args, **kwargs)

    setattr(owner, name, type(descriptor)(wrapped))
    try:
        yield
    finally:
        setattr(owner, name, descriptor)


@contextmanager
def planner_device_defaults(device):
    """During synchronous construction only, align evaluator and graph config."""
    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.wrap.reacher.evaluator import TrajEvaluatorConfig
    from curobo.rollout.arm_base import ArmBaseConfig

    tensor_args = TensorDeviceType(device=torch.device(device))
    with ExitStack() as stack:
        stack.enter_context(forwarded_default(TrajEvaluatorConfig, "from_basic", "tensor_args", tensor_args))
        stack.enter_context(forwarded_default(ArmBaseConfig, "from_dict", "tensor_args", tensor_args))
        yield


def check_evaluator_devices(planner):
    """Fail during setup if either trajectory evaluator escaped the device."""
    expected = str(planner.tensor_args.device)
    devices = {str(emb): {
        name: str(getattr(generator, name).minimum_trajectory_dt.device)
        for name in ("trajopt_solver", "finetune_trajopt_solver")}
        for emb, generator in planner.mg.items()}
    if any(actual != expected for group in devices.values() for actual in group.values()):
        raise RuntimeError("Trajectory evaluator and motion planner devices disagree")
    return devices
