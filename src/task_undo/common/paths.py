"""Locate checkout resources and freeze the package used by a run.

Core code is installable; cases, configs and experiment data live in the checkout.
Workers receive copied inputs explicitly and never need checkout resources.
"""

from pathlib import Path
import shutil

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def project_root():
    """Use the editable checkout, or the current directory for a wheel install."""
    checkout = PACKAGE_ROOT.parent.parent
    if (checkout / "pyproject.toml").is_file() and (checkout / "cases").is_dir():
        return checkout
    return Path.cwd()


def freeze_package(output):
    """Return the import root of a complete, independently runnable source copy."""
    source = output / "source" / "src"
    shutil.copytree(PACKAGE_ROOT, source / "task_undo",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"))
    return source
