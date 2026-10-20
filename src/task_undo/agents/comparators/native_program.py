"""真实 Python 语言语义，隔离宿主文件、网络、进程与 GPU。

Linux / CPython 3.10 / bubblewrap 是这个后端的显式依赖。
隔离不可用时报基础设施错误，绝不退回宿主进程 exec。
"""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile

from task_undo.agents.comparators.program import ProgramError


LIMITS = {
    "source_chars": 32000, "input_bytes": 4 * 1024 * 1024,
    "opcodes": 200000, "cpu_seconds": 3, "wall_seconds": 10,
    "memory_bytes": 256 * 1024 * 1024, "output_bytes": 65536,
}
PYTHON = Path("/usr/bin/python3.10")
BWRAP = Path("/usr/bin/bwrap")
STDLIB = Path("/usr/lib/python3.10")
WORKER = Path(__file__).with_name("native_worker.py")


class NativeProgramError(ProgramError):
    def __init__(self, message, *, name=None):
        super().__init__(message)
        self.name = name


@lru_cache(maxsize=1)
def runtime_identity():
    """保存解释器与标准库指纹，避免把升级后的执行器当作原版本重放。"""
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    try:
        version = subprocess.run([str(PYTHON), "-I", "-S", "--version"],
                                 check=True, capture_output=True, text=True, timeout=5).stdout.strip()
        wrapper = subprocess.run([str(BWRAP), "--version"], check=True,
                                 capture_output=True, text=True, timeout=5).stdout.strip()
        files = {str(p.relative_to(STDLIB)): digest(p) for p in sorted(STDLIB.rglob("*"))
                 if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".so")}
        if not files:
            raise OSError("Python standard library is missing")
        return {
            "backend": "cpython_bubblewrap_v1", "python": version, "bubblewrap": wrapper,
            "python_sha256": digest(PYTHON), "bubblewrap_sha256": digest(BWRAP),
            "worker_sha256": digest(WORKER),
            "stdlib_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        }
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Native CaP requires CPython 3.10 and working bubblewrap") from exc


def sandbox_command():
    # 不绑定整个 /usr，尤其不暴露第三方 site-packages、仓库或用户目录。
    return [
        str(BWRAP), "--unshare-all", "--die-with-parent", "--new-session",
        "--cap-drop", "ALL", "--clearenv",
        "--setenv", "PYTHONHASHSEED", "0", "--setenv", "LC_ALL", "C.UTF-8",
        "--ro-bind", str(PYTHON), str(PYTHON),
        "--ro-bind", str(STDLIB), str(STDLIB),
        "--ro-bind", "/usr/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu",
        "--ro-bind", "/usr/lib64", "/usr/lib64",
        "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
        "--ro-bind", str(WORKER), "/worker.py",
        "--proc", "/proc", "--dev", "/dev", "--dir", "/work",
        "--chdir", "/work", "--remount-ro", "/",
        "--", str(PYTHON), "-s", "-S", "-B", "/worker.py",
    ]


class NativeProgram:
    def __init__(self, source):
        if not isinstance(source, str) or len(source) > LIMITS["source_chars"]:
            raise ProgramError("Program must be a string within the source character budget")
        self.source = source
        self.steps = 0

    def run(self, public):
        runtime_identity()
        payload = json.dumps({"program": self.source, "public": public, "limits": LIMITS},
                             allow_nan=False).encode()
        if len(payload) > LIMITS["input_bytes"]:
            raise ProgramError("Program input byte budget exceeded")
        # 文件输出受 worker 的 RLIMIT_FSIZE 约束，不会无限堆积在父进程内存。
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as error:
            try:
                process = subprocess.Popen(sandbox_command(), stdin=subprocess.PIPE,
                                           stdout=output, stderr=error, close_fds=True,
                                           start_new_session=True, env={})
            except OSError as exc:
                raise RuntimeError("Cannot launch the native CaP sandbox") from exc
            try:
                process.communicate(payload, timeout=LIMITS["wall_seconds"])
            except BaseException as exc:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
                if isinstance(exc, subprocess.TimeoutExpired):
                    raise NativeProgramError("Program wall time budget exceeded") from exc
                raise
            output.seek(0)
            raw = output.read(LIMITS["output_bytes"] + 1)
            error.seek(0)
            stderr = error.read(2048).decode(errors="replace")
            if process.returncode and stderr:
                # bwrap / Python 启动错误未经过 worker，不计作模型程序错误。
                raise RuntimeError("Native CaP sandbox failed: " + stderr.strip())
            if process.returncode or not raw:
                raise NativeProgramError("Program terminated without a result or exceeded a resource budget")
            if len(raw) > LIMITS["output_bytes"]:
                raise NativeProgramError("Program output byte budget exceeded")
            try:
                result = json.loads(raw)
                self.steps = result["steps"]
                if type(result["ok"]) is not bool or type(self.steps) is not int:
                    raise ValueError("Invalid result fields")
                if not result["ok"]:
                    raise NativeProgramError(result["error"], name=result.get("undefined_helper"))
                return result["goal"]
            except NativeProgramError:
                raise
            except (ValueError, KeyError, TypeError) as exc:
                raise NativeProgramError("Program returned an invalid execution envelope") from exc
