"""仅在隔离进程中执行的 CPython 入口；不导入项目，不接收评测标签。"""

import json
import os
import random
import resource
import sys


POLICY_FILE = "<policy>"


def main():
    request = json.load(sys.stdin)
    limits = request["limits"]
    # hard limits 不能由生成代码提高。进程数设为零，禁止创建子进程。
    for name, value in (
        (resource.RLIMIT_CPU, limits["cpu_seconds"]),
        (resource.RLIMIT_AS, limits["memory_bytes"]),
        (resource.RLIMIT_FSIZE, limits["output_bytes"]),
        (resource.RLIMIT_NPROC, 0),
        (resource.RLIMIT_CORE, 0),
        (resource.RLIMIT_NOFILE, 32),
    ):
        resource.setrlimit(name, (value, value))
    response = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    # print、底层 write 和第三方调试输出都不能混入结果协议。
    with open(os.devnull, "r+b", buffering=0) as quiet:
        for fd in (0, 1, 2):
            os.dup2(quiet.fileno(), fd)
    random.seed(0)
    steps = 0

    def trace(frame, event, arg):
        nonlocal steps
        if frame.f_code.co_filename == POLICY_FILE:
            frame.f_trace_opcodes = True
            if event == "opcode":
                steps += 1
                if steps > limits["opcodes"]:
                    raise RuntimeError("Program opcode budget exceeded")
            return trace
        return None

    try:
        namespace = {"__name__": "__policy__", "public": request["public"]}
        code = compile(request["program"], POLICY_FILE, "exec")
        sys.settrace(trace)
        exec(code, namespace)
        function = namespace.get("build_goal")
        if not callable(function):
            raise ValueError("Define callable build_goal(public)")
        goal = function(request["public"])
        sys.settrace(None)
        result = {"ok": True, "goal": goal, "steps": steps}
        encoded = json.dumps(result, allow_nan=False, ensure_ascii=True)
        if len(encoded.encode()) > limits["output_bytes"]:
            raise ValueError("Program output byte budget exceeded")
    except BaseException as exc:
        sys.settrace(None)
        line = getattr(exc, "lineno", None)
        tb = exc.__traceback__
        while tb:
            if tb.tb_frame.f_code.co_filename == POLICY_FILE:
                line = tb.tb_lineno
            tb = tb.tb_next
        message = f"{type(exc).__name__}"
        if line is not None:
            message += f" at policy line {line}"
        message += f": {str(exc)[:1000]}"
        result = {"ok": False, "error": message, "steps": steps}
        if isinstance(exc, NameError) and exc.name:
            result["undefined_helper"] = exc.name
        encoded = json.dumps(result, ensure_ascii=True)
    response.write(encoded)
    response.flush()


if __name__ == "__main__":
    main()
