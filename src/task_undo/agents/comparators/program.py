"""有界 Python 子集解释器：数据只来自 public，不执行生成的 Python 字节码。

支持函数、条件、循环和容器，供 CaP 目标程序使用。没有文件、网络、
模块、任意属性或宿主对象访问；操作数、递归和容器大小均有上限。
这不是完整 Python，也不提供任何撤销规则函数。
"""

import ast
import copy
import operator


class ProgramError(ValueError):
    pass


class MissingFunction(ProgramError):
    def __init__(self, name):
        self.name = name
        super().__init__(f"Undefined helper function: {name}")


class Returned(Exception):
    def __init__(self, value):
        self.value = value


class BreakLoop(Exception):
    pass


class ContinueLoop(Exception):
    pass


LIMITS = {"steps": 20000, "source_chars": 32000, "ast_nodes": 4000,
          "items": 2048, "string_chars": 32000, "call_depth": 24}


class Program:
    def __init__(self, source):
        if not isinstance(source, str) or len(source) > LIMITS["source_chars"]:
            raise ProgramError("Program exceeds source limit")
        try:
            tree = ast.parse(source)
        except (SyntaxError, RecursionError) as exc:
            raise ProgramError("Invalid program syntax") from exc
        if len(list(ast.walk(tree))) > LIMITS["ast_nodes"]:
            raise ProgramError("Program exceeds AST limit")
        self.functions = {}
        self.steps = 0
        self.depth = 0
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            if not isinstance(node, ast.FunctionDef):
                raise ProgramError("Top level must contain function definitions only")
            args = node.args
            if (node.name.startswith("_") or node.name in self.functions or node.decorator_list
                    or node.returns or args.defaults or args.kwonlyargs or args.posonlyargs
                    or args.vararg or args.kwarg or any(a.annotation for a in args.args)):
                raise ProgramError("Use unique plain functions with positional parameters")
            self.functions[node.name] = node
        if "build_goal" not in self.functions:
            raise ProgramError("Define build_goal(public)")
        # Reject host access even in branches that this input does not reach.
        forbidden = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Lambda, ast.With,
                     ast.Try, ast.Raise, ast.Global, ast.Nonlocal, ast.AsyncFunctionDef,
                     ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)
        for node in ast.walk(tree):
            if (isinstance(node, forbidden)
                    or isinstance(node, ast.Name) and node.id.startswith("_")
                    or isinstance(node, ast.Attribute) and node.attr.startswith("_")):
                raise ProgramError("Unsupported program capability")

    def tick(self):
        self.steps += 1
        if self.steps > LIMITS["steps"]:
            raise ProgramError("Program instruction budget exhausted")

    def bounded(self, value, depth=0, seen=None):
        self.tick()
        if depth > 24:
            raise ProgramError("Value nesting limit exceeded")
        if type(value) in (str, list, tuple, dict, set):
            limit = LIMITS["string_chars"] if type(value) is str else LIMITS["items"]
            if len(value) > limit:
                raise ProgramError("Value size limit exceeded")
            if type(value) is not str:
                seen = set() if seen is None else seen
                if id(value) in seen:
                    raise ProgramError("Cyclic values are unsupported")
                seen.add(id(value))
                for item in (list(value.keys()) + list(value.values()) if type(value) is dict else value):
                    self.bounded(item, depth + 1, seen)
                seen.remove(id(value))
        elif type(value) is int:
            if value.bit_length() > 64:
                raise ProgramError("Integer size limit exceeded")
        elif value is not None and type(value) is not bool:
            raise ProgramError("Use JSON-like data and finite integers")
        return value

    def function(self, name, values):
        self.tick()
        if name not in self.functions:
            raise MissingFunction(name)
        node = self.functions[name]
        if len(values) != len(node.args.args):
            raise ProgramError("Function argument count mismatch")
        self.depth += 1
        if self.depth > LIMITS["call_depth"]:
            raise ProgramError("Function depth limit exceeded")
        try:
            self.block(node.body, dict(zip((a.arg for a in node.args.args), values)))
        except Returned as result:
            return self.bounded(result.value)
        finally:
            self.depth -= 1
        return None

    def assign(self, node, value, env):
        self.tick()
        if isinstance(node, ast.Name):
            env[node.id] = value
        elif isinstance(node, (ast.Tuple, ast.List)):
            if type(value) not in (list, tuple) or len(node.elts) != len(value):
                raise ProgramError("Unpacking arity mismatch")
            for key, item in zip(node.elts, value):
                self.assign(key, item, env)
        elif isinstance(node, ast.Subscript):
            target, key = self.expr(node.value, env), self.expr(node.slice, env)
            if type(target) not in (dict, list):
                raise ProgramError("Assignment requires dict or list")
            target[key] = value
            self.bounded(target)
        else:
            raise ProgramError("Unsupported assignment")

    def block(self, nodes, env):
        for node in nodes:
            self.tick()
            if isinstance(node, ast.Return):
                raise Returned(self.expr(node.value, env) if node.value else None)
            if isinstance(node, ast.Assign):
                value = self.expr(node.value, env)
                for target in node.targets:
                    self.assign(target, value, env)
            elif isinstance(node, ast.AugAssign):
                self.assign(node.target, self.binary(node.op, self.expr(node.target, env),
                                                    self.expr(node.value, env)), env)
            elif isinstance(node, ast.Expr):
                self.expr(node.value, env)
            elif isinstance(node, ast.If):
                self.block(node.body if self.expr(node.test, env) else node.orelse, env)
            elif isinstance(node, (ast.For, ast.While)):
                if isinstance(node, ast.For):
                    values = iter(self.expr(node.iter, env))
                broken = False
                while True:
                    self.tick()
                    if isinstance(node, ast.For):
                        try:
                            value = next(values)
                        except StopIteration:
                            break
                        self.assign(node.target, value, env)
                    elif not self.expr(node.test, env):
                        break
                    try:
                        self.block(node.body, env)
                    except ContinueLoop:
                        continue
                    except BreakLoop:
                        broken = True
                        break
                if not broken:
                    self.block(node.orelse, env)
            elif isinstance(node, ast.Break):
                raise BreakLoop()
            elif isinstance(node, ast.Continue):
                raise ContinueLoop()
            elif not isinstance(node, ast.Pass):
                raise ProgramError(f"Unsupported statement: {type(node).__name__}")

    def binary(self, op, left, right):
        operations = {ast.Add: operator.add, ast.Sub: operator.sub,
                      ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
        if type(op) not in operations:
            raise ProgramError("Unsupported arithmetic")
        if isinstance(op, ast.Mult) and (type(left) is not int or type(right) is not int):
            raise ProgramError("Only integer multiplication is supported")
        if isinstance(op, ast.Mod) and (type(left) is not int or type(right) is not int):
            raise ProgramError("Only integer remainder is supported")
        return self.bounded(operations[type(op)](left, right))

    def expr(self, node, env):
        self.tick()
        if isinstance(node, ast.Constant):
            value = node.value
        elif isinstance(node, ast.Name):
            if node.id not in env:
                raise ProgramError(f"Undefined variable: {node.id}")
            value = env[node.id]
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            values = [self.expr(x, env) for x in node.elts]
            value = {ast.List: list, ast.Tuple: tuple, ast.Set: set}[type(node)](values)
        elif isinstance(node, ast.Dict):
            if any(k is None for k in node.keys):
                raise ProgramError("Dictionary unpacking is unsupported")
            value = {self.expr(k, env): self.expr(v, env) for k, v in zip(node.keys, node.values)}
        elif isinstance(node, ast.Subscript):
            target = self.expr(node.value, env)
            if isinstance(node.slice, ast.Slice):
                n = node.slice
                key = slice(*(self.expr(x, env) if x else None for x in (n.lower, n.upper, n.step)))
            else:
                key = self.expr(node.slice, env)
            value = target[key]
        elif isinstance(node, ast.BinOp):
            value = self.binary(node.op, self.expr(node.left, env), self.expr(node.right, env))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub, ast.UAdd)):
            value = {ast.Not: operator.not_, ast.USub: operator.neg, ast.UAdd: operator.pos}[type(node.op)](self.expr(node.operand, env))
        elif isinstance(node, ast.BoolOp):
            value = self.expr(node.values[0], env)
            for item in node.values[1:]:
                if isinstance(node.op, ast.And) and not value or isinstance(node.op, ast.Or) and value:
                    break
                value = self.expr(item, env)
        elif isinstance(node, ast.Compare):
            comparisons = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
                           ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
                           ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
                           ast.Is: operator.is_, ast.IsNot: operator.is_not}
            left, value = self.expr(node.left, env), True
            for op, other in zip(node.ops, node.comparators):
                right = self.expr(other, env)
                if not comparisons[type(op)](left, right):
                    value = False
                    break
                left = right
        elif isinstance(node, ast.IfExp):
            value = self.expr(node.body if self.expr(node.test, env) else node.orelse, env)
        elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
            collected = []
            def collect(index, local):
                self.tick()
                if index == len(node.generators):
                    collected.append((self.expr(node.key, local), self.expr(node.value, local))
                                     if isinstance(node, ast.DictComp) else self.expr(node.elt, local))
                    self.bounded(collected)
                    return
                gen = node.generators[index]
                if gen.is_async:
                    raise ProgramError("Async comprehensions are unsupported")
                for item in self.expr(gen.iter, local):
                    self.assign(gen.target, item, local)
                    if all(self.expr(test, local) for test in gen.ifs):
                        collect(index + 1, local.copy())
            collect(0, env.copy())
            value = dict(collected) if isinstance(node, ast.DictComp) else set(collected) if isinstance(node, ast.SetComp) else collected
        elif isinstance(node, ast.Call):
            if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
                raise ProgramError("Use positional arguments only")
            args = [self.expr(a, env) for a in node.args]
            if isinstance(node.func, ast.Name):
                name = node.func.id
                builtins = {"len": len, "list": list, "dict": dict, "set": set, "tuple": tuple,
                            "sorted": sorted, "str": str, "int": int, "bool": bool,
                            "min": min, "max": max, "any": any, "all": all, "abs": abs}
                if name in self.functions:
                    value = self.function(name, args)
                elif name == "range":
                    seq = range(*args)
                    if len(seq) > LIMITS["items"]:
                        raise ProgramError("Range size limit exceeded")
                    value = list(seq)
                elif name in ("enumerate", "zip", "reversed"):
                    value = list({"enumerate": enumerate, "zip": zip, "reversed": reversed}[name](*args))
                elif name in builtins:
                    value = builtins[name](*args)
                elif name in ("open", "eval", "exec", "compile", "getattr", "setattr", "globals", "locals", "type", "input"):
                    raise ProgramError("Host access is unavailable")
                else:
                    raise MissingFunction(name)
            elif isinstance(node.func, ast.Attribute):
                target = self.expr(node.func.value, env)
                allowed = {str: ("lower", "upper", "strip", "split", "rsplit", "startswith", "endswith", "replace", "find", "join"),
                           dict: ("get", "keys", "values", "items", "copy", "update", "pop", "setdefault"),
                           list: ("append", "extend", "pop", "index", "count", "copy", "remove"),
                           set: ("add", "discard", "union", "intersection", "difference")}
                name = node.func.attr
                if name not in allowed.get(type(target), ()):
                    raise ProgramError("Unsupported data method")
                value = getattr(target, name)(*args)
                if type(target) is dict and name in ("keys", "values", "items"):
                    value = list(value)
                self.bounded(target)
            else:
                raise ProgramError("Only named functions and listed data methods are callable")
        else:
            raise ProgramError(f"Unsupported expression: {type(node).__name__}")
        return self.bounded(value)

    def run(self, public):
        self.bounded(public)
        try:
            return self.function("build_goal", [copy.deepcopy(public)])
        except (KeyError, IndexError, TypeError, ZeroDivisionError, OverflowError,
                RecursionError, BreakLoop, ContinueLoop) as exc:
            raise ProgramError(f"Program execution failed: {type(exc).__name__}") from None
