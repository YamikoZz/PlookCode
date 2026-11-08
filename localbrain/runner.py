"""Lightweight test runner for model-written solutions, no Docker required.

Layers of protection (accident-proofing for generated code, NOT a hard security
boundary like a VM or container -- see README):
  * static check before execution (Python AST allowlist, JavaScript token blocklist)
  * Python: audit hook blocks file/network/process/ctypes events after startup
  * JavaScript: Node permission model (no fs writes, child processes, workers)
    and dangerous globals shadowed
  * TypeScript: types stripped by Node's built-in stripper (parsing only), then run as JavaScript
  * Lua: token blocklist, then loaded as text into an environment without io/os/load/require/debug
  * separate process in an empty temporary directory, minimal environment
  * wall-clock timeout, output cap, and on Windows a Job Object limiting memory
    and forbidding child processes (POSIX: rlimits)
"""
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

OUTPUT_LIMIT = 16384
MEMORY_LIMIT = 256 * 1024 * 1024

PY_MODULES = {"math", "itertools", "functools", "collections", "string", "re", "heapq",
              "bisect", "statistics", "operator", "typing", "fractions", "decimal",
              "dataclasses", "enum", "copy", "random", "datetime", "json", "textwrap"}
PY_BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals",
                   "vars", "getattr", "setattr", "delattr", "input", "breakpoint", "exit",
                   "quit", "help", "memoryview", "__builtins__", "__loader__", "__spec__"}
JS_BANNED = re.compile(r"\b(require|import|process|globalThis|global|eval|Function|constructor|"
                       r"__proto__|prototype\s*\.\s*constructor|fetch|WebSocket|Worker|"
                       r"Deno|Bun|module|exports|setInterval)\b")


LUA_BANNED = re.compile(r"\b(require|dofile|loadfile|loadstring|load|os|io|debug|package|"
                        r"collectgarbage|setfenv|getfenv|_G)\b")


class Rejected(ValueError):
    """Code refused by the static check; it was never executed."""


def check_python(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise Rejected(f"SyntaxError: {error.msg} (line {error.lineno})") from None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in PY_MODULES:
                    raise Rejected(f"import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in PY_MODULES:
                raise Rejected(f"import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in PY_BANNED_NAMES:
            raise Rejected(f"use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise Rejected(f"dunder attribute '{node.attr}' is not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)) and False:
            pass
    return tree


def check_javascript(code):
    match = JS_BANNED.search(code)
    if match:
        raise Rejected(f"use of '{match.group(0)}' is not allowed")


def check_lua(code):
    match = LUA_BANNED.search(code)
    if match:
        raise Rejected(f"use of '{match.group(0)}' is not allowed")


PY_HARNESS = r'''
import json, sys
for _name in %(modules)r:
    __import__(_name)
_src = open(sys.argv[1], encoding="utf-8").read()
_cases = json.load(open(sys.argv[2], encoding="utf-8"))
_fn_name = sys.argv[3]
_allowed = set(sys.modules)
def _hook(event, args):
    if event == "import" and args[0] not in _allowed:
        raise RuntimeError("import blocked: " + str(args[0]))
    if event == "open" or event.split(".")[0] in {"socket", "subprocess", "os", "shutil",
                                                  "ctypes", "winreg", "_winapi", "msvcrt"}:
        if event not in {"os.listdir"}:
            raise RuntimeError("blocked operation: " + event)
sys.addaudithook(_hook)
_out = sys.__stdout__
class _Quiet:
    """User prints are discarded so they cannot crowd out the result line."""
    def write(self, text): return len(text)
    def flush(self): pass
sys.stdout = sys.stderr = _Quiet()
def _report(obj):
    _out.write("@@RESULT@@" + json.dumps(obj) + "\n"); _out.flush()
try:
    _scope = {"__name__": "solution"}
    exec(compile(_src, "solution.py", "exec"), _scope)
    _fn = _scope.get(_fn_name)
    if not callable(_fn):
        _report({"status": "error", "message": "function '%%s' is not defined" %% _fn_name, "passed": 0}); sys.exit(0)
except BaseException as _e:
    _report({"status": "error", "message": type(_e).__name__ + ": " + str(_e)[:300], "passed": 0}); sys.exit(0)
_passed = 0
for _i, _case in enumerate(_cases):
    try:
        _got = _fn(*_case["args"])
        if isinstance(_got, tuple): _got = list(_got)
        _got = json.loads(json.dumps(_got))
    except BaseException as _e:
        _report({"status": "error", "passed": _passed, "case": _i,
                 "message": "%%s(%%s) raised %%s: %%s" %% (_fn_name, json.dumps(_case["args"])[1:-1], type(_e).__name__, str(_e)[:200])})
        sys.exit(0)
    if _got != _case["expected"]:
        _report({"status": "fail", "passed": _passed, "case": _i,
                 "message": "%%s(%%s) returned %%s, expected %%s" %% (_fn_name, json.dumps(_case["args"])[1:-1], json.dumps(_got)[:200], json.dumps(_case["expected"])[:200])})
        sys.exit(0)
    _passed += 1
_report({"status": "pass", "passed": _passed})
'''

JS_HARNESS = r'''
"use strict";
const __log = console.log.bind(console);
const __cases = %(cases)s;
const __name = %(name)s;
let __fn;
try {
  const __quiet = {log() {}, error() {}, warn() {}, info() {}, debug() {}};
  __fn = (function (require, process, module, exports, globalThis, global, fetch, WebSocket,
                    Worker, setInterval, Function, console) {
    %(code)s
    ;return typeof %(name_raw)s === "function" ? %(name_raw)s : undefined;
  })(undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined,
     undefined, undefined, undefined, __quiet);
} catch (e) {
  __log("@@RESULT@@" + JSON.stringify({status: "error", passed: 0, message: String(e).slice(0, 300)}));
}
if (__fn === undefined) {
  __log("@@RESULT@@" + JSON.stringify({status: "error", passed: 0, message: "function '" + __name + "' is not defined"}));
} else {
  let __passed = 0, __done = false;
  for (let i = 0; i < __cases.length && !__done; i++) {
    const c = __cases[i];
    const args = JSON.stringify(c.args).slice(1, -1);
    try {
      let got = __fn(...JSON.parse(JSON.stringify(c.args)));
      const g = JSON.stringify(got === undefined ? null : got), e = JSON.stringify(c.expected);
      if (g !== e) {
        __log("@@RESULT@@" + JSON.stringify({status: "fail", passed: __passed, case: i,
          message: __name + "(" + args + ") returned " + String(g).slice(0, 200) + ", expected " + e.slice(0, 200)}));
        __done = true;
      } else __passed++;
    } catch (err) {
      __log("@@RESULT@@" + JSON.stringify({status: "error", passed: __passed, case: i,
        message: __name + "(" + args + ") threw " + String(err).slice(0, 200)}));
      __done = true;
    }
  }
  if (!__done) __log("@@RESULT@@" + JSON.stringify({status: "pass", passed: __passed}));
}
'''


LUA_HARNESS = r'''
local out = io.stdout
local fh = io.open(arg[1], "rb"); local src = fh:read("*a"); fh:close()
local name = arg[2]
local CASES = %(cases)s
local unpack = table.unpack or unpack

local function enc(v)
  local t = type(v)
  if v == nil then return "null"
  elseif t == "boolean" then return tostring(v)
  elseif t == "number" then
    if v ~= v or v == math.huge or v == -math.huge then return "null" end
    if (math.type and math.type(v) == "integer") or (v == math.floor(v) and math.abs(v) < 2^53) then
      return string.format("%%d", v)
    end
    return string.format("%%.17g", v)
  elseif t == "string" then
    return '"' .. (v:gsub('[%%c"\\]', function(c) return string.format("\\u%%04x", c:byte()) end)) .. '"'
  elseif t == "table" then
    local n, count = #v, 0
    for _ in pairs(v) do count = count + 1 end
    if count == n then
      local parts = {}
      for i = 1, n do parts[i] = enc(v[i]) end
      return "[" .. table.concat(parts, ",") .. "]"
    end
    local keys, parts = {}, {}
    for k in pairs(v) do keys[#keys + 1] = k end
    table.sort(keys, function(p, q) return tostring(p) < tostring(q) end)
    for _, k in ipairs(keys) do parts[#parts + 1] = enc(tostring(k)) .. ":" .. enc(v[k]) end
    return "{" .. table.concat(parts, ",") .. "}"
  end
  return '"<' .. t .. '>"'
end

local function report(status, passed, case, message)
  local s = '{"status":' .. enc(status) .. ',"passed":' .. passed
  if case then s = s .. ',"case":' .. case end
  if message then s = s .. ',"message":' .. enc(message) end
  out:write("@@RESULT@@" .. s .. "}\n"); out:flush()
end

local function copy(t) local r = {} for k, v in pairs(t) do r[k] = v end return r end
local env = {math = copy(math), string = copy(string), table = copy(table), utf8 = utf8 and copy(utf8),
  pairs = pairs, ipairs = ipairs, next = next, select = select, tostring = tostring, tonumber = tonumber,
  type = type, error = error, pcall = pcall, assert = assert, unpack = unpack, setmetatable = setmetatable,
  getmetatable = getmetatable, rawequal = rawequal, rawget = rawget, rawset = rawset, rawlen = rawlen,
  print = function() end}
env.string.dump = nil
local load = load
io, os, debug, package, require, dofile, loadfile = nil, nil, nil, nil, nil, nil, nil

local chunk, err = load(src .. "\nreturn " .. name, "=solution", "t", env)
if not chunk then chunk, err = load(src, "=solution", "t", env) end
if not chunk then report("error", 0, nil, tostring(err)) return end
local ok, result = pcall(chunk)
if not ok then report("error", 0, nil, tostring(result)) return end
local fn = type(result) == "function" and result or env[name]
if type(fn) ~= "function" then report("error", 0, nil, "function '" .. name .. "' is not defined") return end
local passed = 0
for i, c in ipairs(CASES) do
  local shown = enc(c.args):sub(2, -2)
  local args = {}
  for j = 1, c.n do args[j] = c.args[j] end
  local good, got = pcall(fn, unpack(args, 1, c.n))
  if not good then
    report("error", passed, i - 1, name .. "(" .. shown .. ") raised " .. tostring(got):sub(1, 200))
    return
  end
  local g = enc(got)
  if g ~= c.expected then
    report("fail", passed, i - 1, name .. "(" .. shown .. ") returned " .. g:sub(1, 200) .. ", expected " .. c.expected:sub(1, 200))
    return
  end
  passed = passed + 1
end
report("pass", passed)
'''


def lua_string(text):
    """A Lua string literal (control characters as decimal escapes, UTF-8 kept as raw bytes)."""
    out = []
    for c in text:
        if c in '"\\':
            out.append("\\" + c)
        elif ord(c) < 32 or ord(c) == 127:
            out.append("\\%03d" % ord(c))
        else:
            out.append(c)
    return '"' + "".join(out) + '"'


def lua_literal(value):
    """A JSON value as a Lua constructor (lists -> array tables, dicts -> keyed tables)."""
    if value is None:
        return "nil"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return lua_string(value)
    if isinstance(value, list):
        return "{" + ", ".join(lua_literal(v) for v in value) + "}"
    return "{" + ", ".join(f"[{lua_string(str(k))}] = {lua_literal(v)}" for k, v in value.items()) + "}"


def lua_expected(value):
    """The expected value encoded exactly like the Lua harness encodes results."""
    if isinstance(value, float) and value.is_integer() and abs(value) < 2 ** 53:
        value = int(value)
    if isinstance(value, list):
        return "[" + ",".join(lua_expected(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(json.dumps(str(k)) + ":" + lua_expected(v) for k, v in
                              sorted(value.items(), key=lambda kv: str(kv[0]))) + "}" if value else "[]"
    if isinstance(value, str):
        return '"' + "".join(f"\\u{ord(c):04x}" if ord(c) < 32 or c in '"\\' or ord(c) == 127 else c
                             for c in value) + '"'
    if isinstance(value, float):
        return "%.17g" % value
    return json.dumps(value)


def lua_cases(cases):
    rows = [f"{{args = {lua_literal(c['args'])}, n = {len(c['args'])}, expected = {lua_string(lua_expected(c['expected']))}}}"
            for c in cases]
    return "{" + ",\n".join(rows) + "}"


_TS_STRIP = ("const m = require('module'); let s = ''; process.stdin.setEncoding('utf8');"
             "process.stdin.on('data', d => s += d).on('end', () => {"
             "try { process.stdout.write('OK' + m.stripTypeScriptTypes(s)); }"
             "catch (e) { process.stdout.write('ER' + String(e.message || e).slice(0, 300)); } });")
_TS_SUPPORT = {}


def typescript_supported(node):
    if node not in _TS_SUPPORT:
        try:
            probe = subprocess.run([node, "-e", "process.stdout.write(typeof require('module')"
                                    ".stripTypeScriptTypes === 'function' ? 'yes' : 'no')"],
                                   capture_output=True, text=True, timeout=20)
            _TS_SUPPORT[node] = probe.stdout.strip() == "yes"
        except (OSError, subprocess.TimeoutExpired):
            _TS_SUPPORT[node] = False
    return _TS_SUPPORT[node]


def strip_types(node, code):
    """TypeScript -> JavaScript with Node's type stripper. It only parses the code, never runs it."""
    done = subprocess.run([node, "-e", _TS_STRIP], input=code, capture_output=True, text=True,
                          encoding="utf-8", timeout=20)
    if done.stdout.startswith("OK"):
        return done.stdout[2:], None
    detail = done.stdout[2:] if done.stdout.startswith("ER") else done.stderr.strip()[:300]
    return None, "TypeScript: " + detail


def toolchains():
    """Interpreters this machine can use for checking solutions."""
    # A venv python.exe on Windows is a launcher that spawns a child; use the base interpreter.
    found = {"python": getattr(sys, "_base_executable", None) or sys.executable}
    node = shutil.which("node")
    if node:
        found["javascript"] = node
        if typescript_supported(node):
            found["typescript"] = node
    lua = next((p for p in map(shutil.which, ("lua5.4", "lua54", "lua5.3", "lua")) if p), None)
    if lua:
        found["lua"] = lua
    return found


def _limits_windows(pid, memory=MEMORY_LIMIT, processes=1):
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("r", "w", "o", "rb", "wb", "ob")]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("Basic", BASIC), ("Io", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.OpenProcess.restype = wintypes.HANDLE
    job = kernel.CreateJobObjectW(None, None)
    info = EXTENDED()
    # PROCESS_MEMORY | ACTIVE_PROCESS | KILL_ON_JOB_CLOSE | DIE_ON_UNHANDLED_EXCEPTION
    info.Basic.LimitFlags = 0x100 | 0x8 | 0x2000 | 0x400
    info.Basic.ActiveProcessLimit = processes
    info.ProcessMemoryLimit = memory
    if not kernel.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
    handle = kernel.OpenProcess(0x0001 | 0x0100, False, pid)  # TERMINATE | SET_QUOTA
    if not handle or not kernel.AssignProcessToJobObject(job, handle):
        raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
    kernel.CloseHandle(handle)
    return job, kernel


def _execute(command, cwd, timeout, memory=MEMORY_LIMIT, processes=1, env_extra=None,
             output_limit=OUTPUT_LIMIT):
    env = {"PATH": os.environ.get("PATH", ""), "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
           "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1", **(env_extra or {})}
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000 | 0x00004000  # CREATE_NO_WINDOW | BELOW_NORMAL
    else:
        def limit():
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (memory * 4, memory * 4))
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
            os.nice(10)
        kwargs["preexec_fn"] = limit
    proc = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)
    job = None
    if os.name == "nt":
        try:
            job = _limits_windows(proc.pid, memory, processes)
        except OSError:
            proc.kill()
            proc.wait()
            raise
    output, timed_out = bytearray(), False

    def drain():
        while block := proc.stdout.read1(4096):
            if len(output) < output_limit:
                output.extend(block[:output_limit - len(output)])

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        proc.wait()
    reader.join(timeout=5)
    proc.stdout.close()
    if job:
        job[1].CloseHandle(job[0])
    return output.decode("utf-8", errors="replace"), timed_out


def run_tests(language, code, function, cases, timeout=5):
    """Run `function` from `code` on JSON test cases. Returns a result dict."""
    total = len(cases)
    if not isinstance(code, str) or len(code.encode("utf-8")) > 65536:
        return {"status": "rejected", "passed": 0, "total": total, "message": "code too large"}
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", function):
        raise ValueError("invalid function name")
    chains = toolchains()
    if language not in chains:
        return {"status": "unavailable", "passed": 0, "total": total,
                "message": f"no {language} toolchain installed"}
    if language == "typescript":
        code, problem = strip_types(chains["typescript"], code)
        if problem:
            return {"status": "error", "passed": 0, "total": total, "message": problem}
        language = "javascript"
    try:
        {"python": check_python, "lua": check_lua}.get(language, check_javascript)(code)
    except Rejected as error:
        return {"status": "rejected", "passed": 0, "total": total, "message": str(error)}
    with tempfile.TemporaryDirectory(prefix="lb-run-") as temp:
        temp = Path(temp)
        if language == "python":
            (temp / "solution.py").write_text(code, encoding="utf-8")
            (temp / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
            (temp / "harness.py").write_text(PY_HARNESS % {"modules": sorted(PY_MODULES)}, encoding="utf-8")
            command = [chains["python"], "-I", "-S", "-B", str(temp / "harness.py"),
                       str(temp / "solution.py"), str(temp / "cases.json"), function]
        elif language == "lua":
            (temp / "solution.lua").write_text(code, encoding="utf-8")
            (temp / "harness.lua").write_text(LUA_HARNESS % {"cases": lua_cases(cases)}, encoding="utf-8")
            command = [chains["lua"], str(temp / "harness.lua"), str(temp / "solution.lua"), function]
        else:
            script = temp / "harness.js"
            script.write_text(JS_HARNESS % {"cases": json.dumps(cases), "name": json.dumps(function),
                                            "name_raw": function, "code": code}, encoding="utf-8")
            command = [chains["javascript"], "--permission", f"--allow-fs-read={script}",
                       "--max-old-space-size=128", str(script)]
        output, timed_out = _execute(command, temp, timeout)
    if timed_out:
        return {"status": "timeout", "passed": 0, "total": total, "message": f"timed out after {timeout}s"}
    for line in reversed(output.splitlines()):
        if line.startswith("@@RESULT@@"):
            result = json.loads(line[len("@@RESULT@@"):])
            result["total"] = total
            result.setdefault("message", "")
            return result
    lines = [l.strip() for l in output.strip().splitlines() if l.strip()]
    errors = [l for l in lines if re.match(r"[A-Za-z]*Error[: ]", l)]
    detail = errors[0] if errors else " | ".join(lines[-3:])
    return {"status": "error", "passed": 0, "total": total,
            "message": ("crashed: " + detail)[:400] if lines else "crashed without output"}
