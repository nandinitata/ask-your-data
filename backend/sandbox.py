"""Run model-written Python against a dataframe, without letting it run wild.

The code the model writes is never trusted. It runs in a separate Python
process with a time limit, a trimmed set of builtins, and an import allow-list.
It can touch pandas, numpy and matplotlib and the dataframe, and nothing else.

This is a guard for running on your own data locally, not a hard security
boundary. Don't point it at code you actively distrust.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# what the generated code is allowed to import
ALLOWED_IMPORTS = {
    "pandas", "numpy", "math", "statistics", "datetime", "collections",
    "itertools", "functools", "re", "json", "matplotlib",
}

TIMEOUT_SECONDS = 25


@dataclass
class RunResult:
    ok: bool
    stdout: str
    result_repr: str | None   # the `result` variable, printed
    chart_path: str | None    # PNG written by the code, if any
    error: str | None


# This template wraps the model's code. It sets up a locked-down namespace,
# loads the dataframe as `df`, runs the code, then reports back as JSON.
_RUNNER = r'''
import json, sys, builtins, importlib

CSV_PATH = {csv_path!r}
CHART_PATH = {chart_path!r}
ALLOWED = set({allowed!r})

# matplotlib without a screen
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv(CSV_PATH)

# only let the code import from the allow-list
_real_import = builtins.__import__
def _guarded_import(name, *a, **k):
    root = name.split(".")[0]
    if root not in ALLOWED:
        raise ImportError("import of '%s' is not allowed in the sandbox" % root)
    return _real_import(name, *a, **k)

# a trimmed builtins: no open/eval/exec/input, no raw __import__ escape
_BLOCKED = {{"open", "eval", "exec", "input", "compile", "__import__", "memoryview"}}
safe_builtins = {{k: getattr(builtins, k) for k in dir(builtins) if k not in _BLOCKED}}
safe_builtins["__import__"] = _guarded_import

ns = {{"__builtins__": safe_builtins, "df": df, "pd": pd, "np": np, "plt": plt, "result": None}}

CODE = json.loads(sys.stdin.read())["code"]

out = {{"ok": False, "result_repr": None, "chart": False, "error": None}}
import io, contextlib
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        exec(CODE, ns)
    result = ns.get("result", None)
    out["result_repr"] = None if result is None else repr(result)[:4000]
    # save a chart if the code drew one
    if plt.get_fignums():
        plt.tight_layout()
        plt.savefig(CHART_PATH, dpi=120, bbox_inches="tight")
        out["chart"] = True
    out["ok"] = True
except Exception as e:
    import traceback
    out["error"] = "".join(traceback.format_exception_only(type(e), e)).strip()
out["stdout"] = buf.getvalue()[:4000]
print("<<<SANDBOX_JSON>>>" + json.dumps(out))
'''


def run_code(code: str, csv_path: str) -> RunResult:
    chart_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    chart_file.close()
    runner_src = _RUNNER.format(
        csv_path=str(csv_path),
        chart_path=chart_file.name,
        allowed=sorted(ALLOWED_IMPORTS),
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as rf:
        rf.write(runner_src)
        runner_path = rf.name

    try:
        proc = subprocess.run(
            [sys.executable, runner_path],
            input=json.dumps({"code": code}),
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return RunResult(False, "", None, None, f"Code ran longer than {TIMEOUT_SECONDS}s and was stopped.")
    finally:
        Path(runner_path).unlink(missing_ok=True)

    marker = "<<<SANDBOX_JSON>>>"
    if marker not in proc.stdout:
        # the process crashed before reporting (e.g. killed, syntax error at parse)
        err = (proc.stderr or proc.stdout or "the sandbox process failed").strip()[:2000]
        return RunResult(False, proc.stdout, None, None, err)

    payload = json.loads(proc.stdout.split(marker, 1)[1].strip())
    chart_path = chart_file.name if payload.get("chart") else None
    if not payload.get("chart"):
        Path(chart_file.name).unlink(missing_ok=True)

    return RunResult(
        ok=payload["ok"],
        stdout=payload.get("stdout", ""),
        result_repr=payload.get("result_repr"),
        chart_path=chart_path,
        error=payload.get("error"),
    )
