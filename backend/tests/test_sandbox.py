"""Sandbox tests. Fast, no model needed."""
from pathlib import Path

import sandbox

CSV = Path(__file__).parent.parent / "datasets" / "sales.csv"


def test_normal_code_runs_and_returns_result():
    r = sandbox.run_code("result = int(df.shape[0])", str(CSV))
    assert r.ok
    assert r.result_repr == "600"
    assert r.error is None


def test_code_can_draw_a_chart():
    code = "df['region'].value_counts().plot(kind='bar')\nresult = 'done'"
    r = sandbox.run_code(code, str(CSV))
    assert r.ok
    assert r.chart_path is not None
    assert Path(r.chart_path).exists()


def test_blocked_import_is_refused():
    r = sandbox.run_code("import os\nresult = os.getcwd()", str(CSV))
    assert not r.ok
    assert "not allowed" in (r.error or "")


def test_open_is_not_available():
    r = sandbox.run_code("result = open('/etc/passwd').read()", str(CSV))
    assert not r.ok


def test_infinite_loop_times_out():
    r = sandbox.run_code("while True:\n    pass", str(CSV))
    assert not r.ok
    assert "longer than" in (r.error or "")


def test_runtime_error_is_captured_not_raised():
    r = sandbox.run_code("result = df['nope_missing_column']", str(CSV))
    assert not r.ok
    assert "KeyError" in (r.error or "") or "nope_missing_column" in (r.error or "")
