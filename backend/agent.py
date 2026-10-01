"""The analyst agent: plan, write code, run it, check the result, retry if needed.

Built as a small LangGraph state machine so the loop is explicit:

    plan -> write_code -> run -> check --(retry)--> write_code
                                   \--(good/giveup)--> END

Every node drops a line into `steps` so the UI can show its work.
"""
from __future__ import annotations

import re
from typing import TypedDict

import pandas as pd
from langgraph.graph import StateGraph, END

import llm
import sandbox

MAX_ATTEMPTS = 3


class State(TypedDict, total=False):
    question: str
    csv_path: str
    schema: str
    plan: str
    code: str
    run: sandbox.RunResult
    answer: str
    success: bool
    attempts: int
    fix_hint: str
    steps: list[dict]
    chart_path: str | None


def schema_of(csv_path: str) -> str:
    """A compact description of the data for the model to read."""
    df = pd.read_csv(csv_path)
    lines = [f"{len(df)} rows, {len(df.columns)} columns.", "Columns:"]
    for col in df.columns:
        sample = df[col].dropna().head(3).tolist()
        lines.append(f"  - {col} ({df[col].dtype}) e.g. {sample}")
    return "\n".join(lines)


def _strip_code_fence(text: str) -> str:
    """Models like to wrap code in ```python ... ```. Pull the code back out."""
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def _step(state: State, title: str, detail: str, **extra) -> None:
    state.setdefault("steps", []).append({"title": title, "detail": detail, **extra})


# ---- nodes ----

def plan_node(state: State) -> State:
    system = (
        "You are a careful data analyst. Given a dataset description and a "
        "question, write a SHORT plan (2-3 lines) for how to answer it with "
        "pandas. No code yet, just the approach."
    )
    user = f"Data:\n{state['schema']}\n\nQuestion: {state['question']}"
    plan = llm.ask(llm.reasoner(), system, user)
    _step(state, "Plan", plan)
    return {"plan": plan, "attempts": 0}


def code_node(state: State) -> State:
    retry = state.get("attempts", 0) > 0
    system = (
        "You write Python for a sandbox. A pandas DataFrame named `df` is "
        "already loaded. Rules: use only pandas (pd), numpy (np), and matplotlib "
        "(plt). Assign the answer to a variable named `result`. `result` can be a "
        "number, string, dict, pandas Series, or small DataFrame, whatever holds "
        "the answer. Do NOT try to format it into a sentence, just leave the value "
        "in `result`. If a chart helps, draw it with plt (do NOT call plt.show). "
        "No file reads, no imports of os/sys/network. Output ONLY code, no explanation."
    )
    parts = [f"Data:\n{state['schema']}", f"Question: {state['question']}", f"Plan:\n{state['plan']}"]
    if retry:
        prev = state.get("run")
        parts.append(f"Your previous code:\n{state['code']}")
        parts.append(f"It failed or was wrong. Problem: {state.get('fix_hint') or (prev.error if prev else 'unknown')}")
        parts.append("Write corrected code.")
    # first try is deterministic; retries warm up so it stops repeating the same bug
    temp = 0.0 if not retry else min(0.3 + 0.2 * state.get("attempts", 0), 0.7)
    code = _strip_code_fence(llm.ask(llm.coder(temp), system, "\n\n".join(parts)))
    _step(state, "Write code" + (" (retry)" if retry else ""), code, code=code)
    return {"code": code, "attempts": state.get("attempts", 0) + 1}


def run_node(state: State) -> State:
    result = sandbox.run_code(state["code"], state["csv_path"])
    detail = result.error if not result.ok else (result.result_repr or result.stdout or "(ran, no explicit result)")
    _step(state, "Run in sandbox", detail or "(no output)",
          ok=result.ok, chart=bool(result.chart_path))
    return {"run": result, "chart_path": result.chart_path}


def check_node(state: State) -> State:
    run = state["run"]
    if not run.ok:
        # a hard error always means retry (until we run out of tries)
        _step(state, "Check", f"Code errored: {run.error}")
        return {"success": False, "fix_hint": run.error or "the code raised an error"}

    system = (
        "You verify an analyst's result. You get the question and the result value "
        "the code produced. If the result holds the information needed to answer "
        "the question, reply 'good' and WRITE the answer yourself as one plain "
        "sentence using the numbers in the result. A number, string, dict, pandas "
        "Series, or small table all count as valid results, read the values and "
        "answer. Only reply 'retry' if there is no result, the result is empty, or "
        "it is clearly unrelated to the question. Do not reject a correct result "
        "just because it isn't a sentence. Reply in exactly this shape:\n"
        "VERDICT: good OR retry\n"
        "ANSWER: <one plain sentence answering the question, if good>\n"
        "FIX: <what to change, if retry>"
    )
    user = (
        f"Question: {state['question']}\n\n"
        f"Result the code produced: {run.result_repr}\n"
        f"Printed output: {run.stdout}\n"
        f"A chart was drawn: {bool(run.chart_path)}"
    )
    verdict_text = llm.ask(llm.reasoner(), system, user)
    verdict = "good" if re.search(r"verdict:\s*good", verdict_text, re.I) else "retry"
    answer = _after(verdict_text, "ANSWER:")
    fix = _after(verdict_text, "FIX:")
    _step(state, "Check", verdict_text)
    if verdict == "good":
        return {"success": True, "answer": answer or run.result_repr}
    return {"success": False, "fix_hint": fix or "the result did not answer the question"}


def _after(text: str, label: str) -> str:
    m = re.search(re.escape(label) + r"\s*(.+)", text, re.I)
    return m.group(1).strip() if m else ""


def _route(state: State) -> str:
    if state.get("success"):
        return "done"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "giveup"
    return "retry"


def giveup_node(state: State) -> State:
    run = state.get("run")
    msg = "I couldn't get a clean answer after a few tries."
    if run and run.result_repr:
        msg += f" The closest result was: {run.result_repr}"
    _step(state, "Give up", msg)
    return {"answer": msg, "success": False}


def build_graph():
    g = StateGraph(State)
    g.add_node("plan", plan_node)
    g.add_node("code", code_node)
    g.add_node("run", run_node)
    g.add_node("check", check_node)
    g.add_node("giveup", giveup_node)
    g.set_entry_point("plan")
    g.add_edge("plan", "code")
    g.add_edge("code", "run")
    g.add_edge("run", "check")
    g.add_conditional_edges("check", _route, {"retry": "code", "done": END, "giveup": "giveup"})
    g.add_edge("giveup", END)
    return g.compile()


_graph = None


def analyze(question: str, csv_path: str) -> dict:
    """Run the full agent and return a plain dict for the API."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    schema = schema_of(csv_path)
    final = _graph.invoke(
        {"question": question, "csv_path": csv_path, "schema": schema, "steps": []},
        {"recursion_limit": 25},
    )
    return {
        "question": question,
        "answer": final.get("answer", ""),
        "success": final.get("success", False),
        "code": final.get("code", ""),
        "attempts": final.get("attempts", 0),
        "chart_path": final.get("chart_path"),
        "steps": final.get("steps", []),
        "schema": schema,
    }
