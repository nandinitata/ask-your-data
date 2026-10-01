"""Measure how often the agent gets the answer right.

Each question has a ground-truth value computed straight from the data with
pandas. The agent runs, and we check whether that value shows up in its answer
or the code's result. Prints a pass/fail table and the overall accuracy.

Run (needs Ollama running):
    python eval/evaluate.py
    python eval/evaluate.py --csv datasets/sales.csv --runs 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent  # noqa: E402

# Each case: a question, plus a function that returns the string(s) that must
# appear in a correct answer, computed from the dataframe itself.
CASES = [
    ("How many rows are in the dataset?",
     lambda df: [str(len(df))]),
    ("How many unique regions are there?",
     lambda df: [str(df["region"].nunique())]),
    ("Which region has the highest total revenue?",
     lambda df: [df.groupby("region")["revenue"].sum().idxmax().lower()]),
    ("Which product category has the highest average unit price?",
     lambda df: [df.groupby("category")["unit_price"].mean().idxmax().lower()]),
    ("How many unique product categories are there?",
     lambda df: [str(df["category"].nunique())]),
    ("Which region appears in the most rows?",
     lambda df: [df["region"].value_counts().idxmax().lower()]),
]


def _hit(expected: list[str], text: str) -> bool:
    text = text.lower()
    return all(e.lower() in text for e in expected)


def run(csv_path: str, runs: int):
    df = pd.read_csv(csv_path)
    print(f"Evaluating {len(CASES)} questions x {runs} run(s) on {Path(csv_path).name}\n")
    print(f"{'#':>2} {'ok':>4} {'tries':>5}  question")
    print("-" * 68)

    total = 0
    passed = 0
    for i, (question, truth_fn) in enumerate(CASES, 1):
        expected = truth_fn(df)
        for _ in range(runs):
            total += 1
            out = agent.analyze(question, csv_path)
            haystack = " ".join([
                out.get("answer", ""),
                out.get("code", ""),
                next((s["detail"] for s in reversed(out["steps"]) if s["title"].startswith("Run")), ""),
            ])
            ok = _hit(expected, haystack)
            passed += int(ok)
            mark = "PASS" if ok else "FAIL"
            print(f"{i:>2} {mark:>4} {out['attempts']:>5}  {question}")
            if not ok:
                print(f"       expected {expected}, got: {out.get('answer','')[:80]}")

    print("-" * 68)
    print(f"accuracy: {passed}/{total} = {passed / total:.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Path(__file__).parent.parent / "datasets" / "sales.csv"))
    ap.add_argument("--runs", type=int, default=1, help="runs per question (the models aren't fully deterministic)")
    args = ap.parse_args()
    run(args.csv, args.runs)


if __name__ == "__main__":
    main()
