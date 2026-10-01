"""Agent + schema tests.

schema_of is fast and always runs. The full agent test calls the local models,
so it's slow and only runs when you ask for it:
    AYD_RUN_LLM_TESTS=1 pytest -q
"""
import os
from pathlib import Path

import pytest

import agent

CSV = str(Path(__file__).parent.parent / "datasets" / "sales.csv")


def test_schema_describes_the_columns():
    s = agent.schema_of(CSV)
    assert "600 rows" in s
    assert "region" in s and "revenue" in s


def test_strip_code_fence():
    assert agent._strip_code_fence("```python\nx = 1\n```") == "x = 1"
    assert agent._strip_code_fence("x = 1") == "x = 1"


def test_graph_builds():
    g = agent.build_graph()
    assert g is not None


@pytest.mark.skipif(
    os.environ.get("AYD_RUN_LLM_TESTS") != "1",
    reason="needs Ollama; set AYD_RUN_LLM_TESTS=1 to run",
)
def test_agent_answers_a_real_question():
    out = agent.analyze("Which region has the highest total revenue?", CSV)
    assert out["success"] is True
    assert "west" in out["answer"].lower()
    assert out["code"]
