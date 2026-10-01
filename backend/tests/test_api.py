"""API tests. Validation + health run fast; the full /ask path needs Ollama."""
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_sample_describes_the_dataset():
    r = client.get("/sample")
    assert r.status_code == 200
    assert "region" in r.json()["schema"]


def test_empty_question_is_rejected():
    r = client.post("/ask", data={"question": "   "})
    assert r.status_code == 400


@pytest.mark.skipif(
    os.environ.get("AYD_RUN_LLM_TESTS") != "1",
    reason="needs Ollama; set AYD_RUN_LLM_TESTS=1 to run",
)
def test_ask_end_to_end():
    r = client.post("/ask", data={"question": "How many rows are in the data?"})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert "600" in (body["answer"] + body["code"])
