"""The web API for Ask Your Data.

POST /ask with a question (and optionally a CSV). The agent plans, writes code,
runs it in the sandbox, checks itself, and returns the answer, the code it wrote,
a chart if it made one, and the step-by-step trace. Runs fully locally.
"""
from __future__ import annotations

import base64
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import agent
import llm

app = FastAPI(title="Ask Your Data", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE = Path(__file__).parent / "datasets" / "sales.csv"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "coder_model": llm.CODER_MODEL, "reason_model": llm.REASON_MODEL}


@app.get("/sample")
def sample() -> dict:
    """Describe the bundled dataset so the UI can offer it without an upload."""
    return {"name": "sales.csv", "schema": agent.schema_of(str(SAMPLE))}


@app.post("/ask")
async def ask(
    question: str = Form(...),
    file: UploadFile | None = File(None),
) -> dict:
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Ask a question.")

    tmp_path = None
    try:
        if file is not None:
            raw = await file.read()
            if not raw:
                raise HTTPException(status_code=400, detail="The uploaded file is empty.")
            tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
            tmp.write(raw)
            tmp.close()
            tmp_path = tmp.name
            csv_path = tmp_path
        else:
            csv_path = str(SAMPLE)  # fall back to the sample dataset

        result = agent.analyze(question, csv_path)
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    chart_uri = None
    if result.get("chart_path"):
        data = Path(result["chart_path"]).read_bytes()
        chart_uri = "data:image/png;base64," + base64.b64encode(data).decode("ascii")
        Path(result["chart_path"]).unlink(missing_ok=True)

    return {
        "question": result["question"],
        "answer": result["answer"],
        "success": result["success"],
        "code": result["code"],
        "attempts": result["attempts"],
        "steps": result["steps"],
        "schema": result["schema"],
        "chart_data_uri": chart_uri,
    }
