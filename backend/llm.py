"""Thin wrappers around local Ollama models. All open, nothing leaves the machine."""
from __future__ import annotations

import os

from langchain_ollama import ChatOllama

# a coder model writes the Python, a general model plans and checks the work
CODER_MODEL = os.environ.get("AYD_CODER_MODEL", "qwen2.5-coder:3b-instruct")
REASON_MODEL = os.environ.get("AYD_REASON_MODEL", "llama3")


def coder(temperature: float = 0.0) -> ChatOllama:
    # retries use a higher temperature so the model doesn't repeat the same
    # wrong code over and over.
    return ChatOllama(model=CODER_MODEL, temperature=temperature)


def reasoner() -> ChatOllama:
    return ChatOllama(model=REASON_MODEL, temperature=0.1)


def ask(model: ChatOllama, system: str, user: str) -> str:
    """One-shot system+user call, returns the text."""
    resp = model.invoke([("system", system), ("human", user)])
    return (resp.content or "").strip()
