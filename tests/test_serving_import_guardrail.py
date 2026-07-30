"""Guardrails: serving entrypoints must not pull ingest/Docling/torch."""

from __future__ import annotations

import subprocess
import sys


_GUARD_SCRIPT = """
from importlib import import_module
import sys

import_module({module!r})

banned_prefixes = (
    "docling",
    "torch",
    "torchvision",
    "RAG_Agent.infrastructure.ingestion",
    "RAG_Agent.infrastructure.composition.ingest",
    "RAG_Agent.infrastructure.indexing.semantic_chunker",
)
loaded = [
    name
    for name in sys.modules
    if name == "docling"
    or name == "torch"
    or name == "torchvision"
    or any(name == p or name.startswith(p + ".") for p in banned_prefixes)
]
assert not loaded, f"serving import loaded banned modules: {{loaded}}"
"""


def _assert_clean_serving_import(module: str) -> None:
    script = _GUARD_SCRIPT.format(module=module)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_main_chat_does_not_import_ingest_stack():
    _assert_clean_serving_import("RAG_Agent.infrastructure.api.main_chat")


def test_serving_composition_does_not_import_ingest_stack():
    _assert_clean_serving_import("RAG_Agent.infrastructure.composition.serving")
