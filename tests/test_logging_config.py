from __future__ import annotations

import logging

from RAG_Agent.infrastructure.logging_config import configure_app_logging


def test_configure_app_logging_is_idempotent() -> None:
    configure_app_logging()
    configure_app_logging()
    package = logging.getLogger("RAG_Agent")
    marked = [h for h in package.handlers if getattr(h, "_rag_agent_stderr", False)]
    assert len(marked) == 1
    assert package.level == logging.INFO
    assert package.propagate is False
    assert logging.getLogger("RAG_Agent.infrastructure.api.main_ingest").isEnabledFor(
        logging.INFO
    )
