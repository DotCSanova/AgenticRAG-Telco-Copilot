"""App logging that survives uvicorn and reaches Cloud Run stderr.

Uvicorn only wires ``uvicorn.*`` loggers and typically leaves the root logger
at WARNING. Child loggers under ``RAG_Agent`` then drop INFO. This attaches
one stderr handler to the package logger so the same INFO lines show in a
local terminal and in Cloud Logging.
"""

from __future__ import annotations

import logging
import sys

_PACKAGE = "RAG_Agent"
_HANDLER_MARK = "_rag_agent_stderr"


def configure_app_logging(*, level: int = logging.INFO) -> None:
    """Idempotent: one stderr handler on ``RAG_Agent``, no propagate to root."""
    package = logging.getLogger(_PACKAGE)
    package.setLevel(level)
    package.disabled = False
    package.propagate = False

    if not any(getattr(handler, _HANDLER_MARK, False) for handler in package.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
        setattr(handler, _HANDLER_MARK, True)
        package.addHandler(handler)

    manager = logging.root.manager
    for name in manager.loggerDict:
        if name == _PACKAGE or name.startswith(f"{_PACKAGE}."):
            logging.getLogger(name).disabled = False
