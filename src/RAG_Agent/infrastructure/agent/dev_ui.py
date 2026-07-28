"""ADK Dev UI entrypoint (separate from the product FastAPI API).

Run locally::

    uv run uvicorn RAG_Agent.infrastructure.agent.dev_ui:app --host 0.0.0.0 --port 8080

Then open http://127.0.0.1:8080
"""

from __future__ import annotations

from pathlib import Path

from google.adk.cli.fast_api import get_fast_api_app

_AGENTS_DIR = str((Path(__file__).resolve().parent / "dev_agents"))

app = get_fast_api_app(
    agents_dir=_AGENTS_DIR,
    web=True,
    auto_create_session=True,
    allow_origins=["*"],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
