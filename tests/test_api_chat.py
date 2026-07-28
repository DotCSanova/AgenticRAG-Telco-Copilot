from types import SimpleNamespace

from fastapi.testclient import TestClient

from RAG_Agent.config import settings
from RAG_Agent.infrastructure.api.main import app


class _FakeRunner:
    def __init__(self, text: str = "answer from agent") -> None:
        self.text = text
        self.calls: list[dict] = []

    async def run_async(self, **kwargs):
        self.calls.append(kwargs)
        yield SimpleNamespace(
            is_final_response=lambda: True,
            content=SimpleNamespace(parts=[SimpleNamespace(text=self.text)]),
        )


def test_chat_creates_session_and_returns_message(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test-dummy-key")
    monkeypatch.setattr(settings, "qdrant_enable_sparse", False)
    monkeypatch.setattr(settings, "qdrant_mode", "memory")

    fake_runner = _FakeRunner()
    with TestClient(app) as client:
        client.app.state.runner = fake_runner
        response = client.post(
            "/chat",
            json={"message": "What is Near-RT RIC?", "user_id": "u1"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "answer from agent"
    assert body["user_id"] == "u1"
    assert body["session_id"]
    assert fake_runner.calls[0]["user_id"] == "u1"
    assert fake_runner.calls[0]["session_id"] == body["session_id"]


def test_reset_memory_recreates_session(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test-dummy-key")
    monkeypatch.setattr(settings, "qdrant_enable_sparse", False)
    monkeypatch.setattr(settings, "qdrant_mode", "memory")

    with TestClient(app) as client:
        client.app.state.runner = _FakeRunner("before reset")
        created = client.post(
            "/chat",
            json={"message": "hi", "user_id": "u2", "session_id": "sess-fixed"},
        )
        assert created.status_code == 200

        client.app.state.runner = _FakeRunner("after reset")
        reset = client.post(
            "/reset-memory",
            json={"user_id": "u2", "session_id": "sess-fixed"},
        )
        assert reset.status_code == 200
        assert reset.json()["reset"] is True

        again = client.post(
            "/chat",
            json={"message": "hi again", "user_id": "u2", "session_id": "sess-fixed"},
        )
        assert again.status_code == 200
        assert again.json()["session_id"] == "sess-fixed"
        assert again.json()["message"] == "after reset"
