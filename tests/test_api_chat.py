from fastapi.testclient import TestClient

from RAG_Agent.application.chat_service.chat import ChatService
from RAG_Agent.config import settings
from RAG_Agent.infrastructure.api.main import app


class _FakeRuntime:
    def __init__(self, text: str = "answer from agent") -> None:
        self.text = text
        self.calls: list[dict] = []

    async def run_turn(self, *, user_id: str, session_id: str, message: str) -> str:
        self.calls.append(
            {"user_id": user_id, "session_id": session_id, "message": message}
        )
        return self.text


def _install_fake_runtime(client: TestClient, runtime: _FakeRuntime) -> ChatService:
    sessions = client.app.state.session_service
    app_name = client.app.state.agent_app_name
    service = ChatService(runtime=runtime, sessions=sessions, app_name=app_name)
    client.app.state.chat_service = service
    return service


def test_chat_creates_session_and_returns_message(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test-dummy-key")
    monkeypatch.setattr(settings, "qdrant_enable_sparse", False)
    monkeypatch.setattr(settings, "qdrant_in_memory", True)
    monkeypatch.setattr(settings, "sessions_db_url", None)
    monkeypatch.setattr(settings, "cloudsql_instance", None)

    fake_runtime = _FakeRuntime()
    with TestClient(app) as client:
        _install_fake_runtime(client, fake_runtime)
        response = client.post(
            "/chat",
            json={"message": "What is Near-RT RIC?", "user_id": "u1"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "answer from agent"
    assert body["user_id"] == "u1"
    assert body["session_id"]
    assert fake_runtime.calls[0]["user_id"] == "u1"
    assert fake_runtime.calls[0]["session_id"] == body["session_id"]


def test_reset_memory_recreates_session(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test-dummy-key")
    monkeypatch.setattr(settings, "qdrant_enable_sparse", False)
    monkeypatch.setattr(settings, "qdrant_in_memory", True)
    monkeypatch.setattr(settings, "sessions_db_url", None)
    monkeypatch.setattr(settings, "cloudsql_instance", None)

    with TestClient(app) as client:
        _install_fake_runtime(client, _FakeRuntime("before reset"))
        created = client.post(
            "/chat",
            json={"message": "hi", "user_id": "u2", "session_id": "sess-fixed"},
        )
        assert created.status_code == 200

        _install_fake_runtime(client, _FakeRuntime("after reset"))
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
