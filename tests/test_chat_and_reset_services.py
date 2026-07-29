import asyncio

import pytest

from RAG_Agent.application.chat_service.chat import ChatService
from RAG_Agent.application.reset_memory_service.reset_memory import ResetMemoryService
from RAG_Agent.domain.exceptions import AgentEmptyResponseError


class _Session:
    def __init__(self, session_id: str) -> None:
        self.id = session_id


class _FakeSessions:
    def __init__(self) -> None:
        self.sessions: dict[tuple[str, str, str], _Session] = {}

    async def get_session(self, *, app_name, user_id, session_id):
        return self.sessions.get((app_name, user_id, session_id))

    async def create_session(self, *, app_name, user_id, session_id=None):
        sid = session_id or "auto-id"
        session = _Session(sid)
        self.sessions[(app_name, user_id, sid)] = session
        return session

    async def delete_session(self, *, app_name, user_id, session_id):
        self.sessions.pop((app_name, user_id, session_id), None)


class _FakeRuntime:
    def __init__(self, text: str = "ok") -> None:
        self.text = text
        self.calls: list[dict] = []

    async def run_turn(self, *, user_id, session_id, message):
        self.calls.append(
            {"user_id": user_id, "session_id": session_id, "message": message}
        )
        return self.text


def test_chat_service_reuses_existing_session():
    async def _run():
        sessions = _FakeSessions()
        await sessions.create_session(app_name="app", user_id="u", session_id="s1")
        runtime = _FakeRuntime("hello")
        service = ChatService(runtime=runtime, sessions=sessions, app_name="app")

        result = await service.execute(user_id="u", message="q", session_id="s1")
        assert result.session_id == "s1"
        assert result.message == "hello"
        assert runtime.calls[0]["session_id"] == "s1"

    asyncio.run(_run())


def test_chat_service_rejects_empty_agent_reply():
    async def _run():
        service = ChatService(
            runtime=_FakeRuntime("   "),
            sessions=_FakeSessions(),
            app_name="app",
        )
        with pytest.raises(AgentEmptyResponseError):
            await service.execute(user_id="u", message="q")

    asyncio.run(_run())


def test_reset_memory_deletes_and_recreates():
    async def _run():
        sessions = _FakeSessions()
        await sessions.create_session(app_name="app", user_id="u", session_id="s1")
        service = ResetMemoryService(sessions=sessions, app_name="app")

        result = await service.execute(user_id="u", session_id="s1")
        assert result.reset is True
        assert ("app", "u", "s1") in sessions.sessions

    asyncio.run(_run())
