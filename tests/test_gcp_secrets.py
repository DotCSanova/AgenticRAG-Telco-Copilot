from RAG_Agent.config import settings
from RAG_Agent.infrastructure.secrets import gcp_secrets


def test_apply_ingest_secrets_noop_when_flag_false(monkeypatch):
    monkeypatch.setattr(settings, "use_secret_manager", False)
    called = []

    def _boom(*_a, **_k):
        called.append(True)
        raise AssertionError("should not call Secret Manager")

    monkeypatch.setattr(gcp_secrets, "get_secret", _boom)
    gcp_secrets.apply_ingest_secrets_from_secret_manager()
    assert called == []


def test_apply_ingest_secrets_hydrates_settings(monkeypatch):
    monkeypatch.setattr(settings, "use_secret_manager", True)
    monkeypatch.setattr(settings, "cohere_api_key", None)
    monkeypatch.setattr(settings, "qdrant_url", None)
    monkeypatch.setattr(settings, "qdrant_api_key", None)

    secrets = {
        "cohere-api-key": "cohere-secret",
        "qdrant-url": "https://qdrant.example",
        "qdrant-api-key": "q-key",
    }
    monkeypatch.setattr(gcp_secrets, "get_secret", lambda sid: secrets[sid])

    gcp_secrets.apply_ingest_secrets_from_secret_manager()

    assert settings.cohere_api_key == "cohere-secret"
    assert settings.qdrant_url == "https://qdrant.example"
    assert settings.qdrant_api_key == "q-key"


def test_get_secret_strips_quotes(monkeypatch):
    class _Payload:
        data = b'"quoted-value"'

    class _Response:
        payload = _Payload()

    class _Client:
        def access_secret_version(self, request):
            assert "secrets/my-secret/versions/latest" in request["name"]
            return _Response()

    monkeypatch.setattr(settings, "google_cloud_project", "proj-1")

    import sys
    from types import ModuleType

    fake_sm = ModuleType("google.cloud.secretmanager")
    fake_sm.SecretManagerServiceClient = lambda: _Client()  # type: ignore[attr-defined]
    google = ModuleType("google")
    cloud = ModuleType("google.cloud")
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.secretmanager", fake_sm)

    assert gcp_secrets.get_secret("my-secret") == "quoted-value"
