# tests/test_history.py
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, Session, create_engine
from sqlmodel.pool import StaticPool

TEST_DATABASE_URL = "sqlite://"

@pytest.fixture(name="engine")
def engine_fixture():
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    yield engine
    SQLModel.metadata.drop_all(engine)

@pytest.fixture(name="client")
def client_fixture(engine):
    from history.database import get_session
    from api.main import app

    def get_test_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = get_test_session
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()

TRANSLATE_PAYLOAD = {
    "user_id":     "user_test_001",
    "action_type": "translate",
    "src_lang":    "francais",
    "tgt_lang":    "moore",
    "source_text": "Bonjour tout le monde",
    "result_text": "Ne y welame",
}

TTS_PAYLOAD = {
    "user_id":     "user_test_001",
    "action_type": "tts",
    "src_lang":    "moore",
    "source_text": "Ne y welame",
    "speed":       1.0,
}

PIPELINE_PAYLOAD = {
    "user_id":     "user_test_001",
    "action_type": "translate_and_speak",
    "src_lang":    "francais",
    "tgt_lang":    "moore",
    "source_text": "Bonjour",
    "result_text": "Ne y welame",
    "speed":       1.0,
}


class TestCreateHistory:
    def test_create_translate_entry(self, client):
        response = client.post("/history", json=TRANSLATE_PAYLOAD)
        assert response.status_code == 201
        data = response.json()
        assert data["action_type"] == "translate"
        assert data["src_lang"]    == "francais"
        assert data["tgt_lang"]    == "moore"
        assert "id" in data
        assert "created_at" in data

    def test_create_tts_entry(self, client):
        response = client.post("/history", json=TTS_PAYLOAD)
        assert response.status_code == 201
        assert response.json()["speed"] == 1.0

    def test_create_pipeline_entry(self, client):
        response = client.post("/history", json=PIPELINE_PAYLOAD)
        assert response.status_code == 201
        assert response.json()["action_type"] == "translate_and_speak"

    def test_create_anonymous_entry(self, client):
        payload = {**TRANSLATE_PAYLOAD, "user_id": None}
        response = client.post("/history", json=payload)
        assert response.status_code == 201
        assert response.json()["user_id"] is None

    def test_create_returns_valid_uuid(self, client):
        response = client.post("/history", json=TRANSLATE_PAYLOAD)
        assert response.status_code == 201
        uuid.UUID(response.json()["id"])

    def test_unknown_src_lang_returns_422(self, client):
        response = client.post("/history", json={**TRANSLATE_PAYLOAD, "src_lang": "klingon"})
        assert response.status_code == 422

    def test_unknown_tgt_lang_returns_422(self, client):
        response = client.post("/history", json={**TRANSLATE_PAYLOAD, "tgt_lang": "martien"})
        assert response.status_code == 422

    def test_missing_source_text_returns_422(self, client):
        response = client.post("/history", json={**TRANSLATE_PAYLOAD, "source_text": None})
        assert response.status_code == 422

    def test_speed_too_high_returns_422(self, client):
        response = client.post("/history", json={**TTS_PAYLOAD, "speed": 5.0})
        assert response.status_code == 422

    def test_speed_too_low_returns_422(self, client):
        response = client.post("/history", json={**TTS_PAYLOAD, "speed": 0.1})
        assert response.status_code == 422

    def test_invalid_action_type_returns_422(self, client):
        response = client.post("/history", json={**TRANSLATE_PAYLOAD, "action_type": "teleportation"})
        assert response.status_code == 422


class TestListHistory:
    def _seed(self, client, n=3):
        return [client.post("/history", json=TRANSLATE_PAYLOAD).json()["id"] for _ in range(n)]

    def test_list_returns_all_entries(self, client):
        self._seed(client, 3)
        data = client.get("/history").json()
        assert data["total"] == 3
        assert len(data["items"]) == 3

    def test_list_pagination_limit(self, client):
        self._seed(client, 3)
        data = client.get("/history?limit=2").json()
        assert data["total"] == 3
        assert len(data["items"]) == 2

    def test_list_pagination_offset(self, client):
        self._seed(client, 3)
        data = client.get("/history?offset=2").json()
        assert len(data["items"]) == 1

    def test_list_filter_by_user_id(self, client):
        client.post("/history", json=TRANSLATE_PAYLOAD)
        client.post("/history", json={**TRANSLATE_PAYLOAD, "user_id": "other_user"})
        data = client.get("/history?user_id=user_test_001").json()
        assert data["total"] == 1
        assert data["items"][0]["user_id"] == "user_test_001"

    def test_list_filter_by_action_type(self, client):
        client.post("/history", json=TRANSLATE_PAYLOAD)
        client.post("/history", json=TTS_PAYLOAD)
        data = client.get("/history?action_type=tts").json()
        assert data["total"] == 1
        assert data["items"][0]["action_type"] == "tts"

    def test_list_filter_by_lang(self, client):
        client.post("/history", json=TRANSLATE_PAYLOAD)
        client.post("/history", json={**TRANSLATE_PAYLOAD, "src_lang": "anglais", "tgt_lang": "dioula"})
        data = client.get("/history?lang=moore").json()
        assert data["total"] == 1

    def test_list_sorted_desc(self, client):
        self._seed(client, 3)
        items = client.get("/history").json()["items"]
        dates = [i["created_at"] for i in items]
        assert dates == sorted(dates, reverse=True)

    def test_list_empty(self, client):
        data = client.get("/history").json()
        assert data["total"] == 0
        assert data["items"] == []


class TestDeleteHistoryEntry:
    def test_delete_returns_204(self, client):
        entry_id = client.post("/history", json=TRANSLATE_PAYLOAD).json()["id"]
        assert client.delete(f"/history/{entry_id}").status_code == 204

    def test_delete_removes_from_list(self, client):
        entry_id = client.post("/history", json=TRANSLATE_PAYLOAD).json()["id"]
        client.delete(f"/history/{entry_id}")
        assert client.get("/history").json()["total"] == 0

    def test_delete_nonexistent_returns_404(self, client):
        assert client.delete(f"/history/{uuid.uuid4()}").status_code == 404

    def test_delete_invalid_uuid_returns_422(self, client):
        assert client.delete("/history/pas-un-uuid").status_code == 422


class TestClearHistory:
    def test_clear_returns_204(self, client):
        client.post("/history", json=TRANSLATE_PAYLOAD)
        assert client.delete("/history?user_id=user_test_001").status_code == 204

    def test_clear_removes_all_user_entries(self, client):
        client.post("/history", json=TRANSLATE_PAYLOAD)
        client.post("/history", json=TRANSLATE_PAYLOAD)
        client.delete("/history?user_id=user_test_001")
        assert client.get("/history?user_id=user_test_001").json()["total"] == 0

    def test_clear_does_not_affect_other_users(self, client):
        client.post("/history", json=TRANSLATE_PAYLOAD)
        client.post("/history", json={**TRANSLATE_PAYLOAD, "user_id": "other_user"})
        client.delete("/history?user_id=user_test_001")
        assert client.get("/history?user_id=other_user").json()["total"] == 1

    def test_clear_idempotent(self, client):
        assert client.delete("/history?user_id=user_inexistant").status_code == 204

    def test_clear_requires_user_id(self, client):
        assert client.delete("/history").status_code == 422