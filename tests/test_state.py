from datetime import datetime, timezone

from importlib import reload

from src import config as config_module
from src import state as state_module
from src.schemas import SessionState


def test_session_store_roundtrip(monkeypatch, dynamodb_table):
    monkeypatch.setattr(config_module, "get_dynamodb_resource", lambda: dynamodb_table)
    reload(state_module)

    store = state_module.SessionStore(ttl_hours=24)
    wa_id = "628999999999"
    session = SessionState(
        wa_id=wa_id,
        last_intent="faq",
        last_reply="Hai",
        updated_at=datetime.now(timezone.utc),
        escalation=False,
        attributes={"foo": "bar"},
    )

    store.put_session(wa_id, session)
    retrieved = store.get_session(wa_id)

    assert retrieved is not None
    assert retrieved.last_intent == "faq"
    assert retrieved.attributes["foo"] == "bar"
