from types import SimpleNamespace

import pytest

from src import config as config_module
from src import vector_store


class DummyIndex:
    def __init__(self, response):
        self._response = response
        self.queries = []

    def query(self, **kwargs):
        self.queries.append(kwargs)
        return self._response


class DummyClient:
    def __init__(self, *, response, embed_vector):
        self._response = response
        self.embed_vector = embed_vector
        self.embed_calls = []

    def Index(self, name):
        return DummyIndex(self._response)

    class _Inference:
        def __init__(self, parent):
            self._parent = parent

        def embed(self, **kwargs):
            self._parent.embed_calls.append(kwargs)
            return {"data": [{"values": self._parent.embed_vector}]}

    @property
    def inference(self):
        return self._Inference(self)


def _settings(**overrides):
    base = {
        "pinecone_api_key": "key",
        "pinecone_environment": "us-east-1",
        "pinecone_index": "chatbot-bedrock-faq",
        "pinecone_top_k": 3,
        "pinecone_score_threshold": 0.5,
        "pinecone_embedding_model": "intfloat/multilingual-e5-large",
    }
    base.update(overrides)

    class DummySettings(SimpleNamespace):
        @property
        def pinecone_enabled(self) -> bool:  # noqa: D401
            return bool(self.pinecone_api_key and self.pinecone_index)

    return DummySettings(**base)


@pytest.fixture(autouse=True)
def _reset_caches():
    vector_store.clear_caches()
    yield
    vector_store.clear_caches()


def test_search_chunks_filters_by_score(monkeypatch):
    response = {
        "matches": [
            {"score": 0.8, "metadata": {"text": "Promo berlaku sampai akhir bulan."}},
            {"score": 0.3, "metadata": {"text": "Harusnya tidak lolos threshold."}},
        ]
    }

    dummy_client = DummyClient(response=response, embed_vector=[0.1, 0.2, 0.3])

    def fake_client():
        return dummy_client

    fake_client.cache_clear = lambda: None  # type: ignore[attr-defined]

    def fake_index():
        return dummy_client.Index("chatbot-bedrock-faq")

    fake_index.cache_clear = lambda: None  # type: ignore[attr-defined]

    monkeypatch.setattr(vector_store, "_pinecone_client", fake_client)
    monkeypatch.setattr(vector_store, "_pinecone_index", fake_index)
    monkeypatch.setattr(config_module, "get_settings", lambda: _settings())

    results = vector_store.search_chunks("Apa promo terbaru?")

    assert results == [
        {"text": "Promo berlaku sampai akhir bulan.", "score": pytest.approx(0.8)},
    ]


def test_search_chunks_returns_empty_when_disabled(monkeypatch):
    monkeypatch.setattr(config_module, "get_settings", lambda: _settings(pinecone_api_key=None))
    assert vector_store.search_chunks("ada?") == []
