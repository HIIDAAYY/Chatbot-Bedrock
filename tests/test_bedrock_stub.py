import json

import pytest

from schemas import GeneratedAnswer
from config import get_settings
import config as config_module
import bedrock_client
from bedrock_client import BedrockClient
from src import vector_store


class DummyBody:
    def __init__(self, payload: dict):
        self.payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self.payload


class StubRuntime:
    def __init__(self, payload: dict):
        self.payload = payload
        self.invocations = []

    def invoke_model(self, **kwargs):
        self.invocations.append(kwargs)
        return {"body": DummyBody(self.payload)}


class StubAgentRuntime:
    def __init__(self, response: dict):
        self.response = response
        self.calls = []

    def retrieve_and_generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_answer_plain_extracts_text(monkeypatch):
    payload = {
        "content": [{"type": "text", "text": "Jawaban langsung"}],
        "stop_reason": "end_turn",
    }
    runtime = StubRuntime(payload)

    monkeypatch.setattr(config_module, "get_bedrock_runtime_client", lambda: runtime)
    monkeypatch.setattr(bedrock_client, "config", config_module)
    monkeypatch.setattr(bedrock_client.config, "get_bedrock_runtime_client", lambda: runtime)
    pinecone_stub = lambda question: [{"text": "Promo 50% untuk member", "score": 0.81}]
    monkeypatch.setattr(vector_store, "search_chunks", pinecone_stub)
    monkeypatch.setattr(bedrock_client.vector_store, "search_chunks", pinecone_stub)

    client = BedrockClient(
        region=get_settings().aws_region,
        model_id=get_settings().bedrock_model_id,
    )
    answer = client.answer_plain("Halo", None)

    assert isinstance(answer, GeneratedAnswer)
    assert answer.answer == "Jawaban langsung"
    assert answer.confidence >= 0.7
    assert runtime.invocations
    payload = json.loads(runtime.invocations[0]["body"].decode("utf-8"))
    prompt_text = payload["messages"][0]["content"][0]["text"]
    assert "Promo 50% untuk member" in prompt_text


def test_answer_with_rag_uses_scores(monkeypatch):
    runtime = StubRuntime(
        {"content": [{"type": "text", "text": "Fallback"}], "stop_reason": "end_turn"}
    )
    response = {
        "output": {"text": "Jawaban RAG"},
        "citations": [
            {
                "sources": [
                    {"score": 0.8},
                    {"score": 0.6},
                ]
            }
        ],
    }

    stub_agent = StubAgentRuntime(response)

    monkeypatch.setattr(config_module, "get_bedrock_runtime_client", lambda: runtime)
    monkeypatch.setattr(
        config_module, "get_bedrock_agent_runtime_client", lambda: stub_agent
    )
    monkeypatch.setattr(bedrock_client, "config", config_module)
    monkeypatch.setattr(bedrock_client.config, "get_bedrock_runtime_client", lambda: runtime)
    monkeypatch.setattr(
        bedrock_client.config, "get_bedrock_agent_runtime_client", lambda: stub_agent
    )
    monkeypatch.setenv("KNOWLEDGE_BASE_ID", "kb-123")

    config_module.get_settings.cache_clear()
    config_module.get_twilio_secrets.cache_clear()
    config_module._boto_session.cache_clear()

    client = BedrockClient(
        region=get_settings().aws_region,
        model_id=get_settings().bedrock_model_id,
        kb_id="kb-123",
    )

    answer = client.answer_with_rag("Apa promo?", None)
    assert answer.answer == "Jawaban RAG"
    assert pytest.approx(answer.confidence, rel=0.01) == 0.8

    assert stub_agent.calls, "Retrieve and generate should be invoked"
    prompt_template = (
        stub_agent.calls[0][
            "retrieveAndGenerateConfiguration"
        ]["knowledgeBaseConfiguration"]["generationConfiguration"]["promptTemplate"][
            "textPromptTemplate"
        ]
    )
    assert "$search_results$" in prompt_template

    config_module.get_settings.cache_clear()
    config_module.get_twilio_secrets.cache_clear()
    config_module._boto_session.cache_clear()
