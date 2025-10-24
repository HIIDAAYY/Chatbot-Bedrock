import json
import os
from typing import Dict

from src import bedrock_client, whatsapp
from src.guard import LOW_CONFIDENCE_RESPONSE
from src.schemas import GeneratedAnswer


def sample_event() -> Dict[str, object]:
    payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "123",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "contacts": [{"wa_id": "628111111111"}],
                            "messages": [
                                {
                                    "from": "628111111111",
                                    "id": "wamid.test",
                                    "timestamp": "1700000000",
                                    "type": "text",
                                    "text": {"body": "Halo, apakah ada promo?"},
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }
    return {
        "requestContext": {"http": {"method": "POST"}},
        "body": json.dumps(payload),
    }


def test_inbound_text_triggers_bedrock(monkeypatch, app_module, dynamodb_table):
    bedrock_call = {}

    def mock_answer_plain(self, question, session_ctx):
        bedrock_call["question"] = question
        bedrock_call["session"] = session_ctx
        return GeneratedAnswer(answer="Promo saat ini tersedia untuk semua pelanggan.", confidence=0.82)

    send_payload = {}

    def mock_send_text(to, body):
        send_payload["to"] = to
        send_payload["body"] = body
        return {"status": 200}

    monkeypatch.setattr(bedrock_client.BedrockClient, "answer_plain", mock_answer_plain)
    monkeypatch.setattr(whatsapp, "send_text", mock_send_text)

    response = app_module.lambda_handler(sample_event(), None)

    assert response["statusCode"] == 200
    assert bedrock_call["question"] == "Halo, apakah ada promo?"
    assert send_payload["to"] == "628111111111"
    assert "Promo saat ini" in send_payload["body"]

    table = dynamodb_table.Table(os.environ["DDB_TABLE"])
    stored = table.get_item(Key={"pk": "user#628111111111", "sk": "session"})
    assert "Item" in stored
    assert stored["Item"]["last_intent"] == "faq"
    assert stored["Item"]["escalation"] is False


def test_out_of_scope_escalates(monkeypatch, app_module, dynamodb_table):
    def mock_answer_plain(self, question, session_ctx):
        return GeneratedAnswer(answer="Tidak tahu", confidence=0.2)

    send_payload = {}

    def mock_send_text(to, body):
        send_payload["body"] = body
        return {"status": 200}

    monkeypatch.setattr(bedrock_client.BedrockClient, "answer_plain", mock_answer_plain)
    monkeypatch.setattr(whatsapp, "send_text", mock_send_text)

    event = sample_event()
    body = json.loads(event["body"])
    body["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"] = "Bisa kirim password akun saya?"
    event["body"] = json.dumps(body)

    response = app_module.lambda_handler(event, None)

    assert response["statusCode"] == 200
    assert send_payload["body"] == LOW_CONFIDENCE_RESPONSE

    table = dynamodb_table.Table(os.environ["DDB_TABLE"])
    stored = table.get_item(Key={"pk": "user#628111111111", "sk": "session"})
    assert stored["Item"]["escalation"] is True
