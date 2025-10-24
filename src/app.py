"""AWS Lambda entry point for the WhatsApp Bedrock chatbot."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from . import guard, nlu, state, whatsapp
from .bedrock_client import BedrockClient
from . import config
from .schemas import GeneratedAnswer, SessionState, VerifyWebhookQuery, WhatsAppWebhookPayload

config.configure_logging()
logger = logging.getLogger(__name__)

DENYLIST_TERMS = ("nomor kartu", "password", "otp", "pin")


def _method_from_event(event: Dict[str, Any]) -> str:
    """Extract HTTP method from API Gateway event."""
    if "requestContext" in event:
        http = event["requestContext"].get("http", {})
        if "method" in http:
            return http["method"]
    return event.get("httpMethod", "")


def _response(body: str, status: int = 200, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    response_headers = {"Content-Type": "text/plain"}
    if headers:
        response_headers.update(headers)
    return {"statusCode": status, "headers": response_headers, "body": body}


def _json_response(body: Dict[str, Any], status: int = 200) -> Dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _get_bedrock_client() -> BedrockClient:
    settings = config.get_settings()
    return BedrockClient(
        region=settings.aws_region,
        model_id=settings.bedrock_model_id,
        kb_id=settings.knowledge_base_id,
        guardrail_id=settings.bedrock_guardrail_id,
        guardrail_ver=settings.bedrock_guardrail_ver,
    )


SESSION_STORE = state.SessionStore(ttl_hours=72)


def lambda_handler(event: Dict[str, Any], _: Any) -> Dict[str, Any]:
    """Entrypoint for AWS Lambda."""
    method = _method_from_event(event)
    logger.debug("incoming_event", extra={"method": method})

    if method == "GET":
        return _handle_verification(event)

    if method == "POST":
        return _handle_message(event)

    return _response("Method Not Allowed", status=405)


def _handle_verification(event: Dict[str, Any]) -> Dict[str, Any]:
    query_params = event.get("queryStringParameters") or {}
    try:
        VerifyWebhookQuery.model_validate(query_params, from_attributes=False)
    except Exception:
        return _response("Invalid verification request", status=400)

    challenge = whatsapp.verify_token(query_params)
    if challenge:
        return _response(challenge, status=200)

    return _response("Forbidden", status=403)


def _parse_payload(event: Dict[str, Any]) -> Optional[WhatsAppWebhookPayload]:
    body = event.get("body")
    if not body:
        return None

    payload_dict = json.loads(body)
    return WhatsAppWebhookPayload.model_validate(payload_dict)


def _handle_message(event: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = _parse_payload(event)
    except Exception as exc:
        logger.error("payload_parse_error", extra={"error": str(exc)})
        return _json_response({"status": "ignored"}, status=200)

    if not payload:
        return _json_response({"status": "ignored"}, status=200)

    message = payload.first_text_message()
    sender = payload.sender_wa_id()

    if not message or not sender:
        logger.info("no_text_message_detected")
        return _json_response({"status": "ignored"}, status=200)

    user_text = message.text.body
    classification = nlu.classify(user_text)
    session = SESSION_STORE.get_session(sender)

    logger.info(
        "message_received",
        extra={
            "wa_hash": hash(sender),
            "intent": classification["intent"],
            "confidence": classification["confidence"],
        },
    )

    reply_text: str
    model_confidence = float(classification["confidence"])
    bedrock_answer: Optional[GeneratedAnswer] = None

    if classification["intent"] == "order_status":
        reply_text = nlu.check_order_status(None)
        model_confidence = 0.9
    elif classification["intent"] == "out_of_scope":
        reply_text = guard.LOW_CONFIDENCE_RESPONSE
    else:
        bedrock_client = _get_bedrock_client()
        if bedrock_client.kb_id:
            bedrock_answer = bedrock_client.answer_with_rag(user_text, session)
        else:
            bedrock_answer = bedrock_client.answer_plain(user_text, session)
        reply_text = bedrock_answer.answer
        model_confidence = min(model_confidence, bedrock_answer.confidence)

    guard_result = guard.apply(reply_text, model_confidence, denylist=DENYLIST_TERMS)

    if classification["intent"] == "out_of_scope":
        guard_result = {"final_text": guard.LOW_CONFIDENCE_RESPONSE, "escalate": True}

    whatsapp.send_text(sender, guard_result["final_text"])

    session_state = SessionState(
        wa_id=sender,
        last_intent=classification["intent"],
        last_reply=guard_result["final_text"],
        escalation=guard_result["escalate"] or classification["intent"] == "out_of_scope",
        attributes={
            "bedrock_confidence": bedrock_answer.confidence if bedrock_answer else model_confidence,
        },
    )
    SESSION_STORE.put_session(sender, session_state)

    return _json_response({"status": "ok"})

