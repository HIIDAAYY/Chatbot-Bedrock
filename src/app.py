"""AWS Lambda entry point for the Twilio WhatsApp Bedrock chatbot."""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

from urllib.parse import parse_qs

import config, guard, nlu, state, whatsapp
from discord_integration import (
    is_discord_request,
    handle_interaction_event,
    process_followup_task,
)
from bedrock_client import BedrockClient
from schemas import GeneratedAnswer, SessionState, TwilioWebhookPayload

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


def _json_response_cors(body: Dict[str, Any], status: int = 200) -> Dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
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


def _decode_body(event: Dict[str, Any]) -> str:
    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    return body


def _parse_json(event: Dict[str, Any]) -> Dict[str, Any]:
    raw = _decode_body(event)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _parse_twilio_payload(event: Dict[str, Any]) -> Tuple[Optional[TwilioWebhookPayload], Dict[str, str]]:
    raw_body = _decode_body(event)
    if not raw_body:
        return None, {}

    parsed = {key: values[0] for key, values in parse_qs(raw_body).items() if values}
    if not parsed:
        return None, {}

    try:
        payload = TwilioWebhookPayload.model_validate(parsed)
    except Exception as exc:
        logger.error("payload_parse_error", extra={"error": str(exc)})
        return None, parsed

    return payload, parsed


def _full_request_url(event: Dict[str, Any]) -> str:
    headers = event.get("headers") or {}
    proto = headers.get("x-forwarded-proto") or headers.get("X-Forwarded-Proto", "https")
    host = headers.get("host") or headers.get("Host", "")
    path = event.get("requestContext", {}).get("http", {}).get("path", "")
    query = event.get("rawQueryString") or ""
    url = f"{proto}://{host}{path}"
    if query:
        url = f"{url}?{query}"
    return url


def _validate_twilio_signature(event: Dict[str, Any], params: Dict[str, str]) -> bool:
    settings = config.get_settings()
    if not settings.twilio_validate_signature:
        return True

    headers = event.get("headers") or {}
    signature = headers.get("x-twilio-signature") or headers.get("X-Twilio-Signature")
    if not signature:
        logger.warning("missing_twilio_signature")
        return False

    validator = config.get_twilio_validator()
    url = _full_request_url(event)
    try:
        return bool(validator.validate(url, params, signature))
    except Exception as exc:
        logger.error("twilio_signature_validation_error", extra={"error": str(exc)})
        return False


SESSION_STORE = state.SessionStore(ttl_hours=72)


def lambda_handler(event: Dict[str, Any], _: Any) -> Dict[str, Any]:
    """Entrypoint for AWS Lambda."""
    method = _method_from_event(event)
    logger.debug("incoming_event", extra={"method": method})

    # Internal async tasks (Discord follow-up worker)
    internal = event.get("internal") if isinstance(event, dict) else None
    if isinstance(internal, dict) and internal.get("type") == "discord_followup":
        process_followup_task(internal)
        return _json_response({"status": "ok"})

    if method != "POST":
        # Provide a tiny test UI for convenience
        path = event.get("requestContext", {}).get("http", {}).get("path", "")
        if path.endswith("/ui") and method in {"GET"}:
            html = (
                "<!doctype html><meta charset='utf-8'><title>Chat Test</title>"
                "<style>body{font-family:sans-serif;max-width:680px;margin:40px auto;}textarea{width:100%;height:100px}pre{background:#111;color:#0f0;padding:12px;white-space:pre-wrap}</style>"
                "<h2>Chat Test</h2><p>Ketik pesan dan kirim ke endpoint JSON.</p>"
                "<textarea id=q placeholder='halo'></textarea><br><button onclick=send()>Kirim</button>"
                "<pre id=o></pre>"
                "<script>async function send(){const r=await fetch('./chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:document.getElementById('q').value})});document.getElementById('o').textContent=await r.text();}</script>"
            )
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "text/html", "Access-Control-Allow-Origin": "*"},
                "body": html,
            }
        return _response("Method Not Allowed", status=405)

    # Route based on headers/path: Discord Interactions vs JSON chat vs Twilio webhook
    headers = event.get("headers") or {}
    path = event.get("requestContext", {}).get("http", {}).get("path", "")
    if path.endswith("/discord") or is_discord_request(headers):
        raw_body = _decode_body(event)
        return handle_interaction_event(event, raw_body)
    if path.endswith("/chat"):
        body = _parse_json(event)
        text = (body.get("text") or body.get("q") or "").strip()
        user_id = (body.get("user") or "webtester").strip() or "webtester"
        if not text:
            return _json_response_cors({"error": "text is required"}, status=400)
        answer = _handle_text_common(text, user_id)
        return _json_response_cors(answer, status=200)

    payload, params = _parse_twilio_payload(event)
    if not payload:
        return _json_response({"status": "ignored"}, status=200)

    if not _validate_twilio_signature(event, params):
        return _response("Forbidden", status=403)

    return _handle_message(payload)


def _handle_message(payload: TwilioWebhookPayload) -> Dict[str, Any]:
    if payload.num_media > 0:
        logger.info("non_text_message_ignored", extra={"wa_hash": hash(payload.wa_id or payload.from_number)})
        return _json_response({"status": "ignored"}, status=200)

    user_text = payload.body
    sender = payload.wa_id or payload.from_number
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

    whatsapp.send_text(payload.from_number, guard_result["final_text"])
    logger.info(
        "twilio_message_sent",
        extra={
            "wa_hash": hash(payload.from_number),
            "escalate": guard_result["escalate"],
        },
    )

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

    return _response("OK")


def _handle_text_common(user_text: str, user_id: str) -> Dict[str, Any]:
    """Shared logic to process plain text (for JSON chat)."""
    classification = nlu.classify(user_text)
    session = SESSION_STORE.get_session(user_id)

    reply_text: str
    model_confidence = float(classification["confidence"]) or 0.5
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

    session_state = SessionState(
        wa_id=user_id,
        last_intent=classification["intent"],
        last_reply=guard_result["final_text"],
        escalation=guard_result["escalate"] or classification["intent"] == "out_of_scope",
        attributes={
            "bedrock_confidence": bedrock_answer.confidence if bedrock_answer else model_confidence,
        },
    )
    SESSION_STORE.put_session(user_id, session_state)

    return {
        "answer": guard_result["final_text"],
        "intent": classification["intent"],
        "escalate": bool(session_state.escalation),
    }
