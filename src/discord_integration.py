"""Discord Interactions (slash commands) integration for AWS Lambda.

This module verifies Discord signatures and helps handle slash-command
interactions in a serverless-friendly way. We immediately ACK the
interaction (type 5) and invoke an asynchronous follow-up task (the
same Lambda) to generate the Bedrock answer and post it back using the
interaction webhook URL.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import boto3
import requests
from nacl.signing import VerifyKey
from nacl.exceptions import BadSignatureError

import config
import guard
import nlu
from bedrock_client import BedrockClient
from schemas import GeneratedAnswer, SessionState
import state

logger = logging.getLogger(__name__)


def _json_response(body: Dict[str, Any], status: int = 200) -> Dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def is_discord_request(headers: Dict[str, str]) -> bool:
    """Heuristic: Discord Interactions include these headers."""
    if not headers:
        return False
    return (
        "x-signature-ed25519" in headers
        or "X-Signature-Ed25519" in headers
        or "x-signature-timestamp" in headers
        or "X-Signature-Timestamp" in headers
    )


def _verify_discord_signature(headers: Dict[str, str], raw_body: str) -> bool:
    settings = config.get_settings()
    if not settings.discord_validate_signature:
        return True

    public_key = settings.discord_public_key
    if not public_key:
        logger.error("discord_missing_public_key")
        return False

    sig = headers.get("x-signature-ed25519") or headers.get("X-Signature-Ed25519")
    ts = headers.get("x-signature-timestamp") or headers.get("X-Signature-Timestamp")
    if not sig or not ts:
        logger.warning("discord_missing_signature_headers")
        return False

    try:
        verify_key = VerifyKey(bytes.fromhex(public_key))
        verify_key.verify(f"{ts}{raw_body}".encode(), bytes.fromhex(sig))
        return True
    except BadSignatureError:
        logger.warning("discord_signature_invalid")
        return False
    except Exception as exc:
        logger.error("discord_signature_error", extra={"error": str(exc)})
        return False


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


def handle_interaction_event(event: Dict[str, Any], raw_body: str) -> Dict[str, Any]:
    """Handle the initial Interactions POST from Discord."""
    headers = event.get("headers") or {}
    if not _verify_discord_signature(headers, raw_body):
        return {"statusCode": 401, "body": "invalid request signature"}

    try:
        payload = json.loads(raw_body)
    except Exception:
        return {"statusCode": 400, "body": "invalid json"}

    t = payload.get("type")
    # PING -> respond PONG
    if t == 1:
        return _json_response({"type": 1})

    if t != 2:
        # Not an application command; ignore politely
        return _json_response({"type": 4, "data": {"content": "Unsupported interaction."}})

    data = payload.get("data", {})
    name = (data.get("name") or "").lower()
    # We expect a slash command named /chat with a string option 'q'
    if name not in {"chat", "ask"}:
        return _json_response({"type": 4, "data": {"content": "Gunakan perintah /chat."}})

    options = data.get("options", []) or []
    question = None
    for opt in options:
        if (opt.get("name") or "").lower() in {"q", "prompt", "text", "pesan"}:
            question = opt.get("value")
            break
    if not question:
        return _json_response({"type": 4, "data": {"content": "Masukkan pertanyaan setelah /chat."}})

    member = payload.get("member") or {}
    user = member.get("user") or payload.get("user") or {}
    user_id = user.get("id") or "anonymous"
    token = payload.get("token")
    app_id = config.get_settings().discord_app_id

    # Fire-and-forget a follow-up worker invocation to generate and send the reply.
    _invoke_followup_worker(
        question=question,
        user_id=user_id,
        interaction_token=token,
        application_id=app_id,
    )

    # Deferred response so Discord doesn't time out (3s limit)
    return _json_response({"type": 5})


def _invoke_followup_worker(*, question: str, user_id: str, interaction_token: str, application_id: Optional[str]) -> None:
    """Invoke the same Lambda asynchronously to do the heavy lifting."""
    try:
        function_name = os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
        if not function_name:
            logger.error("missing_function_name_for_worker")
            return

        payload = {
            "internal": {
                "type": "discord_followup",
                "question": question,
                "user_id": user_id,
                "interaction_token": interaction_token,
                "application_id": application_id,
            }
        }
        client = boto3.client("lambda")
        client.invoke(
            FunctionName=function_name,
            InvocationType="Event",
            Payload=json.dumps(payload).encode("utf-8"),
        )
    except Exception as exc:
        logger.error("invoke_worker_error", extra={"error": str(exc)})


def process_followup_task(task: Dict[str, Any]) -> None:
    """Generate the answer and post via Discord webhook."""
    question: str = task.get("question", "")
    user_id: str = task.get("user_id", "user")
    token: str = task.get("interaction_token", "")
    application_id: Optional[str] = task.get("application_id")
    if not token or not application_id:
        logger.error("discord_missing_followup_identifiers")
        return

    # Generate reply using existing Bedrock flow + guardrails
    classification = nlu.classify(question)
    session = SESSION_STORE.get_session(user_id)

    reply_text: str
    confidence: float = float(classification["confidence"]) or 0.5
    answer: Optional[GeneratedAnswer] = None

    try:
        if classification["intent"] == "order_status":
            reply_text = nlu.check_order_status(None)
            confidence = 0.9
        elif classification["intent"] == "out_of_scope":
            reply_text = guard.LOW_CONFIDENCE_RESPONSE
        else:
            client = _get_bedrock_client()
            if client.kb_id:
                answer = client.answer_with_rag(question, session)
            else:
                answer = client.answer_plain(question, session)
            reply_text = answer.answer
            confidence = min(confidence, answer.confidence)

        guard_result = guard.apply(reply_text, confidence)
        if classification["intent"] == "out_of_scope":
            guard_result = {"final_text": guard.LOW_CONFIDENCE_RESPONSE, "escalate": True}

        # Persist session
        session_state = SessionState(
            wa_id=user_id,
            last_intent=classification["intent"],
            last_reply=guard_result["final_text"],
            escalation=guard_result["escalate"] or classification["intent"] == "out_of_scope",
            attributes={
                "bedrock_confidence": answer.confidence if answer else confidence,
            },
        )
        SESSION_STORE.put_session(user_id, session_state)

        # Post follow-up message to Discord
        url = f"https://discord.com/api/v10/webhooks/{application_id}/{token}"
        resp = requests.post(url, json={"content": guard_result["final_text"]}, timeout=10)
        if resp.status_code >= 300:
            logger.error(
                "discord_followup_post_failed",
                extra={"status": resp.status_code, "body": resp.text[:200]},
            )
    except Exception as exc:
        logger.error("discord_followup_error", extra={"error": str(exc)})

