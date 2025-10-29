"""Twilio WhatsApp API helpers."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict

from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client

import config

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _twilio_client() -> Client:
    secrets = config.get_twilio_secrets()
    return Client(secrets.account_sid, secrets.auth_token)


def send_text(to: str, body: str) -> Dict[str, str]:
    """Send a text message via Twilio WhatsApp."""
    settings = config.get_settings()
    client = _twilio_client()

    # Normalize/validate WhatsApp addressing to avoid 'Invalid From and To pair' errors
    normalized_to = to
    if not normalized_to.startswith("whatsapp:"):
        if normalized_to.startswith("+"):
            normalized_to = f"whatsapp:{normalized_to}"
        else:
            logger.error("twilio_invalid_to_channel", extra={"to_sample": normalized_to[:6]})
            raise ValueError("Twilio To must be in format 'whatsapp:+<country><number>'")

    message_args = {"to": normalized_to, "body": body}
    if settings.twilio_messaging_service_sid:
        message_args["messaging_service_sid"] = settings.twilio_messaging_service_sid
    else:
        from_addr = settings.twilio_whatsapp_from or ""
        if not from_addr.startswith("whatsapp:"):
            logger.error("twilio_invalid_from_channel", extra={"from_sample": from_addr[:9]})
            raise ValueError(
                "Set TWILIO_WHATSAPP_FROM like 'whatsapp:+1...' or use TWILIO_MESSAGING_SERVICE_SID"
            )
        message_args["from_"] = from_addr

    try:
        message = client.messages.create(**message_args)
    except TwilioRestException as exc:
        logger.error(
            "twilio_send_error",
            extra={
                "status": exc.status,
                "code": exc.code,
                "message": str(exc),
                "using_messaging_service": bool(settings.twilio_messaging_service_sid),
                "wa_hash": hash(to),
            },
        )
        raise

    return {"sid": message.sid}
