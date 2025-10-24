"""WhatsApp Cloud API helpers."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import requests
from requests import Response

from . import config

logger = logging.getLogger(__name__)


def verify_token(query_params: Dict[str, str]) -> Optional[str]:
    """Validate incoming webhook verification request and return the challenge."""
    secrets = config.get_whatsapp_secrets()
    mode = query_params.get("hub.mode")
    verify_token_value = query_params.get("hub.verify_token")
    challenge = query_params.get("hub.challenge")

    if mode == "subscribe" and verify_token_value == secrets.verify_token:
        return challenge
    return None


def _build_headers(access_token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


def _http_post_with_retry(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Response:
    attempts = 0
    backoff = 0.5
    while True:
        attempts += 1
        response = requests.post(url, headers=headers, json=payload, timeout=5)
        if response.status_code < 500 or attempts >= 3:
            return response
        time.sleep(backoff)
        backoff *= 2


def send_text(to: str, body: str) -> Dict[str, Any]:
    """Send a text message to WhatsApp."""
    secrets = config.get_whatsapp_secrets()
    settings = config.get_settings()

    url = (
        f"{settings.whatsapp_graph_base}/{settings.whatsapp_graph_version}/"
        f"{settings.whatsapp_phone_number_id}/messages"
    )

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": body},
    }

    response = _http_post_with_retry(url, _build_headers(secrets.access_token), payload)

    if response.status_code >= 400:
        logger.error(
            "whatsapp_send_error",
            extra={
                "status": response.status_code,
                "wa_hash": hash(to),
                "error": response.text[:200],
            },
        )
        response.raise_for_status()

    return response.json() if response.content else {"status": response.status_code}

