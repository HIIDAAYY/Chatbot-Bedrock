"""DynamoDB session persistence utilities."""

from __future__ import annotations

import logging
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from botocore.exceptions import ClientError

import config
from schemas import SessionState

logger = logging.getLogger(__name__)


def _session_key(wa_id: str) -> Dict[str, str]:
    """Construct the DynamoDB key for a WhatsApp session."""
    return {"pk": f"user#{wa_id}", "sk": "session"}


def _to_dynamodb(value: Any) -> Any:
    """Convert Python values to DynamoDB compatible formats."""
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {key: _to_dynamodb(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_dynamodb(item) for item in value]
    return value


def _from_dynamodb(value: Any) -> Any:
    """Convert DynamoDB types back to native Python types."""
    if isinstance(value, Decimal):
        return float(value) if value % 1 else int(value)
    if isinstance(value, dict):
        return {key: _from_dynamodb(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_from_dynamodb(item) for item in value]
    return value


class SessionStore:
    """Thin wrapper over DynamoDB table for session management."""

    def __init__(self, ttl_hours: Optional[int] = None):
        settings = config.get_settings()
        self._table = config.get_dynamodb_resource().Table(settings.dynamodb_table)
        self._ttl_hours = ttl_hours

    def get_session(self, wa_id: str) -> Optional[SessionState]:
        """Fetch session data for the provided WhatsApp ID."""
        try:
            response = self._table.get_item(Key=_session_key(wa_id))
        except ClientError as exc:
            logger.error("dynamodb_get_item_error", extra={"error": str(exc), "wa_hash": hash(wa_id)})
            raise

        item = response.get("Item")
        if not item:
            return None

        updated_at_value = item.get("updated_at")

        return SessionState(
            wa_id=wa_id,
            last_intent=item.get("last_intent"),
            last_reply=item.get("last_reply"),
            updated_at=datetime.fromtimestamp(
                int(_from_dynamodb(updated_at_value)), tz=timezone.utc
            )
            if updated_at_value
            else datetime.now(timezone.utc),
            escalation=item.get("escalation", False),
            window_24h=item.get("window_24h", True),
            attributes=_from_dynamodb(item.get("attributes", {})),
        )

    def put_session(self, wa_id: str, data: SessionState) -> None:
        """Persist session state to DynamoDB."""
        ttl = None
        if self._ttl_hours:
            ttl = int(
                (datetime.now(timezone.utc) + timedelta(hours=self._ttl_hours)).timestamp()
            )

        item: Dict[str, Any] = {
            **_session_key(wa_id),
            "last_intent": data.last_intent,
            "last_reply": data.last_reply,
            "updated_at": int(data.updated_at.timestamp()),
            "escalation": data.escalation,
            "window_24h": data.window_24h,
            "attributes": _to_dynamodb(data.attributes),
        }
        if ttl:
            item["ttl"] = ttl

        try:
            self._table.put_item(Item=item)
        except ClientError as exc:
            logger.error("dynamodb_put_item_error", extra={"error": str(exc), "wa_hash": hash(wa_id)})
            raise
