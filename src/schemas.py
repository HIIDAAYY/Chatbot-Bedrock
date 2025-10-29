"""Pydantic models for Twilio WhatsApp payloads and internal data structures."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TwilioWebhookPayload(BaseModel):
    """Simplified view of Twilio WhatsApp webhook payload."""

    model_config = ConfigDict(extra="allow")

    from_number: str = Field(alias="From")
    to_number: Optional[str] = Field(default=None, alias="To")
    wa_id: Optional[str] = Field(default=None, alias="WaId")
    body: str = Field(alias="Body")
    num_media: int = Field(default=0, alias="NumMedia")
    message_sid: Optional[str] = Field(default=None, alias="MessageSid")

    @field_validator("num_media", mode="before")
    @classmethod
    def _parse_num_media(cls, value: Any) -> int:
        if value in (None, ""):
            return 0
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0


class SessionState(BaseModel):
    """Representation of a conversation session persisted in DynamoDB."""

    wa_id: str
    last_intent: Optional[str] = None
    last_reply: Optional[str] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    escalation: bool = False
    window_24h: bool = True
    attributes: Dict[str, Any] = Field(default_factory=dict)


class GeneratedAnswer(BaseModel):
    """Structured model for Bedrock answers."""

    answer: str
    confidence: float
    citations: Optional[list[Dict[str, Any]]] = None
