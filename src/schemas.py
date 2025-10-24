"""Pydantic models for WhatsApp payloads and internal data structures."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


class VerifyWebhookQuery(BaseModel):
    """Model for WhatsApp webhook verification query parameters."""

    model_config = ConfigDict(populate_by_name=True)

    hub_mode: str = Field(alias="hub.mode")
    hub_verify_token: str = Field(alias="hub.verify_token")
    hub_challenge: str = Field(alias="hub.challenge")


class MessageText(BaseModel):
    """Text content of a WhatsApp message."""

    body: str


class Message(BaseModel):
    """Represents a WhatsApp message entry."""

    model_config = ConfigDict(populate_by_name=True)

    from_: str = Field(alias="from")
    id: Optional[str] = None
    timestamp: Optional[str] = None
    type: str
    text: Optional[MessageText] = None


class Contact(BaseModel):
    """Minimal contact metadata."""

    wa_id: str


class ChangeValue(BaseModel):
    """Change value component from WhatsApp webhook."""

    messaging_product: Optional[str] = None
    contacts: List[Contact] = Field(default_factory=list)
    messages: Optional[List[Message]] = None


class Change(BaseModel):
    """Change element from webhook."""

    field: str
    value: ChangeValue


class Entry(BaseModel):
    """Entry component from webhook."""

    id: Optional[str] = None
    changes: List[Change]


class WhatsAppWebhookPayload(BaseModel):
    """Root payload for WhatsApp webhook events."""

    object: str
    entry: List[Entry]

    def first_text_message(self) -> Optional[Message]:
        """Return the first text message in the payload if present."""
        for entry in self.entry:
            for change in entry.changes:
                if not change.value.messages:
                    continue
                for message in change.value.messages:
                    if message.type == "text" and message.text and message.text.body:
                        return message
        return None

    def sender_wa_id(self) -> Optional[str]:
        """Retrieve the sender WA ID from contacts or message field."""
        for entry in self.entry:
            for change in entry.changes:
                if change.value.contacts:
                    return change.value.contacts[0].wa_id
                if change.value.messages:
                    msg = change.value.messages[0]
                    if msg.from_:
                        return msg.from_
        return None


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
    citations: Optional[List[Dict[str, Any]]] = None
