"""Application configuration and secrets loading."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _require_env(name: str, *, default: Optional[str] = None) -> str:
    """Fetch an environment variable or raise if it is missing."""
    value = os.getenv(name, default)
    if value is None or value == "":
        raise ConfigurationError(f"Environment variable {name} is required")
    return value


def _bool_env(name: str, default: bool = True) -> bool:
    """Parse boolean environment variable values."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y"}


@dataclass(frozen=True)
class Settings:
    """In-memory representation of runtime settings."""

    app_env: str
    aws_region: str
    twilio_secret_name: str
    twilio_whatsapp_from: Optional[str]
    twilio_messaging_service_sid: Optional[str]
    twilio_validate_signature: bool
    dynamodb_table: str
    bedrock_model_id: Optional[str]
    knowledge_base_id: Optional[str]
    bedrock_guardrail_id: Optional[str]
    bedrock_guardrail_ver: Optional[int]
    bedrock_inference_profile_arn: Optional[str]
    log_level: str
    # Discord (optional for testing via slash commands)
    discord_public_key: Optional[str]
    discord_app_id: Optional[str]
    discord_validate_signature: bool
    # Inline FAQ (optional fallback if Knowledge Base is unavailable)
    faq_inline_path: Optional[str]
    faq_inline_s3_uri: Optional[str]
    faq_inline_max_chars: int
    # Pinecone vector store (optional RAG fallback)
    pinecone_api_key: Optional[str]
    pinecone_environment: Optional[str]
    pinecone_index: Optional[str]
    pinecone_top_k: int
    pinecone_score_threshold: float
    pinecone_embedding_model: Optional[str]

    @property
    def pinecone_enabled(self) -> bool:
        return bool(self.pinecone_api_key and self.pinecone_index)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache application settings from the environment."""
    bedrock_guardrail_ver_raw = os.getenv("BEDROCK_GUARDRAIL_VER")
    guardrail_version = int(bedrock_guardrail_ver_raw) if bedrock_guardrail_ver_raw else None

    pinecone_top_k_raw = os.getenv("PINECONE_TOP_K", "3")
    pinecone_score_threshold_raw = os.getenv("PINECONE_SCORE_THRESHOLD", "0.6")
    try:
        pinecone_top_k = max(1, int(pinecone_top_k_raw))
    except (TypeError, ValueError):
        pinecone_top_k = 3
    try:
        pinecone_score_threshold = float(pinecone_score_threshold_raw)
    except (TypeError, ValueError):
        pinecone_score_threshold = 0.0

    settings = Settings(
        app_env=os.getenv("APP_ENV", "dev"),
        aws_region=_require_env("AWS_REGION"),
        twilio_secret_name=_require_env("TWILIO_SECRET_NAME"),
        twilio_whatsapp_from=os.getenv("TWILIO_WHATSAPP_FROM"),
        twilio_messaging_service_sid=os.getenv("TWILIO_MESSAGING_SERVICE_SID"),
        twilio_validate_signature=_bool_env("TWILIO_VALIDATE_SIGNATURE", default=True),
        dynamodb_table=_require_env("DDB_TABLE"),
        bedrock_model_id=os.getenv("BEDROCK_MODEL_ID"),
        knowledge_base_id=os.getenv("KNOWLEDGE_BASE_ID"),
        bedrock_guardrail_id=os.getenv("BEDROCK_GUARDRAIL_ID"),
        bedrock_guardrail_ver=guardrail_version,
        bedrock_inference_profile_arn=os.getenv("BEDROCK_INFERENCE_PROFILE_ARN"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        discord_public_key=os.getenv("DISCORD_PUBLIC_KEY"),
        discord_app_id=os.getenv("DISCORD_APP_ID"),
        discord_validate_signature=_bool_env("DISCORD_VALIDATE_SIGNATURE", default=True),
        faq_inline_path=os.getenv("FAQ_INLINE_PATH"),
        faq_inline_s3_uri=os.getenv("FAQ_INLINE_S3_URI"),
        faq_inline_max_chars=int(os.getenv("FAQ_INLINE_MAX_CHARS", "18000")),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_environment=os.getenv("PINECONE_ENV"),
        pinecone_index=os.getenv("PINECONE_INDEX"),
        pinecone_top_k=pinecone_top_k,
        pinecone_score_threshold=pinecone_score_threshold,
        pinecone_embedding_model=os.getenv("PINECONE_EMBEDDING_MODEL"),
    )

    if not settings.twilio_whatsapp_from and not settings.twilio_messaging_service_sid:
        raise ConfigurationError(
            "Set TWILIO_WHATSAPP_FROM or TWILIO_MESSAGING_SERVICE_SID for outbound messages"
        )

    return settings


@lru_cache(maxsize=1)
def _boto_session():
    """Create and cache a boto3 session bound to the configured region."""
    settings = get_settings()
    return boto3.session.Session(region_name=settings.aws_region)


def get_bedrock_runtime_client():
    """Return a boto3 Bedrock runtime client."""
    return _boto_session().client("bedrock-runtime")


def get_bedrock_agent_runtime_client():
    """Return a boto3 Bedrock Agent runtime client for Knowledge Bases."""
    return _boto_session().client("bedrock-agent-runtime")


def get_dynamodb_resource():
    """Return a boto3 DynamoDB resource."""
    return _boto_session().resource("dynamodb")


def get_secrets_manager_client():
    """Return a boto3 Secrets Manager client."""
    return _boto_session().client("secretsmanager")


def get_s3_client():
    """Return a boto3 S3 client."""
    return _boto_session().client("s3")


class TwilioSecrets(Dict[str, str]):
    """Typed dictionary for Twilio secret payloads."""

    account_sid_key = "TWILIO_ACCOUNT_SID"
    auth_token_key = "TWILIO_AUTH_TOKEN"

    @property
    def account_sid(self) -> str:
        return self[self.account_sid_key]

    @property
    def auth_token(self) -> str:
        return self[self.auth_token_key]


@lru_cache(maxsize=1)
def get_twilio_secrets() -> TwilioSecrets:
    """
    Fetch Twilio credentials from AWS Secrets Manager.

    Environment variable overrides (useful for local development) take precedence when present.
    """
    account_sid = os.getenv(TwilioSecrets.account_sid_key)
    auth_token = os.getenv(TwilioSecrets.auth_token_key)
    if account_sid and auth_token:
        logger.debug("Using Twilio secrets from environment overrides")
        return TwilioSecrets(
            {
                TwilioSecrets.account_sid_key: account_sid,
                TwilioSecrets.auth_token_key: auth_token,
            }
        )

    settings = get_settings()
    client = get_secrets_manager_client()
    try:
        response = client.get_secret_value(SecretId=settings.twilio_secret_name)
    except ClientError as exc:
        raise ConfigurationError("Failed to fetch Twilio secrets") from exc

    secret_string = response.get("SecretString")
    if not secret_string:
        raise ConfigurationError("Secrets Manager returned empty Twilio secret")

    try:
        payload = json.loads(secret_string)
    except json.JSONDecodeError as exc:
        raise ConfigurationError("Secrets Manager Twilio secret is not valid JSON") from exc

    missing = [
        key
        for key in (TwilioSecrets.account_sid_key, TwilioSecrets.auth_token_key)
        if key not in payload
    ]
    if missing:
        raise ConfigurationError(f"Twilio secret missing keys: {', '.join(missing)}")

    return TwilioSecrets({k: payload[k] for k in payload})


def configure_logging():
    """Configure the root logger once using settings from the environment."""
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@lru_cache(maxsize=1)
def get_twilio_validator():
    """Return a Twilio request validator initialized with the auth token."""
    from twilio.request_validator import RequestValidator

    secrets = get_twilio_secrets()
    return RequestValidator(secrets.auth_token)
