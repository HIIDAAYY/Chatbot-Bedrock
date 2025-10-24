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


@dataclass(frozen=True)
class Settings:
    """In-memory representation of runtime settings."""

    app_env: str
    aws_region: str
    whatsapp_graph_base: str
    whatsapp_graph_version: str
    whatsapp_phone_number_id: str
    whatsapp_secret_name: str
    dynamodb_table: str
    bedrock_model_id: Optional[str]
    knowledge_base_id: Optional[str]
    bedrock_guardrail_id: Optional[str]
    bedrock_guardrail_ver: Optional[int]
    log_level: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache application settings from the environment."""
    bedrock_guardrail_ver_raw = os.getenv("BEDROCK_GUARDRAIL_VER")
    guardrail_version = int(bedrock_guardrail_ver_raw) if bedrock_guardrail_ver_raw else None

    settings = Settings(
        app_env=os.getenv("APP_ENV", "dev"),
        aws_region=_require_env("AWS_REGION"),
        whatsapp_graph_base=_require_env("WHATSAPP_GRAPH_BASE"),
        whatsapp_graph_version=_require_env("WHATSAPP_GRAPH_VERSION"),
        whatsapp_phone_number_id=_require_env("WHATSAPP_PHONE_NUMBER_ID"),
        whatsapp_secret_name=_require_env("WHATSAPP_SECRET_NAME"),
        dynamodb_table=_require_env("DDB_TABLE"),
        bedrock_model_id=os.getenv("BEDROCK_MODEL_ID"),
        knowledge_base_id=os.getenv("KNOWLEDGE_BASE_ID"),
        bedrock_guardrail_id=os.getenv("BEDROCK_GUARDRAIL_ID"),
        bedrock_guardrail_ver=guardrail_version,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
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


class WhatsAppSecrets(Dict[str, str]):
    """Typed dictionary for WhatsApp secret payloads."""

    access_token_key = "WHATSAPP_ACCESS_TOKEN"
    verify_token_key = "VERIFY_TOKEN"

    @property
    def access_token(self) -> str:
        return self[self.access_token_key]

    @property
    def verify_token(self) -> str:
        return self[self.verify_token_key]


@lru_cache(maxsize=1)
def get_whatsapp_secrets() -> WhatsAppSecrets:
    """
    Fetch WhatsApp secrets from AWS Secrets Manager.

    Environment variable overrides (useful for local development) take precedence when present.
    """
    access_token = os.getenv("WHATSAPP_ACCESS_TOKEN")
    verify_token = os.getenv("VERIFY_TOKEN")
    if access_token and verify_token:
        logger.debug("Using WhatsApp secrets from environment overrides")
        return WhatsAppSecrets(
            {
                WhatsAppSecrets.access_token_key: access_token,
                WhatsAppSecrets.verify_token_key: verify_token,
            }
        )

    settings = get_settings()
    client = get_secrets_manager_client()
    try:
        response = client.get_secret_value(SecretId=settings.whatsapp_secret_name)
    except ClientError as exc:
        raise ConfigurationError("Failed to fetch WhatsApp secrets") from exc

    secret_string = response.get("SecretString")
    if not secret_string:
        raise ConfigurationError("Secrets Manager returned empty WhatsApp secret")

    try:
        payload = json.loads(secret_string)
    except json.JSONDecodeError as exc:
        raise ConfigurationError("Secrets Manager WhatsApp secret is not valid JSON") from exc

    missing = [
        key
        for key in (WhatsAppSecrets.access_token_key, WhatsAppSecrets.verify_token_key)
        if key not in payload
    ]
    if missing:
        raise ConfigurationError(f"WhatsApp secret missing keys: {', '.join(missing)}")

    return WhatsAppSecrets({k: payload[k] for k in payload})


def configure_logging():
    """Configure the root logger once using settings from the environment."""
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

