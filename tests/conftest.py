import os
import sys
from importlib import import_module, reload
from typing import Generator

import boto3
import pytest
from moto import mock_dynamodb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

# Ensure absolute imports like `import config` resolve to the same module as `src.config`
MODULE_ALIASES = {
    "config": "src.config",
    "bedrock_client": "src.bedrock_client",
    "schemas": "src.schemas",
}

for alias, target in MODULE_ALIASES.items():
    if alias not in sys.modules:
        sys.modules[alias] = import_module(target)


REQUIRED_ENV = {
    "APP_ENV": "test",
    "AWS_REGION": "us-east-1",
    "AWS_ACCESS_KEY_ID": "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "AWS_SESSION_TOKEN": "testing",
    "TWILIO_SECRET_NAME": "twilio-bot-secrets",
    "TWILIO_WHATSAPP_FROM": "whatsapp:+14155238886",
    "TWILIO_VALIDATE_SIGNATURE": "false",
    "DDB_TABLE": "test-sessions",
    "BEDROCK_MODEL_ID": "anthropic.claude-3-sonnet-20240229-v1:0",
}


@pytest.fixture(autouse=True)
def _env_vars(monkeypatch) -> None:
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "auth-token")


@pytest.fixture
def aws_mock(monkeypatch) -> Generator[None, None, None]:
    with mock_dynamodb():
        from src import config
        from src.config import get_settings, get_twilio_secrets, get_twilio_validator, _boto_session
        try:
            from src import vector_store
        except ImportError:  # pragma: no cover - optional dependency not loaded
            vector_store = None  # type: ignore

        get_settings.cache_clear()
        get_twilio_secrets.cache_clear()
        get_twilio_validator.cache_clear()
        _boto_session.cache_clear()
        if vector_store is not None:
            vector_store.clear_caches()

        yield

        get_settings.cache_clear()
        get_twilio_secrets.cache_clear()
        get_twilio_validator.cache_clear()
        _boto_session.cache_clear()
        if vector_store is not None:
            vector_store.clear_caches()


@pytest.fixture
def dynamodb_table(aws_mock) -> boto3.resources.base.ServiceResource:
    session = boto3.session.Session(region_name=REQUIRED_ENV["AWS_REGION"])
    dynamodb = session.resource("dynamodb")
    dynamodb.create_table(
        TableName=REQUIRED_ENV["DDB_TABLE"],
        KeySchema=[
            {"AttributeName": "pk", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    return dynamodb


@pytest.fixture
def app_module(monkeypatch, dynamodb_table):
    from src import config, state, app, whatsapp

    monkeypatch.setattr(config, "get_dynamodb_resource", lambda: dynamodb_table)
    whatsapp._twilio_client.cache_clear()

    reload(state)
    reload(app)
    return app
