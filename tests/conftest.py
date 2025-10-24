import os
import sys
from importlib import reload
from typing import Generator

import boto3
import pytest
from moto import mock_dynamodb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


REQUIRED_ENV = {
    "APP_ENV": "test",
    "AWS_REGION": "us-east-1",
    "AWS_ACCESS_KEY_ID": "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "AWS_SESSION_TOKEN": "testing",
    "WHATSAPP_GRAPH_BASE": "https://graph.facebook.com",
    "WHATSAPP_GRAPH_VERSION": "v20.0",
    "WHATSAPP_PHONE_NUMBER_ID": "123456",
    "WHATSAPP_SECRET_NAME": "wa-bot-secrets",
    "DDB_TABLE": "test-sessions",
    "BEDROCK_MODEL_ID": "anthropic.claude-3-sonnet-20240229-v1:0",
}


@pytest.fixture(autouse=True)
def _env_vars(monkeypatch) -> None:
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "local-token")
    monkeypatch.setenv("VERIFY_TOKEN", "verify-me")


@pytest.fixture
def aws_mock(monkeypatch) -> Generator[None, None, None]:
    with mock_dynamodb():
        from src import config
        from src.config import get_settings, get_whatsapp_secrets, _boto_session

        get_settings.cache_clear()
        get_whatsapp_secrets.cache_clear()
        _boto_session.cache_clear()

        yield

        get_settings.cache_clear()
        get_whatsapp_secrets.cache_clear()
        _boto_session.cache_clear()


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
    from src import config, state, app

    monkeypatch.setattr(config, "get_dynamodb_resource", lambda: dynamodb_table)

    reload(state)
    reload(app)
    return app
