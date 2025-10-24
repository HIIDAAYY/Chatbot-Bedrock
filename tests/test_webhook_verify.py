from typing import Dict

from src.whatsapp import verify_token


def test_verify_token_success():
    params = {
        "hub.mode": "subscribe",
        "hub.verify_token": "verify-me",
        "hub.challenge": "42",
    }
    assert verify_token(params) == "42"


def test_lambda_get_verification(app_module):
    event: Dict[str, object] = {
        "requestContext": {"http": {"method": "GET"}},
        "queryStringParameters": {
            "hub.mode": "subscribe",
            "hub.verify_token": "verify-me",
            "hub.challenge": "4242",
        },
    }

    response = app_module.lambda_handler(event, None)
    assert response["statusCode"] == 200
    assert response["body"] == "4242"
