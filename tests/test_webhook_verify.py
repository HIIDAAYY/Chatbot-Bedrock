from typing import Dict

def test_lambda_get_method_not_allowed(app_module):
    event: Dict[str, object] = {
        "requestContext": {"http": {"method": "GET"}},
    }
    response = app_module.lambda_handler(event, None)
    assert response["statusCode"] == 405


def test_lambda_ignores_empty_body(app_module):
    event: Dict[str, object] = {
        "requestContext": {"http": {"method": "POST"}},
        "body": "",
    }
    response = app_module.lambda_handler(event, None)
    assert response["statusCode"] == 200
    assert response["body"] == '{"status": "ignored"}'
