import asyncio
import json
import unittest

import httpx
import requests

from titangpt import AsyncTitanGPT, TitanGPT
from titangpt.exceptions import (
    APIError,
    AuthenticationError,
    AuthorizationError,
    ModelNotFoundError,
    RateLimitError,
    ValidationError,
)


def make_requests_response(status_code, body, headers=None):
    response = requests.Response()
    response.status_code = status_code
    response.headers.update(headers or {})
    if isinstance(body, (dict, list)):
        response._content = json.dumps(body).encode("utf-8")
        response.headers["Content-Type"] = "application/json"
    else:
        response._content = str(body).encode("utf-8")
    return response


class SyncErrorHandlingTests(unittest.TestCase):
    def setUp(self):
        self.client = TitanGPT(api_key="test-key")

    def tearDown(self):
        self.client.close()

    def test_openai_error_fields_are_preserved(self):
        body = {
            "error": {
                "message": "Incorrect API key provided",
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_api_key",
            }
        }
        response = make_requests_response(
            401, body, headers={"x-request-id": "req_auth"}
        )

        with self.assertRaises(AuthenticationError) as raised:
            self.client._handle_error(response)

        error = raised.exception
        self.assertEqual(error.message, "Incorrect API key provided")
        self.assertEqual(error.status_code, 401)
        self.assertEqual(error.request_id, "req_auth")
        self.assertEqual(error.error_type, "invalid_request_error")
        self.assertIsNone(error.param)
        self.assertEqual(error.code, "invalid_api_key")
        self.assertEqual(error.response_body, body)

    def test_nested_fastapi_detail_uses_model_error_code(self):
        body = {
            "detail": {
                "error": {
                    "message": "Resource is unavailable",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "model_not_found",
                }
            }
        }
        response = make_requests_response(404, body)

        with self.assertRaises(ModelNotFoundError) as raised:
            self.client._handle_error(response)

        self.assertEqual(raised.exception.code, "model_not_found")
        self.assertEqual(raised.exception.message, "Resource is unavailable")

    def test_fastapi_validation_list_is_readable(self):
        body = {
            "detail": [
                {
                    "loc": ["body", "model"],
                    "msg": "Field required",
                    "type": "missing",
                }
            ]
        }
        response = make_requests_response(422, body)

        with self.assertRaisesRegex(
            ValidationError, "body -> model: Field required"
        ):
            self.client._handle_error(response)

    def test_string_error_body_does_not_crash_parser(self):
        response = make_requests_response(403, {"error": "Access denied"})

        with self.assertRaisesRegex(AuthorizationError, "Access denied"):
            self.client._handle_error(response)

    def test_unknown_json_body_falls_back_to_status_code(self):
        response = make_requests_response(409, {"unexpected": "shape"})

        with self.assertRaisesRegex(APIError, "Error code: 409") as raised:
            self.client._handle_error(response)

        self.assertEqual(raised.exception.response_body, {"unexpected": "shape"})

    def test_cloudflare_rate_limit_html_has_safe_message(self):
        response = make_requests_response(
            429, "<!doctype html><title>Error 1015</title>You are being rate limited"
        )

        with self.assertRaisesRegex(
            RateLimitError, "rate-limited by Cloudflare"
        ):
            self.client._handle_error(response)

    def test_server_error_preserves_request_id_and_structured_type(self):
        body = {
            "error": {
                "message": "The server had an error processing your request.",
                "type": "server_error",
                "param": None,
                "code": None,
            }
        }
        response = make_requests_response(
            503, body, headers={"x-request-id": "req_server"}
        )

        with self.assertRaises(APIError) as raised:
            self.client._handle_error(response)

        self.assertEqual(raised.exception.request_id, "req_server")
        self.assertEqual(raised.exception.error_type, "server_error")


class AsyncErrorHandlingTests(unittest.TestCase):
    def test_async_client_uses_the_same_parser(self):
        client = AsyncTitanGPT(api_key="test-key")
        response = httpx.Response(
            401,
            json={
                "error": {
                    "message": "Account deactivated",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "account_deactivated",
                }
            },
            headers={"x-request-id": "req_async"},
            request=httpx.Request("GET", "https://api.titangpt.xyz/v1/models"),
        )

        with self.assertRaises(AuthenticationError) as raised:
            asyncio.run(client._handle_error(response))

        self.assertEqual(raised.exception.code, "account_deactivated")
        self.assertEqual(raised.exception.request_id, "req_async")


if __name__ == "__main__":
    unittest.main()
