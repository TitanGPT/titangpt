import json
from dataclasses import dataclass
from typing import Any, NoReturn, Optional, Type

from titangpt.exceptions import (
    APIError,
    AuthenticationError,
    AuthorizationError,
    ModelNotFoundError,
    NotFoundError,
    RateLimitError,
    TitanGPTException,
    ValidationError,
)


@dataclass
class ParsedError:
    message: str
    error_type: Optional[str] = None
    param: Any = None
    code: Any = None


def _as_message(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _extract_validation_message(detail: Any) -> Optional[str]:
    if isinstance(detail, list):
        parts = []
        for item in detail:
            if not isinstance(item, dict):
                message = _as_message(item)
                if message:
                    parts.append(message)
                continue

            location = " -> ".join(str(chunk) for chunk in item.get("loc", []))
            message = _as_message(item.get("msg")) or "Validation error"
            parts.append(
                "{0}: {1}".format(location, message) if location else message
            )
        return "; ".join(parts) if parts else None

    return _as_message(detail)


def _extract_gateway_error_message(body: str) -> Optional[str]:
    if not body:
        return None

    normalized = " ".join(body.lower().split())
    if "error 1015" in normalized or "you are being rate limited" in normalized:
        return "Request was rate-limited by Cloudflare or an upstream gateway"
    if "<!doctype html>" in normalized and "access denied" in normalized:
        return "Gateway rejected the request before it reached the API"
    return None


def _parse_mapping(body: dict) -> Optional[ParsedError]:
    error = body.get("error")
    if isinstance(error, dict):
        message = (
            _as_message(error.get("message"))
            or _as_message(error.get("detail"))
            or _as_message(error.get("error"))
        )
        if message:
            return ParsedError(
                message=message,
                error_type=_as_message(error.get("type")),
                param=error.get("param"),
                code=error.get("code"),
            )
    elif error is not None:
        message = _as_message(error)
        if message:
            return ParsedError(message=message)

    detail = body.get("detail")
    if isinstance(detail, dict):
        parsed = _parse_mapping(detail)
        if parsed is not None:
            return parsed

    message = _as_message(body.get("message"))
    if message:
        return ParsedError(
            message=message,
            error_type=_as_message(body.get("type")),
            param=body.get("param"),
            code=body.get("code"),
        )

    detail_message = _extract_validation_message(detail)
    if detail_message:
        return ParsedError(message=detail_message)
    return None


def parse_error_response(
    response_body: Any, raw_text: str, status_code: int
) -> ParsedError:
    parsed = None
    if isinstance(response_body, dict):
        parsed = _parse_mapping(response_body)
    elif isinstance(response_body, list):
        message = _extract_validation_message(response_body)
        if message:
            parsed = ParsedError(message=message)
    elif isinstance(response_body, str):
        message = _extract_gateway_error_message(response_body) or _as_message(
            response_body
        )
        if message:
            parsed = ParsedError(message=message)
    else:
        message = _as_message(response_body)
        if message:
            parsed = ParsedError(message=message)

    if parsed is not None:
        return parsed

    fallback = _extract_gateway_error_message(raw_text)
    if fallback is None and not isinstance(response_body, (dict, list)):
        fallback = _as_message(raw_text)
    return ParsedError(message=fallback or "Error code: {0}".format(status_code))


def raise_api_error(
    *,
    status_code: int,
    response_body: Any,
    raw_text: str,
    request_id: Optional[str] = None,
) -> NoReturn:
    parsed = parse_error_response(response_body, raw_text, status_code)
    exception_type: Type[TitanGPTException]

    if status_code in (400, 422):
        exception_type = ValidationError
    elif status_code == 401:
        exception_type = AuthenticationError
    elif status_code == 403:
        exception_type = AuthorizationError
    elif status_code == 404:
        code = str(parsed.code or "").strip().lower()
        if code == "model_not_found" or "model" in parsed.message.lower():
            exception_type = ModelNotFoundError
        else:
            exception_type = NotFoundError
    elif status_code == 429:
        exception_type = RateLimitError
    else:
        exception_type = APIError

    raise exception_type(
        parsed.message,
        status_code=status_code,
        response_body=response_body,
        request_id=request_id,
        error_type=parsed.error_type,
        param=parsed.param,
        code=parsed.code,
    )
