from titangpt.client import TitanGPT
from titangpt.async_client import AsyncTitanGPT
from titangpt.exceptions import (
    APIError,
    AuthenticationError,
    AuthorizationError,
    ConnectionError,
    DataError,
    ModelNotFoundError,
    NotFoundError,
    RateLimitError,
    TimeoutError,
    TitanGPTException,
    ValidationError,
)

__version__ = "0.2.2"
__author__ = "TitanGPT"
__license__ = "MIT"

__all__ = [
    "TitanGPT",
    "AsyncTitanGPT",
    "TitanGPTException",
    "APIError",
    "AuthenticationError",
    "AuthorizationError",
    "ValidationError",
    "RateLimitError",
    "ModelNotFoundError",
    "NotFoundError",
    "TimeoutError",
    "ConnectionError",
    "DataError",
]

_client = None

def get_client(
    api_key: str = None,
    base_url: str = "https://api.titangpt.ru",
    timeout: int = 60,
    max_retries: int = 3,
) -> TitanGPT:
    global _client

    if (
        _client is None
        or _client.base_url != base_url.rstrip("/")
        or (api_key and _client.api_key != api_key)
    ):
        _client = TitanGPT(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    return _client

def set_api_key(api_key: str) -> None:
    global _client
    _client = TitanGPT(api_key=api_key)
