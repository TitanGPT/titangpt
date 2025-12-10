__version__ = "0.1.7"
__author__ = "TitanGPT"
__license__ = "MIT"

from titangpt.client import TitanGPT
from titangpt.async_client import AsyncTitanGPT
from titangpt.exceptions import (
    TitanGPTException,
    APIError,
    AuthenticationError,
    AuthorizationError,
    ValidationError,
    RateLimitError,
    ModelNotFoundError,
    TimeoutError,
    ConnectionError
)

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
    "TimeoutError",
    "ConnectionError"
]

_client = None

def get_client(api_key: str = None, base_url: str = "https://api.titangpt.ru") -> TitanGPT:
    global _client
    
    if _client is None:
        _client = TitanGPT(api_key=api_key, base_url=base_url)
    
    return _client

def set_api_key(api_key: str) -> None:
    global _client
    _client = TitanGPT(api_key=api_key)
