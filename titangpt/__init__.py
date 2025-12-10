"""
TitanGPT Python Client Library

A comprehensive Python client for interacting with the TitanGPT API.
"""

__version__ = "0.1.5"
__author__ = "TitanGPT"
__license__ = "MIT"

from titangpt.client import TitanGPT
from titangpt.async_client import AsyncTitanGPT
from titangpt.exceptions import (
    TitanGPTException,
    APIError,
    AuthenticationError,
    ValidationError,
    RateLimitError,
)

__all__ = [
    "TitanGPT",
    "AsyncTitanGPT",
    "TitanGPTException",
    "APIError",
    "AuthenticationError",
    "ValidationError",
    "RateLimitError",
]

_client = None

def get_client(api_key: str = None, base_url: str = "https://api.titangpt.ru") -> TitanGPT:
    """
    Get or create a TitanGPT client instance.
    """
    global _client
    
    if _client is None:
        _client = TitanGPT(api_key=api_key, base_url=base_url)
    
    return _client


def set_api_key(api_key: str) -> None:
    """
    Set or update the API key for the default client.
    """
    global _client
    _client = TitanGPT(api_key=api_key)
