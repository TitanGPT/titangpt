"""
TitanGPT Python Client Library

A comprehensive Python client for interacting with the TitanGPT API.
"""

__version__ = "0.1.0"
__author__ = "TitanGPT"
__license__ = "MIT"

# Import main client class
from titangpt.client import TitanGPTClient
from titangpt.exceptions import (
    TitanGPTException,
    APIError,
    AuthenticationError,
    ValidationError,
    RateLimitError,
)

# Public API
__all__ = [
    "TitanGPTClient",
    "TitanGPTException",
    "APIError",
    "AuthenticationError",
    "ValidationError",
    "RateLimitError",
]

# Default client instance
_client = None


def get_client(api_key: str = None, base_url: str = None) -> TitanGPTClient:
    """
    Get or create a TitanGPT client instance.
    
    Args:
        api_key: API key for authentication. If not provided, will use TITANGPT_API_KEY env var.
        base_url: Base URL for the API. Defaults to official TitanGPT API endpoint.
    
    Returns:
        TitanGPTClient: Initialized client instance.
    """
    global _client
    
    if _client is None:
        _client = TitanGPTClient(api_key=api_key, base_url=base_url)
    
    return _client


def set_api_key(api_key: str) -> None:
    """
    Set or update the API key for the default client.
    
    Args:
        api_key: The API key to use for authentication.
    """
    global _client
    
    if _client is None:
        _client = TitanGPTClient(api_key=api_key)
    else:
        _client.set_api_key(api_key)
