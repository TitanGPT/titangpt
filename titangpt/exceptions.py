"""
Custom exception classes for TitanGPT.

This module defines all custom exceptions used throughout the TitanGPT application
for more granular error handling and better error reporting.
"""


class TitanGPTException(Exception):
    """Base exception class for all TitanGPT-specific exceptions."""

    pass


class ConfigurationError(TitanGPTException):
    """Raised when there is an error in configuration."""

    pass


class AuthenticationError(TitanGPTException):
    """Raised when authentication fails."""

    pass


class AuthorizationError(TitanGPTException):
    """Raised when user is not authorized to perform an action."""

    pass


class APIError(TitanGPTException):
    """Raised when an API call fails."""

    pass


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""

    pass


class ValidationError(TitanGPTException):
    """Raised when data validation fails."""

    pass


class ModelNotFoundError(TitanGPTException):
    """Raised when a requested model is not found."""

    pass


class PromptError(TitanGPTException):
    """Raised when there is an error with prompt processing."""

    pass


class TimeoutError(TitanGPTException):
    """Raised when an operation times out."""

    pass


class ConnectionError(TitanGPTException):
    """Raised when a connection error occurs."""

    pass


class DataError(TitanGPTException):
    """Raised when there is an error with data processing."""

    pass


class NotImplementedError(TitanGPTException):
    """Raised when a feature is not yet implemented."""

    pass
