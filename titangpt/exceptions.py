from typing import Any, Optional


class TitanGPTException(Exception):
    """Base exception for all SDK errors."""

    def __init__(
        self,
        message: str = "",
        *,
        status_code: Optional[int] = None,
        response_body: Any = None,
        request_id: Optional[str] = None,
        error_type: Optional[str] = None,
        param: Any = None,
        code: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_body = response_body
        self.request_id = request_id
        self.error_type = error_type
        self.param = param
        self.code = code

    def __str__(self) -> str:
        return self.message or self.__class__.__name__


class ConfigurationError(TitanGPTException):
    """Raised when the client is configured incorrectly."""


class AuthenticationError(TitanGPTException):
    """Raised when authentication with the API fails."""


class AuthorizationError(TitanGPTException):
    """Raised when the API key is valid, but access is forbidden."""


class APIError(TitanGPTException):
    """Raised for generic API and transport level failures."""


class NotFoundError(APIError):
    """Raised when a requested resource does not exist."""


class RateLimitError(APIError):
    """Raised when the API rate limit is exceeded."""


class ValidationError(APIError):
    """Raised when request validation fails."""


class ModelNotFoundError(NotFoundError):
    """Raised when a requested model does not exist."""


class PromptError(TitanGPTException):
    """Raised when prompt formatting or prompt content is invalid."""


class TimeoutError(TitanGPTException):
    """Raised when a request exceeds the configured timeout."""


class ConnectionError(TitanGPTException):
    """Raised when the client cannot reach the API."""


class DataError(TitanGPTException):
    """Raised when the API returns data in an unexpected format."""


class NotImplementedError(TitanGPTException):
    """Raised when a requested SDK feature is not implemented."""
