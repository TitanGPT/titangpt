class TitanGPTException(Exception):
    pass

class ConfigurationError(TitanGPTException):
    pass

class AuthenticationError(TitanGPTException):
    pass

class AuthorizationError(TitanGPTException):
    pass

class APIError(TitanGPTException):
    pass

class RateLimitError(APIError):
    pass

class ValidationError(TitanGPTException):
    pass

class ModelNotFoundError(TitanGPTException):
    pass

class PromptError(TitanGPTException):
    pass

class TimeoutError(TitanGPTException):
    pass

class ConnectionError(TitanGPTException):
    pass

class DataError(TitanGPTException):
    pass

class NotImplementedError(TitanGPTException):
    pass
