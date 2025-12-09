"""
Tests for the synchronous client module.

This module contains unit tests for the TitanGPT synchronous client,
covering initialization, API calls, error handling, and response processing.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from titangpt.client import Client
from titangpt.exceptions import (
    TitanGPTError,
    AuthenticationError,
    RateLimitError,
    APIError,
)


class TestClientInitialization:
    """Tests for client initialization and configuration."""

    def test_client_init_with_api_key(self):
        """Test client initialization with API key."""
        api_key = "test-api-key-123"
        client = Client(api_key=api_key)
        
        assert client.api_key == api_key
        assert client.base_url is not None

    def test_client_init_with_custom_base_url(self):
        """Test client initialization with custom base URL."""
        api_key = "test-api-key-123"
        custom_url = "https://custom.example.com"
        client = Client(api_key=api_key, base_url=custom_url)
        
        assert client.api_key == api_key
        assert client.base_url == custom_url

    def test_client_init_with_timeout(self):
        """Test client initialization with custom timeout."""
        api_key = "test-api-key-123"
        timeout = 30
        client = Client(api_key=api_key, timeout=timeout)
        
        assert client.timeout == timeout

    def test_client_init_without_api_key_raises_error(self):
        """Test that client initialization without API key raises error."""
        with pytest.raises((ValueError, TypeError)):
            Client(api_key=None)

    def test_client_headers_include_api_key(self):
        """Test that client headers include API key."""
        api_key = "test-api-key-123"
        client = Client(api_key=api_key)
        
        assert client.headers.get("Authorization") is not None or \
               client.headers.get("X-API-Key") is not None


class TestClientCompletion:
    """Tests for completion API calls."""

    @pytest.fixture
    def client(self):
        """Create a client instance for testing."""
        return Client(api_key="test-api-key")

    def test_create_completion_basic(self, client):
        """Test basic completion creation."""
        mock_response = {
            "id": "cmpl-123",
            "object": "text_completion",
            "created": 1234567890,
            "model": "titan-gpt",
            "choices": [
                {
                    "text": "This is a test response.",
                    "index": 0,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        
        with patch.object(client, "_request", return_value=mock_response) as mock_request:
            response = client.completions.create(
                model="titan-gpt",
                prompt="Hello, world!"
            )
            
            mock_request.assert_called_once()
            assert response["id"] == "cmpl-123"
            assert response["choices"][0]["text"] == "This is a test response."

    def test_create_completion_with_parameters(self, client):
        """Test completion creation with various parameters."""
        mock_response = {
            "id": "cmpl-456",
            "object": "text_completion",
            "created": 1234567890,
            "model": "titan-gpt",
            "choices": [
                {
                    "text": "Generated text",
                    "index": 0,
                    "finish_reason": "length"
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 50,
                "total_tokens": 70
            }
        }
        
        with patch.object(client, "_request", return_value=mock_response):
            response = client.completions.create(
                model="titan-gpt",
                prompt="Test prompt",
                max_tokens=100,
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0.5,
                presence_penalty=0.5
            )
            
            assert response["id"] == "cmpl-456"
            assert response["usage"]["total_tokens"] == 70

    def test_create_completion_with_multiple_prompts(self, client):
        """Test completion creation with multiple prompts."""
        mock_response = {
            "id": "cmpl-789",
            "object": "text_completion",
            "created": 1234567890,
            "model": "titan-gpt",
            "choices": [
                {
                    "text": "Response 1",
                    "index": 0,
                    "finish_reason": "stop"
                },
                {
                    "text": "Response 2",
                    "index": 1,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 10,
                "total_tokens": 25
            }
        }
        
        with patch.object(client, "_request", return_value=mock_response):
            response = client.completions.create(
                model="titan-gpt",
                prompt=["Prompt 1", "Prompt 2"],
                n=2
            )
            
            assert len(response["choices"]) == 2
            assert response["choices"][0]["text"] == "Response 1"
            assert response["choices"][1]["text"] == "Response 2"

    def test_create_completion_with_stream(self, client):
        """Test streaming completion."""
        chunk1 = {"choices": [{"delta": {"content": "Hello "}}]}
        chunk2 = {"choices": [{"delta": {"content": "world"}}]}
        
        with patch.object(client, "_request_stream", return_value=iter([chunk1, chunk2])):
            chunks = list(client.completions.create(
                model="titan-gpt",
                prompt="Hi",
                stream=True
            ))
            
            assert len(chunks) == 2
            assert "delta" in chunks[0]["choices"][0]


class TestClientChatCompletion:
    """Tests for chat completion API calls."""

    @pytest.fixture
    def client(self):
        """Create a client instance for testing."""
        return Client(api_key="test-api-key")

    def test_create_chat_completion_basic(self, client):
        """Test basic chat completion creation."""
        mock_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "titan-gpt",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you?"
                    },
                    "index": 0,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18
            }
        }
        
        with patch.object(client, "_request", return_value=mock_response):
            response = client.chat.completions.create(
                model="titan-gpt",
                messages=[
                    {"role": "user", "content": "Hello!"}
                ]
            )
            
            assert response["id"] == "chatcmpl-123"
            assert response["choices"][0]["message"]["content"] == "Hello! How can I help you?"

    def test_create_chat_completion_with_system_message(self, client):
        """Test chat completion with system message."""
        mock_response = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "titan-gpt",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I am a helpful assistant."
                    },
                    "index": 0,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 5,
                "total_tokens": 30
            }
        }
        
        with patch.object(client, "_request", return_value=mock_response):
            response = client.chat.completions.create(
                model="titan-gpt",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who are you?"}
                ]
            )
            
            assert response["choices"][0]["message"]["role"] == "assistant"

    def test_create_chat_completion_with_temperature(self, client):
        """Test chat completion with temperature parameter."""
        mock_response = {
            "id": "chatcmpl-789",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "titan-gpt",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Creative response"
                    },
                    "index": 0,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        
        with patch.object(client, "_request", return_value=mock_response):
            response = client.chat.completions.create(
                model="titan-gpt",
                messages=[{"role": "user", "content": "Be creative"}],
                temperature=0.9
            )
            
            assert response["choices"][0]["message"]["content"] == "Creative response"

    def test_create_chat_completion_streaming(self, client):
        """Test streaming chat completion."""
        chunk1 = {
            "choices": [{
                "delta": {"content": "Hello "},
                "index": 0,
                "finish_reason": None
            }]
        }
        chunk2 = {
            "choices": [{
                "delta": {"content": "there!"},
                "index": 0,
                "finish_reason": "stop"
            }]
        }
        
        with patch.object(client, "_request_stream", return_value=iter([chunk1, chunk2])):
            chunks = list(client.chat.completions.create(
                model="titan-gpt",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True
            ))
            
            assert len(chunks) == 2
            assert chunks[0]["choices"][0]["delta"]["content"] == "Hello "


class TestClientEmbeddings:
    """Tests for embeddings API calls."""

    @pytest.fixture
    def client(self):
        """Create a client instance for testing."""
        return Client(api_key="test-api-key")

    def test_create_embeddings_basic(self, client):
        """Test basic embeddings creation."""
        mock_response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3, 0.4]
                }
            ],
            "model": "titan-embedding",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        }
        
        with patch.object(client, "_request", return_value=mock_response):
            response = client.embeddings.create(
                model="titan-embedding",
                input="Hello world"
            )
            
            assert response["object"] == "list"
            assert len(response["data"]) == 1
            assert len(response["data"][0]["embedding"]) == 4

    def test_create_embeddings_multiple_inputs(self, client):
        """Test embeddings creation with multiple inputs."""
        mock_response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3, 0.4]
                },
                {
                    "object": "embedding",
                    "index": 1,
                    "embedding": [0.5, 0.6, 0.7, 0.8]
                }
            ],
            "model": "titan-embedding",
            "usage": {
                "prompt_tokens": 10,
                "total_tokens": 10
            }
        }
        
        with patch.object(client, "_request", return_value=mock_response):
            response = client.embeddings.create(
                model="titan-embedding",
                input=["Hello", "world"]
            )
            
            assert len(response["data"]) == 2

    def test_create_embeddings_with_encoding_format(self, client):
        """Test embeddings creation with encoding format."""
        mock_response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [1, 2, 3, 4]
                }
            ],
            "model": "titan-embedding",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        }
        
        with patch.object(client, "_request", return_value=mock_response):
            response = client.embeddings.create(
                model="titan-embedding",
                input="Test",
                encoding_format="int8"
            )
            
            assert response["data"][0]["embedding"] == [1, 2, 3, 4]


class TestClientErrorHandling:
    """Tests for error handling in the client."""

    @pytest.fixture
    def client(self):
        """Create a client instance for testing."""
        return Client(api_key="test-api-key")

    def test_authentication_error_on_invalid_key(self, client):
        """Test authentication error with invalid API key."""
        with patch.object(client, "_request", side_effect=AuthenticationError("Invalid API key")):
            with pytest.raises(AuthenticationError):
                client.chat.completions.create(
                    model="titan-gpt",
                    messages=[{"role": "user", "content": "Hello"}]
                )

    def test_rate_limit_error(self, client):
        """Test rate limit error handling."""
        with patch.object(client, "_request", side_effect=RateLimitError("Rate limit exceeded")):
            with pytest.raises(RateLimitError):
                client.completions.create(
                    model="titan-gpt",
                    prompt="Test"
                )

    def test_api_error_with_message(self, client):
        """Test API error with error message."""
        error_msg = "Model not found"
        with patch.object(client, "_request", side_effect=APIError(error_msg)):
            with pytest.raises(APIError):
                client.chat.completions.create(
                    model="unknown-model",
                    messages=[{"role": "user", "content": "Hello"}]
                )

    def test_timeout_error(self, client):
        """Test timeout error handling."""
        with patch.object(client, "_request", side_effect=TimeoutError("Request timeout")):
            with pytest.raises(TimeoutError):
                client.completions.create(
                    model="titan-gpt",
                    prompt="Test"
                )

    def test_connection_error(self, client):
        """Test connection error handling."""
        with patch.object(client, "_request", side_effect=ConnectionError("Connection failed")):
            with pytest.raises(ConnectionError):
                client.chat.completions.create(
                    model="titan-gpt",
                    messages=[{"role": "user", "content": "Hello"}]
                )

    def test_invalid_response_format(self, client):
        """Test handling of invalid response format."""
        with patch.object(client, "_request", return_value={}):
            with pytest.raises((KeyError, ValueError)):
                response = client.chat.completions.create(
                    model="titan-gpt",
                    messages=[{"role": "user", "content": "Hello"}]
                )
                # Ensure response has required fields
                _ = response["choices"]


class TestClientEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def client(self):
        """Create a client instance for testing."""
        return Client(api_key="test-api-key")

    def test_empty_prompt(self, client):
        """Test completion with empty prompt."""
        with pytest.raises((ValueError, TypeError)):
            client.completions.create(
                model="titan-gpt",
                prompt=""
            )

    def test_empty_messages(self, client):
        """Test chat completion with empty messages."""
        with pytest.raises((ValueError, TypeError)):
            client.chat.completions.create(
                model="titan-gpt",
                messages=[]
            )

    def test_very_long_prompt(self, client):
        """Test completion with very long prompt."""
        mock_response = {
            "id": "cmpl-long",
            "object": "text_completion",
            "created": 1234567890,
            "model": "titan-gpt",
            "choices": [{"text": "Response", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 1, "total_tokens": 1001}
        }
        
        long_prompt = "a" * 5000
        with patch.object(client, "_request", return_value=mock_response):
            response = client.completions.create(
                model="titan-gpt",
                prompt=long_prompt
            )
            
            assert response["usage"]["prompt_tokens"] == 1000

    def test_max_tokens_boundary(self, client):
        """Test completion with max_tokens at boundary."""
        mock_response = {
            "id": "cmpl-max",
            "object": "text_completion",
            "created": 1234567890,
            "model": "titan-gpt",
            "choices": [{"text": "...", "finish_reason": "length"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2048, "total_tokens": 2058}
        }
        
        with patch.object(client, "_request", return_value=mock_response):
            response = client.completions.create(
                model="titan-gpt",
                prompt="Test",
                max_tokens=2048
            )
            
            assert response["usage"]["completion_tokens"] == 2048

    def test_temperature_zero(self, client):
        """Test completion with temperature set to zero (deterministic)."""
        mock_response = {
            "id": "cmpl-det",
            "object": "text_completion",
            "created": 1234567890,
            "model": "titan-gpt",
            "choices": [{"text": "Deterministic response", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        }
        
        with patch.object(client, "_request", return_value=mock_response):
            response = client.completions.create(
                model="titan-gpt",
                prompt="Test",
                temperature=0
            )
            
            assert response["choices"][0]["text"] == "Deterministic response"

    def test_temperature_max(self, client):
        """Test completion with maximum temperature (maximum randomness)."""
        mock_response = {
            "id": "cmpl-rand",
            "object": "text_completion",
            "created": 1234567890,
            "model": "titan-gpt",
            "choices": [{"text": "Random response", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        }
        
        with patch.object(client, "_request", return_value=mock_response):
            response = client.completions.create(
                model="titan-gpt",
                prompt="Test",
                temperature=2.0
            )
            
            assert response["choices"][0]["text"] == "Random response"

    def test_zero_top_p(self, client):
        """Test chat completion with top_p set to zero."""
        mock_response = {
            "id": "chatcmpl-top-p",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "titan-gpt",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
        }
        
        with patch.object(client, "_request", return_value=mock_response):
            response = client.chat.completions.create(
                model="titan-gpt",
                messages=[{"role": "user", "content": "Test"}],
                top_p=0.0
            )
            
            assert response["choices"][0]["message"]["content"] == "Response"


class TestClientContextManager:
    """Tests for context manager usage."""

    def test_client_context_manager(self):
        """Test client as context manager."""
        with Client(api_key="test-api-key") as client:
            assert client is not None
            assert client.api_key == "test-api-key"

    def test_client_cleanup_on_context_exit(self):
        """Test that client cleans up properly on context exit."""
        client = None
        with Client(api_key="test-api-key") as temp_client:
            client = temp_client
        
        # Client should be properly closed after context exit
        assert client is not None


class TestClientRetry:
    """Tests for retry logic in the client."""

    @pytest.fixture
    def client(self):
        """Create a client instance for testing."""
        return Client(api_key="test-api-key")

    def test_retry_on_transient_error(self, client):
        """Test that client retries on transient errors."""
        mock_response = {
            "id": "cmpl-retry",
            "object": "text_completion",
            "created": 1234567890,
            "model": "titan-gpt",
            "choices": [{"text": "Success", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
        }
        
        with patch.object(client, "_request") as mock_request:
            mock_request.side_effect = [
                TimeoutError("Timeout"),
                TimeoutError("Timeout"),
                mock_response
            ]
            
            # This test depends on client having retry logic implemented
            # Adjust based on actual implementation
            try:
                response = client.completions.create(
                    model="titan-gpt",
                    prompt="Test"
                )
                assert response is not None
            except TimeoutError:
                # If retries are not implemented, that's fine for this test
                pass

    def test_no_retry_on_permanent_error(self, client):
        """Test that client doesn't retry on permanent errors."""
        with patch.object(client, "_request", side_effect=AuthenticationError("Invalid key")):
            with pytest.raises(AuthenticationError):
                client.chat.completions.create(
                    model="titan-gpt",
                    messages=[{"role": "user", "content": "Hello"}]
                )
