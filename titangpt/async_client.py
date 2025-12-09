"""
Asynchronous client for TitanGPT API.

This module provides an async-based client for interacting with the TitanGPT API,
enabling non-blocking operations and efficient concurrent request handling.
"""

import asyncio
import json
from typing import Any, Dict, Optional, List, AsyncIterator
from urllib.parse import urljoin

import aiohttp


class AsyncClient:
    """Asynchronous client for TitanGPT API interactions."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.titangpt.com",
        timeout: Optional[float] = 30.0,
        max_retries: int = 3,
    ):
        """
        Initialize the async client.

        Args:
            api_key: API key for authentication.
            base_url: Base URL for the API endpoint.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts for failed requests.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure an active client session exists."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "TitanGPT-AsyncClient/1.0",
        }

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Make an async HTTP request.

        Args:
            method: HTTP method (GET, POST, etc.).
            endpoint: API endpoint path.
            **kwargs: Additional arguments to pass to the request.

        Returns:
            Response data as dictionary.

        Raises:
            aiohttp.ClientError: If the request fails.
        """
        session = await self._ensure_session()
        url = urljoin(self.base_url, endpoint)
        headers = self._get_headers()

        timeout = aiohttp.ClientTimeout(total=self.timeout)

        for attempt in range(self.max_retries):
            try:
                async with session.request(
                    method,
                    url,
                    headers=headers,
                    timeout=timeout,
                    **kwargs,
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 401:
                        raise ValueError("Unauthorized: Invalid API key")
                    elif response.status == 429:
                        # Rate limited, retry with backoff
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        raise RuntimeError("Rate limit exceeded")
                    elif response.status >= 500:
                        # Server error, retry
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        raise RuntimeError(f"Server error: {response.status}")
                    else:
                        error_text = await response.text()
                        raise RuntimeError(
                            f"Request failed with status {response.status}: {error_text}"
                        )
            except asyncio.TimeoutError:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise TimeoutError(f"Request timeout after {self.timeout}s")

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "titan-gpt-3.5",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Create a chat completion.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            model: Model identifier.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens in the response.
            stream: Whether to stream the response.
            **kwargs: Additional parameters.

        Returns:
            Completion response data.
        """
        payload = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "stream": stream,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        payload.update(kwargs)

        return await self._request(
            "POST",
            "/v1/chat/completions",
            json=payload,
        )

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = "titan-gpt-3.5",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream a chat completion response.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            model: Model identifier.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens in the response.
            **kwargs: Additional parameters.

        Yields:
            Streamed response chunks.
        """
        session = await self._ensure_session()
        url = urljoin(self.base_url, "/v1/chat/completions")
        headers = self._get_headers()

        payload = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "stream": True,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        payload.update(kwargs)

        timeout = aiohttp.ClientTimeout(total=None)

        async with session.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout,
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"Stream request failed: {response.status}")

            async for line in response.content:
                decoded_line = line.decode("utf-8").strip()
                if decoded_line.startswith("data: "):
                    data = decoded_line[6:]
                    if data != "[DONE]":
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            pass

    async def get_models(self) -> Dict[str, Any]:
        """
        Get list of available models.

        Returns:
            Models data.
        """
        return await self._request("GET", "/v1/models")

    async def get_model(self, model_id: str) -> Dict[str, Any]:
        """
        Get information about a specific model.

        Args:
            model_id: Model identifier.

        Returns:
            Model information.
        """
        return await self._request("GET", f"/v1/models/{model_id}")

    async def create_embedding(
        self,
        text: str,
        model: str = "titan-embedding-1",
    ) -> Dict[str, Any]:
        """
        Create embeddings for text.

        Args:
            text: Text to embed.
            model: Embedding model identifier.

        Returns:
            Embedding response data.
        """
        payload = {
            "input": text,
            "model": model,
        }
        return await self._request(
            "POST",
            "/v1/embeddings",
            json=payload,
        )

    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the API service.

        Returns:
            Health status information.
        """
        return await self._request("GET", "/v1/health")


# Example usage and utility function
async def create_async_client(
    api_key: str,
    **kwargs: Any,
) -> AsyncClient:
    """
    Create and initialize an async client.

    Args:
        api_key: API key for authentication.
        **kwargs: Additional arguments for AsyncClient initialization.

    Returns:
        Initialized AsyncClient instance.
    """
    client = AsyncClient(api_key, **kwargs)
    await client._ensure_session()
    return client
