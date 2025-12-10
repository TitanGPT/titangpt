import os
import json
import asyncio
import aiofiles
from typing import Any, Dict, Optional, List, Union
import aiohttp
from titangpt.exceptions import (
    APIError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    ModelNotFoundError,
    TitanGPTException
)

class TitanResponse(dict):
    def __getattr__(self, name):
        try:
            value = self[name]
            if isinstance(value, dict):
                return TitanResponse(value)
            if isinstance(value, list):
                return [TitanResponse(i) if isinstance(i, dict) else i for i in value]
            return value
        except KeyError:
            raise AttributeError(f"'TitanResponse' object has no attribute '{name}'")

class AsyncCompletions:
    def __init__(self, client):
        self._client = client

    async def create(self, model: str, messages: List[Dict[str, str]], **kwargs) -> TitanResponse:
        payload = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        return await self._client._post("v1/chat/completions", json=payload)

class AsyncChat:
    def __init__(self, client):
        self.completions = AsyncCompletions(client)

class AsyncImages:
    def __init__(self, client):
        self._client = client

    async def generate(self, prompt: str, model: str = "flux", n: int = 1, size: str = "1024x1024", **kwargs) -> TitanResponse:
        payload = {
            "prompt": prompt,
            "model": model,
            "n": n,
            "size": size,
            **kwargs
        }
        return await self._client._post("v1/images/generations", json=payload)

class AsyncAudio:
    def __init__(self, client):
        self.transcriptions = AsyncTranscriptions(client)

class AsyncTranscriptions:
    def __init__(self, client):
        self._client = client

    async def create(self, file, model: str = "whisper-1", **kwargs) -> TitanResponse:
        data = aiohttp.FormData()
        if isinstance(file, str):
            data.add_field('file', open(file, 'rb'))
        else:
            data.add_field('file', file)
            
        data.add_field('model', model)
        for k, v in kwargs.items():
            data.add_field(k, str(v))
            
        return await self._client._post("v1/audio/transcriptions", data=data)

class AsyncMusic:
    def __init__(self, client):
        self._client = client
    
    async def search(self, query: str) -> TitanResponse:
        return await self._client._post("v2/music/search", json={"query": query})

    async def lyrics(self, video_id: str) -> TitanResponse:
        return await self._client._get(f"v2/music/lyrics/{video_id}")

    async def download(self, video_id: str, save_path: str) -> str:
        await self._client._ensure_session()
        url = f"{self._client.base_url}/v2/music/download/{video_id}"
        
        try:
            timeout = aiohttp.ClientTimeout(total=300) 
            async with self._client._session.get(url, timeout=timeout) as resp:
                if resp.status >= 400:
                    await self._client._handle_error(resp)
                
                if os.path.isdir(save_path):
                    filename = f"{video_id}.mp3"
                    save_path = os.path.join(save_path, filename)
                
                async with aiofiles.open(save_path, mode='wb') as f:
                    async for chunk in resp.content.iter_chunked(8192):
                        await f.write(chunk)
                
                return save_path
        except Exception as e:
            raise APIError(f"Download failed: {str(e)}")

class AsyncModels:
    def __init__(self, client):
        self._client = client

    async def list(self) -> TitanResponse:
        return await self._client._post("v1/models")

class AsyncTitanGPT:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.titangpt.ru",
        timeout: int = 60,
        user_id: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("TITANGPT_API_KEY")
        if not self.api_key:
            raise ValueError("The api_key client option must be set either by passing api_key to the client or by setting the TITANGPT_API_KEY environment variable")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.user_id = user_id
        self._session: Optional[aiohttp.ClientSession] = None

        self.chat = AsyncChat(self)
        self.images = AsyncImages(self)
        self.audio = AsyncAudio(self)
        self.music = AsyncMusic(self)
        self.models = AsyncModels(self)

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "TitanGPT-Python-Async/1.0"
            }
            if self.user_id:
                headers["x-user-id"] = str(self.user_id)
                
            self._session = aiohttp.ClientSession(headers=headers)

    async def check_health(self) -> Dict[str, str]:
        await self._ensure_session()
        url = f"{self.base_url}/"
        try:
            async with self._session.get(url, timeout=10) as response:
                if response.status == 200:
                    return await response.json()
                text = await response.text()
                raise APIError(f"Health check failed with status {response.status}: {text}")
        except Exception as e:
            raise APIError(f"Health check failed: {str(e)}")

    async def _request(self, method: str, path: str, json: dict = None, data = None, params: dict = None) -> TitanResponse:
        await self._ensure_session()
        url = f"{self.base_url}/{path}"
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            async with self._session.request(method, url, json=json, data=data, params=params, timeout=timeout) as resp:
                if resp.status >= 400:
                    await self._handle_error(resp)

                result = await resp.json()
                return TitanResponse(result)

        except aiohttp.ClientError as e:
            raise APIError(f"Connection error: {str(e)}")
        except Exception as e:
            if isinstance(e, TitanGPTException):
                raise e
            raise APIError(f"Unexpected error: {str(e)}")

    async def _post(self, path: str, json: dict = None, data = None) -> TitanResponse:
        return await self._request("POST", path, json=json, data=data)

    async def _get(self, path: str, params: dict = None) -> TitanResponse:
        return await self._request("GET", path, params=params)

    async def _handle_error(self, response):
        try:
            error_data = await response.json()
            message = error_data.get("error", {}).get("message") or error_data.get("message")
            if not message and "detail" in error_data:
                message = error_data["detail"]
        except:
            message = await response.text()

        if not message:
            message = f"Error code: {response.status}"

        if response.status == 400:
            raise ValidationError(message)
        elif response.status == 401:
            raise AuthenticationError(f"Authentication failed: {message}")
        elif response.status == 403:
            raise AuthenticationError(f"Permission denied (Invalid API Key): {message}")
        elif response.status == 404:
            raise ModelNotFoundError(message)
        elif response.status == 429:
            raise RateLimitError(message)
        else:
            raise APIError(f"TitanGPT API Error {response.status}: {message}")

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
