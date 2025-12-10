import os
import json
import asyncio
from typing import Any, Dict, Optional, List, Union
import aiohttp


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



class AsyncTitanGPT:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.titangpt.ru",
        timeout: int = 60,
    ):
        self.api_key = api_key or os.getenv("TITANGPT_API_KEY")
        if not self.api_key:
            raise ValueError("The api_key client option must be set either by passing api_key to the client or by setting the TITANGPT_API_KEY environment variable")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

        self.chat = AsyncChat(self)
        self.images = AsyncImages(self)
        self.audio = AsyncAudio(self)
        self.music = AsyncMusic(self)

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "TitanGPT-Python-Async/1.0"
            }
            self._session = aiohttp.ClientSession(headers=headers)

    async def _post(self, path: str, json: dict = None, data = None) -> TitanResponse:
        await self._ensure_session()
        url = f"{self.base_url}/{path}"
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            if json is not None:
                resp = await self._session.post(url, json=json, timeout=timeout)
                resp = await self._session.post(url, data=data, timeout=timeout)
                
            if resp.status >= 400:
                try:
                    error_data = await resp.json()
                    error_msg = error_data.get("error", {}).get("message", await resp.text())
                except:
                    error_msg = await resp.text()
                raise Exception(f"API Error {resp.status}: {error_msg}")

            result = await resp.json()
            return TitanResponse(result)

        except Exception as e:
            raise e

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
