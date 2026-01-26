import os
import aiofiles
from typing import Any, Dict, Optional, List, Union
import httpx
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

    async def create(self, model: str, messages: List[Dict[str, str]], **kwargs) -> Union[TitanResponse, List]:
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

    async def generate(self, prompt: str, model: str = "flux", n: int = 1, size: str = "1024x1024", **kwargs) -> Union[TitanResponse, List]:
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

    async def create(self, file, model: str = "whisper-1", **kwargs) -> Union[TitanResponse, List]:
        files = {}
        data = {'model': model}
        
        for k, v in kwargs.items():
            data[k] = str(v)

        file_obj = None
        should_close = False

        try:
            if isinstance(file, str):
                file_obj = open(file, 'rb')
                should_close = True
                files['file'] = file_obj
            else:
                files['file'] = file
            
            await self._client._ensure_client()
            return await self._client._request("POST", "v1/audio/transcriptions", data=data, files=files)
        finally:
            if should_close and file_obj:
                file_obj.close()


class BaseMusicDownloader:
    def __init__(self, client):
        self._client = client

    async def _download_file(self, url: str, save_path: str, file_id: str, method: str = "GET", json_body: Optional[dict] = None, ext: str = "mp3") -> str:
        await self._client._ensure_client()
        
        try:
            request_kwargs: Dict[str, Any] = {"timeout": 300.0}
            if method == "POST" and json_body:
                request_kwargs["json"] = json_body

            async with self._client._session.stream(method, url, **request_kwargs) as resp:
                if resp.status_code >= 400:
                    await self._client._handle_error(resp)
                if os.path.isdir(save_path):
                    filename = f"{file_id}.{ext}"
                    save_path = os.path.join(save_path, filename)
                
                async with aiofiles.open(save_path, mode='wb') as f:
                    async for chunk in resp.aiter_bytes():
                        await f.write(chunk)
                
                return save_path
        except Exception as e:
            if isinstance(e, TitanGPTException):
                raise e
            raise APIError(f"Download failed: {str(e)}")

class AsyncYandexMusic(BaseMusicDownloader):
    
    async def search(self, query: str) -> Union[TitanResponse, List]:
        return await self._client._post("v2/yandex/search", json={"query": query})

    async def lyrics(self, track_id: str) -> Union[TitanResponse, List]:
        return await self._client._get(f"v2/yandex/lyrics/{track_id}")

 
    async def download(self, track_id: str, save_path: str, lossless: bool = False) -> str:
        """Скачивает трек, всегда используя метод POST."""
        url = f"{self._client.base_url}/v2/yandex/download/{track_id}"
        return await self._download_file(
            url,
            save_path,
            track_id,
            method="POST",
            json_body={"lossless": lossless},
            ext="flac" if lossless else "mp3"
        )

class AsyncYouTubeMusic(BaseMusicDownloader):

    async def search(self, query: str) -> Union[TitanResponse, List]:
        return await self._client._post("v2/youtube/music/search", json={"query": query})

    async def lyrics(self, video_id: str) -> Union[TitanResponse, List]:
        return await self._client._get(f"v2/youtube/music/lyrics/{video_id}")

    async def download(self, video_id: str, save_path: str) -> str:
        url = f"{self._client.base_url}/v2/youtube/music/download/{video_id}"
        return await self._download_file(url, save_path, video_id, method="GET", ext="mp3")

class AsyncMusic:
    def __init__(self, client):
        self.yandex = AsyncYandexMusic(client)
        self.youtube = AsyncYouTubeMusic(client)

class AsyncModerations:
    def __init__(self, client):
        self._client = client

    async def create(self, input: str) -> Union[TitanResponse, List]:
        return await self._client._post("v1/beta/moderations", json={"input": input})

class AsyncThreads:
    def __init__(self, client):
        self._client = client

    async def create(self, metadata: Optional[Dict] = None) -> Union[TitanResponse, List]:
        payload = {}
        if metadata:
            payload["metadata"] = metadata
        return await self._client._post("beta/v1/threads", json=payload)

    async def add_message(self, thread_id: str, content: str, role: str = "user") -> Union[TitanResponse, List]:
        payload = {
            "role": role,
            "content": content
        }
        return await self._client._post(f"beta/v1/threads/{thread_id}/messages", json=payload)

    async def run(self, thread_id: str, assistant_id: str, model: str = "gpt-4o", instructions: Optional[str] = None) -> Union[TitanResponse, List]:
        payload = {
            "assistant_id": assistant_id,
            "model": model
        }
        if instructions:
            payload["instructions"] = instructions
        return await self._client._post(f"beta/v1/threads/{thread_id}/runs", json=payload)

    async def list_messages(self, thread_id: str) -> Union[TitanResponse, List]:
        return await self._client._get(f"beta/v1/threads/{thread_id}/messages")

class AsyncModels:
    def __init__(self, client):
        self._client = client

    async def list(self) -> Union[TitanResponse, List]:
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
        self._session: Optional[httpx.AsyncClient] = None 
        self.chat = AsyncChat(self)
        self.images = AsyncImages(self)
        self.audio = AsyncAudio(self)
        self.music = AsyncMusic(self)
        self.moderations = AsyncModerations(self)
        self.threads = AsyncThreads(self)
        self.models = AsyncModels(self)

    async def _ensure_client(self):
        is_closed = getattr(self._session, "is_closed", False) if self._session else True
        
        if self._session is None or is_closed:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "TitanGPT-Python-Async/1.2 (HTTP/2)",
                "Content-Type": "application/json"
            }
            if self.user_id:
                headers["x-user-id"] = str(self.user_id)
            
            self._session = httpx.AsyncClient(
                headers=headers, 
                http2=True,
                timeout=self.timeout
            )

    async def check_health(self) -> Dict[str, str]:
        await self._ensure_client()
        assert self._session is not None
        
        url = f"{self.base_url}/"
        try:
            response = await self._session.get(url, timeout=10.0)
            if response.status_code == 200:
                return response.json()
            text = response.text
            raise APIError(f"Health check failed with status {response.status_code}: {text}")
        except Exception as e:
            raise APIError(f"Health check failed: {str(e)}")

    async def _request(self, method: str, path: str, json: Optional[dict] = None, data = None, params: Optional[dict] = None, files = None) -> Union[TitanResponse, List]:
        await self._ensure_client()
        assert self._session is not None
        
        url = f"{self.base_url}/{path}"
        request_headers = self._session.headers.copy()
        if files:
            if "Content-Type" in request_headers:
                del request_headers["Content-Type"]

        try:
            resp = await self._session.request(
                method, 
                url, 
                json=json, 
                data=data, 
                params=params, 
                files=files,
                headers=request_headers
            )
            
            if resp.status_code >= 400:
                await self._handle_error(resp)

            result = resp.json()
            if isinstance(result, list):
                return [TitanResponse(i) if isinstance(i, dict) else i for i in result]
            return TitanResponse(result)

        except httpx.RequestError as e:
            raise APIError(f"Connection error: {str(e)}")
        except Exception as e:
            if isinstance(e, TitanGPTException):
                raise e
            raise APIError(f"Unexpected error: {str(e)}")

    async def _post(self, path: str, json: Optional[dict] = None, data = None) -> Union[TitanResponse, List]:
        return await self._request("POST", path, json=json, data=data)

    async def _get(self, path: str, params: Optional[dict] = None) -> Union[TitanResponse, List]:
        return await self._request("GET", path, params=params)

    async def _handle_error(self, response: httpx.Response):
        try:
            error_data = response.json()
            message = error_data.get("error", {}).get("message") or error_data.get("message")
            if not message and "detail" in error_data:
                message = error_data["detail"]
        except:
            message = response.text

        if not message:
            message = f"Error code: {response.status_code}"

        status = response.status_code

        if status == 400:
            raise ValidationError(message)
        elif status == 401:
            raise AuthenticationError(f"Authentication failed: {message}")
        elif status == 403:
            raise AuthenticationError(f"Permission denied (Invalid API Key): {message}")
        elif status == 404:
            raise ModelNotFoundError(message)
        elif status == 429:
            raise RateLimitError(message)
        else:
            raise APIError(f"TitanGPT API Error {status}: {message}")

    async def close(self):
        if self._session and not getattr(self._session, "is_closed", False):
            await self._session.aclose()
        self._session = None

    async def __aenter__(self):
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()