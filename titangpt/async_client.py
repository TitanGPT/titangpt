import asyncio
import os
from typing import Any, Dict, List, Optional, Sequence, Union

import aiofiles
import httpx

from titangpt._error_handling import raise_api_error
from titangpt.client import (
    IDEMPOTENT_METHODS,
    RETRYABLE_STATUS_CODES,
    JsonObject,
    ResponseData,
    _build_file_payload,
    _join_url,
    _wrap_data,
)
from titangpt.exceptions import (
    APIError,
    ConnectionError,
    DataError,
    NotFoundError,
    TimeoutError,
    TitanGPTException,
    ValidationError,
)


class AsyncCompletions:
    def __init__(self, client: "AsyncTitanGPT") -> None:
        self._client = client

    async def create(
        self, model: str, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> ResponseData:
        payload = {"model": model, "messages": messages, **kwargs}
        return await self._client._post("v1/chat/completions", json=payload)


class AsyncChat:
    def __init__(self, client: "AsyncTitanGPT") -> None:
        self.completions = AsyncCompletions(client)


class AsyncAudio:
    def __init__(self, client: "AsyncTitanGPT") -> None:
        self.transcriptions = AsyncTranscriptions(client)


class AsyncTranscriptions:
    def __init__(self, client: "AsyncTitanGPT") -> None:
        self._client = client

    async def create(
        self, file: Any, model: str = "whisper-1", **kwargs: Any
    ) -> ResponseData:
        data = {"model": model}
        for key, value in kwargs.items():
            if value is not None:
                data[key] = str(value)
        return await self._client._upload_file(
            "v1/audio/transcriptions", file=file, data=data
        )


class BaseMusicDownloader:
    def __init__(self, client: "AsyncTitanGPT") -> None:
        self._client = client

    async def _download_file(
        self,
        url: str,
        save_path: str,
        file_id: str,
        *,
        method: str = "GET",
        json_body: Optional[JsonObject] = None,
        ext: str = "mp3",
    ) -> str:
        await self._client._ensure_client()
        stream_url = _join_url(self._client.base_url, url)
        try:
            async with self._client._session.stream(
                method,
                stream_url,
                json=json_body,
                timeout=self._client.timeout,
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    await self._client._handle_error(response)

                if os.path.isdir(save_path):
                    save_path = os.path.join(save_path, "{0}.{1}".format(file_id, ext))

                async with aiofiles.open(save_path, "wb") as file_obj:
                    async for chunk in response.aiter_bytes():
                        if chunk:
                            await file_obj.write(chunk)
                return save_path
        except httpx.TimeoutException as exc:
            raise TimeoutError("Download timed out: {0}".format(exc))
        except httpx.ConnectError as exc:
            raise ConnectionError("Connection error: {0}".format(exc))
        except TitanGPTException:
            raise
        except Exception as exc:
            raise APIError("Download failed: {0}".format(exc))


class AsyncYandexMusic(BaseMusicDownloader):
    async def search(self, query: str) -> ResponseData:
        return await self._client._post("v2/yandex/search", json={"query": query})

    async def lyrics(self, track_id: str) -> ResponseData:
        return await self._client._get("v2/yandex/lyrics/{0}".format(track_id))

    async def download(
        self, track_id: str, save_path: str, lossless: bool = False
    ) -> str:
        return await self._download_file(
            "v2/yandex/download/{0}".format(track_id),
            save_path,
            track_id,
            method="POST",
            json_body={"lossless": lossless},
            ext="flac" if lossless else "mp3",
        )


class AsyncYouTubeMusic(BaseMusicDownloader):
    async def search(self, query: str) -> ResponseData:
        return await self._client._post("v2/youtube/music/search", json={"query": query})

    async def lyrics(self, video_id: str) -> ResponseData:
        return await self._client._get("v2/youtube/music/lyrics/{0}".format(video_id))

    async def download(self, video_id: str, save_path: str) -> str:
        return await self._download_file(
            "v2/youtube/music/download/{0}".format(video_id),
            save_path,
            video_id,
            method="GET",
            ext="mp3",
        )


class AsyncMusic:
    def __init__(self, client: "AsyncTitanGPT") -> None:
        self.yandex = AsyncYandexMusic(client)
        self.youtube = AsyncYouTubeMusic(client)

    async def search(self, query: str, provider: str = "youtube") -> ResponseData:
        provider_name = provider.lower()
        if provider_name in {"youtube", "yt"}:
            return await self.youtube.search(query)
        if provider_name == "yandex":
            return await self.yandex.search(query)
        raise ValidationError(
            "Unsupported music provider: {0}".format(provider_name)
        )


class AsyncThreads:
    _paths = ("v1/threads", "beta/v1/threads")

    def __init__(self, client: "AsyncTitanGPT") -> None:
        self._client = client

    async def create(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        tool_resources: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> ResponseData:
        payload: JsonObject = {}
        if messages is not None:
            payload["messages"] = messages
        if tool_resources is not None:
            payload["tool_resources"] = tool_resources
        if metadata is not None:
            payload["metadata"] = metadata
        return await self._client._post_any(self._paths, json=payload or None)

    async def retrieve(self, thread_id: str) -> ResponseData:
        return await self._client._get_any(
            ["{0}/{1}".format(path, thread_id) for path in self._paths]
        )

    async def add_message(
        self,
        thread_id: str,
        content: Union[str, List[Dict[str, Any]]],
        role: str = "user",
        metadata: Optional[Dict[str, str]] = None,
    ) -> ResponseData:
        payload: JsonObject = {"role": role, "content": content}
        if metadata is not None:
            payload["metadata"] = metadata
        return await self._client._post_any(
            ["{0}/{1}/messages".format(path, thread_id) for path in self._paths],
            json=payload,
        )

    async def list_messages(
        self, thread_id: str, limit: int = 20, order: str = "desc"
    ) -> ResponseData:
        return await self._client._get_any(
            ["{0}/{1}/messages".format(path, thread_id) for path in self._paths],
            params={"limit": limit, "order": order},
        )

    async def run(
        self,
        thread_id: str,
        assistant_id: str,
        model: Optional[str] = None,
        instructions: Optional[str] = None,
    ) -> ResponseData:
        payload: JsonObject = {"assistant_id": assistant_id}
        if model is not None:
            payload["model"] = model
        if instructions is not None:
            payload["instructions"] = instructions
        return await self._client._post_any(
            ["{0}/{1}/runs".format(path, thread_id) for path in self._paths],
            json=payload,
        )


class AsyncModels:
    def __init__(self, client: "AsyncTitanGPT") -> None:
        self._client = client

    async def list(self) -> ResponseData:
        return await self._client._get_any(
            ["v1/models", "models"], fallback_methods=("POST",)
        )


class AsyncFiles:
    def __init__(self, client: "AsyncTitanGPT") -> None:
        self._client = client

    async def create(
        self, file: Any, purpose: str = "assistants", ttl: Optional[int] = None
    ) -> ResponseData:
        data: Dict[str, str] = {"purpose": purpose}
        if ttl is not None:
            data["ttl"] = str(ttl)
        return await self._client._upload_file("v1/files", file=file, data=data)

    async def list(self) -> ResponseData:
        return await self._client._get("v1/files")

    async def retrieve(self, file_id: str) -> ResponseData:
        return await self._client._get("v1/files/{0}".format(file_id))

    async def delete(self, file_id: str) -> ResponseData:
        return await self._client._delete("v1/files/{0}".format(file_id))

    async def content(
        self, file_id: str, *, decode: bool = False, encoding: str = "utf-8"
    ) -> Union[bytes, str]:
        response = await self._client._get_raw("v1/files/{0}/content".format(file_id))
        if decode:
            return response.content.decode(encoding)
        return response.content


class AsyncUsage:
    def __init__(self, client: "AsyncTitanGPT") -> None:
        self._client = client

    async def get(self) -> ResponseData:
        return await self._client._get("v1/usage")


class AsyncTitanGPT:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.titangpt.xyz",
        timeout: int = 60,
        max_retries: int = 3,
        user_id: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("TITANGPT_API_KEY")
        if not self.api_key:
            raise ValueError(
                "The api_key client option must be set either by passing api_key "
                "to the client or by setting the TITANGPT_API_KEY environment variable"
            )

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.user_id = user_id
        self._session: Optional[httpx.AsyncClient] = None

        self.chat = AsyncChat(self)
        self.audio = AsyncAudio(self)
        self.music = AsyncMusic(self)
        self.threads = AsyncThreads(self)
        self.models = AsyncModels(self)
        self.files = AsyncFiles(self)
        self.usage = AsyncUsage(self)

    async def _ensure_client(self) -> None:
        is_closed = getattr(self._session, "is_closed", False) if self._session else True
        if self._session is None or is_closed:
            headers = {
                "Authorization": "Bearer {0}".format(self.api_key),
                "User-Agent": "TitanGPT-Python-Async/0.2.4"
            }
            if self.user_id:
                headers["x-user-id"] = str(self.user_id)

            self._session = httpx.AsyncClient(headers=headers, http2=True, timeout=self.timeout)

    async def check_health(self) -> ResponseData:
        return await self._get("")

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        await self._ensure_client()
        assert self._session is not None

        url = _join_url(self.base_url, path)
        method_upper = method.upper()
        attempts = self.max_retries + 1 if method_upper in IDEMPOTENT_METHODS else 1
        last_error: Optional[TitanGPTException] = None

        for attempt in range(attempts):
            try:
                response = await self._session.request(
                    method_upper,
                    url,
                    **kwargs
                )
            except httpx.TimeoutException as exc:
                last_error = TimeoutError("Request timed out: {0}".format(exc))
            except httpx.ConnectError as exc:
                last_error = ConnectionError("Connection error: {0}".format(exc))
            except httpx.RequestError as exc:
                raise APIError("Connection error: {0}".format(exc)) from exc
            else:
                if (
                    response.status_code in RETRYABLE_STATUS_CODES
                    and attempt < attempts - 1
                ):
                    await asyncio.sleep(min(2 ** attempt, 5))
                    continue
                if response.status_code >= 400:
                    await self._handle_error(response)
                return response

            if attempt < attempts - 1:
                await asyncio.sleep(min(2 ** attempt, 5))

        if last_error is not None:
            raise last_error
        raise APIError("Request failed before reaching the API")

    async def _request_any(
        self,
        method: str,
        paths: Sequence[str],
        *,
        fallback_methods: Sequence[str] = (),
        **kwargs: Any
    ) -> httpx.Response:
        last_error: Optional[TitanGPTException] = None

        for candidate_method in (method,) + tuple(fallback_methods):
            for path in paths:
                try:
                    return await self._request(candidate_method, path, **kwargs)
                except NotFoundError as exc:
                    last_error = exc
                    continue

        if last_error is not None:
            raise last_error
        raise APIError("No request paths were provided")

    def _process_response(self, response: httpx.Response) -> ResponseData:
        if response.status_code == 204 or not response.content:
            return None

        content_type = response.headers.get("Content-Type", "").lower()
        if "application/json" in content_type or "+json" in content_type:
            try:
                return _wrap_data(response.json())
            except ValueError as exc:
                raise DataError(
                    "Expected JSON response but received invalid data",
                    status_code=response.status_code,
                    response_body=response.text,
                ) from exc

        if content_type.startswith("text/"):
            return response.text

        return response.content

    async def _post(
        self, path: str, json: Optional[JsonObject] = None, data: Any = None
    ) -> ResponseData:
        response = await self._request("POST", path, json=json, data=data)
        return self._process_response(response)

    async def _post_any(
        self,
        paths: Sequence[str],
        json: Optional[JsonObject] = None,
        data: Any = None,
        *,
        fallback_methods: Sequence[str] = (),
    ) -> ResponseData:
        response = await self._request_any(
            "POST", paths, json=json, data=data, fallback_methods=fallback_methods
        )
        return self._process_response(response)

    async def _get(self, path: str, params: Optional[JsonObject] = None) -> ResponseData:
        response = await self._request("GET", path, params=params)
        return self._process_response(response)

    async def _get_any(
        self,
        paths: Sequence[str],
        params: Optional[JsonObject] = None,
        *,
        fallback_methods: Sequence[str] = (),
    ) -> ResponseData:
        response = await self._request_any(
            "GET", paths, params=params, fallback_methods=fallback_methods
        )
        return self._process_response(response)

    async def _get_raw(
        self, path: str, params: Optional[JsonObject] = None
    ) -> httpx.Response:
        return await self._request("GET", path, params=params)

    async def _delete(self, path: str) -> ResponseData:
        response = await self._request("DELETE", path)
        return self._process_response(response)

    async def _upload_file(
        self, path: str, *, file: Any, data: Optional[Dict[str, str]] = None
    ) -> ResponseData:
        await self._ensure_client()
        assert self._session is not None

        opened_file = None
        try:
            if isinstance(file, (str, os.PathLike)):
                opened_file = open(os.fspath(file), "rb")
                file_payload = _build_file_payload(opened_file)
            else:
                file_payload = _build_file_payload(file)

            response = await self._request(
                "POST",
                path,
                files={"file": file_payload},
                data=data,
            )
            return self._process_response(response)
        finally:
            if opened_file is not None:
                opened_file.close()

    async def _handle_error(self, response: httpx.Response) -> None:
        request_id = response.headers.get("x-request-id") or response.headers.get(
            "request-id"
        )
        response_body: Any = response.text

        try:
            response_body = response.json()
        except ValueError:
            pass

        raise_api_error(
            status_code=response.status_code,
            response_body=response_body,
            raw_text=response.text,
            request_id=request_id,
        )

    async def close(self) -> None:
        if self._session and not getattr(self._session, "is_closed", False):
            await self._session.aclose()
        self._session = None

    async def __aenter__(self) -> "AsyncTitanGPT":
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
