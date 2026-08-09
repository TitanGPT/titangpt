import os
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import requests

from titangpt._error_handling import raise_api_error
from titangpt.exceptions import (
    APIError,
    ConnectionError,
    DataError,
    NotFoundError,
    TimeoutError,
    TitanGPTException,
    ValidationError,
)


ResponseData = Union["TitanResponse", List[Any], str, bytes, None]
JsonObject = Dict[str, Any]
IDEMPOTENT_METHODS = {"GET", "HEAD", "OPTIONS"}
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class TitanResponse(dict):
    """Dictionary wrapper that also exposes keys as attributes."""

    def __getattr__(self, name: str) -> Any:
        try:
            return _wrap_data(self[name])
        except KeyError as exc:
            raise AttributeError(
                "'TitanResponse' object has no attribute '{0}'".format(name)
            ) from exc


def _wrap_data(value: Any) -> Any:
    if isinstance(value, dict) and not isinstance(value, TitanResponse):
        return TitanResponse({key: _wrap_data(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_wrap_data(item) for item in value]
    return value


def _join_url(base_url: str, path: str) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return "{0}/{1}".format(base_url.rstrip("/"), path.lstrip("/"))


def _build_file_payload(file: Any) -> Any:
    filename = getattr(file, "name", "upload.bin")
    return (os.path.basename(filename), file)


class Completions:
    def __init__(self, client: "TitanGPT") -> None:
        self._client = client

    def create(
        self, model: str, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> ResponseData:
        payload = {"model": model, "messages": messages, **kwargs}
        return self._client._post("v1/chat/completions", json=payload)


class Chat:
    def __init__(self, client: "TitanGPT") -> None:
        self.completions = Completions(client)


class Audio:
    def __init__(self, client: "TitanGPT") -> None:
        self.transcriptions = Transcriptions(client)


class Transcriptions:
    def __init__(self, client: "TitanGPT") -> None:
        self._client = client

    def create(self, file: Any, model: str = "whisper-1", **kwargs: Any) -> ResponseData:
        data = {"model": model, **kwargs}
        return self._client._upload_file("v1/audio/transcriptions", file=file, data=data)


class BaseMusicDownloader:
    def __init__(self, client: "TitanGPT") -> None:
        self._client = client

    def _download_file(
        self,
        url: str,
        save_path: str,
        file_id: str,
        *,
        method: str = "GET",
        json_body: Optional[JsonObject] = None,
        ext: str = "mp3",
    ) -> str:
        response = self._client._request(method, url, json=json_body, stream=True)

        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "{0}.{1}".format(file_id, ext))

        with open(save_path, "wb") as file_obj:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file_obj.write(chunk)
        return save_path


class YandexMusic(BaseMusicDownloader):
    def search(self, query: str) -> ResponseData:
        return self._client._post("v2/yandex/search", json={"query": query})

    def lyrics(self, track_id: str) -> ResponseData:
        return self._client._get("v2/yandex/lyrics/{0}".format(track_id))

    def download(self, track_id: str, save_path: str, lossless: bool = False) -> str:
        return self._download_file(
            url="v2/yandex/download/{0}".format(track_id),
            save_path=save_path,
            file_id=track_id,
            method="POST",
            json_body={"lossless": lossless},
            ext="flac" if lossless else "mp3",
        )


class YouTubeMusic(BaseMusicDownloader):
    def search(self, query: str) -> ResponseData:
        return self._client._post("v2/youtube/music/search", json={"query": query})

    def lyrics(self, video_id: str) -> ResponseData:
        return self._client._get("v2/youtube/music/lyrics/{0}".format(video_id))

    def download(self, video_id: str, save_path: str) -> str:
        return self._download_file(
            url="v2/youtube/music/download/{0}".format(video_id),
            save_path=save_path,
            file_id=video_id,
            method="GET",
            ext="mp3",
        )


class Music:
    def __init__(self, client: "TitanGPT") -> None:
        self.yandex = YandexMusic(client)
        self.youtube = YouTubeMusic(client)

    def search(self, query: str, provider: str = "youtube") -> ResponseData:
        provider_name = provider.lower()
        if provider_name in {"youtube", "yt"}:
            return self.youtube.search(query)
        if provider_name == "yandex":
            return self.yandex.search(query)
        raise ValidationError(
            "Unsupported music provider: {0}".format(provider_name)
        )


class Threads:
    _paths = ("v1/threads", "beta/v1/threads")

    def __init__(self, client: "TitanGPT") -> None:
        self._client = client

    def create(
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
        return self._client._post_any(self._paths, json=payload or None)

    def retrieve(self, thread_id: str) -> ResponseData:
        return self._client._get_any(
            ["{0}/{1}".format(path, thread_id) for path in self._paths]
        )

    def add_message(
        self,
        thread_id: str,
        content: Union[str, List[Dict[str, Any]]],
        role: str = "user",
        metadata: Optional[Dict[str, str]] = None,
    ) -> ResponseData:
        payload: JsonObject = {"role": role, "content": content}
        if metadata is not None:
            payload["metadata"] = metadata
        return self._client._post_any(
            ["{0}/{1}/messages".format(path, thread_id) for path in self._paths],
            json=payload,
        )

    def list_messages(
        self, thread_id: str, limit: int = 20, order: str = "desc"
    ) -> ResponseData:
        return self._client._get_any(
            ["{0}/{1}/messages".format(path, thread_id) for path in self._paths],
            params={"limit": limit, "order": order},
        )

    def run(
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
        return self._client._post_any(
            ["{0}/{1}/runs".format(path, thread_id) for path in self._paths],
            json=payload,
        )


class Models:
    def __init__(self, client: "TitanGPT") -> None:
        self._client = client

    def list(self) -> ResponseData:
        return self._client._get_any(["v1/models", "models"], fallback_methods=("POST",))


class Files:
    def __init__(self, client: "TitanGPT") -> None:
        self._client = client

    def create(
        self, file: Any, purpose: str = "assistants", ttl: Optional[int] = None
    ) -> ResponseData:
        data: JsonObject = {"purpose": purpose}
        if ttl is not None:
            data["ttl"] = ttl
        return self._client._upload_file("v1/files", file=file, data=data)

    def list(self) -> ResponseData:
        return self._client._get("v1/files")

    def retrieve(self, file_id: str) -> ResponseData:
        return self._client._get("v1/files/{0}".format(file_id))

    def delete(self, file_id: str) -> ResponseData:
        return self._client._delete("v1/files/{0}".format(file_id))

    def content(
        self, file_id: str, *, decode: bool = False, encoding: str = "utf-8"
    ) -> Union[bytes, str]:
        response = self._client._get_raw("v1/files/{0}/content".format(file_id))
        if decode:
            return response.content.decode(encoding)
        return response.content


class Usage:
    def __init__(self, client: "TitanGPT") -> None:
        self._client = client

    def get(self) -> ResponseData:
        return self._client._get("v1/usage")


class TitanGPT:
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
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": "Bearer {0}".format(self.api_key),
                "User-Agent": "TitanGPT-Python/0.2.4"
            }
        )
        if user_id:
            self.session.headers["x-user-id"] = str(user_id)

        self.chat = Chat(self)
        self.audio = Audio(self)
        self.music = Music(self)
        self.threads = Threads(self)
        self.models = Models(self)
        self.files = Files(self)
        self.usage = Usage(self)

    def check_health(self) -> ResponseData:
        return self._get("")

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = _join_url(self.base_url, path)
        method_upper = method.upper()
        attempts = self.max_retries + 1 if method_upper in IDEMPOTENT_METHODS else 1
        last_error: Optional[TitanGPTException] = None

        for attempt in range(attempts):
            try:
                response = self.session.request(
                    method_upper,
                    url,
                    timeout=self.timeout,
                    **kwargs
                )
            except requests.Timeout as exc:
                last_error = TimeoutError("Request timed out: {0}".format(exc))
            except requests.ConnectionError as exc:
                last_error = ConnectionError("Connection error: {0}".format(exc))
            except requests.RequestException as exc:
                raise APIError("Connection error: {0}".format(exc)) from exc
            else:
                if (
                    response.status_code in RETRYABLE_STATUS_CODES
                    and attempt < attempts - 1
                ):
                    time.sleep(min(2 ** attempt, 5))
                    continue
                if response.status_code >= 400:
                    self._handle_error(response)
                return response

            if attempt < attempts - 1:
                time.sleep(min(2 ** attempt, 5))

        if last_error is not None:
            raise last_error
        raise APIError("Request failed before reaching the API")

    def _request_any(
        self,
        method: str,
        paths: Sequence[str],
        *,
        fallback_methods: Sequence[str] = (),
        **kwargs: Any
    ) -> requests.Response:
        last_error: Optional[TitanGPTException] = None

        for candidate_method in (method,) + tuple(fallback_methods):
            for path in paths:
                try:
                    return self._request(candidate_method, path, **kwargs)
                except NotFoundError as exc:
                    last_error = exc
                    continue

        if last_error is not None:
            raise last_error
        raise APIError("No request paths were provided")

    def _process_response(self, response: requests.Response) -> ResponseData:
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

    def _post(self, path: str, json: Optional[JsonObject] = None, data: Any = None) -> ResponseData:
        response = self._request("POST", path, json=json, data=data)
        return self._process_response(response)

    def _post_any(
        self,
        paths: Sequence[str],
        json: Optional[JsonObject] = None,
        data: Any = None,
        *,
        fallback_methods: Sequence[str] = (),
    ) -> ResponseData:
        response = self._request_any(
            "POST", paths, json=json, data=data, fallback_methods=fallback_methods
        )
        return self._process_response(response)

    def _get(self, path: str, params: Optional[JsonObject] = None) -> ResponseData:
        response = self._request("GET", path, params=params)
        return self._process_response(response)

    def _get_any(
        self,
        paths: Sequence[str],
        params: Optional[JsonObject] = None,
        *,
        fallback_methods: Sequence[str] = (),
    ) -> ResponseData:
        response = self._request_any(
            "GET", paths, params=params, fallback_methods=fallback_methods
        )
        return self._process_response(response)

    def _get_raw(self, path: str, params: Optional[JsonObject] = None) -> requests.Response:
        return self._request("GET", path, params=params)

    def _delete(self, path: str) -> ResponseData:
        response = self._request("DELETE", path)
        return self._process_response(response)

    def _upload_file(
        self, path: str, *, file: Any, data: Optional[JsonObject] = None
    ) -> ResponseData:
        opened_file = None
        try:
            if isinstance(file, (str, os.PathLike)):
                opened_file = open(os.fspath(file), "rb")
                file_payload = _build_file_payload(opened_file)
            else:
                file_payload = _build_file_payload(file)

            response = self._request(
                "POST",
                path,
                files={"file": file_payload},
                data=data,
            )
            return self._process_response(response)
        finally:
            if opened_file is not None:
                opened_file.close()

    def _handle_error(self, response: requests.Response) -> None:
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

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "TitanGPT":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
