import os
import json
from typing import Any, Dict, Optional, List, Union
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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

class Completions:
    def __init__(self, client):
        self._client = client

    def create(self, model: str, messages: List[Dict[str, str]], **kwargs) -> TitanResponse:
        payload = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        return self._client._post("v1/chat/completions", json=payload)

class Chat:
    def __init__(self, client):
        self.completions = Completions(client)

class Images:
    def __init__(self, client):
        self._client = client

    def generate(self, prompt: str, model: str = "flux", n: int = 1, size: str = "1024x1024", **kwargs) -> TitanResponse:
        payload = {
            "prompt": prompt,
            "model": model,
            "n": n,
            "size": size,
            **kwargs
        }
        return self._client._post("v1/images/generations", json=payload)

class Audio:
    def __init__(self, client):
        self.transcriptions = Transcriptions(client)

class Transcriptions:
    def __init__(self, client):
        self._client = client

    def create(self, file, model: str = "whisper-1", **kwargs) -> TitanResponse:
        if isinstance(file, str):
             with open(file, "rb") as f:
                 files = {"file": f}
                 data = {"model": model, **kwargs}
                 return self._client._post("v1/audio/transcriptions", files=files, data=data)
        
        files = {"file": file}
        data = {"model": model, **kwargs}
        return self._client._post("v1/audio/transcriptions", files=files, data=data)

class Music:
    def __init__(self, client):
        self._client = client
    
    def search(self, query: str) -> TitanResponse:
        return self._client._post("v2/music/search", json={"query": query})

    def lyrics(self, video_id: str) -> TitanResponse:
        return self._client._get(f"v2/music/lyrics/{video_id}")

    def download(self, video_id: str, save_path: str) -> str:
        response = self._client._get_binary(f"v2/music/download/{video_id}")
        
        if os.path.isdir(save_path):
            filename = f"{video_id}.mp3"
            save_path = os.path.join(save_path, filename)

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return save_path

class Models:
    def __init__(self, client):
        self._client = client

    def list(self) -> TitanResponse:
        return self._client._post("v1/models")

class TitanGPT:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.titangpt.ru",
        timeout: int = 60,
        max_retries: int = 3,
        user_id: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("TITANGPT_API_KEY")
        if not self.api_key:
            raise ValueError("The api_key client option must be set either by passing api_key to the client or by setting the TITANGPT_API_KEY environment variable")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        
        self.session = requests.Session()
        
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        auth_val = f"Bearer {self.api_key}"
        headers = {
            "Authorization": auth_val.encode('utf-8'), 
            "User-Agent": "TitanGPT-Python/1.0",
        }
        if user_id:
            headers["x-user-id"] = str(user_id)
            
        self.session.headers.update(headers)

        self.chat = Chat(self)
        self.images = Images(self)
        self.audio = Audio(self)
        self.music = Music(self)
        self.models = Models(self)

    def check_health(self) -> Dict[str, str]:
        url = f"{self.base_url}/" 
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                raise APIError(f"Health check failed with status {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            raise APIError(f"Health check failed: {str(e)}")

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self.base_url}/{path}"
        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            if response.status_code >= 400:
                self._handle_error(response)
            return response
        except requests.exceptions.RequestException as e:
            if isinstance(e, TitanGPTException):
                raise e
            raise APIError(f"Connection error: {str(e)}") from e
        except Exception as e:
            if isinstance(e, TitanGPTException):
                raise e
            raise APIError(f"Unexpected error: {str(e)}")

    def _post(self, path: str, json: dict = None, files=None, data=None) -> TitanResponse:
        response = self._request("POST", path, json=json, files=files, data=data)
        return TitanResponse(response.json())

    def _get(self, path: str, params: dict = None) -> TitanResponse:
        response = self._request("GET", path, params=params)
        return TitanResponse(response.json())

    def _get_binary(self, path: str) -> requests.Response:
        url = f"{self.base_url}/{path}"
        try:
            response = self.session.get(url, stream=True, timeout=self.timeout * 3)
            if response.status_code >= 400:
                self._handle_error(response)
            return response
        except requests.exceptions.RequestException as e:
            if isinstance(e, TitanGPTException):
                raise e
            raise APIError(f"Connection error: {str(e)}") from e

    def _handle_error(self, response):
        try:
            error_json = response.json()
            message = error_json.get("error", {}).get("message") or error_json.get("message")
            if not message and "detail" in error_json:
                message = error_json["detail"]
        except ValueError:
            message = response.text

        if not message:
            message = f"Error code: {response.status_code}"

        if response.status_code == 400:
            raise ValidationError(message)
        elif response.status_code == 401:
            raise AuthenticationError(f"Authentication failed: {message}")
        elif response.status_code == 403:
            raise AuthenticationError(f"Permission denied (Invalid API Key or Model): {message}")
        elif response.status_code == 404:
            raise ModelNotFoundError(message)
        elif response.status_code == 429:
            raise RateLimitError(message)
        else:
            raise APIError(f"TitanGPT API Error {response.status_code}: {message}")

    def close(self):
        self.session.close()

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
