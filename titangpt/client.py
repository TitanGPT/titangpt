import os
import json
from typing import Any, Dict, Optional, List, Union
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
        files = {"file": file}
        data = {"model": model, **kwargs}
        return self._client._post("v1/audio/transcriptions", files=files, data=data)


class Music:
    def __init__(self, client):
        self._client = client
    
    def search(self, query: str) -> TitanResponse:
        return self._client._post("v2/music/search", json={"query": query})



class TitanGPT:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.titangpt.ru",
        timeout: int = 60,
        max_retries: int = 3,
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
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        

        auth_val = f"Bearer {self.api_key}"
        self.session.headers.update({
            "Authorization": auth_val.encode('utf-8'), 
            "User-Agent": "TitanGPT-Python/1.0",
        })

        self.chat = Chat(self)
        self.images = Images(self)
        self.audio = Audio(self)
        self.music = Music(self)

    def _post(self, path: str, json: dict = None, files=None, data=None) -> TitanResponse:
        url = f"{self.base_url}/{path}"
        
        try:
            response = self.session.post(
                url, 
                json=json, 
                files=files, 
                data=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return TitanResponse(response.json())
            
        except requests.exceptions.HTTPError as e:
            try:
                err_text = response.json().get("error", {}).get("message", response.text)
            except:
                err_text = response.text
            raise Exception(f"API Error {response.status_code}: {err_text}") from e
        except Exception as e:
            raise e

    def close(self):
        self.session.close()

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
