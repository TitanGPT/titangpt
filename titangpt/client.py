import os
from typing import Any, Dict, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class TitanGPTClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.titangpt.ru",
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.getenv("TITANGPT_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required. Set TITANGPT_API_KEY environment variable or pass api_key."
            )
        
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = self._create_session(max_retries)
    
    def _create_session(self, max_retries: int) -> requests.Session:
        session = requests.Session()
        
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "TitanGPT-Python-Client/1.0",
        })
        
        return session
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            error_message = f"API Error: {response.status_code} - {response.text}"
            raise requests.exceptions.HTTPError(error_message) from e
        
        return response.json() if response.content else {}
    
    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.timeout
        
        response = self.session.request(method, url, **kwargs)
        return self._handle_response(response)
    
    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return self._request("GET", endpoint, params=params, **kwargs)
    
    def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            endpoint,
            data=data,
            json=json,
            **kwargs,
        )
    
    def put(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return self._request(
            "PUT",
            endpoint,
            data=data,
            json=json,
            **kwargs,
        )
    
    def delete(
        self,
        endpoint: str,
        **kwargs,
    ) -> Dict[str, Any]:
        return self._request("DELETE", endpoint, **kwargs)

    def chat_completions(self, json: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self.post("v1/chat/completions", json=json, **kwargs)
    
    def close(self) -> None:
        if self.session:
            self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
