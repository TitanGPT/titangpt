"""
Synchronous client for TitanGPT API.
"""

import os
from typing import Any, Dict, List, Optional, Union
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class TitanGPTClient:
    """
    Synchronous client for interacting with the TitanGPT API.
    
    This client provides methods for making requests to the TitanGPT API
    with built-in retry logic and error handling.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.titangpt.com/v1",
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize the TitanGPT synchronous client.
        
        Args:
            api_key: API key for authentication. Defaults to TITANGPT_API_KEY environment variable.
            base_url: Base URL for the API endpoint.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
        """
        self.api_key = api_key or os.getenv("TITANGPT_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required. Set TITANGPT_API_KEY environment variable or pass api_key."
            )
        
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        
        # Create session with retry strategy
        self.session = self._create_session(max_retries)
    
    def _create_session(self, max_retries: int) -> requests.Session:
        """
        Create a requests session with retry strategy.
        
        Args:
            max_retries: Maximum number of retries.
            
        Returns:
            Configured requests.Session object.
        """
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers
        session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "TitanGPT-Python-Client/1.0",
        })
        
        return session
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Handle API response and raise exceptions for errors.
        
        Args:
            response: The response object from requests.
            
        Returns:
            Parsed JSON response.
            
        Raises:
            requests.exceptions.HTTPError: If response status is 4xx or 5xx.
        """
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
        """
        Make a request to the API.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE).
            endpoint: API endpoint path.
            **kwargs: Additional arguments to pass to requests.
            
        Returns:
            Parsed JSON response.
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Set timeout if not provided
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
        """
        Make a GET request to the API.
        
        Args:
            endpoint: API endpoint path.
            params: Query parameters.
            **kwargs: Additional arguments to pass to requests.
            
        Returns:
            Parsed JSON response.
        """
        return self._request("GET", endpoint, params=params, **kwargs)
    
    def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make a POST request to the API.
        
        Args:
            endpoint: API endpoint path.
            data: Form data.
            json: JSON body data.
            **kwargs: Additional arguments to pass to requests.
            
        Returns:
            Parsed JSON response.
        """
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
        """
        Make a PUT request to the API.
        
        Args:
            endpoint: API endpoint path.
            data: Form data.
            json: JSON body data.
            **kwargs: Additional arguments to pass to requests.
            
        Returns:
            Parsed JSON response.
        """
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
        """
        Make a DELETE request to the API.
        
        Args:
            endpoint: API endpoint path.
            **kwargs: Additional arguments to pass to requests.
            
        Returns:
            Parsed JSON response.
        """
        return self._request("DELETE", endpoint, **kwargs)
    
    def close(self) -> None:
        """Close the session and clean up resources."""
        if self.session:
            self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
