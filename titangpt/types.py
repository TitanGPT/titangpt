from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class Message:
    role: str
    content: str
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:

    
    id: str
    title: str
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:

    
    name: str
    model_id: str
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 30
    retry_attempts: int = 10
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIResponse:

    
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    status_code: int = 200
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class User:

    
    id: str
    username: str
    email: str
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


@dataclass
class Token:

    
    access_token: str
    token_type: str = "Bearer"
    expires_in: Optional[int] = None
    refresh_token: Optional[str] = None
    scope: Optional[str] = None
    issued_at: datetime = field(default_factory=datetime.now)


@dataclass
class CompletionRequest:

    
    prompt: str
    model: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop_sequences: List[str] = field(default_factory=list)
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionResponse:

    
    id: str
    text: str
    model: str
    tokens_used: int
    finish_reason: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Error:

    
    code: str
    message: str
    details: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    trace_id: Optional[str] = None
