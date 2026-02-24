"""
schemas.py
----------
Pydantic request and response models for the API.
No business logic — purely data shape definitions.
"""

from pydantic import BaseModel, Field
from config import TOP_K


class Message(BaseModel):
    """A single conversation turn."""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    """Incoming request body for POST /chat."""
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[Message] = Field(default_factory=list)
    top_k: int = Field(default=TOP_K, ge=1, le=10)


class HealthResponse(BaseModel):
    status: str


class ConfigResponse(BaseModel):
    model: str
    top_k: int
    index_loaded: bool
