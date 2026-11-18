"""Public interface for the Agentic RAG SDK."""

from .client import ResearchAssistantClient
from .exceptions import (
    AnswerGenerationError,
    AuthenticationError,
    NetworkError,
    NotFoundError,
    ResearchAssistantError,
    ServiceUnavailableError,
    TimeoutError,
    ValidationError,
)
from .models import AskResponse, HealthResponse, SearchChunk, SearchResponse

__all__ = [
    "AnswerGenerationError",
    "AskResponse",
    "AuthenticationError",
    "HealthResponse",
    "NetworkError",
    "NotFoundError",
    "ResearchAssistantClient",
    "ResearchAssistantError",
    "SearchChunk",
    "SearchResponse",
    "ServiceUnavailableError",
    "TimeoutError",
    "ValidationError",
]
