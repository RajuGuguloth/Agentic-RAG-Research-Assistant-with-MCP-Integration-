"""Exceptions raised by the Agentic RAG SDK."""


class ResearchAssistantError(Exception):
    """Base exception for SDK failures."""


class AuthenticationError(ResearchAssistantError):
    """The API key was rejected by the server."""


class NotFoundError(ResearchAssistantError):
    """The requested API route was not found."""


class ValidationError(ResearchAssistantError):
    """The request failed API validation."""


class AnswerGenerationError(ResearchAssistantError):
    """The server failed while generating an answer."""


class ServiceUnavailableError(ResearchAssistantError):
    """The RAG service or index is unavailable."""


class NetworkError(ResearchAssistantError):
    """The SDK could not communicate with the server."""


class TimeoutError(NetworkError):
    """The request exceeded the configured timeout."""
