"""Synchronous client for the Agentic RAG HTTP API."""

from typing import Any, Mapping, Optional

import httpx

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
from .models import AskResponse, HealthResponse, SearchResponse


class ResearchAssistantClient:
    """Call the supported health and question endpoints of the RAG service."""

    def __init__(self, base_url: str, api_key: str, timeout: float) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string.")
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("api_key must be a non-empty string.")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero.")

        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def health(self) -> HealthResponse:
        """Return the service and index health state."""
        return HealthResponse.from_mapping(self._request("GET", "/v1/health"))

    def ask(self, question: str, user_id: Optional[str] = None) -> AskResponse:
        """Ask the indexed documents a question."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string.")
        if len(question) > 8000:
            raise ValueError("question must be at most 8000 characters.")
        if user_id is not None:
            if not isinstance(user_id, str) or not user_id.strip():
                raise ValueError("user_id must be a non-empty string when provided.")
            if len(user_id) > 128:
                raise ValueError("user_id must be at most 128 characters.")

        payload = {"question": question}
        if user_id is not None:
            payload["user_id"] = user_id
        return AskResponse.from_mapping(self._request("POST", "/v1/query", json=payload))

    def search(self, collection: str, query: str, top_k: int = 6) -> SearchResponse:
        """Return ranked chunks retrieved for a query."""
        if not isinstance(collection, str) or not collection.strip():
            raise ValueError("collection must be a non-empty string.")
        if len(collection) > 64:
            raise ValueError("collection must be at most 64 characters.")
        if (
            collection != collection.strip()
            or not collection[0].isalnum()
            or not collection[-1].isalnum()
            or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in collection)
        ):
            raise ValueError(
                "collection must contain lowercase letters, digits, hyphens, or underscores."
            )
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if len(query) > 8000:
            raise ValueError("query must be at most 8000 characters.")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 20:
            raise ValueError("top_k must be an integer between 1 and 20.")

        return SearchResponse.from_mapping(
            self._request(
                "POST",
                "/v1/search",
                json={"collection": collection, "query": query, "top_k": top_k},
            )
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        try:
            response = httpx.request(
                method,
                f"{self._base_url}{path}",
                headers={"X-API-Key": self._api_key},
                json=json,
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise TimeoutError("RAG API request timed out.") from exc
        except httpx.RequestError as exc:
            raise NetworkError("RAG API request failed.") from exc

        if response.status_code == 401:
            raise AuthenticationError("RAG API authentication failed.")
        if response.status_code == 404:
            raise NotFoundError("RAG API route was not found.")
        if response.status_code == 422:
            raise ValidationError("RAG API request validation failed.")
        if response.status_code == 502:
            raise AnswerGenerationError("RAG API answer generation failed.")
        if response.status_code == 503:
            raise ServiceUnavailableError("RAG API service is unavailable.")
        if response.is_error:
            raise ResearchAssistantError(
                f"RAG API request failed with HTTP {response.status_code}."
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise ResearchAssistantError("RAG API returned invalid JSON.") from exc
        if not isinstance(payload, Mapping):
            raise ResearchAssistantError("RAG API returned an invalid response.")
        return payload
