"""Typed response models for the Agentic RAG SDK."""

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional


@dataclass(frozen=True)
class HealthResponse:
    """Health and index state returned by the RAG service."""

    status: str
    index_ready: bool
    index_path: str
    children_indexed: int
    parents_stored: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "HealthResponse":
        return cls(
            status=payload["status"],
            index_ready=payload["index_ready"],
            index_path=payload["index_path"],
            children_indexed=payload["children_indexed"],
            parents_stored=payload["parents_stored"],
        )


@dataclass(frozen=True)
class AskResponse:
    """Answer, source information, and trace ID returned by the query API."""

    answered: bool
    answer: str
    sources: str
    trace_id: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AskResponse":
        return cls(
            answered=payload["answered"],
            answer=payload["answer"],
            sources=payload["sources"],
            trace_id=payload["trace_id"],
        )


@dataclass(frozen=True)
class SearchChunk:
    """One retrieved chunk returned by the search API."""

    text: str
    source: Optional[str]
    source_id: Optional[str]
    page: Optional[int]
    retrieval_score: Optional[float]
    reranker_score: Optional[float]
    child_rrf_score: Optional[float]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SearchChunk":
        return cls(
            text=payload["text"],
            source=payload["source"],
            source_id=payload["source_id"],
            page=payload["page"],
            retrieval_score=payload["retrieval_score"],
            reranker_score=payload["reranker_score"],
            child_rrf_score=payload["child_rrf_score"],
        )


@dataclass(frozen=True)
class SearchResponse:
    """Ranked chunks returned by the search API."""

    query: str
    chunks: List[SearchChunk]
    count: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SearchResponse":
        return cls(
            query=payload["query"],
            chunks=[
                SearchChunk.from_mapping(chunk)
                for chunk in payload["chunks"]
            ],
            count=payload["count"],
        )
