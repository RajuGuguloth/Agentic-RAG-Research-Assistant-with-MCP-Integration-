"""Process-wide retriever and index lifecycle shared by the app and API."""

from __future__ import annotations

import os
import logging
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Callable, Dict, List

from langchain_core.documents import Document

from retrieval.dense_retriever import DenseRetriever
from retrieval.multimodal_retriever import MultimodalRetriever
from retrieval.sparse_retriever import SparseRetriever

log = logging.getLogger("rag_runtime")

_runtime_lock = threading.RLock()
_hybrid_retriever: MultimodalRetriever | None = None
_index_store_path = Path(
    os.getenv(
        "RAG_INDEX_PATH",
        str(Path(__file__).resolve().parents[1] / "index_store"),
    )
).expanduser().resolve()


class CollectionError(RuntimeError):
    """Base error for collection index lifecycle failures."""


class InvalidCollectionIDError(CollectionError, ValueError):
    """The collection identifier is not safe to use as a directory name."""


class CollectionNotFoundError(CollectionError, FileNotFoundError):
    """The requested collection does not have a persisted index."""


class EmptyCollectionError(CollectionError, ValueError):
    """The collection exists but does not contain a complete usable index."""


class CollectionIndexManager:
    """Manage lazily loaded, independently persisted collection retrievers."""

    def __init__(self, index_root: Path | str | None = None):
        self._index_root = (
            Path(index_root).expanduser().resolve()
            if index_root is not None
            else _index_store_path
        )
        self._cache: Dict[str, MultimodalRetriever] = {}
        self._lock = threading.RLock()

    def get(self, collection_id: str) -> MultimodalRetriever:
        """Return a cached or persisted retriever for a collection."""
        collection_id = self._validate_collection_id(collection_id)
        with self._lock:
            cached = self._cache.get(collection_id)
            if cached is not None:
                return cached

            collection_path = self._collection_path(collection_id)
            if not collection_path.is_dir():
                raise CollectionNotFoundError(
                    f"Collection '{collection_id}' does not exist."
                )

            retriever = self._new_retriever()
            if not retriever.load(str(collection_path)):
                raise EmptyCollectionError(
                    f"Collection '{collection_id}' has no complete index."
                )
            self._cache[collection_id] = retriever
            return retriever

    def build_or_replace(
        self,
        collection_id: str,
        parent_docs: List[Document],
        child_docs: List[Document],
    ) -> dict:
        """Build and atomically replace one collection index."""
        collection_id = self._validate_collection_id(collection_id)
        if not parent_docs or not child_docs:
            raise EmptyCollectionError(
                f"Collection '{collection_id}' requires parent and child documents."
            )

        with self._lock:
            collection_path = self._collection_path(collection_id)
            candidate = self._new_retriever()
            candidate.dense.build_index(list(child_docs))
            candidate.sparse.build_index(list(child_docs))
            candidate.add_parents(list(parent_docs))

            def validate_persisted_index() -> None:
                persisted = self._new_retriever()
                if not persisted.load(str(collection_path)):
                    raise EmptyCollectionError(
                        f"Collection '{collection_id}' failed persistence validation."
                    )

            _persist_candidate(
                candidate,
                collection_path,
                validate=validate_persisted_index,
            )
            self._cache[collection_id] = candidate

            from retrieval.query_cache import get_query_cache

            get_query_cache().clear()
            return {
                "collection_id": collection_id,
                "parents_indexed": len(parent_docs),
                "children_indexed": len(child_docs),
                "index_path": str(collection_path),
            }

    def remove(self, collection_id: str) -> None:
        """Remove one collection from the cache and persisted storage."""
        collection_id = self._validate_collection_id(collection_id)
        with self._lock:
            collection_path = self._collection_path(collection_id)
            if not collection_path.exists():
                raise CollectionNotFoundError(
                    f"Collection '{collection_id}' does not exist."
                )
            if collection_path.is_symlink():
                raise CollectionError(
                    f"Collection '{collection_id}' resolves to a symlink."
                )
            shutil.rmtree(collection_path)
            self._cache.pop(collection_id, None)

    def _new_retriever(self) -> MultimodalRetriever:
        return MultimodalRetriever(DenseRetriever(), SparseRetriever())

    def _collection_path(self, collection_id: str) -> Path:
        return self._index_root / collection_id

    @staticmethod
    def _validate_collection_id(collection_id: str) -> str:
        if not isinstance(collection_id, str):
            raise InvalidCollectionIDError("collection_id must be a string.")
        value = collection_id.strip()
        if (
            not value
            or len(value) > 64
            or value[0] not in "abcdefghijklmnopqrstuvwxyz0123456789"
            or value[-1] not in "abcdefghijklmnopqrstuvwxyz0123456789"
            or any(
                char not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
                for char in value
            )
        ):
            raise InvalidCollectionIDError(
                "collection_id must contain 1-64 lowercase letters, digits, "
                "hyphens, or underscores and start and end with a letter or digit."
            )
        return value


def get_index_store_path() -> Path:
    return _index_store_path


def get_runtime_lock() -> threading.RLock:
    return _runtime_lock


def get_hybrid_retriever() -> MultimodalRetriever:
    """Load the persisted retrieval engine once in this process."""
    global _hybrid_retriever
    with _runtime_lock:
        if _hybrid_retriever is None:
            dense = DenseRetriever()
            sparse = SparseRetriever()
            retriever = MultimodalRetriever(dense, sparse)
            if retriever.load(str(_index_store_path)):
                print(f"[rag_runtime] Restored index from {_index_store_path}")
            else:
                print(f"[rag_runtime] No persisted index at {_index_store_path}")
            _hybrid_retriever = retriever
        return _hybrid_retriever


def rebuild_index(
    parent_docs: List[Document],
    child_docs: List[Document],
    *,
    replace: bool = True,
) -> dict:
    """Build and persist an index before swapping it into the active runtime."""
    if not parent_docs or not child_docs:
        raise ValueError("Cannot index an empty set of parent or child documents.")

    retriever = get_hybrid_retriever()
    with _runtime_lock:
        if replace or not retriever.dense.is_ready:
            indexed_parents = list(parent_docs)
            indexed_children = list(child_docs)
        else:
            indexed_parents = list(retriever.parent_store.values()) + list(parent_docs)
            indexed_children = list(retriever.dense.documents) + list(child_docs)

        candidate = _new_candidate_retriever(retriever)
        candidate.dense.build_index(indexed_children)
        candidate.sparse.build_index(indexed_children)
        candidate.parent_store = {}
        candidate.add_parents(indexed_parents)

        _persist_candidate(candidate)

        retriever.dense = candidate.dense
        retriever.sparse = candidate.sparse
        retriever.parent_store = candidate.parent_store

        from retrieval.query_cache import get_query_cache

        get_query_cache().clear()

        return {
            "parents_indexed": len(parent_docs),
            "children_indexed": len(child_docs),
            "total_parents": len(indexed_parents),
            "total_children": len(indexed_children),
            "index_path": str(_index_store_path),
            "replace": replace,
        }


def _new_candidate_retriever(retriever):
    """Create fresh index containers while reusing loaded model instances."""
    import copy

    candidate = copy.copy(retriever)
    candidate.dense = copy.copy(retriever.dense)
    candidate.dense.index = None
    candidate.dense.documents = []
    candidate.dense.dimension = None

    candidate.sparse = copy.copy(retriever.sparse)
    candidate.sparse.bm25 = None
    candidate.sparse.documents = []
    candidate.sparse.tokenized_corpus = []
    candidate.parent_store = {}
    return candidate


def _persist_candidate(
    candidate,
    index_path: Path | None = None,
    *,
    validate: Callable[[], None] | None = None,
) -> None:
    """Write an index to a staging directory and atomically replace the active one."""
    index_path = index_path or _index_store_path
    index_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = Path(
        tempfile.mkdtemp(prefix=".index-build-", dir=str(index_path.parent))
    )
    backup_path = index_path.with_name(
        f".{index_path.name}.backup-{uuid.uuid4().hex}"
    )
    moved_current = False

    try:
        candidate.save(str(staging_path))
        if index_path.exists():
            os.replace(index_path, backup_path)
            moved_current = True
        try:
            os.replace(staging_path, index_path)
        except Exception:
            if moved_current:
                os.replace(backup_path, index_path)
                moved_current = False
            raise
        if validate is not None:
            try:
                validate()
            except Exception as exc:
                if index_path.is_dir():
                    shutil.rmtree(index_path)
                elif index_path.exists():
                    index_path.unlink()
                if moved_current:
                    os.replace(backup_path, index_path)
                    moved_current = False
                raise CollectionError(
                    f"Persisted index validation failed for '{index_path.name}'."
                ) from exc
    finally:
        if staging_path.exists():
            shutil.rmtree(staging_path)

    if moved_current:
        try:
            shutil.rmtree(backup_path)
        except OSError:
            log.warning("Could not remove previous index backup at %s", backup_path)
