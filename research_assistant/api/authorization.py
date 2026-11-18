"""Prototype API-key authorization for collection-scoped operations."""

from __future__ import annotations

import hmac
import json
import os
from typing import Dict, List

from fastapi import HTTPException

from retrieval.runtime import CollectionIndexManager


COLLECTION_PERMISSIONS_ENV = "RAG_COLLECTION_PERMISSIONS"


class CollectionAuthorizationConfigError(RuntimeError):
    """Collection authorization is missing or malformed."""


def authorize_collection(api_key: str | None, collection_id: str) -> None:
    """Authenticate an API key and authorize it for one collection.

    ``RAG_COLLECTION_PERMISSIONS`` must be a JSON object whose keys are API
    keys and whose values are arrays of permitted collection IDs, for example:
    ``{"local-dev-key": ["papers", "notes"]}``.
    """
    collection_id = CollectionIndexManager._validate_collection_id(collection_id)
    expected_key = os.getenv("RAG_API_KEY", "")
    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="RAG API authentication is not configured. Set RAG_API_KEY.",
        )
    if not api_key or not hmac.compare_digest(api_key, expected_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

    permissions = _load_collection_permissions()
    allowed_collections = permissions.get(expected_key, [])
    if collection_id not in allowed_collections:
        raise HTTPException(
            status_code=403,
            detail="API key is not authorized for this collection.",
        )


def _load_collection_permissions() -> Dict[str, List[str]]:
    raw_permissions = os.getenv(COLLECTION_PERMISSIONS_ENV)
    if not raw_permissions:
        raise CollectionAuthorizationConfigError(
            f"{COLLECTION_PERMISSIONS_ENV} is not configured."
        )
    try:
        parsed = json.loads(raw_permissions)
    except json.JSONDecodeError as exc:
        raise CollectionAuthorizationConfigError(
            f"{COLLECTION_PERMISSIONS_ENV} must contain valid JSON."
        ) from exc

    if not isinstance(parsed, dict) or not parsed:
        raise CollectionAuthorizationConfigError(
            f"{COLLECTION_PERMISSIONS_ENV} must be a non-empty JSON object."
        )

    permissions: Dict[str, List[str]] = {}
    for api_key, collections in parsed.items():
        if (
            not isinstance(api_key, str)
            or not api_key
            or not isinstance(collections, list)
            or not collections
            or not all(isinstance(collection, str) for collection in collections)
        ):
            raise CollectionAuthorizationConfigError(
                f"{COLLECTION_PERMISSIONS_ENV} must map API keys to "
                "non-empty arrays of collection IDs."
            )
        try:
            permissions[api_key] = [
                CollectionIndexManager._validate_collection_id(collection)
                for collection in collections
            ]
        except ValueError as exc:
            raise CollectionAuthorizationConfigError(
                f"{COLLECTION_PERMISSIONS_ENV} contains an invalid collection ID."
            ) from exc
    return permissions
