import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestCollectionAuthorization(unittest.TestCase):
    def setUp(self):
        from api.authorization import authorize_collection

        self.authorize_collection = authorize_collection

    def _environment(self, **overrides):
        environment = {
            "RAG_API_KEY": "test-key",
            "RAG_COLLECTION_PERMISSIONS": json.dumps(
                {"test-key": ["collection-a"]}
            ),
        }
        environment.update(overrides)
        return patch.dict(os.environ, environment, clear=True)

    def test_allows_permitted_collection(self):
        with self._environment():
            self.authorize_collection("test-key", "collection-a")

    def test_denies_collection_without_access(self):
        with self._environment():
            with self.assertRaisesRegex(HTTPException, "not authorized"):
                self.authorize_collection("test-key", "collection-b")

    def test_rejects_invalid_api_key(self):
        with self._environment():
            with self.assertRaises(HTTPException) as context:
                self.authorize_collection("wrong-key", "collection-a")
        self.assertEqual(context.exception.status_code, 401)

    def test_rejects_missing_api_key_configuration(self):
        with self._environment(RAG_API_KEY=""):
            with self.assertRaises(HTTPException) as context:
                self.authorize_collection("test-key", "collection-a")
        self.assertEqual(context.exception.status_code, 503)

    def test_rejects_missing_authorization_configuration(self):
        with self._environment(RAG_COLLECTION_PERMISSIONS=""):
            from api.authorization import CollectionAuthorizationConfigError

            with self.assertRaises(CollectionAuthorizationConfigError):
                self.authorize_collection("test-key", "collection-a")

    def test_rejects_malformed_authorization_configuration(self):
        with self._environment(RAG_COLLECTION_PERMISSIONS="{not-json"):
            from api.authorization import CollectionAuthorizationConfigError

            with self.assertRaises(CollectionAuthorizationConfigError):
                self.authorize_collection("test-key", "collection-a")

    def test_invalid_collection_is_rejected_before_authorization_lookup(self):
        with self._environment(RAG_COLLECTION_PERMISSIONS="{not-json"):
            from retrieval.runtime import InvalidCollectionIDError

            with self.assertRaises(InvalidCollectionIDError):
                self.authorize_collection("test-key", "../outside")

    def test_api_key_is_not_exposed_in_errors(self):
        secret = "super-secret-test-key"
        with self._environment(RAG_API_KEY=secret):
            with self.assertRaises(HTTPException) as context:
                self.authorize_collection("wrong-key", "collection-a")
        self.assertNotIn(secret, str(context.exception))
        self.assertNotIn("wrong-key", str(context.exception))
