import unittest
from unittest.mock import patch

import httpx

from rag_sdk import (
    AskResponse,
    AuthenticationError,
    HealthResponse,
    ResearchAssistantClient,
    SearchChunk,
    SearchResponse,
    TimeoutError,
    ValidationError,
)


class TestResearchAssistantClient(unittest.TestCase):
    def setUp(self):
        self.client = ResearchAssistantClient(
            "http://localhost:7860/",
            "secret-api-key",
            timeout=12.5,
        )

    @patch("rag_sdk.client.httpx.request")
    def test_health_request_and_typed_response(self, request):
        request.return_value = httpx.Response(
            200,
            json={
                "status": "ready",
                "index_ready": True,
                "index_path": "/data/index_store",
                "children_indexed": 6,
                "parents_stored": 3,
            },
        )

        result = self.client.health()

        self.assertIsInstance(result, HealthResponse)
        self.assertEqual(result.status, "ready")
        request.assert_called_once_with(
            "GET",
            "http://localhost:7860/v1/health",
            headers={"X-API-Key": "secret-api-key"},
            json=None,
            timeout=12.5,
        )

    @patch("rag_sdk.client.httpx.request")
    def test_ask_request_headers_and_typed_response(self, request):
        request.return_value = httpx.Response(
            200,
            json={
                "answered": True,
                "answer": "The policy allows paid leave.",
                "sources": "policy.pdf, page 2",
                "trace_id": "trace-1",
            },
        )

        result = self.client.ask("What is the leave policy?", user_id="employee-1")

        self.assertIsInstance(result, AskResponse)
        self.assertEqual(result.answer, "The policy allows paid leave.")
        request.assert_called_once_with(
            "POST",
            "http://localhost:7860/v1/query",
            headers={"X-API-Key": "secret-api-key"},
            json={
                "question": "What is the leave policy?",
                "user_id": "employee-1",
            },
            timeout=12.5,
        )

    @patch("rag_sdk.client.httpx.request")
    def test_invalid_api_key_is_converted_without_exposing_key(self, request):
        request.return_value = httpx.Response(401, json={"detail": "Invalid API key."})

        with self.assertRaises(AuthenticationError) as raised:
            self.client.health()

        self.assertNotIn("secret-api-key", str(raised.exception))

    @patch("rag_sdk.client.httpx.request")
    def test_validation_error_is_converted(self, request):
        request.return_value = httpx.Response(422, json={"detail": "Invalid request."})

        with self.assertRaises(ValidationError):
            self.client.ask("What is the leave policy?")

    @patch("rag_sdk.client.httpx.request")
    def test_timeout_is_converted_without_exposing_api_key(self, request):
        request.side_effect = httpx.ReadTimeout(
            "timed out",
            request=httpx.Request("GET", "http://localhost:7860/v1/health"),
        )

        with self.assertRaises(TimeoutError) as raised:
            self.client.health()

        self.assertNotIn("secret-api-key", str(raised.exception))

    def test_invalid_question_is_rejected_locally(self):
        with self.assertRaises(ValueError):
            self.client.ask("   ")

    @patch("rag_sdk.client.httpx.request")
    def test_search_request_and_typed_response(self, request):
        request.return_value = httpx.Response(
            200,
            json={
                "query": "leave policy",
                "chunks": [
                    {
                        "text": "Employees receive paid leave.",
                        "source": "policy.pdf",
                        "source_id": "parent-1",
                        "page": 2,
                        "retrieval_score": 0.81,
                        "reranker_score": 0.74,
                        "child_rrf_score": 0.04,
                    }
                ],
                "count": 1,
            },
        )

        result = self.client.search("hr-policies", "leave policy", top_k=3)

        self.assertIsInstance(result, SearchResponse)
        self.assertIsInstance(result.chunks[0], SearchChunk)
        self.assertEqual(result.chunks[0].source_id, "parent-1")
        self.assertEqual(result.chunks[0].reranker_score, 0.74)
        request.assert_called_once_with(
            "POST",
            "http://localhost:7860/v1/search",
            headers={"X-API-Key": "secret-api-key"},
            json={
                "collection": "hr-policies",
                "query": "leave policy",
                "top_k": 3,
            },
            timeout=12.5,
        )

    def test_search_rejects_invalid_values_locally(self):
        with self.assertRaises(ValueError):
            self.client.search("", "query")
        with self.assertRaises(ValueError):
            self.client.search("../outside", "query")
        with self.assertRaises(ValueError):
            self.client.search("collection-a", "   ")
        with self.assertRaises(ValueError):
            self.client.search("collection-a", "query", top_k=0)
        with self.assertRaises(ValueError):
            self.client.search("collection-a", "query", top_k=21)
        with self.assertRaises(ValueError):
            self.client.search("collection-a", "query", top_k=True)

    def test_search_requires_collection(self):
        with self.assertRaises(TypeError):
            self.client.search("leave policy")

    @patch("rag_sdk.client.httpx.request")
    def test_search_authentication_error_does_not_expose_api_key(self, request):
        request.return_value = httpx.Response(401, json={"detail": "Invalid API key."})

        with self.assertRaises(AuthenticationError) as raised:
            self.client.search("hr-policies", "leave policy")

        self.assertNotIn("secret-api-key", str(raised.exception))
