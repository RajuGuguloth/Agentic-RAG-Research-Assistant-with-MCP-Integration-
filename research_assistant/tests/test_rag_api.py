import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestRagApiAuth(unittest.TestCase):
    def setUp(self):
        from api import rag_router

        self.rag_router = rag_router
        self.api = FastAPI()
        self.api.include_router(rag_router.router)
        self.client = TestClient(self.api)

    def test_ingest_request_accepts_legacy_and_generic_path_fields(self):
        legacy = self.rag_router.IngestRequest(pdf_paths=["/tmp/paper.pdf"])
        current = self.rag_router.IngestRequest(file_paths=["/tmp/notes.docx"])
        self.assertEqual(legacy.file_paths, ["/tmp/paper.pdf"])
        self.assertEqual(current.file_paths, ["/tmp/notes.docx"])

    def test_ingest_accepts_a_supported_file_inside_configured_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "notes.txt"
            source.write_text("approved content", encoding="utf-8")
            with patch.dict(os.environ, {"RAG_API_KEY": "test-key"}), patch.object(
                self.rag_router, "INGEST_ROOT", root.resolve()
            ), patch.object(
                self.rag_router,
                "_index_paths",
                return_value={"total_children": 1, "files": ["notes.txt"]},
            ) as index_paths:
                response = self.client.post(
                    "/v1/ingest",
                    headers={"X-API-Key": "test-key"},
                    json={"file_paths": [str(source)]},
                )

        self.assertEqual(response.status_code, 200, response.text)
        index_paths.assert_called_once()

    def test_ingest_rejects_files_outside_configured_root(self):
        with tempfile.TemporaryDirectory() as root_dir, tempfile.TemporaryDirectory() as outside_dir:
            source = Path(outside_dir) / "notes.txt"
            source.write_text("private content", encoding="utf-8")
            with patch.dict(os.environ, {"RAG_API_KEY": "test-key"}), patch.object(
                self.rag_router, "INGEST_ROOT", Path(root_dir)
            ), patch.object(self.rag_router, "_index_paths") as index_paths:
                response = self.client.post(
                    "/v1/ingest",
                    headers={"X-API-Key": "test-key"},
                    json={"file_paths": [str(source)]},
                )

        self.assertEqual(response.status_code, 403)
        index_paths.assert_not_called()

    def test_ingest_rejects_symlink_to_file_outside_configured_root(self):
        with tempfile.TemporaryDirectory() as root_dir, tempfile.TemporaryDirectory() as outside_dir:
            root = Path(root_dir)
            outside_file = Path(outside_dir) / "notes.txt"
            outside_file.write_text("private content", encoding="utf-8")
            link = root / "linked.txt"
            link.symlink_to(outside_file)
            with patch.dict(os.environ, {"RAG_API_KEY": "test-key"}), patch.object(
                self.rag_router, "INGEST_ROOT", root
            ), patch.object(self.rag_router, "_index_paths") as index_paths:
                response = self.client.post(
                    "/v1/ingest",
                    headers={"X-API-Key": "test-key"},
                    json={"file_paths": [str(link)]},
                )

        self.assertEqual(response.status_code, 403)
        index_paths.assert_not_called()

    def test_protected_route_rejects_missing_and_invalid_keys(self):
        with patch.dict(os.environ, {"RAG_API_KEY": "test-key"}):
            self.assertEqual(self.client.get("/v1/papers").status_code, 401)
            self.assertEqual(
                self.client.get("/v1/papers", headers={"X-API-Key": "wrong"}).status_code,
                401,
            )

    def test_protected_route_accepts_configured_key(self):
        fake = type(
            "FakeRetriever",
            (),
            {
                "parent_store": {
                    "p1": Document(
                        page_content="text",
                        metadata={"source": "paper.pdf"},
                    )
                }
            },
        )()
        with patch.dict(os.environ, {"RAG_API_KEY": "test-key"}), patch.object(
            self.rag_router, "get_hybrid_retriever", return_value=fake
        ):
            response = self.client.get(
                "/v1/papers",
                headers={"X-API-Key": "test-key"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["distinct_sources"], 1)
        self.assertEqual(response.json()["papers"][0]["source"], "paper.pdf")

    def test_protected_route_fails_closed_without_configured_key(self):
        with patch.dict(os.environ, {"RAG_API_KEY": ""}):
            response = self.client.get(
                "/v1/papers",
                headers={"X-API-Key": "anything"},
            )
        self.assertEqual(response.status_code, 503)

    def test_upload_indexes_pdf_bytes_through_shared_service(self):
        uploaded = []

        def capture_upload(collection, paths, **kwargs):
            uploaded.append(paths[0].read_bytes())
            return {"total_children": 1, "files": ["paper.pdf"]}

        with patch.dict(
            os.environ,
            {
                "RAG_API_KEY": "test-key",
                "RAG_COLLECTION_PERMISSIONS": '{"test-key":["collection-a"]}',
            },
        ), patch.object(
            self.rag_router,
            "_index_collection_paths",
            side_effect=capture_upload,
        ):
            response = self.client.post(
                "/v1/upload",
                headers={"X-API-Key": "test-key"},
                files=[("files", ("paper.pdf", b"%PDF-test content", "application/pdf"))],
                data={
                    "collection": "collection-a",
                    "replace": "true",
                    "with_vision": "false",
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["total_children"], 1)
        self.assertEqual(uploaded, [b"%PDF-test content"])

    def test_upload_accepts_local_text_documents(self):
        uploaded = []

        def capture_upload(collection, paths, **kwargs):
            uploaded.append((paths[0].suffix, paths[0].read_text(encoding="utf-8")))
            return {"total_children": 1, "files": ["notes.txt"]}

        with patch.dict(
            os.environ,
            {
                "RAG_API_KEY": "test-key",
                "RAG_COLLECTION_PERMISSIONS": '{"test-key":["collection-a"]}',
            },
        ), patch.object(
            self.rag_router,
            "_index_collection_paths",
            side_effect=capture_upload,
        ):
            response = self.client.post(
                "/v1/upload",
                headers={"X-API-Key": "test-key"},
                files=[("files", ("notes.txt", b"Local document text", "text/plain"))],
                data={"collection": "collection-a", "replace": "true"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(uploaded, [(".txt", "Local document text")])

    def test_upload_rejects_unsupported_document_formats(self):
        with patch.dict(
            os.environ,
            {
                "RAG_API_KEY": "test-key",
                "RAG_COLLECTION_PERMISSIONS": '{"test-key":["collection-a"]}',
            },
        ):
            response = self.client.post(
                "/v1/upload",
                headers={"X-API-Key": "test-key"},
                files=[("files", ("notes.rtf", b"content", "text/rtf"))],
                data={"collection": "collection-a"},
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported document format", response.json()["detail"])

    def test_search_returns_ranked_parent_chunks(self):
        from types import SimpleNamespace

        fake = type(
            "FakeRetriever",
            (),
            {
                "dense": SimpleNamespace(is_ready=True),
                "search": lambda self, query, k: [
                    Document(
                        page_content="evidence",
                        metadata={
                            "source": "paper.pdf",
                            "parent_id": "parent-1",
                            "page": 2,
                            "retrieval_score": 0.8,
                        },
                    )
                ],
            },
        )()
        with patch.dict(os.environ, {"RAG_API_KEY": "test-key"}), patch.object(
            self.rag_router.collection_index_manager,
            "get",
            return_value=fake,
        ):
            response = self.client.post(
                "/v1/search",
                headers={"X-API-Key": "test-key"},
                json={
                    "collection": "collection-a",
                    "query": "question",
                    "top_k": 1,
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["chunks"][0]["source_id"], "parent-1")
        self.assertEqual(response.json()["chunks"][0]["text"], "evidence")

    def test_search_rejects_invalid_collection_id(self):
        with patch.dict(os.environ, {"RAG_API_KEY": "test-key"}), patch.object(
            self.rag_router.collection_index_manager,
            "get",
        ) as get_collection:
            response = self.client.post(
                "/v1/search",
                headers={"X-API-Key": "test-key"},
                json={"collection": "../outside", "query": "question"},
            )
        self.assertEqual(response.status_code, 422)
        get_collection.assert_not_called()

    def test_search_returns_not_found_for_missing_collection(self):
        from retrieval.runtime import CollectionNotFoundError

        with patch.dict(os.environ, {"RAG_API_KEY": "test-key"}), patch.object(
            self.rag_router.collection_index_manager,
            "get",
            side_effect=CollectionNotFoundError("Collection 'missing' does not exist."),
        ):
            response = self.client.post(
                "/v1/search",
                headers={"X-API-Key": "test-key"},
                json={"collection": "missing", "query": "question"},
            )
        self.assertEqual(response.status_code, 404)

    def test_search_returns_conflict_for_empty_collection(self):
        from retrieval.runtime import EmptyCollectionError

        with patch.dict(os.environ, {"RAG_API_KEY": "test-key"}), patch.object(
            self.rag_router.collection_index_manager,
            "get",
            side_effect=EmptyCollectionError("Collection 'empty' has no complete index."),
        ):
            response = self.client.post(
                "/v1/search",
                headers={"X-API-Key": "test-key"},
                json={"collection": "empty", "query": "question"},
            )
        self.assertEqual(response.status_code, 409)

    def test_search_uses_requested_collection_without_global_fallback(self):
        from types import SimpleNamespace

        collection_retriever = type(
            "CollectionRetriever",
            (),
            {
                "dense": SimpleNamespace(is_ready=True),
                "search": lambda self, query, k: [],
            },
        )()
        global_retriever = type(
            "GlobalRetriever",
            (),
            {
                "dense": SimpleNamespace(is_ready=True),
                "search": lambda self, query, k: [
                    Document(page_content="global", metadata={})
                ],
            },
        )()
        with patch.dict(os.environ, {"RAG_API_KEY": "test-key"}), patch.object(
            self.rag_router.collection_index_manager,
            "get",
            return_value=collection_retriever,
        ) as get_collection, patch.object(
            self.rag_router,
            "get_hybrid_retriever",
            return_value=global_retriever,
        ) as get_global:
            response = self.client.post(
                "/v1/search",
                headers={"X-API-Key": "test-key"},
                json={"collection": "collection-a", "query": "question"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["chunks"], [])
        get_collection.assert_called_once_with("collection-a")
        get_global.assert_not_called()

    def test_search_isolates_two_collections(self):
        from types import SimpleNamespace

        def retriever_for(name):
            return type(
                f"{name.title()}Retriever",
                (),
                {
                    "dense": SimpleNamespace(is_ready=True),
                    "search": lambda self, query, k, name=name: [
                        Document(
                            page_content=f"{name} evidence",
                            metadata={"source": f"{name}.txt"},
                        )
                    ],
                },
            )()

        retrievers = {
            "collection-a": retriever_for("a"),
            "collection-b": retriever_for("b"),
        }
        with patch.dict(os.environ, {"RAG_API_KEY": "test-key"}), patch.object(
            self.rag_router.collection_index_manager,
            "get",
            side_effect=retrievers.__getitem__,
        ):
            first = self.client.post(
                "/v1/search",
                headers={"X-API-Key": "test-key"},
                json={"collection": "collection-a", "query": "question"},
            )
            second = self.client.post(
                "/v1/search",
                headers={"X-API-Key": "test-key"},
                json={"collection": "collection-b", "query": "question"},
            )

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(first.json()["chunks"][0]["source"], "a.txt")
        self.assertEqual(second.json()["chunks"][0]["source"], "b.txt")

    def test_search_requires_authentication(self):
        response = self.client.post(
            "/v1/search",
            json={"collection": "collection-a", "query": "question"},
        )
        self.assertEqual(response.status_code, 401)

    def test_query_returns_answer_source_and_trace(self):
        from types import SimpleNamespace

        fake = type("FakeRetriever", (), {"dense": SimpleNamespace(is_ready=True)})()
        expected = {
            "answer": "grounded answer",
            "sources": "- paper.pdf (page 2)",
            "trace_id": "trace-1",
        }
        with patch.dict(os.environ, {"RAG_API_KEY": "test-key"}), patch.object(
            self.rag_router, "get_hybrid_retriever", return_value=fake
        ), patch.object(self.rag_router, "_answer_question", return_value=expected):
            response = self.client.post(
                "/v1/query",
                headers={"X-API-Key": "test-key"},
                json={"question": "question"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"answered": True, **expected})


class TestSharedIndexRebuild(unittest.TestCase):
    def test_replace_and_append_keep_parent_child_index_consistent(self):
        from retrieval import runtime

        class FakeDense:
            documents = []
            is_ready = False

            def build_index(self, documents):
                self.documents = list(documents)
                self.is_ready = True

        class FakeSparse:
            def build_index(self, documents):
                self.documents = list(documents)

        class FakeRetriever:
            def __init__(self):
                self.dense = FakeDense()
                self.sparse = FakeSparse()
                self.parent_store = {"old": Document(page_content="old")}
                self.saved_path = None

            def add_parents(self, documents):
                for document in documents:
                    self.parent_store[document.metadata["parent_id"]] = document

            def save(self, path):
                self.saved_path = path

        retriever = FakeRetriever()
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            runtime, "_hybrid_retriever", retriever
        ), patch.object(runtime, "_index_store_path", Path(temp_dir)):
            parent_a = Document(page_content="new parent", metadata={"parent_id": "a"})
            child_a = Document(page_content="new child", metadata={"parent_id": "a"})
            replaced = runtime.rebuild_index([parent_a], [child_a], replace=True)
            self.assertEqual(replaced["total_parents"], 1)
            self.assertEqual(set(retriever.parent_store), {"a"})

            parent_b = Document(page_content="another parent", metadata={"parent_id": "b"})
            child_b = Document(page_content="another child", metadata={"parent_id": "b"})
            appended = runtime.rebuild_index([parent_b], [child_b], replace=False)
            self.assertEqual(appended["total_parents"], 2)
            self.assertEqual(appended["total_children"], 2)
            self.assertEqual(set(retriever.parent_store), {"a", "b"})
            self.assertEqual(len(retriever.dense.documents), len(retriever.sparse.documents))

    def test_failed_candidate_persistence_preserves_active_index_and_disk(self):
        from retrieval import runtime

        class FakeDense:
            documents = []
            is_ready = False

            def build_index(self, documents):
                self.documents = list(documents)
                self.is_ready = True

        class FakeSparse:
            def build_index(self, documents):
                self.documents = list(documents)

        class FakeRetriever:
            def __init__(self):
                self.dense = FakeDense()
                self.dense.documents = ["existing-child"]
                self.sparse = FakeSparse()
                self.sparse.documents = ["existing-child"]
                self.parent_store = {"existing": Document(page_content="existing-parent")}

            def add_parents(self, documents):
                for document in documents:
                    self.parent_store[document.metadata["parent_id"]] = document

            def save(self, path):
                raise OSError("simulated persistence failure")

        retriever = FakeRetriever()
        old_dense = retriever.dense
        old_sparse = retriever.sparse
        old_parent_store = retriever.parent_store
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "index"
            index_path.mkdir()
            marker = index_path / "previous-index"
            marker.write_text("still available", encoding="utf-8")
            with patch.object(runtime, "_hybrid_retriever", retriever), patch.object(
                runtime, "_index_store_path", index_path
            ):
                with self.assertRaisesRegex(OSError, "simulated persistence failure"):
                    runtime.rebuild_index(
                        [
                            Document(
                                page_content="new parent",
                                metadata={"parent_id": "new"},
                            )
                        ],
                        [
                            Document(
                                page_content="new child",
                                metadata={"parent_id": "new"},
                            )
                        ],
                    )
                self.assertIs(retriever.dense, old_dense)
                self.assertIs(retriever.sparse, old_sparse)
                self.assertIs(retriever.parent_store, old_parent_store)
                self.assertEqual(old_dense.documents, ["existing-child"])
                self.assertTrue(marker.exists())
                self.assertEqual(marker.read_text(encoding="utf-8"), "still available")
