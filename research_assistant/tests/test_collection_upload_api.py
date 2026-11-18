import json
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


class TestCollectionUploadAPI(unittest.TestCase):
    def setUp(self):
        from api import rag_router

        self.rag_router = rag_router
        self.api = FastAPI()
        self.api.include_router(rag_router.router)
        self.client = TestClient(self.api)

    def _environment(self, allowed=("collection-a", "collection-b")):
        return patch.dict(
            os.environ,
            {
                "RAG_API_KEY": "test-key",
                "RAG_COLLECTION_PERMISSIONS": json.dumps({"test-key": list(allowed)}),
            },
            clear=True,
        )

    def _upload(self, collection="collection-a", content=b"document text", **data):
        form_data = {"collection": collection, **data}
        return self.client.post(
            "/v1/upload",
            headers={"X-API-Key": "test-key"},
            files=[("files", ("notes.txt", content, "text/plain"))],
            data=form_data,
        )

    def test_successful_upload_targets_selected_collection(self):
        with self._environment(), patch.object(
            self.rag_router,
            "_index_collection_paths",
            return_value={
                "collection_id": "collection-a",
                "files": ["notes.txt"],
                "parents_indexed": 1,
                "children_indexed": 1,
            },
        ) as index_collection, patch.object(
            self.rag_router, "rebuild_index"
        ) as rebuild_index:
            response = self._upload()

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["collection_id"], "collection-a")
        index_collection.assert_called_once()
        rebuild_index.assert_not_called()

    def test_missing_collection_is_rejected(self):
        with self._environment():
            response = self.client.post(
                "/v1/upload",
                headers={"X-API-Key": "test-key"},
                files=[("files", ("notes.txt", b"text", "text/plain"))],
            )
        self.assertEqual(response.status_code, 422)

    def test_replace_false_is_rejected_without_replacing_collection(self):
        with self._environment(), patch.object(
            self.rag_router, "_index_collection_paths"
        ) as index_collection:
            response = self._upload(replace="false")

        self.assertEqual(response.status_code, 400)
        self.assertIn("replace=false", response.json()["detail"])
        index_collection.assert_not_called()

    def test_invalid_collection_is_rejected_before_file_processing(self):
        with self._environment(), patch.object(
            self.rag_router, "_index_collection_paths"
        ) as index_collection:
            response = self._upload(collection="../outside")
        self.assertEqual(response.status_code, 422)
        index_collection.assert_not_called()

    def test_unauthorized_collection_is_rejected(self):
        with self._environment(allowed=("collection-a",)):
            response = self._upload(collection="collection-b")
        self.assertEqual(response.status_code, 403)

    def test_collection_isolation(self):
        collections = {}

        def build(collection, paths, **kwargs):
            collections[collection] = paths[0].read_text(encoding="utf-8")
            return {"collection_id": collection, "files": ["notes.txt"]}

        with self._environment(), patch.object(
            self.rag_router, "_index_collection_paths", side_effect=build
        ):
            self.assertEqual(
                self._upload(collection="collection-a", content=b"alpha").status_code,
                200,
            )
            self.assertEqual(
                self._upload(collection="collection-b", content=b"beta").status_code,
                200,
            )

        self.assertEqual(collections, {"collection-a": "alpha", "collection-b": "beta"})

    def test_replacing_a_does_not_alter_b(self):
        collections = {"collection-b": "beta"}

        def replace(collection, paths, **kwargs):
            collections[collection] = paths[0].read_text(encoding="utf-8")
            return {"collection_id": collection}

        with self._environment(), patch.object(
            self.rag_router, "_index_collection_paths", side_effect=replace
        ):
            self.assertEqual(
                self._upload(collection="collection-a", content=b"old").status_code,
                200,
            )
            self.assertEqual(
                self._upload(collection="collection-b", content=b"beta").status_code,
                200,
            )
            self.assertEqual(
                self._upload(collection="collection-a", content=b"new").status_code,
                200,
            )

        self.assertEqual(collections, {"collection-a": "new", "collection-b": "beta"})

    def test_failed_upload_preserves_existing_collection(self):
        collections = {"collection-a": "old"}

        def failed_replace(collection, paths, **kwargs):
            raise RuntimeError("candidate build failed")

        with self._environment(), patch.object(
            self.rag_router,
            "_index_collection_paths",
            side_effect=failed_replace,
        ):
            response = self._upload(collection="collection-a", content=b"new")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(collections["collection-a"], "old")

    def test_existing_file_safety_validation_is_preserved(self):
        with self._environment(), patch.object(
            self.rag_router, "_index_collection_paths"
        ) as index_collection:
            unsupported = self.client.post(
                "/v1/upload",
                headers={"X-API-Key": "test-key"},
                files=[("files", ("notes.rtf", b"text", "text/rtf"))],
                data={"collection": "collection-a"},
            )
            invalid_pdf = self.client.post(
                "/v1/upload",
                headers={"X-API-Key": "test-key"},
                files=[("files", ("paper.pdf", b"not-a-pdf", "application/pdf"))],
                data={"collection": "collection-a"},
            )

        self.assertEqual(unsupported.status_code, 400)
        self.assertEqual(invalid_pdf.status_code, 400)
        index_collection.assert_not_called()

    def test_upload_persists_collection_and_fresh_manager_searches_it(self):
        from retrieval import runtime

        class Dense:
            def __init__(self):
                self.documents = []
                self.is_ready = False

            def build_index(self, documents):
                self.documents = list(documents)
                self.is_ready = bool(self.documents)

            def save_index(self, path):
                Path(path, "dense.json").write_text("ready", encoding="utf-8")

            def load_index(self, path):
                ready = Path(path, "dense.json").exists()
                self.is_ready = ready
                return ready

        class Sparse:
            def __init__(self):
                self.documents = []
                self.is_ready = False

            def build_index(self, documents):
                self.documents = list(documents)
                self.is_ready = bool(self.documents)

            def save_index(self, path):
                Path(path, "sparse.json").write_text("ready", encoding="utf-8")

            def load_index(self, path):
                ready = Path(path, "sparse.json").exists()
                self.is_ready = ready
                return ready

        class Retriever:
            def __init__(self, dense, sparse):
                self.dense = dense
                self.sparse = sparse
                self.parent_store = {}

            def add_parents(self, documents):
                self.parent_store = {
                    document.metadata["parent_id"]: document for document in documents
                }

            def save(self, path):
                Path(path).mkdir(parents=True, exist_ok=True)
                self.dense.save_index(path)
                self.sparse.save_index(path)
                (Path(path) / "parents.json").write_text(
                    next(iter(self.parent_store.values())).page_content,
                    encoding="utf-8",
                )

            def load(self, path):
                if not self.dense.load_index(path) or not self.sparse.load_index(path):
                    return False
                parent_path = Path(path) / "parents.json"
                if not parent_path.exists():
                    return False
                self.parent_store = {
                    "p1": Document(
                        page_content=parent_path.read_text(encoding="utf-8"),
                        metadata={"parent_id": "p1", "source": "notes.txt"},
                    )
                }
                return True

            def search(self, query, k=None):
                return list(self.parent_store.values())[:k]

        def load_documents(paths, *, with_vision=False):
            return [Document(page_content="alpha evidence", metadata={"source": "notes.txt"})]

        def chunk_documents(documents):
            parent = Document(
                page_content=documents[0].page_content,
                metadata={"parent_id": "p1", "source": "notes.txt"},
            )
            child = Document(
                page_content=documents[0].page_content,
                metadata={"parent_id": "p1", "source": "notes.txt"},
            )
            return [parent], [child]

        with tempfile.TemporaryDirectory() as temp_dir, self._environment(), patch.object(
            runtime, "DenseRetriever", Dense
        ), patch.object(
            runtime, "SparseRetriever", Sparse
        ), patch.object(
            runtime, "MultimodalRetriever", Retriever
        ), patch(
            "etl.document_loader.load_documents_from_files", load_documents
        ), patch(
            "etl.chunker.chunk_documents_parent_child", chunk_documents
        ):
            first_manager = runtime.CollectionIndexManager(temp_dir)
            with patch.object(self.rag_router, "collection_index_manager", first_manager):
                upload = self._upload(content=b"alpha")
            self.assertEqual(upload.status_code, 200, upload.text)

            fresh_manager = runtime.CollectionIndexManager(temp_dir)
            with patch.object(self.rag_router, "collection_index_manager", fresh_manager):
                search = self.client.post(
                    "/v1/search",
                    headers={"X-API-Key": "test-key"},
                    json={
                        "collection": "collection-a",
                        "query": "alpha",
                        "top_k": 1,
                    },
                )

        self.assertEqual(search.status_code, 200, search.text)
        self.assertEqual(search.json()["chunks"][0]["text"], "alpha evidence")


if __name__ == "__main__":
    unittest.main()
