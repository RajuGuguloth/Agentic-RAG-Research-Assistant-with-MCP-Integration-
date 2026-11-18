import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document


class FakeDense:
    def __init__(self):
        self.documents = []
        self.is_ready = False

    def build_index(self, documents):
        self.documents = list(documents)
        self.is_ready = bool(self.documents)


class PersistedDense(FakeDense):
    def save_index(self, path):
        (Path(path) / "dense.json").write_text(
            json.dumps([document.page_content for document in self.documents]),
            encoding="utf-8",
        )

    def load_index(self, path):
        index_path = Path(path) / "dense.json"
        if not index_path.exists():
            return False
        self.documents = [
            Document(page_content=text)
            for text in json.loads(index_path.read_text(encoding="utf-8"))
        ]
        self.is_ready = bool(self.documents)
        return self.is_ready


class FakeSparse:
    def __init__(self):
        self.documents = []
        self.is_ready = False

    def build_index(self, documents):
        self.documents = list(documents)
        self.is_ready = bool(self.documents)


class PersistedSparse(FakeSparse):
    def save_index(self, path):
        (Path(path) / "sparse.json").write_text("saved", encoding="utf-8")

    def load_index(self, path):
        return (Path(path) / "sparse.json").exists()


class FakeRetriever:
    def __init__(self, dense, sparse):
        self.dense = dense
        self.sparse = sparse
        self.parent_store = {}

    def add_parents(self, documents):
        for document in documents:
            self.parent_store[document.metadata["parent_id"]] = document

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        payload = {
            "parents": {
                key: {
                    "text": value.page_content,
                    "metadata": value.metadata,
                }
                for key, value in self.parent_store.items()
            }
        }
        (Path(path) / "index.json").write_text(json.dumps(payload), encoding="utf-8")

    def load(self, path):
        index_path = Path(path) / "index.json"
        if not index_path.exists():
            return False
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        self.parent_store = {
            key: Document(
                page_content=value["text"],
                metadata=value["metadata"],
            )
            for key, value in payload["parents"].items()
        }
        self.dense.documents = list(self.parent_store.values())
        self.dense.is_ready = True
        self.sparse.documents = list(self.parent_store.values())
        self.sparse.is_ready = True
        return True


class TestCollectionIndexManager(unittest.TestCase):
    def setUp(self):
        from retrieval import runtime

        self.runtime = runtime
        self.patches = [
            patch.object(runtime, "DenseRetriever", FakeDense),
            patch.object(runtime, "SparseRetriever", FakeSparse),
            patch.object(runtime, "MultimodalRetriever", FakeRetriever),
        ]
        for active_patch in self.patches:
            active_patch.start()

    def tearDown(self):
        for active_patch in reversed(self.patches):
            active_patch.stop()

    @staticmethod
    def _docs(name):
        parent = Document(
            page_content=f"{name} parent",
            metadata={"parent_id": f"{name}-parent"},
        )
        child = Document(
            page_content=f"{name} child",
            metadata={"parent_id": f"{name}-parent"},
        )
        return [parent], [child]

    def test_loads_one_collection_lazily(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.runtime.CollectionIndexManager(temp_dir)
            parent_docs, child_docs = self._docs("a")
            manager.build_or_replace("collection-a", parent_docs, child_docs)

            reloaded = self.runtime.CollectionIndexManager(temp_dir)
            retriever = reloaded.get("collection-a")

            self.assertEqual(list(retriever.parent_store), ["a-parent"])
            self.assertIs(retriever, reloaded.get("collection-a"))

    def test_keeps_two_collections_isolated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.runtime.CollectionIndexManager(temp_dir)
            manager.build_or_replace("collection-a", *self._docs("a"))
            manager.build_or_replace("collection-b", *self._docs("b"))

            self.assertEqual(
                list(manager.get("collection-a").parent_store),
                ["a-parent"],
            )
            self.assertEqual(
                list(manager.get("collection-b").parent_store),
                ["b-parent"],
            )

    def test_replacing_a_does_not_change_b(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.runtime.CollectionIndexManager(temp_dir)
            manager.build_or_replace("collection-a", *self._docs("old-a"))
            manager.build_or_replace("collection-b", *self._docs("b"))

            manager.build_or_replace("collection-a", *self._docs("new-a"))

            self.assertEqual(
                list(manager.get("collection-a").parent_store),
                ["new-a-parent"],
            )
            self.assertEqual(
                list(manager.get("collection-b").parent_store),
                ["b-parent"],
            )

    def test_missing_collection_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.runtime.CollectionIndexManager(temp_dir)

            with self.assertRaises(self.runtime.CollectionNotFoundError):
                manager.get("missing")

    def test_invalid_collection_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.runtime.CollectionIndexManager(temp_dir)

            with self.assertRaises(self.runtime.InvalidCollectionIDError):
                manager.get("../outside")

    def test_failed_replacement_preserves_old_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.runtime.CollectionIndexManager(temp_dir)
            manager.build_or_replace("collection-a", *self._docs("old-a"))

            with patch.object(
                self.runtime,
                "_persist_candidate",
                side_effect=OSError("simulated persistence failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated persistence failure"):
                    manager.build_or_replace("collection-a", *self._docs("new-a"))

            self.assertEqual(
                list(manager.get("collection-a").parent_store),
                ["old-a-parent"],
            )

    def test_failed_reload_validation_preserves_old_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.runtime.CollectionIndexManager(temp_dir)
            manager.build_or_replace("collection-a", *self._docs("old-a"))
            old_retriever = manager.get("collection-a")

            class FailedLoadRetriever(FakeRetriever):
                def load(self, path):
                    return False

            new_candidate = FakeRetriever(FakeDense(), FakeSparse())
            failed_validation = FailedLoadRetriever(FakeDense(), FakeSparse())
            with patch.object(
                manager,
                "_new_retriever",
                side_effect=[new_candidate, failed_validation],
            ):
                with self.assertRaisesRegex(
                    self.runtime.CollectionError,
                    "Persisted index validation failed",
                ):
                    manager.build_or_replace("collection-a", *self._docs("new-a"))

            self.assertIs(manager.get("collection-a"), old_retriever)
            self.assertEqual(
                list(manager.get("collection-a").parent_store),
                ["old-a-parent"],
            )

    def test_real_hybrid_persistence_round_trip(self):
        from retrieval.multimodal_retriever import MultimodalRetriever

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            self.runtime,
            "MultimodalRetriever",
            MultimodalRetriever,
        ), patch.object(
            self.runtime,
            "DenseRetriever",
            PersistedDense,
        ), patch.object(
            self.runtime,
            "SparseRetriever",
            PersistedSparse,
        ), patch(
            "retrieval.hybrid_retriever.HAS_CROSS_ENCODER",
            False,
        ):
            manager = self.runtime.CollectionIndexManager(temp_dir)
            parent_docs, child_docs = self._docs("real")
            manager.build_or_replace("collection-a", parent_docs, child_docs)

            reloaded = self.runtime.CollectionIndexManager(temp_dir)
            retriever = reloaded.get("collection-a")

            self.assertTrue(retriever.dense.is_ready)
            self.assertEqual(
                retriever.parent_store["real-parent"].page_content,
                "real parent",
            )

    def test_removing_one_collection_does_not_affect_another(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.runtime.CollectionIndexManager(temp_dir)
            manager.build_or_replace("collection-a", *self._docs("a"))
            manager.build_or_replace("collection-b", *self._docs("b"))

            manager.remove("collection-a")

            with self.assertRaises(self.runtime.CollectionNotFoundError):
                manager.get("collection-a")
            self.assertEqual(
                list(manager.get("collection-b").parent_store),
                ["b-parent"],
            )


if __name__ == "__main__":
    unittest.main()
