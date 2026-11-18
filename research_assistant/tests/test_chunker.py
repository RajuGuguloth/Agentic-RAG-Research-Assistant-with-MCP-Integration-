import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class FakeSemanticChunker:
    def __init__(self, embeddings, breakpoint_threshold_type):
        self.embeddings = embeddings

    def split_text(self, text):
        return [part.strip() for part in text.split("|") if part.strip()]


class TestChunkingStrategies(unittest.TestCase):
    def setUp(self):
        from etl import chunker

        self.chunker = chunker
        self.document = Document(
            page_content="First idea. | Second idea.",
            metadata={"source": "paper.pdf", "page": 1},
        )

    def assert_parent_child_links_match(self, parents, children):
        parent_ids = {doc.metadata["parent_id"] for doc in parents}
        self.assertTrue(children)
        self.assertTrue(all(doc.metadata["parent_id"] in parent_ids for doc in children))
        self.assertTrue(all(doc.metadata["doc_type"] == "parent" for doc in parents))
        self.assertTrue(all(doc.metadata["doc_type"] == "child" for doc in children))

    def test_existing_strategy_keeps_semantic_children_linked_to_parents(self):
        with patch.object(self.chunker, "_get_embeddings", return_value=object()), patch.object(
            self.chunker, "SemanticChunker", FakeSemanticChunker
        ):
            parents, children = self.chunker.chunk_documents_parent_child([self.document])

        self.assertEqual(len(parents), 1)
        self.assertEqual(len(children), 2)
        self.assert_parent_child_links_match(parents, children)
        self.assertTrue(
            all(
                child.metadata["chunking_strategy"]
                == self.chunker.STRATEGY_PARENT_CHILD_SEMANTIC
                for child in children
            )
        )

    def test_recursive_strategy_uses_matching_parent_and_child_chunks(self):
        with patch.object(self.chunker.Config, "CHUNK_SIZE", 12), patch.object(
            self.chunker.Config, "CHUNK_OVERLAP", 0
        ):
            parents, children = self.chunker.chunk_documents_parent_child(
                [self.document],
                strategy=self.chunker.STRATEGY_RECURSIVE_FIXED,
            )

        self.assertGreater(len(parents), 1)
        self.assertEqual(len(parents), len(children))
        self.assert_parent_child_links_match(parents, children)
        self.assertEqual(
            [doc.page_content for doc in parents],
            [doc.page_content for doc in children],
        )

    def test_semantic_only_creates_one_linked_parent_per_semantic_chunk(self):
        with patch.object(self.chunker, "_get_embeddings", return_value=object()), patch.object(
            self.chunker, "SemanticChunker", FakeSemanticChunker
        ):
            parents, children = self.chunker.chunk_documents_parent_child(
                [self.document],
                strategy=self.chunker.STRATEGY_SEMANTIC_ONLY,
            )

        self.assertEqual(len(parents), 2)
        self.assertEqual(len(children), 2)
        self.assert_parent_child_links_match(parents, children)
        self.assertEqual(
            [doc.page_content for doc in parents],
            [doc.page_content for doc in children],
        )

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported chunking strategy"):
            self.chunker.chunk_documents_parent_child(
                [self.document],
                strategy="unknown",
            )


if __name__ == "__main__":
    unittest.main()
