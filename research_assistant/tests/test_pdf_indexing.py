import inspect
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class FakeRetriever:
    class Dense:
        is_ready = False

    def __init__(self):
        self.dense = self.Dense()


with patch("retrieval.runtime.get_hybrid_retriever", return_value=FakeRetriever()):
    import app


class TestDocumentIndexingStages(unittest.TestCase):
    def setUp(self):
        self.files = [type("Upload", (), {"name": "/tmp/paper.pdf"})()]
        self.docs = [
            Document(page_content="Extracted paper text", metadata={"page": 1})
        ]
        self.parents = [
            Document(page_content="parent", metadata={"parent_id": "p1"})
        ]
        self.children = [
            Document(page_content="child", metadata={"parent_id": "p1"})
        ]

    def test_vision_is_off_by_default_for_pdf_indexing(self):
        default = inspect.signature(app.build_index_from_documents).parameters[
            "with_vision"
        ].default
        self.assertFalse(default)
        with patch.object(
            app, "load_documents_from_files", return_value=self.docs
        ) as load_pdfs, patch.object(
            app,
            "chunk_documents_parent_child",
            return_value=(self.parents, self.children),
        ), patch.object(
            app,
            "rebuild_index",
            return_value={"total_children": 1},
        ):
            result = app._run_document_indexing(
                self.files,
                app.STRATEGY_PARENT_CHILD_SEMANTIC,
                False,
                lambda *_args, **_kwargs: None,
            )

        self.assertEqual(result["total_children"], 1)
        load_pdfs.assert_called_once_with(
            ["/tmp/paper.pdf"],
            progress=unittest.mock.ANY,
            with_vision=False,
        )

    def test_empty_extraction_fails_before_index_mutation(self):
        with patch.object(app, "load_documents_from_files", return_value=[]), patch.object(
            app, "rebuild_index"
        ) as rebuild:
            with self.assertRaises(app.DocumentIndexingFailure) as raised:
                app._run_document_indexing(
                    self.files,
                    app.STRATEGY_PARENT_CHILD_SEMANTIC,
                    False,
                    lambda *_args, **_kwargs: None,
                )

        self.assertEqual(raised.exception.stage, "local document text extraction")
        self.assertIn("Scanned or image-only PDFs", str(raised.exception.cause))
        rebuild.assert_not_called()

    def test_failed_index_does_not_trigger_summary_of_previous_index(self):
        failure = app.DocumentIndexingFailure(
            "local document text extraction",
            ValueError("no text"),
        )
        with patch.object(app, "_run_document_indexing", side_effect=failure), patch.object(
            app, "launch_simple_summary"
        ) as summarize:
            result = app.build_index_and_summarize_documents(
                self.files,
                progress=lambda *_args, **_kwargs: None,
            )

        self.assertTrue(result[0].startswith("❌ Document indexing failed"))
        self.assertIn("Any previous index was preserved", result[0])
        self.assertEqual(result[1:], ("", "", "", ""))
        summarize.assert_not_called()

    def test_nested_vision_rate_limit_is_reported_without_provider_response_body(self):
        response = httpx.Response(
            429,
            request=httpx.Request("POST", "https://api.example.test/chat"),
            text="provider detail that should not be shown",
        )
        provider_error = httpx.HTTPStatusError(
            "rate limited",
            request=response.request,
            response=response,
        )
        try:
            raise provider_error
        except httpx.HTTPStatusError as cause:
            try:
                raise RuntimeError("vision analysis wrapper") from cause
            except RuntimeError as wrapped:
                failure = app.DocumentIndexingFailure(
                    "local document extraction and PDF vision analysis",
                    wrapped,
                )

        with patch.object(app.Config, "VISION_PROVIDER", "openai"):
            message = app._document_indexing_error_message(failure)

        self.assertIn("OpenAI vision provider rate-limited", message)
        self.assertIn("HTTP 429", message)
        self.assertNotIn("provider detail", message)


if __name__ == "__main__":
    unittest.main()
