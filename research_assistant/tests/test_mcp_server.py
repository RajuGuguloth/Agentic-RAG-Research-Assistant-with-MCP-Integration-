import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp_server import build_index_from_paths, search_papers


class TestMcpIndexingInputs(unittest.TestCase):
    def test_empty_paths_raise_an_error(self):
        with self.assertRaisesRegex(ValueError, "At least one document path"):
            asyncio.run(build_index_from_paths("collection-a", []))

    def test_missing_paths_raise_an_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "Unsupported or missing document paths"):
            asyncio.run(build_index_from_paths("collection-a", ["/missing/paper.pdf"]))

    def test_collection_is_required(self):
        with self.assertRaises(TypeError):
            asyncio.run(build_index_from_paths(file_paths=[]))

    def test_indexing_forwards_collection(self):
        with self.subTest("request payload"):
            with tempfile.NamedTemporaryFile(suffix=".txt") as document:
                document.write(b"content")
                document.flush()
                with patch(
                    "mcp_server._api_request",
                    new=AsyncMock(return_value={"collection_id": "collection-a"}),
                ) as api_request:
                    asyncio.run(
                        build_index_from_paths("collection-a", [document.name])
                    )

        api_request.assert_awaited_once()
        self.assertEqual(api_request.await_args.args[:2], ("POST", "/v1/upload"))
        self.assertEqual(
            api_request.await_args.kwargs["data"]["collection"],
            "collection-a",
        )


class TestMcpSearchContract(unittest.TestCase):
    def test_search_forwards_collection(self):
        response = {
            "query": "leave policy",
            "chunks": [{"text": "leave evidence"}],
            "count": 1,
        }
        with patch(
            "mcp_server._api_request",
            new=AsyncMock(return_value=response),
        ) as api_request:
            result = asyncio.run(
                search_papers("hr-policies", "leave policy", top_k=3)
            )

        self.assertEqual(result, response)
        api_request.assert_awaited_once_with(
            "POST",
            "/v1/search",
            json_body={
                "collection": "hr-policies",
                "query": "leave policy",
                "top_k": 3,
            },
        )


if __name__ == "__main__":
    unittest.main()
