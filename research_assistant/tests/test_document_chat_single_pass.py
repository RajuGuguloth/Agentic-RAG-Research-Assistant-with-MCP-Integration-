import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchestration import document_chat


class FakeChain:
    def __init__(self):
        self.calls = []

    async def ainvoke(self, values):
        self.calls.append(values)
        return type("Response", (), {"content": "Answer based on the cited paper."})()


class FakePrompt:
    def __init__(self, chain):
        self.chain = chain

    def __or__(self, llm):
        return self.chain


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def search(self, query, *, k):
        self.calls.append((query, k))
        return self.docs


class TestDocumentChatSinglePass(unittest.IsolatedAsyncioTestCase):
    async def test_retrieves_once_and_generates_exactly_once(self):
        docs = [
            Document(
                page_content="The paper describes its proposed method.",
                metadata={"source": "paper.pdf", "page": 2},
            )
        ]
        retriever = FakeRetriever(docs)
        chain = FakeChain()
        llm = object()

        with (
            patch.object(document_chat, "HAS_USER_MEMORY", False),
            patch.object(document_chat, "HAS_QUERY_UTILS", False),
            patch.object(document_chat, "HAS_PROMPT_OPTIMIZER", False),
            patch.object(document_chat.Config, "TOP_K_FINAL", 5),
            patch.object(
                document_chat.ChatPromptTemplate,
                "from_messages",
                return_value=FakePrompt(chain),
            ),
            patch.object(document_chat, "log_trace") as trace,
        ):
            result = await document_chat.answer_document_question(
                "What method does the paper propose?",
                [],
                llm,
                retriever,
            )

        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(retriever.calls[0][1], 5)
        self.assertEqual(len(chain.calls), 1)
        self.assertIn(
            "The paper describes its proposed method.",
            chain.calls[0]["context"],
        )
        self.assertEqual(result["answer"], "Answer based on the cited paper.")
        self.assertIn("paper.pdf", result["sources"])
        trace.assert_called_once()
        trace_metadata = trace.call_args.kwargs["verification_results"]
        self.assertEqual(trace_metadata["mode"], "single_pass")
        self.assertEqual(trace_metadata["llm_generation_calls"], 1)

    async def test_empty_retrieval_returns_without_generation(self):
        retriever = FakeRetriever([])
        chain = FakeChain()

        with (
            patch.object(document_chat, "HAS_USER_MEMORY", False),
            patch.object(document_chat, "HAS_QUERY_UTILS", False),
            patch.object(document_chat, "HAS_PROMPT_OPTIMIZER", False),
            patch.object(
                document_chat.ChatPromptTemplate,
                "from_messages",
                return_value=FakePrompt(chain),
            ),
            patch.object(document_chat, "log_trace"),
        ):
            result = await document_chat.answer_document_question(
                "Question with no matches",
                [],
                object(),
                retriever,
            )

        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(chain.calls, [])
        self.assertEqual(result["sources"], "")
        self.assertIn("don't have enough confidence", result["answer"])


if __name__ == "__main__":
    unittest.main()
