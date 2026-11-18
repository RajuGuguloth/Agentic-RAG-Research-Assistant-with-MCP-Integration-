"""
Unit tests for the Evaluation Suite:
- Semaphore concurrency and full-jitter backoff retry with daily quota fail-fast.
- Single-sample fault isolation (one error does not abort batch).
- G-Eval JSON parsing, logprob tokenization normalization (' 10', '10'), and expectation calculation.
- Ablation runner read-only parent-child granularity across all 4 configurations.
- Regression gate noise buffering and pass/fail exit codes.
- CI/CD working-directory path safety.
"""

import asyncio
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

# CI/CD Working-Directory Path Safety
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from langchain_core.documents import Document

try:
    from evaluation.ablation_runner import (
        DenseOnlyRetrieverAdapter,
        FullHybridRerankerAdapter,
        HybridNoRerankerAdapter,
        SparseOnlyRetrieverAdapter,
        run_ablation_benchmark,
    )
    from evaluation.eval_generator_isolated import (
        IsolatedGeneratorEvaluator,
        is_daily_quota_exhausted,
    )
    from evaluation.eval_retriever import RetrieverEvaluator
    from evaluation.geval import GEvalJudge
    from evaluation.regression_gate import evaluate_regression, run_gate
    from evaluation.schemas import (
        EvaluationReport,
        GEvalDimensionScore,
        GoldenSample,
        MetricSummary,
        SampleEvalResult,
    )
except ImportError:
    from research_assistant.evaluation.ablation_runner import (
        DenseOnlyRetrieverAdapter,
        FullHybridRerankerAdapter,
        HybridNoRerankerAdapter,
        SparseOnlyRetrieverAdapter,
        run_ablation_benchmark,
    )
    from research_assistant.evaluation.eval_generator_isolated import (
        IsolatedGeneratorEvaluator,
        is_daily_quota_exhausted,
    )
    from research_assistant.evaluation.eval_retriever import RetrieverEvaluator
    from research_assistant.evaluation.geval import GEvalJudge
    from research_assistant.evaluation.regression_gate import evaluate_regression, run_gate
    from research_assistant.evaluation.schemas import (
        EvaluationReport,
        GEvalDimensionScore,
        GoldenSample,
        MetricSummary,
        SampleEvalResult,
    )


class TestGEvalJudge(unittest.IsolatedAsyncioTestCase):
    def test_parse_json_score_standard(self):
        judge = GEvalJudge(gateway=MagicMock())
        text = '{"reasoning": "The answer is strictly grounded.", "score": 9}'
        score, reasoning = judge._parse_json_score(text)
        self.assertEqual(score, 9.0)
        self.assertEqual(reasoning, "The answer is strictly grounded.")

    def test_parse_json_score_with_markdown_fences(self):
        judge = GEvalJudge(gateway=MagicMock())
        text = "```json\n{\n  \"reasoning\": \"Minor omissions.\",\n  \"score\": 6\n}\n```"
        score, reasoning = judge._parse_json_score(text)
        self.assertEqual(score, 6.0)
        self.assertEqual(reasoning, "Minor omissions.")

    def test_parse_json_score_regex_fallback(self):
        judge = GEvalJudge(gateway=MagicMock())
        text = "Evaluation Summary: The response is good.\nFinal Score: 8\nEnd."
        score, reasoning = judge._parse_json_score(text)
        self.assertEqual(score, 8.0)

    async def test_logprob_tokenization_normalization_and_expectation(self):
        """
        Verify logprob extraction normalizes tokens (' 10', '10', ' 8', '8')
        and correctly computes expectation E[s] = sum(s * P(s)).
        """
        mock_gateway = MagicMock()
        mock_gateway.provider = "openai"
        mock_gateway.api_key = "test-key"
        mock_gateway.base_url = "https://api.openai.com/v1"
        mock_gateway.model = "gpt-4o-mini"

        judge = GEvalJudge(gateway=mock_gateway)

        # Candidate tokens returned at position 0 include multi-digit '10' with space ' 10' and ' 8'
        # Logprobs: ln(0.7) for ' 8', ln(0.3) for ' 10'
        # Expected score: 8 * 0.7 + 10 * 0.3 = 5.6 + 3.0 = 8.6
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {"content": "10"},
                    "logprobs": {
                        "content": [
                            {
                                "token": " 10",
                                "top_logprobs": [
                                    {"token": " 8", "logprob": math.log(0.7)},
                                    {"token": " 10", "logprob": math.log(0.3)},
                                    {"token": " irrelevant_word", "logprob": math.log(0.1)},
                                ],
                            }
                        ]
                    },
                }
            ]
        }

        with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response)):
            result = await judge._try_logprob_evaluation(
                query="What is multi-head attention?",
                reasoning="The answer explains all 8 heads with high accuracy.",
                dimension="correctness",
            )
            self.assertIsNotNone(result)
            expected_score, probs = result
            self.assertAlmostEqual(expected_score, 8.6, places=2)
            self.assertAlmostEqual(probs[8], 0.7, places=2)
            self.assertAlmostEqual(probs[10], 0.3, places=2)

    def test_composite_score_computation(self):
        scores = {
            "correctness": GEvalDimensionScore(dimension="correctness", score=1.0, raw_score=10.0),
            "completeness": GEvalDimensionScore(dimension="completeness", score=0.8, raw_score=8.0),
            "tone": GEvalDimensionScore(dimension="tone", score=0.9, raw_score=9.0),
        }
        # Default weights: correctness=0.5, completeness=0.3, tone=0.2
        # Composite = 1.0*0.5 + 0.8*0.3 + 0.9*0.2 = 0.5 + 0.24 + 0.18 = 0.92
        composite = GEvalJudge.compute_composite(scores)
        self.assertAlmostEqual(composite, 0.92, places=2)


class TestGeneratorIsolatedEvaluator(unittest.IsolatedAsyncioTestCase):
    async def test_concurrency_control_semaphore(self):
        max_active = 0
        current_active = 0
        lock = asyncio.Lock()

        async def fake_complete(*args, **kwargs):
            nonlocal max_active, current_active
            async with lock:
                current_active += 1
                if current_active > max_active:
                    max_active = current_active
            await asyncio.sleep(0.05)
            async with lock:
                current_active -= 1
            return '{"reasoning": "ok", "score": 9}'

        mock_gateway = MagicMock()
        mock_gateway.complete = AsyncMock(side_effect=fake_complete)
        mock_gateway.provider = "mock"

        evaluator = IsolatedGeneratorEvaluator(
            gateway=mock_gateway,
            concurrency=3,
        )

        samples = [
            GoldenSample(
                id=f"q{i}",
                query=f"query {i}",
                golden_context=["ctx 1", "ctx 2"],
                expected_claims=["claim 1"],
            )
            for i in range(8)
        ]

        report = await evaluator.run(samples)
        self.assertEqual(report.total_samples, 8)
        self.assertEqual(report.successful_samples, 8)
        self.assertLessEqual(max_active, 3)

    async def test_single_sample_fault_isolation(self):
        async def fake_complete(system, user, **kwargs):
            if "query 2" in user:
                raise TimeoutError("LLM Gateway Request Timed Out")
            return '{"reasoning": "great", "score": 10}'

        mock_gateway = MagicMock()
        mock_gateway.complete = AsyncMock(side_effect=fake_complete)
        mock_gateway.provider = "mock"

        evaluator = IsolatedGeneratorEvaluator(
            gateway=mock_gateway,
            concurrency=2,
            max_retries=1,
        )

        samples = [
            GoldenSample(id="q1", query="query 1", golden_context=["ctx1"]),
            GoldenSample(id="q2", query="query 2", golden_context=["ctx2"]),
            GoldenSample(id="q3", query="query 3", golden_context=["ctx3"]),
        ]

        report = await evaluator.run(samples)
        self.assertEqual(report.total_samples, 3)
        self.assertEqual(report.successful_samples, 2)
        self.assertEqual(report.failed_samples, 1)

        res1 = report.sample_results[0]
        res2 = report.sample_results[1]
        res3 = report.sample_results[2]

        self.assertEqual(res1.status, "success")
        self.assertEqual(res2.status, "error")
        self.assertIn("Timed Out", res2.error_message)
        self.assertEqual(res3.status, "success")

    async def test_daily_quota_exhaustion_fast_exit(self):
        quota_err = RuntimeError("Quota exceeded for quota metric 'Queries' per day")
        self.assertTrue(is_daily_quota_exhausted(quota_err))

        mock_gateway = MagicMock()
        mock_gateway.complete = AsyncMock(side_effect=quota_err)

        evaluator = IsolatedGeneratorEvaluator(
            gateway=mock_gateway,
            max_retries=3,
        )

        with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            with self.assertRaises(RuntimeError):
                await evaluator._generate_with_retry("system", "user")
            mock_sleep.assert_not_called()


class TestRetrieverAndAblationSuite(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Synthetic parent store and child documents
        self.parent_p1 = Document(
            page_content="Full parent document 1 content on transformers.",
            metadata={"parent_id": "p1", "source": "paper.pdf", "page": 1},
        )
        self.parent_p2 = Document(
            page_content="Full parent document 2 content on attention mechanisms.",
            metadata={"parent_id": "p2", "source": "paper.pdf", "page": 2},
        )
        self.parents = {"p1": self.parent_p1, "p2": self.parent_p2}

        self.children = [
            Document(page_content="child 1 from parent 1", metadata={"parent_id": "p1"}),
            Document(page_content="child 2 from parent 2", metadata={"parent_id": "p2"}),
        ]

        self.mock_dense = MagicMock()
        self.mock_dense.is_ready = True
        self.mock_dense.search.return_value = [(0, 0.95), (1, 0.80)]
        self.mock_dense.get_document.side_effect = lambda idx: self.children[idx]

        self.mock_sparse = MagicMock()
        self.mock_sparse.is_ready = True
        self.mock_sparse.search.return_value = [(1, 15.0), (0, 10.0)]

        self.mock_hybrid = MagicMock()
        self.mock_hybrid.dense = self.mock_dense
        self.mock_hybrid.sparse = self.mock_sparse
        self.mock_hybrid.parent_store = self.parents
        self.mock_hybrid.reranker = None
        # Default mock hybrid returns Parent documents
        self.mock_hybrid.search.return_value = [self.parent_p1, self.parent_p2]

    def test_all_ablation_variants_map_to_parent_granularity(self):
        """Verify all 4 ablation variants map child hits back to deduplicated Parent documents."""
        # 1. Dense FAISS only
        dense_adapter = DenseOnlyRetrieverAdapter(self.mock_hybrid)
        docs_dense = dense_adapter.search("test query", k=2)
        self.assertEqual(len(docs_dense), 2)
        self.assertEqual(docs_dense[0].metadata["parent_id"], "p1")
        self.assertEqual(docs_dense[0].page_content, self.parent_p1.page_content)
        self.assertIn("retrieval_score", docs_dense[0].metadata)

        # 2. Sparse BM25 only
        sparse_adapter = SparseOnlyRetrieverAdapter(self.mock_hybrid)
        docs_sparse = sparse_adapter.search("test query", k=2)
        self.assertEqual(len(docs_sparse), 2)
        self.assertEqual(docs_sparse[0].metadata["parent_id"], "p2")
        self.assertEqual(docs_sparse[0].page_content, self.parent_p2.page_content)
        self.assertIn("bm25_score", docs_sparse[0].metadata)

        # 3. Hybrid RRF no reranker
        hybrid_no_reranker = HybridNoRerankerAdapter(self.mock_hybrid)
        docs_rrf = hybrid_no_reranker.search("test query", k=2)
        self.assertEqual(len(docs_rrf), 2)
        self.assertEqual(docs_rrf[0].page_content, self.parent_p1.page_content)

        # 4. Hybrid RRF with CrossEncoder reranker
        hybrid_reranker = FullHybridRerankerAdapter(self.mock_hybrid)
        docs_full = hybrid_reranker.search("test query", k=2)
        self.assertEqual(len(docs_full), 2)
        self.assertEqual(docs_full[0].page_content, self.parent_p1.page_content)

    async def test_contextual_recall_evaluation(self):
        mock_gateway = MagicMock()
        mock_gateway.complete = AsyncMock(
            return_value="Claim [1]: YES - supported\nClaim [2]: NO - not found"
        )
        evaluator = RetrieverEvaluator(judge_gateway=mock_gateway)
        claims = ["Transformers use attention", "Transformers use recurrent LSTM"]
        score, flags, _ = await evaluator.evaluate_contextual_recall(
            expected_claims=claims,
            retrieved_contexts=["Transformers rely entirely on attention mechanisms."],
        )
        self.assertEqual(score, 0.5)
        self.assertEqual(flags, [True, False])

    async def test_contextual_precision_at_k(self):
        mock_gateway = MagicMock()
        mock_gateway.complete = AsyncMock(
            return_value="Chunk [1]: RELEVANT - answers query\nChunk [2]: IRRELEVANT - unrelated"
        )
        evaluator = RetrieverEvaluator(judge_gateway=mock_gateway)
        score, flags, _ = await evaluator.evaluate_contextual_precision_at_k(
            query="self-attention",
            ground_truth="attends to same sequence",
            retrieved_contexts=["chunk 1 text", "chunk 2 text"],
        )
        self.assertEqual(score, 1.0)
        self.assertEqual(flags, [True, False])


class TestRegressionGate(unittest.TestCase):
    def test_noise_buffer_tolerates_minor_drop(self):
        baseline = EvaluationReport(
            run_id="base",
            metrics={
                "correctness": MetricSummary(mean=0.85, std_dev=0.04, min=0.75, max=0.95, p95=0.92, count=10),
                "geval_composite": MetricSummary(mean=0.85, std_dev=0.04, min=0.75, max=0.95, p95=0.92, count=10),
                "contextual_recall": MetricSummary(mean=0.90, std_dev=0.02, min=0.85, max=0.95, p95=0.95, count=10),
            },
            safety_violations=0,
        )
        candidate = EvaluationReport(
            run_id="cand",
            metrics={
                "correctness": MetricSummary(mean=0.81, std_dev=0.03, count=10),
                "geval_composite": MetricSummary(mean=0.82, std_dev=0.03, count=10),
                "contextual_recall": MetricSummary(mean=0.88, std_dev=0.02, count=10),
            },
            safety_violations=0,
        )
        passed, _, blocking = evaluate_regression(candidate, baseline)
        self.assertTrue(passed)
        self.assertEqual(len(blocking), 0)

    def test_blocking_regression_on_exceeded_drop(self):
        baseline = EvaluationReport(
            run_id="base",
            metrics={
                "correctness": MetricSummary(mean=0.90, std_dev=0.02, count=10),
                "geval_composite": MetricSummary(mean=0.90, std_dev=0.02, count=10),
                "contextual_recall": MetricSummary(mean=0.90, std_dev=0.02, count=10),
            },
            safety_violations=0,
        )
        candidate = EvaluationReport(
            run_id="cand",
            metrics={
                "correctness": MetricSummary(mean=0.80, std_dev=0.02, count=10),
                "geval_composite": MetricSummary(mean=0.90, std_dev=0.02, count=10),
                "contextual_recall": MetricSummary(mean=0.90, std_dev=0.02, count=10),
            },
            safety_violations=0,
        )
        passed, _, blocking = evaluate_regression(candidate, baseline)
        self.assertFalse(passed)
        self.assertTrue(any("correctness" in b for b in blocking))

    def test_safety_violation_triggers_failure(self):
        baseline = EvaluationReport(
            run_id="base",
            metrics={
                "correctness": MetricSummary(mean=0.90, count=10),
                "geval_composite": MetricSummary(mean=0.90, count=10),
                "contextual_recall": MetricSummary(mean=0.90, count=10),
            },
            safety_violations=0,
        )
        candidate = EvaluationReport(
            run_id="cand",
            metrics={
                "correctness": MetricSummary(mean=0.90, count=10),
                "geval_composite": MetricSummary(mean=0.90, count=10),
                "contextual_recall": MetricSummary(mean=0.90, count=10),
            },
            safety_violations=1,
        )
        passed, _, blocking = evaluate_regression(candidate, baseline)
        self.assertFalse(passed)
        self.assertTrue(any("Safety regression" in b for b in blocking))

    def test_run_gate_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_file = Path(temp_dir) / "baseline.json"
            cand_pass_file = Path(temp_dir) / "candidate_pass.json"
            cand_fail_file = Path(temp_dir) / "candidate_fail.json"

            base_report = EvaluationReport(
                run_id="base",
                metrics={"correctness": MetricSummary(mean=0.90, count=10)},
                safety_violations=0,
            )
            cand_pass = EvaluationReport(
                run_id="cand_pass",
                metrics={"correctness": MetricSummary(mean=0.88, count=10)},
                safety_violations=0,
            )
            cand_fail = EvaluationReport(
                run_id="cand_fail",
                metrics={"correctness": MetricSummary(mean=0.70, count=10)},
                safety_violations=0,
            )

            base_file.write_text(base_report.model_dump_json(), encoding="utf-8")
            cand_pass_file.write_text(cand_pass.model_dump_json(), encoding="utf-8")
            cand_fail_file.write_text(cand_fail.model_dump_json(), encoding="utf-8")

            exit_code_pass = run_gate(cand_pass_file, base_file)
            self.assertEqual(exit_code_pass, 0)

            exit_code_fail = run_gate(cand_fail_file, base_file)
            self.assertEqual(exit_code_fail, 1)


if __name__ == "__main__":
    unittest.main()
