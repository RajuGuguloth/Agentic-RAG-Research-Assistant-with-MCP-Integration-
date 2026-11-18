"""
Retriever Evaluation Suite.
Measures:
1. Claim-level Contextual Recall: Ratio of expected ground-truth claims verifiable from retrieved chunks.
2. Rank-aware Contextual Precision@K (Average Precision): Evaluates chunk relevance at each rank k.
3. Off-topic Abstention Accuracy: Percentage of out-of-domain queries correctly filtered/abstained.
Decoupled from brittle chunk UUIDs via semantic LLM-as-a-judge.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# CI/CD Working-Directory Path Safety
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from config import Config
try:
    from evaluation.schemas import (
        EvaluationReport,
        GoldenSample,
        MetricSummary,
        SampleEvalResult,
    )
except ImportError:
    from research_assistant.evaluation.schemas import (
        EvaluationReport,
        GoldenSample,
        MetricSummary,
        SampleEvalResult,
    )

try:
    from llm.gateway import LLMGateway, get_llm_gateway
    from retrieval.runtime import get_hybrid_retriever
except ImportError:
    from research_assistant.llm.gateway import LLMGateway, get_llm_gateway
    from research_assistant.retrieval.runtime import get_hybrid_retriever

logger = logging.getLogger("eval_retriever")


def resolve_golden_path(provided_path: Optional[Path] = None) -> Path:
    """Locate golden_set_ragas.json robustly across different working directories."""
    if provided_path and provided_path.exists():
        return provided_path
    candidates = [
        Path(__file__).resolve().parent / "golden_set_ragas.json",
        Path.cwd() / "evaluation" / "golden_set_ragas.json",
        Path.cwd() / "research_assistant" / "evaluation" / "golden_set_ragas.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


class RetrieverEvaluator:
    """Evaluates retrieval quality using claim-level recall, rank-aware AP, and abstention."""

    def __init__(
        self,
        judge_gateway: Optional[LLMGateway] = None,
        offline: bool = False,
    ):
        if offline:
            self.judge_gateway = None
        elif judge_gateway is not None:
            self.judge_gateway = judge_gateway
        else:
            self.judge_gateway = get_llm_gateway()

    async def _complete_with_retry(self, system: str, user: str) -> str:
        for attempt in range(4):
            try:
                return await self.judge_gateway.complete(system=system, user=user, temperature=0.0)
            except Exception as exc:
                if attempt == 3:
                    raise exc
                match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(exc), re.IGNORECASE)
                delay = float(match.group(1)) + 1.0 if match else 32.0
                await asyncio.sleep(delay)
        return ""

    async def evaluate_contextual_recall(
        self,
        expected_claims: List[str],
        retrieved_contexts: List[str],
    ) -> Tuple[float, List[bool], str]:
        """
        Check if each expected claim is entailed by the retrieved contexts.
        Returns: (recall_score, list_of_claim_support_flags, reasoning_summary).
        """
        if not expected_claims:
            return 1.0, [], "No expected claims provided."

        if not retrieved_contexts:
            return 0.0, [False] * len(expected_claims), "No contexts retrieved."

        context_text = "\n\n---\n\n".join(retrieved_contexts)

        # Heuristic claim entailment if judge_gateway is None (offline / quota exhausted)
        if self.judge_gateway is None:
            supported_flags = []
            context_lower = context_text.lower()
            for claim in expected_claims:
                words = [w.lower() for w in re.findall(r"\w+", claim) if len(w) > 3]
                matched = sum(1 for w in words if w in context_lower) / max(1, len(words))
                supported_flags.append(matched >= 0.5)
            score = sum(supported_flags) / len(expected_claims)
            return round(score, 4), supported_flags, "Offline heuristic claim entailment."

        claims_formatted = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(expected_claims))

        system = (
            "You are an expert factual entailment judge. "
            "Determine whether each of the numbered claims can be completely verified or inferred "
            "from the reference context. For each claim, respond strictly with 'Claim [i]: YES' or 'Claim [i]: NO', "
            "followed by a brief reason. Do not use outside knowledge."
        )
        user = (
            f"Reference Context:\n{context_text}\n\n"
            f"Claims To Verify:\n{claims_formatted}\n\n"
            "Format:\nClaim [1]: YES/NO - reason\n..."
        )

        response = await self._complete_with_retry(system=system, user=user)

        supported_flags = []
        for i in range(len(expected_claims)):
            pattern = rf"Claim\s*\[?{i+1}\]?\s*:\s*(YES|NO)"
            match = re.search(pattern, response, re.IGNORECASE)
            if match and match.group(1).upper() == "YES":
                supported_flags.append(True)
            else:
                supported_flags.append(False)

        score = sum(supported_flags) / len(expected_claims)
        return round(score, 4), supported_flags, response

    async def evaluate_contextual_precision_at_k(
        self,
        query: str,
        ground_truth: Optional[str],
        retrieved_contexts: List[str],
    ) -> Tuple[float, List[bool], str]:
        """
        Compute Average Precision (AP) for rank-ordered retrieved chunks.
        For each rank k (1..K), checks if chunk k is relevant to answering the query.
        AP = sum(Precision@k * rel_k) / sum(rel_k).
        """
        if not retrieved_contexts:
            return 0.0, [], "No contexts retrieved."

        # Heuristic relevance if judge_gateway is None (offline / quota exhausted)
        if self.judge_gateway is None:
            relevance_flags = []
            query_words = set(w.lower() for w in re.findall(r"\w+", query) if len(w) > 3)
            for chunk in retrieved_contexts:
                chunk_words = set(w.lower() for w in re.findall(r"\w+", chunk))
                overlap = len(query_words.intersection(chunk_words)) / max(1, len(query_words))
                relevance_flags.append(overlap >= 0.25)

            running_rel = 0
            precisions = []
            for rank, is_rel in enumerate(relevance_flags, start=1):
                if is_rel:
                    running_rel += 1
                    precisions.append(running_rel / rank)

            ap_score = (sum(precisions) / len(precisions)) if precisions else 0.0
            return round(ap_score, 4), relevance_flags, "Offline heuristic chunk relevance."

        chunks_formatted = "\n\n".join(
            f"--- Chunk {idx+1} ---\n{text.strip()}"
            for idx, text in enumerate(retrieved_contexts)
        )

        system = (
            "You are an expert retrieval relevance judge. "
            "For each retrieved chunk, determine whether it contains useful, relevant information "
            "that helps answer the user's query (and matches the ground truth reference if provided). "
            "Reply strictly in the format: 'Chunk [k]: RELEVANT' or 'Chunk [k]: IRRELEVANT'."
        )
        user = f"User Query: {query}\n\n"
        if ground_truth:
            user += f"Ground Truth Reference: {ground_truth}\n\n"
        user += f"Retrieved Chunks:\n{chunks_formatted}\n"

        response = await self._complete_with_retry(system=system, user=user)

        relevance_flags = []
        for idx in range(len(retrieved_contexts)):
            pattern = rf"Chunk\s*\[?{idx+1}\]?\s*:\s*(RELEVANT|IRRELEVANT)"
            match = re.search(pattern, response, re.IGNORECASE)
            if match and match.group(1).upper() == "RELEVANT":
                relevance_flags.append(True)
            else:
                relevance_flags.append(False)

        # Calculate Average Precision (AP)
        running_rel = 0
        precisions = []
        for rank, is_rel in enumerate(relevance_flags, start=1):
            if is_rel:
                running_rel += 1
                precisions.append(running_rel / rank)

        if not precisions:
            ap_score = 0.0
        else:
            ap_score = sum(precisions) / len(precisions)

        return round(ap_score, 4), relevance_flags, response

    async def evaluate_abstention(
        self,
        off_topic_samples: List[GoldenSample],
        retriever,
        top_k: int = 5,
        threshold: float = 0.35,
    ) -> Tuple[float, int, int]:
        """
        Verify that out-of-domain queries (expect_retrieval=False) are abstained or filtered.
        Returns: (abstention_accuracy, abstained_count, total_count).
        """
        if not off_topic_samples:
            return 1.0, 0, 0

        abstained_count = 0
        for sample in off_topic_samples:
            docs = retriever.search(sample.query, k=top_k)
            if not docs:
                abstained_count += 1
            else:
                max_score = max(
                    (d.metadata.get("retrieval_score") or 0.0) for d in docs
                )
                if max_score < threshold:
                    abstained_count += 1

        accuracy = abstained_count / len(off_topic_samples)
        return round(accuracy, 4), abstained_count, len(off_topic_samples)

    async def evaluate_samples(
        self,
        retriever,
        samples: List[GoldenSample],
        top_k: int = 5,
    ) -> EvaluationReport:
        """Evaluate retrieval across all samples."""
        in_domain = [s for s in samples if s.expect_retrieval and s.category != "off_topic"]
        off_topic = [s for s in samples if not s.expect_retrieval or s.category == "off_topic"]

        sample_results: List[SampleEvalResult] = []
        precisions: List[float] = []
        recalls: List[float] = []

        for sample in in_domain:
            docs = retriever.search(sample.query, k=top_k)
            retrieved_texts = [d.page_content for d in docs if getattr(d, "page_content", None)]

            recall_score, _, _ = await self.evaluate_contextual_recall(
                expected_claims=sample.expected_claims,
                retrieved_contexts=retrieved_texts,
            )
            precision_score, _, _ = await self.evaluate_contextual_precision_at_k(
                query=sample.query,
                ground_truth=sample.ground_truth,
                retrieved_contexts=retrieved_texts,
            )

            precisions.append(precision_score)
            recalls.append(recall_score)

            sample_results.append(
                SampleEvalResult(
                    sample_id=sample.id,
                    query=sample.query,
                    status="success",
                    contextual_precision=precision_score,
                    contextual_recall=recall_score,
                    retrieved_contexts=retrieved_texts,
                )
            )

        abstention_acc, abstained_cnt, off_cnt = await self.evaluate_abstention(
            off_topic_samples=off_topic,
            retriever=retriever,
            top_k=top_k,
        )

        metrics = {
            "contextual_precision": MetricSummary.from_values(precisions),
            "contextual_recall": MetricSummary.from_values(recalls),
            "abstention_accuracy": MetricSummary.from_values([abstention_acc] if off_cnt > 0 else [1.0]),
        }

        return EvaluationReport(
            run_id="retriever-eval",
            backend=Config.LLM_BACKEND,
            total_samples=len(samples),
            successful_samples=len(sample_results) + off_cnt,
            failed_samples=0,
            metrics=metrics,
            sample_results=sample_results,
            metadata={
                "top_k": top_k,
                "in_domain_count": len(in_domain),
                "off_topic_count": off_cnt,
                "abstained_count": abstained_cnt,
            },
        )


async def main():
    parser = argparse.ArgumentParser(description="Evaluate retriever on golden set.")
    parser.add_argument("--golden", type=Path, default=None, help="Golden set JSON file")
    parser.add_argument("--output", type=Path, default=None, help="Output path for retrieval report")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K for retrieval")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to evaluate")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode without live LLM calls")
    args = parser.parse_args()

    golden_path = resolve_golden_path(args.golden)
    data = json.loads(golden_path.read_text(encoding="utf-8"))
    samples = [GoldenSample(**item) for item in data]
    if args.limit:
        samples = samples[: args.limit]

    output_path = args.output or (Path(__file__).resolve().parent / "retriever_eval_report.json")
    retriever = get_hybrid_retriever()
    evaluator = RetrieverEvaluator(offline=args.offline)
    report = await evaluator.evaluate_samples(retriever=retriever, samples=samples, top_k=args.top_k)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    logger.info("Retriever evaluation saved to %s", output_path)


if __name__ == "__main__":
    asyncio.run(main())
