"""
Retriever Ablation Runner.
Benchmarks 4 retrieval configurations without mutating index_store/:
1. Dense FAISS only (mapped to deduplicated Parent documents)
2. BM25 Sparse only (mapped to deduplicated Parent documents)
3. Hybrid RRF without CrossEncoder reranker (mapped to deduplicated Parent documents)
4. Hybrid RRF + CrossEncoder reranker (mapped to deduplicated Parent documents)
Ensures identical Parent-level text granularity across all 4 configurations.
Outputs ablation_report.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document

# CI/CD Working-Directory Path Safety
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from config import Config
try:
    from evaluation.eval_retriever import RetrieverEvaluator
    from evaluation.schemas import EvaluationReport, GoldenSample, MetricSummary
except ImportError:
    from research_assistant.evaluation.eval_retriever import RetrieverEvaluator
    from research_assistant.evaluation.schemas import EvaluationReport, GoldenSample, MetricSummary

try:
    from llm.gateway import LLMGateway, get_llm_gateway
    from retrieval.hybrid_retriever import HybridRetriever
    from retrieval.runtime import get_hybrid_retriever
except ImportError:
    from research_assistant.llm.gateway import LLMGateway, get_llm_gateway
    from research_assistant.retrieval.hybrid_retriever import HybridRetriever
    from research_assistant.retrieval.runtime import get_hybrid_retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ablation_runner")


class DenseOnlyRetrieverAdapter:
    """
    Read-only adapter for Dense FAISS retrieval without mutating index_store.
    Maps child hits back to deduplicated Parent documents for identical granularity.
    """

    def __init__(self, hybrid: HybridRetriever):
        self.hybrid = hybrid

    def search(self, query: str, k: int = 5) -> List[Document]:
        dense_results = self.hybrid.dense.search(query, k=Config.TOP_K_DENSE)
        unique_parents: Dict[str, Document] = {}

        for doc_idx, score in dense_results:
            child_doc = self.hybrid.dense.get_document(doc_idx)
            pid = child_doc.metadata.get("parent_id")
            if pid and pid in self.hybrid.parent_store and pid not in unique_parents:
                parent = self.hybrid.parent_store[pid]
                cloned = Document(
                    page_content=parent.page_content,
                    metadata={
                        **parent.metadata,
                        "retrieval_score": score,
                        "child_dense_score": score,
                    },
                )
                unique_parents[pid] = cloned
            if len(unique_parents) >= k:
                break

        return list(unique_parents.values())[:k]


class SparseOnlyRetrieverAdapter:
    """
    Read-only adapter for Sparse BM25 retrieval without mutating index_store.
    Maps child hits back to deduplicated Parent documents for identical granularity.
    """

    def __init__(self, hybrid: HybridRetriever):
        self.hybrid = hybrid

    def search(self, query: str, k: int = 5) -> List[Document]:
        sparse_results = self.hybrid.sparse.search(query, k=Config.TOP_K_SPARSE)
        unique_parents: Dict[str, Document] = {}

        for doc_idx, score in sparse_results:
            child_doc = self.hybrid.dense.get_document(doc_idx)
            pid = child_doc.metadata.get("parent_id")
            if pid and pid in self.hybrid.parent_store and pid not in unique_parents:
                parent = self.hybrid.parent_store[pid]
                cloned = Document(
                    page_content=parent.page_content,
                    metadata={
                        **parent.metadata,
                        "retrieval_score": score,
                        "bm25_score": score,
                    },
                )
                unique_parents[pid] = cloned
            if len(unique_parents) >= k:
                break

        return list(unique_parents.values())[:k]


class HybridNoRerankerAdapter:
    """
    Read-only adapter for Hybrid RRF without CrossEncoder reranking.
    Maps child hits back to deduplicated Parent documents and sorts by child_rrf_score.
    """

    def __init__(self, hybrid: HybridRetriever):
        self.hybrid = hybrid

    def search(self, query: str, k: int = 5) -> List[Document]:
        saved_reranker = self.hybrid.reranker
        try:
            self.hybrid.reranker = None
            return self.hybrid.search(query, k=k)
        finally:
            self.hybrid.reranker = saved_reranker


class FullHybridRerankerAdapter:
    """
    Read-only adapter for Hybrid RRF with CrossEncoder reranking.
    Maps child hits back to deduplicated Parent documents and reranks with CrossEncoder.
    """

    def __init__(self, hybrid: HybridRetriever):
        self.hybrid = hybrid

    def search(self, query: str, k: int = 5) -> List[Document]:
        return self.hybrid.search(query, k=k)


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


async def run_ablation_benchmark(
    hybrid: HybridRetriever,
    samples: List[GoldenSample],
    judge_gateway: Optional[LLMGateway] = None,
    top_k: int = 5,
    offline: bool = False,
) -> Dict[str, Any]:
    """Run comparative benchmark across all 4 configurations at identical Parent text granularity."""
    evaluator = RetrieverEvaluator(judge_gateway=judge_gateway, offline=offline)

    configs = {
        "1_dense_faiss_only": DenseOnlyRetrieverAdapter(hybrid),
        "2_bm25_sparse_only": SparseOnlyRetrieverAdapter(hybrid),
        "3_hybrid_rrf_no_reranker": HybridNoRerankerAdapter(hybrid),
        "4_hybrid_rrf_with_reranker": FullHybridRerankerAdapter(hybrid),
    }

    ablation_results: Dict[str, Any] = {}

    for config_name, adapter in configs.items():
        logger.info("Evaluating configuration at Parent granularity: %s", config_name)
        report = await evaluator.evaluate_samples(
            retriever=adapter,
            samples=samples,
            top_k=top_k,
        )
        ablation_results[config_name] = {
            "contextual_precision_mean": report.metrics["contextual_precision"].mean,
            "contextual_precision_p95": report.metrics["contextual_precision"].p95,
            "contextual_recall_mean": report.metrics["contextual_recall"].mean,
            "contextual_recall_p95": report.metrics["contextual_recall"].p95,
            "abstention_accuracy": report.metrics["abstention_accuracy"].mean,
            "report": report.model_dump(),
        }

    return ablation_results


async def main():
    parser = argparse.ArgumentParser(description="Run 4-way retriever ablation benchmark.")
    parser.add_argument(
        "--golden",
        type=Path,
        default=None,
        help="Golden set JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for ablation report",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-K for retrieval")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to evaluate")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode without live LLM calls")
    args = parser.parse_args()

    golden_path = resolve_golden_path(args.golden)
    data = json.loads(golden_path.read_text(encoding="utf-8"))
    samples = [GoldenSample(**item) for item in data]
    if args.limit:
        samples = samples[: args.limit]

    output_path = args.output or (Path(__file__).resolve().parent / "ablation_report.json")

    retriever = get_hybrid_retriever()
    judge_gateway = None if args.offline else get_llm_gateway()
    results = await run_ablation_benchmark(
        retriever,
        samples,
        judge_gateway=judge_gateway,
        top_k=args.top_k,
        offline=args.offline,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Ablation benchmark saved to %s", output_path)


if __name__ == "__main__":
    asyncio.run(main())
