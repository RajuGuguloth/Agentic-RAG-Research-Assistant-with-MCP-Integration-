"""
Evaluation Schemas and Pydantic v2 Models.
Decoupled data structures for offline evaluation, G-Eval, retriever diagnostics, and regression gating.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

# CI/CD Working-Directory Path Safety
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))


class GoldenSample(BaseModel):
    """A reference benchmark sample for evaluation."""

    model_config = ConfigDict(extra="allow")

    id: str
    query: str
    golden_context: List[str] = Field(default_factory=list)
    ground_truth: Optional[str] = None
    expected_claims: List[str] = Field(default_factory=list)
    expect_retrieval: bool = True
    category: str = "in_domain"


class GEvalDimensionScore(BaseModel):
    """Detailed score and Chain-of-Thought reasoning for a single G-Eval dimension."""

    model_config = ConfigDict(extra="allow")

    dimension: str
    score: float = Field(
        ...,
        description="Normalized score in [0.0, 1.0] or numeric score aligned with metric scale.",
    )
    raw_score: float = Field(
        ...,
        description="Raw integer/floating score on the 0-10 rubric scale.",
    )
    reasoning: str = Field(
        default="",
        description="Step-by-step Chain-of-Thought evaluation reasoning.",
    )
    logprobs: Optional[Dict[int, float]] = Field(
        default=None,
        description="Probability distribution across rubric scores (0-10) when available.",
    )
    is_logprob_weighted: bool = Field(
        default=False,
        description="True if score was computed via token logprob expectation.",
    )


class SampleEvalResult(BaseModel):
    """Evaluation result for an individual sample."""

    model_config = ConfigDict(extra="allow")

    sample_id: str
    query: str
    generated_answer: Optional[str] = None
    status: str = Field(
        default="success",
        description="'success' | 'error' | 'skipped'",
    )
    error_message: Optional[str] = None
    latency_seconds: Optional[float] = None
    token_count_estimate: Optional[int] = None
    geval_scores: Dict[str, GEvalDimensionScore] = Field(default_factory=dict)
    contextual_precision: Optional[float] = None
    contextual_recall: Optional[float] = None
    retrieved_contexts: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MetricSummary(BaseModel):
    """Descriptive statistics for a numeric evaluation metric."""

    model_config = ConfigDict(extra="allow")

    mean: float = 0.0
    std_dev: float = 0.0
    min: float = 0.0
    max: float = 0.0
    p95: float = 0.0
    count: int = 0

    @classmethod
    def from_values(cls, values: List[float]) -> MetricSummary:
        """Compute statistical summary from a list of floats."""
        valid = [float(v) for v in values if v is not None and not math.isnan(v)]
        if not valid:
            return cls(mean=0.0, std_dev=0.0, min=0.0, max=0.0, p95=0.0, count=0)

        n = len(valid)
        mean_val = sum(valid) / n
        variance = sum((x - mean_val) ** 2 for x in valid) / n if n > 1 else 0.0
        std_val = math.sqrt(variance)

        sorted_vals = sorted(valid)
        min_val = sorted_vals[0]
        max_val = sorted_vals[-1]

        # 95th percentile index
        p95_idx = int(math.ceil(0.95 * n)) - 1
        p95_val = sorted_vals[max(0, min(p95_idx, n - 1))]

        return cls(
            mean=round(mean_val, 4),
            std_dev=round(std_val, 4),
            min=round(min_val, 4),
            max=round(max_val, 4),
            p95=round(p95_val, 4),
            count=n,
        )


class EvaluationReport(BaseModel):
    """Aggregated evaluation report across all evaluation dimensions."""

    model_config = ConfigDict(extra="allow")

    run_id: str = "eval-run"
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    backend: str = "unknown"
    total_samples: int = 0
    successful_samples: int = 0
    failed_samples: int = 0
    metrics: Dict[str, MetricSummary] = Field(default_factory=dict)
    safety_violations: int = 0
    sample_results: List[SampleEvalResult] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
