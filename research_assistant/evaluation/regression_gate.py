"""
Automated CI/CD Regression Gate.
Compares a candidate EvaluationReport against a baseline EvaluationReport.
Applies standard-deviation noise buffers: drop > max(0.05, 1.5 * baseline_std_dev)
to prevent false positive alerts caused by stochastic LLM-judge variance.
Exits with code 1 on blocking regressions (>5% drop in correctness, geval_composite,
contextual_recall, or any safety violation) and 0 on pass.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# CI/CD Working-Directory Path Safety
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

try:
    from evaluation.schemas import EvaluationReport, MetricSummary
except ImportError:
    from research_assistant.evaluation.schemas import EvaluationReport, MetricSummary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("regression_gate")

BLOCKING_METRICS = ["correctness", "geval_composite", "contextual_recall"]


def load_report(source: Any) -> EvaluationReport:
    """Load EvaluationReport from path, dict, or existing instance."""
    if isinstance(source, EvaluationReport):
        return source
    if isinstance(source, dict):
        return EvaluationReport(**source)
    if isinstance(source, (str, Path)):
        path = Path(source)
        data = json.loads(path.read_text(encoding="utf-8"))
        return EvaluationReport(**data)
    raise TypeError(f"Unsupported report source type: {type(source).__name__}")


def evaluate_regression(
    candidate: EvaluationReport,
    baseline: EvaluationReport,
    monitored_metrics: Optional[List[str]] = None,
) -> Tuple[bool, List[Dict[str, Any]], List[str]]:
    """
    Compare candidate against baseline with standard deviation noise buffers.
    Returns: (passed: bool, comparison_rows: List[dict], blocking_reasons: List[str]).
    """
    metrics_to_check = monitored_metrics or BLOCKING_METRICS
    comparison_rows = []
    blocking_reasons = []

    # 1. Check Safety Violations
    cand_safety = candidate.safety_violations
    base_safety = baseline.safety_violations
    safety_passed = cand_safety == 0 and cand_safety <= base_safety
    if not safety_passed:
        blocking_reasons.append(
            f"Safety regression: Candidate has {cand_safety} violations (baseline had {base_safety})."
        )

    comparison_rows.append(
        {
            "metric": "safety_violations",
            "baseline_mean": base_safety,
            "baseline_std": 0.0,
            "candidate_mean": cand_safety,
            "diff": cand_safety - base_safety,
            "threshold": 0.0,
            "status": "PASS" if safety_passed else "FAIL",
            "blocking": True,
        }
    )

    # 2. Check Numeric Metrics with Std-Dev Noise Buffers
    for m in metrics_to_check:
        base_summary = baseline.metrics.get(m)
        cand_summary = candidate.metrics.get(m)

        if not base_summary or not cand_summary:
            continue

        base_mean = base_summary.mean
        base_std = base_summary.std_dev
        cand_mean = cand_summary.mean

        drop = base_mean - cand_mean
        threshold = max(0.05, 1.5 * base_std)

        regressed = drop > threshold
        if regressed and m in BLOCKING_METRICS:
            blocking_reasons.append(
                f"Blocking drop on '{m}': baseline={base_mean:.4f} (std={base_std:.4f}), "
                f"candidate={cand_mean:.4f}, drop={drop:.4f} > threshold={threshold:.4f}"
            )

        comparison_rows.append(
            {
                "metric": m,
                "baseline_mean": round(base_mean, 4),
                "baseline_std": round(base_std, 4),
                "candidate_mean": round(cand_mean, 4),
                "diff": round(cand_mean - base_mean, 4),
                "threshold": round(threshold, 4),
                "status": "FAIL" if regressed else "PASS",
                "blocking": m in BLOCKING_METRICS,
            }
        )

    passed = len(blocking_reasons) == 0
    return passed, comparison_rows, blocking_reasons


def format_table(rows: List[Dict[str, Any]]) -> str:
    """Format comparison rows into a clean Markdown table."""
    headers = [
        "Metric",
        "Baseline Mean",
        "Baseline Std",
        "Candidate Mean",
        "Diff",
        "Allowed Drop",
        "Status",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        diff_str = f"{r['diff']:+.4f}" if isinstance(r['diff'], (int, float)) else str(r['diff'])
        line = (
            f"| {r['metric']} | {r['baseline_mean']} | {r['baseline_std']} | "
            f"{r['candidate_mean']} | {diff_str} | {r['threshold']} | {r['status']} |"
        )
        lines.append(line)
    return "\n".join(lines)


def run_gate(
    candidate_path: Path,
    baseline_path: Path,
) -> int:
    """Run regression gate CLI and return exit code (0 for pass, 1 for fail)."""
    candidate = load_report(candidate_path)
    baseline = load_report(baseline_path)

    passed, rows, blocking_reasons = evaluate_regression(candidate, baseline)

    print("\n" + "=" * 60)
    print("           EVALUATION REGRESSION GATE REPORT")
    print("=" * 60)
    print(format_table(rows))
    print("=" * 60)

    if passed:
        print("\n✅ REGRESSION GATE PASSED: All metrics within acceptable noise bounds.\n")
        return 0
    else:
        print("\n❌ REGRESSION GATE FAILED: Blocking regressions detected:")
        for r in blocking_reasons:
            print(f"  - {r}")
        print("\nAborting CI pipeline with exit code 1.\n")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Evaluate candidate vs baseline regression gate.")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate EvaluationReport JSON")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline EvaluationReport JSON")
    args = parser.parse_args()

    sys.exit(run_gate(args.candidate, args.baseline))


if __name__ == "__main__":
    main()
