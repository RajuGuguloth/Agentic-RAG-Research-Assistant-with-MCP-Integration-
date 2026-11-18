"""
G-Eval: Chain-of-Thought LLM-as-a-Judge Evaluation Framework.
Computes evaluation scores across 'correctness', 'completeness', and 'tone'
using Chain-of-Thought (CoT) reasoning, strict 0-10 rubrics, and token log-probability
expectation when available, with automatic fallback to structured JSON integer scoring.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# CI/CD Working-Directory Path Safety
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

import httpx

from config import Config
try:
    from evaluation.schemas import GEvalDimensionScore
except ImportError:
    from research_assistant.evaluation.schemas import GEvalDimensionScore

try:
    from llm.gateway import LLMGateway, get_llm_gateway
except ImportError:
    from research_assistant.llm.gateway import LLMGateway, get_llm_gateway

DIMENSION_PROMPTS = {
    "correctness": {
        "definition": "Factual consistency and truthfulness of the generated answer with respect to the reference context and ground truth.",
        "steps": [
            "1. Read the provided query, reference context, and ground truth carefully.",
            "2. Identify the key factual claims made in the generated answer.",
            "3. Cross-reference each claim against the context and ground truth.",
            "4. Check for any contradictions, unverified claims, or hallucinations.",
            "5. Assign a score from 0 to 10 based on the rubric.",
        ],
        "rubric": (
            "Score 0-4 (Poor): Contains direct factual contradictions, major hallucinations, or completely incorrect information.\n"
            "Score 5-8 (Fair to Good): Mostly accurate with minor ambiguities or slight inaccuracies, but core facts are supported.\n"
            "Score 9-10 (Excellent): Completely accurate, truthful, and strictly grounded with zero hallucinations or contradictions."
        ),
    },
    "completeness": {
        "definition": "Exhaustiveness of the generated answer in addressing all components, conditions, and aspects of the user query.",
        "steps": [
            "1. Analyze all questions, sub-questions, and implicit requirements in the user query.",
            "2. Verify whether the generated answer addresses all aspects requested.",
            "3. Check if any essential context, formulas, definitions, or qualifiers were omitted.",
            "4. Assign a score from 0 to 10 based on the rubric.",
        ],
        "rubric": (
            "Score 0-4 (Poor): Misses most of the essential information or fails to answer the primary question.\n"
            "Score 5-8 (Fair to Good): Answers the main question but leaves out important secondary details or nuances.\n"
            "Score 9-10 (Excellent): Fully answers the query with all relevant details, comprehensive context, and necessary distinctions."
        ),
    },
    "tone": {
        "definition": "Clarity, objectivity, professionalism, and conciseness of the response.",
        "steps": [
            "1. Assess the language used: is it clear, objective, neutral, and academic/professional?",
            "2. Check for excessive hedging, bias, emotional language, or overly conversational fluff.",
            "3. Check for clarity and structural readability.",
            "4. Assign a score from 0 to 10 based on the rubric.",
        ],
        "rubric": (
            "Score 0-4 (Poor): Inappropriate, overly casual, evasive, aggressive, or unreadable tone.\n"
            "Score 5-8 (Fair to Good): Generally acceptable tone, slightly wordy or mildly conversational, but understandable.\n"
            "Score 9-10 (Excellent): Clear, crisp, academic, highly professional, and objective tone."
        ),
    },
}

VALID_SCORE_STRINGS = {str(i): i for i in range(11)}


class GEvalJudge:
    """Chain-of-Thought G-Eval judge supporting logprob scoring and JSON fallback."""

    def __init__(self, gateway: Optional[LLMGateway] = None):
        self.gateway = gateway or get_llm_gateway()

    def _format_context(self, context: Union[str, List[str]]) -> str:
        if isinstance(context, list):
            return "\n\n---\n\n".join(c.strip() for c in context if c.strip())
        return str(context).strip()

    def _build_cot_prompt(
        self,
        query: str,
        context: str,
        generated_answer: str,
        ground_truth: Optional[str],
        dimension: str,
    ) -> tuple[str, str]:
        dim_info = DIMENSION_PROMPTS.get(dimension, DIMENSION_PROMPTS["correctness"])
        system = (
            "You are an expert AI Evaluation Judge. Your job is to rigorously evaluate "
            f"a generated answer on the dimension of '{dimension.upper()}'.\n\n"
            f"Dimension Definition: {dim_info['definition']}\n\n"
            "Evaluation Steps (Chain of Thought):\n"
            + "\n".join(dim_info["steps"])
            + "\n\nScoring Rubric (0 to 10 integer):\n"
            + dim_info["rubric"]
            + "\n\nYou must output valid JSON with two fields: 'reasoning' (your step-by-step chain of thought) and 'score' (an integer from 0 to 10)."
        )

        user_content = (
            f"User Query:\n{query}\n\n"
            f"Reference Context:\n{context}\n\n"
        )
        if ground_truth:
            user_content += f"Ground Truth Reference:\n{ground_truth}\n\n"
        user_content += (
            f"Generated Answer To Evaluate:\n{generated_answer}\n\n"
            "Now, evaluate the generated answer according to the steps and rubric.\n"
            "Output JSON:\n"
            "{\n"
            '  "reasoning": "<step-by-step chain of thought>",\n'
            '  "score": <integer between 0 and 10>\n'
            "}"
        )
        return system, user_content

    async def _try_logprob_evaluation(
        self,
        query: str,
        reasoning: str,
        dimension: str,
    ) -> Optional[tuple[float, Dict[int, float]]]:
        """
        Extract normalized token logprobs for the score token on OpenAI-compatible endpoints.
        Elicits a single score token immediately following the Chain-of-Thought block.
        """
        provider = getattr(self.gateway, "provider", "").lower()
        if provider not in ("openai", "together", "groq", "openai_compat", "perplexity"):
            return None

        api_key = getattr(self.gateway, "api_key", None)
        base_url = getattr(self.gateway, "base_url", None) or "https://api.openai.com/v1"
        model = getattr(self.gateway, "model", "gpt-4o-mini")

        if not api_key:
            return None

        system = (
            "You are an expert AI Evaluation Judge. "
            f"Based on the evaluation reasoning for '{dimension.upper()}', "
            "output strictly a single integer score from 0 to 10."
        )
        prompt_with_score_prefix = (
            f"Evaluation Reasoning:\n{reasoning}\n\n"
            "Based on the rubric and reasoning above, the final integer score (0-10) is: "
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt_with_score_prefix},
            ],
            "temperature": 0.0,
            "max_tokens": 2,
            "logprobs": True,
            "top_logprobs": 15,
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                url = base_url.rstrip("/") + "/chat/completions"
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code != 200:
                    return None
                data = resp.json()

            choice = data["choices"][0]
            logprob_content = choice.get("logprobs", {}).get("content", [])
            if not logprob_content:
                return None

            # Check the first token position immediately after the prompt
            first_token_logprobs = logprob_content[0].get("top_logprobs", [])
            if not first_token_logprobs:
                return None

            unnormalized: Dict[int, float] = {}
            for item in first_token_logprobs:
                raw_token = item.get("token", "")
                norm_token = raw_token.strip()
                if norm_token in VALID_SCORE_STRINGS:
                    score_val = VALID_SCORE_STRINGS[norm_token]
                    lp = item.get("logprob", -99.0)
                    prob = math.exp(lp)
                    unnormalized[score_val] = unnormalized.get(score_val, 0.0) + prob

            if not unnormalized:
                return None

            total_p = sum(unnormalized.values())
            if total_p <= 0.0:
                return None

            normalized_probs = {s: p / total_p for s, p in unnormalized.items()}
            expected_score = sum(s * p for s, p in normalized_probs.items())

            return expected_score, normalized_probs

        except Exception:
            return None

    def _parse_json_score(self, text: str) -> tuple[float, str]:
        """Extract reasoning and numeric score from JSON or fallback regex."""
        cleaned = text.strip()

        # Strip leading/trailing markdown fences
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        # 1. Attempt direct json.loads on cleaned text
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and "score" in parsed:
                score = float(parsed["score"])
                reasoning = str(parsed.get("reasoning", "")).strip()
                return max(0.0, min(10.0, score)), reasoning
        except (json.JSONDecodeError, ValueError):
            pass

        # 2. Attempt to parse first complete JSON block {...}
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict) and "score" in parsed:
                    score = float(parsed["score"])
                    reasoning = str(parsed.get("reasoning", "")).strip()
                    return max(0.0, min(10.0, score)), reasoning
            except (json.JSONDecodeError, ValueError):
                pass

        # 3. Regex fallback: if truncated at max_output_tokens, extract "score" key
        score_match = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        if not score_match:
            score_match = re.search(
                r"(?:score|final score|rating)\s*[:=]\s*(\d+(?:\.\d+)?)",
                text,
                re.IGNORECASE,
            )
        if score_match:
            try:
                score = float(score_match.group(1))
                reasoning_match = re.search(r'"reasoning"\s*:\s*"((?:\\.|[^"\\])*)', text)
                reasoning = reasoning_match.group(1).strip() if reasoning_match else text
                return max(0.0, min(10.0, score)), reasoning
            except ValueError:
                pass

        return 5.0, text

    async def evaluate_dimension(
        self,
        query: str,
        context: Union[str, List[str]],
        generated_answer: str,
        ground_truth: Optional[str] = None,
        dimension: str = "correctness",
    ) -> GEvalDimensionScore:
        """Evaluate a single dimension with CoT and logprobs or JSON fallback."""
        dim_key = dimension.lower()
        if dim_key not in DIMENSION_PROMPTS:
            dim_key = "correctness"

        formatted_context = self._format_context(context)
        system, user = self._build_cot_prompt(
            query=query,
            context=formatted_context,
            generated_answer=generated_answer,
            ground_truth=ground_truth,
            dimension=dim_key,
        )

        # 1. Generate Chain-of-Thought reasoning + base integer score with rate-limit retry
        response_text = ""
        for attempt in range(4):
            try:
                response_text = await self.gateway.complete(
                    system=system,
                    user=user,
                    temperature=0.0,
                    max_tokens=1024,
                )
                break
            except Exception as exc:
                if attempt == 3:
                    raise exc
                match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(exc), re.IGNORECASE)
                delay = float(match.group(1)) + 1.0 if match else 32.0
                await asyncio.sleep(delay)

        raw_score, reasoning = self._parse_json_score(response_text)

        # 2. Try token log-probability expectation (if backend supports it)
        logprob_result = await self._try_logprob_evaluation(
            query=query,
            reasoning=reasoning,
            dimension=dim_key,
        )
        if logprob_result is not None:
            expected_score, prob_dist = logprob_result
            norm_score = max(0.0, min(1.0, expected_score / 10.0))
            return GEvalDimensionScore(
                dimension=dim_key,
                score=round(norm_score, 4),
                raw_score=round(expected_score, 2),
                reasoning=reasoning,
                logprobs=prob_dist,
                is_logprob_weighted=True,
            )

        # 3. Fallback to structured JSON integer score
        norm_score = max(0.0, min(1.0, raw_score / 10.0))
        return GEvalDimensionScore(
            dimension=dim_key,
            score=round(norm_score, 4),
            raw_score=round(raw_score, 2),
            reasoning=reasoning,
            logprobs=None,
            is_logprob_weighted=False,
        )

    async def evaluate_all(
        self,
        query: str,
        context: Union[str, List[str]],
        generated_answer: str,
        ground_truth: Optional[str] = None,
        dimensions: Optional[List[str]] = None,
    ) -> Dict[str, GEvalDimensionScore]:
        """Evaluate all requested dimensions (default: correctness, completeness, tone)."""
        dims = dimensions or ["correctness", "completeness", "tone"]
        results: Dict[str, GEvalDimensionScore] = {}
        for dim in dims:
            results[dim] = await self.evaluate_dimension(
                query=query,
                context=context,
                generated_answer=generated_answer,
                ground_truth=ground_truth,
                dimension=dim,
            )
        return results

    @staticmethod
    def compute_composite(
        scores: Dict[str, GEvalDimensionScore],
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute weighted composite score (default: correctness 0.5, completeness 0.3, tone 0.2)."""
        w = weights or {"correctness": 0.5, "completeness": 0.3, "tone": 0.2}
        total_weight = 0.0
        weighted_sum = 0.0
        for dim, weight in w.items():
            if dim in scores:
                weighted_sum += scores[dim].score * weight
                total_weight += weight

        if total_weight == 0.0:
            return 0.0
        return round(weighted_sum / total_weight, 4)
