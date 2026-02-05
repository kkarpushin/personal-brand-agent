"""
Meta-evaluator factory -- thin wrapper around ``SingleCallEvaluator``.

The orchestrator expects ``create_meta_evaluator()`` returning an object
with an ``evaluate_draft(draft, content_type)`` method.  This module bridges
the gap between that API and ``SingleCallEvaluator.evaluate()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from src.meta_agent.single_call_evaluator import SingleCallEvaluator
from src.models import ContentType, DraftPost


@dataclass
class MetaEvaluationResult:
    """Simplified result matching the attributes the orchestrator reads."""

    score: float
    feedback: str
    suggestions: List[str]
    passes_threshold: bool
    raw_evaluation: object  # full SingleCallEvaluation


class MetaEvaluator:
    """Wraps :class:`SingleCallEvaluator` with the ``evaluate_draft`` API."""

    def __init__(self, evaluator: SingleCallEvaluator) -> None:
        self._evaluator = evaluator

    async def evaluate_draft(
        self,
        draft: DraftPost,
        content_type: ContentType,
    ) -> MetaEvaluationResult:
        """Evaluate a draft post and return a simplified result."""
        post_text = getattr(draft, "full_text", None) or str(draft)
        evaluation = await self._evaluator.evaluate(
            post_content=post_text,
            content_type=content_type,
        )
        feedback_parts = list(evaluation.weaknesses or [])
        return MetaEvaluationResult(
            score=evaluation.weighted_total,
            feedback="; ".join(feedback_parts) if feedback_parts else "No issues",
            suggestions=list(evaluation.specific_suggestions or []),
            passes_threshold=evaluation.passes_threshold,
            raw_evaluation=evaluation,
        )


async def create_meta_evaluator() -> MetaEvaluator:
    """Factory expected by the orchestrator."""
    evaluator = SingleCallEvaluator()
    return MetaEvaluator(evaluator)
