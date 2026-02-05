"""
Single-call evaluation system for the LinkedIn Super Agent.

Replaces multi-turn Critic Agent dialogue with one structured LLM call.
Reduces latency, cost, and debugging complexity while maintaining evaluation
quality through a detailed rubric.

Provides:
    - SingleCallEvaluator: Text quality evaluation via single LLM call
    - VisualQualityEvaluator: Visual asset quality and coherence evaluation

Architecture reference: architecture.md lines 22937-23466
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.tools.claude_client import ClaudeClient
from src.meta_agent.models import (
    SingleCallEvaluation,
    EvaluationCriterion,
    evaluation_rubric,
    VisualEvaluation,
)
from src.models import ContentType, VisualAsset, HumanizedPost
from src.config import ThresholdConfig, THRESHOLD_CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Factory helper: provides a default ClaudeClient when none is injected.
# This avoids requiring callers to instantiate the client themselves.
# ---------------------------------------------------------------------------

def _get_default_claude() -> ClaudeClient:
    """Create a default ClaudeClient instance.

    Uses the ``ANTHROPIC_API_KEY`` environment variable for authentication.

    Returns:
        A new ``ClaudeClient`` instance.
    """
    return ClaudeClient()


# ===========================================================================
# SINGLE-CALL TEXT EVALUATOR
# ===========================================================================


class SingleCallEvaluator:
    """
    Evaluates post quality in a single LLM call with structured output.

    Replaces the multi-turn Critic dialogue with a deterministic rubric-based
    evaluation. The LLM receives the full rubric, post content, and content
    type, then returns a structured ``SingleCallEvaluation`` in one shot.

    Args:
        claude_client: Injected ``ClaudeClient`` for LLM calls.
            Defaults to a new ``ClaudeClient()`` via ``_get_default_claude()``.
        threshold_config: Centralized threshold configuration.
            Defaults to the global ``THRESHOLD_CONFIG`` singleton.

    Usage::

        evaluator = SingleCallEvaluator()
        result = await evaluator.evaluate(
            post_content="My LinkedIn post...",
            content_type=ContentType.ENTERPRISE_CASE,
        )
        if result.passes_threshold:
            print("Post passes QC!")
        else:
            print(f"Revisions needed: {result.recommended_revisions}")
    """

    EVALUATION_PROMPT: str = """
    Evaluate this LinkedIn post against the rubric below.
    Be harsh but constructive. The goal is genuine improvement.

    === POST ===
    {post_content}
    === END POST ===

    Context:
    - Content Type: {content_type}
    - Target: LinkedIn professionals interested in AI
    - Goal: Maximize engagement + deliver value

    === RUBRIC ===
    {rubric_text}
    === END RUBRIC ===

    EVALUATION INSTRUCTIONS:

    For EACH criterion:
    1. Quote the specific part of the post you're evaluating (max 50 chars)
    2. Give a score 1-10 based on the rubric
    3. Explain your score in 1-2 sentences

    Then provide:
    - 2-3 specific STRENGTHS (what's working well)
    - 2-3 specific WEAKNESSES (what needs improvement)
    - 3-5 ACTIONABLE SUGGESTIONS (how to fix the weaknesses)

    Also identify:
    - Recurring PATTERNS across criteria (e.g., "consistently vague")
    - KNOWLEDGE GAPS the Writer should research

    IMPORTANT - Be SPECIFIC:
    BAD: "hook is weak"
    GOOD: "The hook starts with 'Recently...' which is generic.
       Instead, try leading with the specific result: '40 hours -> 2 minutes.'"

    Return as structured JSON with this exact schema:
    {{
        "scores": {{"criterion_name": score_int, ...}},
        "criterion_feedback": {{
            "criterion_name": {{
                "quote": "relevant excerpt (max 50 chars)",
                "score": score_int,
                "explanation": "1-2 sentence explanation"
            }},
            ...
        }},
        "strengths": ["strength 1", "strength 2"],
        "weaknesses": ["weakness 1", "weakness 2"],
        "specific_suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"],
        "patterns_detected": ["pattern 1"],
        "knowledge_gaps": ["gap 1"]
    }}
    """

    def __init__(
        self,
        claude_client: Optional[ClaudeClient] = None,
        threshold_config: Optional[ThresholdConfig] = None,
    ) -> None:
        self.claude = claude_client or _get_default_claude()
        self.rubric: Dict[str, EvaluationCriterion] = evaluation_rubric
        self.threshold_config = threshold_config or THRESHOLD_CONFIG

    async def evaluate(
        self,
        post_content: str,
        content_type: ContentType,
    ) -> SingleCallEvaluation:
        """
        Evaluate a post in a single LLM call and return a complete evaluation.

        Steps:
        1. Format the rubric into human-readable text
        2. Call LLM with structured output prompt
        3. Calculate weighted total from scores
        4. Compare against content-type-specific threshold
        5. Generate prioritized revision recommendations if needed

        Args:
            post_content: Full text of the LinkedIn post to evaluate.
            content_type: The ``ContentType`` of the post (determines threshold
                multiplier).

        Returns:
            A fully populated ``SingleCallEvaluation`` with scores, feedback,
            pass/fail decision, and optional revision recommendations.
        """
        logger.info(
            "[EVALUATOR] Starting single-call evaluation for content_type=%s",
            content_type.value,
        )

        rubric_text = self._format_rubric()

        prompt = self.EVALUATION_PROMPT.format(
            post_content=post_content,
            content_type=content_type.value,
            rubric_text=rubric_text,
        )

        # Call LLM for structured evaluation
        response_data: Dict[str, Any] = await self.claude.generate_structured(
            prompt=prompt,
        )

        # Build the evaluation dataclass from the LLM response
        scores: Dict[str, int] = response_data.get("scores", {})
        criterion_feedback: Dict[str, dict] = response_data.get(
            "criterion_feedback", {}
        )
        strengths: List[str] = response_data.get("strengths", [])
        weaknesses: List[str] = response_data.get("weaknesses", [])
        specific_suggestions: List[str] = response_data.get(
            "specific_suggestions", []
        )
        patterns_detected: List[str] = response_data.get("patterns_detected", [])
        knowledge_gaps: List[str] = response_data.get("knowledge_gaps", [])

        # Calculate weighted total
        weighted_total = self._calculate_weighted_total(scores)

        # Use centralized threshold with content-type multiplier
        pass_threshold = self.threshold_config.get_pass_threshold(content_type)
        passes_threshold = weighted_total >= pass_threshold

        logger.info(
            "[EVALUATOR] Evaluation complete: weighted_total=%.2f, "
            "threshold=%.2f, passes=%s",
            weighted_total,
            pass_threshold,
            passes_threshold,
        )

        # Build revision recommendations if the post does not pass
        recommended_revisions: Optional[List[str]] = None

        evaluation = SingleCallEvaluation(
            scores=scores,
            weighted_total=weighted_total,
            criterion_feedback=criterion_feedback,
            strengths=strengths,
            weaknesses=weaknesses,
            specific_suggestions=specific_suggestions,
            passes_threshold=passes_threshold,
            recommended_revisions=None,
            patterns_detected=patterns_detected,
            knowledge_gaps=knowledge_gaps,
        )

        if not passes_threshold:
            evaluation.recommended_revisions = self._prioritize_revisions(evaluation)
            logger.info(
                "[EVALUATOR] Post below threshold. Recommended revisions: %s",
                evaluation.recommended_revisions,
            )

        return evaluation

    def _calculate_weighted_total(self, scores: Dict[str, int]) -> float:
        """
        Calculate the weighted average score across all criteria.

        Each criterion's score is multiplied by its weight from the rubric.
        Unknown criteria (not in the rubric) are logged and skipped.

        Args:
            scores: Mapping of criterion name to integer score (1-10).

        Returns:
            Weighted total score, rounded to 2 decimal places.
        """
        total = 0.0
        for criterion_name, score in scores.items():
            criterion = self.rubric.get(criterion_name)
            if criterion is None:
                logger.warning(
                    "[EVALUATOR] Unknown criterion '%s' in scores, skipping",
                    criterion_name,
                )
                continue
            total += score * criterion.weight
        return round(total, 2)

    def _prioritize_revisions(
        self, evaluation: SingleCallEvaluation
    ) -> List[str]:
        """
        Prioritize revision recommendations by impact.

        Impact is calculated as ``weight * (10 - score)`` for each criterion
        scoring below 8.  The top 3 highest-impact items are returned with
        matching suggestions from the evaluation.

        Args:
            evaluation: The completed evaluation with scores and suggestions.

        Returns:
            Up to 3 actionable revision strings, sorted by impact.
        """
        improvements: List[tuple] = []

        for criterion_name, score in evaluation.scores.items():
            if score < 8:
                criterion = self.rubric.get(criterion_name)
                if criterion is None:
                    continue
                impact = criterion.weight * (10 - score)
                improvements.append((criterion_name, impact, score))

        # Sort by impact descending (highest improvement potential first)
        improvements.sort(key=lambda x: x[1], reverse=True)

        # Return top 3 with matched suggestions
        revisions: List[str] = []
        for criterion_name, impact, score in improvements[:3]:
            # Try to find a matching suggestion from the LLM's output
            suggestion = next(
                (
                    s
                    for s in evaluation.specific_suggestions
                    if criterion_name.lower().replace("_", " ") in s.lower()
                    or criterion_name.lower() in s.lower()
                ),
                f"Improve {criterion_name} (current score: {score}/10)",
            )
            revisions.append(suggestion)

        return revisions

    def _format_rubric(self) -> str:
        """
        Format the evaluation rubric into human-readable text for the LLM prompt.

        Each criterion is presented with its name, weight, guiding question,
        and scoring guide (from 10 down to 2).

        Returns:
            Multi-line string representation of the full rubric.
        """
        lines: List[str] = []
        for name, criterion in self.rubric.items():
            lines.append(f"\n### {criterion.name} (weight: {criterion.weight})")
            lines.append(f"Question: {criterion.evaluation_prompt}")
            lines.append("Scoring guide:")
            for score, desc in sorted(criterion.rubric.items(), reverse=True):
                lines.append(f"  {score}: {desc}")
        return "\n".join(lines)


# ===========================================================================
# VISUAL QUALITY EVALUATOR
# ===========================================================================


class VisualQualityEvaluator:
    """
    Evaluates visual asset quality and content-visual coherence.

    Checks:
    1. Format appropriateness for content type
    2. Text-visual coherence (via LLM when possible)
    3. Technical quality (dimensions, resolution)
    4. Brand consistency and mobile optimization
    5. Author photo usage (engagement boost)

    Args:
        claude_client: Injected ``ClaudeClient`` for coherence evaluation.
            Defaults to a new ``ClaudeClient()`` via ``_get_default_claude()``.

    Usage::

        evaluator = VisualQualityEvaluator()
        result = await evaluator.evaluate(
            visual=visual_asset,
            post_content="My post text...",
            content_type=ContentType.AUTOMATION_CASE,
        )
        if result.score >= 6.0:
            print("Visual quality acceptable")
    """

    # Expected visual format by content type
    RECOMMENDED_FORMATS: Dict[ContentType, List[str]] = {
        ContentType.ENTERPRISE_CASE: [
            "metrics_card",
            "case_study_visual",
            "logo_showcase",
        ],
        ContentType.PRIMARY_SOURCE: [
            "data_chart",
            "quote_card",
            "source_screenshot",
        ],
        ContentType.AUTOMATION_CASE: [
            "workflow_diagram",
            "before_after",
            "screenshot",
        ],
        ContentType.COMMUNITY_CONTENT: [
            "photo_with_overlay",
            "meme_format",
            "carousel",
        ],
        ContentType.TOOL_RELEASE: [
            "product_screenshot",
            "feature_comparison",
            "demo_gif",
        ],
    }

    def __init__(
        self,
        claude_client: Optional[ClaudeClient] = None,
    ) -> None:
        self.claude = claude_client or _get_default_claude()

    async def evaluate(
        self,
        visual: VisualAsset,
        post_content: str,
        content_type: ContentType,
    ) -> VisualEvaluation:
        """
        Evaluate visual quality against multiple criteria.

        Criteria evaluated:
        1. Format appropriateness for content type
        2. Text-visual coherence (LLM-based)
        3. Technical quality (resolution, dimensions)
        4. Brand consistency
        5. Author photo usage

        Score is computed from a base of 7.0 with adjustments:
        - +1.0 / -1.0 for format appropriateness
        - +1.0 / -1.5 for coherence (>= 8.0 / < 6.0)
        - +0.5 / -0.5 for brand consistency
        - +0.5 for author photo usage
        - Clamped to [1.0, 10.0]

        Args:
            visual: The ``VisualAsset`` to evaluate.
            post_content: The post text to check coherence against.
            content_type: The ``ContentType`` of the post.

        Returns:
            A ``VisualEvaluation`` with score, flags, issues, and strengths.
        """
        logger.info(
            "[VISUAL_EVAL] Starting evaluation for content_type=%s",
            content_type.value,
        )

        issues: List[str] = []
        strengths: List[str] = []

        # ---- 1. Format appropriateness ------------------------------------
        recommended = self.RECOMMENDED_FORMATS.get(content_type, [])
        visual_style = getattr(visual, "visual_style", None) or visual.metadata.get(
            "visual_style", ""
        )
        format_appropriate = visual_style in recommended

        if not format_appropriate:
            issues.append(
                f"Visual style '{visual_style}' not ideal for "
                f"{content_type.value}. Consider: {', '.join(recommended)}"
            )
        else:
            strengths.append(
                f"Visual style '{visual_style}' appropriate for "
                f"{content_type.value}"
            )

        # ---- 2. Content-visual coherence (LLM-based) ---------------------
        coherence_score = await self._evaluate_coherence(visual, post_content)

        if coherence_score < 7.0:
            issues.append(
                f"Visual doesn't strongly match post content "
                f"(coherence: {coherence_score}/10)"
            )

        # ---- 3. Technical quality -----------------------------------------
        technical_issues = self._check_technical_quality(visual)
        issues.extend(technical_issues)

        # ---- 4. Brand consistency -----------------------------------------
        brand_check = getattr(
            visual, "brand_consistency_check",
            visual.metadata.get("brand_consistency_check", True),
        )
        mobile_optimized = getattr(
            visual, "mobile_optimized",
            visual.metadata.get("mobile_optimized", True),
        )
        brand_ok = brand_check and mobile_optimized

        if not brand_ok:
            if not brand_check:
                issues.append("Visual doesn't match brand guidelines")
            if not mobile_optimized:
                issues.append("Visual not optimized for mobile viewing")
        else:
            strengths.append("Brand consistent and mobile optimized")

        # ---- 5. Author photo usage ----------------------------------------
        photo_used = getattr(
            visual, "photo_used",
            visual.metadata.get("photo_used", False),
        )
        if photo_used:
            strengths.append(
                "Includes author photo (typically +15-25% engagement)"
            )

        # ---- Calculate final score ----------------------------------------
        base_score = 7.0
        score = base_score

        # Format appropriateness adjustment
        if format_appropriate:
            score += 1.0
        else:
            score -= 1.0

        # Coherence adjustment
        if coherence_score >= 8.0:
            score += 1.0
        elif coherence_score < 6.0:
            score -= 1.5

        # Brand consistency adjustment
        if brand_ok:
            score += 0.5
        else:
            score -= 0.5

        # Author photo bonus
        if photo_used:
            score += 0.5

        # Clamp to 1-10
        score = max(1.0, min(10.0, score))

        logger.info(
            "[VISUAL_EVAL] Evaluation complete: score=%.1f, "
            "format_ok=%s, coherence=%.1f, brand_ok=%s",
            score,
            format_appropriate,
            coherence_score,
            brand_ok,
        )

        return VisualEvaluation(
            score=score,
            format_appropriate=format_appropriate,
            content_match_score=coherence_score,
            technical_quality=8.0 if not technical_issues else 6.0,
            brand_consistency=brand_ok,
            issues=issues,
            strengths=strengths,
        )

    async def _evaluate_coherence(
        self,
        visual: VisualAsset,
        post_content: str,
    ) -> float:
        """
        Use LLM to evaluate how well the visual matches the post content.

        Falls back to the ``visual_content_match_score`` attribute if
        available (e.g., pre-computed by Visual Creator), or returns a
        neutral 7.0 if LLM evaluation fails.

        Args:
            visual: The ``VisualAsset`` to evaluate.
            post_content: The post text for coherence comparison.

        Returns:
            Coherence score as a float (1.0 - 10.0).
        """
        # Use pre-computed score if available
        precomputed = getattr(visual, "visual_content_match_score", None)
        if precomputed is not None:
            return float(precomputed)

        # Extract visual description attributes with safe fallbacks
        visual_format = getattr(visual, "format", None) or visual.asset_type
        visual_style = getattr(visual, "visual_style", None) or visual.metadata.get(
            "visual_style", "unknown"
        )
        alt_text = getattr(visual, "alt_text", None) or visual.metadata.get(
            "alt_text", ""
        )
        prompt_used = visual.prompt_used or ""

        prompt = f"""
        Rate how well this visual matches the post content on a scale of 1-10.

        POST CONTENT:
        {post_content[:500]}

        VISUAL DESCRIPTION:
        - Format: {visual_format}
        - Style: {visual_style}
        - Alt text: {alt_text}
        - Prompt used: {prompt_used[:200]}

        Consider:
        1. Does the visual reinforce the post's main message?
        2. Is the visual style appropriate for the tone?
        3. Would the visual make sense without the text?

        Return ONLY a single number 1-10, nothing else.
        """

        try:
            response_text = await self.claude.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=16,
            )
            # Parse the number from the response
            score = float(response_text.strip().split()[0])
            return max(1.0, min(10.0, score))
        except (ValueError, IndexError) as exc:
            logger.warning(
                "[VISUAL_EVAL] Failed to parse coherence score from LLM "
                "response, defaulting to 7.0: %s",
                exc,
            )
            return 7.0
        except Exception as exc:
            logger.error(
                "[VISUAL_EVAL] LLM coherence evaluation failed, "
                "defaulting to 7.0: %s",
                exc,
            )
            return 7.0

    def _check_technical_quality(self, visual: VisualAsset) -> List[str]:
        """
        Check technical aspects of the visual asset.

        Currently verifies:
        - Minimum width (800px for desktop quality)
        - Carousel minimum width (1080px per LinkedIn recommendations)

        Uses ``visual.width`` / ``visual.height`` from the ``VisualAsset``
        dataclass, or falls back to a ``dimensions`` string attribute
        (e.g., ``"1080x1080"``).

        Args:
            visual: The ``VisualAsset`` to check.

        Returns:
            List of technical issue strings (empty if all checks pass).
        """
        issues: List[str] = []

        # Resolve dimensions from the VisualAsset
        width = visual.width
        height = visual.height

        # Fallback: try a "dimensions" attribute or metadata entry
        if not width:
            dimensions_str = getattr(
                visual, "dimensions", visual.metadata.get("dimensions", "")
            )
            if dimensions_str and "x" in str(dimensions_str):
                try:
                    parts = str(dimensions_str).split("x")
                    width = int(parts[0])
                    height = int(parts[1])
                except (ValueError, IndexError):
                    pass

        if width:
            if width < 800:
                issues.append(
                    f"Visual width {width}px may appear low quality on desktop"
                )
            if visual.asset_type == "carousel" and width < 1080:
                issues.append(
                    "Carousel images should be at least 1080px wide"
                )

        return issues


# ===========================================================================
# PUBLIC API
# ===========================================================================

__all__ = [
    "SingleCallEvaluator",
    "VisualQualityEvaluator",
]
