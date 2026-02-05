"""
Quality Control (QC) Agent for the LinkedIn Super Agent system.

Evaluates post quality using content-type-aware scoring rubrics and makes
pass / revise / reject decisions via a single structured LLM call.

Architecture overview
---------------------
1. Load type-specific scoring rubric based on ``ContentType``.
2. Evaluate universal criteria: hook_strength, value_density, humanness,
   visual_match, controversy_safety, tone_match.
3. Evaluate type-specific criteria (e.g., metrics_credibility for
   ENTERPRISE_CASE, reproducibility for AUTOMATION_CASE).
4. Calculate weighted aggregate score using type-specific weights.
5. Make decision: PASS (>= threshold), REVISE (>= reject threshold), REJECT.
6. Generate targeted revision feedback with agent-routing information.

Error philosophy: NO FALLBACKS, FAIL FAST.  If Claude returns unparseable
JSON or scores are missing, the agent raises immediately.

References
----------
- ``architecture.md`` lines 10120-10760   (QC scoring rubrics)
- ``architecture.md`` lines 11049-11135   (QCResult / QCOutput schemas)
- ``architecture.md`` lines 22937-23466   (Single-Call Evaluator)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.models import (
    ContentType,
    DraftPost,
    HumanizedPost,
    QCOutput,
    QCResult,
    VisualAsset,
)
from src.utils import generate_id, utc_now
from src.tools.claude_client import ClaudeClient

logger = logging.getLogger("QCAgent")


# =============================================================================
# UNIVERSAL SCORING CRITERIA
# =============================================================================

UNIVERSAL_CRITERIA: Dict[str, Dict[str, Any]] = {
    "hook_strength": {
        "base_weight": 0.20,
        "rubric": {
            10: "Impossible to not click. Perfect curiosity gap.",
            8: "Very compelling, would stop most scrollers.",
            6: "Good hook, but seen similar before.",
            4: "Generic, doesn't stand out.",
            2: "Boring, would scroll past.",
        },
        "prompt_hint": "Does it create curiosity? Is it specific? Would it stop the scroll?",
    },
    "value_density": {
        "base_weight": 0.25,
        "rubric": {
            10: "Every sentence delivers value. Dense with insights.",
            8: "High value, minimal filler.",
            6: "Good value but some padding.",
            4: "More fluff than substance.",
            2: "Says nothing new.",
        },
        "prompt_hint": "How many actionable insights? Is there filler content?",
    },
    "humanness": {
        "base_weight": 0.15,
        "rubric": {
            10: "Sounds exactly like a real person. Unique voice.",
            8: "Very natural, occasional AI tells.",
            6: "Mostly human but some robotic phrases.",
            4: "Clearly AI-assisted.",
            2: "Obviously AI-generated.",
        },
        "prompt_hint": "AI phrases? Personal touches? Sentence rhythm variety?",
    },
    "visual_match": {
        "base_weight": 0.15,
        "rubric": {
            10: "Image perfectly amplifies the message.",
            8: "Strong visual that supports content.",
            6: "Relevant but not exceptional.",
            4: "Generic stock-photo feel.",
            2: "Irrelevant or distracting.",
        },
        "prompt_hint": "Does the image add to the message? Is it the right visual type?",
    },
    "controversy_safety": {
        "base_weight": 0.15,
        "rubric": {
            10: "Thought-provoking without being offensive.",
            8: "Takes a stance, but respectfully.",
            6: "Safe, unlikely to spark negative reactions.",
            4: "Could be misinterpreted.",
            2: "Likely to cause backlash.",
        },
        "prompt_hint": "Could it offend? Are claims well-supported? Tone respectful?",
    },
    "tone_match": {
        "base_weight": 0.10,
        "rubric": {
            10: "Perfect tone for content type.",
            8: "Good tone match, minor deviations.",
            6: "Acceptable but not optimal.",
            4: "Tone mismatch noticeable.",
            2: "Wrong tone entirely.",
        },
        "prompt_hint": "Does it sound like the right voice for this content type?",
    },
}


# =============================================================================
# TYPE-SPECIFIC ADDITIONAL CRITERIA AND WEIGHT OVERRIDES
# =============================================================================

TYPE_CRITERIA: Dict[ContentType, Dict[str, Any]] = {
    ContentType.ENTERPRISE_CASE: {
        "metrics_credibility": {
            "weight": 0.15,
            "desc": "Are numbers specific, believable, sourced?",
            "rubric": {
                10: "Specific, sourced, believable metrics with context.",
                8: "Good metrics, clearly attributed.",
                6: "Some numbers but vague or unsourced.",
                4: "Mostly qualitative, lacking specifics.",
                2: "No real metrics or clearly exaggerated.",
            },
        },
        "weight_overrides": {
            "hook_strength": 0.20,
            "value_density": 0.25,
            "humanness": 0.15,
            "visual_match": 0.15,
            "controversy_safety": 0.10,
        },
        "tone_expectation": "professional credibility with insight",
    },
    ContentType.PRIMARY_SOURCE: {
        "intellectual_depth": {
            "weight": 0.15,
            "desc": "Does it engage meaningfully with research?",
            "rubric": {
                10: "Substantive engagement, adds real insight.",
                8: "Good understanding, fair representation.",
                6: "Accurate but surface-level.",
                4: "Oversimplified or slightly misrepresents.",
                2: "Misunderstands or sensationalizes.",
            },
        },
        "weight_overrides": {
            "hook_strength": 0.20,
            "value_density": 0.25,
            "humanness": 0.15,
            "visual_match": 0.10,
            "controversy_safety": 0.15,
        },
        "tone_expectation": "intellectual engagement with accessibility",
    },
    ContentType.AUTOMATION_CASE: {
        "reproducibility": {
            "weight": 0.20,
            "desc": "Can reader actually reproduce this?",
            "rubric": {
                10: "Clear steps, specific tools, complete instructions.",
                8: "Good detail, reader could figure out gaps.",
                6: "General approach clear but missing specifics.",
                4: "Vague, would need significant research.",
                2: "Not reproducible, too abstract.",
            },
        },
        "weight_overrides": {
            "hook_strength": 0.15,
            "value_density": 0.30,
            "humanness": 0.15,
            "visual_match": 0.15,
            "controversy_safety": 0.05,
        },
        "tone_expectation": "practitioner authenticity with generosity",
    },
    ContentType.COMMUNITY_CONTENT: {
        "community_authenticity": {
            "weight": 0.15,
            "desc": "Does it feel connected to community?",
            "rubric": {
                10: "Feels like community insider sharing wisdom.",
                8: "Good connection, proper attribution.",
                6: "Covers community content but feels distant.",
                4: "More like reporting on than participating in.",
                2: "Disconnected, appropriating without credit.",
            },
        },
        "weight_overrides": {
            "hook_strength": 0.20,
            "value_density": 0.20,
            "humanness": 0.20,
            "visual_match": 0.10,
            "controversy_safety": 0.15,
        },
        "tone_expectation": "conversational warmth with connection",
    },
    ContentType.TOOL_RELEASE: {
        "evaluation_balance": {
            "weight": 0.15,
            "desc": "Is assessment fair and balanced?",
            "rubric": {
                10: "Balanced, honest, helps reader decide.",
                8: "Good assessment, minor bias.",
                6: "Leans promotional or dismissive.",
                4: "Clearly unbalanced, agenda apparent.",
                2: "Pure promotion or unfair criticism.",
            },
        },
        "weight_overrides": {
            "hook_strength": 0.25,
            "value_density": 0.20,
            "humanness": 0.15,
            "visual_match": 0.15,
            "controversy_safety": 0.10,
        },
        "tone_expectation": "balanced assessment with hands-on credibility",
    },
}


# =============================================================================
# IMPROVEMENT SUGGESTIONS BY TYPE
# =============================================================================

IMPROVEMENT_SUGGESTIONS: Dict[ContentType, Dict[str, List[str]]] = {
    ContentType.ENTERPRISE_CASE: {
        "hook_strength": [
            "Lead with the most impressive metric",
            "Name the company in the hook for credibility",
            "Use a lessons-learned angle",
        ],
        "value_density": [
            "Add more specific metrics with context",
            "Include timeline for implementation",
            "Extract replicable lessons",
        ],
        "humanness": [
            "Add interpretive commentary ('What struck me was...')",
            "Include a measured opinion on the approach",
        ],
        "visual_match": [
            "Use data visualization or metrics card",
            "Show architecture diagram if applicable",
        ],
        "controversy_safety": [
            "Ensure claims about the company are verifiable",
            "Add caveats about context-specific results",
        ],
        "tone_match": [
            "Keep professional credibility tone",
            "Add insight-driven commentary, not just facts",
        ],
        "metrics_credibility": [
            "Add specific numbers (X% not 'significant')",
            "Include timeframe and baseline",
            "Attribute metrics to source",
        ],
    },
    ContentType.PRIMARY_SOURCE: {
        "hook_strength": [
            "Lead with the counterintuitive finding",
            "Challenge a common assumption",
            "Create a curiosity gap about implications",
        ],
        "value_density": [
            "Explain why this matters for practitioners",
            "Add your interpretation of implications",
            "Include debate angles",
        ],
        "humanness": [
            "Add intellectual reactions ('This made me reconsider...')",
            "Include nuanced takes, not just summary",
        ],
        "visual_match": [
            "Use concept illustration or quote card",
            "Show data chart from the source",
        ],
        "controversy_safety": [
            "Fairly represent the source research",
            "Acknowledge limitations of the study",
        ],
        "tone_match": [
            "Balance intellectual depth with accessibility",
            "Engage meaningfully, don't just summarize",
        ],
        "intellectual_depth": [
            "Go beyond summary to interpretation",
            "Acknowledge nuances and limitations",
            "Connect to practical implications",
        ],
    },
    ContentType.AUTOMATION_CASE: {
        "hook_strength": [
            "Lead with time/cost savings",
            "Promise specific outcome",
            "Address a relatable pain point",
        ],
        "value_density": [
            "Add step-by-step specifics",
            "Name exact tools and versions",
            "Include gotchas and tips",
        ],
        "humanness": [
            "Add practitioner empathy ('I know this pain...')",
            "Share what you learned building it",
        ],
        "visual_match": [
            "Use workflow diagram or screenshot",
            "Show before/after comparison",
        ],
        "controversy_safety": [
            "Be honest about limitations of the approach",
            "Mention when this won't work",
        ],
        "tone_match": [
            "Sound like a helpful practitioner, not a lecturer",
            "Be generous with practical details",
        ],
        "reproducibility": [
            "Name all tools explicitly",
            "Add configuration details",
            "Include warnings about common issues",
        ],
    },
    ContentType.COMMUNITY_CONTENT: {
        "hook_strength": [
            "Highlight the most surprising insight",
            "Create FOMO about the discussion",
            "Promise curated wisdom",
        ],
        "value_density": [
            "Synthesize multiple perspectives",
            "Extract practitioner signals",
            "Add your meta-takeaway",
        ],
        "humanness": [
            "Position yourself as community participant",
            "Add genuine reactions to insights",
        ],
        "visual_match": [
            "Use quote card or platform-style visual",
            "Show carousel of insights",
        ],
        "controversy_safety": [
            "Represent diverse viewpoints fairly",
            "Don't mischaracterize community sentiment",
        ],
        "tone_match": [
            "Sound connected and warm",
            "Be a participant, not an observer",
        ],
        "community_authenticity": [
            "Add proper attributions",
            "Include diverse viewpoints",
            "Invite continued discussion",
        ],
    },
    ContentType.TOOL_RELEASE: {
        "hook_strength": [
            "Emphasize timeliness ('Just dropped')",
            "Lead with killer feature or comparison",
            "Create urgency for relevant users",
        ],
        "value_density": [
            "Be specific about features",
            "Clarify who should care",
            "Include access information",
        ],
        "humanness": [
            "Add hands-on experience notes",
            "Include honest first impressions",
        ],
        "visual_match": [
            "Use product screenshot or feature comparison",
            "Show demo GIF or interface visual",
        ],
        "controversy_safety": [
            "Be fair in comparison to competitors",
            "Label first impressions clearly",
        ],
        "tone_match": [
            "Balance excitement with honest assessment",
            "Sound credible, not promotional",
        ],
        "evaluation_balance": [
            "Add limitations or caveats",
            "Mention who shouldn't use this",
            "Compare fairly to alternatives",
        ],
    },
}


# =============================================================================
# TONE EXPECTATIONS BY CONTENT TYPE
# =============================================================================

TONE_EXPECTATIONS: Dict[ContentType, str] = {
    ContentType.ENTERPRISE_CASE: "professional credibility with insight",
    ContentType.PRIMARY_SOURCE: "intellectual engagement with accessibility",
    ContentType.AUTOMATION_CASE: "practitioner authenticity with generosity",
    ContentType.COMMUNITY_CONTENT: "conversational warmth with connection",
    ContentType.TOOL_RELEASE: "balanced assessment with hands-on credibility",
}


# =============================================================================
# DEFAULT THRESHOLDS (loaded from config/scoring_weights.json if available)
# =============================================================================


def _load_qc_thresholds() -> Dict[str, Any]:
    """Load QC thresholds from config/scoring_weights.json."""
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "scoring_weights.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("qc_thresholds", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


_QC_THRESHOLDS = _load_qc_thresholds()
DEFAULT_PASS_THRESHOLD: float = _QC_THRESHOLDS.get("default_pass", 8.0)
DEFAULT_REJECT_THRESHOLD: float = _QC_THRESHOLDS.get("default_reject", 5.5)


# =============================================================================
# QC AGENT
# =============================================================================


class QCAgent:
    """
    Quality Control Agent using single-call evaluation.

    Evaluates post content against universal and type-specific criteria in ONE
    structured LLM call, then makes a pass / revise / reject decision.

    The agent focuses on WHAT needs fixing, not HOW to fix it.  Revision
    instructions are objective (e.g., "Hook doesn't meet 210-char limit"),
    leaving creative alternatives to downstream agents.

    Args:
        claude: Async Claude API client for LLM calls.
        pass_threshold: Minimum aggregate score to pass (default 8.0).
        reject_threshold: Score below which content is rejected (default 5.5).
    """

    # ------------------------------------------------------------------
    # Single-call evaluation prompt template
    # ------------------------------------------------------------------
    _EVALUATION_PROMPT = """Evaluate this LinkedIn post against ALL criteria below.
Be harsh but constructive. The goal is genuine improvement.

=== POST TEXT ===
{post_text}
=== END POST TEXT ===

=== VISUAL INFORMATION ===
Visual type: {visual_type}
Visual description: {visual_description}
=== END VISUAL ===

=== CONTEXT ===
Content Type: {content_type}
Expected Tone: {tone_expectation}
Target Audience: LinkedIn professionals interested in AI
Goal: Maximize engagement + deliver genuine value
=== END CONTEXT ===

=== SCORING RUBRIC ===
{rubric_text}
=== END RUBRIC ===

INSTRUCTIONS:
For EACH criterion listed in the rubric:
1. Give a score from 1.0 to 10.0 (use decimals for precision)
2. Provide a concise feedback string explaining the score

Return ONLY valid JSON in this exact format (no markdown fences):
{{
  "scores": {{
    "criterion_name": {{
      "score": 8.0,
      "feedback": "Specific feedback explaining the score"
    }}
  }}
}}

Score ALL of these criteria: {criteria_names}
"""

    def __init__(
        self,
        claude: ClaudeClient,
        pass_threshold: float = DEFAULT_PASS_THRESHOLD,
        reject_threshold: float = DEFAULT_REJECT_THRESHOLD,
    ) -> None:
        self.claude = claude
        self.pass_threshold = pass_threshold
        self.reject_threshold = reject_threshold
        self._type_overrides: Dict[str, Dict[str, float]] = _QC_THRESHOLDS.get(
            "content_type_overrides", {}
        )
        self.logger = logger

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    async def run(
        self,
        humanized_post: HumanizedPost,
        visual_asset: Optional[VisualAsset],
        draft_post: DraftPost,
        content_type: ContentType,
    ) -> QCOutput:
        """
        Evaluate post quality and make pass / revise / reject decision.

        Uses single-call evaluation: ONE Claude call scores ALL criteria at
        once and returns structured JSON.

        Args:
            humanized_post: The humanized post text to evaluate.
            visual_asset: Optional visual asset accompanying the post.
            draft_post: The original draft post (for hook / CTA reference).
            content_type: The content type determining rubric weights.

        Returns:
            ``QCOutput`` with result, scoring weights, and type adjustments.

        Raises:
            json.JSONDecodeError: If Claude returns unparseable JSON.
            KeyError: If expected score keys are missing from response.
        """
        self.logger.info(
            "Starting QC evaluation for content_type=%s", content_type.value
        )

        # 1. Determine all criteria to evaluate
        all_criteria = self._get_all_criteria(content_type)
        criteria_names = list(all_criteria.keys())

        # 2. Build weights map
        weights = self._get_weights(content_type)

        # 3. Build and execute single-call evaluation prompt
        prompt = self._build_evaluation_prompt(
            humanized_post=humanized_post,
            visual_asset=visual_asset,
            draft_post=draft_post,
            content_type=content_type,
            criteria_names=criteria_names,
        )

        raw_response = await self.claude.generate(
            prompt,
            system=(
                "You are a strict LinkedIn content quality evaluator. "
                "You MUST return valid JSON only, no markdown fences, "
                "no explanatory text outside the JSON object."
            ),
            temperature=0.3,
            max_tokens=4096,
        )

        # 4. Parse scores from response
        scores = self._parse_scores(raw_response, criteria_names)

        # 5. Calculate weighted aggregate
        aggregate = self._calculate_weighted_score(scores, weights)

        # 6. Make decision (uses content-type-specific thresholds)
        decision = self._make_decision(aggregate, content_type)

        self.logger.info(
            "QC evaluation complete: aggregate=%.2f decision=%s",
            aggregate,
            decision,
        )

        # 7. Generate feedback and revision instructions when not passing
        feedback_dict: Optional[Dict[str, Any]] = None
        revision_instructions: Optional[List[str]] = None
        revision_target: Optional[str] = None

        if decision != "pass":
            feedback_dict = self._generate_feedback(scores, content_type)
            revision_instructions = self._build_revision_instructions(
                scores, content_type
            )
            if decision == "revise":
                revision_target = self._determine_revision_target(scores)

        # 8. Separate universal and type-specific scores
        universal_scores = {
            k: v for k, v in scores.items() if k in UNIVERSAL_CRITERIA
        }
        type_specific_scores = {
            k: v for k, v in scores.items() if k not in UNIVERSAL_CRITERIA
        }

        # 9. Build QCResult
        result = QCResult(
            total_score=aggregate,
            scores=scores,
            decision=self._decision_to_qc_format(decision, revision_target),
            feedback=self._format_feedback_string(feedback_dict),
            revision_target=revision_target,
            type_specific_feedback={
                "universal_scores": universal_scores,
                "type_specific_scores": type_specific_scores,
                "type_specific_issues": self._get_type_issues(
                    scores, content_type
                ),
                "tone_match_assessment": (
                    f"Tone {'matches' if scores.get('tone_match', 0) >= 7 else 'needs adjustment for'} "
                    f"{content_type.value}"
                ),
                "type_requirements_met": [
                    k for k, v in scores.items() if v >= 7.0
                ],
                "type_requirements_missing": [
                    k for k, v in scores.items() if v < 7.0
                ],
                "weights_used": weights,
            },
        )

        # 10. Build QCOutput
        resolved_pass, resolved_reject = self._resolve_thresholds(content_type)
        output = QCOutput(
            result=result,
            scoring_weights_used=weights,
            type_adjustments_applied={
                "content_type": content_type.value,
                "pass_threshold": resolved_pass,
                "reject_threshold": resolved_reject,
                "default_pass_threshold": self.pass_threshold,
                "default_reject_threshold": self.reject_threshold,
                "threshold_overridden": resolved_pass != self.pass_threshold
                or resolved_reject != self.reject_threshold,
                "weight_overrides_applied": bool(
                    self._get_type_weight_overrides(content_type)
                ),
            },
        )

        return output

    # ------------------------------------------------------------------
    # PROMPT CONSTRUCTION
    # ------------------------------------------------------------------

    def _build_evaluation_prompt(
        self,
        humanized_post: HumanizedPost,
        visual_asset: Optional[VisualAsset],
        draft_post: DraftPost,
        content_type: ContentType,
        criteria_names: List[str],
    ) -> str:
        """Build the single-call evaluation prompt with full rubric."""
        rubric_text = self._format_rubric(content_type)

        visual_type = "none"
        visual_description = "No visual asset provided"
        if visual_asset is not None:
            visual_type = visual_asset.asset_type or "unknown"
            visual_description = visual_asset.prompt_used or "No description available"

        tone_expectation = TONE_EXPECTATIONS.get(
            content_type, "professional with insight"
        )

        return self._EVALUATION_PROMPT.format(
            post_text=humanized_post.humanized_text,
            visual_type=visual_type,
            visual_description=visual_description,
            content_type=content_type.value,
            tone_expectation=tone_expectation,
            rubric_text=rubric_text,
            criteria_names=", ".join(criteria_names),
        )

    def _format_rubric(self, content_type: ContentType) -> str:
        """Format all criteria (universal + type-specific) into prompt text."""
        lines: List[str] = []

        # Universal criteria
        lines.append("--- UNIVERSAL CRITERIA ---")
        for name, spec in UNIVERSAL_CRITERIA.items():
            lines.append(f"\n### {name} (weight: {spec['base_weight']})")
            lines.append(f"Question: {spec['prompt_hint']}")
            lines.append("Scoring guide:")
            for score in sorted(spec["rubric"].keys(), reverse=True):
                lines.append(f"  {score}: {spec['rubric'][score]}")

        # Type-specific criteria
        type_config = TYPE_CRITERIA.get(content_type, {})
        type_specific_names = [
            k
            for k in type_config.keys()
            if k not in ("weight_overrides", "tone_expectation")
        ]
        if type_specific_names:
            lines.append(
                f"\n--- TYPE-SPECIFIC CRITERIA ({content_type.value}) ---"
            )
            for name in type_specific_names:
                spec = type_config[name]
                lines.append(f"\n### {name} (weight: {spec['weight']})")
                lines.append(f"Description: {spec['desc']}")
                rubric = spec.get("rubric", {})
                if rubric:
                    lines.append("Scoring guide:")
                    for score in sorted(rubric.keys(), reverse=True):
                        lines.append(f"  {score}: {rubric[score]}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # SCORE PARSING
    # ------------------------------------------------------------------

    def _parse_scores(
        self, raw_response: str, expected_criteria: List[str]
    ) -> Dict[str, float]:
        """
        Parse Claude's JSON response into a flat criterion->score dict.

        Handles both raw JSON and markdown-fenced JSON. Raises on
        unparseable or incomplete responses (fail-fast).
        """
        cleaned = raw_response.strip()

        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            first_newline = cleaned.find("\n")
            if first_newline != -1:
                cleaned = cleaned[first_newline + 1 :]
            else:
                cleaned = cleaned[3:]
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[:-3]
            cleaned = cleaned.strip()

        parsed: Dict[str, Any] = json.loads(cleaned)

        # Extract scores from the nested structure
        raw_scores: Dict[str, Any] = parsed.get("scores", parsed)

        scores: Dict[str, float] = {}
        for criterion in expected_criteria:
            if criterion not in raw_scores:
                # Fail-fast: missing criterion defaults to 0 to surface issues
                self.logger.warning(
                    "Missing score for criterion '%s' - defaulting to 0.0 (fail-fast)",
                    criterion,
                )
                scores[criterion] = 0.0
                continue

            value = raw_scores[criterion]
            if isinstance(value, dict):
                score_val = value.get("score", 0.0)
            elif isinstance(value, (int, float)):
                score_val = value
            else:
                self.logger.warning(
                    "Unexpected score format for '%s': %r - defaulting to 0.0",
                    criterion,
                    value,
                )
                score_val = 0.0

            # Clamp to valid range
            scores[criterion] = max(0.0, min(10.0, float(score_val)))

        return scores

    # ------------------------------------------------------------------
    # WEIGHT RESOLUTION
    # ------------------------------------------------------------------

    def _get_all_criteria(
        self, content_type: ContentType
    ) -> Dict[str, Dict[str, Any]]:
        """Return merged dict of all criteria (universal + type-specific)."""
        criteria: Dict[str, Dict[str, Any]] = dict(UNIVERSAL_CRITERIA)

        type_config = TYPE_CRITERIA.get(content_type, {})
        for key, value in type_config.items():
            if key not in ("weight_overrides", "tone_expectation"):
                criteria[key] = value

        return criteria

    def _get_weights(self, content_type: ContentType) -> Dict[str, float]:
        """
        Build the final weights dict for a given content type.

        Starts with universal base weights, applies type-specific overrides,
        then adds type-specific criterion weights.
        """
        weights: Dict[str, float] = {
            name: spec["base_weight"]
            for name, spec in UNIVERSAL_CRITERIA.items()
        }

        # Apply type-specific weight overrides for universal criteria
        overrides = self._get_type_weight_overrides(content_type)
        for criterion_name, override_weight in overrides.items():
            if criterion_name in weights:
                weights[criterion_name] = override_weight

        # Add type-specific criterion weights
        type_config = TYPE_CRITERIA.get(content_type, {})
        for key, value in type_config.items():
            if key not in ("weight_overrides", "tone_expectation"):
                weights[key] = value["weight"]

        return weights

    def _get_type_weight_overrides(
        self, content_type: ContentType
    ) -> Dict[str, float]:
        """Extract weight_overrides from type config, if present."""
        type_config = TYPE_CRITERIA.get(content_type, {})
        return type_config.get("weight_overrides", {})

    # ------------------------------------------------------------------
    # SCORE AGGREGATION
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_weighted_score(
        scores: Dict[str, float], weights: Dict[str, float]
    ) -> float:
        """
        Calculate weighted aggregate score.

        Uses a proper weighted average: sum(score * weight) / sum(weights).
        Missing scores default to 0 (fail-fast philosophy -- surfaces issues
        immediately rather than hiding them).

        Args:
            scores: Criterion name to score (1-10) mapping.
            weights: Criterion name to weight mapping.

        Returns:
            Weighted average score rounded to two decimals.
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for criterion, weight in weights.items():
            score = scores.get(criterion, 0.0)
            weighted_sum += score * weight
            total_weight += weight

        if total_weight == 0.0:
            return 0.0

        return round(weighted_sum / total_weight, 2)

    # ------------------------------------------------------------------
    # THRESHOLD RESOLUTION
    # ------------------------------------------------------------------

    def _resolve_thresholds(self, content_type: ContentType) -> Tuple[float, float]:
        """Resolve pass/reject thresholds, preferring content-type overrides.

        Looks up the ``content_type_overrides`` section of the QC threshold
        config.  If the current content type has an entry, those values take
        precedence over the instance-level defaults.

        Args:
            content_type: The content type to resolve thresholds for.

        Returns:
            A ``(pass_threshold, reject_threshold)`` tuple.
        """
        overrides = self._type_overrides.get(content_type.value, {})
        pass_t: float = overrides.get("pass", self.pass_threshold)
        reject_t: float = overrides.get("reject", self.reject_threshold)
        return pass_t, reject_t

    # ------------------------------------------------------------------
    # DECISION LOGIC
    # ------------------------------------------------------------------

    def _make_decision(self, aggregate_score: float, content_type: ContentType) -> str:
        """
        Make pass / revise / reject decision based on aggregate score.

        Uses content-type-specific thresholds when available, falling back
        to the instance-level defaults.

        Args:
            aggregate_score: Weighted aggregate (0-10).
            content_type: The content type (used to resolve thresholds).

        Returns:
            One of ``"pass"``, ``"revise"``, ``"reject"``.
        """
        pass_t, reject_t = self._resolve_thresholds(content_type)
        if aggregate_score >= pass_t:
            return "pass"
        if aggregate_score >= reject_t:
            return "revise"
        return "reject"

    @staticmethod
    def _decision_to_qc_format(
        decision: str, revision_target: Optional[str]
    ) -> str:
        """
        Convert internal decision + target to ``QCResult.decision`` format.

        Expected values: PASS / REVISE_WRITER / REVISE_HUMANIZER /
        REVISE_VISUAL / REJECT.
        """
        if decision == "pass":
            return "PASS"
        if decision == "reject":
            return "REJECT"
        # decision == "revise"
        if revision_target == "humanizer":
            return "REVISE_HUMANIZER"
        if revision_target == "visual":
            return "REVISE_VISUAL"
        return "REVISE_WRITER"

    # ------------------------------------------------------------------
    # REVISION TARGET ROUTING
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_revision_target(scores: Dict[str, float]) -> str:
        """
        Determine which agent should handle the revision.

        Routing logic:
        - humanness or tone_match is the lowest score  -> ``"humanizer"``
        - visual_match is the lowest score              -> ``"visual"``
        - Otherwise (hook_strength, value_density, etc) -> ``"writer"``

        When multiple criteria are equally low, priority is:
        writer > humanizer > visual (writer can address the most issues).
        """
        # Identify lowest-scoring criteria buckets
        humanizer_criteria = {"humanness", "tone_match"}
        visual_criteria = {"visual_match"}
        # Everything else routes to writer

        # Find the single lowest score and its criterion
        if not scores:
            return "writer"

        sorted_scores: List[Tuple[str, float]] = sorted(
            scores.items(), key=lambda x: x[1]
        )
        lowest_criterion = sorted_scores[0][0]

        if lowest_criterion in humanizer_criteria:
            return "humanizer"
        if lowest_criterion in visual_criteria:
            return "visual"
        return "writer"

    # ------------------------------------------------------------------
    # FEEDBACK GENERATION
    # ------------------------------------------------------------------

    def _generate_feedback(
        self, scores: Dict[str, float], content_type: ContentType
    ) -> Dict[str, Any]:
        """
        Generate structured feedback dict for revision.

        Identifies low-scoring criteria, adds type-specific issues, and
        surfaces targeted improvement suggestions.
        """
        weights = self._get_weights(content_type)

        feedback: Dict[str, Any] = {
            "content_type": content_type.value,
            "overall_score": self._calculate_weighted_score(scores, weights),
            "low_scores": [],
            "type_specific_issues": self._get_type_issues(scores, content_type),
            "specific_suggestions": [],
        }

        suggestions_map = IMPROVEMENT_SUGGESTIONS.get(content_type, {})

        for criterion, score in scores.items():
            if score < 7.0:
                criterion_suggestions = suggestions_map.get(criterion, [])
                primary_suggestion = (
                    criterion_suggestions[0]
                    if criterion_suggestions
                    else f"Improve {criterion} (current: {score}/10)"
                )

                feedback["low_scores"].append(
                    {
                        "criterion": criterion,
                        "current_score": score,
                        "target_score": 8.0,
                        "weight": weights.get(criterion, 0.0),
                        "suggestion": primary_suggestion,
                    }
                )
                feedback["specific_suggestions"].append(primary_suggestion)

        # Sort low_scores by impact (weight * improvement potential)
        feedback["low_scores"].sort(
            key=lambda x: x["weight"] * (10.0 - x["current_score"]),
            reverse=True,
        )

        return feedback

    def _build_revision_instructions(
        self, scores: Dict[str, float], content_type: ContentType
    ) -> List[str]:
        """
        Build objective revision instructions stating WHAT needs to change.

        These are objective statements (e.g., "Hook doesn't create curiosity
        gap"), NOT creative suggestions (that is the downstream agent's job).
        """
        instructions: List[str] = []
        weights = self._get_weights(content_type)

        # Build impact-sorted list of failing criteria
        failing: List[Tuple[str, float, float]] = []
        for criterion, score in scores.items():
            if score < 7.0:
                weight = weights.get(criterion, 0.0)
                impact = weight * (10.0 - score)
                failing.append((criterion, score, impact))

        failing.sort(key=lambda x: x[2], reverse=True)

        for criterion, score, _impact in failing:
            instruction = self._criterion_to_instruction(
                criterion, score, content_type
            )
            instructions.append(instruction)

        return instructions

    @staticmethod
    def _criterion_to_instruction(
        criterion: str, score: float, content_type: ContentType
    ) -> str:
        """Convert a low-scoring criterion into an objective instruction."""
        instruction_templates: Dict[str, str] = {
            "hook_strength": (
                f"Hook scored {score:.1f}/10 -- lacks scroll-stopping power "
                f"for {content_type.value} content"
            ),
            "value_density": (
                f"Value density scored {score:.1f}/10 -- contains filler or "
                f"lacks actionable insights"
            ),
            "humanness": (
                f"Humanness scored {score:.1f}/10 -- contains AI-tell phrases "
                f"or unnatural patterns"
            ),
            "visual_match": (
                f"Visual match scored {score:.1f}/10 -- image does not "
                f"amplify the message for {content_type.value}"
            ),
            "controversy_safety": (
                f"Controversy safety scored {score:.1f}/10 -- claims may be "
                f"unsupported or tone may provoke negative reactions"
            ),
            "tone_match": (
                f"Tone match scored {score:.1f}/10 -- tone does not match "
                f"expectations for {content_type.value}"
            ),
            "metrics_credibility": (
                f"Metrics credibility scored {score:.1f}/10 -- numbers lack "
                f"specificity, sourcing, or believability"
            ),
            "intellectual_depth": (
                f"Intellectual depth scored {score:.1f}/10 -- analysis is "
                f"surface-level or does not engage meaningfully with research"
            ),
            "reproducibility": (
                f"Reproducibility scored {score:.1f}/10 -- steps are not "
                f"specific enough for a reader to follow"
            ),
            "community_authenticity": (
                f"Community authenticity scored {score:.1f}/10 -- post does "
                f"not feel connected to the community"
            ),
            "evaluation_balance": (
                f"Evaluation balance scored {score:.1f}/10 -- assessment is "
                f"not balanced or fair"
            ),
        }
        return instruction_templates.get(
            criterion,
            f"{criterion} scored {score:.1f}/10 -- needs improvement",
        )

    # ------------------------------------------------------------------
    # TYPE-SPECIFIC ISSUE DETECTION
    # ------------------------------------------------------------------

    @staticmethod
    def _get_type_issues(
        scores: Dict[str, float], content_type: ContentType
    ) -> List[str]:
        """
        Return list of type-specific issues based on low scores.

        These are objective statements about WHAT requirement failed, not
        suggestions for HOW to fix them.
        """
        issues: List[str] = []

        if content_type == ContentType.ENTERPRISE_CASE:
            if scores.get("metrics_credibility", 0) < 7:
                issues.append(
                    "Metrics need more specificity. Add exact numbers, "
                    "timeframes, and source attribution."
                )

        elif content_type == ContentType.PRIMARY_SOURCE:
            if scores.get("intellectual_depth", 0) < 7:
                issues.append(
                    "Analysis lacks depth. Post needs more interpretive "
                    "value beyond surface-level summary."
                )

        elif content_type == ContentType.AUTOMATION_CASE:
            if scores.get("reproducibility", 0) < 7:
                issues.append(
                    "Steps need more detail. Specific tools, versions, "
                    "and configuration are missing."
                )

        elif content_type == ContentType.COMMUNITY_CONTENT:
            if scores.get("community_authenticity", 0) < 7:
                issues.append(
                    "Community connection is weak. Attributions, diverse "
                    "perspectives, or discussion invitation are missing."
                )

        elif content_type == ContentType.TOOL_RELEASE:
            if scores.get("evaluation_balance", 0) < 7:
                issues.append(
                    "Evaluation is unbalanced. Limitations, caveats, or "
                    "fair comparison to alternatives are missing."
                )

        return issues

    # ------------------------------------------------------------------
    # FORMATTING HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _format_feedback_string(
        feedback_dict: Optional[Dict[str, Any]],
    ) -> str:
        """Convert feedback dict to a human-readable summary string."""
        if feedback_dict is None:
            return "Post passed quality evaluation."

        parts: List[str] = [
            f"Overall score: {feedback_dict.get('overall_score', 'N/A')}/10"
        ]

        low_scores = feedback_dict.get("low_scores", [])
        if low_scores:
            parts.append("Low-scoring criteria:")
            for ls in low_scores:
                parts.append(
                    f"  - {ls['criterion']}: {ls['current_score']}/10 "
                    f"(target: {ls['target_score']}) -- {ls['suggestion']}"
                )

        type_issues = feedback_dict.get("type_specific_issues", [])
        if type_issues:
            parts.append("Type-specific issues:")
            for issue in type_issues:
                parts.append(f"  - {issue}")

        return "\n".join(parts)
