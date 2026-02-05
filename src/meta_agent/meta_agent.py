"""
Main MetaAgent orchestrator for self-improvement cycles.

The ``MetaAgent`` is the top-level coordinator for the self-improvement
subsystem. It evaluates drafts against quality thresholds, decides when
to trigger improvement research, applies recommendations through the
modification safety system, and manages A/B experiments.

Architecture reference:
    - ``architecture.md`` lines 14305-15480  (Meta-Agent overview)
    - ``architecture.md`` lines 22438-22936  (Modification Safety)
    - ``architecture.md`` lines 22937-23466  (Single-Call Evaluation)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.meta_agent.models import (
    ModificationRequest,
    ModificationRiskLevel,
)
from src.meta_agent.single_call_evaluator import SingleCallEvaluator
from src.meta_agent.experimentation import ExperimentationEngine
from src.models import ContentType
from src.config import THRESHOLD_CONFIG, AutonomyLevel

if TYPE_CHECKING:
    from src.database import SupabaseDB

logger = logging.getLogger(__name__)


class MetaAgent:
    """
    Main orchestrator for the self-improvement meta-agent subsystem.

    Responsibilities:
    1. **Evaluate drafts** against content-type-specific quality thresholds.
    2. **Run improvement cycles** -- research, apply recommendations, check
       experiments.
    3. **Classify risk** of proposed modifications by component type.
    4. **Suggest experiments** based on performance gap analysis.
    5. **Notify changes** with structured logging.

    Args:
        evaluator: ``SingleCallEvaluator`` for post quality assessment.
        researcher: Research agent for knowledge gap filling.
        safety_system: ``ModificationSafetySystem`` for safe change application.
        experimenter: ``ExperimentationEngine`` for A/B testing.
        db: ``SupabaseDB`` database client.

    Usage::

        meta = MetaAgent(
            evaluator=evaluator,
            researcher=research_agent,
            safety_system=mod_safety,
            experimenter=experiment_engine,
            db=database,
        )

        # Evaluate a draft
        result = await meta.evaluate_draft(
            draft="My LinkedIn post...",
            context={"content_type": ContentType.ENTERPRISE_CASE},
        )
        if result["action"] == "pass":
            print("Draft approved!")

        # Run periodic improvement cycle
        await meta.run_improvement_cycle()
    """

    def __init__(
        self,
        evaluator: SingleCallEvaluator,
        researcher: Any,  # ResearchAgent (not yet created as a module)
        safety_system: Any,  # ModificationSafetySystem (not yet created as a module)
        experimenter: ExperimentationEngine,
        db: SupabaseDB,
    ) -> None:
        self.evaluator = evaluator
        self.researcher = researcher
        self.safety_system = safety_system
        self.experimenter = experimenter
        self.db = db

    async def evaluate_draft(
        self,
        draft: str,
        context: dict,
    ) -> dict:
        """
        Evaluate a post draft against quality thresholds.

        Uses the ``SingleCallEvaluator`` to score the draft, then compares
        the weighted total against the content-type-specific pass threshold
        from ``THRESHOLD_CONFIG``.

        The returned dict includes the score, detailed feedback, a decision
        action (``"pass"`` or ``"revise"``), and the iteration count.

        Args:
            draft: Full text of the LinkedIn post draft.
            context: Pipeline context dict. Expected keys:
                - ``content_type`` (``ContentType``): The post content type.
                - ``iteration`` (``int``, optional): Current revision iteration
                  (defaults to ``1``).

        Returns:
            Dict with keys:
                - ``score`` (``float``): Weighted evaluation score.
                - ``feedback`` (``dict``): Full evaluation feedback including
                  strengths, weaknesses, suggestions.
                - ``action`` (``str``): ``"pass"`` if score meets threshold,
                  ``"revise"`` otherwise.
                - ``iteration`` (``int``): Current iteration number.
                - ``threshold`` (``float``): The pass threshold used.
                - ``content_type`` (``str``): The content type value.
        """
        content_type = context.get("content_type", ContentType.ENTERPRISE_CASE)
        iteration = context.get("iteration", 1)

        logger.info(
            "[META_AGENT] Evaluating draft: content_type=%s, iteration=%d",
            content_type.value,
            iteration,
        )

        # Run evaluation
        evaluation = await self.evaluator.evaluate(
            post_content=draft,
            content_type=content_type,
        )

        # Determine pass threshold using centralized config
        pass_threshold = THRESHOLD_CONFIG.get_pass_threshold(content_type)
        passes = evaluation.weighted_total >= pass_threshold

        action = "pass" if passes else "revise"

        logger.info(
            "[META_AGENT] Evaluation result: score=%.2f, threshold=%.2f, action=%s",
            evaluation.weighted_total,
            pass_threshold,
            action,
        )

        return {
            "score": evaluation.weighted_total,
            "feedback": {
                "scores": evaluation.scores,
                "strengths": evaluation.strengths,
                "weaknesses": evaluation.weaknesses,
                "suggestions": evaluation.specific_suggestions,
                "patterns": evaluation.patterns_detected,
                "knowledge_gaps": evaluation.knowledge_gaps,
                "recommended_revisions": evaluation.recommended_revisions,
            },
            "action": action,
            "iteration": iteration,
            "threshold": pass_threshold,
            "content_type": content_type.value,
        }

    async def run_improvement_cycle(self) -> None:
        """
        Run a full improvement cycle.

        Steps:
        1. Check if research should be triggered (based on performance
           trends, time since last research, etc.).
        2. If triggered, execute research via the researcher agent.
        3. Apply research recommendations through the safety system,
           classifying each by risk level.
        4. Check experiment status and log progress.

        This method is designed to be called periodically (e.g., after
        each post creation cycle or on a schedule).
        """
        logger.info("[META_AGENT] Starting improvement cycle")

        # --- Step 1: Check if research is needed ---------------------------
        should_research = await self._should_research()

        if should_research:
            logger.info("[META_AGENT] Research triggered, executing...")

            # --- Step 2: Execute research ----------------------------------
            try:
                report = await self.researcher.run_research_cycle()
                logger.info(
                    "[META_AGENT] Research complete: %d recommendations",
                    len(getattr(report, "recommendations", [])),
                )

                # --- Step 3: Apply recommendations through safety system ---
                recommendations = getattr(report, "recommendations", [])
                if recommendations:
                    changes_applied: List[dict] = []

                    for rec in recommendations:
                        # Convert recommendation to dict if needed
                        rec_dict = (
                            rec if isinstance(rec, dict) else {
                                "component": getattr(rec, "component", "unknown"),
                                "change": getattr(rec, "change", ""),
                                "priority": getattr(rec, "priority", 3),
                                "confidence": getattr(rec, "confidence", 0.5),
                            }
                        )

                        # Classify risk
                        risk_level = self._classify_risk(rec_dict)
                        rec_dict["risk_level"] = risk_level

                        logger.info(
                            "[META_AGENT] Applying recommendation: "
                            "component=%s, risk=%s, change=%s",
                            rec_dict.get("component", "unknown"),
                            risk_level,
                            rec_dict.get("change", "")[:100],
                        )

                        try:
                            result = await self.safety_system.process_modification(
                                rec_dict
                            )
                            changes_applied.append({
                                "component": rec_dict.get("component"),
                                "change": rec_dict.get("change"),
                                "risk": risk_level,
                                "result": str(result),
                            })
                        except Exception as exc:
                            logger.error(
                                "[META_AGENT] Failed to apply recommendation "
                                "for '%s': %s",
                                rec_dict.get("component", "unknown"),
                                exc,
                            )

                    if changes_applied:
                        await self._notify_changes(changes_applied)

            except Exception as exc:
                logger.error(
                    "[META_AGENT] Research cycle failed: %s", exc
                )
        else:
            logger.info("[META_AGENT] No research needed at this time")

        # --- Step 4: Check experiment status -------------------------------
        experiment_status = await self.experimenter.get_status()
        if experiment_status:
            logger.info(
                "[META_AGENT] Active experiment: '%s' (status=%s, "
                "control_avg=%.2f, treatment_avg=%.2f)",
                experiment_status.get("name", "unknown"),
                experiment_status.get("status", "unknown"),
                experiment_status.get("control", {}).get("avg_score", 0),
                experiment_status.get("treatment", {}).get("avg_score", 0),
            )
        else:
            logger.info("[META_AGENT] No active experiment")

        logger.info("[META_AGENT] Improvement cycle complete")

    def _classify_risk(self, rec: dict) -> str:
        """
        Classify modification risk level based on the target component.

        Risk classification rules:
        - ``prompt`` or ``system_prompt`` in component -> ``MEDIUM``
        - ``threshold`` or ``scoring`` in component -> ``HIGH``
        - Everything else -> ``LOW``

        This is a simplified heuristic. The full ``ModificationSafetySystem``
        uses the ``modification_risk_classification`` mapping for
        fine-grained risk assessment.

        Args:
            rec: Recommendation dict with at least a ``component`` key.

        Returns:
            Risk level string: ``"low"``, ``"medium"``, or ``"high"``.
        """
        component = rec.get("component", "").lower()

        if "prompt" in component or "system_prompt" in component:
            return ModificationRiskLevel.MEDIUM.value

        if "threshold" in component or "scoring" in component:
            return ModificationRiskLevel.HIGH.value

        return ModificationRiskLevel.LOW.value

    async def suggest_experiment(self) -> Optional[dict]:
        """
        Analyze performance gaps and suggest a new A/B experiment.

        Reviews recent post performance to identify the weakest evaluation
        criterion, then proposes an experiment to test an improvement
        strategy for that area.

        Returns:
            Experiment suggestion dict with keys ``name``, ``hypothesis``,
            ``variable``, ``control_config``, ``treatment_config``, or
            ``None`` if no significant gap is found.
        """
        logger.info("[META_AGENT] Analyzing performance for experiment suggestions")

        try:
            recent_posts = await self.db.get_recent_posts(limit=20)
        except Exception as exc:
            logger.error(
                "[META_AGENT] Failed to get recent posts for analysis: %s", exc
            )
            return None

        if not recent_posts or len(recent_posts) < 5:
            logger.info(
                "[META_AGENT] Not enough posts (%d) for experiment suggestion",
                len(recent_posts) if recent_posts else 0,
            )
            return None

        # Aggregate scores by criterion from recent evaluations
        criterion_scores: Dict[str, List[float]] = {}
        for post in recent_posts:
            post_data = post if isinstance(post, dict) else {}
            evaluation = post_data.get("meta_evaluation", {})
            scores = evaluation.get("scores", {}) if isinstance(evaluation, dict) else {}

            for criterion, score in scores.items():
                if criterion not in criterion_scores:
                    criterion_scores[criterion] = []
                try:
                    criterion_scores[criterion].append(float(score))
                except (ValueError, TypeError):
                    continue

        if not criterion_scores:
            logger.info("[META_AGENT] No criterion scores found in recent posts")
            return None

        # Find the weakest criterion
        criterion_averages = {
            name: sum(scores) / len(scores)
            for name, scores in criterion_scores.items()
            if scores
        }

        if not criterion_averages:
            return None

        weakest = min(criterion_averages, key=criterion_averages.get)  # type: ignore[arg-type]
        weakest_avg = criterion_averages[weakest]

        # Only suggest if there's a meaningful gap (below 7.0)
        if weakest_avg >= 7.0:
            logger.info(
                "[META_AGENT] No significant performance gap (weakest: %s=%.2f)",
                weakest,
                weakest_avg,
            )
            return None

        suggestion = {
            "name": f"Improve {weakest.replace('_', ' ').title()}",
            "hypothesis": (
                f"Targeted improvements to {weakest} will raise "
                f"average score from {weakest_avg:.1f} to 7.0+"
            ),
            "variable": weakest,
            "control_config": {"strategy": "current"},
            "treatment_config": {"strategy": f"enhanced_{weakest}"},
        }

        logger.info(
            "[META_AGENT] Experiment suggestion: %s (current avg: %.2f)",
            suggestion["name"],
            weakest_avg,
        )
        return suggestion

    async def _should_research(self) -> bool:
        """
        Determine whether a research cycle should be triggered.

        Checks:
        1. Time since last research (> 24 hours triggers research).
        2. Recent performance trend (average score below 7.0 triggers
           research).

        Returns:
            ``True`` if research should be triggered.
        """
        from datetime import timedelta
        from src.utils import utc_now as _utc_now

        # Check time since last research
        try:
            last_research = await self.db.get_last_research_date()
            if last_research is not None:
                hours_since = (
                    _utc_now() - last_research
                ).total_seconds() / 3600.0
                if hours_since < 24.0:
                    logger.info(
                        "[META_AGENT] Last research %.1f hours ago, skipping",
                        hours_since,
                    )
                    return False
        except Exception as exc:
            logger.warning(
                "[META_AGENT] Failed to check last research date: %s", exc
            )

        # Check recent performance
        try:
            avg_score = await self.db.get_average_score()
            if avg_score < 7.0:
                logger.info(
                    "[META_AGENT] Average score %.2f below 7.0, triggering research",
                    avg_score,
                )
                return True
        except Exception as exc:
            logger.warning(
                "[META_AGENT] Failed to check average score: %s", exc
            )

        # Default: trigger research if enough time has passed
        return True

    async def _notify_changes(self, changes: list) -> None:
        """
        Log applied changes with structured [META_AGENT] prefix.

        Each change is logged at INFO level with details about the
        component affected, the change made, and the risk level.

        Args:
            changes: List of change dicts, each with keys ``component``,
                ``change``, ``risk``, and ``result``.
        """
        logger.info(
            "[META_AGENT] === Changes Applied: %d ===", len(changes)
        )
        for i, change in enumerate(changes, 1):
            logger.info(
                "[META_AGENT] Change %d/%d: component=%s, risk=%s, "
                "change=%s, result=%s",
                i,
                len(changes),
                change.get("component", "unknown"),
                change.get("risk", "unknown"),
                str(change.get("change", ""))[:200],
                str(change.get("result", ""))[:100],
            )
        logger.info("[META_AGENT] === End Changes ===")


# ===========================================================================
# PUBLIC API
# ===========================================================================

__all__ = [
    "MetaAgent",
]
