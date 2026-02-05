"""
A/B testing framework for the LinkedIn Super Agent meta-agent subsystem.

Provides the ``ExperimentationEngine`` which manages the full lifecycle of
content strategy experiments: creation, variant assignment, result recording,
statistical analysis, and automatic application of winning strategies.

Only one experiment can run at a time. Variants alternate based on post
counts (even/odd) to ensure balanced assignment without randomisation
complexity.

Architecture reference:
    - ``architecture.md`` Experimentation Engine section
    - ``architecture.md`` lines 14305-15480  (Meta-Agent overview)
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.meta_agent.models import (
    Experiment,
    ExperimentVariant,
    ExperimentStatus,
)
from src.utils import generate_id, utc_now

if TYPE_CHECKING:
    from src.database import SupabaseDB

logger = logging.getLogger(__name__)


class ExperimentationEngine:
    """
    A/B testing framework for content strategies.

    Manages experiment lifecycle:
    1. **Create** -- define hypothesis, control, and treatment variants.
    2. **Start** -- activate the experiment (only one can run at a time).
    3. **Assign** -- alternate variant assignment based on post counts.
    4. **Record** -- log scores per variant per post.
    5. **Check completion** -- evaluate when min sample size is reached.
    6. **Complete** -- determine winner, calculate lift, optionally apply.

    Args:
        db: ``SupabaseDB`` database client for experiment persistence.
        modification_engine: Optional engine for auto-applying winning
            strategies. When ``None``, winners are logged but not applied.

    Usage::

        engine = ExperimentationEngine(db=database)
        exp_id = await engine.create_experiment(
            name="Hook Style Test",
            hypothesis="Question hooks increase engagement by 15%",
            variable="hook_style",
            control_config={"style": "statement"},
            treatment_config={"style": "question"},
        )
        await engine.start_experiment(exp_id)

        # During post creation:
        variant = await engine.assign_variant()
        if variant:
            # Apply variant.config to the pipeline
            pass

        # After post scoring:
        await engine.record_result(post_id="...", variant_name="treatment", score=8.5)
    """

    def __init__(
        self,
        db: SupabaseDB,
        modification_engine: Any = None,
    ) -> None:
        self.db = db
        self.modification_engine = modification_engine
        self.current_experiment: Optional[Experiment] = None

    async def create_experiment(
        self,
        name: str,
        hypothesis: str,
        variable: str,
        control_config: Dict[str, Any],
        treatment_config: Dict[str, Any],
        control_description: str = "Current approach (control)",
        treatment_description: str = "New approach (treatment)",
        min_posts_per_variant: int = 10,
        max_posts_per_variant: int = 30,
        confidence_threshold: float = 0.95,
    ) -> str:
        """
        Create a new A/B experiment.

        Validates that no experiment is currently running before creating
        a new one. The experiment starts in ``DRAFT`` status.

        Args:
            name: Human-readable experiment name.
            hypothesis: What we expect the treatment to achieve.
            variable: The variable being tested (e.g. ``"hook_style"``).
            control_config: Configuration dict for the control variant.
            treatment_config: Configuration dict for the treatment variant.
            control_description: Description of the control variant.
            treatment_description: Description of the treatment variant.
            min_posts_per_variant: Minimum posts per variant for significance.
            max_posts_per_variant: Maximum posts per variant (hard stop).
            confidence_threshold: Required confidence to declare winner.

        Returns:
            UUID string of the created experiment.

        Raises:
            ValueError: If an experiment is already running.
        """
        # Validate no running experiment
        active = await self.db.get_active_experiments()
        if active:
            raise ValueError(
                f"Cannot create experiment: {len(active)} experiment(s) "
                f"already active. Stop current experiment(s) first."
            )

        experiment_id = generate_id()

        control = ExperimentVariant(
            name="control",
            config=control_config,
            description=control_description,
        )
        treatment = ExperimentVariant(
            name="treatment",
            config=treatment_config,
            description=treatment_description,
        )

        experiment = Experiment(
            id=experiment_id,
            name=name,
            hypothesis=hypothesis,
            variable=variable,
            control=control,
            treatment=treatment,
            status=ExperimentStatus.DRAFT,
            min_posts_per_variant=min_posts_per_variant,
            max_posts_per_variant=max_posts_per_variant,
            confidence_threshold=confidence_threshold,
        )

        # Persist to database
        experiment_data = self._experiment_to_dict(experiment)
        await self.db.save_experiment(experiment_data)

        self.current_experiment = experiment
        logger.info(
            "[EXPERIMENT] Created experiment '%s' (id=%s): %s",
            name,
            experiment_id,
            hypothesis,
        )
        return experiment_id

    async def start_experiment(self, experiment_id: str) -> None:
        """
        Start a draft experiment, making it active.

        Transitions the experiment from ``DRAFT`` to ``RUNNING`` status
        and sets the ``started_at`` timestamp.

        Args:
            experiment_id: UUID of the experiment to start.

        Raises:
            ValueError: If the experiment is not in ``DRAFT`` status.
        """
        experiment = await self._load_experiment(experiment_id)

        if experiment.status != ExperimentStatus.DRAFT:
            raise ValueError(
                f"Cannot start experiment '{experiment.name}': "
                f"status is '{experiment.status.value}', expected 'draft'"
            )

        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = utc_now()

        await self.db.update_experiment({
            "id": experiment_id,
            "status": ExperimentStatus.RUNNING.value,
            "started_at": experiment.started_at.isoformat(),
        })

        self.current_experiment = experiment
        logger.info(
            "[EXPERIMENT] Started experiment '%s' (id=%s)",
            experiment.name,
            experiment_id,
        )

    async def assign_variant(self) -> Optional[ExperimentVariant]:
        """
        Assign the next variant for a new post.

        Uses simple alternation based on total post counts across variants
        (even count -> control, odd count -> treatment). This ensures
        balanced assignment without requiring randomisation infrastructure.

        Returns:
            The ``ExperimentVariant`` to use for the next post,
            or ``None`` if no experiment is running.
        """
        if self.current_experiment is None:
            return None

        if self.current_experiment.status != ExperimentStatus.RUNNING:
            return None

        # Alternate based on total post count
        total_posts = (
            self.current_experiment.control.post_count
            + self.current_experiment.treatment.post_count
        )

        if total_posts % 2 == 0:
            variant = self.current_experiment.control
        else:
            variant = self.current_experiment.treatment

        logger.info(
            "[EXPERIMENT] Assigned variant '%s' (total_posts=%d)",
            variant.name,
            total_posts,
        )
        return variant

    async def record_result(
        self,
        post_id: str,
        variant_name: str,
        score: float,
    ) -> None:
        """
        Record a result for a specific variant after post scoring.

        Updates the variant's post count, total score, and post ID list.
        After recording, checks whether the experiment has reached
        completion criteria.

        Args:
            post_id: UUID of the scored post.
            variant_name: Name of the variant (``"control"`` or ``"treatment"``).
            score: QC score for the post (1-10 scale).

        Raises:
            ValueError: If no experiment is running or variant name is invalid.
        """
        if self.current_experiment is None:
            raise ValueError("No experiment is currently running")

        if self.current_experiment.status != ExperimentStatus.RUNNING:
            raise ValueError(
                f"Experiment '{self.current_experiment.name}' is not running "
                f"(status: {self.current_experiment.status.value})"
            )

        # Find the correct variant
        if variant_name == "control":
            variant = self.current_experiment.control
        elif variant_name == "treatment":
            variant = self.current_experiment.treatment
        else:
            raise ValueError(
                f"Unknown variant '{variant_name}'. "
                f"Expected 'control' or 'treatment'."
            )

        # Update variant stats
        variant.post_count += 1
        variant.total_score += score
        variant.post_ids.append(post_id)

        logger.info(
            "[EXPERIMENT] Recorded result for '%s': post=%s, score=%.2f "
            "(total posts: %d, avg: %.2f)",
            variant_name,
            post_id,
            score,
            variant.post_count,
            variant.total_score / variant.post_count if variant.post_count > 0 else 0,
        )

        # Persist updated experiment state
        await self.db.update_experiment(
            self._experiment_to_dict(self.current_experiment)
        )

        # Check if experiment should complete
        await self._check_completion()

    async def _check_completion(self) -> None:
        """
        Check whether the current experiment has reached completion criteria.

        Completion occurs when:
        - Both variants have at least ``min_posts_per_variant`` posts, OR
        - Either variant has reached ``max_posts_per_variant`` posts.

        Uses simple average comparison for statistical analysis. Production
        systems should integrate ``scipy.stats.ttest_ind`` for proper
        significance testing.
        """
        if self.current_experiment is None:
            return

        exp = self.current_experiment
        control = exp.control
        treatment = exp.treatment

        # Check minimum sample size reached
        min_reached = (
            control.post_count >= exp.min_posts_per_variant
            and treatment.post_count >= exp.min_posts_per_variant
        )

        # Check maximum sample size (hard stop)
        max_reached = (
            control.post_count >= exp.max_posts_per_variant
            or treatment.post_count >= exp.max_posts_per_variant
        )

        if not min_reached and not max_reached:
            return

        # Calculate simple averages
        control_avg = (
            control.total_score / control.post_count
            if control.post_count > 0
            else 0.0
        )
        treatment_avg = (
            treatment.total_score / treatment.post_count
            if treatment.post_count > 0
            else 0.0
        )

        logger.info(
            "[EXPERIMENT] Checking completion for '%s': "
            "control_avg=%.2f (%d posts), treatment_avg=%.2f (%d posts)",
            exp.name,
            control_avg,
            control.post_count,
            treatment_avg,
            treatment.post_count,
        )

        # Simple comparison (production: use scipy.stats.ttest_ind)
        # For now, we use a simple heuristic: the difference must be at least
        # 0.5 points to declare a winner; otherwise it's inconclusive.
        difference = abs(treatment_avg - control_avg)
        significant = difference >= 0.5

        early_stop = max_reached and not min_reached

        if significant or early_stop:
            await self._complete_experiment(early_stop=early_stop)
        else:
            logger.info(
                "[EXPERIMENT] Difference (%.2f) not yet significant, continuing",
                difference,
            )

    async def _complete_experiment(self, early_stop: bool = False) -> None:
        """
        Complete the current experiment and determine the winner.

        Calculates which variant performed better, the percentage lift,
        and whether the result meets the confidence threshold. If
        confidence is sufficient and a ``modification_engine`` is
        configured, automatically applies the winning strategy.

        Args:
            early_stop: Whether the experiment was stopped early (before
                min sample size) due to reaching max posts.
        """
        if self.current_experiment is None:
            return

        exp = self.current_experiment
        control = exp.control
        treatment = exp.treatment

        control_avg = (
            control.total_score / control.post_count
            if control.post_count > 0
            else 0.0
        )
        treatment_avg = (
            treatment.total_score / treatment.post_count
            if treatment.post_count > 0
            else 0.0
        )

        # Determine winner
        if treatment_avg > control_avg:
            winner = "treatment"
            lift = (
                ((treatment_avg - control_avg) / control_avg * 100.0)
                if control_avg > 0
                else 0.0
            )
        elif control_avg > treatment_avg:
            winner = "control"
            lift = (
                ((control_avg - treatment_avg) / treatment_avg * 100.0)
                if treatment_avg > 0
                else 0.0
            )
        else:
            winner = "tie"
            lift = 0.0

        # Simple confidence heuristic based on sample size and difference
        # Production: replace with proper t-test p-value calculation
        total_posts = control.post_count + treatment.post_count
        min_posts = min(control.post_count, treatment.post_count)
        difference = abs(treatment_avg - control_avg)

        # Heuristic confidence: scales with sample size and effect size
        if min_posts >= exp.min_posts_per_variant and difference >= 0.5:
            confidence = min(0.7 + (min_posts / 100.0) + (difference / 10.0), 0.99)
        else:
            confidence = 0.5

        exp.winner = winner
        exp.lift = round(lift, 2)
        exp.status = ExperimentStatus.STOPPED if early_stop else ExperimentStatus.COMPLETED
        exp.completed_at = utc_now()

        # Persist final state
        await self.db.update_experiment(
            self._experiment_to_dict(exp)
        )

        logger.info(
            "[EXPERIMENT] Experiment '%s' completed: winner=%s, lift=%.2f%%, "
            "confidence=%.2f, early_stop=%s",
            exp.name,
            winner,
            lift,
            confidence,
            early_stop,
        )

        # Auto-apply winning strategy if confidence is high enough
        if (
            confidence >= exp.confidence_threshold
            and winner != "tie"
            and self.modification_engine is not None
        ):
            winning_variant = treatment if winner == "treatment" else control
            try:
                await self.modification_engine.apply_recommendations([
                    {
                        "component": exp.variable,
                        "change": f"Apply winning experiment config: {winning_variant.config}",
                        "priority": 1,
                        "confidence": confidence,
                    }
                ])
                logger.info(
                    "[EXPERIMENT] Auto-applied winning strategy from '%s'",
                    exp.name,
                )
            except Exception as exc:
                logger.error(
                    "[EXPERIMENT] Failed to auto-apply winning strategy: %s", exc
                )
        elif winner != "tie" and confidence < exp.confidence_threshold:
            logger.info(
                "[EXPERIMENT] Winner '%s' not auto-applied: "
                "confidence %.2f < threshold %.2f",
                winner,
                confidence,
                exp.confidence_threshold,
            )

        # Clear current experiment
        self.current_experiment = None

    async def get_status(self) -> Optional[dict]:
        """
        Get the status of the current experiment.

        Returns:
            Dict with experiment status information, or ``None`` if no
            experiment is loaded. Keys include:
            ``id``, ``name``, ``status``, ``hypothesis``, ``variable``,
            ``control`` (variant stats), ``treatment`` (variant stats),
            ``winner``, ``lift``.
        """
        if self.current_experiment is None:
            return None

        exp = self.current_experiment
        control_avg = (
            exp.control.total_score / exp.control.post_count
            if exp.control.post_count > 0
            else 0.0
        )
        treatment_avg = (
            exp.treatment.total_score / exp.treatment.post_count
            if exp.treatment.post_count > 0
            else 0.0
        )

        return {
            "id": exp.id,
            "name": exp.name,
            "status": exp.status.value,
            "hypothesis": exp.hypothesis,
            "variable": exp.variable,
            "control": {
                "name": exp.control.name,
                "post_count": exp.control.post_count,
                "avg_score": round(control_avg, 2),
            },
            "treatment": {
                "name": exp.treatment.name,
                "post_count": exp.treatment.post_count,
                "avg_score": round(treatment_avg, 2),
            },
            "winner": exp.winner,
            "lift": exp.lift,
        }

    async def _get_post_scores(self, post_ids: List[str]) -> List[float]:
        """
        Retrieve QC scores for a list of post IDs from the database.

        Args:
            post_ids: List of post UUID strings.

        Returns:
            List of float scores corresponding to the post IDs.
            Posts not found in the database are excluded from the result.
        """
        scores: List[float] = []
        for post_id in post_ids:
            try:
                post_data = await self.db.get_post(post_id)
                if post_data and "qc_score" in post_data:
                    scores.append(float(post_data["qc_score"]))
            except Exception as exc:
                logger.warning(
                    "[EXPERIMENT] Failed to get score for post %s: %s",
                    post_id,
                    exc,
                )
        return scores

    async def _load_experiment(self, experiment_id: str) -> Experiment:
        """
        Load an experiment from the database by ID.

        Args:
            experiment_id: UUID of the experiment.

        Returns:
            The loaded ``Experiment`` instance.

        Raises:
            ValueError: If the experiment is not found.
        """
        data = await self.db.get_experiment(experiment_id)
        if data is None:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        return self._dict_to_experiment(data)

    def _experiment_to_dict(self, experiment: Experiment) -> Dict[str, Any]:
        """
        Convert an ``Experiment`` to a flat dict for database persistence.

        Serializes nested ``ExperimentVariant`` objects and enum values
        into JSON-compatible formats for Supabase storage.

        Args:
            experiment: The ``Experiment`` to serialize.

        Returns:
            Flat dict suitable for database insert/update.
        """
        return {
            "id": experiment.id,
            "name": experiment.name,
            "hypothesis": experiment.hypothesis,
            "variable": experiment.variable,
            "control_value": asdict(experiment.control),
            "treatment_value": asdict(experiment.treatment),
            "status": experiment.status.value,
            "min_posts_per_variant": experiment.min_posts_per_variant,
            "max_posts_per_variant": experiment.max_posts_per_variant,
            "confidence_threshold": experiment.confidence_threshold,
            "winner": experiment.winner,
            "lift": experiment.lift,
            "created_at": experiment.created_at.isoformat(),
            "started_at": (
                experiment.started_at.isoformat()
                if experiment.started_at
                else None
            ),
            "completed_at": (
                experiment.completed_at.isoformat()
                if experiment.completed_at
                else None
            ),
        }

    def _dict_to_experiment(self, data: Dict[str, Any]) -> Experiment:
        """
        Convert a database dict back to an ``Experiment`` instance.

        Deserializes nested variant dicts and status strings into their
        proper dataclass and enum types.

        Args:
            data: Raw dict from database query.

        Returns:
            Reconstructed ``Experiment`` instance.
        """
        from datetime import datetime

        control_data = data.get("control_value", {})
        treatment_data = data.get("treatment_value", {})

        control = ExperimentVariant(
            name=control_data.get("name", "control"),
            config=control_data.get("config", {}),
            description=control_data.get("description", ""),
            post_count=control_data.get("post_count", 0),
            total_score=control_data.get("total_score", 0.0),
            post_ids=control_data.get("post_ids", []),
        )

        treatment = ExperimentVariant(
            name=treatment_data.get("name", "treatment"),
            config=treatment_data.get("config", {}),
            description=treatment_data.get("description", ""),
            post_count=treatment_data.get("post_count", 0),
            total_score=treatment_data.get("total_score", 0.0),
            post_ids=treatment_data.get("post_ids", []),
        )

        # Parse status enum
        status_str = data.get("status", "draft")
        try:
            status = ExperimentStatus(status_str)
        except ValueError:
            status = ExperimentStatus.DRAFT

        # Parse timestamps
        def _parse_ts(val: Any) -> Optional[datetime]:
            if val is None:
                return None
            if isinstance(val, datetime):
                return val
            return datetime.fromisoformat(str(val))

        return Experiment(
            id=data["id"],
            name=data.get("name", ""),
            hypothesis=data.get("hypothesis", ""),
            variable=data.get("variable", ""),
            control=control,
            treatment=treatment,
            status=status,
            min_posts_per_variant=data.get("min_posts_per_variant", 10),
            max_posts_per_variant=data.get("max_posts_per_variant", 30),
            confidence_threshold=data.get("confidence_threshold", 0.95),
            winner=data.get("winner"),
            lift=data.get("lift"),
            created_at=_parse_ts(data.get("created_at")) or utc_now(),
            started_at=_parse_ts(data.get("started_at")),
            completed_at=_parse_ts(data.get("completed_at")),
        )


# ===========================================================================
# PUBLIC API
# ===========================================================================

__all__ = [
    "ExperimentationEngine",
]
