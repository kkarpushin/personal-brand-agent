"""
Continuous Learning Engine for the LinkedIn Super Agent.

Learns from EVERY iteration, starting from the first post.
Extracts micro-learnings from evaluation feedback, tracks confidence,
and formats learnings for prompt injection.

This is the "shallow but constant" learning component (vs. ResearchAgent
which is "deep but periodic").

Architecture reference:
    - ``architecture.md`` lines 14564-15255 (ContinuousLearningEngine full spec)
    - ``architecture.md`` lines 15259-15303 (Integration with Pipeline)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.meta_agent.models import (
    LearningType,
    LearningSource,
    MicroLearning,
    IterationLearnings,
)
from src.utils import utc_now, generate_id

if TYPE_CHECKING:
    from src.database import SupabaseDB
    from src.meta_agent.knowledge_base import KnowledgeBase
    from src.models import ContentType

logger = logging.getLogger("ContinuousLearningEngine")


# =============================================================================
# BOOTSTRAP LEARNINGS
# Initial high-confidence rules to guide first posts
# =============================================================================

BOOTSTRAP_LEARNINGS: List[Dict[str, Any]] = [
    {
        "learning_type": LearningType.HOOK_PATTERN,
        "description": "Start with a number or surprising statistic to grab attention",
        "rule": "hook_start_with_number",
        "affected_component": "writer",
        "confidence": 0.7,
    },
    {
        "learning_type": LearningType.HOOK_PATTERN,
        "description": "Ask a provocative question that challenges assumptions",
        "rule": "hook_provocative_question",
        "affected_component": "writer",
        "confidence": 0.65,
    },
    {
        "learning_type": LearningType.CONTENT_STRUCTURE,
        "description": "Keep paragraphs short (2-3 lines max) for mobile readability",
        "rule": "short_paragraphs",
        "affected_component": "writer",
        "confidence": 0.8,
    },
    {
        "learning_type": LearningType.CONTENT_STRUCTURE,
        "description": "Use white space generously - LinkedIn rewards scannability",
        "rule": "generous_whitespace",
        "affected_component": "writer",
        "confidence": 0.75,
    },
    {
        "learning_type": LearningType.VISUAL_STYLE,
        "description": "Photos with human faces get 38% more engagement",
        "rule": "prefer_human_faces",
        "affected_component": "visual_creator",
        "confidence": 0.7,
    },
    {
        "learning_type": LearningType.TONE_ADJUSTMENT,
        "description": "Use conversational tone, avoid corporate jargon",
        "rule": "conversational_tone",
        "affected_component": "humanizer",
        "confidence": 0.75,
    },
]


class ContinuousLearningEngine:
    """Engine that learns from EVERY iteration, starting from first post.

    KEY DIFFERENCE FROM RESEARCH AGENT:
    - ResearchAgent: Deep research triggered by schedule/events
    - ContinuousLearningEngine: Shallow but constant learning every iteration

    Both work together:
    - CLE handles immediate micro-learnings
    - ResearchAgent handles deep strategic research

    Usage::

        from src.meta_agent.learning_engine import ContinuousLearningEngine
        from src.database import get_db

        db = await get_db()
        engine = ContinuousLearningEngine(db=db)

        # Bootstrap for first post
        if is_first_post:
            await engine.handle_first_post()

        # Learn from every iteration
        learnings = await engine.learn_from_iteration(
            post_id=run_id,
            content_type=content_type,
            qc_score=7.5,
            meta_feedback={"weaknesses": [...], "suggestions": [...]},
            qc_feedback="Hook is weak. Consider starting with a statistic.",
        )

        # Get learnings for prompt injection
        writer_learnings = engine.get_learnings_for_prompt("writer", content_type)
        prompt_text = engine.format_learnings_for_prompt(writer_learnings)
    """

    def __init__(
        self,
        db: SupabaseDB,
        kb: Optional[KnowledgeBase] = None,
    ) -> None:
        """Initialize the learning engine.

        Args:
            db: Database client for persisting learnings.
            kb: Optional KnowledgeBase for advanced learning integration.
        """
        self.db = db
        self.kb = kb

        # In-memory cache of active learnings
        self.learnings: Dict[str, MicroLearning] = {}
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        """Load learnings from database if not already loaded."""
        if self._loaded:
            return

        try:
            stored = await self.db.get_all_micro_learnings(limit=500)
            for record in stored:
                learning = MicroLearning.from_dict(record)
                if learning.is_active:
                    self.learnings[learning.id] = learning
            logger.info(
                "[LEARN] Loaded %d active learnings from database",
                len(self.learnings),
            )
        except Exception as exc:
            logger.warning("[LEARN] Failed to load learnings from database: %s", exc)

        self._loaded = True

    async def handle_first_post(self) -> List[MicroLearning]:
        """Bootstrap learnings for the first post.

        Creates initial high-confidence rules based on known best practices.
        These will be confirmed or contradicted as the system gains experience.

        Returns:
            List of bootstrap MicroLearnings created.
        """
        await self._ensure_loaded()

        created: List[MicroLearning] = []

        for bootstrap in BOOTSTRAP_LEARNINGS:
            learning_id = f"bootstrap_{generate_id()[:8]}"
            learning = MicroLearning(
                id=learning_id,
                learning_type=bootstrap["learning_type"],
                source=LearningSource.EXPLICIT_RULE,
                description=bootstrap["description"],
                rule=bootstrap["rule"],
                affected_component=bootstrap["affected_component"],
                confidence=bootstrap["confidence"],
                is_bootstrap=True,
                is_active=True,
            )
            created.append(learning)
            self.learnings[learning_id] = learning

        # Persist to database
        if created:
            try:
                await self.db.save_micro_learnings([l.to_dict() for l in created])
                logger.info(
                    "[LEARN] Bootstrapped %d learnings for first post",
                    len(created),
                )
            except Exception as exc:
                logger.warning("[LEARN] Failed to persist bootstrap learnings: %s", exc)

        return created

    async def learn_from_iteration(
        self,
        post_id: str,
        content_type: ContentType,
        qc_score: float,
        meta_feedback: Dict[str, Any],
        qc_feedback: str,
    ) -> IterationLearnings:
        """Extract learnings from a single iteration.

        Called AFTER every evaluation, BEFORE proceeding to next step.
        This is the core of continuous learning.

        Args:
            post_id: ID of the post being evaluated.
            content_type: The content type of the post.
            qc_score: QC score from evaluation.
            meta_feedback: Feedback dict from meta-agent evaluation.
            qc_feedback: Text feedback from QC agent.

        Returns:
            IterationLearnings containing new and updated learnings.
        """
        await self._ensure_loaded()

        iteration_id = f"iter_{utc_now().strftime('%Y%m%d_%H%M%S')}_{generate_id()[:6]}"

        logger.info("[LEARN] Starting learning extraction for post %s", post_id)

        new_learnings: List[MicroLearning] = []
        confirmed: List[str] = []
        contradicted: List[str] = []

        # Extract text feedback
        weaknesses = meta_feedback.get("weaknesses", [])
        suggestions = meta_feedback.get("suggestions", [])
        strengths = meta_feedback.get("strengths", [])
        text_feedback = weaknesses + suggestions

        # Process QC feedback
        if qc_feedback:
            text_feedback.append(qc_feedback)

        content_type_str = (
            content_type.value if hasattr(content_type, "value") else str(content_type)
        )

        # Extract learnings from feedback
        for feedback in text_feedback:
            learning = self._extract_learning_from_feedback(
                feedback=feedback,
                component="writer",
                content_type=content_type_str,
                source=LearningSource.QC_FEEDBACK,
            )
            if learning:
                existing = self._find_similar_learning(learning)
                if existing:
                    if self._confirms(learning, existing):
                        if existing.confirm():
                            confirmed.append(existing.id)
                            logger.debug("[LEARN] Confirmed: %s", existing.description)
                    else:
                        existing.contradict()
                        contradicted.append(existing.id)
                        logger.debug("[LEARN] Contradicted: %s", existing.description)
                else:
                    new_learnings.append(learning)
                    self.learnings[learning.id] = learning
                    logger.info("[LEARN] NEW: %s", learning.description)

        # Extract positive learnings from strengths
        for strength in strengths:
            learning = self._extract_learning_from_feedback(
                feedback=f"GOOD: {strength}",
                component="writer",
                content_type=content_type_str,
                source=LearningSource.QC_FEEDBACK,
            )
            if learning:
                learning.confidence = 0.6  # Strengths start higher
                existing = self._find_similar_learning(learning)
                if existing:
                    if existing.confirm():
                        confirmed.append(existing.id)
                else:
                    new_learnings.append(learning)
                    self.learnings[learning.id] = learning

        # Persist changes
        try:
            if new_learnings:
                await self.db.save_micro_learnings([l.to_dict() for l in new_learnings])

            if confirmed or contradicted:
                updates = [
                    self.learnings[lid].to_dict()
                    for lid in confirmed + contradicted
                    if lid in self.learnings
                ]
                if updates:
                    await self.db.update_micro_learnings(updates)
        except Exception as exc:
            logger.warning("[LEARN] Failed to persist learnings: %s", exc)

        result = IterationLearnings(
            iteration_id=iteration_id,
            post_id=post_id,
            text_feedback=text_feedback,
            visual_feedback=[],
            structure_feedback=[],
            new_learnings=new_learnings,
            confirmed_learnings=confirmed,
            contradicted_learnings=contradicted,
            prompt_adjustments={},
            config_adjustments={},
        )

        logger.info(
            "[LEARN] Iteration complete: %d new, %d confirmed, %d contradicted",
            len(new_learnings),
            len(confirmed),
            len(contradicted),
        )

        return result

    def _extract_learning_from_feedback(
        self,
        feedback: str,
        component: str,
        content_type: str,
        source: LearningSource,
    ) -> Optional[MicroLearning]:
        """Extract a structured learning from free-text feedback.

        Uses pattern matching to identify actionable feedback and convert
        it into a MicroLearning. This is a simple rule-based approach;
        could be enhanced with LLM extraction for better results.

        Args:
            feedback: Text feedback string.
            component: Which component this affects.
            content_type: Content type this applies to.
            source: Source of the feedback.

        Returns:
            MicroLearning if feedback is actionable, None otherwise.
        """
        feedback_lower = feedback.lower()

        # Skip very short or generic feedback
        if len(feedback) < 20:
            return None

        # Determine learning type based on keywords
        learning_type = LearningType.CONTENT_STRUCTURE  # default

        if any(kw in feedback_lower for kw in ["hook", "opening", "first line", "start"]):
            learning_type = LearningType.HOOK_PATTERN
        elif any(kw in feedback_lower for kw in ["visual", "image", "photo", "diagram"]):
            learning_type = LearningType.VISUAL_STYLE
        elif any(kw in feedback_lower for kw in ["tone", "voice", "authentic", "human"]):
            learning_type = LearningType.TONE_ADJUSTMENT
        elif any(kw in feedback_lower for kw in ["time", "schedule", "post when"]):
            learning_type = LearningType.TIMING_INSIGHT

        # Generate a rule identifier from the feedback
        # Simple approach: use first significant words
        words = [w for w in feedback_lower.split() if len(w) > 3][:5]
        rule = "_".join(words) if words else "generic_feedback"

        return MicroLearning(
            id=f"learn_{generate_id()[:8]}",
            learning_type=learning_type,
            source=source,
            description=feedback[:200],  # Truncate long feedback
            rule=rule,
            affected_component=component,
            confidence=0.4,  # Start with moderate confidence
            content_type=content_type,
        )

    def _find_similar_learning(self, new: MicroLearning) -> Optional[MicroLearning]:
        """Find existing learning that's similar to new one."""
        for existing in self.learnings.values():
            if not existing.is_active:
                continue
            if (
                existing.learning_type == new.learning_type
                and existing.affected_component == new.affected_component
                and existing.rule == new.rule
            ):
                return existing
        return None

    def _confirms(self, new: MicroLearning, existing: MicroLearning) -> bool:
        """Check if new learning confirms or contradicts existing.

        For now, same rule = confirmation. Could use semantic similarity.
        """
        return new.rule == existing.rule

    def get_learnings_for_prompt(
        self,
        component: str,
        content_type: Optional[ContentType] = None,
    ) -> List[MicroLearning]:
        """Get relevant learnings to inject into a component's prompt.

        Called at the START of each generation to provide context.

        Args:
            component: Component name (writer, visual_creator, humanizer).
            content_type: Optional content type to filter by.

        Returns:
            List of relevant MicroLearnings, sorted by confidence.
        """
        content_type_str = (
            content_type.value
            if content_type and hasattr(content_type, "value")
            else str(content_type) if content_type else None
        )

        relevant: List[MicroLearning] = []

        for learning in self.learnings.values():
            if not learning.is_active:
                continue
            if learning.affected_component != component:
                continue
            if learning.confidence < 0.5:
                continue
            # Check content type compatibility
            if content_type_str and learning.content_type:
                if learning.content_type != content_type_str:
                    continue
            relevant.append(learning)

        # Sort by confidence (highest first), limit to top 10
        return sorted(relevant, key=lambda l: l.confidence, reverse=True)[:10]

    def format_learnings_for_prompt(
        self,
        learnings: List[MicroLearning],
    ) -> str:
        """Format learnings as prompt injection text.

        Args:
            learnings: List of MicroLearnings to format.

        Returns:
            Formatted string to inject into prompt, or empty string if none.
        """
        if not learnings:
            return ""

        lines = ["\n=== LEARNED FROM PREVIOUS ITERATIONS ===\n"]

        for learning in learnings:
            confidence_emoji = "[RULE]" if learning.is_promoted_to_rule else "[LEARNED]"
            lines.append(f"{confidence_emoji} {learning.description}")
            lines.append(f"   Confidence: {learning.confidence:.0%}")
            lines.append("")

        lines.append("========================================\n")

        return "\n".join(lines)


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "ContinuousLearningEngine",
    "BOOTSTRAP_LEARNINGS",
]
