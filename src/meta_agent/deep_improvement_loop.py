"""
Deep improvement loop for the LinkedIn Super Agent meta-agent subsystem.

Implements the full self-improvement cycle:
    Create -> Critique -> Reflect -> Research -> Modify -> Validate

Uses ``SingleCallEvaluator`` for structured critique (replacing the legacy
multi-turn CriticAgent dialogue) and drives the reflection, research, and
code/prompt evolution stages.

Architecture reference:
    - ``architecture.md`` lines 19656-19775  (DeepImprovementLoop)
    - ``architecture.md`` lines 18550-18700  (ReflectionEngine)
    - ``architecture.md`` lines 19150-19650  (CodeEvolutionEngine, KnowledgeBase)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.meta_agent.models import (
    ImprovementResult,
    DialogueSummary,
    Learning,
    Reflection,
)
from src.meta_agent.single_call_evaluator import SingleCallEvaluator
from src.models import ContentType
from src.utils import generate_id, utc_now

if TYPE_CHECKING:
    from src.database import SupabaseDB

logger = logging.getLogger(__name__)


class DeepImprovementLoop:
    """
    Full improvement loop: Create -> Critique -> Reflect -> Research -> Modify -> Validate.

    Orchestrates one complete self-improvement iteration. Given a draft post
    and pipeline context, the loop:

    1. Evaluates the draft via ``SingleCallEvaluator`` (structured critique).
    2. Converts evaluation output into a ``DialogueSummary``.
    3. Passes the summary to the ``ReflectionEngine`` for self-reflection.
    4. If knowledge gaps are identified, executes research via ``ResearchAgent``.
    5. Stores any new learnings in the knowledge base.
    6. If code or prompt changes are recommended, generates them via
       ``CodeEvolutionEngine`` and validates before committing.
    7. Returns a complete ``ImprovementResult`` with all artifacts.

    Args:
        critic: ``SingleCallEvaluator`` for structured post evaluation.
        reflector: ``ReflectionEngine`` for self-reflection on critique.
        researcher: ``ResearchAgent`` for filling knowledge gaps.
        code_evolver: ``CodeEvolutionEngine`` for generating code/prompt changes.
        knowledge_base: ``KnowledgeBase`` for storing and querying learnings.
        db: ``SupabaseDB`` database client for persistence.

    Usage::

        loop = DeepImprovementLoop(
            critic=evaluator,
            reflector=reflection_engine,
            researcher=research_agent,
            code_evolver=code_evolution_engine,
            knowledge_base=kb,
            db=database,
        )
        result = await loop.run(
            draft="My LinkedIn post draft...",
            context={"content_type": ContentType.ENTERPRISE_CASE},
        )
        if result.success:
            print(f"Improvements applied: {len(result.modifications)}")
    """

    def __init__(
        self,
        critic: SingleCallEvaluator,
        reflector: Any,  # ReflectionEngine (not yet created as a module)
        researcher: Any,  # ResearchAgent (not yet created as a module)
        code_evolver: Any,  # CodeEvolutionEngine (not yet created as a module)
        knowledge_base: Any,  # KnowledgeBase (not yet created as a module)
        db: SupabaseDB,
    ) -> None:
        self.critic = critic
        self.reflector = reflector
        self.researcher = researcher
        self.code_evolver = code_evolver
        self.kb = knowledge_base
        self.db = db

    async def run(self, draft: str, context: dict) -> ImprovementResult:
        """
        Run the full deep improvement loop.

        Steps:
            1. Evaluate draft via ``SingleCallEvaluator``.
            2. Convert evaluation to ``DialogueSummary``.
            3. Reflect on critique using ``ReflectionEngine``.
            4. Research knowledge gaps (if any) via ``ResearchAgent``.
            5. Store new learnings in the knowledge base.
            6. Generate code/prompt modifications (if recommended).
            7. Validate and return ``ImprovementResult``.

        Args:
            draft: The post text to evaluate and improve.
            context: Pipeline context dict. Expected keys:
                - ``content_type`` (``ContentType``): The type of content.
                - Additional context as needed by downstream components.

        Returns:
            An ``ImprovementResult`` containing the original draft, critique
            summary, reflection, knowledge gained, and any modifications.
        """
        logger.info("[DEEP_LOOP] Starting deep improvement loop")

        # --- Step 1: Evaluate draft ----------------------------------------
        content_type = context.get("content_type", ContentType.ENTERPRISE_CASE)
        evaluation = await self.critic.evaluate(
            post_content=draft,
            content_type=content_type,
        )
        logger.info(
            "[DEEP_LOOP] Evaluation complete: score=%.2f, passes=%s",
            evaluation.weighted_total,
            evaluation.passes_threshold,
        )

        # --- Step 2: Convert to DialogueSummary ----------------------------
        dialogue_summary = DialogueSummary(
            weaknesses=evaluation.weaknesses,
            suggestions=evaluation.specific_suggestions,
            knowledge_gaps=evaluation.knowledge_gaps,
            research_queries=evaluation.knowledge_gaps,  # gaps become queries
            confidence_in_suggestions=min(evaluation.weighted_total / 10.0, 1.0),
        )
        logger.info(
            "[DEEP_LOOP] DialogueSummary created: %d weaknesses, %d suggestions",
            len(dialogue_summary.weaknesses),
            len(dialogue_summary.suggestions),
        )

        # --- Step 3: Reflect on critique -----------------------------------
        historical_posts = await self.db.get_recent_posts(limit=10)
        historical_content: List[str] = []
        for post in historical_posts:
            if isinstance(post, dict):
                historical_content.append(post.get("content", ""))
            else:
                historical_content.append(getattr(post, "content", ""))

        reflection: Reflection = await self.reflector.reflect(
            original_work=draft,
            critique=dialogue_summary,
            historical_work=historical_content,
        )
        logger.info(
            "[DEEP_LOOP] Reflection complete: critique_valid=%s, "
            "research_needed=%d, code_changes=%d, prompt_changes=%d",
            reflection.critique_valid,
            len(reflection.research_needed),
            len(reflection.code_changes),
            len(reflection.prompt_changes),
        )

        # --- Step 4: Research knowledge gaps (if any) ----------------------
        knowledge: Dict[str, Any] = {}
        if reflection.research_needed:
            logger.info(
                "[DEEP_LOOP] Researching %d knowledge gaps",
                len(reflection.research_needed),
            )
            for query in reflection.research_needed:
                try:
                    results = await self.researcher.execute_query(query)
                    # Use query source or purpose as the topic key
                    topic_key = getattr(query, "purpose", None) or getattr(
                        query, "query", str(query)
                    )
                    knowledge[topic_key] = results
                except Exception as exc:
                    logger.error(
                        "[DEEP_LOOP] Research query failed: %s", exc
                    )
                    # Fail-fast philosophy: log but continue with remaining queries
                    continue

            # Synthesize raw research into structured knowledge
            if knowledge:
                knowledge = await self._synthesize_knowledge(knowledge)
                logger.info(
                    "[DEEP_LOOP] Synthesized %d knowledge topics", len(knowledge)
                )

        # --- Step 5: Store learnings ---------------------------------------
        for topic, content in knowledge.items():
            now = utc_now()
            learning = Learning(
                id=generate_id(),
                topic=topic,
                content=json.dumps(content) if not isinstance(content, str) else content,
                source="critique_research",
                confidence=reflection.confidence_in_changes,
                learned_at=now,
            )
            try:
                await self.kb.store_learning(learning)
                logger.info("[DEEP_LOOP] Stored learning: topic=%s", topic)
            except Exception as exc:
                logger.error(
                    "[DEEP_LOOP] Failed to store learning '%s': %s", topic, exc
                )

        # --- Step 6: Generate code/prompt modifications --------------------
        modifications: list = []

        if reflection.code_changes:
            logger.info(
                "[DEEP_LOOP] Generating code module for: %s",
                reflection.code_changes[0],
            )
            try:
                module = await self.code_evolver.generate_module(
                    purpose=reflection.code_changes[0],
                    knowledge=knowledge,
                    reflection=reflection,
                )
                # Validate the generated module
                test_passed = await self._test_module(module)
                if test_passed:
                    modifications.append(("code", module))
                    logger.info("[DEEP_LOOP] Code module validated and committed")
                else:
                    logger.warning(
                        "[DEEP_LOOP] Code module failed validation, skipping"
                    )
            except Exception as exc:
                logger.error(
                    "[DEEP_LOOP] Code module generation failed: %s", exc
                )

        if reflection.prompt_changes:
            logger.info(
                "[DEEP_LOOP] Evolving prompt for: %s",
                reflection.prompt_changes[0],
            )
            try:
                evolution = await self.code_evolver.evolve_prompt(
                    current_prompt_path="prompts/writer_system.txt",
                    reflection=reflection,
                    knowledge=knowledge,
                )
                # Validate the evolved prompt
                test_passed = await self._test_prompt(evolution)
                if test_passed:
                    modifications.append(("prompt", evolution))
                    logger.info("[DEEP_LOOP] Prompt evolution validated and committed")
                else:
                    logger.warning(
                        "[DEEP_LOOP] Prompt evolution failed validation, skipping"
                    )
            except Exception as exc:
                logger.error(
                    "[DEEP_LOOP] Prompt evolution failed: %s", exc
                )

        # --- Step 7: Return result -----------------------------------------
        success = True  # Loop itself completed; modifications may or may not exist
        result = ImprovementResult(
            original_draft=draft,
            critique_summary=dialogue_summary,
            reflection=reflection,
            knowledge_gained=knowledge,
            modifications=modifications,
            success=success,
        )
        logger.info(
            "[DEEP_LOOP] Improvement loop complete: %d modifications, success=%s",
            len(modifications),
            success,
        )
        return result

    async def _synthesize_knowledge(self, raw_knowledge: dict) -> dict:
        """
        Synthesize raw research results into structured, actionable knowledge.

        Processes each topic's raw research output into a normalized dict
        containing ``summary``, ``actionable_insights``, and ``confidence``.

        Args:
            raw_knowledge: Mapping of topic -> raw research results.

        Returns:
            Mapping of topic -> synthesized knowledge dict with keys:
            ``summary``, ``actionable_insights``, ``confidence``.
        """
        synthesized: Dict[str, Any] = {}

        for topic, raw_data in raw_knowledge.items():
            if isinstance(raw_data, dict):
                # Already structured -- extract key fields
                synthesized[topic] = {
                    "summary": raw_data.get("summary", str(raw_data)),
                    "actionable_insights": raw_data.get(
                        "actionable_insights",
                        raw_data.get("findings", []),
                    ),
                    "confidence": raw_data.get("confidence", 0.5),
                }
            elif isinstance(raw_data, list):
                # List of findings -- aggregate
                synthesized[topic] = {
                    "summary": f"Aggregated {len(raw_data)} findings",
                    "actionable_insights": raw_data,
                    "confidence": 0.5,
                }
            else:
                # String or other primitive
                synthesized[topic] = {
                    "summary": str(raw_data),
                    "actionable_insights": [],
                    "confidence": 0.3,
                }

        return synthesized

    async def _test_module(self, module: Any) -> bool:
        """
        Validate a generated Python module passes basic checks.

        Checks:
        1. The module has a ``validated`` attribute set to ``True`` by the
           code evolution engine's internal syntax check.
        2. The module has non-empty ``code`` content.

        More sophisticated validation (import testing, unit test execution)
        should be added as the system matures.

        Args:
            module: A ``GeneratedModule`` (or compatible object) to validate.

        Returns:
            ``True`` if the module passes all validation checks.
        """
        # Check basic validation flag from code evolution engine
        if not getattr(module, "validated", False):
            logger.warning(
                "[DEEP_LOOP] Module '%s' did not pass code evolution validation",
                getattr(module, "name", "unknown"),
            )
            return False

        # Check non-empty code
        code = getattr(module, "code", "")
        if not code or not code.strip():
            logger.warning(
                "[DEEP_LOOP] Module '%s' has empty code",
                getattr(module, "name", "unknown"),
            )
            return False

        logger.info(
            "[DEEP_LOOP] Module '%s' passed validation",
            getattr(module, "name", "unknown"),
        )
        return True

    async def _test_prompt(self, evolution: Any) -> bool:
        """
        Validate an evolved prompt is safe and well-formed.

        Checks:
        1. The evolution has non-empty ``new_prompt`` content.
        2. The new prompt is not too short (minimum 50 characters) to
           avoid degenerate prompt evolution.

        Args:
            evolution: A ``PromptEvolution`` (or compatible object) to validate.

        Returns:
            ``True`` if the prompt evolution passes all validation checks.
        """
        new_prompt = getattr(evolution, "new_prompt", "")
        if not new_prompt or not new_prompt.strip():
            logger.warning("[DEEP_LOOP] Prompt evolution has empty new_prompt")
            return False

        # Sanity: prompt should be reasonably long
        if len(new_prompt.strip()) < 50:
            logger.warning(
                "[DEEP_LOOP] Prompt evolution too short (%d chars), likely degenerate",
                len(new_prompt.strip()),
            )
            return False

        logger.info(
            "[DEEP_LOOP] Prompt evolution for '%s' passed validation",
            getattr(evolution, "original_path", "unknown"),
        )
        return True


# ===========================================================================
# PUBLIC API
# ===========================================================================

__all__ = [
    "DeepImprovementLoop",
]
