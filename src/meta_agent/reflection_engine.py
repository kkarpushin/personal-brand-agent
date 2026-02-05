"""
Reflection Engine for the LinkedIn Super Agent meta-improvement layer.

Provides "aha moment" analysis by deeply reflecting on critique feedback,
detecting recurring patterns across historical work, identifying knowledge
gaps, and proposing concrete process / prompt / code changes.

The ReflectionEngine sits between the Critic (or SingleCallEvaluator) and
the Knowledge Base in the deep improvement loop:

    Critic  -->  ReflectionEngine  -->  KnowledgeBase + ResearchAgent
    (feedback)   (pattern detection)    (persistent storage + gap filling)

Architecture references:
    - ``architecture.md`` lines 18601-18684  (ReflectionEngine + Reflection)
    - ``architecture.md`` lines 19656-19700  (DeepImprovementLoop integration)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from src.tools.claude_client import ClaudeClient, get_claude
from src.meta_agent.models import (
    Reflection,
    DialogueSummary,
    ResearchQuery,
)

logger = logging.getLogger("ReflectionEngine")


class ReflectionEngine:
    """Agent that reflects on critique feedback and detects patterns.

    Uses Claude to perform a structured self-reflection on received
    feedback, comparing it against historical work to identify recurring
    issues, knowledge gaps, and actionable changes.

    The reflection process answers five key questions:

    1. Is the criticism valid?  Why or why not?
    2. Is this a **pattern** in past work, or a one-time issue?
    3. What knowledge am I missing to do this better?
    4. What specific research would help me learn?
    5. How should I change my approach going forward?

    Args:
        claude_client: Optional :class:`ClaudeClient` for LLM calls.
            Defaults to a new client via :func:`get_claude`.

    Architecture reference: ``architecture.md`` lines 18604-18657.

    Usage::

        engine = ReflectionEngine()
        reflection = await engine.reflect(
            original_work="My LinkedIn post text...",
            critique=dialogue_summary,
            historical_work=["post 1...", "post 2...", "post 3..."],
        )
        if reflection.is_recurring_pattern:
            print(f"Recurring issue: {reflection.pattern_description}")
    """

    # -----------------------------------------------------------------
    # SYSTEM PROMPT
    # -----------------------------------------------------------------

    REFLECTION_PROMPT: str = (
        "You are reflecting on feedback you received about your work.\n"
        "Think deeply about:\n\n"
        "1. Is the criticism valid? Why or why not?\n"
        "2. Is this a PATTERN in my work, or a one-time issue?\n"
        "3. What knowledge am I missing to do this better?\n"
        "4. What specific research would help me learn?\n"
        "5. How should I change my approach going forward?\n\n"
        "Be honest with yourself. The goal is genuine improvement."
    )

    def __init__(
        self,
        claude_client: Optional[ClaudeClient] = None,
    ) -> None:
        self.claude = claude_client or get_claude()

    # -----------------------------------------------------------------
    # MAIN REFLECTION METHOD
    # -----------------------------------------------------------------

    async def reflect(
        self,
        original_work: str,
        critique: DialogueSummary,
        historical_work: List[str],
    ) -> Reflection:
        """Perform deep reflection on critique feedback.

        Builds a comprehensive prompt with the original work, critique
        weaknesses and suggestions, and up to the last 10 historical
        works (each truncated to 200 characters for context), then asks
        Claude to produce a structured :class:`Reflection`.

        Args:
            original_work: The full text of the content that was critiqued.
            critique: A :class:`DialogueSummary` containing the weaknesses,
                suggestions, knowledge gaps, and research queries from the
                critique session.
            historical_work: List of past post texts for pattern detection.
                Only the last 10 items are used, each truncated to 200
                characters.

        Returns:
            A :class:`Reflection` dataclass with validation, pattern
            detection, knowledge gaps, and proposed changes.
        """
        # Truncate historical works for prompt efficiency
        history_snippets = [
            w[:200] for w in historical_work[-10:]
        ]

        prompt = (
            "My work:\n"
            f"{original_work}\n\n"
            "Critique I received:\n"
            f"- Weaknesses: {json.dumps(critique.weaknesses, ensure_ascii=False)}\n"
            f"- Suggestions: {json.dumps(critique.suggestions, ensure_ascii=False)}\n\n"
            "My past work (for pattern detection):\n"
            f"{json.dumps(history_snippets, ensure_ascii=False)}\n\n"
            "Reflect on the following questions and return structured JSON:\n"
            "1. Is this criticism valid? (and reasoning)\n"
            "2. Do I see this pattern in my past work? "
            "(is_recurring, description, frequency)\n"
            "3. What knowledge am I missing? (knowledge_gaps list)\n"
            "4. What should I research to improve? "
            "(research queries with source hints and priority)\n"
            "5. What concrete changes should I make?\n"
            "   - process_changes: changes to my content creation process\n"
            "   - prompt_changes: specific prompt rule modifications\n"
            "   - code_changes: new modules or functions needed\n\n"
            "Return JSON with this exact schema:\n"
            "{\n"
            '  "critique_valid": true/false,\n'
            '  "critique_validity_reasoning": "explanation",\n'
            '  "is_recurring_pattern": true/false,\n'
            '  "pattern_description": "description or null",\n'
            '  "pattern_frequency": "e.g. 3 out of last 10 posts or null",\n'
            '  "knowledge_gaps": ["gap1", "gap2"],\n'
            '  "research_needed": [\n'
            '    {"query": "...", "source_hint": "perplexity", "priority": 1}\n'
            "  ],\n"
            '  "process_changes": ["change1"],\n'
            '  "prompt_changes": ["change1"],\n'
            '  "code_changes": ["change1"],\n'
            '  "confidence_in_changes": 0.7\n'
            "}"
        )

        logger.info("[REFLECTION] Starting reflection on critique")

        response_data: Dict[str, Any] = await self.claude.generate_structured(
            prompt=prompt,
            system=self.REFLECTION_PROMPT,
        )

        # Parse research queries from the response
        research_needed_raw = response_data.get("research_needed", [])
        research_queries: List[ResearchQuery] = []
        for rq in research_needed_raw:
            if isinstance(rq, dict):
                research_queries.append(
                    ResearchQuery(
                        source=rq.get("source_hint", "perplexity"),
                        query=rq.get("query", ""),
                        purpose=rq.get("purpose", "Fill knowledge gap"),
                        priority=int(rq.get("priority", 3)),
                    )
                )
            elif isinstance(rq, str):
                research_queries.append(
                    ResearchQuery(
                        source="perplexity",
                        query=rq,
                        purpose="Fill knowledge gap",
                        priority=3,
                    )
                )

        reflection = Reflection(
            critique_valid=bool(response_data.get("critique_valid", True)),
            critique_validity_reasoning=str(
                response_data.get("critique_validity_reasoning", "")
            ),
            is_recurring_pattern=bool(
                response_data.get("is_recurring_pattern", False)
            ),
            pattern_description=response_data.get("pattern_description"),
            pattern_frequency=response_data.get("pattern_frequency"),
            knowledge_gaps=response_data.get("knowledge_gaps", []),
            research_needed=research_queries,
            process_changes=response_data.get("process_changes", []),
            prompt_changes=response_data.get("prompt_changes", []),
            code_changes=response_data.get("code_changes", []),
            confidence_in_changes=float(
                response_data.get("confidence_in_changes", 0.5)
            ),
        )

        logger.info(
            "[REFLECTION] Reflection complete: valid=%s, recurring=%s, "
            "gaps=%d, changes=%d, confidence=%.2f",
            reflection.critique_valid,
            reflection.is_recurring_pattern,
            len(reflection.knowledge_gaps),
            len(reflection.process_changes)
            + len(reflection.prompt_changes)
            + len(reflection.code_changes),
            reflection.confidence_in_changes,
        )

        return reflection


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "ReflectionEngine",
]
