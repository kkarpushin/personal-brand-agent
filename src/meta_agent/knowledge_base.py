"""
Knowledge Base for the LinkedIn Super Agent meta-improvement layer.

Provides persistent storage and retrieval for agent learnings using
Supabase as the sole database backend.  The KnowledgeBase is a key
component in the deep improvement loop:

    ReflectionEngine  -->  KnowledgeBase  -->  Pipeline Agents
    (generates insights)   (stores/retrieves)  (apply learnings)

Semantic search is approximated with text matching against the ``topic``
and ``content`` fields.  A full vector-based semantic search can be
added later by enabling pgvector in Supabase and adding an embedding
column, but the current implementation is intentionally simple and
functional without external vector stores (no Pinecone, no Redis).

Architecture references:
    - ``architecture.md`` lines 19641-19652  (Learning dataclass)
    - ``architecture.md`` lines 19656-19700  (DeepImprovementLoop + KnowledgeBase)
    - ``architecture.md`` lines 21445-21446  (Knowledge base persistent storage)
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from src.meta_agent.models import Learning
from src.utils import utc_now

logger = logging.getLogger("KnowledgeBase")


class KnowledgeBase:
    """Persistent storage for agent learnings backed by Supabase.

    All operations go through the :class:`SupabaseDB` client.  The
    ``learnings`` table is the single source of truth for accumulated
    knowledge, with confidence-based retrieval ensuring that only
    reliable learnings influence pipeline behaviour.

    Args:
        db: An initialised :class:`SupabaseDB` instance.

    Usage::

        from src.database import get_db
        db = await get_db()
        kb = KnowledgeBase(db)

        await kb.store_learning(learning)
        relevant = await kb.query_relevant("enterprise case hooks", limit=5)
        rules = await kb.get_applicable_rules("enterprise_case")
    """

    def __init__(self, db: Any) -> None:
        self.db = db

    # -----------------------------------------------------------------
    # STORE
    # -----------------------------------------------------------------

    async def store_learning(self, learning: Learning) -> None:
        """Insert a new learning into the ``learnings`` table.

        Converts the :class:`Learning` dataclass to a dict and persists
        it via the database client's ``save_learnings`` batch method.

        Args:
            learning: The :class:`Learning` to store.

        Raises:
            DatabaseError: If the insert fails (propagated from
                :meth:`SupabaseDB.save_learnings`).
        """
        record = asdict(learning)

        # Ensure datetime fields are ISO-formatted strings for Supabase
        if hasattr(learning.learned_at, "isoformat"):
            record["learned_at"] = learning.learned_at.isoformat()

        await self.db.save_learnings([record])

        logger.info(
            "[KB] Stored learning id=%s topic='%s' confidence=%.2f",
            learning.id,
            learning.topic,
            learning.confidence,
        )

    # -----------------------------------------------------------------
    # QUERY
    # -----------------------------------------------------------------

    async def query_relevant(
        self,
        context: str,
        limit: int = 5,
    ) -> List[Learning]:
        """Query learnings relevant to a given context string.

        Performs a text-matching search against the ``learnings`` table,
        filtering for active learnings whose ``topic`` or ``content``
        fields contain tokens from the *context* string.  Results are
        ordered by confidence descending.

        Note: This is a simplified approximation of semantic search.
        For production use, consider enabling pgvector in Supabase and
        storing embeddings alongside learnings for true cosine-similarity
        retrieval.

        Args:
            context: A context string describing what learnings are needed
                (e.g., ``"enterprise case hooks"``).
            limit: Maximum number of learnings to return (default ``5``).

        Returns:
            List of :class:`Learning` objects, ordered by confidence
            descending, limited to *limit* results.
        """
        # Fetch all active learnings sorted by confidence
        all_learnings = await self.db.get_all_learnings(limit=200)

        if not all_learnings:
            return []

        # Simple text matching: tokenize context and score each learning
        context_lower = context.lower()
        context_tokens = set(context_lower.split())

        scored: List[tuple] = []
        for record in all_learnings:
            topic = str(record.get("topic", "")).lower()
            content = str(record.get("content", "")).lower()
            combined = f"{topic} {content}"

            # Count matching tokens
            match_count = sum(
                1 for token in context_tokens if token in combined
            )

            # Also check if the full context string appears as a substring
            if context_lower in combined:
                match_count += len(context_tokens)

            if match_count > 0:
                scored.append((record, match_count))

        # Sort by match count (descending), then by confidence (descending)
        scored.sort(
            key=lambda x: (
                x[1],
                float(x[0].get("confidence", 0)),
            ),
            reverse=True,
        )

        results: List[Learning] = []
        for record, _score in scored[:limit]:
            results.append(_record_to_learning(record))

        logger.debug(
            "[KB] query_relevant context='%s' matched=%d returned=%d",
            context[:50],
            len(scored),
            len(results),
        )

        return results

    # -----------------------------------------------------------------
    # GET APPLICABLE RULES
    # -----------------------------------------------------------------

    async def get_applicable_rules(
        self, content_type: str
    ) -> List[str]:
        """Get high-confidence learning rules applicable to a content type.

        Filters the ``learnings`` table for entries whose ``topic`` field
        contains the *content_type* string and whose ``confidence`` is
        above ``0.7``, then returns just the ``content`` (rule text) for
        each.

        These rules are intended to be injected into agent prompts to
        guide content generation.

        Args:
            content_type: Content type identifier (e.g.,
                ``"enterprise_case"``, ``"automation_case"``).

        Returns:
            List of rule strings, ordered by confidence descending.
        """
        all_learnings = await self.db.get_all_learnings(limit=200)

        if not all_learnings:
            return []

        content_type_lower = content_type.lower()
        rules: List[str] = []

        for record in all_learnings:
            confidence = float(record.get("confidence", 0))
            if confidence < 0.7:
                continue

            topic = str(record.get("topic", "")).lower()
            if content_type_lower in topic:
                content = str(record.get("content", ""))
                if content:
                    rules.append(content)

        logger.debug(
            "[KB] get_applicable_rules type='%s' found=%d",
            content_type,
            len(rules),
        )

        return rules

    # -----------------------------------------------------------------
    # UPDATE STATS
    # -----------------------------------------------------------------

    async def update_learning_stats(
        self,
        learning_id: str,
        success: bool,
    ) -> None:
        """Update application statistics for a learning.

        Increments ``applied_count`` by 1 and recalculates ``success_rate``
        as a running average.

        Args:
            learning_id: UUID of the learning to update.
            success: Whether the learning was successfully applied.
        """
        # Fetch current state
        all_learnings = await self.db.get_all_learnings(limit=2000)
        target = None
        for record in all_learnings:
            if record.get("id") == learning_id:
                target = record
                break

        if target is None:
            logger.warning(
                "[KB] update_learning_stats: learning %s not found",
                learning_id,
            )
            return

        old_count = int(target.get("applied_count", 0))
        old_rate = target.get("success_rate")

        new_count = old_count + 1

        # Calculate new success rate as running average
        if old_rate is None:
            new_rate = 1.0 if success else 0.0
        else:
            old_rate_f = float(old_rate)
            new_rate = (
                (old_rate_f * old_count + (1.0 if success else 0.0))
                / new_count
            )

        update_record = {
            "id": learning_id,
            "applied_count": new_count,
            "success_rate": round(new_rate, 4),
        }

        await self.db.update_learnings([update_record])

        logger.info(
            "[KB] Updated learning %s: applied_count=%d success_rate=%.2f",
            learning_id,
            new_count,
            new_rate,
        )


# =============================================================================
# HELPERS
# =============================================================================


def _record_to_learning(record: Dict[str, Any]) -> Learning:
    """Convert a database record dict to a :class:`Learning` instance.

    Handles safe type conversion for fields that may arrive as strings
    from the database.

    Args:
        record: Row dict from the ``learnings`` Supabase table.

    Returns:
        A populated :class:`Learning` dataclass.
    """
    from datetime import datetime, timezone

    learned_at = record.get("learned_at")
    if isinstance(learned_at, str):
        try:
            learned_at = datetime.fromisoformat(learned_at)
        except (ValueError, TypeError):
            learned_at = utc_now()
    elif learned_at is None:
        learned_at = utc_now()

    return Learning(
        id=str(record.get("id", "")),
        topic=str(record.get("topic", "")),
        content=str(record.get("content", "")),
        source=str(record.get("source", "unknown")),
        confidence=float(record.get("confidence", 0.0)),
        learned_at=learned_at,
        applied_count=int(record.get("applied_count", 0)),
        success_rate=(
            float(record["success_rate"])
            if record.get("success_rate") is not None
            else None
        ),
    )


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "KnowledgeBase",
]
