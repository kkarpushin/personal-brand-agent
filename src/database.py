"""
Unified async database client for all system operations.

ALL database operations go through the SupabaseDB class defined here.
No direct Supabase calls should appear anywhere else in the codebase.

Usage::

    from src.database import SupabaseDB, get_db

    # In async context:
    db = await get_db()
    post_id = await db.save_post({"text_content": "...", "content_type": "..."})

Architecture reference: architecture.md lines 308-988, 21260-21286, 23467-23824, 23825-24315
"""

import asyncio
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Union

from supabase import AsyncClient, create_async_client

from src.exceptions import DatabaseError, ValidationError
from src.utils import utc_now

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION HELPERS
# =============================================================================


def validate_not_empty(value: Any, name: str) -> None:
    """Validate that *value* is not ``None`` or an empty string.

    Args:
        value: The value to check.
        name: Human-readable field name used in error messages.

    Raises:
        ValidationError: If *value* is ``None`` or a blank string.
    """
    if value is None:
        raise ValidationError(f"{name} cannot be None")
    if isinstance(value, str) and not value.strip():
        raise ValidationError(f"{name} cannot be empty string")


def validate_positive(value: Union[int, float], name: str) -> None:
    """Validate that *value* is strictly positive (> 0).

    Args:
        value: The numeric value to check.
        name: Human-readable field name used in error messages.

    Raises:
        ValidationError: If *value* is ``None`` or not positive.
    """
    if value is None:
        raise ValidationError(f"{name} cannot be None")
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class SupabaseConfig:
    """Supabase configuration loaded from environment variables.

    Attributes:
        url: The Supabase project URL (``SUPABASE_URL``).
        key: The service-role key for full server-side access
            (``SUPABASE_SERVICE_KEY``).
    """

    url: str
    key: str  # service_role key for full server-side access

    @classmethod
    def from_env(cls) -> "SupabaseConfig":
        """Create a config instance from environment variables.

        Reads ``SUPABASE_URL`` and ``SUPABASE_SERVICE_KEY``.

        Raises:
            ValueError: If either variable is missing or empty.
        """
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_KEY")

        if not url or not key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set"
            )

        return cls(url=url, key=key)


# =============================================================================
# SUPABASE DATABASE CLIENT
# =============================================================================


class SupabaseDB:
    """Unified **async** database client for all system operations.

    ALL database operations go through this class.
    No direct Supabase calls elsewhere in the codebase.

    **Important:** Use the :meth:`create` factory method instead of
    ``__init__`` directly -- the underlying async client requires an
    ``await`` during initialisation.
    """

    _instance: Optional["SupabaseDB"] = None

    def __init__(self, client: AsyncClient) -> None:
        """Private constructor.  Use :meth:`create` factory method."""
        self.client = client

    @classmethod
    async def create(
        cls, config: Optional[SupabaseConfig] = None
    ) -> "SupabaseDB":
        """Factory method to create an async :class:`SupabaseDB` instance.

        Args:
            config: Optional configuration.  When ``None``,
                :meth:`SupabaseConfig.from_env` is used.

        Returns:
            A fully initialised :class:`SupabaseDB` instance.
        """
        config = config or SupabaseConfig.from_env()
        client = await create_async_client(config.url, config.key)
        return cls(client)

    # -----------------------------------------------------------------
    # POSTS
    # -----------------------------------------------------------------

    async def save_post(self, post: Dict[str, Any]) -> str:
        """Save a published post.

        Args:
            post: Post data dict.  Must contain ``text_content`` (non-empty)
                and ``content_type``.

        Returns:
            The UUID of the inserted post.

        Raises:
            ValidationError: On missing / invalid fields.
            DatabaseError: When the insert returns no data.
        """
        if not post:
            raise ValidationError("post cannot be None or empty")
        if "text_content" not in post or not post["text_content"]:
            raise ValidationError("post must have text_content")
        if "content_type" not in post:
            raise ValidationError("post must have content_type")

        result = await self.client.table("posts").insert(post).execute()
        if not result.data:
            raise DatabaseError("Insert succeeded but returned no data")
        return result.data[0]["id"]

    async def upsert_imported_posts(
        self, posts: List[Dict[str, Any]]
    ) -> int:
        """Upsert imported posts into the ``posts`` table.

        Each post dict is expected to contain at least ``linkedin_post_id``
        and ``text_content``.  Existing rows with the same
        ``linkedin_post_id`` are updated; new rows are inserted.

        Args:
            posts: List of post dicts to upsert.

        Returns:
            Number of rows upserted.
        """
        if not posts:
            return 0

        result = await (
            self.client.table("posts")
            .upsert(posts, on_conflict="linkedin_post_id")
            .execute()
        )
        return len(result.data) if result.data else 0

    async def get_post(self, post_id: str) -> Optional[Dict[str, Any]]:
        """Get a post by ID.

        Args:
            post_id: UUID of the post.

        Returns:
            Post dict or ``None`` if not found.
        """
        validate_not_empty(post_id, "post_id")

        result = (
            await self.client.table("posts")
            .select("*")
            .eq("id", post_id)
            .execute()
        )
        return result.data[0] if result.data else None

    async def get_recent_posts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent posts ordered by creation date descending.

        Args:
            limit: Maximum number of posts to return (must be > 0).

        Returns:
            List of post dicts.
        """
        validate_positive(limit, "limit")

        result = await (
            self.client.table("posts")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data

    async def get_total_post_count(self) -> int:
        """Get total number of posts.

        Returns:
            Integer count of all posts.
        """
        result = await (
            self.client.table("posts")
            .select("id", count="exact")
            .execute()
        )
        return result.count or 0

    async def get_posts_by_content_type(
        self, content_type: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get posts filtered by content type.

        Args:
            content_type: The content type to filter by (e.g.
                ``"enterprise_case"``).
            limit: Maximum results to return.

        Returns:
            List of post dicts ordered by ``created_at`` descending.
        """
        validate_not_empty(content_type, "content_type")
        validate_positive(limit, "limit")

        result = await (
            self.client.table("posts")
            .select("*")
            .eq("content_type", content_type)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data

    # -----------------------------------------------------------------
    # POST METRICS (Analytics)
    # -----------------------------------------------------------------

    async def store_metrics_snapshot(
        self, snapshot: Dict[str, Any]
    ) -> str:
        """Store a metrics snapshot for a post.

        Args:
            snapshot: Metrics data dict.  Must contain ``post_id``.

        Returns:
            UUID of the inserted snapshot row.

        Raises:
            ValidationError: On missing / invalid fields.
            DatabaseError: When the insert returns no data.
        """
        if not snapshot:
            raise ValidationError("snapshot cannot be None or empty")
        if "post_id" not in snapshot:
            raise ValidationError("snapshot must have 'post_id'")

        result = await (
            self.client.table("post_metrics").insert(snapshot).execute()
        )
        if not result.data:
            raise DatabaseError("Insert succeeded but returned no data")
        return result.data[0]["id"]

    async def get_metrics_history(
        self, post_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get metrics history for a post.

        Args:
            post_id: UUID of the post.
            limit: Maximum snapshots to return.

        Returns:
            List of metric snapshot dicts ordered by ``collected_at``
            descending.
        """
        validate_not_empty(post_id, "post_id")
        validate_positive(limit, "limit")

        result = await (
            self.client.table("post_metrics")
            .select("*")
            .eq("post_id", post_id)
            .order("collected_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data

    async def get_average_metrics_at(
        self, minutes_after_post: int
    ) -> Dict[str, float]:
        """Get average metrics at a specific number of minutes after posting.

        Uses the Supabase RPC function ``get_average_metrics_at_minutes``.

        Args:
            minutes_after_post: The time checkpoint in minutes.

        Returns:
            Dict with keys ``avg_likes``, ``avg_comments``, etc., or
            empty dict if no data.
        """
        validate_positive(minutes_after_post, "minutes_after_post")

        result = await self.client.rpc(
            "get_average_metrics_at_minutes",
            {"minutes": minutes_after_post},
        ).execute()
        return result.data[0] if result.data else {}

    async def get_average_score(self) -> float:
        """Get average QC score across all posts.

        Uses the Supabase RPC function ``get_average_qc_score``.

        Returns:
            Average score as a float.  Defaults to ``7.0`` when there
            are no scored posts.
        """
        result = await self.client.rpc("get_average_qc_score").execute()
        return result.data[0]["avg_score"] if result.data else 7.0

    async def get_percentile(
        self, likes: int, minutes_after_post: int
    ) -> float:
        """Calculate percentile rank for a post's likes at a given checkpoint.

        Uses the Supabase RPC function ``get_likes_percentile``.

        Args:
            likes: Number of likes the post received.
            minutes_after_post: Minutes after publication.

        Returns:
            Percentile as a float (e.g. ``90.0`` means top 10 %).
            Defaults to ``50.0`` when there is no data.
        """
        result = await self.client.rpc(
            "get_likes_percentile",
            {"likes_count": likes, "minutes": minutes_after_post},
        ).execute()
        return result.data[0]["percentile"] if result.data else 50.0

    # -----------------------------------------------------------------
    # LEARNINGS (Continuous Learning Engine)
    # -----------------------------------------------------------------

    async def save_learnings(
        self, learnings: List[Dict[str, Any]]
    ) -> None:
        """Save new learnings.

        Args:
            learnings: List of learning dicts to insert.  If the list is
                empty or ``None``, the call is a no-op.
        """
        if learnings:
            await self.client.table("learnings").insert(learnings).execute()

    async def get_all_learnings(
        self, limit: int = 1000, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get active learnings with pagination.

        Args:
            limit: Maximum number of learnings to return.
            offset: Number of rows to skip (for pagination).

        Returns:
            List of learning dicts ordered by ``confidence`` descending.
        """
        validate_positive(limit, "limit")

        result = await (
            self.client.table("learnings")
            .select("*")
            .eq("is_active", True)
            .order("confidence", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return result.data

    async def update_learnings(
        self, learnings: List[Dict[str, Any]]
    ) -> None:
        """Update existing learnings (confirmations, contradictions).

        Uses ``upsert`` with ``on_conflict="id"`` so existing rows are
        updated and new rows are inserted in a single query.

        Args:
            learnings: List of learning dicts (each must include ``id``).
        """
        if not learnings:
            return

        await (
            self.client.table("learnings")
            .upsert(learnings, on_conflict="id")
            .execute()
        )

    async def get_learnings_for_component(
        self,
        component: str,
        content_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get learnings for a specific component.

        Args:
            component: The affected component name (e.g. ``"writer"``).
            content_type: Optional content type filter.  When provided,
                results include learnings for that type **or** learnings
                with a ``NULL`` content type (i.e. universal learnings).

        Returns:
            Up to 10 active learnings with confidence >= 0.5, ordered by
            confidence descending.
        """
        validate_not_empty(component, "component")

        query = (
            self.client.table("learnings")
            .select("*")
            .eq("affected_component", component)
            .eq("is_active", True)
            .gte("confidence", 0.5)
        )

        if content_type:
            query = query.or_(
                f"content_type.eq.{content_type},content_type.is.null"
            )

        result = await query.order("confidence", desc=True).limit(10).execute()
        return result.data

    # -----------------------------------------------------------------
    # MODIFICATIONS (Self-Modifying Code)
    # -----------------------------------------------------------------

    async def save_modification(self, modification: Dict[str, Any]) -> str:
        """Save a code modification record.

        Args:
            modification: Modification data dict.  Required fields:
                ``gap_type``, ``gap_description``, ``module_name``,
                ``file_path``, ``code_content``.

        Returns:
            UUID of the inserted row.

        Raises:
            ValidationError: On missing / invalid fields.
            DatabaseError: When the insert returns no data.
        """
        if not modification:
            raise ValidationError("modification cannot be None or empty")

        required_fields: Set[str] = {
            "gap_type",
            "gap_description",
            "module_name",
            "file_path",
            "code_content",
        }
        missing = required_fields - set(modification.keys())
        if missing:
            raise ValidationError(
                f"modification missing required fields: {missing}"
            )

        result = await (
            self.client.table("code_modifications")
            .insert(modification)
            .execute()
        )
        if not result.data:
            raise DatabaseError("Insert succeeded but returned no data")
        return result.data[0]["id"]

    async def get_modification(
        self, modification_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get modification by ID.

        Args:
            modification_id: UUID of the modification.

        Returns:
            Modification dict or ``None`` if not found.
        """
        validate_not_empty(modification_id, "modification_id")

        result = await (
            self.client.table("code_modifications")
            .select("*")
            .eq("id", modification_id)
            .execute()
        )
        return result.data[0] if result.data else None

    async def update_modification(
        self, modification: Dict[str, Any]
    ) -> None:
        """Update modification status.

        Args:
            modification: Dict containing at least ``id`` and the fields
                to update.

        Raises:
            ValidationError: If *modification* is empty or lacks ``id``.
        """
        if not modification:
            raise ValidationError("modification cannot be None or empty")
        if "id" not in modification:
            raise ValidationError("modification must have 'id' for update")

        await (
            self.client.table("code_modifications")
            .update(modification)
            .eq("id", modification["id"])
            .execute()
        )

    async def get_modifications(
        self, days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get recent modifications.

        Args:
            days: How many days back to look (default 30).

        Returns:
            List of modification dicts ordered by ``created_at``
            descending.
        """
        validate_positive(days, "days")

        from_date = (utc_now() - timedelta(days=days)).isoformat()
        result = await (
            self.client.table("code_modifications")
            .select("*")
            .gte("created_at", from_date)
            .order("created_at", desc=True)
            .execute()
        )
        return result.data

    # -----------------------------------------------------------------
    # EXPERIMENTS (A/B Testing)
    # -----------------------------------------------------------------

    async def save_experiment(self, experiment: Dict[str, Any]) -> str:
        """Save an experiment.

        Args:
            experiment: Experiment data dict.  Required fields:
                ``name``, ``hypothesis``, ``variable``,
                ``control_value``, ``treatment_value``.

        Returns:
            UUID of the inserted experiment.

        Raises:
            ValidationError: On missing / invalid fields.
            DatabaseError: When the insert returns no data.
        """
        if not experiment:
            raise ValidationError("experiment cannot be None or empty")

        required_fields: Set[str] = {
            "name",
            "hypothesis",
            "variable",
            "control_value",
            "treatment_value",
        }
        missing = required_fields - set(experiment.keys())
        if missing:
            raise ValidationError(
                f"experiment missing required fields: {missing}"
            )

        result = await (
            self.client.table("experiments").insert(experiment).execute()
        )
        if not result.data:
            raise DatabaseError("Insert succeeded but returned no data")
        return result.data[0]["id"]

    async def get_experiment(
        self, experiment_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get experiment by ID.

        Args:
            experiment_id: UUID of the experiment.

        Returns:
            Experiment dict or ``None`` if not found.
        """
        validate_not_empty(experiment_id, "experiment_id")

        result = await (
            self.client.table("experiments")
            .select("*")
            .eq("id", experiment_id)
            .execute()
        )
        return result.data[0] if result.data else None

    async def update_experiment(
        self, experiment: Dict[str, Any]
    ) -> None:
        """Update an experiment.

        Args:
            experiment: Dict containing at least ``id`` and the fields
                to update.

        Raises:
            ValidationError: If *experiment* is empty or lacks ``id``.
        """
        if not experiment:
            raise ValidationError("experiment cannot be None or empty")
        if "id" not in experiment:
            raise ValidationError("experiment must have 'id' for update")

        await (
            self.client.table("experiments")
            .update(experiment)
            .eq("id", experiment["id"])
            .execute()
        )

    async def get_active_experiments(self) -> List[Dict[str, Any]]:
        """Get all active experiments.

        Returns:
            List of experiment dicts with ``status == 'active'``.
        """
        result = await (
            self.client.table("experiments")
            .select("*")
            .eq("status", "active")
            .execute()
        )
        return result.data

    # -----------------------------------------------------------------
    # RESEARCH (Research Agent)
    # -----------------------------------------------------------------

    async def save_research_report(
        self, report: Dict[str, Any]
    ) -> str:
        """Save a research report.

        Args:
            report: Report data dict.  Required fields:
                ``trigger_type``, ``findings``, ``recommendations``.

        Returns:
            UUID of the inserted report.

        Raises:
            ValidationError: On missing / invalid fields.
            DatabaseError: When the insert returns no data.
        """
        if not report:
            raise ValidationError("report cannot be None or empty")

        required_fields: Set[str] = {
            "trigger_type",
            "findings",
            "recommendations",
        }
        missing = required_fields - set(report.keys())
        if missing:
            raise ValidationError(
                f"report missing required fields: {missing}"
            )

        result = await (
            self.client.table("research_reports").insert(report).execute()
        )
        if not result.data:
            raise DatabaseError("Insert succeeded but returned no data")
        return result.data[0]["id"]

    async def get_last_research_date(self) -> Optional[datetime]:
        """Get date of last research report.

        Returns:
            Timezone-aware ``datetime`` of the most recent report, or
            ``None`` if no reports exist.
        """
        result = await (
            self.client.table("research_reports")
            .select("created_at")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if result.data:
            return datetime.fromisoformat(result.data[0]["created_at"])
        return None

    async def get_past_experiments(
        self, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get past experiment results for research context.

        Args:
            limit: Maximum number of completed experiments to return.

        Returns:
            List of completed experiment dicts ordered by
            ``completed_at`` descending.
        """
        validate_positive(limit, "limit")

        result = await (
            self.client.table("experiments")
            .select("*")
            .eq("status", "completed")
            .order("completed_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data

    # -----------------------------------------------------------------
    # PROMPTS (Prompt Management)
    # -----------------------------------------------------------------

    async def get_prompt(self, component: str) -> Optional[str]:
        """Get current active prompt for a component.

        Args:
            component: Component name (e.g. ``"writer"``, ``"qc"``).

        Returns:
            The prompt content string, or ``None`` if no active prompt
            exists for the component.
        """
        validate_not_empty(component, "component")

        result = await (
            self.client.table("prompts")
            .select("content")
            .eq("component", component)
            .eq("is_active", True)
            .order("version", desc=True)
            .limit(1)
            .execute()
        )
        return result.data[0]["content"] if result.data else None

    async def save_prompt(
        self, component: str, content: str, reason: str
    ) -> str:
        """Save a new prompt version atomically.

        Uses the Supabase RPC function ``save_prompt_atomic`` to
        atomically:

        1. Determine the next version number.
        2. Deactivate all previous prompts for the component.
        3. Insert the new prompt as the active version.

        This prevents race conditions when multiple processes try to
        update prompts concurrently.

        Args:
            component: Component name.
            content: The full prompt text.
            reason: Why this prompt was changed.

        Returns:
            UUID of the newly created prompt row.
        """
        validate_not_empty(component, "component")
        validate_not_empty(content, "content")
        validate_not_empty(reason, "reason")

        result = await self.client.rpc(
            "save_prompt_atomic",
            {
                "p_component": component,
                "p_content": content,
                "p_reason": reason,
            },
        ).execute()

        return result.data[0]["id"]

    # -----------------------------------------------------------------
    # TREND TOPICS (Scout Cache)
    # -----------------------------------------------------------------

    async def cache_topics(
        self, topics: List[Dict[str, Any]], source: str
    ) -> None:
        """Cache discovered topics.

        Stamps each topic with ``source`` and ``cached_at`` before
        inserting the batch.

        Args:
            topics: List of topic dicts.
            source: Origin of the topics (e.g. ``"hackernews"``,
                ``"twitter"``).
        """
        validate_not_empty(source, "source")

        if not topics:
            return

        for topic in topics:
            topic["source"] = source
            topic["cached_at"] = utc_now().isoformat()

        await self.client.table("topic_cache").insert(topics).execute()

    async def get_cached_topics(
        self, hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get recently cached topics.

        Args:
            hours: Look-back window in hours (default 24).

        Returns:
            List of topic dicts ordered by ``score`` descending.
        """
        validate_positive(hours, "hours")

        from_time = (utc_now() - timedelta(hours=hours)).isoformat()
        result = await (
            self.client.table("topic_cache")
            .select("*")
            .gte("cached_at", from_time)
            .order("score", desc=True)
            .execute()
        )
        return result.data

    # -----------------------------------------------------------------
    # PHOTOS (Author photo library)
    # -----------------------------------------------------------------

    async def get_all_photos(self) -> List[Dict[str, Any]]:
        """Get all non-disabled photos from the library.

        Returns photos ordered by ``times_used`` ascending so that
        less-used photos appear first.

        Returns:
            List of photo dicts.
        """
        result = await (
            self.client.table("author_photos")
            .select("*")
            .eq("disabled", False)
            .order("times_used", desc=False)
            .execute()
        )
        return result.data

    async def update_photo_usage(
        self, photo_id: str, post_id: str
    ) -> None:
        """Update photo usage statistics.

        Uses the Supabase RPC function ``increment_photo_usage`` to
        atomically increment the usage counter and record the last
        usage.

        Args:
            photo_id: UUID of the photo.
            post_id: UUID of the post the photo was used in.
        """
        validate_not_empty(photo_id, "photo_id")
        validate_not_empty(post_id, "post_id")

        await self.client.rpc(
            "increment_photo_usage",
            {"p_photo_id": photo_id, "p_post_id": post_id},
        ).execute()

    async def save_photo_metadata(
        self, metadata: Dict[str, Any]
    ) -> str:
        """Save new photo metadata.

        Args:
            metadata: Photo metadata dict.  Required fields:
                ``file_name``, ``file_path``.

        Returns:
            UUID of the inserted photo row.

        Raises:
            ValidationError: On missing / invalid fields.
            DatabaseError: When the insert returns no data.
        """
        if not metadata:
            raise ValidationError("metadata cannot be None or empty")

        required_fields: Set[str] = {"file_name", "file_path"}
        missing = required_fields - set(metadata.keys())
        if missing:
            raise ValidationError(
                f"metadata missing required fields: {missing}"
            )

        result = await (
            self.client.table("author_photos").insert(metadata).execute()
        )
        if not result.data:
            raise DatabaseError("Insert succeeded but returned no data")
        return result.data[0]["id"]

    # -----------------------------------------------------------------
    # DRAFTS (Work in Progress)
    # -----------------------------------------------------------------

    async def save_draft(self, draft: Dict[str, Any]) -> str:
        """Save a draft post.

        Args:
            draft: Draft data dict.  Required fields: ``hook``,
                ``body``, ``full_text``, ``content_type``.

        Returns:
            UUID of the inserted draft row.

        Raises:
            ValidationError: On missing / invalid fields.
            DatabaseError: When the insert returns no data.
        """
        if not draft:
            raise ValidationError("draft cannot be None or empty")

        required_fields: Set[str] = {
            "hook",
            "body",
            "full_text",
            "content_type",
        }
        missing = required_fields - set(draft.keys())
        if missing:
            raise ValidationError(
                f"draft missing required fields: {missing}"
            )

        result = await (
            self.client.table("drafts").insert(draft).execute()
        )
        if not result.data:
            raise DatabaseError("Insert succeeded but returned no data")
        return result.data[0]["id"]

    async def get_draft(
        self, draft_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get draft by ID.

        Args:
            draft_id: UUID of the draft.

        Returns:
            Draft dict or ``None`` if not found.
        """
        validate_not_empty(draft_id, "draft_id")

        result = await (
            self.client.table("drafts")
            .select("*")
            .eq("id", draft_id)
            .execute()
        )
        return result.data[0] if result.data else None

    async def update_draft(self, draft: Dict[str, Any]) -> None:
        """Update a draft.

        Args:
            draft: Dict containing at least ``id`` and the fields to
                update.

        Raises:
            ValidationError: If *draft* is empty or lacks ``id``.
        """
        if not draft:
            raise ValidationError("draft cannot be None or empty")
        if "id" not in draft:
            raise ValidationError("draft must have 'id' for update")

        await (
            self.client.table("drafts")
            .update(draft)
            .eq("id", draft["id"])
            .execute()
        )

    # -----------------------------------------------------------------
    # AGENT LOGS
    # -----------------------------------------------------------------

    async def save_agent_log(self, log_entry: Dict[str, Any]) -> str:
        """Save an agent log entry.

        Args:
            log_entry: Log entry dict.  Must contain ``timestamp``
                and ``level``.

        Returns:
            UUID of the inserted log row.

        Raises:
            ValidationError: On missing / invalid fields.
            DatabaseError: When the insert returns no data.
        """
        if not log_entry:
            raise ValidationError("log_entry cannot be None or empty")
        if "timestamp" not in log_entry or "level" not in log_entry:
            raise ValidationError(
                "log_entry must have 'timestamp' and 'level'"
            )

        result = await (
            self.client.table("agent_logs").insert(log_entry).execute()
        )
        if not result.data:
            raise DatabaseError("Insert succeeded but returned no data")
        return result.data[0]["id"]

    async def get_agent_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[int] = None,
        component: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query agent logs with optional filters.

        Args:
            start_time: Only logs at or after this time.
            end_time: Only logs at or before this time.
            level: Minimum log level (e.g. ``20`` for INFO).
            component: Filter by component name.
            limit: Maximum number of rows (default 100).

        Returns:
            List of log dicts ordered by ``timestamp`` descending.
        """
        validate_positive(limit, "limit")

        query = self.client.table("agent_logs").select("*")

        if start_time:
            query = query.gte("timestamp", start_time.isoformat())
        if end_time:
            query = query.lte("timestamp", end_time.isoformat())
        if level:
            query = query.gte("level", level)
        if component:
            query = query.eq("component", component)

        result = await (
            query.order("timestamp", desc=True).limit(limit).execute()
        )
        return result.data

    # -----------------------------------------------------------------
    # PIPELINE ERRORS
    # -----------------------------------------------------------------

    async def save_pipeline_error(self, error: Dict[str, Any]) -> str:
        """Save a pipeline error for post-mortem analysis.

        Args:
            error: Error data dict.

        Returns:
            UUID of the inserted error row.

        Raises:
            ValidationError: If *error* is empty.
            DatabaseError: When the insert returns no data.
        """
        if not error:
            raise ValidationError("error cannot be None or empty")

        result = await (
            self.client.table("pipeline_errors").insert(error).execute()
        )
        if not result.data:
            raise DatabaseError("Insert succeeded but returned no data")
        return result.data[0]["id"]

    # -----------------------------------------------------------------
    # PENDING APPROVALS
    # -----------------------------------------------------------------

    async def save_pending_approval(
        self, approval: Dict[str, Any]
    ) -> str:
        """Save a pending approval request.

        Args:
            approval: Approval data dict.  Must contain ``run_id``.

        Returns:
            UUID of the inserted approval row.

        Raises:
            ValidationError: If *approval* is empty or lacks ``run_id``.
            DatabaseError: When the insert returns no data.
        """
        if not approval or "run_id" not in approval:
            raise ValidationError("approval must have 'run_id'")

        result = await (
            self.client.table("pending_approvals")
            .insert(approval)
            .execute()
        )
        if not result.data:
            raise DatabaseError("Insert succeeded but returned no data")
        return result.data[0]["id"]

    async def get_pending_approvals(
        self, status: str = "pending"
    ) -> List[Dict[str, Any]]:
        """Get pending approvals by status.

        Args:
            status: Approval status filter (default ``"pending"``).

        Returns:
            List of approval dicts ordered by ``requested_at``
            ascending (oldest first).
        """
        validate_not_empty(status, "status")

        result = await (
            self.client.table("pending_approvals")
            .select("*")
            .eq("status", status)
            .order("requested_at", desc=False)
            .execute()
        )
        return result.data

    # -----------------------------------------------------------------
    # SCHEDULED POSTS
    # -----------------------------------------------------------------

    async def save_scheduled_post(
        self, post: Dict[str, Any]
    ) -> str:
        """Save a scheduled post.

        Args:
            post: Scheduled post data dict.  Must contain
                ``content``, ``scheduled_time``, and ``status``.

        Returns:
            UUID of the inserted scheduled post row.

        Raises:
            ValidationError: On missing / invalid fields.
            DatabaseError: When the insert returns no data.
        """
        if not post:
            raise ValidationError(
                "scheduled post cannot be None or empty"
            )

        required_fields: Set[str] = {
            "content",
            "scheduled_time",
            "status",
        }
        missing = required_fields - set(post.keys())
        if missing:
            raise ValidationError(
                f"scheduled post missing required fields: {missing}"
            )

        result = await (
            self.client.table("scheduled_posts")
            .insert(post)
            .execute()
        )
        if not result.data:
            raise DatabaseError("Insert succeeded but returned no data")
        return result.data[0]["id"]

    async def get_due_posts(self) -> List[Dict[str, Any]]:
        """Get posts that are due for publishing.

        Returns all posts with status ``"scheduled"`` whose
        ``scheduled_time`` is at or before the current UTC time.

        Returns:
            List of scheduled post dicts ordered by
            ``scheduled_time`` ascending.
        """
        now = utc_now().isoformat()
        result = await (
            self.client.table("scheduled_posts")
            .select("*")
            .eq("status", "scheduled")
            .lte("scheduled_time", now)
            .order("scheduled_time", desc=False)
            .execute()
        )
        return result.data

    async def claim_post(self, post_id: str) -> bool:
        """Atomically claim a scheduled post for publishing.

        Transitions the post from ``"scheduled"`` to ``"publishing"``
        status.  Only succeeds if the post currently has status
        ``"scheduled"``, preventing double-publishing.

        Args:
            post_id: UUID of the scheduled post.

        Returns:
            ``True`` if the claim succeeded, ``False`` if the post was
            already claimed or in a different status.
        """
        validate_not_empty(post_id, "post_id")

        result = await (
            self.client.table("scheduled_posts")
            .update({
                "status": "publishing",
                "claimed_at": utc_now().isoformat(),
            })
            .eq("id", post_id)
            .eq("status", "scheduled")
            .execute()
        )
        # If data is returned, the update matched and the claim succeeded
        return bool(result.data)

    async def update_scheduled_post(
        self, post: Dict[str, Any]
    ) -> None:
        """Update a scheduled post.

        Args:
            post: Dict containing at least ``id`` and the fields to
                update.

        Raises:
            ValidationError: If *post* is empty or lacks ``id``.
        """
        if not post:
            raise ValidationError(
                "scheduled post cannot be None or empty"
            )
        if "id" not in post:
            raise ValidationError(
                "scheduled post must have 'id' for update"
            )

        await (
            self.client.table("scheduled_posts")
            .update(post)
            .eq("id", post["id"])
            .execute()
        )

    # -----------------------------------------------------------------
    # AUTHOR PROFILES
    # -----------------------------------------------------------------

    async def save_author_profile(
        self, profile: Dict[str, Any]
    ) -> str:
        """Save or update an author voice profile.

        Uses ``upsert`` so that an existing profile for the same
        ``author_name`` is updated rather than creating a duplicate.

        Args:
            profile: Author profile data dict.  Must contain
                ``author_name``.

        Returns:
            UUID of the inserted / updated profile row.

        Raises:
            ValidationError: On missing / invalid fields.
            DatabaseError: When the upsert returns no data.
        """
        if not profile:
            raise ValidationError("profile cannot be None or empty")
        if "author_name" not in profile or not profile["author_name"]:
            raise ValidationError("profile must have 'author_name'")

        result = await (
            self.client.table("author_profiles")
            .upsert(profile, on_conflict="author_name")
            .execute()
        )
        if not result.data:
            raise DatabaseError("Upsert succeeded but returned no data")
        return result.data[0]["id"]

    async def get_author_profile(self) -> Optional[Dict[str, Any]]:
        """Get the author voice profile.

        For a single-author system, returns the most recently updated
        profile.

        Returns:
            Author profile dict or ``None`` if no profile exists.
        """
        result = await (
            self.client.table("author_profiles")
            .select("*")
            .order("last_updated", desc=True)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None


# =============================================================================
# GLOBAL DATABASE INSTANCE (Singleton)
# =============================================================================

# One async connection for the entire application.
_db_instance: Optional[SupabaseDB] = None
_db_lock: Optional[asyncio.Lock] = None

# Thread lock for safe initialisation of the async lock itself.
_init_lock = threading.Lock()


async def get_db() -> SupabaseDB:
    """Get the global async database instance.

    Thread-safe **and** async-safe.  The first call creates the
    :class:`SupabaseDB` singleton; subsequent calls return the same
    instance.

    Returns:
        The singleton :class:`SupabaseDB` instance.
    """
    global _db_instance, _db_lock

    # Thread-safe lazy initialisation of the async lock.
    if _db_lock is None:
        with _init_lock:
            # Double-check after acquiring thread lock.
            if _db_lock is None:
                _db_lock = asyncio.Lock()

    if _db_instance is None:
        async with _db_lock:
            # Double-check after acquiring async lock.
            if _db_instance is None:
                _db_instance = await SupabaseDB.create()

    return _db_instance
