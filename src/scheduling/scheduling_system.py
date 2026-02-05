"""
Scheduling system for optimal LinkedIn post timing and conflict avoidance.

``SchedulingSystem`` manages the post scheduling lifecycle: finding optimal
time slots, creating scheduled posts, handling status transitions, and
recovering from stuck states.

All database interactions go through the ``db`` parameter (a
:class:`~src.database.SupabaseDB` instance).

Architecture reference: architecture.md lines 23902-24180
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from src.scheduling.models import PostStatus, PublishingSlot, ScheduledPost
from src.utils import generate_id, utc_now

logger = logging.getLogger(__name__)


class SchedulingSystem:
    """Manages post scheduling with optimal timing and conflict avoidance.

    Uses a list of ``PublishingSlot`` definitions to determine the best
    times for LinkedIn posts.  Ensures no two posts are scheduled within
    a configurable conflict window (default: 2 hours).

    Args:
        db: Database client (:class:`~src.database.SupabaseDB`).
        slots: Optional list of preferred publishing slots.  When
            ``None``, uses ``DEFAULT_SLOTS`` which target peak LinkedIn
            engagement windows (Tue-Thu mornings, lunch, and evening).
    """

    # ----------------------------------------------------------------
    # Default slots based on LinkedIn engagement research:
    # Tuesday-Thursday, 8-10am, 12pm, 5-6pm
    # ----------------------------------------------------------------
    DEFAULT_SLOTS: List[PublishingSlot] = [
        # Tuesday morning
        PublishingSlot(day_of_week=1, hour=8, minute=0, priority=2, reason="Tue 8am - early engagement"),
        PublishingSlot(day_of_week=1, hour=9, minute=0, priority=1, reason="Tue 9am - peak morning"),
        PublishingSlot(day_of_week=1, hour=10, minute=0, priority=2, reason="Tue 10am - late morning"),
        # Tuesday lunch & evening
        PublishingSlot(day_of_week=1, hour=12, minute=0, priority=3, reason="Tue 12pm - lunch break"),
        PublishingSlot(day_of_week=1, hour=17, minute=0, priority=2, reason="Tue 5pm - evening commute"),
        PublishingSlot(day_of_week=1, hour=18, minute=0, priority=3, reason="Tue 6pm - post-work"),
        # Wednesday morning
        PublishingSlot(day_of_week=2, hour=8, minute=0, priority=2, reason="Wed 8am - early engagement"),
        PublishingSlot(day_of_week=2, hour=9, minute=0, priority=1, reason="Wed 9am - peak morning"),
        PublishingSlot(day_of_week=2, hour=10, minute=0, priority=2, reason="Wed 10am - late morning"),
        # Wednesday lunch & evening
        PublishingSlot(day_of_week=2, hour=12, minute=0, priority=3, reason="Wed 12pm - lunch break"),
        PublishingSlot(day_of_week=2, hour=17, minute=0, priority=2, reason="Wed 5pm - evening commute"),
        PublishingSlot(day_of_week=2, hour=18, minute=0, priority=3, reason="Wed 6pm - post-work"),
        # Thursday morning
        PublishingSlot(day_of_week=3, hour=8, minute=0, priority=2, reason="Thu 8am - early engagement"),
        PublishingSlot(day_of_week=3, hour=9, minute=0, priority=1, reason="Thu 9am - peak morning"),
        PublishingSlot(day_of_week=3, hour=10, minute=0, priority=2, reason="Thu 10am - late morning"),
        # Thursday lunch & evening
        PublishingSlot(day_of_week=3, hour=12, minute=0, priority=3, reason="Thu 12pm - lunch break"),
        PublishingSlot(day_of_week=3, hour=17, minute=0, priority=2, reason="Thu 5pm - evening commute"),
        PublishingSlot(day_of_week=3, hour=18, minute=0, priority=3, reason="Thu 6pm - post-work"),
    ]

    # Minimum gap between any two scheduled posts (hours)
    CONFLICT_WINDOW_HOURS: int = 2

    # How many days ahead to search for available slots
    MAX_SEARCH_DAYS: int = 7

    # Timeout for stuck post recovery (minutes)
    STUCK_TIMEOUT_MINUTES: int = 10

    def __init__(
        self,
        db: "SupabaseDB",  # noqa: F821
        slots: Optional[List[PublishingSlot]] = None,
    ) -> None:
        self.db = db
        self.slots: List[PublishingSlot] = slots if slots is not None else self.DEFAULT_SLOTS

    # ================================================================
    # SLOT FINDING
    # ================================================================

    async def find_next_slot(self, after: Optional[datetime] = None) -> datetime:
        """Find the next available publishing slot that avoids conflicts.

        Iterates through preferred slots for the next ``MAX_SEARCH_DAYS``
        days, ordered by priority, returning the first slot that does not
        conflict with an already-scheduled post.

        Args:
            after: Only consider slots after this time.  Defaults to
                the current UTC time.

        Returns:
            A timezone-aware UTC ``datetime`` for the next available slot.

        Raises:
            RuntimeError: If no available slot is found within the search
                window.
        """
        if after is None:
            after = utc_now()

        # Ensure timezone-aware
        if after.tzinfo is None:
            after = after.replace(tzinfo=timezone.utc)

        # Sort slots by priority (lower = better)
        sorted_slots = sorted(self.slots, key=lambda s: s.priority)

        for days_ahead in range(self.MAX_SEARCH_DAYS):
            check_date = after + timedelta(days=days_ahead)

            for slot in sorted_slots:
                if check_date.weekday() != slot.day_of_week:
                    continue

                candidate = check_date.replace(
                    hour=slot.hour,
                    minute=slot.minute,
                    second=0,
                    microsecond=0,
                    tzinfo=timezone.utc,
                )

                # Skip slots in the past
                if candidate <= after:
                    continue

                # Check for conflicts with existing scheduled posts
                has_conflict = await self._check_conflict(candidate)
                if not has_conflict:
                    logger.info(
                        "[SCHEDULER] Found available slot: %s (%s)",
                        candidate.isoformat(),
                        slot.reason,
                    )
                    return candidate

        raise RuntimeError(
            f"[SCHEDULER] No available slot found in next {self.MAX_SEARCH_DAYS} days. "
            "Consider adjusting constraints or clearing the queue."
        )

    # ================================================================
    # POST MANAGEMENT
    # ================================================================

    async def schedule_post(
        self,
        run_id: str,
        content_type: str,
        text: str,
        visual_path: Optional[str] = None,
        qc_score: float = 0.0,
    ) -> ScheduledPost:
        """Schedule a post for publication at the next optimal time.

        Finds the next available slot, creates a ``ScheduledPost``, and
        persists it to the database.

        Args:
            run_id: Pipeline run ID that produced this content.
            content_type: Type of content (e.g. ``"enterprise_case"``).
            text: Full post text.
            visual_path: Optional path to a visual asset (image).
            qc_score: Quality control score from the QC agent.

        Returns:
            The newly created ``ScheduledPost`` with status
            ``SCHEDULED``.
        """
        slot_time = await self.find_next_slot()

        post = ScheduledPost(
            id=generate_id(),
            run_id=run_id,
            content_type=content_type,
            text=text,
            visual_path=visual_path,
            scheduled_at=slot_time,
            status=PostStatus.SCHEDULED,
            qc_score=qc_score,
        )

        # Persist to Supabase
        await self.db.save_scheduled_post({
            "id": post.id,
            "content": post.text,
            "content_type": post.content_type,
            "run_id": post.run_id,
            "visual_path": post.visual_path,
            "scheduled_time": post.scheduled_at.isoformat(),
            "status": post.status.value,
            "qc_score": post.qc_score,
            "created_at": post.created_at.isoformat(),
        })

        logger.info(
            "[SCHEDULER] Post %s scheduled for %s (run=%s, type=%s, qc=%.1f)",
            post.id,
            post.scheduled_at.isoformat(),
            post.run_id,
            post.content_type,
            post.qc_score,
        )

        return post

    async def get_scheduled_posts(
        self,
        status: Optional[PostStatus] = None,
        limit: int = 20,
    ) -> List[ScheduledPost]:
        """Query scheduled posts from the database.

        Args:
            status: Optional filter by post status.  When ``None``,
                returns posts in all statuses.
            limit: Maximum number of posts to return (default 20).

        Returns:
            List of ``ScheduledPost`` objects ordered by scheduled time.
        """
        # Build query
        query = (
            self.db.client.table("scheduled_posts")
            .select("*")
            .order("scheduled_time", desc=False)
            .limit(limit)
        )

        if status is not None:
            query = query.eq("status", status.value)

        result = await query.execute()

        posts: List[ScheduledPost] = []
        for row in result.data:
            posts.append(self._row_to_post(row))

        logger.debug(
            "[SCHEDULER] Retrieved %d scheduled posts (filter=%s)",
            len(posts),
            status.value if status else "all",
        )
        return posts

    async def cancel_post(self, post_id: str) -> ScheduledPost:
        """Cancel a scheduled post.

        Sets the post status to ``CANCELLED``.  Only posts with status
        ``SCHEDULED`` can be cancelled.

        Args:
            post_id: UUID of the scheduled post to cancel.

        Returns:
            The updated ``ScheduledPost`` with status ``CANCELLED``.

        Raises:
            ValueError: If the post is not in ``SCHEDULED`` status.
        """
        posts = await self.get_scheduled_posts()
        post = next((p for p in posts if p.id == post_id), None)

        if post is None:
            raise ValueError(f"[SCHEDULER] Post {post_id} not found")

        if post.status != PostStatus.SCHEDULED:
            raise ValueError(
                f"[SCHEDULER] Cannot cancel post {post_id} with status '{post.status.value}'"
            )

        post.status = PostStatus.CANCELLED
        await self.db.update_scheduled_post({
            "id": post.id,
            "status": PostStatus.CANCELLED.value,
        })

        logger.info("[SCHEDULER] Post %s cancelled", post_id)
        return post

    async def mark_published(
        self,
        post_id: str,
        linkedin_post_id: str,
    ) -> None:
        """Mark a post as successfully published.

        Updates the post status to ``PUBLISHED``, records the LinkedIn
        post ID, and sets the publication timestamp.

        Args:
            post_id: UUID of the scheduled post.
            linkedin_post_id: LinkedIn's identifier for the published post.
        """
        now = utc_now()
        await self.db.update_scheduled_post({
            "id": post_id,
            "status": PostStatus.PUBLISHED.value,
            "published_at": now.isoformat(),
            "linkedin_post_id": linkedin_post_id,
        })

        logger.info(
            "[SCHEDULER] Post %s published (linkedin_id=%s)",
            post_id,
            linkedin_post_id,
        )

    async def mark_failed(self, post_id: str, error: str) -> None:
        """Mark a post as failed.

        Updates the post status to ``FAILED`` and records the error
        message for debugging.

        Args:
            post_id: UUID of the scheduled post.
            error: Error message describing the failure.
        """
        await self.db.update_scheduled_post({
            "id": post_id,
            "status": PostStatus.FAILED.value,
            "error": error,
        })

        logger.error(
            "[SCHEDULER] Post %s failed: %s",
            post_id,
            error,
        )

    # ================================================================
    # CONFLICT DETECTION
    # ================================================================

    async def _check_conflict(self, slot_time: datetime) -> bool:
        """Check if any post is already scheduled within the conflict window.

        The conflict window is ``CONFLICT_WINDOW_HOURS`` hours on each
        side of the proposed ``slot_time``.

        Args:
            slot_time: The proposed scheduling time (timezone-aware UTC).

        Returns:
            ``True`` if a conflict exists (another post is too close),
            ``False`` if the slot is available.
        """
        window_start = slot_time - timedelta(hours=self.CONFLICT_WINDOW_HOURS)
        window_end = slot_time + timedelta(hours=self.CONFLICT_WINDOW_HOURS)

        # Query for posts in SCHEDULED or PUBLISHING status within the window
        result = await (
            self.db.client.table("scheduled_posts")
            .select("id")
            .in_("status", [PostStatus.SCHEDULED.value, PostStatus.PUBLISHING.value])
            .gte("scheduled_time", window_start.isoformat())
            .lte("scheduled_time", window_end.isoformat())
            .execute()
        )

        has_conflict = bool(result.data)
        if has_conflict:
            logger.debug(
                "[SCHEDULER] Conflict detected at %s (%d nearby posts in window %s - %s)",
                slot_time.isoformat(),
                len(result.data),
                window_start.isoformat(),
                window_end.isoformat(),
            )

        return has_conflict

    # ================================================================
    # RECOVERY
    # ================================================================

    async def recover_stuck_posts(self) -> None:
        """Find and recover posts stuck in ``PUBLISHING`` status.

        Posts that have been in ``PUBLISHING`` status for longer than
        ``STUCK_TIMEOUT_MINUTES`` are considered stuck (e.g. the
        publishing process crashed).  These are marked as ``FAILED``
        with a descriptive error message.
        """
        cutoff = utc_now() - timedelta(minutes=self.STUCK_TIMEOUT_MINUTES)

        # Find posts stuck in PUBLISHING status
        result = await (
            self.db.client.table("scheduled_posts")
            .select("*")
            .eq("status", PostStatus.PUBLISHING.value)
            .lte("claimed_at", cutoff.isoformat())
            .execute()
        )

        if not result.data:
            return

        for row in result.data:
            post_id = row["id"]
            await self.db.update_scheduled_post({
                "id": post_id,
                "status": PostStatus.FAILED.value,
                "error": (
                    f"Publishing stuck for >{self.STUCK_TIMEOUT_MINUTES} minutes. "
                    "Marked as failed by recovery process."
                ),
            })
            logger.warning(
                "[SCHEDULER] Recovered stuck post %s (was in PUBLISHING for >%d min)",
                post_id,
                self.STUCK_TIMEOUT_MINUTES,
            )

        logger.info(
            "[SCHEDULER] Recovery complete: %d stuck posts marked as FAILED",
            len(result.data),
        )

    # ================================================================
    # INTERNAL HELPERS
    # ================================================================

    @staticmethod
    def _row_to_post(row: Dict) -> ScheduledPost:
        """Convert a database row dict to a ``ScheduledPost`` dataclass.

        Args:
            row: Dict from Supabase query result.

        Returns:
            A ``ScheduledPost`` instance.
        """
        published_at = None
        if row.get("published_at"):
            published_at = datetime.fromisoformat(row["published_at"])

        created_at = utc_now()
        if row.get("created_at"):
            created_at = datetime.fromisoformat(row["created_at"])

        scheduled_at = datetime.fromisoformat(row["scheduled_time"])

        return ScheduledPost(
            id=row["id"],
            run_id=row.get("run_id", ""),
            content_type=row.get("content_type", ""),
            text=row.get("content", ""),
            visual_path=row.get("visual_path"),
            scheduled_at=scheduled_at,
            status=PostStatus(row.get("status", "draft")),
            published_at=published_at,
            linkedin_post_id=row.get("linkedin_post_id"),
            error=row.get("error"),
            qc_score=float(row.get("qc_score", 0.0)),
            created_at=created_at,
        )


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "SchedulingSystem",
]
