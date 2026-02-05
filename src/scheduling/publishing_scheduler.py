"""
Background publishing scheduler that publishes posts at their scheduled times.

``PublishingScheduler`` runs as an asyncio background task, periodically
checking for posts that are due for publication and publishing them via
the LinkedIn client.  Includes stuck-post recovery to handle cases where
the publishing process crashes mid-flight.

Architecture reference: architecture.md lines 24182-24312
"""

import asyncio
import logging
from datetime import timezone
from typing import Optional

from src.scheduling.models import PostStatus, ScheduledPost
from src.scheduling.scheduling_system import SchedulingSystem
from src.utils import utc_now

logger = logging.getLogger(__name__)


class PublishingScheduler:
    """Background task that publishes scheduled posts at their designated times.

    Runs an asyncio loop that periodically:
    1. Checks for posts due for publication (status ``SCHEDULED``,
       ``scheduled_at <= now``).
    2. Transitions each due post to ``PUBLISHING`` status.
    3. Publishes via the LinkedIn client.
    4. Marks as ``PUBLISHED`` on success or ``FAILED`` on error.
    5. Periodically recovers posts stuck in ``PUBLISHING`` status.

    Args:
        scheduling_system: The scheduling system managing post lifecycle.
        linkedin_client: LinkedIn API client for publishing posts
            (:class:`~src.tools.linkedin_client.LinkedInClient`).
        check_interval_seconds: How often to check for due posts
            (default: 60 seconds).
    """

    # How often to run stuck-post recovery (every N check cycles)
    RECOVERY_INTERVAL_CYCLES: int = 10

    def __init__(
        self,
        scheduling_system: SchedulingSystem,
        linkedin_client: "LinkedInClient",  # noqa: F821
        check_interval_seconds: int = 60,
    ) -> None:
        self.scheduling_system = scheduling_system
        self.linkedin_client = linkedin_client
        self.check_interval_seconds = check_interval_seconds
        self._running: bool = False
        self._cycle_count: int = 0

    # ================================================================
    # LIFECYCLE
    # ================================================================

    async def start(self) -> None:
        """Start the publishing scheduler background loop.

        Sets ``_running`` to ``True`` and enters the main check loop.
        The loop runs until :meth:`stop` is called or an unrecoverable
        error occurs.
        """
        self._running = True
        self._cycle_count = 0
        logger.info(
            "[SCHEDULER] Publishing scheduler started (interval=%ds)",
            self.check_interval_seconds,
        )

        while self._running:
            try:
                await self._check_and_publish()
                self._cycle_count += 1

                # Run stuck-post recovery periodically
                if self._cycle_count % self.RECOVERY_INTERVAL_CYCLES == 0:
                    await self._recover_stuck()

            except asyncio.CancelledError:
                logger.info("[SCHEDULER] Publishing scheduler cancelled")
                break
            except Exception:
                logger.exception(
                    "[SCHEDULER] Unexpected error in publishing scheduler loop"
                )

            # Wait for next check cycle
            try:
                await asyncio.sleep(self.check_interval_seconds)
            except asyncio.CancelledError:
                logger.info("[SCHEDULER] Publishing scheduler sleep cancelled")
                break

        logger.info("[SCHEDULER] Publishing scheduler stopped")

    async def stop(self) -> None:
        """Stop the publishing scheduler.

        Sets ``_running`` to ``False`` which causes the main loop in
        :meth:`start` to exit after the current iteration completes.
        """
        self._running = False
        logger.info("[SCHEDULER] Publishing scheduler stop requested")

    # ================================================================
    # CORE CHECK LOOP
    # ================================================================

    async def _check_and_publish(self) -> None:
        """Check for posts due for publication and publish them.

        Queries the database for posts with status ``SCHEDULED`` and
        ``scheduled_at <= now``.  For each due post, atomically claims
        it (transitions to ``PUBLISHING``), then attempts to publish
        via the LinkedIn client.
        """
        now = utc_now()

        # Get all posts that are due for publishing
        due_posts = await self.scheduling_system.db.get_due_posts()

        if not due_posts:
            return

        logger.info(
            "[SCHEDULER] Found %d posts due for publishing",
            len(due_posts),
        )

        for row in due_posts:
            post_id = row["id"]

            # Atomically claim the post (SCHEDULED -> PUBLISHING)
            claimed = await self.scheduling_system.db.claim_post(post_id)
            if not claimed:
                logger.debug(
                    "[SCHEDULER] Post %s already claimed, skipping",
                    post_id,
                )
                continue

            # Convert row to ScheduledPost for publishing
            post = SchedulingSystem._row_to_post(row)
            post.status = PostStatus.PUBLISHING

            try:
                await self._publish_post(post)
            except Exception as exc:
                logger.error(
                    "[SCHEDULER] Failed to publish post %s: %s",
                    post_id,
                    exc,
                )
                await self.scheduling_system.mark_failed(
                    post_id,
                    str(exc),
                )

    # ================================================================
    # PUBLISHING
    # ================================================================

    async def _publish_post(self, post: ScheduledPost) -> None:
        """Publish a single post to LinkedIn.

        Calls the LinkedIn client to publish the post text and optional
        visual asset.  On success, marks the post as ``PUBLISHED`` with
        the LinkedIn post ID.  On failure, the exception propagates to
        the caller for error handling.

        Args:
            post: The ``ScheduledPost`` to publish (must be in
                ``PUBLISHING`` status).

        Raises:
            LinkedInAPIError: On LinkedIn API failures.
            LinkedInRateLimitError: If rate-limited by LinkedIn.
        """
        logger.info(
            "[SCHEDULER] Publishing post %s (type=%s, text_len=%d)",
            post.id,
            post.content_type,
            len(post.text),
        )

        # Publish via LinkedIn client
        result = await self.linkedin_client.publish_post(
            text=post.text,
            image_path=post.visual_path,
        )

        linkedin_post_id = result.get("post_id", "")

        # Mark as published
        await self.scheduling_system.mark_published(
            post.id,
            linkedin_post_id,
        )

        logger.info(
            "[SCHEDULER] Successfully published post %s (linkedin_id=%s)",
            post.id,
            linkedin_post_id,
        )

    # ================================================================
    # RECOVERY
    # ================================================================

    async def _recover_stuck(self) -> None:
        """Trigger stuck-post recovery on the scheduling system.

        Delegates to :meth:`SchedulingSystem.recover_stuck_posts` to
        find and handle posts that have been in ``PUBLISHING`` status
        for too long.
        """
        logger.debug("[SCHEDULER] Running stuck-post recovery check")
        await self.scheduling_system.recover_stuck_posts()


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "PublishingScheduler",
]
