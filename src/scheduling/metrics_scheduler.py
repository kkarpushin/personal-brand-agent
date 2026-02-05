"""
Background metrics scheduler that collects post metrics at scheduled intervals.

``MetricsScheduler`` runs as an asyncio background task, periodically checking
for recently published posts and collecting LinkedIn engagement metrics at
predetermined checkpoints after publication.

Collection schedule (Golden Hour Focus):
    - T+0min:   Post published -> record baseline
    - T+15min:  First check -> early momentum signal
    - T+30min:  Second check -> velocity calculation
    - T+60min:  Golden hour complete -> CRITICAL snapshot (Telegram notification)
    - T+3h:     Mid-day check
    - T+24h:    Day-1 performance
    - T+48h:    Final metrics (LinkedIn engagement peaks around 48h)

Architecture reference: architecture.md lines 13246-13265
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.database import get_db
from src.models import PostMetricsSnapshot
from src.tools.linkedin_client import LinkedInClient
from src.utils import utc_now

logger = logging.getLogger("MetricsScheduler")


# =============================================================================
# COLLECTION SCHEDULE
# =============================================================================

# Each entry is (minutes_after_post, label).
# Ordered by time so earlier checkpoints are processed first.
COLLECTION_CHECKPOINTS: List[Tuple[int, str]] = [
    (0, "baseline"),
    (15, "early_momentum"),
    (30, "velocity_check"),
    (60, "golden_hour"),
    (180, "mid_day"),
    (1440, "day_1"),
    (2880, "final_48h"),
]

# The checkpoint (in minutes) that triggers a Telegram notification.
GOLDEN_HOUR_CHECKPOINT: int = 60


class MetricsScheduler:
    """Background task that collects post metrics at scheduled intervals.

    After a post is published, the scheduler tracks it through a series of
    checkpoints (T+0, T+15min, T+30min, T+60min, T+3h, T+24h, T+48h).
    At each checkpoint it fetches engagement metrics from LinkedIn, calculates
    velocity, stores a snapshot in the database, and (for the golden-hour
    checkpoint) sends a Telegram notification.

    The scheduler runs as an asyncio loop similar to
    :class:`~src.scheduling.publishing_scheduler.PublishingScheduler`.

    Args:
        linkedin_client: LinkedIn API client for fetching metrics.
        telegram_notifier: Optional Telegram notifier for golden-hour alerts.
            When ``None``, Telegram notifications are silently skipped.
        check_interval_seconds: How often to scan for due checkpoints
            (default: 30 seconds).
    """

    def __init__(
        self,
        linkedin_client: LinkedInClient,
        telegram_notifier: Optional[Any] = None,
        check_interval_seconds: int = 30,
    ) -> None:
        self.linkedin_client = linkedin_client
        self.telegram_notifier = telegram_notifier
        self.check_interval_seconds = check_interval_seconds
        self._running: bool = False

        # Internal tracking: maps post_id -> set of completed checkpoint minutes.
        # Prevents duplicate collections within the same scheduler lifetime.
        self._completed_checkpoints: Dict[str, set] = {}

    # ================================================================
    # LIFECYCLE
    # ================================================================

    async def start(self) -> None:
        """Start the metrics scheduler background loop.

        Sets ``_running`` to ``True`` and enters the main check loop.
        The loop runs until :meth:`stop` is called or an unrecoverable
        error occurs.
        """
        self._running = True
        logger.info(
            "[METRICS] Metrics scheduler started (interval=%ds)",
            self.check_interval_seconds,
        )

        while self._running:
            try:
                await self._check_and_collect()
            except asyncio.CancelledError:
                logger.info("[METRICS] Metrics scheduler cancelled")
                break
            except Exception:
                logger.exception(
                    "[METRICS] Unexpected error in metrics scheduler loop"
                )

            # Wait for next check cycle
            try:
                await asyncio.sleep(self.check_interval_seconds)
            except asyncio.CancelledError:
                logger.info("[METRICS] Metrics scheduler sleep cancelled")
                break

        logger.info("[METRICS] Metrics scheduler stopped")

    async def stop(self) -> None:
        """Stop the metrics scheduler.

        Sets ``_running`` to ``False`` which causes the main loop in
        :meth:`start` to exit after the current iteration completes.
        """
        self._running = False
        logger.info("[METRICS] Metrics scheduler stop requested")

    # ================================================================
    # CORE CHECK LOOP
    # ================================================================

    async def _check_and_collect(self) -> None:
        """Scan for published posts with due metric checkpoints and collect.

        Queries the database for recently published posts (within the last
        48 hours), then for each post determines which checkpoints are due
        and have not yet been collected. Collects metrics for each due
        checkpoint.
        """
        db = await get_db()
        now = utc_now()

        # Get posts published within the last 48 hours that might still
        # have pending checkpoints.
        recent_posts = await self._get_recently_published_posts(db)

        if not recent_posts:
            return

        for post in recent_posts:
            post_id = post["id"]
            linkedin_post_id = post.get("linkedin_post_id")

            if not linkedin_post_id:
                logger.debug(
                    "[METRICS] Post %s has no linkedin_post_id, skipping",
                    post_id,
                )
                continue

            published_at = self._parse_timestamp(post.get("published_at"))
            if published_at is None:
                logger.warning(
                    "[METRICS] Post %s has no valid published_at, skipping",
                    post_id,
                )
                continue

            # Determine which checkpoints are due for this post
            due_checkpoints = self._get_due_checkpoints(
                post_id, published_at, now
            )

            for checkpoint_minutes, label in due_checkpoints:
                try:
                    await self._collect_checkpoint(
                        post_id=post_id,
                        linkedin_post_id=linkedin_post_id,
                        published_at=published_at,
                        checkpoint_minutes=checkpoint_minutes,
                        label=label,
                        db=db,
                    )
                except Exception:
                    # Metrics collection failures are non-critical.
                    # Log warning and continue to next checkpoint.
                    logger.warning(
                        "[METRICS] Failed to collect %s checkpoint for "
                        "post %s (T+%dmin)",
                        label,
                        post_id,
                        checkpoint_minutes,
                        exc_info=True,
                    )

    # ================================================================
    # POST DISCOVERY
    # ================================================================

    async def _get_recently_published_posts(
        self, db: Any
    ) -> List[Dict[str, Any]]:
        """Get posts published within the last 48 hours.

        Queries the ``scheduled_posts`` table for posts with status
        ``published`` and ``published_at`` within the last 48 hours.

        Args:
            db: The database client instance.

        Returns:
            List of published post dicts.
        """
        cutoff = (utc_now() - timedelta(hours=48)).isoformat()

        result = await (
            db.client.table("scheduled_posts")
            .select("*")
            .eq("status", "published")
            .gte("published_at", cutoff)
            .order("published_at", desc=False)
            .execute()
        )
        return result.data

    # ================================================================
    # CHECKPOINT RESOLUTION
    # ================================================================

    def _get_due_checkpoints(
        self,
        post_id: str,
        published_at: datetime,
        now: datetime,
    ) -> List[Tuple[int, str]]:
        """Determine which checkpoints are due and not yet collected.

        A checkpoint is due when the elapsed time since publication is
        greater than or equal to the checkpoint offset (in minutes).

        Args:
            post_id: UUID of the post.
            published_at: When the post was published (timezone-aware).
            now: Current UTC time.

        Returns:
            List of ``(minutes, label)`` tuples for due checkpoints.
        """
        elapsed = now - published_at
        elapsed_minutes = elapsed.total_seconds() / 60.0

        completed = self._completed_checkpoints.get(post_id, set())

        due: List[Tuple[int, str]] = []
        for checkpoint_minutes, label in COLLECTION_CHECKPOINTS:
            if checkpoint_minutes in completed:
                continue
            if elapsed_minutes >= checkpoint_minutes:
                due.append((checkpoint_minutes, label))

        return due

    # ================================================================
    # METRIC COLLECTION
    # ================================================================

    async def _collect_checkpoint(
        self,
        post_id: str,
        linkedin_post_id: str,
        published_at: datetime,
        checkpoint_minutes: int,
        label: str,
        db: Any,
    ) -> None:
        """Collect metrics for a single checkpoint and store the snapshot.

        Fetches current metrics from LinkedIn, calculates velocity based on
        the previous snapshot, builds a :class:`PostMetricsSnapshot`, and
        stores it in the database.

        Args:
            post_id: Internal post UUID.
            linkedin_post_id: LinkedIn post URN/ID for API calls.
            published_at: When the post was published (timezone-aware).
            checkpoint_minutes: Scheduled checkpoint offset in minutes.
            label: Human-readable checkpoint label.
            db: The database client instance.
        """
        now = utc_now()

        logger.info(
            "[METRICS] Collecting %s checkpoint for post %s (T+%dmin)",
            label,
            post_id,
            checkpoint_minutes,
        )

        # Fetch current metrics from LinkedIn
        raw_metrics = await self.linkedin_client.get_post_metrics(
            linkedin_post_id
        )

        # Fetch reactions breakdown
        reactions_by_type = await self._fetch_reactions_breakdown(
            linkedin_post_id
        )

        # Calculate actual minutes since publication
        actual_minutes = int(
            (now - published_at).total_seconds() / 60.0
        )

        # Calculate collection drift (how far off schedule we are)
        collection_drift_seconds = int(
            (now - published_at).total_seconds()
            - (checkpoint_minutes * 60)
        )

        # Calculate likes velocity from previous snapshot
        likes_velocity = await self._calculate_likes_velocity(
            db=db,
            post_id=post_id,
            current_likes=raw_metrics.get("likes", 0),
            current_time=now,
        )

        # Calculate engagement rate if impressions are available
        likes = raw_metrics.get("likes", 0)
        comments = raw_metrics.get("comments", 0)
        reposts = raw_metrics.get("shares", 0)
        impressions = raw_metrics.get("impressions") or None

        engagement_rate: Optional[float] = None
        if impressions and impressions > 0:
            engagement_rate = (likes + comments + reposts) / impressions

        # Build the snapshot
        snapshot = PostMetricsSnapshot(
            post_id=post_id,
            likes=likes,
            comments=comments,
            reposts=reposts,
            reactions_by_type=reactions_by_type,
            impressions=impressions,
            clicks=raw_metrics.get("clicks") or None,
            engagement_rate=engagement_rate,
            likes_velocity=likes_velocity,
            minutes_after_post=actual_minutes,
            collected_at=now,
            scheduled_checkpoint=checkpoint_minutes,
            collection_drift_seconds=collection_drift_seconds,
        )

        # Store in database
        snapshot_dict = self._snapshot_to_dict(snapshot)
        await db.store_metrics_snapshot(snapshot_dict)

        # Mark checkpoint as completed
        if post_id not in self._completed_checkpoints:
            self._completed_checkpoints[post_id] = set()
        self._completed_checkpoints[post_id].add(checkpoint_minutes)

        logger.info(
            "[METRICS] Stored %s snapshot for post %s: "
            "likes=%d, comments=%d, reposts=%d, velocity=%.2f likes/min",
            label,
            post_id,
            likes,
            comments,
            reposts,
            likes_velocity or 0.0,
        )

        # Send Telegram notification for golden hour checkpoint
        if checkpoint_minutes == GOLDEN_HOUR_CHECKPOINT:
            await self._notify_golden_hour(
                post_id=post_id,
                snapshot=snapshot,
            )

    # ================================================================
    # REACTIONS BREAKDOWN
    # ================================================================

    async def _fetch_reactions_breakdown(
        self, linkedin_post_id: str
    ) -> Dict[str, int]:
        """Fetch and tally reactions by type from LinkedIn.

        Args:
            linkedin_post_id: LinkedIn post URN/ID.

        Returns:
            Dict mapping reaction type to count. Falls back to default
            zeroes if the API call fails (metrics collection is
            non-critical).
        """
        default_reactions: Dict[str, int] = {
            "LIKE": 0,
            "CELEBRATE": 0,
            "SUPPORT": 0,
            "LOVE": 0,
            "INSIGHTFUL": 0,
            "FUNNY": 0,
        }

        try:
            raw_reactions = await self.linkedin_client.get_post_reactions(
                linkedin_post_id
            )
        except Exception:
            logger.warning(
                "[METRICS] Failed to fetch reactions for %s, "
                "using zeroes",
                linkedin_post_id,
                exc_info=True,
            )
            return default_reactions

        reactions_by_type = dict(default_reactions)
        for reaction in raw_reactions:
            rtype = reaction.get("type", "LIKE")
            if rtype in reactions_by_type:
                reactions_by_type[rtype] += 1
            else:
                # Unknown reaction type; count as LIKE
                reactions_by_type["LIKE"] += 1

        return reactions_by_type

    # ================================================================
    # VELOCITY CALCULATION
    # ================================================================

    async def _calculate_likes_velocity(
        self,
        db: Any,
        post_id: str,
        current_likes: int,
        current_time: datetime,
    ) -> Optional[float]:
        """Calculate likes velocity (likes per minute) since the last snapshot.

        Velocity = delta_likes / delta_minutes between the current collection
        and the most recent stored snapshot for this post.

        Args:
            db: Database client instance.
            post_id: UUID of the post.
            current_likes: Number of likes at the current collection time.
            current_time: Current UTC timestamp.

        Returns:
            Likes per minute as a float, or ``None`` if there is no
            previous snapshot to compare against.
        """
        history = await db.get_metrics_history(post_id, limit=1)

        if not history:
            return None

        previous = history[0]
        previous_likes = previous.get("likes", 0)
        previous_collected_at = self._parse_timestamp(
            previous.get("collected_at")
        )

        if previous_collected_at is None:
            return None

        delta_minutes = (
            current_time - previous_collected_at
        ).total_seconds() / 60.0

        if delta_minutes <= 0:
            return None

        delta_likes = current_likes - previous_likes
        return delta_likes / delta_minutes

    # ================================================================
    # GOLDEN HOUR NOTIFICATION
    # ================================================================

    async def _notify_golden_hour(
        self,
        post_id: str,
        snapshot: PostMetricsSnapshot,
    ) -> None:
        """Send a Telegram notification for the golden hour milestone.

        The golden hour (T+60min) is the single most important checkpoint
        for LinkedIn engagement prediction. This notification gives the
        author immediate feedback on early post performance.

        Args:
            post_id: UUID of the post.
            snapshot: The golden-hour metrics snapshot.
        """
        if self.telegram_notifier is None:
            logger.debug(
                "[METRICS] No Telegram notifier configured; "
                "skipping golden hour notification for post %s",
                post_id,
            )
            return

        velocity_str = (
            f"{snapshot.likes_velocity:.2f} likes/min"
            if snapshot.likes_velocity is not None
            else "N/A"
        )

        message = (
            "*Golden Hour Report (T+60min)*\n\n"
            f"Post: `{post_id[:8]}...`\n"
            f"Likes: {snapshot.likes}\n"
            f"Comments: {snapshot.comments}\n"
            f"Reposts: {snapshot.reposts}\n"
            f"Velocity: {velocity_str}\n"
        )

        if snapshot.impressions is not None:
            message += f"Impressions: {snapshot.impressions}\n"

        if snapshot.engagement_rate is not None:
            message += (
                f"Engagement Rate: {snapshot.engagement_rate:.2%}\n"
            )

        try:
            await self.telegram_notifier.send(message)
            logger.info(
                "[METRICS] Golden hour notification sent for post %s",
                post_id,
            )
        except Exception:
            # Notification failure is non-critical
            logger.warning(
                "[METRICS] Failed to send golden hour notification "
                "for post %s",
                post_id,
                exc_info=True,
            )

    # ================================================================
    # HELPERS
    # ================================================================

    @staticmethod
    def _parse_timestamp(value: Any) -> Optional[datetime]:
        """Parse an ISO-format timestamp string into a timezone-aware datetime.

        Handles both timezone-aware and naive ISO strings. Naive datetimes
        are assumed to be UTC.

        Args:
            value: ISO-format string, datetime, or ``None``.

        Returns:
            Timezone-aware UTC datetime, or ``None`` if parsing fails.
        """
        if value is None:
            return None

        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)

        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
                if dt.tzinfo is None:
                    return dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except (ValueError, TypeError):
                return None

        return None

    @staticmethod
    def _snapshot_to_dict(snapshot: PostMetricsSnapshot) -> Dict[str, Any]:
        """Convert a :class:`PostMetricsSnapshot` to a dict for DB storage.

        Serializes the ``collected_at`` datetime to an ISO string so that
        it is compatible with Supabase's TIMESTAMPTZ column.

        Args:
            snapshot: The metrics snapshot to convert.

        Returns:
            A plain dict suitable for ``db.store_metrics_snapshot()``.
        """
        data = asdict(snapshot)
        # Serialize datetime to ISO string for Supabase compatibility
        if isinstance(data.get("collected_at"), datetime):
            data["collected_at"] = data["collected_at"].isoformat()
        return data

    def cleanup_completed_checkpoints(self, post_id: str) -> None:
        """Remove tracking data for a post that has completed all checkpoints.

        Call this after the final (T+48h) checkpoint to free memory in
        long-running scheduler instances.

        Args:
            post_id: UUID of the post to remove from tracking.
        """
        self._completed_checkpoints.pop(post_id, None)
        logger.debug(
            "[METRICS] Cleaned up checkpoint tracking for post %s",
            post_id,
        )


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "MetricsScheduler",
    "COLLECTION_CHECKPOINTS",
    "GOLDEN_HOUR_CHECKPOINT",
]
