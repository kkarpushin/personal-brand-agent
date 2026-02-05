"""
Scheduling data models: PostStatus, ScheduledPost, PublishingSlot.

Defines the core data structures used by the scheduling subsystem:
- ``PostStatus``: Lifecycle status of a scheduled post.
- ``ScheduledPost``: A post that has been scheduled for future publication.
- ``PublishingSlot``: A recurring time slot for optimal LinkedIn publishing.

Architecture reference: architecture.md lines 23830-23900
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List

from src.utils import utc_now


# =============================================================================
# POST STATUS ENUM
# =============================================================================


class PostStatus(Enum):
    """Lifecycle status of a scheduled post.

    Transitions:
        DRAFT -> SCHEDULED -> PUBLISHING -> PUBLISHED
                                         -> FAILED
                 SCHEDULED -> CANCELLED
    """

    DRAFT = "draft"
    SCHEDULED = "scheduled"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Check if status is terminal (no further transitions allowed)."""
        return self in {PostStatus.PUBLISHED, PostStatus.CANCELLED, PostStatus.FAILED}


# =============================================================================
# SCHEDULED POST
# =============================================================================


@dataclass
class ScheduledPost:
    """A post scheduled for future publication on LinkedIn.

    Attributes:
        id: Unique identifier (UUID).
        run_id: Pipeline run that produced this post.
        content_type: Type of content (e.g. ``"enterprise_case"``).
        text: Full post text content.
        visual_path: Optional path to visual asset (image).
        scheduled_at: When the post should be published (timezone-aware UTC).
        status: Current lifecycle status.
        published_at: Actual publication timestamp (set after publishing).
        linkedin_post_id: LinkedIn's identifier for the published post.
        error: Error message if publishing failed.
        qc_score: Quality control score assigned during pipeline.
        created_at: When this scheduled post record was created.
    """

    # Required fields
    id: str
    run_id: str
    content_type: str
    text: str
    visual_path: Optional[str]
    scheduled_at: datetime

    # Status tracking
    status: PostStatus = PostStatus.DRAFT
    published_at: Optional[datetime] = None
    linkedin_post_id: Optional[str] = None
    error: Optional[str] = None

    # Quality
    qc_score: float = 0.0

    # Metadata
    created_at: datetime = field(default_factory=utc_now)


# =============================================================================
# PUBLISHING SLOT
# =============================================================================


@dataclass
class PublishingSlot:
    """A recurring time slot for optimal LinkedIn publishing.

    Represents a specific day-of-week and time combination when posts
    tend to perform well. Used by ``SchedulingSystem`` to find the best
    times to schedule new posts.

    Attributes:
        day_of_week: Day of week as integer (0=Monday, 6=Sunday).
        hour: Hour of day in 24h format (0-23).
        minute: Minute of hour (0-59).
        priority: Slot priority where lower values are better.
        reason: Human-readable explanation of why this slot is preferred.
    """

    day_of_week: int  # 0=Monday, 6=Sunday
    hour: int  # 0-23
    minute: int = 0
    priority: int = 1  # lower = better
    reason: str = ""  # why this slot is good


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "PostStatus",
    "ScheduledPost",
    "PublishingSlot",
]
