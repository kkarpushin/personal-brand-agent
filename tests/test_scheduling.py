"""Tests for the scheduling subsystem models and module imports.

Validates:
- PostStatus enum values and is_terminal property
- ScheduledPost dataclass creation, defaults, and timezone awareness
- PublishingSlot dataclass creation and default values
- SchedulingSystem and PublishingScheduler can be imported
"""

from datetime import datetime, timezone

import pytest

from src.scheduling.models import PostStatus, PublishingSlot, ScheduledPost


# =============================================================================
# PostStatus enum tests
# =============================================================================


class TestPostStatus:
    """Tests for the PostStatus enum."""

    def test_post_status_has_exactly_six_values(self):
        """PostStatus should have exactly 6 members."""
        assert len(PostStatus) == 6

    def test_post_status_members(self):
        """PostStatus should contain DRAFT, SCHEDULED, PUBLISHING, PUBLISHED, FAILED, CANCELLED."""
        expected = {"DRAFT", "SCHEDULED", "PUBLISHING", "PUBLISHED", "FAILED", "CANCELLED"}
        actual = {member.name for member in PostStatus}
        assert actual == expected

    @pytest.mark.parametrize(
        "status",
        [PostStatus.PUBLISHED, PostStatus.CANCELLED, PostStatus.FAILED],
        ids=["PUBLISHED", "CANCELLED", "FAILED"],
    )
    def test_is_terminal_returns_true_for_terminal_statuses(self, status):
        """is_terminal should return True for PUBLISHED, CANCELLED, and FAILED."""
        assert status.is_terminal is True

    @pytest.mark.parametrize(
        "status",
        [PostStatus.DRAFT, PostStatus.SCHEDULED, PostStatus.PUBLISHING],
        ids=["DRAFT", "SCHEDULED", "PUBLISHING"],
    )
    def test_is_terminal_returns_false_for_non_terminal_statuses(self, status):
        """is_terminal should return False for DRAFT, SCHEDULED, and PUBLISHING."""
        assert status.is_terminal is False

    def test_post_status_string_values(self):
        """Each PostStatus member should have the expected lowercase string value."""
        assert PostStatus.DRAFT.value == "draft"
        assert PostStatus.SCHEDULED.value == "scheduled"
        assert PostStatus.PUBLISHING.value == "publishing"
        assert PostStatus.PUBLISHED.value == "published"
        assert PostStatus.FAILED.value == "failed"
        assert PostStatus.CANCELLED.value == "cancelled"


# =============================================================================
# ScheduledPost dataclass tests
# =============================================================================


class TestScheduledPost:
    """Tests for the ScheduledPost dataclass."""

    @pytest.fixture
    def scheduled_at(self):
        """A fixed timezone-aware datetime for scheduled_at."""
        return datetime(2025, 7, 1, 9, 0, 0, tzinfo=timezone.utc)

    @pytest.fixture
    def minimal_post(self, scheduled_at):
        """A ScheduledPost created with only the required fields."""
        return ScheduledPost(
            id="post-001",
            run_id="run-abc",
            content_type="enterprise_case",
            text="This is a test post about enterprise architecture.",
            visual_path=None,
            scheduled_at=scheduled_at,
        )

    def test_creation_with_required_fields(self, minimal_post, scheduled_at):
        """ScheduledPost should be creatable with only the required fields."""
        assert minimal_post.id == "post-001"
        assert minimal_post.run_id == "run-abc"
        assert minimal_post.content_type == "enterprise_case"
        assert minimal_post.text == "This is a test post about enterprise architecture."
        assert minimal_post.visual_path is None
        assert minimal_post.scheduled_at == scheduled_at

    def test_default_status_is_draft(self, minimal_post):
        """ScheduledPost default status should be PostStatus.DRAFT."""
        assert minimal_post.status is PostStatus.DRAFT

    def test_default_qc_score_is_zero(self, minimal_post):
        """ScheduledPost default qc_score should be 0.0."""
        assert minimal_post.qc_score == 0.0

    def test_created_at_is_timezone_aware(self, minimal_post):
        """ScheduledPost created_at should be timezone-aware (have tzinfo set)."""
        assert minimal_post.created_at.tzinfo is not None

    def test_created_at_is_utc(self, minimal_post):
        """ScheduledPost created_at should be in UTC."""
        assert minimal_post.created_at.tzinfo == timezone.utc

    def test_optional_fields_default_to_none(self, minimal_post):
        """Optional fields published_at, linkedin_post_id, error should default to None."""
        assert minimal_post.published_at is None
        assert minimal_post.linkedin_post_id is None
        assert minimal_post.error is None

    def test_creation_with_all_fields(self, scheduled_at):
        """ScheduledPost should accept all fields including optional ones."""
        published_at = datetime(2025, 7, 1, 9, 5, 0, tzinfo=timezone.utc)
        created_at = datetime(2025, 6, 30, 18, 0, 0, tzinfo=timezone.utc)

        post = ScheduledPost(
            id="post-002",
            run_id="run-xyz",
            content_type="thought_leadership",
            text="Deep dive into microservices patterns.",
            visual_path="/assets/images/microservices.png",
            scheduled_at=scheduled_at,
            status=PostStatus.PUBLISHED,
            published_at=published_at,
            linkedin_post_id="li-12345",
            error=None,
            qc_score=8.5,
            created_at=created_at,
        )

        assert post.id == "post-002"
        assert post.visual_path == "/assets/images/microservices.png"
        assert post.status is PostStatus.PUBLISHED
        assert post.published_at == published_at
        assert post.linkedin_post_id == "li-12345"
        assert post.qc_score == 8.5
        assert post.created_at == created_at

    def test_created_at_is_auto_generated(self, scheduled_at):
        """Two posts created at different times should have different created_at values (or at least both are set)."""
        post = ScheduledPost(
            id="post-auto",
            run_id="run-auto",
            content_type="tip",
            text="Quick tip.",
            visual_path=None,
            scheduled_at=scheduled_at,
        )
        # created_at should be a datetime instance, auto-populated by utc_now()
        assert isinstance(post.created_at, datetime)


# =============================================================================
# PublishingSlot dataclass tests
# =============================================================================


class TestPublishingSlot:
    """Tests for the PublishingSlot dataclass."""

    def test_creation_with_all_fields(self):
        """PublishingSlot should accept all fields."""
        slot = PublishingSlot(
            day_of_week=1,
            hour=9,
            minute=30,
            priority=2,
            reason="Tuesday morning peak",
        )
        assert slot.day_of_week == 1
        assert slot.hour == 9
        assert slot.minute == 30
        assert slot.priority == 2
        assert slot.reason == "Tuesday morning peak"

    def test_default_minute_is_zero(self):
        """PublishingSlot default minute should be 0."""
        slot = PublishingSlot(day_of_week=2, hour=10)
        assert slot.minute == 0

    def test_default_priority_is_one(self):
        """PublishingSlot default priority should be 1."""
        slot = PublishingSlot(day_of_week=3, hour=12)
        assert slot.priority == 1

    def test_default_reason_is_empty_string(self):
        """PublishingSlot default reason should be an empty string."""
        slot = PublishingSlot(day_of_week=4, hour=17)
        assert slot.reason == ""

    def test_all_defaults_together(self):
        """PublishingSlot created with only required fields should have correct defaults."""
        slot = PublishingSlot(day_of_week=0, hour=8)
        assert slot.minute == 0
        assert slot.priority == 1
        assert slot.reason == ""


# =============================================================================
# Module import tests
# =============================================================================


class TestSchedulingImports:
    """Tests that key scheduling classes can be imported from their modules."""

    def test_scheduling_system_can_be_imported(self):
        """SchedulingSystem should be importable from src.scheduling.scheduling_system."""
        from src.scheduling.scheduling_system import SchedulingSystem

        assert SchedulingSystem is not None
        assert callable(SchedulingSystem)

    def test_publishing_scheduler_can_be_imported(self):
        """PublishingScheduler should be importable from src.scheduling.publishing_scheduler."""
        from src.scheduling.publishing_scheduler import PublishingScheduler

        assert PublishingScheduler is not None
        assert callable(PublishingScheduler)

    def test_package_level_exports(self):
        """All key classes should be importable from the scheduling package."""
        from src.scheduling import (
            PostStatus,
            PublishingSlot,
            ScheduledPost,
            PublishingScheduler,
            SchedulingSystem,
        )

        assert PostStatus is not None
        assert PublishingSlot is not None
        assert ScheduledPost is not None
        assert PublishingScheduler is not None
        assert SchedulingSystem is not None
