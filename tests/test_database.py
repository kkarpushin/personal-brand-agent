"""Tests for the src.database module.

Covers:
- SupabaseConfig construction and environment-based creation.
- SupabaseDB importability and method availability.
- validate_not_empty and validate_positive helper functions.
"""

import pytest

from src.database import SupabaseConfig, SupabaseDB, validate_not_empty, validate_positive
from src.exceptions import ValidationError


# =============================================================================
# SupabaseConfig tests
# =============================================================================


class TestSupabaseConfig:
    """Tests for the SupabaseConfig dataclass."""

    def test_create_with_url_and_key(self):
        """SupabaseConfig can be created with explicit url and key."""
        config = SupabaseConfig(url="https://example.supabase.co", key="test-key-123")
        assert config.url == "https://example.supabase.co"
        assert config.key == "test-key-123"

    def test_from_env_raises_when_vars_missing(self, monkeypatch):
        """SupabaseConfig.from_env raises ValueError when env vars are not set.

        Note: The conftest autouse fixture already clears these env vars,
        but we explicitly ensure they are absent for clarity.
        """
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)

        with pytest.raises(ValueError, match="SUPABASE_URL and SUPABASE_SERVICE_KEY must be set"):
            SupabaseConfig.from_env()

    def test_from_env_raises_when_url_missing(self, monkeypatch):
        """SupabaseConfig.from_env raises ValueError when only SUPABASE_URL is missing."""
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "sk-test-key")

        with pytest.raises(ValueError, match="SUPABASE_URL and SUPABASE_SERVICE_KEY must be set"):
            SupabaseConfig.from_env()

    def test_from_env_raises_when_key_missing(self, monkeypatch):
        """SupabaseConfig.from_env raises ValueError when only SUPABASE_SERVICE_KEY is missing."""
        monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
        monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)

        with pytest.raises(ValueError, match="SUPABASE_URL and SUPABASE_SERVICE_KEY must be set"):
            SupabaseConfig.from_env()

    def test_from_env_succeeds_when_vars_set(self, monkeypatch):
        """SupabaseConfig.from_env creates a config when both env vars are set."""
        monkeypatch.setenv("SUPABASE_URL", "https://test-project.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test")

        config = SupabaseConfig.from_env()

        assert config.url == "https://test-project.supabase.co"
        assert config.key == "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test"


# =============================================================================
# SupabaseDB importability and interface tests
# =============================================================================


class TestSupabaseDBInterface:
    """Tests that SupabaseDB can be imported and has the expected public methods."""

    def test_supabase_db_importable(self):
        """SupabaseDB class can be imported from src.database."""
        assert SupabaseDB is not None

    def test_has_create_classmethod(self):
        """SupabaseDB has a 'create' class method."""
        assert hasattr(SupabaseDB, "create")
        assert callable(getattr(SupabaseDB, "create"))

    @pytest.mark.parametrize(
        "method_name",
        [
            "save_post",
            "get_post",
            "get_recent_posts",
            "store_metrics_snapshot",
            "get_metrics_history",
            "save_learnings",
            "get_all_learnings",
            "save_modification",
            "get_modification",
            "save_experiment",
            "get_experiment",
            "save_scheduled_post",
            "get_due_posts",
            "claim_post",
            "save_author_profile",
            "get_author_profile",
            "save_draft",
            "get_draft",
            "save_agent_log",
            "get_agent_logs",
            "save_pipeline_error",
            "save_pending_approval",
            "get_pending_approvals",
            "save_research_report",
            "get_prompt",
            "save_prompt",
            "cache_topics",
            "get_cached_topics",
            "get_all_photos",
            "update_photo_usage",
            "save_photo_metadata",
        ],
    )
    def test_has_expected_method(self, method_name):
        """SupabaseDB has all expected async CRUD methods."""
        assert hasattr(SupabaseDB, method_name), (
            f"SupabaseDB is missing expected method '{method_name}'"
        )


# =============================================================================
# validate_not_empty tests
# =============================================================================


class TestValidateNotEmpty:
    """Tests for the validate_not_empty helper function."""

    def test_raises_for_empty_string(self):
        """validate_not_empty raises ValidationError for an empty string."""
        with pytest.raises(ValidationError, match="field_name cannot be empty string"):
            validate_not_empty("", "field_name")

    def test_raises_for_whitespace_only_string(self):
        """validate_not_empty raises ValidationError for a whitespace-only string."""
        with pytest.raises(ValidationError, match="cannot be empty string"):
            validate_not_empty("   ", "test_field")

    def test_raises_for_none(self):
        """validate_not_empty raises ValidationError for None."""
        with pytest.raises(ValidationError, match="my_field cannot be None"):
            validate_not_empty(None, "my_field")

    def test_passes_for_non_empty_string(self):
        """validate_not_empty does not raise for a non-empty string."""
        # Should not raise -- no assertion needed, just verify no exception
        validate_not_empty("hello", "field_name")

    def test_passes_for_non_string_truthy_values(self):
        """validate_not_empty does not raise for non-string truthy values like integers."""
        validate_not_empty(42, "field_name")
        validate_not_empty(["item"], "field_name")


# =============================================================================
# validate_positive tests
# =============================================================================


class TestValidatePositive:
    """Tests for the validate_positive helper function."""

    def test_raises_for_zero(self):
        """validate_positive raises ValidationError for zero."""
        with pytest.raises(ValidationError, match="count must be positive, got 0"):
            validate_positive(0, "count")

    def test_raises_for_negative_integer(self):
        """validate_positive raises ValidationError for a negative integer."""
        with pytest.raises(ValidationError, match="limit must be positive, got -5"):
            validate_positive(-5, "limit")

    def test_raises_for_negative_float(self):
        """validate_positive raises ValidationError for a negative float."""
        with pytest.raises(ValidationError, match="score must be positive, got -0.1"):
            validate_positive(-0.1, "score")

    def test_raises_for_none(self):
        """validate_positive raises ValidationError for None."""
        with pytest.raises(ValidationError, match="value cannot be None"):
            validate_positive(None, "value")

    def test_passes_for_positive_integer(self):
        """validate_positive does not raise for a positive integer."""
        validate_positive(1, "count")
        validate_positive(100, "limit")

    def test_passes_for_positive_float(self):
        """validate_positive does not raise for a positive float."""
        validate_positive(0.001, "threshold")
        validate_positive(3.14, "pi")
