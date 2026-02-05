"""Tests for src.exceptions -- custom exception hierarchy.

Validates the full exception hierarchy, attribute storage, message formatting,
and raise/catch mechanics for every exception class in the module.
"""

import pytest

from src.exceptions import (
    # Base
    AgentBaseError,
    # Core
    ValidationError,
    DatabaseError,
    ConfigurationError,
    SecurityError,
    RetryExhaustedError,
    CircuitOpenError,
    NodeTimeoutError,
    # LinkedIn API
    LinkedInRateLimitError,
    LinkedInSessionExpiredError,
    LinkedInAPIError,
    # Agent Pipeline
    TrendScoutError,
    TopicSelectionError,
    AnalyzerError,
    WriterError,
    VisualizerError,
    ImageGenerationError,
    SchedulingConflictError,
    # Validation
    MetadataTypeMismatchError,
    ExtractionTypeMismatchError,
    BoundaryValidationError,
    # Configuration
    ConfigurationBackupError,
    ConfigurationCorruptedError,
    ConfigurationAccessError,
    ConfigurationWriteError,
)


# =========================================================================
# Hierarchy tests -- isinstance checks
# =========================================================================


class TestExceptionHierarchy:
    """Verify that every exception sits in the correct inheritance chain."""

    # -- AgentBaseError subtree -------------------------------------------

    @pytest.mark.parametrize(
        "exc_cls",
        [
            LinkedInRateLimitError,
            LinkedInSessionExpiredError,
            LinkedInAPIError,
            TrendScoutError,
            TopicSelectionError,
            AnalyzerError,
            WriterError,
            VisualizerError,
            ImageGenerationError,
            SchedulingConflictError,
        ],
    )
    def test_agent_base_error_subclasses(self, exc_cls):
        """All agent-specific exceptions must be instances of AgentBaseError."""
        err = exc_cls("test message")
        assert isinstance(err, AgentBaseError)
        assert isinstance(err, Exception)

    def test_agent_base_error_is_exception(self):
        err = AgentBaseError("base")
        assert isinstance(err, Exception)

    # -- ValidationError subtree ------------------------------------------

    @pytest.mark.parametrize(
        "exc_cls",
        [
            MetadataTypeMismatchError,
            ExtractionTypeMismatchError,
        ],
    )
    def test_validation_error_subclasses_simple(self, exc_cls):
        """Simple ValidationError subclasses are also ValueError instances."""
        err = exc_cls("bad data")
        assert isinstance(err, ValidationError)
        assert isinstance(err, ValueError)
        assert isinstance(err, Exception)

    def test_boundary_validation_error_is_validation_and_value_error(self):
        err = BoundaryValidationError("AgentA", "AgentB", ["issue"])
        assert isinstance(err, ValidationError)
        assert isinstance(err, ValueError)
        assert isinstance(err, Exception)

    def test_validation_error_is_value_error(self):
        err = ValidationError("invalid")
        assert isinstance(err, ValueError)

    # -- ConfigurationError subtree ---------------------------------------

    @pytest.mark.parametrize(
        "exc_cls",
        [
            ConfigurationBackupError,
            ConfigurationCorruptedError,
            ConfigurationAccessError,
            ConfigurationWriteError,
        ],
    )
    def test_configuration_error_subclasses(self, exc_cls):
        """All configuration subclasses are instances of ConfigurationError."""
        err = exc_cls("config problem")
        assert isinstance(err, ConfigurationError)
        assert isinstance(err, Exception)

    # -- Standalone core exceptions ---------------------------------------

    def test_database_error_is_exception(self):
        assert isinstance(DatabaseError("db fail"), Exception)

    def test_security_error_is_exception(self):
        assert isinstance(SecurityError("forbidden"), Exception)

    def test_circuit_open_error_is_exception(self):
        assert isinstance(CircuitOpenError("open"), Exception)

    # -- Negative checks: no accidental cross-contamination ---------------

    def test_agent_error_not_value_error(self):
        """AgentBaseError subclasses must NOT be ValueError."""
        err = TrendScoutError("oops")
        assert not isinstance(err, ValueError)

    def test_configuration_error_not_agent_base_error(self):
        err = ConfigurationBackupError("backup failed")
        assert not isinstance(err, AgentBaseError)

    def test_database_error_not_agent_base_error(self):
        err = DatabaseError("connection lost")
        assert not isinstance(err, AgentBaseError)


# =========================================================================
# RetryExhaustedError
# =========================================================================


class TestRetryExhaustedError:
    """Verify attribute storage and message format for RetryExhaustedError."""

    def test_stores_operation(self):
        original = RuntimeError("boom")
        err = RetryExhaustedError("fetch_trends", 5, original)
        assert err.operation == "fetch_trends"

    def test_stores_attempts(self):
        original = RuntimeError("boom")
        err = RetryExhaustedError("fetch_trends", 5, original)
        assert err.attempts == 5

    def test_stores_last_error(self):
        original = RuntimeError("boom")
        err = RetryExhaustedError("fetch_trends", 5, original)
        assert err.last_error is original

    def test_message_format(self):
        original = ValueError("invalid input")
        err = RetryExhaustedError("analyze_post", 3, original)
        msg = str(err)
        assert "analyze_post" in msg
        assert "3" in msg
        assert "invalid input" in msg

    def test_message_contains_operation_and_attempts(self):
        err = RetryExhaustedError("write_draft", 10, IOError("disk"))
        msg = str(err)
        assert "write_draft failed after 10 attempts" in msg

    def test_message_contains_last_error(self):
        last = ConnectionError("timeout")
        err = RetryExhaustedError("api_call", 2, last)
        assert "Last error: timeout" in str(err)

    def test_is_exception(self):
        err = RetryExhaustedError("op", 1, RuntimeError())
        assert isinstance(err, Exception)

    def test_not_agent_base_error(self):
        err = RetryExhaustedError("op", 1, RuntimeError())
        assert not isinstance(err, AgentBaseError)


# =========================================================================
# NodeTimeoutError
# =========================================================================


class TestNodeTimeoutError:
    """Verify attribute storage and message format for NodeTimeoutError."""

    def test_stores_node_name(self):
        err = NodeTimeoutError("trend_scout", 30)
        assert err.node_name == "trend_scout"

    def test_stores_timeout(self):
        err = NodeTimeoutError("trend_scout", 30)
        assert err.timeout == 30

    def test_message_format(self):
        err = NodeTimeoutError("writer_agent", 60)
        msg = str(err)
        assert "writer_agent" in msg
        assert "60" in msg

    def test_message_exact_format(self):
        err = NodeTimeoutError("qc_agent", 45)
        assert str(err) == "Node 'qc_agent' timed out after 45 seconds"

    def test_is_exception(self):
        err = NodeTimeoutError("x", 1)
        assert isinstance(err, Exception)

    def test_not_agent_base_error(self):
        err = NodeTimeoutError("x", 1)
        assert not isinstance(err, AgentBaseError)


# =========================================================================
# BoundaryValidationError
# =========================================================================


class TestBoundaryValidationError:
    """Verify attribute storage and message format for BoundaryValidationError."""

    def test_stores_source_agent(self):
        err = BoundaryValidationError("writer", "humanizer", ["missing hook"])
        assert err.source_agent == "writer"

    def test_stores_target_agent(self):
        err = BoundaryValidationError("writer", "humanizer", ["missing hook"])
        assert err.target_agent == "humanizer"

    def test_stores_issues_list(self):
        issues = ["missing hook", "too long", "no CTA"]
        err = BoundaryValidationError("writer", "qc", issues)
        assert err.issues == issues
        assert len(err.issues) == 3

    def test_issues_list_identity(self):
        """The stored list must be the exact same object passed in."""
        issues = ["a", "b"]
        err = BoundaryValidationError("x", "y", issues)
        assert err.issues is issues

    def test_message_contains_agents(self):
        err = BoundaryValidationError("analyzer", "writer", ["missing field"])
        msg = str(err)
        assert "analyzer" in msg
        assert "writer" in msg

    def test_message_contains_boundary_arrow(self):
        err = BoundaryValidationError("analyzer", "writer", ["x"])
        assert "analyzer -> writer" in str(err)

    def test_message_contains_issues(self):
        issues = ["field_a missing", "field_b wrong type"]
        err = BoundaryValidationError("a", "b", issues)
        msg = str(err)
        assert "field_a missing" in msg
        assert "field_b wrong type" in msg

    def test_empty_issues_list(self):
        err = BoundaryValidationError("src", "tgt", [])
        assert err.issues == []
        assert "src -> tgt" in str(err)

    def test_is_validation_error(self):
        err = BoundaryValidationError("a", "b", [])
        assert isinstance(err, ValidationError)

    def test_is_value_error(self):
        err = BoundaryValidationError("a", "b", [])
        assert isinstance(err, ValueError)


# =========================================================================
# Raise / catch mechanics
# =========================================================================


class TestRaiseCatchMechanics:
    """Verify that exceptions can be raised and caught at each hierarchy level."""

    def test_raise_and_catch_agent_base_error(self):
        with pytest.raises(AgentBaseError):
            raise AgentBaseError("agent failure")

    def test_catch_subclass_as_agent_base_error(self):
        with pytest.raises(AgentBaseError):
            raise TrendScoutError("no trends found")

    def test_catch_subclass_as_specific_type(self):
        with pytest.raises(TrendScoutError):
            raise TrendScoutError("no trends found")

    def test_catch_validation_error_as_value_error(self):
        with pytest.raises(ValueError):
            raise ValidationError("bad value")

    def test_catch_metadata_mismatch_as_value_error(self):
        with pytest.raises(ValueError):
            raise MetadataTypeMismatchError("mismatch")

    def test_catch_extraction_mismatch_as_validation_error(self):
        with pytest.raises(ValidationError):
            raise ExtractionTypeMismatchError("mismatch")

    def test_catch_boundary_as_value_error(self):
        with pytest.raises(ValueError):
            raise BoundaryValidationError("a", "b", ["issue"])

    def test_catch_configuration_subclass_as_configuration_error(self):
        with pytest.raises(ConfigurationError):
            raise ConfigurationWriteError("cannot write")

    def test_catch_retry_exhausted_as_exception(self):
        with pytest.raises(Exception):
            raise RetryExhaustedError("op", 3, RuntimeError("err"))

    def test_catch_node_timeout_as_exception(self):
        with pytest.raises(Exception):
            raise NodeTimeoutError("node", 30)

    def test_catch_circuit_open_as_exception(self):
        with pytest.raises(CircuitOpenError):
            raise CircuitOpenError("circuit is open")

    def test_catch_database_error(self):
        with pytest.raises(DatabaseError):
            raise DatabaseError("connection refused")

    def test_catch_security_error(self):
        with pytest.raises(SecurityError):
            raise SecurityError("unauthorized")

    def test_linkedin_rate_limit_caught_as_agent_base(self):
        with pytest.raises(AgentBaseError):
            raise LinkedInRateLimitError("rate limited")

    def test_linkedin_session_expired_caught_as_agent_base(self):
        with pytest.raises(AgentBaseError):
            raise LinkedInSessionExpiredError("session gone")

    def test_linkedin_api_error_caught_as_agent_base(self):
        with pytest.raises(AgentBaseError):
            raise LinkedInAPIError("api 500")

    def test_scheduling_conflict_caught_as_agent_base(self):
        with pytest.raises(AgentBaseError):
            raise SchedulingConflictError("slot taken")

    def test_image_generation_caught_as_agent_base(self):
        with pytest.raises(AgentBaseError):
            raise ImageGenerationError("DALL-E failed")

    def test_writer_error_caught_as_agent_base(self):
        with pytest.raises(AgentBaseError):
            raise WriterError("draft generation failed")

    def test_visualizer_error_caught_as_agent_base(self):
        with pytest.raises(AgentBaseError):
            raise VisualizerError("visual creation failed")

    def test_analyzer_error_caught_as_agent_base(self):
        with pytest.raises(AgentBaseError):
            raise AnalyzerError("analysis failed")

    def test_topic_selection_error_caught_as_agent_base(self):
        with pytest.raises(AgentBaseError):
            raise TopicSelectionError("no suitable topic")

    @pytest.mark.parametrize(
        "exc_cls",
        [
            ConfigurationBackupError,
            ConfigurationCorruptedError,
            ConfigurationAccessError,
            ConfigurationWriteError,
        ],
    )
    def test_configuration_subclasses_caught_as_configuration_error(self, exc_cls):
        with pytest.raises(ConfigurationError):
            raise exc_cls("config issue")


# =========================================================================
# Message preservation
# =========================================================================


class TestMessagePreservation:
    """Verify that simple exceptions preserve their string message."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            AgentBaseError,
            ValidationError,
            DatabaseError,
            ConfigurationError,
            SecurityError,
            CircuitOpenError,
            LinkedInRateLimitError,
            LinkedInSessionExpiredError,
            LinkedInAPIError,
            TrendScoutError,
            TopicSelectionError,
            AnalyzerError,
            WriterError,
            VisualizerError,
            ImageGenerationError,
            SchedulingConflictError,
            MetadataTypeMismatchError,
            ExtractionTypeMismatchError,
            ConfigurationBackupError,
            ConfigurationCorruptedError,
            ConfigurationAccessError,
            ConfigurationWriteError,
        ],
    )
    def test_str_message_preserved(self, exc_cls):
        msg = "something went wrong"
        err = exc_cls(msg)
        assert str(err) == msg

    @pytest.mark.parametrize(
        "exc_cls",
        [
            AgentBaseError,
            ValidationError,
            DatabaseError,
            ConfigurationError,
            SecurityError,
            CircuitOpenError,
        ],
    )
    def test_args_tuple(self, exc_cls):
        err = exc_cls("detail")
        assert err.args == ("detail",)


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    """Edge-case coverage for special constructors."""

    def test_retry_exhausted_with_zero_attempts(self):
        err = RetryExhaustedError("op", 0, RuntimeError("nope"))
        assert err.attempts == 0
        assert "0 attempts" in str(err)

    def test_retry_exhausted_with_nested_exception(self):
        inner = ValueError("inner")
        outer = RuntimeError("outer")
        outer.__cause__ = inner
        err = RetryExhaustedError("op", 1, outer)
        assert err.last_error is outer
        assert err.last_error.__cause__ is inner

    def test_node_timeout_with_zero_timeout(self):
        err = NodeTimeoutError("node_x", 0)
        assert err.timeout == 0
        assert "0 seconds" in str(err)

    def test_boundary_validation_with_single_issue(self):
        err = BoundaryValidationError("src", "dest", ["only issue"])
        assert len(err.issues) == 1

    def test_boundary_validation_with_many_issues(self):
        issues = [f"issue_{i}" for i in range(50)]
        err = BoundaryValidationError("a", "b", issues)
        assert len(err.issues) == 50

    def test_retry_exhausted_last_error_type_preserved(self):
        """last_error retains its original exception type."""
        original = ConnectionError("conn refused")
        err = RetryExhaustedError("connect", 3, original)
        assert isinstance(err.last_error, ConnectionError)


# =========================================================================
# __all__ completeness
# =========================================================================


def test_all_exports_are_importable():
    """Every name in __all__ must be importable from src.exceptions."""
    import src.exceptions as mod

    for name in mod.__all__:
        assert hasattr(mod, name), f"{name} listed in __all__ but not defined"


def test_all_exports_count():
    """Sanity check: __all__ should list all 25 exception classes."""
    import src.exceptions as mod

    assert len(mod.__all__) == 25
