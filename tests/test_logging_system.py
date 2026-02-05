"""Tests for the logging module: LogLevel, LogComponent, LogEntry, and module imports."""

import json
from datetime import datetime, timezone

import pytest

from src.logging.models import LogComponent, LogEntry, LogLevel


# ---------------------------------------------------------------------------
# Fixed timestamp used across all tests for determinism
# ---------------------------------------------------------------------------
FIXED_TS = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


# ===================================================================
# LogLevel tests
# ===================================================================


class TestLogLevel:
    """Verify LogLevel enum values, ordering, and name_str property."""

    def test_has_five_levels(self) -> None:
        """LogLevel should define exactly 5 severity levels."""
        assert len(LogLevel) == 5

    def test_numeric_values(self) -> None:
        """Each level should have its expected numeric value."""
        assert LogLevel.DEBUG.value == 10
        assert LogLevel.INFO.value == 20
        assert LogLevel.WARNING.value == 30
        assert LogLevel.ERROR.value == 40
        assert LogLevel.CRITICAL.value == 50

    def test_ordering_debug_less_than_info(self) -> None:
        """DEBUG (10) should compare less than INFO (20)."""
        assert LogLevel.DEBUG.value < LogLevel.INFO.value

    def test_ordering_info_less_than_warning(self) -> None:
        """INFO (20) should compare less than WARNING (30)."""
        assert LogLevel.INFO.value < LogLevel.WARNING.value

    def test_ordering_warning_less_than_error(self) -> None:
        """WARNING (30) should compare less than ERROR (40)."""
        assert LogLevel.WARNING.value < LogLevel.ERROR.value

    def test_ordering_error_less_than_critical(self) -> None:
        """ERROR (40) should compare less than CRITICAL (50)."""
        assert LogLevel.ERROR.value < LogLevel.CRITICAL.value

    def test_full_ordering(self) -> None:
        """All five levels should sort in ascending severity order."""
        ordered = sorted(LogLevel, key=lambda lvl: lvl.value)
        assert ordered == [
            LogLevel.DEBUG,
            LogLevel.INFO,
            LogLevel.WARNING,
            LogLevel.ERROR,
            LogLevel.CRITICAL,
        ]

    def test_name_str_returns_lowercase(self) -> None:
        """name_str property should return the level name in lowercase."""
        assert LogLevel.DEBUG.name_str == "debug"
        assert LogLevel.INFO.name_str == "info"
        assert LogLevel.WARNING.name_str == "warning"
        assert LogLevel.ERROR.name_str == "error"
        assert LogLevel.CRITICAL.name_str == "critical"


# ===================================================================
# LogComponent tests
# ===================================================================


class TestLogComponent:
    """Verify LogComponent enum has the expected agent and infrastructure values."""

    def test_orchestrator_exists(self) -> None:
        """ORCHESTRATOR should be a valid LogComponent."""
        assert LogComponent.ORCHESTRATOR.value == "orchestrator"

    def test_trend_scout_exists(self) -> None:
        """TREND_SCOUT should be a valid LogComponent."""
        assert LogComponent.TREND_SCOUT.value == "trend_scout"

    def test_analyzer_exists(self) -> None:
        """ANALYZER should be a valid LogComponent."""
        assert LogComponent.ANALYZER.value == "analyzer"

    def test_writer_exists(self) -> None:
        """WRITER should be a valid LogComponent."""
        assert LogComponent.WRITER.value == "writer"

    def test_humanizer_exists(self) -> None:
        """HUMANIZER should be a valid LogComponent."""
        assert LogComponent.HUMANIZER.value == "humanizer"

    def test_meta_agent_exists(self) -> None:
        """META_AGENT should be a valid LogComponent."""
        assert LogComponent.META_AGENT.value == "meta_agent"

    def test_scheduler_exists(self) -> None:
        """SCHEDULER should be a valid LogComponent."""
        assert LogComponent.SCHEDULER.value == "scheduler"

    def test_database_exists(self) -> None:
        """DATABASE should be a valid LogComponent."""
        assert LogComponent.DATABASE.value == "database"

    def test_photo_selector_exists(self) -> None:
        """PHOTO_SELECTOR should be a valid LogComponent."""
        assert LogComponent.PHOTO_SELECTOR.value == "photo_selector"

    def test_has_many_components(self) -> None:
        """LogComponent should define a large number of system components."""
        # The enum has ~30 members; assert at least 20 to avoid brittleness
        assert len(LogComponent) >= 20


# ===================================================================
# LogEntry tests
# ===================================================================


class TestLogEntry:
    """Verify LogEntry creation, defaults, and serialization methods."""

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _make_entry(**overrides) -> LogEntry:
        """Create a LogEntry with sensible defaults, overridden as needed."""
        defaults = dict(
            timestamp=FIXED_TS,
            level=LogLevel.INFO,
            component=LogComponent.WRITER,
            message="Test log message",
        )
        defaults.update(overrides)
        return LogEntry(**defaults)

    # ---------------------------------------------------------------
    # Creation / defaults
    # ---------------------------------------------------------------

    def test_creation_with_required_fields(self) -> None:
        """LogEntry should be constructable with only the four required fields."""
        entry = self._make_entry()

        assert entry.timestamp == FIXED_TS
        assert entry.level == LogLevel.INFO
        assert entry.component == LogComponent.WRITER
        assert entry.message == "Test log message"

    def test_optional_fields_default_to_none(self) -> None:
        """Optional scalar fields should default to None."""
        entry = self._make_entry()

        assert entry.run_id is None
        assert entry.post_id is None
        assert entry.error_type is None
        assert entry.error_traceback is None
        assert entry.duration_ms is None

    def test_default_data_is_empty_dict(self) -> None:
        """The data field should default to an empty dict (not None)."""
        entry = self._make_entry()
        assert entry.data == {}
        assert isinstance(entry.data, dict)

    def test_data_default_is_independent_per_instance(self) -> None:
        """Each LogEntry should have its own data dict (no shared mutable default)."""
        entry_a = self._make_entry()
        entry_b = self._make_entry()
        entry_a.data["key"] = "value"
        assert "key" not in entry_b.data

    def test_creation_with_all_optional_fields(self) -> None:
        """LogEntry should accept all optional fields at construction time."""
        entry = self._make_entry(
            run_id="run-001",
            post_id="post-042",
            data={"topic": "AI"},
            error_type="ValueError",
            error_traceback="Traceback ...",
            duration_ms=350,
        )

        assert entry.run_id == "run-001"
        assert entry.post_id == "post-042"
        assert entry.data == {"topic": "AI"}
        assert entry.error_type == "ValueError"
        assert entry.error_traceback == "Traceback ..."
        assert entry.duration_ms == 350

    # ---------------------------------------------------------------
    # to_json
    # ---------------------------------------------------------------

    def test_to_json_returns_valid_json_string(self) -> None:
        """to_json() should return a parseable JSON string."""
        entry = self._make_entry()
        result = entry.to_json()

        assert isinstance(result, str)
        parsed = json.loads(result)  # should not raise
        assert isinstance(parsed, dict)

    def test_to_json_contains_expected_keys(self) -> None:
        """The JSON output should contain all defined fields."""
        entry = self._make_entry(run_id="run-001", duration_ms=100)
        parsed = json.loads(entry.to_json())

        expected_keys = {
            "timestamp",
            "level",
            "level_name",
            "component",
            "message",
            "run_id",
            "post_id",
            "data",
            "error_type",
            "error_traceback",
            "duration_ms",
        }
        assert set(parsed.keys()) == expected_keys

    def test_to_json_serializes_values_correctly(self) -> None:
        """to_json() should serialize level as numeric, component as string value."""
        entry = self._make_entry(
            level=LogLevel.ERROR,
            component=LogComponent.ORCHESTRATOR,
        )
        parsed = json.loads(entry.to_json())

        assert parsed["level"] == 40
        assert parsed["level_name"] == "error"
        assert parsed["component"] == "orchestrator"
        assert parsed["timestamp"] == FIXED_TS.isoformat()
        assert parsed["message"] == "Test log message"

    # ---------------------------------------------------------------
    # to_dict
    # ---------------------------------------------------------------

    def test_to_dict_returns_dict(self) -> None:
        """to_dict() should return a plain Python dict."""
        entry = self._make_entry()
        result = entry.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_contains_expected_keys(self) -> None:
        """to_dict() output should have the same keys as to_json()."""
        entry = self._make_entry()
        d = entry.to_dict()

        expected_keys = {
            "timestamp",
            "level",
            "level_name",
            "component",
            "message",
            "run_id",
            "post_id",
            "data",
            "error_type",
            "error_traceback",
            "duration_ms",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_match_to_json(self) -> None:
        """to_dict() and to_json() should produce equivalent data."""
        entry = self._make_entry(
            run_id="run-x",
            post_id="post-y",
            data={"score": 9.5},
            duration_ms=250,
        )
        d = entry.to_dict()
        j = json.loads(entry.to_json())

        assert d == j

    # ---------------------------------------------------------------
    # to_readable
    # ---------------------------------------------------------------

    def test_to_readable_returns_string(self) -> None:
        """to_readable() should return a human-readable string."""
        entry = self._make_entry()
        result = entry.to_readable()
        assert isinstance(result, str)

    def test_to_readable_contains_level_indicator(self) -> None:
        """to_readable() should include the bracketed level indicator."""
        assert "[INFO]" in self._make_entry(level=LogLevel.INFO).to_readable()
        assert "[DEBUG]" in self._make_entry(level=LogLevel.DEBUG).to_readable()
        assert "[WARN]" in self._make_entry(level=LogLevel.WARNING).to_readable()
        assert "[ERROR]" in self._make_entry(level=LogLevel.ERROR).to_readable()
        assert "[CRIT]" in self._make_entry(level=LogLevel.CRITICAL).to_readable()

    def test_to_readable_contains_time(self) -> None:
        """to_readable() should include the HH:MM:SS formatted time."""
        entry = self._make_entry()
        result = entry.to_readable()
        # FIXED_TS is 12:00:00 UTC
        assert "12:00:00" in result

    def test_to_readable_contains_component(self) -> None:
        """to_readable() should include the component value string."""
        entry = self._make_entry(component=LogComponent.TREND_SCOUT)
        result = entry.to_readable()
        assert "trend_scout" in result

    def test_to_readable_contains_message(self) -> None:
        """to_readable() should include the log message text."""
        entry = self._make_entry(message="Draft generated successfully")
        result = entry.to_readable()
        assert "Draft generated successfully" in result

    def test_to_readable_includes_duration_when_set(self) -> None:
        """When duration_ms is provided, to_readable() should append it."""
        entry = self._make_entry(duration_ms=450)
        result = entry.to_readable()
        assert "(450ms)" in result

    def test_to_readable_excludes_duration_when_not_set(self) -> None:
        """When duration_ms is None, to_readable() should not mention duration."""
        entry = self._make_entry(duration_ms=None)
        result = entry.to_readable()
        assert "ms)" not in result

    def test_to_readable_full_format(self) -> None:
        """Verify the complete format: [LEVEL] [HH:MM:SS] [component] message (Nms)."""
        entry = self._make_entry(
            level=LogLevel.WARNING,
            component=LogComponent.ANALYZER,
            message="Slow API call",
            duration_ms=1200,
        )
        result = entry.to_readable()
        assert result == "[WARN] [12:00:00] [analyzer] Slow API call (1200ms)"


# ===================================================================
# Module import tests
# ===================================================================


class TestModuleImports:
    """Verify that key classes can be imported from their respective modules."""

    def test_agent_logger_importable(self) -> None:
        """AgentLogger should be importable from src.logging.agent_logger."""
        from src.logging.agent_logger import AgentLogger

        assert AgentLogger is not None

    def test_component_logger_importable(self) -> None:
        """ComponentLogger should be importable from src.logging.component_logger."""
        from src.logging.component_logger import ComponentLogger

        assert ComponentLogger is not None

    def test_pipeline_run_logger_importable(self) -> None:
        """PipelineRunLogger should be importable from src.logging.pipeline_run_logger."""
        from src.logging.pipeline_run_logger import PipelineRunLogger

        assert PipelineRunLogger is not None

    def test_timed_operation_importable(self) -> None:
        """TimedOperation should be importable from src.logging.component_logger."""
        from src.logging.component_logger import TimedOperation

        assert TimedOperation is not None

    def test_top_level_package_exports(self) -> None:
        """The src.logging package __init__ should re-export key names."""
        from src.logging import (
            AgentLogger,
            ComponentLogger,
            LogComponent,
            LogEntry,
            LogLevel,
            PipelineRunLogger,
        )

        assert LogLevel is not None
        assert LogComponent is not None
        assert LogEntry is not None
        assert AgentLogger is not None
        assert ComponentLogger is not None
        assert PipelineRunLogger is not None
