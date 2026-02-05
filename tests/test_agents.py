"""
Tests for agent imports and basic construction.

Validates that all 7 agents and the orchestrator can be imported, that their
key classes and functions are accessible, and that core orchestrator helpers
(``create_content_pipeline``, ``initialize_pipeline_state``) produce expected
results without hitting any external services.
"""

import pytest


# =========================================================================
# INDIVIDUAL AGENT IMPORTS
#
# Each test verifies that the agent class can be imported from its module
# and that the imported name is indeed a class (not accidentally a function
# or module).
# =========================================================================


class TestTrendScoutImport:
    """Verify TrendScoutAgent is importable and is a class."""

    def test_import_trend_scout_agent(self):
        from src.agents.trend_scout import TrendScoutAgent

        assert TrendScoutAgent is not None

    def test_trend_scout_is_class(self):
        from src.agents.trend_scout import TrendScoutAgent

        assert isinstance(TrendScoutAgent, type)


class TestAnalyzerImport:
    """Verify AnalyzerAgent is importable and is a class."""

    def test_import_analyzer_agent(self):
        from src.agents.analyzer import AnalyzerAgent

        assert AnalyzerAgent is not None

    def test_analyzer_is_class(self):
        from src.agents.analyzer import AnalyzerAgent

        assert isinstance(AnalyzerAgent, type)


class TestWriterImport:
    """Verify WriterAgent is importable and is a class."""

    def test_import_writer_agent(self):
        from src.agents.writer import WriterAgent

        assert WriterAgent is not None

    def test_writer_is_class(self):
        from src.agents.writer import WriterAgent

        assert isinstance(WriterAgent, type)


class TestHumanizerImport:
    """Verify HumanizerAgent is importable and is a class."""

    def test_import_humanizer_agent(self):
        from src.agents.humanizer import HumanizerAgent

        assert HumanizerAgent is not None

    def test_humanizer_is_class(self):
        from src.agents.humanizer import HumanizerAgent

        assert isinstance(HumanizerAgent, type)


class TestVisualCreatorImport:
    """Verify VisualCreatorAgent is importable and is a class."""

    def test_import_visual_creator_agent(self):
        from src.agents.visual_creator import VisualCreatorAgent

        assert VisualCreatorAgent is not None

    def test_visual_creator_is_class(self):
        from src.agents.visual_creator import VisualCreatorAgent

        assert isinstance(VisualCreatorAgent, type)


class TestPhotoSelectorImport:
    """Verify PhotoSelectorAgent is importable and is a class."""

    def test_import_photo_selector_agent(self):
        from src.agents.photo_selector import PhotoSelectorAgent

        assert PhotoSelectorAgent is not None

    def test_photo_selector_is_class(self):
        from src.agents.photo_selector import PhotoSelectorAgent

        assert isinstance(PhotoSelectorAgent, type)


class TestQCAgentImport:
    """Verify QCAgent is importable and is a class."""

    def test_import_qc_agent(self):
        from src.agents.qc_agent import QCAgent

        assert QCAgent is not None

    def test_qc_is_class(self):
        from src.agents.qc_agent import QCAgent

        assert isinstance(QCAgent, type)


# =========================================================================
# AGENTS __init__ RE-EXPORTS
#
# The agents package __init__.py re-exports all seven agent classes.
# Verify those re-exports are consistent with the direct module imports.
# =========================================================================


class TestAgentsPackageExports:
    """Verify src.agents __init__.py re-exports all agent classes."""

    def test_package_exports_all_agents(self):
        from src.agents import (
            TrendScoutAgent,
            AnalyzerAgent,
            WriterAgent,
            HumanizerAgent,
            VisualCreatorAgent,
            PhotoSelectorAgent,
            QCAgent,
        )

        assert all(
            isinstance(cls, type)
            for cls in [
                TrendScoutAgent,
                AnalyzerAgent,
                WriterAgent,
                HumanizerAgent,
                VisualCreatorAgent,
                PhotoSelectorAgent,
                QCAgent,
            ]
        )

    def test_package_export_matches_direct_import(self):
        """Re-exported classes must be the exact same objects as direct imports."""
        from src.agents import TrendScoutAgent as pkg_ts
        from src.agents.trend_scout import TrendScoutAgent as mod_ts

        from src.agents import AnalyzerAgent as pkg_an
        from src.agents.analyzer import AnalyzerAgent as mod_an

        from src.agents import WriterAgent as pkg_wr
        from src.agents.writer import WriterAgent as mod_wr

        from src.agents import HumanizerAgent as pkg_hu
        from src.agents.humanizer import HumanizerAgent as mod_hu

        from src.agents import VisualCreatorAgent as pkg_vc
        from src.agents.visual_creator import VisualCreatorAgent as mod_vc

        from src.agents import PhotoSelectorAgent as pkg_ps
        from src.agents.photo_selector import PhotoSelectorAgent as mod_ps

        from src.agents import QCAgent as pkg_qc
        from src.agents.qc_agent import QCAgent as mod_qc

        assert pkg_ts is mod_ts
        assert pkg_an is mod_an
        assert pkg_wr is mod_wr
        assert pkg_hu is mod_hu
        assert pkg_vc is mod_vc
        assert pkg_ps is mod_ps
        assert pkg_qc is mod_qc


# =========================================================================
# ORCHESTRATOR IMPORTS
# =========================================================================


class TestOrchestratorImports:
    """Verify orchestrator functions are importable and callable."""

    def test_import_create_content_pipeline(self):
        from src.agents.orchestrator import create_content_pipeline

        assert callable(create_content_pipeline)

    def test_import_run_pipeline(self):
        from src.agents.orchestrator import run_pipeline

        assert callable(run_pipeline)

    def test_import_initialize_pipeline_state(self):
        from src.agents.orchestrator import initialize_pipeline_state

        assert callable(initialize_pipeline_state)


# =========================================================================
# ORCHESTRATOR -- create_content_pipeline
#
# Validates that calling create_content_pipeline() produces a compiled
# LangGraph graph without errors, and that the graph exposes the expected
# node names.
# =========================================================================


class TestCreateContentPipeline:
    """Verify create_content_pipeline builds a valid compiled graph."""

    def test_create_does_not_raise(self):
        """Calling create_content_pipeline() must succeed without errors."""
        from src.agents.orchestrator import create_content_pipeline

        graph = create_content_pipeline()
        assert graph is not None

    def test_compiled_graph_has_invoke(self):
        """The compiled graph should expose an ``invoke`` or ``ainvoke`` method."""
        from src.agents.orchestrator import create_content_pipeline

        graph = create_content_pipeline()
        # A compiled LangGraph StateGraph has both invoke and ainvoke
        assert hasattr(graph, "invoke") or hasattr(graph, "ainvoke")

    def test_compiled_graph_has_expected_nodes(self):
        """The graph should contain all pipeline node names."""
        from src.agents.orchestrator import create_content_pipeline

        graph = create_content_pipeline()

        expected_nodes = [
            "scout",
            "select_topic",
            "analyze",
            "write",
            "meta_evaluate",
            "humanize",
            "visualize",
            "qc",
            "learn",
            "prepare_output",
            "manual_review_queue",
            "handle_error",
            "reset_for_restart",
        ]

        # CompiledGraph stores nodes in a dict-like ``nodes`` attribute.
        # Different langgraph versions may expose this differently, so we
        # check a few common access patterns.
        graph_node_names = set()
        if hasattr(graph, "nodes"):
            graph_node_names = set(graph.nodes.keys())
        elif hasattr(graph, "builder") and hasattr(graph.builder, "nodes"):
            graph_node_names = set(graph.builder.nodes.keys())

        if graph_node_names:
            for name in expected_nodes:
                assert name in graph_node_names, (
                    f"Expected node '{name}' not found in graph. "
                    f"Available nodes: {sorted(graph_node_names)}"
                )

    def test_compiled_graph_entry_point_is_scout(self):
        """The entry point of the pipeline should be 'scout'."""
        from src.agents.orchestrator import create_content_pipeline

        graph = create_content_pipeline()

        # CompiledGraph typically stores the entry point information.
        # Check via the builder's entry_point or first node.
        if hasattr(graph, "builder"):
            builder = graph.builder
            if hasattr(builder, "_entry_point"):
                assert builder._entry_point == "scout"
            elif hasattr(builder, "entry_point"):
                assert builder.entry_point == "scout"


# =========================================================================
# ORCHESTRATOR -- initialize_pipeline_state
#
# Validates that initialize_pipeline_state returns a PipelineState dict
# with all expected keys and correct default values.
# =========================================================================


class TestInitializePipelineState:
    """Verify initialize_pipeline_state produces a correctly defaulted state."""

    def test_returns_dict(self):
        from src.agents.orchestrator import initialize_pipeline_state

        state = initialize_pipeline_state("test-run-001")
        assert isinstance(state, dict)

    def test_run_id_is_set(self):
        from src.agents.orchestrator import initialize_pipeline_state

        state = initialize_pipeline_state("test-run-002")
        assert state["run_id"] == "test-run-002"

    def test_run_timestamp_is_present(self):
        from src.agents.orchestrator import initialize_pipeline_state

        state = initialize_pipeline_state("test-run-003")
        assert "run_timestamp" in state
        assert state["run_timestamp"] is not None

    def test_revision_count_defaults_to_zero(self):
        from src.agents.orchestrator import initialize_pipeline_state

        state = initialize_pipeline_state("test-run-004")
        assert state["revision_count"] == 0

    def test_reject_restart_count_defaults_to_zero(self):
        from src.agents.orchestrator import initialize_pipeline_state

        state = initialize_pipeline_state("test-run-005")
        assert state["_reject_restart_count"] == 0

    def test_errors_defaults_to_empty_list(self):
        from src.agents.orchestrator import initialize_pipeline_state

        state = initialize_pipeline_state("test-run-006")
        assert state["errors"] == []

    def test_warnings_defaults_to_empty_list(self):
        from src.agents.orchestrator import initialize_pipeline_state

        state = initialize_pipeline_state("test-run-007")
        assert state["warnings"] == []

    def test_stage_is_initialized(self):
        from src.agents.orchestrator import initialize_pipeline_state

        state = initialize_pipeline_state("test-run-008")
        assert state["stage"] == "initialized"

    def test_selection_mode_defaults_to_auto(self):
        from src.agents.orchestrator import initialize_pipeline_state

        state = initialize_pipeline_state("test-run-009")
        assert state["selection_mode"] == "auto_top_pick"

    def test_selection_mode_can_be_overridden(self):
        from src.agents.orchestrator import initialize_pipeline_state

        state = initialize_pipeline_state("test-run-010", selection_mode="human_choice")
        assert state["selection_mode"] == "human_choice"

    def test_content_type_is_none(self):
        """Content type is not known until topic selection runs."""
        from src.agents.orchestrator import initialize_pipeline_state

        state = initialize_pipeline_state("test-run-011")
        assert state["content_type"] is None

    def test_all_optional_agent_outputs_are_none(self):
        """All agent output slots should start as None."""
        from src.agents.orchestrator import initialize_pipeline_state

        state = initialize_pipeline_state("test-run-012")

        none_keys = [
            "selected_topic",
            "analysis_brief",
            "draft_post",
            "writer_output",
            "humanized_post",
            "visual_asset",
            "visual_creator_output",
            "qc_result",
            "qc_output",
            "final_content",
            "critical_error",
            "error_stage",
        ]

        for key in none_keys:
            assert state[key] is None, (
                f"Expected state['{key}'] to be None, got {state[key]!r}"
            )

    def test_list_fields_default_to_empty(self):
        """All list fields should start empty."""
        from src.agents.orchestrator import initialize_pipeline_state

        state = initialize_pipeline_state("test-run-013")

        list_keys = [
            "trend_topics",
            "revision_history",
            "_rejected_topics",
            "meta_critique_history",
            "errors",
            "warnings",
            "capabilities_added",
        ]

        for key in list_keys:
            assert state[key] == [], (
                f"Expected state['{key}'] to be [], got {state[key]!r}"
            )

    def test_integer_counters_default_to_zero(self):
        """All numeric counters should start at zero."""
        from src.agents.orchestrator import initialize_pipeline_state

        state = initialize_pipeline_state("test-run-014")

        zero_keys = [
            "revision_count",
            "_reject_restart_count",
            "meta_iteration",
            "human_approval_reminder_count",
            "human_approval_escalation_level",
            "learnings_used_count",
            "code_generation_count",
        ]

        for key in zero_keys:
            assert state[key] == 0, (
                f"Expected state['{key}'] to be 0, got {state[key]!r}"
            )

    def test_meta_passed_defaults_to_false(self):
        from src.agents.orchestrator import initialize_pipeline_state

        state = initialize_pipeline_state("test-run-015")
        assert state["meta_passed"] is False

    def test_is_first_post_defaults_to_false(self):
        from src.agents.orchestrator import initialize_pipeline_state

        state = initialize_pipeline_state("test-run-016")
        assert state["is_first_post"] is False
