"""
LangGraph Orchestrator -- central state machine pipeline for content generation.

Ties all seven agents (Trend Scout, Analyzer, Writer, Meta-Evaluator,
Humanizer, Visual Creator, QC) together in a directed graph with
error-aware routing, timeout enforcement, and revision loops.

Flow
----
scout -> select_topic -> analyze -> write -> meta_evaluate
    -> (humanize | write) -> visualize -> qc -> learn
    -> (prepare_output | write | humanize | visualize | reset_for_restart
        | manual_review_queue)
    -> END

Key design decisions
--------------------
- **Pure functions**: Every node returns a dict; no direct state mutation.
- **Error routing**: ``@with_error_handling`` converts exceptions to
  ``{"critical_error": ...}`` and every edge checks for that key.
- **Timeouts**: ``@with_timeout`` wraps async calls; values are configurable
  via ``NODE_TIMEOUT_<NAME>`` environment variables.
- **ContentType propagation**: Determined at ``select_topic``, loaded into
  ``type_context``, and consumed by every downstream node.

Error philosophy: NO FALLBACKS -- FAIL FAST, retry with exponential backoff.
Supabase is THE ONLY database.

Architecture reference: ``architecture.md`` lines 11139-13176.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Dict, Any, List, Optional

from langgraph.graph import StateGraph, END

from src.models import (
    ContentType,
    PipelineState,
    TrendTopic,
    TrendScoutOutput,
    AnalysisBrief,
    DraftPost,
    WriterOutput,
    HumanizedPost,
    VisualAsset,
    VisualCreatorOutput,
    QCResult,
    QCOutput,
    HookStyle,
    get_hook_styles_for_type,
)
from src.exceptions import (
    TrendScoutError,
    TopicSelectionError,
    AnalyzerError,
    WriterError,
    VisualizerError,
    NodeTimeoutError,
)
from src.utils import utc_now, generate_id
from src.database import get_db

logger = logging.getLogger("Orchestrator")


# =============================================================================
# TIMEOUT CONFIGURATION
# =============================================================================


@dataclass
class NodeTimeoutsConfig:
    """
    Centralized timeout configuration for pipeline nodes.

    Each field default can be overridden by an environment variable
    ``NODE_TIMEOUT_<FIELD_NAME_UPPER>`` (e.g. ``NODE_TIMEOUT_SCOUT=180``).
    """

    # Agent-based nodes (longer timeouts for LLM calls)
    scout: int = 120           # 2 min -- multiple API calls
    analyze: int = 90          # 1.5 min -- deep analysis
    write: int = 60            # 1 min -- draft generation
    meta_evaluate: int = 45    # 45 sec -- evaluation
    humanize: int = 45         # 45 sec -- humanization
    visualize: int = 180       # 3 min -- image generation
    qc: int = 60               # 1 min -- quality check

    # Non-agent nodes (shorter timeouts)
    select_topic: int = 5
    prepare_output: int = 10
    manual_review_queue: int = 5
    handle_error: int = 5

    def __post_init__(self) -> None:
        """Load overrides from environment variables."""
        for field_name in self.__dataclass_fields__:
            env_var = f"NODE_TIMEOUT_{field_name.upper()}"
            env_value = os.environ.get(env_var)
            if env_value is not None:
                try:
                    setattr(self, field_name, int(env_value))
                except ValueError:
                    pass  # Keep default if env var is not a valid integer

    def get(self, node_name: str, default: int = 60) -> int:
        """Return timeout for *node_name*, falling back to *default*."""
        return getattr(self, node_name, default)


NODE_TIMEOUTS_CONFIG = NodeTimeoutsConfig()
NODE_TIMEOUTS: Dict[str, int] = {
    f: getattr(NODE_TIMEOUTS_CONFIG, f)
    for f in NODE_TIMEOUTS_CONFIG.__dataclass_fields__
}


# =============================================================================
# CONSTANTS
# =============================================================================

MAX_META_ITERATIONS: int = 3
"""Maximum number of meta-evaluation rewrite loops before forcing proceed."""

MAX_REVISIONS: int = 3
"""Default maximum QC-driven revisions before escalating to manual review."""

MAX_REJECT_RESTARTS: int = 2
"""Maximum times the pipeline will restart with a new topic before escalating."""

MAX_CRITIQUE_HISTORY: int = 10
"""Maximum critique entries retained in state to prevent memory bloat."""

DEFAULT_PASS_THRESHOLD: float = 7.0
"""Default QC pass threshold when type_context does not specify one."""


# =============================================================================
# DECORATORS
# =============================================================================


def with_error_handling(node_name: str = None):
    """Convert unhandled node exceptions into ``critical_error`` state updates.

    The LangGraph conditional edges check ``state.get("critical_error")`` and
    route to ``handle_error`` when present.  This decorator ensures that no
    exception silently kills the graph.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            name = node_name or func.__name__
            node_logger = logging.getLogger(f"Node.{name}")
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}"
                node_logger.error(
                    "[%s] Exception caught, routing to error handler: %s",
                    name,
                    error_msg,
                )
                return {
                    "critical_error": error_msg,
                    "error_stage": name,
                }

        return wrapper

    return decorator


def with_timeout(timeout_seconds: int = None, node_name: str = None):
    """Add ``asyncio.wait_for`` timeout to an async node function.

    Resolution order for the actual timeout value:
    1. Explicit *timeout_seconds* parameter.
    2. ``NODE_TIMEOUTS[node_name]`` (or func name with ``_node`` stripped).
    3. 60 seconds as a last resort.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            actual = timeout_seconds
            if actual is None:
                name = node_name or func.__name__.replace("_node", "")
                actual = NODE_TIMEOUTS.get(name, 60)
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=actual)
            except asyncio.TimeoutError:
                name = node_name or func.__name__
                raise NodeTimeoutError(name, actual)

        return wrapper

    return decorator


# =============================================================================
# LAZY AGENT ACCESSORS (singleton, initialised on first use)
# =============================================================================

_scout_agent = None
_analyzer_agent = None
_writer_agent = None
_meta_agent = None
_humanizer_agent = None
_visual_creator_agent = None
_qc_agent = None


async def _get_scout_agent():
    global _scout_agent
    if _scout_agent is None:
        from src.agents.trend_scout import create_trend_scout
        _scout_agent = await create_trend_scout()
    return _scout_agent


async def _get_analyzer_agent():
    global _analyzer_agent
    if _analyzer_agent is None:
        from src.agents.analyzer import create_analyzer
        _analyzer_agent = await create_analyzer()
    return _analyzer_agent


async def _get_writer_agent():
    global _writer_agent
    if _writer_agent is None:
        from src.agents.writer import create_writer
        _writer_agent = await create_writer()
    return _writer_agent


async def _get_meta_agent():
    global _meta_agent
    if _meta_agent is None:
        from src.meta_agent.evaluator import create_meta_evaluator
        _meta_agent = await create_meta_evaluator()
    return _meta_agent


async def _get_humanizer_agent():
    global _humanizer_agent
    if _humanizer_agent is None:
        from src.agents.humanizer import create_humanizer
        _humanizer_agent = await create_humanizer()
    return _humanizer_agent


async def _get_visual_creator_agent():
    global _visual_creator_agent
    if _visual_creator_agent is None:
        from src.agents.visual_creator import create_visual_creator
        _visual_creator_agent = await create_visual_creator()
    return _visual_creator_agent


async def _get_qc_agent():
    global _qc_agent
    if _qc_agent is None:
        from src.agents.qc import create_qc_agent
        _qc_agent = await create_qc_agent()
    return _qc_agent


# =============================================================================
# TYPE-SPECIFIC CONTEXT LOADER
# =============================================================================


def load_type_context(content_type: ContentType) -> Dict[str, Any]:
    """Load all type-specific configurations when ``ContentType`` is determined.

    The returned dict flows through the entire pipeline, informing extraction
    focus, template selection, tone calibration, visual format, and QC criteria.

    Architecture reference: ``architecture.md`` lines 11358-11471.
    """

    TYPE_CONTEXTS: Dict[ContentType, Dict[str, Any]] = {
        ContentType.ENTERPRISE_CASE: {
            # Analyzer config
            "extraction_focus": [
                "company", "industry", "problem", "solution",
                "metrics", "timeline", "lessons",
            ],
            "required_fields": ["company", "metrics", "problem_statement"],
            # Writer config
            "preferred_templates": ["METRICS_HERO", "LESSONS_LEARNED", "HOW_THEY_DID_IT"],
            "hook_styles": get_hook_styles_for_type(ContentType.ENTERPRISE_CASE),
            "cta_style": "credibility_expert",
            # Humanizer config
            "humanization_intensity": "medium",
            "tone_markers": ["analytical", "credible", "insider"],
            "avoid_markers": ["hype", "exclamation_heavy", "casual"],
            # Visual config
            "visual_formats": ["metrics_card", "architecture_diagram", "timeline_infographic"],
            "color_scheme": "corporate_blue",
            # QC config
            "extra_criteria": ["metrics_credibility"],
            "weight_adjustments": {"factual_accuracy": 1.3, "engagement_hook": 0.9},
            "pass_threshold": 7.2,
        },
        ContentType.PRIMARY_SOURCE: {
            "extraction_focus": [
                "authors", "thesis", "methodology", "findings",
                "implications", "counterintuitive",
            ],
            "required_fields": ["thesis", "key_findings", "authors"],
            "preferred_templates": ["RESEARCH_INSIGHT", "CONTRARIAN_TAKE", "FUTURE_PREDICTION"],
            "hook_styles": get_hook_styles_for_type(ContentType.PRIMARY_SOURCE),
            "cta_style": "intellectual_discourse",
            "humanization_intensity": "low",
            "tone_markers": ["thoughtful", "nuanced", "intellectual"],
            "avoid_markers": ["oversimplification", "clickbait", "absolutist"],
            "visual_formats": ["concept_diagram", "quote_card_elegant", "data_visualization"],
            "color_scheme": "academic_subtle",
            "extra_criteria": ["intellectual_depth"],
            "weight_adjustments": {"factual_accuracy": 1.4, "engagement_hook": 0.8},
            "pass_threshold": 7.5,
        },
        ContentType.AUTOMATION_CASE: {
            "extraction_focus": [
                "task_automated", "tools_used", "workflow_steps",
                "time_saved", "code_available",
            ],
            "required_fields": ["task_automated", "tools_used", "workflow_steps"],
            "preferred_templates": ["HOW_TO_GUIDE", "TOOL_STACK_REVEAL", "AUTOMATION_STORY"],
            "hook_styles": get_hook_styles_for_type(ContentType.AUTOMATION_CASE),
            "cta_style": "practical_action",
            "humanization_intensity": "high",
            "tone_markers": ["practical", "enthusiastic", "hands_on"],
            "avoid_markers": ["over_technical", "corporate_speak"],
            "visual_formats": ["workflow_diagram", "screenshot_annotated", "carousel_steps"],
            "color_scheme": "tech_vibrant",
            "extra_criteria": ["reproducibility"],
            "weight_adjustments": {"actionability": 1.3, "engagement_hook": 1.1},
            "pass_threshold": 7.0,
        },
        ContentType.COMMUNITY_CONTENT: {
            "extraction_focus": [
                "platform", "author_credibility", "key_insights",
                "engagement_signals", "code_examples",
            ],
            "required_fields": ["platform", "key_insights"],
            "preferred_templates": ["COMMUNITY_SPOTLIGHT", "DISCUSSION_SUMMARY", "PERSONAL_STORY"],
            "hook_styles": get_hook_styles_for_type(ContentType.COMMUNITY_CONTENT),
            "cta_style": "community_engagement",
            "humanization_intensity": "high",
            "tone_markers": ["conversational", "authentic", "curious"],
            "avoid_markers": ["formal", "corporate", "distant"],
            "visual_formats": ["quote_card_casual", "screenshot_highlighted", "meme_professional"],
            "color_scheme": "community_warm",
            "extra_criteria": ["community_authenticity"],
            "weight_adjustments": {"voice_match": 1.2, "engagement_hook": 1.2},
            "pass_threshold": 6.8,
        },
        ContentType.TOOL_RELEASE: {
            "extraction_focus": [
                "tool_name", "company", "key_features",
                "pricing", "demo_url", "competing_tools",
            ],
            "required_fields": ["tool_name", "key_features"],
            "preferred_templates": ["PRODUCT_LAUNCH", "TOOL_COMPARISON", "FIRST_LOOK"],
            "hook_styles": get_hook_styles_for_type(ContentType.TOOL_RELEASE),
            "cta_style": "evaluation_try",
            "humanization_intensity": "medium",
            "tone_markers": ["balanced", "evaluative", "practical"],
            "avoid_markers": ["promotional", "uncritical_hype", "sponsored_feel"],
            "visual_formats": ["product_screenshot", "feature_comparison_table", "demo_gif"],
            "color_scheme": "product_neutral",
            "extra_criteria": ["evaluation_balance"],
            "weight_adjustments": {"factual_accuracy": 1.2, "actionability": 1.1},
            "pass_threshold": 7.0,
        },
    }

    return TYPE_CONTEXTS.get(content_type, {})


# =============================================================================
# NODE IMPLEMENTATIONS
#
# RULES (architecture.md lines 11897-11940):
# 1. NO DIRECT STATE MUTATION -- always return dict updates.
# 2. Validate required inputs at start.
# 3. Use @with_error_handling and @with_timeout decorators.
# 4. Return {"critical_error": "..."} for unrecoverable failures.
# =============================================================================


@with_error_handling(node_name="scout")
@with_timeout(node_name="scout")
async def trend_scout_node(state: PipelineState) -> Dict[str, Any]:
    """Discover and score trending topics, classifying ``ContentType``."""

    scout = await _get_scout_agent()
    scout_output: TrendScoutOutput = await scout.run()

    if not scout_output.topics:
        raise TrendScoutError("Scout found no topics after filtering")

    if scout_output.top_pick is None:
        raise TrendScoutError("Scout failed to select top pick")

    return {
        "stage": "scouted",
        "trend_topics": scout_output.topics,
        "top_pick": scout_output.top_pick,
        "topics_by_type": scout_output.topics_by_type,
        "scout_statistics": {
            "total_sources_scanned": scout_output.total_sources_scanned,
            "topics_before_filter": scout_output.topics_before_filter,
            "topics_after_filter": scout_output.topics_after_filter,
            "exclusion_log": scout_output.exclusion_log,
        },
    }


@with_error_handling(node_name="select_topic")
@with_timeout(node_name="select_topic")
async def topic_selection_node(state: PipelineState) -> Dict[str, Any]:
    """Select a topic and lock ``ContentType`` + type context for the run.

    Selection modes:
    - ``auto_top_pick`` -- use the scout's top-rated topic.
    - ``type_balance``  -- favour underrepresented content types.
    - ``human_choice``  -- use a pre-selected topic or fall back to top pick.
    """

    if not state.get("trend_topics"):
        raise TopicSelectionError("No trend_topics in state")

    selection_mode = state.get("selection_mode", "auto_top_pick")

    if selection_mode == "auto_top_pick":
        selected = state.get("top_pick")
    elif selection_mode == "type_balance":
        selected = select_for_type_balance(state["trend_topics"])
    else:  # human_choice
        selected = state.get("selected_topic") or state.get("top_pick")

    if selected is None:
        raise TopicSelectionError(
            f"Failed to select topic with mode '{selection_mode}'"
        )

    content_type = selected.content_type
    type_context = load_type_context(content_type)

    return {
        "stage": "topic_selected",
        "selected_topic": selected,
        "content_type": content_type,
        "type_context": type_context,
    }


@with_error_handling(node_name="analyze")
@with_timeout(node_name="analyze")
async def analyzer_node(state: PipelineState) -> Dict[str, Any]:
    """Deep analysis with type-specific extraction configuration."""

    topic = state.get("selected_topic")
    content_type = state.get("content_type")
    type_context = state.get("type_context")

    if not all([topic, content_type, type_context]):
        raise AnalyzerError(
            "Analyzer requires selected_topic, content_type, and type_context"
        )

    extraction_focus = type_context.get("extraction_focus", [])
    required_fields = type_context.get("required_fields", [])

    analyzer = await _get_analyzer_agent()
    analysis_brief: AnalysisBrief = await analyzer.run(
        topic=topic,
        content_type=content_type,
        extraction_focus=extraction_focus,
        required_fields=required_fields,
    )

    return {
        "stage": "analyzed",
        "analysis_brief": analysis_brief,
        "extraction_data": analysis_brief.extraction_data,
    }


@with_error_handling(node_name="write")
@with_timeout(node_name="write")
async def writer_node(state: PipelineState) -> Dict[str, Any]:
    """Generate a draft using type-appropriate templates and hooks.

    Handles both first-draft creation and revision (from QC or Meta-Agent).
    """

    content_type = state.get("content_type")
    type_context = state.get("type_context")
    analysis = state.get("analysis_brief")

    if not all([content_type, type_context, analysis]):
        raise WriterError(
            "Writer requires content_type, type_context, and analysis_brief"
        )

    preferred_templates = type_context.get("preferred_templates", [])
    hook_styles = type_context.get("hook_styles", [])
    cta_style = type_context.get("cta_style", "general")

    # Determine revision instructions (if this is a rewrite pass)
    revision_instructions = None
    if state.get("current_revision_target") == "writer":
        qc_output = state.get("qc_output")
        if qc_output:
            revision_instructions = getattr(
                qc_output, "revision_instructions",
                getattr(qc_output.result, "feedback", None) if qc_output else None,
            )
    elif state.get("meta_passed") is False:
        meta_eval = state.get("meta_evaluation") or {}
        revision_instructions = meta_eval.get("suggestions", [])

    writer = await _get_writer_agent()
    writer_output: WriterOutput = await writer.run(
        analysis_brief=analysis,
        content_type=content_type,
        preferred_templates=preferred_templates,
        hook_styles=hook_styles,
        cta_style=cta_style,
        revision_instructions=revision_instructions,
    )

    return {
        "stage": "drafted",
        "draft_post": writer_output.primary_draft,
        "writer_output": writer_output,
        "template_used": writer_output.primary_draft.template_used,
        "hook_style_used": writer_output.primary_draft.hook_style.value
        if isinstance(writer_output.primary_draft.hook_style, HookStyle)
        else str(writer_output.primary_draft.hook_style),
        "visual_brief": writer_output.primary_draft.visual_brief,
        # Clear revision target after processing (State Machine Fix 4.3)
        "current_revision_target": None,
    }


# -------------------------------------------------------------------------
# META-AGENT SELF-EVALUATION NODE
# Integrated between Writer and Humanizer for quality iteration.
# -------------------------------------------------------------------------


@with_error_handling(node_name="meta_evaluate")
@with_timeout(node_name="meta_evaluate")
async def meta_evaluate_node(state: PipelineState) -> Dict[str, Any]:
    """Evaluate draft quality; decide to proceed or request rewrite.

    - Score >= threshold --> proceed to humanizer.
    - Score < threshold and iteration < max --> send back to writer.
    - Max iterations reached --> force proceed with a warning.
    """

    draft = state.get("draft_post")
    content_type = state.get("content_type")

    if not draft or not content_type:
        raise WriterError("meta_evaluate requires draft_post and content_type")

    iteration = state.get("meta_iteration", 0)

    # Force-pass when max iterations are exhausted
    if iteration >= MAX_META_ITERATIONS:
        return {
            "stage": "meta_evaluated",
            "meta_iteration": iteration,
            "meta_passed": True,
            "warnings": state.get("warnings", []) + [
                f"Meta-Agent: Max iterations ({MAX_META_ITERATIONS}) reached, "
                f"forcing proceed"
            ],
        }

    meta = await _get_meta_agent()
    evaluation = await meta.evaluate_draft(
        draft=draft,
        content_type=content_type,
    )

    # Type-specific threshold from context (falls back to global default)
    type_context = state.get("type_context") or {}
    threshold = type_context.get("pass_threshold", DEFAULT_PASS_THRESHOLD)

    score = getattr(evaluation, "score", 0.0)

    critique_entry: Dict[str, Any] = {
        "iteration": iteration,
        "score": score,
        "threshold": threshold,
        "feedback": getattr(evaluation, "feedback", ""),
        "suggestions": getattr(evaluation, "suggestions", []),
        "timestamp": utc_now().isoformat(),
    }
    critique_history = (
        state.get("meta_critique_history", []) + [critique_entry]
    )[-MAX_CRITIQUE_HISTORY:]

    evaluation_dict = (
        evaluation.__dict__
        if hasattr(evaluation, "__dict__")
        else {"score": score}
    )

    if score >= threshold:
        return {
            "stage": "meta_evaluated",
            "meta_evaluation": evaluation_dict,
            "meta_evaluation_score": score,
            "meta_iteration": iteration,
            "meta_passed": True,
            "meta_critique_history": critique_history,
        }

    return {
        "stage": "meta_needs_rewrite",
        "meta_evaluation": evaluation_dict,
        "meta_evaluation_score": score,
        "meta_iteration": iteration + 1,
        "meta_passed": False,
        "meta_critique_history": critique_history,
        "current_revision_target": "writer",
    }


@with_error_handling(node_name="humanize")
@with_timeout(node_name="humanize")
async def humanizer_node(state: PipelineState) -> Dict[str, Any]:
    """Humanise the draft with type-specific intensity and tone markers."""

    content_type = state.get("content_type")
    type_context = state.get("type_context") or {}
    draft = state.get("draft_post")

    if not draft or not content_type:
        raise WriterError("humanize requires draft_post and content_type")

    intensity = type_context.get("humanization_intensity", "medium")
    tone_markers = type_context.get("tone_markers", [])
    avoid_markers = type_context.get("avoid_markers", [])

    # Revision instructions from QC (if revision loop)
    revision_instructions = None
    if state.get("current_revision_target") == "humanizer":
        qc_output = state.get("qc_output")
        if qc_output:
            revision_instructions = getattr(
                qc_output, "revision_instructions",
                getattr(qc_output.result, "feedback", None) if qc_output else None,
            )

    humanizer = await _get_humanizer_agent()
    humanized_post: HumanizedPost = await humanizer.run(
        draft=draft,
        content_type=content_type,
        intensity=intensity,
        tone_markers=tone_markers,
        avoid_markers=avoid_markers,
        revision_instructions=revision_instructions,
    )

    return {
        "humanized_post": humanized_post,
        "humanization_intensity": intensity,
        "stage": "humanized",
        "current_revision_target": None,  # Clear after processing
    }


@with_error_handling(node_name="visualize")
@with_timeout(node_name="visualize")
async def visual_creator_node(state: PipelineState) -> Dict[str, Any]:
    """Generate visuals in a type-appropriate format and colour scheme."""

    content_type = state.get("content_type")
    type_context = state.get("type_context") or {}
    post = state.get("humanized_post")
    draft = state.get("draft_post")

    if not post or not draft or not content_type:
        raise VisualizerError(
            "visualize requires humanized_post, draft_post, and content_type"
        )

    visual_formats = type_context.get("visual_formats", ["single_image"])
    color_scheme = type_context.get("color_scheme", "brand_default")

    # Visual-specific revision instructions
    revision_instructions = None
    if state.get("current_revision_target") == "visual":
        qc_output = state.get("qc_output")
        if qc_output:
            revision_instructions = getattr(
                qc_output, "visual_revision_instructions", None
            )
            if revision_instructions:
                logger.info(
                    "[VISUAL] Processing revision with instructions: %s...",
                    str(revision_instructions)[:100],
                )

    visual_brief = state.get("visual_brief") or getattr(draft, "visual_brief", "")
    suggested_type = getattr(draft, "visual_type", "single_image")

    visual_creator = await _get_visual_creator_agent()
    visual_output: VisualCreatorOutput = await visual_creator.run(
        post=post,
        visual_brief=visual_brief,
        suggested_type=suggested_type,
        content_type=content_type,
        allowed_formats=visual_formats,
        color_scheme=color_scheme,
        revision_instructions=revision_instructions,
    )

    return {
        "visual_asset": visual_output.primary_asset,
        "visual_creator_output": visual_output,
        "visual_format_used": visual_output.format_selected,
        "stage": "visual_created",
        "current_revision_target": None,
    }


@with_error_handling(node_name="qc")
@with_timeout(node_name="qc")
async def qc_node(state: PipelineState) -> Dict[str, Any]:
    """Quality check with type-specific criteria and weight adjustments."""

    content_type = state.get("content_type")
    type_context = state.get("type_context") or {}

    if not content_type:
        raise ValueError("qc requires content_type in state")

    extra_criteria = type_context.get("extra_criteria", [])
    weight_adjustments = type_context.get("weight_adjustments", {})
    pass_threshold = type_context.get("pass_threshold", DEFAULT_PASS_THRESHOLD)

    qc = await _get_qc_agent()
    qc_output: QCOutput = await qc.run(
        humanized_post=state.get("humanized_post"),
        visual_asset=state.get("visual_asset"),
        draft_post=state.get("draft_post"),
        content_type=content_type,
        extra_criteria=extra_criteria,
        weight_adjustments=weight_adjustments,
        pass_threshold=pass_threshold,
    )

    revision_count = state.get("revision_count", 0)
    revision_history = list(state.get("revision_history", []))

    decision = qc_output.result.decision
    if decision.upper() != "PASS":
        revision_history.append({
            "revision_number": revision_count + 1,
            "decision": decision,
            "score": qc_output.result.total_score,
            "feedback": qc_output.result.feedback,
        })

    return {
        "qc_result": qc_output.result,
        "qc_output": qc_output,
        "type_specific_scores": qc_output.result.type_specific_feedback,
        "revision_count": revision_count + (1 if decision.upper() != "PASS" else 0),
        "revision_history": revision_history,
        "stage": "qc_completed",
    }


# -------------------------------------------------------------------------
# CONTINUOUS LEARNING NODE
# Executes AFTER every QC evaluation, BEFORE the routing decision.
# -------------------------------------------------------------------------


@with_error_handling(node_name="learn")
@with_timeout(timeout_seconds=30, node_name="learn")
async def post_evaluation_learning_node(state: PipelineState) -> Dict[str, Any]:
    """Extract learnings from every iteration regardless of pass/fail.

    The QC routing decision is computed here and stored in ``_qc_decision``
    so that ``route_after_learning`` can forward it without re-computing.
    """

    learn_logger = logging.getLogger("ContinuousLearning")

    # Compute QC decision for downstream routing
    qc_decision = _compute_qc_decision(state)

    # Attempt to extract learnings if a learning engine is available
    learnings = None
    try:
        learning_engine = state.get("learning_engine")
        if learning_engine is not None:
            meta_eval = state.get("meta_evaluation") or {}
            qc_result = state.get("qc_result")

            qc_score = (
                qc_result.total_score
                if qc_result and hasattr(qc_result, "total_score")
                else 0.0
            )

            learnings = await learning_engine.learn_from_iteration(
                post_id=state.get("run_id", "unknown"),
                content_type=state.get("content_type"),
                qc_score=qc_score,
                meta_feedback=meta_eval,
                qc_feedback=qc_result.feedback if qc_result else "",
            )

            learn_logger.info(
                "[LEARN] Post %s: extracted learnings, qc_decision=%s",
                state.get("run_id", "?"),
                qc_decision,
            )
    except Exception as exc:
        # Learning failures are non-critical -- log and continue
        learn_logger.warning(
            "[LEARN] Learning extraction failed (non-critical): %s", exc
        )

    return {
        "iteration_learnings": learnings,
        "_qc_decision": qc_decision,
    }


# -------------------------------------------------------------------------
# PREPARE OUTPUT / MANUAL REVIEW / ERROR HANDLER / RESTART
# -------------------------------------------------------------------------


@with_error_handling(node_name="prepare_output")
@with_timeout(node_name="prepare_output")
async def prepare_for_human_approval(state: PipelineState) -> Dict[str, Any]:
    """Build the final content package for human review / auto-approval."""

    humanized_post = state.get("humanized_post")
    if not humanized_post:
        return {"critical_error": "Missing humanized_post for final output"}

    visual_asset = state.get("visual_asset")
    qc_result = state.get("qc_result")
    selected_topic = state.get("selected_topic")

    final_content: Dict[str, Any] = {
        "post_text": humanized_post.humanized_text,
        "visual": visual_asset,
        # Metadata
        "content_type": (
            state["content_type"].value
            if state.get("content_type")
            else None
        ),
        "template_used": state.get("template_used"),
        "hook_style": state.get("hook_style_used"),
        "visual_format": state.get("visual_format_used"),
        # Scores
        "qc_score": qc_result.total_score if qc_result else None,
        "type_specific_scores": state.get("type_specific_scores"),
        # Source attribution
        "source_url": (
            selected_topic.primary_source_url if selected_topic else None
        ),
        "source_title": selected_topic.title if selected_topic else None,
        # Statistics
        "revision_count": state.get("revision_count", 0),
        "revision_history": state.get("revision_history", []),
    }

    return {
        "final_content": final_content,
        "human_approval_status": "pending",
        "stage": "ready_for_approval",
    }


@with_error_handling(node_name="manual_review_queue")
@with_timeout(node_name="manual_review_queue")
async def queue_for_manual_review(state: PipelineState) -> Dict[str, Any]:
    """Queue the post for manual human review.

    Used when maximum revisions are reached but quality remains below the
    pass threshold.  A human can approve, edit, or reject.
    """

    humanized_post = state.get("humanized_post")
    if not humanized_post:
        return {"critical_error": "Missing humanized_post for manual review"}

    qc_result = state.get("qc_result")

    return {
        "stage": "manual_review_required",
        "human_approval_status": "pending_manual_review",
        "warnings": state.get("warnings", []) + [
            f"Max revisions ({state.get('revision_count', 0)}) reached but "
            f"quality below threshold. Queued for manual review."
        ],
        "final_content": {
            "text": humanized_post.humanized_text,
            "visual": state.get("visual_asset"),
            "qc_score": qc_result.total_score if qc_result else None,
            "requires_human_decision": True,
        },
    }


async def error_handler_node(state: PipelineState) -> Dict[str, Any]:
    """Central error handler -- logs, persists to Supabase, and terminates."""

    err_logger = logging.getLogger("PipelineErrorHandler")

    critical_error = state.get("critical_error", "Unknown error")
    errors_list = state.get("errors", [])
    last_stage = state.get("stage", "unknown")

    selected_topic = state.get("selected_topic")
    topic_id = None
    if selected_topic is not None:
        topic_id = (
            selected_topic.get("id")
            if isinstance(selected_topic, dict)
            else getattr(selected_topic, "id", None)
        )

    error_context: Dict[str, Any] = {
        "critical_error": str(critical_error),
        "accumulated_errors": errors_list,
        "last_successful_stage": last_stage,
        "run_id": state.get("run_id"),
        "content_type": (
            state["content_type"].value
            if state.get("content_type") and hasattr(state["content_type"], "value")
            else str(state.get("content_type"))
        ),
        "topic_id": topic_id,
        "revision_count": state.get("revision_count", 0),
    }

    err_logger.error(
        "Pipeline failed at stage '%s': %s\nContext: %s",
        last_stage,
        critical_error,
        error_context,
    )

    # Persist error to Supabase for post-mortem analysis
    try:
        db = await get_db()
        await db.client.table("pipeline_errors").insert({
            "run_id": state.get("run_id"),
            "error_type": "CriticalError",
            "error_message": str(critical_error),
            "stage": last_stage,
            "context": error_context,
            "created_at": utc_now().isoformat(),
        }).execute()
    except Exception as db_exc:
        err_logger.warning("Failed to save error to database: %s", db_exc)

    return {
        "stage": "error",
        "final_content": {
            "status": "error",
            "critical_error": str(critical_error),
            "errors": errors_list,
            "last_successful_stage": last_stage,
            "context": error_context,
            "requires_human_attention": True,
        },
    }


@with_error_handling(node_name="reset_for_restart")
@with_timeout(timeout_seconds=5, node_name="reset_for_restart")
async def reset_for_restart_node(state: PipelineState) -> Dict[str, Any]:
    """Reset state for a fresh topic search after QC rejection.

    Increments ``_reject_restart_count`` (checked by routing to prevent
    infinite loops), clears topic-specific fields, and preserves the run ID
    and learning context.
    """

    current_restart_count = state.get("_reject_restart_count", 0) + 1

    selected_topic = state.get("selected_topic")
    rejected_id = None
    if selected_topic is not None:
        rejected_id = (
            selected_topic.get("id")
            if isinstance(selected_topic, dict)
            else getattr(selected_topic, "id", None)
        )

    logger.info(
        "[RESTART] Resetting for new topic search (restart #%d, rejected: %s)",
        current_restart_count,
        rejected_id or "unknown",
    )

    rejected_topics = list(state.get("_rejected_topics", []))
    if rejected_id:
        rejected_topics.append(rejected_id)

    return {
        "_reject_restart_count": current_restart_count,
        # Clear topic-related state
        "trend_topics": [],
        "selected_topic": None,
        "analysis_brief": None,
        "draft_post": None,
        "humanized_post": None,
        "visual_asset": None,
        "visual_brief": None,
        # Reset revision counters
        "revision_count": 0,
        "meta_iteration": 0,
        "meta_critique_history": [],
        "current_revision_target": None,
        # Clear QC state
        "qc_output": None,
        "qc_result": None,
        "meta_passed": False,
        "meta_evaluation": None,
        # Track rejected topics
        "_rejected_topics": rejected_topics,
        "stage": "restarting",
    }


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================


def route_after_meta_evaluate(state: PipelineState) -> str:
    """Route after meta evaluation: ``humanize`` if passed, ``write`` if not."""
    if state.get("meta_passed", False):
        return "humanize"
    return "write"


def route_after_learning(state: PipelineState) -> str:
    """Forward the pre-computed QC decision stored by the learning node."""
    return state.get("_qc_decision", "pass")


def _compute_qc_decision(state: PipelineState) -> str:
    """Compute the QC routing decision from current state.

    Routes:
    - ``pass``               -- quality passed, proceed to output.
    - ``revise_writer``      -- send back to writer for revision.
    - ``revise_humanizer``   -- send back to humanizer.
    - ``revise_visual``      -- send back to visual creator.
    - ``reject_restart``     -- quality too low, restart with new topic.
    - ``max_revisions_force``-- max revisions reached, manual review.
    """

    qc_output = state.get("qc_output")
    qc_result = state.get("qc_result")
    content_type = state.get("content_type")

    if not all([qc_output, qc_result, content_type]):
        logger.error(
            "Missing required state for QC routing: qc_output=%s, "
            "qc_result=%s, content_type=%s",
            bool(qc_output),
            bool(qc_result),
            content_type,
        )
        return "pass"  # Degrade gracefully in routing (error already logged)

    revision_count = state.get("revision_count", 0)
    reject_restart_count = state.get("_reject_restart_count", 0)

    # Resolve the QC score
    score: float = 0.0
    if isinstance(qc_result, dict):
        score = qc_result.get("total_score", 0.0)
    else:
        score = getattr(qc_result, "total_score", 0.0)

    # Resolve the QC decision string
    decision_raw: str = ""
    if isinstance(qc_result, dict):
        decision_raw = qc_result.get("decision", "")
    else:
        decision_raw = getattr(qc_result, "decision", "")
    decision_upper = decision_raw.upper()

    # Type-specific pass threshold
    type_context = state.get("type_context") or {}
    pass_threshold = type_context.get("pass_threshold", DEFAULT_PASS_THRESHOLD)

    # PASS case
    if decision_upper == "PASS" or score >= pass_threshold:
        return "pass"

    # Max revisions reached -- escalate to manual review
    if revision_count >= MAX_REVISIONS:
        return "max_revisions_force"

    # REJECT case
    if decision_upper == "REJECT":
        if reject_restart_count >= MAX_REJECT_RESTARTS:
            logger.warning(
                "Reached max reject_restart limit (%d). "
                "Routing to manual review instead of another restart.",
                MAX_REJECT_RESTARTS,
            )
            return "max_revisions_force"
        return "reject_restart"

    # REVISE case -- determine which agent should handle it
    return determine_revision_target(qc_output)


def determine_revision_target(qc_output) -> str:
    """Analyse lowest QC scores to determine which agent should revise.

    Criteria-to-agent mapping (State Machine Fix 4.2):
    - ``humanness``, ``tone_match``, ``authenticity`` -> ``revise_humanizer``
    - ``visual_match``, ``visual_quality`` -> ``revise_visual``
    - Everything else -> ``revise_writer``
    """

    if isinstance(qc_output, dict):
        result = qc_output.get("result", {})
        scores = (
            result.get("scores", {})
            if isinstance(result, dict)
            else {}
        )
    else:
        result = getattr(qc_output, "result", None)
        scores = getattr(result, "scores", {}) if result else {}

    if not scores:
        return "revise_writer"  # Default when no score data available

    humanizer_criteria = {"humanness", "tone_match", "authenticity"}
    visual_criteria = {"visual_match", "visual_quality"}

    lowest_score = float("inf")
    lowest_criterion: Optional[str] = None

    for criterion, score_val in scores.items():
        if isinstance(score_val, (int, float)) and score_val < lowest_score:
            lowest_score = score_val
            lowest_criterion = criterion

    if lowest_criterion in humanizer_criteria:
        return "revise_humanizer"
    if lowest_criterion in visual_criteria:
        return "revise_visual"
    return "revise_writer"


def select_for_type_balance(topics: List[TrendTopic]) -> TrendTopic:
    """Select the topic that best balances content-type distribution.

    Favours underrepresented types by boosting their score relative to the
    target distribution.
    """

    target_distribution: Dict[ContentType, float] = {
        ContentType.ENTERPRISE_CASE: 0.25,
        ContentType.PRIMARY_SOURCE: 0.20,
        ContentType.AUTOMATION_CASE: 0.25,
        ContentType.COMMUNITY_CONTENT: 0.15,
        ContentType.TOOL_RELEASE: 0.15,
    }

    # In production this would query the database for recent type counts.
    # Here we use a stub that assumes uniform distribution.
    recent_type_counts: Dict[ContentType, float] = {
        ct: 0.2 for ct in ContentType
    }

    scored: List[tuple] = []
    for topic in topics:
        target = target_distribution.get(topic.content_type, 0.2)
        actual = recent_type_counts.get(topic.content_type, 0.0)
        scarcity_boost = (target - actual) * 2
        combined = topic.score + scarcity_boost
        scored.append((topic, combined))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


# =============================================================================
# WORKFLOW GRAPH CONSTRUCTION
# =============================================================================


def create_content_pipeline() -> Any:
    """Build and compile the LangGraph state machine.

    Returns a compiled ``StateGraph`` ready for ``await pipeline.ainvoke(state)``.
    """

    workflow = StateGraph(PipelineState)

    # ---- Add nodes --------------------------------------------------------
    workflow.add_node("scout", trend_scout_node)
    workflow.add_node("select_topic", topic_selection_node)
    workflow.add_node("analyze", analyzer_node)
    workflow.add_node("write", writer_node)
    workflow.add_node("meta_evaluate", meta_evaluate_node)
    workflow.add_node("humanize", humanizer_node)
    workflow.add_node("visualize", visual_creator_node)
    workflow.add_node("qc", qc_node)
    workflow.add_node("learn", post_evaluation_learning_node)
    workflow.add_node("prepare_output", prepare_for_human_approval)
    workflow.add_node("manual_review_queue", queue_for_manual_review)
    workflow.add_node("handle_error", error_handler_node)
    workflow.add_node("reset_for_restart", reset_for_restart_node)

    # ---- Entry point ------------------------------------------------------
    workflow.set_entry_point("scout")

    # ---- Error-aware routing helper ---------------------------------------

    def make_error_aware_router(next_node: str):
        """Return ``handle_error`` if ``critical_error`` is set, else *next_node*."""

        def router(state: PipelineState) -> str:
            if state.get("critical_error"):
                return "handle_error"
            return next_node

        return router

    # ---- Main flow edges (each with error checking) -----------------------

    workflow.add_conditional_edges(
        "scout",
        make_error_aware_router("select_topic"),
        {"select_topic": "select_topic", "handle_error": "handle_error"},
    )

    workflow.add_conditional_edges(
        "select_topic",
        make_error_aware_router("analyze"),
        {"analyze": "analyze", "handle_error": "handle_error"},
    )

    workflow.add_conditional_edges(
        "analyze",
        make_error_aware_router("write"),
        {"write": "write", "handle_error": "handle_error"},
    )

    # write -> meta_evaluate
    workflow.add_conditional_edges(
        "write",
        make_error_aware_router("meta_evaluate"),
        {"meta_evaluate": "meta_evaluate", "handle_error": "handle_error"},
    )

    # meta_evaluate -> (humanize | write | handle_error)
    def route_after_meta_evaluate_with_error(state: PipelineState) -> str:
        if state.get("critical_error"):
            return "handle_error"
        return route_after_meta_evaluate(state)

    workflow.add_conditional_edges(
        "meta_evaluate",
        route_after_meta_evaluate_with_error,
        {
            "humanize": "humanize",
            "write": "write",
            "handle_error": "handle_error",
        },
    )

    # humanize -> visualize
    workflow.add_conditional_edges(
        "humanize",
        make_error_aware_router("visualize"),
        {"visualize": "visualize", "handle_error": "handle_error"},
    )

    # visualize -> qc
    workflow.add_conditional_edges(
        "visualize",
        make_error_aware_router("qc"),
        {"qc": "qc", "handle_error": "handle_error"},
    )

    # qc -> learn (always go through learning first)
    workflow.add_conditional_edges(
        "qc",
        make_error_aware_router("learn"),
        {"learn": "learn", "handle_error": "handle_error"},
    )

    # learn -> (prepare_output | write | humanize | visualize |
    #           reset_for_restart | manual_review_queue | handle_error)
    def route_after_learning_with_error(state: PipelineState) -> str:
        if state.get("critical_error"):
            return "handle_error"
        decision = route_after_learning(state)
        return decision

    workflow.add_conditional_edges(
        "learn",
        route_after_learning_with_error,
        {
            "pass": "prepare_output",
            "revise_writer": "write",
            "revise_humanizer": "humanize",
            "revise_visual": "visualize",
            "reject_restart": "reset_for_restart",
            "max_revisions_force": "manual_review_queue",
            "handle_error": "handle_error",
        },
    )

    # reset_for_restart -> scout (unconditional)
    workflow.add_edge("reset_for_restart", "scout")

    # Terminal edges
    workflow.add_conditional_edges(
        "prepare_output",
        make_error_aware_router(END),
        {END: END, "handle_error": "handle_error"},
    )

    workflow.add_conditional_edges(
        "manual_review_queue",
        make_error_aware_router(END),
        {END: END, "handle_error": "handle_error"},
    )

    # Error handler always terminates
    workflow.add_edge("handle_error", END)

    return workflow.compile()


# =============================================================================
# PIPELINE STATE INITIALISATION
# =============================================================================


def initialize_pipeline_state(
    run_id: str,
    selection_mode: str = "auto_top_pick",
) -> PipelineState:
    """Create a fully-defaulted ``PipelineState`` for a new pipeline run."""

    return PipelineState(
        # Run tracking
        run_id=run_id,
        run_timestamp=utc_now(),
        stage="initialized",
        # Content type (set after topic selection)
        content_type=None,
        type_context=None,
        # Scout
        trend_topics=[],
        top_pick=None,
        topics_by_type={},
        scout_statistics=None,
        # Topic selection
        selected_topic=None,
        selection_mode=selection_mode,
        # Analyzer
        analysis_brief=None,
        extraction_data=None,
        # Writer
        draft_post=None,
        writer_output=None,
        template_used=None,
        hook_style_used=None,
        # Humanizer
        humanized_post=None,
        humanization_intensity=None,
        # Visual Creator
        visual_brief=None,
        visual_asset=None,
        visual_creator_output=None,
        visual_format_used=None,
        # QC
        qc_result=None,
        qc_output=None,
        type_specific_scores=None,
        # Revision tracking
        revision_count=0,
        revision_history=[],
        current_revision_target=None,
        # Reject / restart tracking
        _reject_restart_count=0,
        _rejected_topics=[],
        _qc_decision=None,
        # Meta-agent self-evaluation
        meta_evaluation=None,
        meta_evaluation_score=None,
        meta_iteration=0,
        meta_passed=False,
        meta_critique_history=[],
        # Final output
        final_content=None,
        human_approval_status=None,
        human_approval_requested_at=None,
        human_approval_reminder_count=0,
        human_approval_escalation_level=0,
        # Error handling
        critical_error=None,
        error_stage=None,
        errors=[],
        warnings=[],
        # Continuous learning (injected by run_pipeline)
        iteration_learnings=None,
        learnings_used_count=0,
        is_first_post=False,
        # Self-modifying code (injected by run_pipeline)
        self_mod_result=None,
        capabilities_added=[],
        code_generation_count=0,
    )


# =============================================================================
# MAIN EXECUTION
# =============================================================================


async def run_pipeline(
    selection_mode: str = "auto_top_pick",
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute the full content generation pipeline.

    Args:
        selection_mode: Topic selection strategy
            (``auto_top_pick`` / ``type_balance`` / ``human_choice``).
        project_root: Root directory of the project (for self-modification
            engine).  Defaults to ``Path.cwd()``.

    Returns:
        A dict with ``run_id``, ``status``, ``content``, and ``statistics``.
    """

    run_logger = logging.getLogger("Pipeline")
    project_root = project_root or Path.cwd()

    # ---- Database --------------------------------------------------------
    db = await get_db()

    # ---- Check first post ------------------------------------------------
    is_first_post = False
    try:
        post_count = await db.get_total_post_count()
        is_first_post = post_count == 0
    except Exception:
        run_logger.warning("Could not determine post count; assuming not first post")

    if is_first_post:
        run_logger.info(
            "[FIRST POST] This is the first post -- will bootstrap with "
            "best practices"
        )

    # ---- Initialise state ------------------------------------------------
    run_id = generate_id()
    initial_state = initialize_pipeline_state(run_id, selection_mode)
    initial_state["is_first_post"] = is_first_post

    run_logger.info(
        "[PIPELINE] Starting run %s (mode=%s, first_post=%s)",
        run_id,
        selection_mode,
        is_first_post,
    )

    # ---- Compile and run -------------------------------------------------
    pipeline = create_content_pipeline()
    final_state = await pipeline.ainvoke(initial_state)

    # ---- Collect statistics ----------------------------------------------
    learning_stats: Dict[str, Any] = {}
    if final_state.get("iteration_learnings") is not None:
        learnings = final_state["iteration_learnings"]
        learning_stats = {
            "new_learnings": len(getattr(learnings, "new_learnings", [])),
            "confirmed_learnings": len(
                getattr(learnings, "confirmed_learnings", [])
            ),
            "contradicted_learnings": len(
                getattr(learnings, "contradicted_learnings", [])
            ),
        }

    content_type_val = None
    if final_state.get("content_type"):
        ct = final_state["content_type"]
        content_type_val = ct.value if hasattr(ct, "value") else str(ct)

    qc_score = None
    qc_result = final_state.get("qc_result")
    if qc_result is not None:
        qc_score = (
            qc_result.total_score
            if hasattr(qc_result, "total_score")
            else qc_result.get("total_score")
            if isinstance(qc_result, dict)
            else None
        )

    return {
        "run_id": run_id,
        "status": (
            "success" if not final_state.get("critical_error") else "error"
        ),
        "content": final_state.get("final_content"),
        "statistics": {
            "content_type": content_type_val,
            "revision_count": final_state.get("revision_count", 0),
            "qc_score": qc_score,
            "learnings_used": final_state.get("learnings_used_count", 0),
            "learning_stats": learning_stats,
            "is_first_post": is_first_post,
        },
    }


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Configuration
    "NodeTimeoutsConfig",
    "NODE_TIMEOUTS_CONFIG",
    "NODE_TIMEOUTS",
    # Decorators
    "with_error_handling",
    "with_timeout",
    # Type context
    "load_type_context",
    # Nodes
    "trend_scout_node",
    "topic_selection_node",
    "analyzer_node",
    "writer_node",
    "meta_evaluate_node",
    "humanizer_node",
    "visual_creator_node",
    "qc_node",
    "post_evaluation_learning_node",
    "prepare_for_human_approval",
    "queue_for_manual_review",
    "error_handler_node",
    "reset_for_restart_node",
    # Routing
    "route_after_meta_evaluate",
    "route_after_learning",
    "determine_revision_target",
    "select_for_type_balance",
    # Workflow
    "create_content_pipeline",
    "initialize_pipeline_state",
    "run_pipeline",
]
