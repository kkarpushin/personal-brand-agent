"""
Centralized shared data types for the LinkedIn Super Agent system.

This module is THE single source of truth for all data models used across the
pipeline.  Every agent reads from and writes to ``PipelineState`` using the
types defined here.

Hierarchy of types
------------------
- **Enums**: ``ContentType``, ``HookStyle``, ``VisualType``
- **Trend Scout models**: ``SuggestedAngle``, ``TopPickSummary``, five metadata
  dataclasses, ``TrendTopic``, ``TrendScoutOutput``
- **Analyzer models**: ``TypeSpecificExtraction``, ``AnalysisBrief``
- **Writer models**: ``DraftPost``, ``WriterOutput``
- **Humanizer models**: ``HumanizedPost``
- **Visual Creator models**: ``VisualAsset``, ``VisualCreatorOutput``
- **QC models**: ``QCResult``, ``QCOutput``
- **Analytics models**: ``PostMetricsSnapshot``
- **Orchestrator state**: ``PipelineState`` (``TypedDict``)
- **Constants & helpers**: ``CONTENT_TYPE_HOOK_STYLES``, validation helpers

References
----------
- ``architecture.md`` lines 3182-3192  (ContentType enum)
- ``architecture.md`` lines 4727-5236  (Trend Scout output schemas)
- ``architecture.md`` lines 4760-4910  (HookStyle, VisualType, hook-style maps)
- ``architecture.md`` lines 6466-6528  (AnalysisBrief)
- ``architecture.md`` lines 7644-7704  (DraftPost, WriterOutput)
- ``architecture.md`` lines 8079-8148  (HumanizedPost)
- ``architecture.md`` lines 10025-10083 (VisualAsset, VisualCreatorOutput)
- ``architecture.md`` lines 11049-11135 (QCResult, QCOutput)
- ``architecture.md`` lines 11224-11352 (PipelineState TypedDict)
- ``architecture.md`` lines 13337-13381 (PostMetricsSnapshot)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
    Union,
)

from src.exceptions import MetadataTypeMismatchError
from src.utils import utc_now


# =============================================================================
# ENUMS
# =============================================================================


class ContentType(Enum):
    """
    Five distinct content verticals requiring different scoring, extraction,
    template selection, and QC criteria.

    Authoritative definition: ``architecture.md`` lines 3182-3192.
    """

    ENTERPRISE_CASE = "enterprise_case"
    PRIMARY_SOURCE = "primary_source"
    AUTOMATION_CASE = "automation_case"
    COMMUNITY_CONTENT = "community_content"
    TOOL_RELEASE = "tool_release"


class HookStyle(Enum):
    """
    Unified hook styles for all content types.

    Each ``ContentType`` maps to a specific subset of allowed hook styles via
    ``CONTENT_TYPE_HOOK_STYLES``.

    Authoritative definition: ``architecture.md`` lines 4771-4831.
    """

    # -- ENTERPRISE_CASE hook styles ----------------------------------------
    METRICS = "metrics"
    LESSONS_LEARNED = "lessons_learned"
    PROBLEM_SOLUTION = "problem_solution"

    # -- PRIMARY_SOURCE hook styles -----------------------------------------
    CONTRARIAN = "contrarian"
    QUESTION = "question"
    SURPRISING_STAT = "surprising_stat"
    SIMPLIFIED_EXPLAINER = "simplified_explainer"
    DEBATE_STARTER = "debate_starter"

    # -- AUTOMATION_CASE hook styles ----------------------------------------
    HOW_TO = "how_to"
    TIME_SAVED = "time_saved"
    BEFORE_AFTER = "before_after"
    RESULTS_STORY = "results_story"
    TOOL_COMPARISON = "tool_comparison"

    # -- COMMUNITY_CONTENT hook styles --------------------------------------
    RELATABLE = "relatable"
    COMMUNITY_REFERENCE = "community_reference"
    PERSONAL = "personal"
    CURATED_INSIGHTS = "curated_insights"
    HOT_TAKE_RESPONSE = "hot_take_response"
    PRACTITIONER_WISDOM = "practitioner_wisdom"

    # -- TOOL_RELEASE hook styles -------------------------------------------
    NEWS_BREAKING = "news_breaking"
    FEATURE_HIGHLIGHT = "feature_highlight"
    COMPARISON = "comparison"
    FIRST_LOOK = "first_look"
    IMPLICATIONS = "implications"

    # -- Universal / additional hook styles (Fix 5.1) -----------------------
    STORY = "story"
    INDUSTRY_IMPACT = "industry_impact"
    EXPERT_QUOTE = "expert_quote"
    TREND_ANALYSIS = "trend_analysis"


class VisualType(str, Enum):
    """
    Types of visual content that can accompany a LinkedIn post.

    Inherits from ``str`` so that ``VisualType.DATA_VIZ == "data_viz"``
    evaluates to ``True``, making JSON serialization painless.

    Authoritative definition: ``architecture.md`` lines 4833-4857.
    """

    DATA_VIZ = "data_viz"
    DIAGRAM = "diagram"
    SCREENSHOT = "screenshot"
    QUOTE_CARD = "quote_card"
    AUTHOR_PHOTO = "author_photo"
    CAROUSEL = "carousel"
    INTERFACE_VISUAL = "interface_visual"
    BEFORE_AFTER = "before_after"

    @classmethod
    def for_content_type(cls, content_type: ContentType) -> List[VisualType]:
        """Return the recommended visual types for *content_type*."""

        mapping: Dict[ContentType, List[VisualType]] = {
            ContentType.ENTERPRISE_CASE: [
                cls.DATA_VIZ,
                cls.QUOTE_CARD,
                cls.SCREENSHOT,
            ],
            ContentType.PRIMARY_SOURCE: [
                cls.DATA_VIZ,
                cls.DIAGRAM,
                cls.QUOTE_CARD,
            ],
            ContentType.AUTOMATION_CASE: [
                cls.DIAGRAM,
                cls.SCREENSHOT,
                cls.BEFORE_AFTER,
            ],
            ContentType.COMMUNITY_CONTENT: [
                cls.AUTHOR_PHOTO,
                cls.QUOTE_CARD,
                cls.CAROUSEL,
            ],
            ContentType.TOOL_RELEASE: [
                cls.SCREENSHOT,
                cls.INTERFACE_VISUAL,
                cls.DIAGRAM,
            ],
        }
        return mapping.get(content_type, [cls.AUTHOR_PHOTO])


# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================

VALID_ENTERPRISE_SCALES: List[str] = [
    "SMB",
    "Mid-Market",
    "Enterprise",
    "Fortune 500",
]

VALID_SOURCE_TYPES: List[str] = [
    "research_paper",
    "think_tank_report",
    "expert_essay",
    "whitepaper",
]

VALID_REPRODUCIBILITY_LEVELS: List[str] = ["high", "medium", "low"]

VALID_PLATFORMS: List[str] = [
    "YouTube",
    "Reddit",
    "HackerNews",
    "Dev.to",
    "Twitter",
    "Medium",
    "Substack",
]

VALID_CONTENT_FORMATS: List[str] = [
    "video",
    "post",
    "comment",
    "thread",
    "article",
    "newsletter",
]

VALID_AUTHOR_CREDIBILITY: List[str] = [
    "verified_expert",
    "practitioner",
    "unknown",
]

VALID_RELEASE_TYPES: List[str] = [
    "new_product",
    "major_update",
    "api_release",
    "open_source",
]


# =============================================================================
# TREND SCOUT - HELPER DATACLASSES
# =============================================================================


@dataclass
class SuggestedAngle:
    """A potential hook / angle for writing about a topic."""

    angle_text: str
    angle_type: str  # e.g. "lessons_learned", "metrics_story", "how_to"
    hook_templates: List[str]
    content_type_fit: float  # 0.0 - 1.0


@dataclass
class TopPickSummary:
    """Summary attached to the daily top-pick topic."""

    why_chosen: str
    key_takeaways: List[str]  # exactly 3 takeaways
    who_should_care: str


# =============================================================================
# TREND SCOUT - TYPE-SPECIFIC METADATA
#
# Each class uses a Literal ``type`` field as a discriminator so that runtime
# type-checking and JSON deserialization can validate that
# ``metadata.type == topic.content_type.value``.
# =============================================================================


@dataclass
class EnterpriseCaseMetadata:
    """Metadata for enterprise AI implementation case studies."""

    # Discriminator - MUST equal ``ContentType.ENTERPRISE_CASE.value``
    type: Literal["enterprise_case"] = "enterprise_case"

    # Required fields
    company: str = ""
    industry: str = ""
    scale: str = ""  # one of VALID_ENTERPRISE_SCALES
    problem_domain: str = ""
    ai_technologies: List[str] = field(default_factory=list)
    metrics: Dict[str, str] = field(default_factory=dict)
    roi_mentioned: bool = False
    architecture_available: bool = False
    lessons_learned: List[str] = field(default_factory=list)

    # Optional fields
    implementation_timeline: Optional[str] = None
    team_size: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate field values against allowed options."""
        if self.scale and self.scale not in VALID_ENTERPRISE_SCALES:
            raise ValueError(
                f"EnterpriseCaseMetadata.scale must be one of "
                f"{VALID_ENTERPRISE_SCALES}, got '{self.scale}'"
            )
        if not self.company:
            raise ValueError(
                "EnterpriseCaseMetadata.company is required and cannot be empty"
            )
        if not self.industry:
            raise ValueError(
                "EnterpriseCaseMetadata.industry is required and cannot be empty"
            )


@dataclass
class PrimarySourceMetadata:
    """Metadata for research papers, reports, and primary sources."""

    # Discriminator
    type: Literal["primary_source"] = "primary_source"

    # Required fields
    authors: List[str] = field(default_factory=list)
    organization: str = ""
    source_type: str = ""  # one of VALID_SOURCE_TYPES
    publication_venue: str = ""
    key_hypothesis: str = ""
    methodology_summary: str = ""
    code_available: bool = False

    # Optional fields
    counterintuitive_finding: Optional[str] = None
    citations_count: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate field values against allowed options."""
        if self.source_type and self.source_type not in VALID_SOURCE_TYPES:
            raise ValueError(
                f"PrimarySourceMetadata.source_type must be one of "
                f"{VALID_SOURCE_TYPES}, got '{self.source_type}'"
            )
        if not self.authors:
            raise ValueError(
                "PrimarySourceMetadata.authors is required and cannot be empty"
            )
        if not self.organization:
            raise ValueError(
                "PrimarySourceMetadata.organization is required and cannot be empty"
            )
        if not self.key_hypothesis:
            raise ValueError(
                "PrimarySourceMetadata.key_hypothesis is required and cannot be empty"
            )


@dataclass
class AutomationCaseMetadata:
    """Metadata for AI automation and agent case studies."""

    # Discriminator
    type: Literal["automation_case"] = "automation_case"

    # Required fields
    agent_type: str = ""
    workflow_components: List[str] = field(default_factory=list)
    integrations: List[str] = field(default_factory=list)
    use_case_domain: str = ""
    metrics: Dict[str, str] = field(default_factory=dict)
    reproducibility: str = ""  # one of VALID_REPRODUCIBILITY_LEVELS
    code_available: bool = False

    # Optional fields
    time_saved: Optional[str] = None
    cost_saved: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate field values against allowed options."""
        if self.reproducibility and self.reproducibility not in VALID_REPRODUCIBILITY_LEVELS:
            raise ValueError(
                f"AutomationCaseMetadata.reproducibility must be one of "
                f"{VALID_REPRODUCIBILITY_LEVELS}, got '{self.reproducibility}'"
            )
        if not self.agent_type:
            raise ValueError(
                "AutomationCaseMetadata.agent_type is required and cannot be empty"
            )
        if not self.workflow_components:
            raise ValueError(
                "AutomationCaseMetadata.workflow_components is required and cannot be empty"
            )
        if not self.use_case_domain:
            raise ValueError(
                "AutomationCaseMetadata.use_case_domain is required and cannot be empty"
            )


@dataclass
class CommunityContentMetadata:
    """Metadata for community discussions, videos, and threads."""

    # Discriminator
    type: Literal["community_content"] = "community_content"

    # Required fields
    platform: str = ""  # one of VALID_PLATFORMS
    format: str = ""  # one of VALID_CONTENT_FORMATS
    engagement_metrics: Dict[str, int] = field(default_factory=dict)
    author_credibility: str = ""  # one of VALID_AUTHOR_CREDIBILITY
    has_code_examples: bool = False
    has_demo: bool = False
    key_contributors: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate field values against allowed options."""
        if self.platform and self.platform not in VALID_PLATFORMS:
            raise ValueError(
                f"CommunityContentMetadata.platform must be one of "
                f"{VALID_PLATFORMS}, got '{self.platform}'"
            )
        if self.format and self.format not in VALID_CONTENT_FORMATS:
            raise ValueError(
                f"CommunityContentMetadata.format must be one of "
                f"{VALID_CONTENT_FORMATS}, got '{self.format}'"
            )
        if self.author_credibility and self.author_credibility not in VALID_AUTHOR_CREDIBILITY:
            raise ValueError(
                f"CommunityContentMetadata.author_credibility must be one of "
                f"{VALID_AUTHOR_CREDIBILITY}, got '{self.author_credibility}'"
            )


@dataclass
class ToolReleaseMetadata:
    """Metadata for new tool and product releases."""

    # Discriminator
    type: Literal["tool_release"] = "tool_release"

    # Required fields
    tool_name: str = ""
    company: str = ""
    release_date: str = ""
    release_type: str = ""  # one of VALID_RELEASE_TYPES
    demo_url: Optional[str] = None
    api_available: bool = False
    pricing_model: Optional[str] = None
    key_features: List[str] = field(default_factory=list)
    competing_tools: List[str] = field(default_factory=list)
    early_reviews: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate field values against allowed options."""
        if self.release_type and self.release_type not in VALID_RELEASE_TYPES:
            raise ValueError(
                f"ToolReleaseMetadata.release_type must be one of "
                f"{VALID_RELEASE_TYPES}, got '{self.release_type}'"
            )
        if not self.tool_name:
            raise ValueError(
                "ToolReleaseMetadata.tool_name is required and cannot be empty"
            )
        if not self.company:
            raise ValueError(
                "ToolReleaseMetadata.company is required and cannot be empty"
            )
        if not self.key_features:
            raise ValueError(
                "ToolReleaseMetadata.key_features is required and cannot be empty"
            )


# Discriminated union of all five metadata types.
TopicMetadata = Union[
    EnterpriseCaseMetadata,
    PrimarySourceMetadata,
    AutomationCaseMetadata,
    CommunityContentMetadata,
    ToolReleaseMetadata,
]


# =============================================================================
# TREND SCOUT - MAIN OUTPUT SCHEMAS
# =============================================================================


@dataclass
class TrendTopic:
    """
    Enhanced topic schema with content-type classification, type-specific
    metadata, and quality-focused scoring.

    Authoritative definition: ``architecture.md`` lines 5159-5211.
    """

    # Identification
    id: str
    title: str
    summary: str  # 2-3 sentences

    # Classification
    content_type: ContentType

    # Sources
    sources: List[str]  # URLs
    primary_source_url: str

    # Scoring
    score: float  # 0-10 quality-adjusted
    score_breakdown: Dict[str, float]
    quality_signals_matched: List[str]

    # Content for downstream agents
    suggested_angles: List[SuggestedAngle]
    related_topics: List[str]
    raw_content: str  # full text for Analyzer

    # Type-specific metadata
    metadata: TopicMetadata

    # Analysis guidance
    analysis_format: str  # guides extraction strategy
    recommended_post_format: str  # insight_thread / contrarian / tutorial / etc.
    recommended_visual_type: str  # data_viz / architecture / screenshot / etc.

    # Top-pick designation
    is_top_pick: bool = False
    top_pick_summary: Optional[TopPickSummary] = None

    # Timestamps
    discovered_at: datetime = field(default_factory=utc_now)
    source_published_at: Optional[datetime] = None

    def __repr__(self) -> str:
        top_marker = "*" if self.is_top_pick else ""
        return (
            f"TrendTopic({top_marker}id='{self.id}', "
            f"title='{self.title[:40]}...', "
            f"type={self.content_type.value}, "
            f"score={self.score:.1f})"
        )


@dataclass
class TrendScoutOutput:
    """Complete output from the Trend Scout agent."""

    run_id: str
    run_timestamp: datetime

    # All scored topics
    topics: List[TrendTopic]

    # Top pick of the day
    top_pick: TrendTopic

    # Statistics
    total_sources_scanned: int
    topics_before_filter: int
    topics_after_filter: int
    exclusion_log: List[Dict[str, Any]]

    # Breakdown by content type
    topics_by_type: Dict[str, int]


# =============================================================================
# ANALYZER - OUTPUT SCHEMAS
# =============================================================================


@dataclass
class TypeSpecificExtraction:
    """
    Extracted fields specific to a ``ContentType``.

    This is a *generic* container used inside ``AnalysisBrief``.  For each
    content type the ``extracted_fields`` dictionary holds the raw key-value
    data while ``required_fields_present`` / ``missing_fields`` track
    completeness.
    """

    content_type: ContentType
    extracted_fields: Dict[str, Any]
    required_fields_present: List[str]
    missing_fields: List[str]
    extraction_confidence: float  # 0.0 - 1.0


@dataclass
class AnalysisBrief:
    """
    Structured brief produced by the Analyzer agent for the Writer.

    Authoritative definition: ``architecture.md`` lines 6466-6528.
    """

    topic_id: str
    content_type: ContentType
    title: str
    key_findings: List[str]
    main_argument: str
    suggested_angle: str
    hook_materials: Dict[str, Any]
    extraction_data: TypeSpecificExtraction
    controversy_level: str  # low / medium / high / spicy
    complexity_level: str  # simplify_heavily / simplify_slightly / keep_technical
    target_audience: str
    recommended_post_format: str
    recommended_visual_type: str


# =============================================================================
# WRITER - OUTPUT SCHEMAS
# =============================================================================


@dataclass
class DraftPost:
    """
    First draft produced by the Writer agent.

    Authoritative definition: ``architecture.md`` lines 7644-7684.
    """

    hook: str  # first line - must fit in 210 chars
    body: str
    cta: str
    hashtags: List[str]
    full_text: str  # combined, formatted
    template_used: str
    hook_style: HookStyle
    content_type: ContentType
    character_count: int
    visual_brief: str  # description for image generation
    visual_type: str  # data_viz / diagram / screenshot / quote_card
    key_terms: List[str]  # for hashtag optimisation


@dataclass
class WriterOutput:
    """Complete output from the Writer agent."""

    primary_draft: DraftPost
    alternatives: List[DraftPost]
    template_category: str
    generation_metadata: Dict[str, Any]


# =============================================================================
# HUMANIZER - OUTPUT SCHEMA
# =============================================================================


@dataclass
class HumanizedPost:
    """
    Post text after humanisation pass.

    Authoritative definition: ``architecture.md`` lines 8079-8148.
    """

    original_text: str
    humanized_text: str
    changes_made: List[str]
    humanization_intensity: str  # low / medium / high
    ai_detection_score_before: Optional[float] = None  # 0.0-1.0
    ai_detection_score_after: Optional[float] = None  # 0.0-1.0


# =============================================================================
# VISUAL CREATOR - OUTPUT SCHEMAS
# =============================================================================


@dataclass
class VisualAsset:
    """
    A generated visual asset for a LinkedIn post.

    Authoritative definition: ``architecture.md`` lines 10025-10067.
    """

    asset_type: str  # single_image / carousel / document
    file_path: Optional[str] = None
    url: Optional[str] = None
    prompt_used: str = ""
    generation_model: str = ""
    width: int = 0
    height: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualCreatorOutput:
    """Complete output from the Visual Creator agent."""

    primary_asset: VisualAsset
    alternatives: List[VisualAsset]
    visual_brief_used: str
    format_selected: str


# =============================================================================
# QC - OUTPUT SCHEMAS
# =============================================================================


@dataclass
class QCResult:
    """
    Quality-control evaluation result.

    Authoritative definition: ``architecture.md`` lines 11049-11104.
    """

    total_score: float
    scores: Dict[str, float]  # criterion -> score
    decision: str  # PASS / REVISE_WRITER / REVISE_HUMANIZER / REVISE_VISUAL / REJECT
    feedback: str
    revision_target: Optional[str] = None  # "writer" / "humanizer" / "visual"
    type_specific_feedback: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QCOutput:
    """Complete QC output with decision and routing information."""

    result: QCResult
    scoring_weights_used: Dict[str, float]
    type_adjustments_applied: Dict[str, Any]


# =============================================================================
# ANALYTICS - SNAPSHOT SCHEMA
# =============================================================================


@dataclass
class PostMetricsSnapshot:
    """
    Single snapshot of post metrics at a point in time.

    Authoritative definition: ``architecture.md`` lines 13337-13381.
    """

    post_id: str
    likes: int = 0
    comments: int = 0
    reposts: int = 0

    # ANALYTICS FIX 7.1: LinkedIn reaction types (not just total likes)
    reactions_by_type: Dict[str, int] = field(default_factory=lambda: {
        "LIKE": 0,
        "CELEBRATE": 0,
        "SUPPORT": 0,
        "LOVE": 0,
        "INSIGHTFUL": 0,
        "FUNNY": 0,
    })

    impressions: Optional[int] = None
    clicks: Optional[int] = None
    engagement_rate: Optional[float] = None
    likes_velocity: Optional[float] = None  # likes per minute since last snapshot
    minutes_after_post: int = 0
    collected_at: datetime = field(default_factory=utc_now)

    # Scheduled checkpoint support (Fix for #27)
    scheduled_checkpoint: Optional[int] = None  # e.g., 15, 30, 60, 180, 1440
    collection_drift_seconds: Optional[int] = None


@dataclass
class PostPerformance:
    """Complete performance record for a published post.

    Includes QC metadata for feedback loop:
    1. Correlation analysis: QC score vs actual performance
    2. Threshold calibration: Adjust QC thresholds based on real data
    3. Criterion weight optimization: Which criteria predict success?

    Authoritative definition: ``architecture.md`` lines 13406-13493.
    """

    post_id: str
    linkedin_url: str
    published_at: datetime

    # Content metadata (from generation)
    content_type: ContentType
    hook_style: HookStyle
    template_used: str
    visual_type: str
    has_author_photo: bool
    topic_summary: str

    # QC metadata (for feedback loop)
    qc_score: float
    qc_criterion_scores: Dict[str, float]
    revision_count: int
    auto_approved: bool
    meta_evaluation_score: Optional[float]
    threshold_used: float

    # Lineage
    pipeline_run_id: str
    topic_id: str

    # A/B Testing support
    experiment_id: Optional[str] = None
    experiment_variant: Optional[str] = None

    # Metrics snapshots
    snapshots: List[PostMetricsSnapshot] = field(default_factory=list)

    # Key milestones
    metrics_15min: Optional[PostMetricsSnapshot] = None
    metrics_30min: Optional[PostMetricsSnapshot] = None
    metrics_1hour: Optional[PostMetricsSnapshot] = None
    metrics_24hour: Optional[PostMetricsSnapshot] = None
    metrics_final: Optional[PostMetricsSnapshot] = None  # 48h

    # Calculated scores
    golden_hour_score: float = 0.0
    final_score: float = 0.0
    percentile_rank: float = 0.0

    # Comparisons
    vs_average: float = 1.0
    vs_content_type_avg: float = 1.0
    vs_same_day_time_avg: float = 1.0


@dataclass
class AnalyticsInsight:
    """Actionable insight derived from analytics.

    Authoritative definition: ``architecture.md`` lines 13496-13511.
    """

    insight_type: str  # content_type_performance / timing_pattern / visual_impact
    description: str
    confidence: float  # 0-1
    sample_size: int
    recommendation: str
    affected_component: str  # trend_scout / writer / visual_creator / scheduler
    parameter_to_adjust: Optional[str] = None
    suggested_value: Optional[Any] = None


# =============================================================================
# CONTENT TYPE <-> HOOK STYLE MAPPING (single source of truth)
# =============================================================================


CONTENT_TYPE_HOOK_STYLES: Dict[ContentType, List[HookStyle]] = {
    ContentType.ENTERPRISE_CASE: [
        HookStyle.METRICS,
        HookStyle.LESSONS_LEARNED,
        HookStyle.PROBLEM_SOLUTION,
    ],
    ContentType.PRIMARY_SOURCE: [
        HookStyle.CONTRARIAN,
        HookStyle.QUESTION,
        HookStyle.SURPRISING_STAT,
        HookStyle.SIMPLIFIED_EXPLAINER,
        HookStyle.DEBATE_STARTER,
    ],
    ContentType.AUTOMATION_CASE: [
        HookStyle.HOW_TO,
        HookStyle.TIME_SAVED,
        HookStyle.BEFORE_AFTER,
        HookStyle.RESULTS_STORY,
        HookStyle.TOOL_COMPARISON,
    ],
    ContentType.COMMUNITY_CONTENT: [
        HookStyle.RELATABLE,
        HookStyle.COMMUNITY_REFERENCE,
        HookStyle.PERSONAL,
        HookStyle.CURATED_INSIGHTS,
        HookStyle.HOT_TAKE_RESPONSE,
        HookStyle.PRACTITIONER_WISDOM,
    ],
    ContentType.TOOL_RELEASE: [
        HookStyle.NEWS_BREAKING,
        HookStyle.FEATURE_HIGHLIGHT,
        HookStyle.COMPARISON,
        HookStyle.FIRST_LOOK,
        HookStyle.IMPLICATIONS,
    ],
}


CONTENT_TYPE_TO_METADATA_TYPE: Dict[ContentType, str] = {
    ContentType.ENTERPRISE_CASE: "enterprise_case",
    ContentType.PRIMARY_SOURCE: "primary_source",
    ContentType.AUTOMATION_CASE: "automation_case",
    ContentType.COMMUNITY_CONTENT: "community_content",
    ContentType.TOOL_RELEASE: "tool_release",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_hook_styles_for_type(content_type: ContentType) -> List[HookStyle]:
    """Return the allowed ``HookStyle`` values for *content_type*."""

    return CONTENT_TYPE_HOOK_STYLES.get(content_type, [])


def validate_hook_style(hook_style: HookStyle, content_type: ContentType) -> bool:
    """Return ``True`` if *hook_style* is allowed for *content_type*."""

    return hook_style in CONTENT_TYPE_HOOK_STYLES.get(content_type, [])


def validate_topic_metadata(topic: TrendTopic) -> bool:
    """
    Validate that ``topic.metadata.type`` matches ``topic.content_type``.

    This check should run:
    1. After Trend Scout creates a ``TrendTopic``
    2. Before passing to the Analyzer
    3. During JSON deserialization

    Raises:
        MetadataTypeMismatchError: If types do not match.

    Returns:
        ``True`` when validation succeeds.
    """

    expected_type = CONTENT_TYPE_TO_METADATA_TYPE.get(topic.content_type)
    actual_type = getattr(topic.metadata, "type", None)

    if actual_type is None:
        raise MetadataTypeMismatchError(
            f"Metadata for topic '{topic.id}' has no 'type' discriminator field"
        )

    if actual_type != expected_type:
        raise MetadataTypeMismatchError(
            f"Topic '{topic.id}' has content_type={topic.content_type.value} "
            f"but metadata.type='{actual_type}' (expected '{expected_type}')"
        )

    return True


# =============================================================================
# PIPELINE STATE (LangGraph TypedDict)
#
# This is THE main state object that flows through the entire LangGraph
# pipeline.  Every agent reads from and writes to this dict.
#
# Authoritative definition: architecture.md lines 11224-11352.
# =============================================================================


class PipelineState(TypedDict, total=False):
    """
    Enhanced pipeline state with ``ContentType`` awareness flowing through
    all stages.

    ``total=False`` marks every key as optional so that the state can be
    incrementally populated as it flows through the graph.
    """

    # -----------------------------------------------------------------
    # RUN TRACKING
    # -----------------------------------------------------------------
    run_id: str
    run_timestamp: datetime
    stage: str  # current stage name for debugging / monitoring

    # -----------------------------------------------------------------
    # CONTENT TYPE CONTEXT (propagates through entire pipeline)
    # -----------------------------------------------------------------
    content_type: Optional[ContentType]
    type_context: Optional[Dict[str, Any]]

    # -----------------------------------------------------------------
    # SCOUT STAGE
    # -----------------------------------------------------------------
    trend_topics: List[TrendTopic]
    top_pick: Optional[TrendTopic]
    topics_by_type: Dict[str, int]
    scout_statistics: Optional[Dict[str, Any]]

    # -----------------------------------------------------------------
    # TOPIC SELECTION
    # -----------------------------------------------------------------
    selected_topic: Optional[TrendTopic]
    selection_mode: str  # "auto_top_pick" / "human_choice" / "type_balance"

    # -----------------------------------------------------------------
    # ANALYZER STAGE
    # -----------------------------------------------------------------
    analysis_brief: Optional[AnalysisBrief]
    extraction_data: Optional[TypeSpecificExtraction]

    # -----------------------------------------------------------------
    # WRITER STAGE
    # -----------------------------------------------------------------
    draft_post: Optional[DraftPost]
    writer_output: Optional[WriterOutput]
    template_used: Optional[str]
    hook_style_used: Optional[str]

    # -----------------------------------------------------------------
    # HUMANIZER STAGE
    # -----------------------------------------------------------------
    humanized_post: Optional[HumanizedPost]
    humanization_intensity: Optional[str]  # low / medium / high

    # -----------------------------------------------------------------
    # VISUAL CREATOR STAGE
    # -----------------------------------------------------------------
    visual_brief: Optional[str]
    visual_asset: Optional[VisualAsset]
    visual_creator_output: Optional[VisualCreatorOutput]
    visual_format_used: Optional[str]

    # -----------------------------------------------------------------
    # QC STAGE
    # -----------------------------------------------------------------
    qc_result: Optional[QCResult]
    qc_output: Optional[QCOutput]
    type_specific_scores: Optional[Dict[str, float]]

    # -----------------------------------------------------------------
    # REVISION TRACKING
    # -----------------------------------------------------------------
    revision_count: int
    revision_history: List[Dict[str, Any]]
    current_revision_target: Optional[str]  # "writer" / "humanizer" / "visual"

    # -----------------------------------------------------------------
    # REJECT / RESTART TRACKING
    # -----------------------------------------------------------------
    _reject_restart_count: int
    _rejected_topics: List[str]  # IDs of topics rejected by QC
    _qc_decision: Optional[str]  # PASS / REVISE_WRITER / REVISE_HUMANIZER / REJECT

    # -----------------------------------------------------------------
    # META-AGENT SELF-EVALUATION LOOP
    # -----------------------------------------------------------------
    meta_evaluation: Optional[Dict[str, Any]]
    meta_evaluation_score: Optional[float]
    meta_iteration: int
    meta_passed: bool
    meta_critique_history: List[Dict[str, Any]]

    # -----------------------------------------------------------------
    # FINAL OUTPUT
    # -----------------------------------------------------------------
    final_content: Optional[Dict[str, Any]]
    human_approval_status: Optional[str]
    human_approval_requested_at: Optional[datetime]
    human_approval_reminder_count: int
    human_approval_escalation_level: int

    # -----------------------------------------------------------------
    # ERROR HANDLING
    # -----------------------------------------------------------------
    critical_error: Optional[str]
    error_stage: Optional[str]
    errors: List[str]
    warnings: List[str]

    # -----------------------------------------------------------------
    # CONTINUOUS LEARNING ENGINE
    # -----------------------------------------------------------------
    learning_engine: Optional[Any]  # Injected by run_pipeline
    iteration_learnings: Optional[Any]
    learnings_used_count: int
    is_first_post: bool

    # -----------------------------------------------------------------
    # SELF-MODIFYING CODE ENGINE
    # -----------------------------------------------------------------
    self_mod_result: Optional[Any]
    capabilities_added: List[str]
    code_generation_count: int


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Enums
    "ContentType",
    "HookStyle",
    "VisualType",
    # Trend Scout
    "SuggestedAngle",
    "TopPickSummary",
    "EnterpriseCaseMetadata",
    "PrimarySourceMetadata",
    "AutomationCaseMetadata",
    "CommunityContentMetadata",
    "ToolReleaseMetadata",
    "TopicMetadata",
    "TrendTopic",
    "TrendScoutOutput",
    # Analyzer
    "TypeSpecificExtraction",
    "AnalysisBrief",
    # Writer
    "DraftPost",
    "WriterOutput",
    # Humanizer
    "HumanizedPost",
    # Visual Creator
    "VisualAsset",
    "VisualCreatorOutput",
    # QC
    "QCResult",
    "QCOutput",
    # Analytics
    "PostMetricsSnapshot",
    "PostPerformance",
    "AnalyticsInsight",
    # Pipeline State
    "PipelineState",
    # Constants
    "CONTENT_TYPE_HOOK_STYLES",
    "CONTENT_TYPE_TO_METADATA_TYPE",
    "VALID_ENTERPRISE_SCALES",
    "VALID_SOURCE_TYPES",
    "VALID_REPRODUCIBILITY_LEVELS",
    "VALID_PLATFORMS",
    "VALID_CONTENT_FORMATS",
    "VALID_AUTHOR_CREDIBILITY",
    "VALID_RELEASE_TYPES",
    # Helpers
    "get_hook_styles_for_type",
    "validate_hook_style",
    "validate_topic_metadata",
]
