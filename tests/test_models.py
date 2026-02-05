"""
Tests for src.models -- the centralized shared data types module.

Covers:
    - Enum correctness (ContentType, HookStyle, VisualType)
    - VisualType.for_content_type mapping
    - Dataclass creation and __post_init__ validation for all metadata types
    - Helper dataclasses (SuggestedAngle, TopPickSummary, DraftPost, QCResult, etc.)
    - PipelineState TypedDict usage
    - Helper functions (get_hook_styles_for_type, validate_hook_style, validate_topic_metadata)
"""

from datetime import datetime, timezone

import pytest

from src.exceptions import MetadataTypeMismatchError
from src.models import (
    CONTENT_TYPE_HOOK_STYLES,
    VALID_AUTHOR_CREDIBILITY,
    VALID_CONTENT_FORMATS,
    VALID_ENTERPRISE_SCALES,
    VALID_PLATFORMS,
    VALID_RELEASE_TYPES,
    VALID_REPRODUCIBILITY_LEVELS,
    VALID_SOURCE_TYPES,
    AnalysisBrief,
    AutomationCaseMetadata,
    CommunityContentMetadata,
    ContentType,
    DraftPost,
    EnterpriseCaseMetadata,
    HookStyle,
    HumanizedPost,
    PipelineState,
    PostMetricsSnapshot,
    PrimarySourceMetadata,
    QCOutput,
    QCResult,
    SuggestedAngle,
    ToolReleaseMetadata,
    TopPickSummary,
    TrendScoutOutput,
    TrendTopic,
    TypeSpecificExtraction,
    VisualAsset,
    VisualCreatorOutput,
    VisualType,
    WriterOutput,
    get_hook_styles_for_type,
    validate_hook_style,
    validate_topic_metadata,
)


# =========================================================================
# ContentType Enum
# =========================================================================


def test_content_type_has_exactly_five_values():
    """ContentType must have exactly 5 members."""
    assert len(ContentType) == 5


def test_content_type_values():
    """Each ContentType member maps to the expected string value."""
    expected = {
        "ENTERPRISE_CASE": "enterprise_case",
        "PRIMARY_SOURCE": "primary_source",
        "AUTOMATION_CASE": "automation_case",
        "COMMUNITY_CONTENT": "community_content",
        "TOOL_RELEASE": "tool_release",
    }
    for name, value in expected.items():
        assert ContentType[name].value == value


def test_content_type_lookup_by_value():
    """ContentType can be looked up by its string value."""
    assert ContentType("enterprise_case") is ContentType.ENTERPRISE_CASE
    assert ContentType("tool_release") is ContentType.TOOL_RELEASE


def test_content_type_invalid_value_raises():
    """Looking up an invalid value raises ValueError."""
    with pytest.raises(ValueError):
        ContentType("nonexistent")


# =========================================================================
# HookStyle Enum
# =========================================================================


def test_hook_style_exists_and_has_members():
    """HookStyle enum must exist and have at least 20 members."""
    assert len(HookStyle) >= 20


def test_hook_style_expected_values():
    """Spot-check a representative subset of HookStyle members."""
    expected_names = [
        "METRICS",
        "LESSONS_LEARNED",
        "CONTRARIAN",
        "HOW_TO",
        "TIME_SAVED",
        "BEFORE_AFTER",
        "RELATABLE",
        "NEWS_BREAKING",
        "STORY",
        "INDUSTRY_IMPACT",
        "EXPERT_QUOTE",
        "TREND_ANALYSIS",
    ]
    for name in expected_names:
        assert hasattr(HookStyle, name), f"HookStyle.{name} is missing"


def test_hook_style_values_are_snake_case():
    """All HookStyle values should be lowercase snake_case strings."""
    for member in HookStyle:
        assert member.value == member.value.lower()
        assert " " not in member.value


# =========================================================================
# VisualType Enum
# =========================================================================


def test_visual_type_is_str_enum():
    """VisualType inherits from str so members compare equal to strings."""
    assert isinstance(VisualType.DATA_VIZ, str)
    assert VisualType.DATA_VIZ == "data_viz"


def test_visual_type_members():
    """VisualType must contain the expected members."""
    expected = [
        "DATA_VIZ",
        "DIAGRAM",
        "SCREENSHOT",
        "QUOTE_CARD",
        "AUTHOR_PHOTO",
        "CAROUSEL",
        "INTERFACE_VISUAL",
        "BEFORE_AFTER",
    ]
    for name in expected:
        assert hasattr(VisualType, name), f"VisualType.{name} is missing"
    assert len(VisualType) == len(expected)


def test_visual_type_for_content_type_enterprise_case():
    """Enterprise case should recommend DATA_VIZ, QUOTE_CARD, SCREENSHOT."""
    result = VisualType.for_content_type(ContentType.ENTERPRISE_CASE)
    assert result == [VisualType.DATA_VIZ, VisualType.QUOTE_CARD, VisualType.SCREENSHOT]


def test_visual_type_for_content_type_primary_source():
    """Primary source should recommend DATA_VIZ, DIAGRAM, QUOTE_CARD."""
    result = VisualType.for_content_type(ContentType.PRIMARY_SOURCE)
    assert result == [VisualType.DATA_VIZ, VisualType.DIAGRAM, VisualType.QUOTE_CARD]


def test_visual_type_for_content_type_automation_case():
    """Automation case should recommend DIAGRAM, SCREENSHOT, BEFORE_AFTER."""
    result = VisualType.for_content_type(ContentType.AUTOMATION_CASE)
    assert result == [VisualType.DIAGRAM, VisualType.SCREENSHOT, VisualType.BEFORE_AFTER]


def test_visual_type_for_content_type_community_content():
    """Community content should recommend AUTHOR_PHOTO, QUOTE_CARD, CAROUSEL."""
    result = VisualType.for_content_type(ContentType.COMMUNITY_CONTENT)
    assert result == [VisualType.AUTHOR_PHOTO, VisualType.QUOTE_CARD, VisualType.CAROUSEL]


def test_visual_type_for_content_type_tool_release():
    """Tool release should recommend SCREENSHOT, INTERFACE_VISUAL, DIAGRAM."""
    result = VisualType.for_content_type(ContentType.TOOL_RELEASE)
    assert result == [VisualType.SCREENSHOT, VisualType.INTERFACE_VISUAL, VisualType.DIAGRAM]


def test_visual_type_for_content_type_covers_all():
    """Every ContentType must have an entry in the for_content_type mapping."""
    for ct in ContentType:
        result = VisualType.for_content_type(ct)
        assert isinstance(result, list)
        assert len(result) > 0, f"No visual types returned for {ct}"


# =========================================================================
# SuggestedAngle
# =========================================================================


def test_suggested_angle_creation():
    """SuggestedAngle can be created with all required fields."""
    angle = SuggestedAngle(
        angle_text="Focus on ROI metrics",
        angle_type="metrics_story",
        hook_templates=["Template A", "Template B"],
        content_type_fit=0.85,
    )
    assert angle.angle_text == "Focus on ROI metrics"
    assert angle.angle_type == "metrics_story"
    assert len(angle.hook_templates) == 2
    assert angle.content_type_fit == 0.85


# =========================================================================
# TopPickSummary
# =========================================================================


def test_top_pick_summary_creation():
    """TopPickSummary can be created with all required fields."""
    summary = TopPickSummary(
        why_chosen="Highest relevance score",
        key_takeaways=["Takeaway 1", "Takeaway 2", "Takeaway 3"],
        who_should_care="AI engineers and product managers",
    )
    assert summary.why_chosen == "Highest relevance score"
    assert len(summary.key_takeaways) == 3
    assert summary.who_should_care == "AI engineers and product managers"


# =========================================================================
# EnterpriseCaseMetadata
# =========================================================================


def test_enterprise_case_metadata_valid_creation():
    """EnterpriseCaseMetadata can be created with valid values."""
    meta = EnterpriseCaseMetadata(
        company="Acme Corp",
        industry="FinTech",
        scale="Enterprise",
        problem_domain="fraud detection",
        ai_technologies=["GPT-4", "LangChain"],
        metrics={"accuracy": "99.2%"},
        roi_mentioned=True,
    )
    assert meta.type == "enterprise_case"
    assert meta.company == "Acme Corp"
    assert meta.industry == "FinTech"
    assert meta.scale == "Enterprise"
    assert meta.roi_mentioned is True


def test_enterprise_case_metadata_all_valid_scales():
    """All VALID_ENTERPRISE_SCALES should be accepted."""
    for scale in VALID_ENTERPRISE_SCALES:
        meta = EnterpriseCaseMetadata(
            company="TestCo",
            industry="Tech",
            scale=scale,
        )
        assert meta.scale == scale


def test_enterprise_case_metadata_invalid_scale_raises():
    """Invalid scale value must raise ValueError."""
    with pytest.raises(ValueError, match="scale must be one of"):
        EnterpriseCaseMetadata(
            company="Acme",
            industry="Tech",
            scale="Startup",
        )


def test_enterprise_case_metadata_empty_company_raises():
    """Empty company string must raise ValueError."""
    with pytest.raises(ValueError, match="company is required"):
        EnterpriseCaseMetadata(
            company="",
            industry="Tech",
            scale="SMB",
        )


def test_enterprise_case_metadata_empty_industry_raises():
    """Empty industry string must raise ValueError."""
    with pytest.raises(ValueError, match="industry is required"):
        EnterpriseCaseMetadata(
            company="Acme",
            industry="",
            scale="SMB",
        )


def test_enterprise_case_metadata_defaults():
    """Optional and default fields should have correct defaults."""
    meta = EnterpriseCaseMetadata(
        company="Acme",
        industry="Tech",
        scale="SMB",
    )
    assert meta.ai_technologies == []
    assert meta.metrics == {}
    assert meta.roi_mentioned is False
    assert meta.architecture_available is False
    assert meta.lessons_learned == []
    assert meta.implementation_timeline is None
    assert meta.team_size is None


# =========================================================================
# PrimarySourceMetadata
# =========================================================================


def test_primary_source_metadata_valid_creation():
    """PrimarySourceMetadata can be created with valid values."""
    meta = PrimarySourceMetadata(
        authors=["Smith, J.", "Doe, A."],
        organization="MIT",
        source_type="research_paper",
        publication_venue="NeurIPS 2024",
        key_hypothesis="Transformers scale better with data",
    )
    assert meta.type == "primary_source"
    assert len(meta.authors) == 2
    assert meta.organization == "MIT"
    assert meta.source_type == "research_paper"
    assert meta.key_hypothesis == "Transformers scale better with data"


def test_primary_source_metadata_all_valid_source_types():
    """All VALID_SOURCE_TYPES should be accepted."""
    for source_type in VALID_SOURCE_TYPES:
        meta = PrimarySourceMetadata(
            authors=["Author"],
            organization="Org",
            source_type=source_type,
            key_hypothesis="Hypothesis",
        )
        assert meta.source_type == source_type


def test_primary_source_metadata_invalid_source_type_raises():
    """Invalid source_type must raise ValueError."""
    with pytest.raises(ValueError, match="source_type must be one of"):
        PrimarySourceMetadata(
            authors=["Author"],
            organization="Org",
            source_type="blog_post",
            key_hypothesis="Something",
        )


def test_primary_source_metadata_empty_authors_raises():
    """Empty authors list must raise ValueError."""
    with pytest.raises(ValueError, match="authors is required"):
        PrimarySourceMetadata(
            authors=[],
            organization="Org",
            source_type="research_paper",
            key_hypothesis="Something",
        )


def test_primary_source_metadata_empty_organization_raises():
    """Empty organization must raise ValueError."""
    with pytest.raises(ValueError, match="organization is required"):
        PrimarySourceMetadata(
            authors=["Author"],
            organization="",
            source_type="research_paper",
            key_hypothesis="Something",
        )


def test_primary_source_metadata_empty_key_hypothesis_raises():
    """Empty key_hypothesis must raise ValueError."""
    with pytest.raises(ValueError, match="key_hypothesis is required"):
        PrimarySourceMetadata(
            authors=["Author"],
            organization="Org",
            source_type="research_paper",
            key_hypothesis="",
        )


def test_primary_source_metadata_defaults():
    """Optional fields should have correct defaults."""
    meta = PrimarySourceMetadata(
        authors=["Author"],
        organization="Org",
        key_hypothesis="Something",
    )
    assert meta.code_available is False
    assert meta.counterintuitive_finding is None
    assert meta.citations_count is None


# =========================================================================
# AutomationCaseMetadata
# =========================================================================


def test_automation_case_metadata_valid_creation():
    """AutomationCaseMetadata can be created with valid values."""
    meta = AutomationCaseMetadata(
        agent_type="RAG pipeline",
        workflow_components=["retriever", "generator"],
        integrations=["Slack", "Jira"],
        use_case_domain="customer support",
        reproducibility="high",
    )
    assert meta.type == "automation_case"
    assert meta.agent_type == "RAG pipeline"
    assert meta.reproducibility == "high"


def test_automation_case_metadata_all_valid_reproducibility():
    """All VALID_REPRODUCIBILITY_LEVELS should be accepted."""
    for level in VALID_REPRODUCIBILITY_LEVELS:
        meta = AutomationCaseMetadata(
            agent_type="Agent",
            workflow_components=["comp"],
            use_case_domain="domain",
            reproducibility=level,
        )
        assert meta.reproducibility == level


def test_automation_case_metadata_invalid_reproducibility_raises():
    """Invalid reproducibility must raise ValueError."""
    with pytest.raises(ValueError, match="reproducibility must be one of"):
        AutomationCaseMetadata(
            agent_type="Agent",
            workflow_components=["comp"],
            use_case_domain="domain",
            reproducibility="very_high",
        )


def test_automation_case_metadata_empty_agent_type_raises():
    """Empty agent_type must raise ValueError."""
    with pytest.raises(ValueError, match="agent_type is required"):
        AutomationCaseMetadata(
            agent_type="",
            workflow_components=["comp"],
            use_case_domain="domain",
        )


def test_automation_case_metadata_empty_workflow_components_raises():
    """Empty workflow_components must raise ValueError."""
    with pytest.raises(ValueError, match="workflow_components is required"):
        AutomationCaseMetadata(
            agent_type="Agent",
            workflow_components=[],
            use_case_domain="domain",
        )


def test_automation_case_metadata_empty_use_case_domain_raises():
    """Empty use_case_domain must raise ValueError."""
    with pytest.raises(ValueError, match="use_case_domain is required"):
        AutomationCaseMetadata(
            agent_type="Agent",
            workflow_components=["comp"],
            use_case_domain="",
        )


# =========================================================================
# CommunityContentMetadata
# =========================================================================


def test_community_content_metadata_valid_creation():
    """CommunityContentMetadata can be created with valid values."""
    meta = CommunityContentMetadata(
        platform="Reddit",
        format="thread",
        author_credibility="practitioner",
    )
    assert meta.type == "community_content"
    assert meta.platform == "Reddit"
    assert meta.format == "thread"
    assert meta.author_credibility == "practitioner"


def test_community_content_metadata_invalid_platform_raises():
    """Invalid platform must raise ValueError."""
    with pytest.raises(ValueError, match="platform must be one of"):
        CommunityContentMetadata(platform="Facebook")


def test_community_content_metadata_invalid_format_raises():
    """Invalid format must raise ValueError."""
    with pytest.raises(ValueError, match="format must be one of"):
        CommunityContentMetadata(format="podcast")


def test_community_content_metadata_invalid_credibility_raises():
    """Invalid author_credibility must raise ValueError."""
    with pytest.raises(ValueError, match="author_credibility must be one of"):
        CommunityContentMetadata(author_credibility="famous")


def test_community_content_metadata_all_valid_platforms():
    """All VALID_PLATFORMS should be accepted."""
    for platform in VALID_PLATFORMS:
        meta = CommunityContentMetadata(platform=platform)
        assert meta.platform == platform


def test_community_content_metadata_all_valid_formats():
    """All VALID_CONTENT_FORMATS should be accepted."""
    for fmt in VALID_CONTENT_FORMATS:
        meta = CommunityContentMetadata(format=fmt)
        assert meta.format == fmt


def test_community_content_metadata_all_valid_credibility():
    """All VALID_AUTHOR_CREDIBILITY should be accepted."""
    for cred in VALID_AUTHOR_CREDIBILITY:
        meta = CommunityContentMetadata(author_credibility=cred)
        assert meta.author_credibility == cred


# =========================================================================
# ToolReleaseMetadata
# =========================================================================


def test_tool_release_metadata_valid_creation():
    """ToolReleaseMetadata can be created with valid values."""
    meta = ToolReleaseMetadata(
        tool_name="Claude Opus 4.5",
        company="Anthropic",
        release_date="2025-05-01",
        release_type="major_update",
        key_features=["Extended thinking", "Improved coding"],
    )
    assert meta.type == "tool_release"
    assert meta.tool_name == "Claude Opus 4.5"
    assert meta.company == "Anthropic"


def test_tool_release_metadata_all_valid_release_types():
    """All VALID_RELEASE_TYPES should be accepted."""
    for rt in VALID_RELEASE_TYPES:
        meta = ToolReleaseMetadata(
            tool_name="Tool",
            company="Co",
            release_type=rt,
            key_features=["feature"],
        )
        assert meta.release_type == rt


def test_tool_release_metadata_invalid_release_type_raises():
    """Invalid release_type must raise ValueError."""
    with pytest.raises(ValueError, match="release_type must be one of"):
        ToolReleaseMetadata(
            tool_name="Tool",
            company="Co",
            release_type="patch",
            key_features=["feature"],
        )


def test_tool_release_metadata_empty_tool_name_raises():
    """Empty tool_name must raise ValueError."""
    with pytest.raises(ValueError, match="tool_name is required"):
        ToolReleaseMetadata(
            tool_name="",
            company="Co",
            key_features=["feature"],
        )


def test_tool_release_metadata_empty_company_raises():
    """Empty company must raise ValueError."""
    with pytest.raises(ValueError, match="company is required"):
        ToolReleaseMetadata(
            tool_name="Tool",
            company="",
            key_features=["feature"],
        )


def test_tool_release_metadata_empty_key_features_raises():
    """Empty key_features must raise ValueError."""
    with pytest.raises(ValueError, match="key_features is required"):
        ToolReleaseMetadata(
            tool_name="Tool",
            company="Co",
            key_features=[],
        )


# =========================================================================
# DraftPost
# =========================================================================


def test_draft_post_creation():
    """DraftPost can be created with all required fields."""
    post = DraftPost(
        hook="Did you know GPT-4 reduced fraud by 73%?",
        body="Here's how a FinTech company did it...\n\nStep 1: ...",
        cta="What's your experience with AI in fraud detection?",
        hashtags=["#AI", "#FinTech", "#FraudDetection"],
        full_text="Did you know GPT-4 reduced fraud by 73%?\n\nHere's how...",
        template_used="metrics_driven",
        hook_style=HookStyle.METRICS,
        content_type=ContentType.ENTERPRISE_CASE,
        character_count=280,
        visual_brief="Bar chart comparing fraud rates before and after AI",
        visual_type="data_viz",
        key_terms=["GPT-4", "fraud detection", "FinTech"],
    )
    assert post.hook_style is HookStyle.METRICS
    assert post.content_type is ContentType.ENTERPRISE_CASE
    assert post.character_count == 280
    assert len(post.hashtags) == 3
    assert len(post.key_terms) == 3


def test_draft_post_hook_style_is_hook_style_enum():
    """DraftPost.hook_style should accept HookStyle enum members."""
    post = DraftPost(
        hook="h",
        body="b",
        cta="c",
        hashtags=[],
        full_text="h b c",
        template_used="t",
        hook_style=HookStyle.CONTRARIAN,
        content_type=ContentType.PRIMARY_SOURCE,
        character_count=5,
        visual_brief="v",
        visual_type="diagram",
        key_terms=[],
    )
    assert isinstance(post.hook_style, HookStyle)


# =========================================================================
# WriterOutput
# =========================================================================


def test_writer_output_creation():
    """WriterOutput wraps a primary draft and alternatives."""
    primary = DraftPost(
        hook="h",
        body="b",
        cta="c",
        hashtags=[],
        full_text="h b c",
        template_used="t",
        hook_style=HookStyle.HOW_TO,
        content_type=ContentType.AUTOMATION_CASE,
        character_count=5,
        visual_brief="v",
        visual_type="diagram",
        key_terms=[],
    )
    output = WriterOutput(
        primary_draft=primary,
        alternatives=[],
        template_category="automation",
        generation_metadata={"model": "claude-opus-4-5-20251101"},
    )
    assert output.primary_draft is primary
    assert output.alternatives == []
    assert output.template_category == "automation"


# =========================================================================
# HumanizedPost
# =========================================================================


def test_humanized_post_creation():
    """HumanizedPost can be created with required and optional fields."""
    hp = HumanizedPost(
        original_text="Original AI-sounding text",
        humanized_text="More natural sounding text",
        changes_made=["Shortened sentences", "Added colloquialism"],
        humanization_intensity="medium",
        ai_detection_score_before=0.85,
        ai_detection_score_after=0.15,
    )
    assert hp.humanization_intensity == "medium"
    assert hp.ai_detection_score_before == 0.85
    assert hp.ai_detection_score_after == 0.15
    assert len(hp.changes_made) == 2


def test_humanized_post_optional_scores_default_to_none():
    """AI detection scores should default to None."""
    hp = HumanizedPost(
        original_text="text",
        humanized_text="text",
        changes_made=[],
        humanization_intensity="low",
    )
    assert hp.ai_detection_score_before is None
    assert hp.ai_detection_score_after is None


# =========================================================================
# VisualAsset and VisualCreatorOutput
# =========================================================================


def test_visual_asset_creation():
    """VisualAsset can be created with defaults."""
    asset = VisualAsset(asset_type="single_image")
    assert asset.asset_type == "single_image"
    assert asset.file_path is None
    assert asset.url is None
    assert asset.width == 0
    assert asset.height == 0
    assert asset.metadata == {}


def test_visual_creator_output_creation():
    """VisualCreatorOutput wraps a primary asset and alternatives."""
    asset = VisualAsset(asset_type="single_image", width=1200, height=628)
    output = VisualCreatorOutput(
        primary_asset=asset,
        alternatives=[],
        visual_brief_used="Bar chart of metrics",
        format_selected="single_image",
    )
    assert output.primary_asset.width == 1200
    assert output.format_selected == "single_image"


# =========================================================================
# QCResult and QCOutput
# =========================================================================


def test_qc_result_creation():
    """QCResult can be created with all fields."""
    result = QCResult(
        total_score=8.5,
        scores={"hook": 9.0, "body": 8.0, "cta": 8.5},
        decision="PASS",
        feedback="Strong post with compelling metrics.",
    )
    assert result.total_score == 8.5
    assert result.decision == "PASS"
    assert result.revision_target is None
    assert result.type_specific_feedback == {}


def test_qc_result_with_revision_target():
    """QCResult revision fields are populated for REVISE decisions."""
    result = QCResult(
        total_score=5.0,
        scores={"hook": 4.0, "body": 6.0},
        decision="REVISE_WRITER",
        feedback="Hook needs work.",
        revision_target="writer",
        type_specific_feedback={"hook_issue": "Too generic"},
    )
    assert result.decision == "REVISE_WRITER"
    assert result.revision_target == "writer"
    assert "hook_issue" in result.type_specific_feedback


def test_qc_output_creation():
    """QCOutput wraps a QCResult with scoring metadata."""
    result = QCResult(
        total_score=7.0,
        scores={"hook": 7.0},
        decision="PASS",
        feedback="OK",
    )
    output = QCOutput(
        result=result,
        scoring_weights_used={"hook": 0.3, "body": 0.5, "cta": 0.2},
        type_adjustments_applied={"enterprise_case": {"metrics_bonus": 0.1}},
    )
    assert output.result.total_score == 7.0
    assert "hook" in output.scoring_weights_used


# =========================================================================
# PostMetricsSnapshot
# =========================================================================


def test_post_metrics_snapshot_creation():
    """PostMetricsSnapshot can be created with required fields."""
    snapshot = PostMetricsSnapshot(
        post_id="post-123",
        likes=42,
        comments=7,
        reposts=3,
        impressions=1500,
        engagement_rate=0.035,
        minutes_after_post=60,
    )
    assert snapshot.post_id == "post-123"
    assert snapshot.likes == 42
    assert snapshot.impressions == 1500
    assert snapshot.engagement_rate == 0.035
    assert snapshot.minutes_after_post == 60


def test_post_metrics_snapshot_defaults():
    """PostMetricsSnapshot optional fields should have correct defaults."""
    snapshot = PostMetricsSnapshot(post_id="post-456")
    assert snapshot.likes == 0
    assert snapshot.comments == 0
    assert snapshot.reposts == 0
    assert snapshot.impressions is None
    assert snapshot.engagement_rate is None
    assert snapshot.minutes_after_post == 0
    # collected_at should be a timezone-aware datetime
    assert snapshot.collected_at.tzinfo is not None


# =========================================================================
# TypeSpecificExtraction
# =========================================================================


def test_type_specific_extraction_creation():
    """TypeSpecificExtraction can be created with all fields."""
    extraction = TypeSpecificExtraction(
        content_type=ContentType.ENTERPRISE_CASE,
        extracted_fields={"company": "Acme", "scale": "Enterprise"},
        required_fields_present=["company", "scale"],
        missing_fields=[],
        extraction_confidence=0.95,
    )
    assert extraction.content_type is ContentType.ENTERPRISE_CASE
    assert extraction.extraction_confidence == 0.95
    assert len(extraction.missing_fields) == 0


# =========================================================================
# AnalysisBrief
# =========================================================================


def test_analysis_brief_creation():
    """AnalysisBrief can be created with all required fields."""
    extraction = TypeSpecificExtraction(
        content_type=ContentType.PRIMARY_SOURCE,
        extracted_fields={"authors": ["Smith"]},
        required_fields_present=["authors"],
        missing_fields=["methodology_summary"],
        extraction_confidence=0.8,
    )
    brief = AnalysisBrief(
        topic_id="topic-001",
        content_type=ContentType.PRIMARY_SOURCE,
        title="New Transformer Architecture",
        key_findings=["Finding 1", "Finding 2"],
        main_argument="Transformers can be more efficient",
        suggested_angle="contrarian",
        hook_materials={"stat": "50% faster"},
        extraction_data=extraction,
        controversy_level="medium",
        complexity_level="simplify_slightly",
        target_audience="ML engineers",
        recommended_post_format="insight_thread",
        recommended_visual_type="data_viz",
    )
    assert brief.topic_id == "topic-001"
    assert brief.content_type is ContentType.PRIMARY_SOURCE
    assert brief.controversy_level == "medium"


# =========================================================================
# TrendTopic
# =========================================================================


def _make_trend_topic(
    content_type: ContentType = ContentType.ENTERPRISE_CASE,
    metadata=None,
    topic_id: str = "topic-001",
) -> TrendTopic:
    """Helper to build a minimal TrendTopic for tests."""
    if metadata is None:
        metadata = EnterpriseCaseMetadata(
            company="Acme Corp",
            industry="Tech",
            scale="Enterprise",
        )
    return TrendTopic(
        id=topic_id,
        title="AI in Enterprise: A Case Study",
        summary="Acme Corp deployed AI for fraud detection.",
        content_type=content_type,
        sources=["https://example.com/article"],
        primary_source_url="https://example.com/article",
        score=8.5,
        score_breakdown={"relevance": 9.0, "freshness": 8.0},
        quality_signals_matched=["has_metrics", "named_company"],
        suggested_angles=[
            SuggestedAngle(
                angle_text="ROI focus",
                angle_type="metrics_story",
                hook_templates=["Template A"],
                content_type_fit=0.9,
            ),
        ],
        related_topics=["AI fraud detection"],
        raw_content="Full article text here...",
        metadata=metadata,
        analysis_format="enterprise_deep_dive",
        recommended_post_format="insight_thread",
        recommended_visual_type="data_viz",
    )


def test_trend_topic_creation():
    """TrendTopic can be created with all required fields."""
    topic = _make_trend_topic()
    assert topic.id == "topic-001"
    assert topic.content_type is ContentType.ENTERPRISE_CASE
    assert topic.score == 8.5
    assert topic.is_top_pick is False
    assert topic.top_pick_summary is None
    assert topic.discovered_at.tzinfo is not None


def test_trend_topic_repr():
    """TrendTopic __repr__ includes key info."""
    topic = _make_trend_topic()
    repr_str = repr(topic)
    assert "topic-001" in repr_str
    assert "enterprise_case" in repr_str
    assert "8.5" in repr_str


def test_trend_topic_repr_with_top_pick():
    """TrendTopic __repr__ shows asterisk for top picks."""
    topic = _make_trend_topic()
    topic.is_top_pick = True
    repr_str = repr(topic)
    assert "*" in repr_str


# =========================================================================
# TrendScoutOutput
# =========================================================================


def test_trend_scout_output_creation():
    """TrendScoutOutput wraps topics and statistics."""
    topic = _make_trend_topic()
    output = TrendScoutOutput(
        run_id="run-001",
        run_timestamp=datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        topics=[topic],
        top_pick=topic,
        total_sources_scanned=50,
        topics_before_filter=20,
        topics_after_filter=5,
        exclusion_log=[],
        topics_by_type={"enterprise_case": 1},
    )
    assert output.run_id == "run-001"
    assert len(output.topics) == 1
    assert output.total_sources_scanned == 50


# =========================================================================
# PipelineState TypedDict
# =========================================================================


def test_pipeline_state_is_typed_dict():
    """PipelineState should be usable as a dict (TypedDict)."""
    state: PipelineState = {
        "run_id": "run-001",
        "stage": "scout",
    }
    assert state["run_id"] == "run-001"
    assert state["stage"] == "scout"


def test_pipeline_state_total_false():
    """PipelineState(total=False) means all keys are optional - empty dict is valid."""
    state: PipelineState = {}
    assert isinstance(state, dict)


def test_pipeline_state_accepts_model_types():
    """PipelineState can hold our model types as values."""
    topic = _make_trend_topic()
    draft = DraftPost(
        hook="h",
        body="b",
        cta="c",
        hashtags=[],
        full_text="h b c",
        template_used="t",
        hook_style=HookStyle.METRICS,
        content_type=ContentType.ENTERPRISE_CASE,
        character_count=5,
        visual_brief="v",
        visual_type="data_viz",
        key_terms=[],
    )
    state: PipelineState = {
        "run_id": "run-002",
        "content_type": ContentType.ENTERPRISE_CASE,
        "selected_topic": topic,
        "draft_post": draft,
        "revision_count": 0,
        "errors": [],
        "warnings": [],
    }
    assert state["content_type"] is ContentType.ENTERPRISE_CASE
    assert state["selected_topic"].id == "topic-001"
    assert state["revision_count"] == 0


# =========================================================================
# CONTENT_TYPE_HOOK_STYLES mapping
# =========================================================================


def test_content_type_hook_styles_covers_all_content_types():
    """Every ContentType must have an entry in CONTENT_TYPE_HOOK_STYLES."""
    for ct in ContentType:
        assert ct in CONTENT_TYPE_HOOK_STYLES, f"{ct} missing from CONTENT_TYPE_HOOK_STYLES"


def test_content_type_hook_styles_values_are_hook_style_enums():
    """All values in CONTENT_TYPE_HOOK_STYLES must be HookStyle members."""
    for ct, styles in CONTENT_TYPE_HOOK_STYLES.items():
        assert len(styles) > 0, f"{ct} has empty hook styles list"
        for style in styles:
            assert isinstance(style, HookStyle), f"{style} is not a HookStyle"


def test_content_type_hook_styles_no_duplicates():
    """No content type should have duplicate hook styles."""
    for ct, styles in CONTENT_TYPE_HOOK_STYLES.items():
        assert len(styles) == len(set(styles)), f"{ct} has duplicate hook styles"


# =========================================================================
# get_hook_styles_for_type
# =========================================================================


def test_get_hook_styles_for_type_enterprise_case():
    """Enterprise case should return METRICS, LESSONS_LEARNED, PROBLEM_SOLUTION."""
    styles = get_hook_styles_for_type(ContentType.ENTERPRISE_CASE)
    assert HookStyle.METRICS in styles
    assert HookStyle.LESSONS_LEARNED in styles
    assert HookStyle.PROBLEM_SOLUTION in styles
    assert len(styles) == 3


def test_get_hook_styles_for_type_primary_source():
    """Primary source should return 5 styles including CONTRARIAN and QUESTION."""
    styles = get_hook_styles_for_type(ContentType.PRIMARY_SOURCE)
    assert HookStyle.CONTRARIAN in styles
    assert HookStyle.QUESTION in styles
    assert HookStyle.SURPRISING_STAT in styles
    assert HookStyle.SIMPLIFIED_EXPLAINER in styles
    assert HookStyle.DEBATE_STARTER in styles
    assert len(styles) == 5


def test_get_hook_styles_for_type_automation_case():
    """Automation case should return 5 styles including HOW_TO and TIME_SAVED."""
    styles = get_hook_styles_for_type(ContentType.AUTOMATION_CASE)
    assert HookStyle.HOW_TO in styles
    assert HookStyle.TIME_SAVED in styles
    assert HookStyle.BEFORE_AFTER in styles
    assert HookStyle.RESULTS_STORY in styles
    assert HookStyle.TOOL_COMPARISON in styles
    assert len(styles) == 5


def test_get_hook_styles_for_type_community_content():
    """Community content should return 6 styles including RELATABLE."""
    styles = get_hook_styles_for_type(ContentType.COMMUNITY_CONTENT)
    assert HookStyle.RELATABLE in styles
    assert HookStyle.COMMUNITY_REFERENCE in styles
    assert HookStyle.PERSONAL in styles
    assert HookStyle.CURATED_INSIGHTS in styles
    assert HookStyle.HOT_TAKE_RESPONSE in styles
    assert HookStyle.PRACTITIONER_WISDOM in styles
    assert len(styles) == 6


def test_get_hook_styles_for_type_tool_release():
    """Tool release should return 5 styles including NEWS_BREAKING."""
    styles = get_hook_styles_for_type(ContentType.TOOL_RELEASE)
    assert HookStyle.NEWS_BREAKING in styles
    assert HookStyle.FEATURE_HIGHLIGHT in styles
    assert HookStyle.COMPARISON in styles
    assert HookStyle.FIRST_LOOK in styles
    assert HookStyle.IMPLICATIONS in styles
    assert len(styles) == 5


def test_get_hook_styles_for_type_matches_mapping():
    """get_hook_styles_for_type must return the same list as CONTENT_TYPE_HOOK_STYLES."""
    for ct in ContentType:
        assert get_hook_styles_for_type(ct) == CONTENT_TYPE_HOOK_STYLES[ct]


# =========================================================================
# validate_hook_style
# =========================================================================


def test_validate_hook_style_valid_combination():
    """validate_hook_style returns True for allowed combinations."""
    assert validate_hook_style(HookStyle.METRICS, ContentType.ENTERPRISE_CASE) is True
    assert validate_hook_style(HookStyle.CONTRARIAN, ContentType.PRIMARY_SOURCE) is True
    assert validate_hook_style(HookStyle.HOW_TO, ContentType.AUTOMATION_CASE) is True
    assert validate_hook_style(HookStyle.RELATABLE, ContentType.COMMUNITY_CONTENT) is True
    assert validate_hook_style(HookStyle.NEWS_BREAKING, ContentType.TOOL_RELEASE) is True


def test_validate_hook_style_invalid_combination():
    """validate_hook_style returns False for disallowed combinations."""
    # METRICS is only for ENTERPRISE_CASE
    assert validate_hook_style(HookStyle.METRICS, ContentType.PRIMARY_SOURCE) is False
    assert validate_hook_style(HookStyle.METRICS, ContentType.AUTOMATION_CASE) is False
    assert validate_hook_style(HookStyle.METRICS, ContentType.COMMUNITY_CONTENT) is False
    assert validate_hook_style(HookStyle.METRICS, ContentType.TOOL_RELEASE) is False


def test_validate_hook_style_cross_type():
    """Hook styles should not validate for content types they don't belong to."""
    # HOW_TO is for AUTOMATION_CASE only
    assert validate_hook_style(HookStyle.HOW_TO, ContentType.ENTERPRISE_CASE) is False
    assert validate_hook_style(HookStyle.HOW_TO, ContentType.PRIMARY_SOURCE) is False
    assert validate_hook_style(HookStyle.HOW_TO, ContentType.COMMUNITY_CONTENT) is False
    assert validate_hook_style(HookStyle.HOW_TO, ContentType.TOOL_RELEASE) is False


def test_validate_hook_style_all_valid_combinations():
    """Every style in CONTENT_TYPE_HOOK_STYLES must validate as True."""
    for ct, styles in CONTENT_TYPE_HOOK_STYLES.items():
        for style in styles:
            assert validate_hook_style(style, ct) is True, (
                f"validate_hook_style({style}, {ct}) should be True"
            )


# =========================================================================
# validate_topic_metadata
# =========================================================================


def test_validate_topic_metadata_matching_types():
    """validate_topic_metadata returns True when metadata.type matches content_type."""
    topic = _make_trend_topic(
        content_type=ContentType.ENTERPRISE_CASE,
        metadata=EnterpriseCaseMetadata(
            company="Acme",
            industry="Tech",
            scale="Enterprise",
        ),
    )
    assert validate_topic_metadata(topic) is True


def test_validate_topic_metadata_primary_source():
    """validate_topic_metadata works for primary_source type."""
    topic = _make_trend_topic(
        content_type=ContentType.PRIMARY_SOURCE,
        metadata=PrimarySourceMetadata(
            authors=["Author"],
            organization="Org",
            key_hypothesis="H",
        ),
    )
    assert validate_topic_metadata(topic) is True


def test_validate_topic_metadata_automation_case():
    """validate_topic_metadata works for automation_case type."""
    topic = _make_trend_topic(
        content_type=ContentType.AUTOMATION_CASE,
        metadata=AutomationCaseMetadata(
            agent_type="RAG",
            workflow_components=["comp"],
            use_case_domain="domain",
        ),
    )
    assert validate_topic_metadata(topic) is True


def test_validate_topic_metadata_community_content():
    """validate_topic_metadata works for community_content type."""
    topic = _make_trend_topic(
        content_type=ContentType.COMMUNITY_CONTENT,
        metadata=CommunityContentMetadata(),
    )
    assert validate_topic_metadata(topic) is True


def test_validate_topic_metadata_tool_release():
    """validate_topic_metadata works for tool_release type."""
    topic = _make_trend_topic(
        content_type=ContentType.TOOL_RELEASE,
        metadata=ToolReleaseMetadata(
            tool_name="Tool",
            company="Co",
            key_features=["feature"],
        ),
    )
    assert validate_topic_metadata(topic) is True


def test_validate_topic_metadata_mismatch_raises():
    """validate_topic_metadata raises MetadataTypeMismatchError on mismatch."""
    topic = _make_trend_topic(
        content_type=ContentType.PRIMARY_SOURCE,
        metadata=EnterpriseCaseMetadata(
            company="Acme",
            industry="Tech",
            scale="Enterprise",
        ),
    )
    with pytest.raises(MetadataTypeMismatchError, match="metadata.type='enterprise_case'"):
        validate_topic_metadata(topic)


def test_validate_topic_metadata_no_type_field_raises():
    """validate_topic_metadata raises if metadata has no 'type' discriminator."""

    class FakeMetadata:
        """Metadata without a type field."""
        pass

    topic = _make_trend_topic()
    topic.metadata = FakeMetadata()  # type: ignore[assignment]
    with pytest.raises(MetadataTypeMismatchError, match="no 'type' discriminator"):
        validate_topic_metadata(topic)


def test_validate_topic_metadata_all_content_types():
    """validate_topic_metadata should pass for all content types with correct metadata."""
    metadata_by_type = {
        ContentType.ENTERPRISE_CASE: EnterpriseCaseMetadata(
            company="Co", industry="Tech", scale="SMB",
        ),
        ContentType.PRIMARY_SOURCE: PrimarySourceMetadata(
            authors=["A"], organization="Org", key_hypothesis="H",
        ),
        ContentType.AUTOMATION_CASE: AutomationCaseMetadata(
            agent_type="Agent", workflow_components=["c"], use_case_domain="d",
        ),
        ContentType.COMMUNITY_CONTENT: CommunityContentMetadata(),
        ContentType.TOOL_RELEASE: ToolReleaseMetadata(
            tool_name="T", company="C", key_features=["f"],
        ),
    }
    for ct, meta in metadata_by_type.items():
        topic = _make_trend_topic(content_type=ct, metadata=meta)
        assert validate_topic_metadata(topic) is True


# =========================================================================
# Validation constants sanity checks
# =========================================================================


def test_valid_enterprise_scales_values():
    """VALID_ENTERPRISE_SCALES should contain the four expected values."""
    assert VALID_ENTERPRISE_SCALES == ["SMB", "Mid-Market", "Enterprise", "Fortune 500"]


def test_valid_source_types_values():
    """VALID_SOURCE_TYPES should contain the expected values."""
    assert "research_paper" in VALID_SOURCE_TYPES
    assert "think_tank_report" in VALID_SOURCE_TYPES
    assert "expert_essay" in VALID_SOURCE_TYPES
    assert "whitepaper" in VALID_SOURCE_TYPES
    assert len(VALID_SOURCE_TYPES) == 4


def test_valid_reproducibility_levels_values():
    """VALID_REPRODUCIBILITY_LEVELS should contain high, medium, low."""
    assert VALID_REPRODUCIBILITY_LEVELS == ["high", "medium", "low"]


def test_valid_platforms_values():
    """VALID_PLATFORMS should contain the expected platforms."""
    expected = ["YouTube", "Reddit", "HackerNews", "Dev.to", "Twitter", "Medium", "Substack"]
    assert VALID_PLATFORMS == expected


def test_valid_content_formats_values():
    """VALID_CONTENT_FORMATS should contain the expected formats."""
    expected = ["video", "post", "comment", "thread", "article", "newsletter"]
    assert VALID_CONTENT_FORMATS == expected


def test_valid_release_types_values():
    """VALID_RELEASE_TYPES should contain the expected types."""
    expected = ["new_product", "major_update", "api_release", "open_source"]
    assert VALID_RELEASE_TYPES == expected
