"""
Tests for src.config module.

Covers:
    - AutonomyLevel enum values
    - SourceThresholdConfig defaults and get_min_score
    - ThresholdConfig defaults, get_decision, threshold accessors
    - Settings defaults and from_yaml
    - Module-level helpers: load_type_context, get_domain_candidate_types, get_url_type_hint
    - Singleton get_settings / reset_settings behaviour
"""

import pytest

from src.config import (
    AutonomyLevel,
    SourceThresholdConfig,
    ThresholdConfig,
    Settings,
    load_type_context,
    get_domain_candidate_types,
    get_url_type_hint,
    reset_settings,
)
from src.models import ContentType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure the Settings singleton is cleared before and after each test."""
    reset_settings()
    yield
    reset_settings()


# ===========================================================================
# 1. AutonomyLevel enum
# ===========================================================================


class TestAutonomyLevel:
    """Tests for AutonomyLevel IntEnum."""

    def test_has_four_levels(self):
        """AutonomyLevel should have exactly 4 members."""
        assert len(AutonomyLevel) == 4

    def test_human_all_value(self):
        assert AutonomyLevel.HUMAN_ALL == 1

    def test_human_posts_value(self):
        assert AutonomyLevel.HUMAN_POSTS == 2

    def test_auto_high_score_value(self):
        assert AutonomyLevel.AUTO_HIGH_SCORE == 3

    def test_full_autonomy_value(self):
        assert AutonomyLevel.FULL_AUTONOMY == 4

    def test_is_int_enum(self):
        """AutonomyLevel values should be usable as plain integers."""
        assert AutonomyLevel.FULL_AUTONOMY > AutonomyLevel.HUMAN_ALL
        assert int(AutonomyLevel.AUTO_HIGH_SCORE) == 3


# ===========================================================================
# 2. SourceThresholdConfig
# ===========================================================================


class TestSourceThresholdConfig:
    """Tests for SourceThresholdConfig defaults and get_min_score."""

    def test_default_hackernews_min_score(self):
        cfg = SourceThresholdConfig()
        assert cfg.hackernews_min_score == 50

    def test_default_twitter_min_engagement(self):
        cfg = SourceThresholdConfig()
        assert cfg.twitter_min_engagement == 1000

    def test_default_product_hunt_min_upvotes(self):
        cfg = SourceThresholdConfig()
        assert cfg.product_hunt_min_upvotes == 200

    def test_default_github_min_stars_velocity(self):
        cfg = SourceThresholdConfig()
        assert cfg.github_min_stars_velocity == 100

    def test_default_reddit_min_score(self):
        cfg = SourceThresholdConfig()
        assert cfg.reddit_min_score == 100

    def test_default_reddit_min_comments(self):
        cfg = SourceThresholdConfig()
        assert cfg.reddit_min_comments == 20

    def test_default_youtube_min_views(self):
        cfg = SourceThresholdConfig()
        assert cfg.youtube_min_views == 10000

    def test_default_devto_min_reactions(self):
        cfg = SourceThresholdConfig()
        assert cfg.devto_min_reactions == 50

    def test_default_medium_min_claps(self):
        cfg = SourceThresholdConfig()
        assert cfg.medium_min_claps == 100

    def test_default_arxiv_min_citations(self):
        cfg = SourceThresholdConfig()
        assert cfg.arxiv_min_citations == 0

    def test_get_min_score_hackernews(self):
        cfg = SourceThresholdConfig()
        assert cfg.get_min_score("hackernews") == 50

    def test_get_min_score_reddit(self):
        cfg = SourceThresholdConfig()
        assert cfg.get_min_score("reddit") == 100

    def test_get_min_score_unknown_source_raises(self):
        cfg = SourceThresholdConfig()
        with pytest.raises(ValueError, match="Unknown source"):
            cfg.get_min_score("unknown_source")

    def test_env_override_hackernews(self, monkeypatch):
        """Environment variable HN_MIN_SCORE should override the default."""
        monkeypatch.setenv("HN_MIN_SCORE", "200")
        cfg = SourceThresholdConfig()
        assert cfg.hackernews_min_score == 200
        assert cfg.get_min_score("hackernews") == 200

    def test_env_override_invalid_value_raises(self, monkeypatch):
        """Invalid env var value should raise ConfigurationError."""
        from src.exceptions import ConfigurationError

        monkeypatch.setenv("HN_MIN_SCORE", "not_a_number")
        with pytest.raises(ConfigurationError, match="Invalid value"):
            SourceThresholdConfig()


# ===========================================================================
# 3-6. ThresholdConfig
# ===========================================================================


class TestThresholdConfig:
    """Tests for ThresholdConfig defaults and decision logic."""

    # -- Default values (test 2) -------------------------------------------

    def test_default_min_score_to_proceed(self):
        cfg = ThresholdConfig()
        assert cfg.min_score_to_proceed == 8.0

    def test_default_auto_publish_threshold(self):
        cfg = ThresholdConfig()
        assert cfg.auto_publish_threshold == 9.0

    def test_default_rejection_threshold(self):
        cfg = ThresholdConfig()
        assert cfg.rejection_threshold == 5.5

    def test_default_revision_threshold(self):
        cfg = ThresholdConfig()
        assert cfg.revision_threshold == 7.0

    def test_default_max_meta_iterations(self):
        cfg = ThresholdConfig()
        assert cfg.max_meta_iterations == 3

    def test_default_max_reject_restarts(self):
        cfg = ThresholdConfig()
        assert cfg.max_reject_restarts == 2

    def test_default_type_multipliers_has_five_keys(self):
        cfg = ThresholdConfig()
        assert len(cfg.type_multipliers) == 5

    def test_default_type_multipliers_keys(self):
        cfg = ThresholdConfig()
        expected_keys = {
            "enterprise_case",
            "primary_source",
            "automation_case",
            "community_content",
            "tool_release",
        }
        assert set(cfg.type_multipliers.keys()) == expected_keys

    # -- get_decision: "pass" (test 3) -------------------------------------

    def test_get_decision_pass_for_high_score(self):
        """Score above pass threshold should return 'pass'."""
        cfg = ThresholdConfig()
        # Default enterprise_case multiplier is 0.90, so pass_threshold = 8.0 * 0.90 = 7.2
        assert cfg.get_decision(8.5, "enterprise_case") == "pass"

    def test_get_decision_pass_at_exact_threshold(self):
        """Score exactly at pass threshold should return 'pass' (>= comparison)."""
        cfg = ThresholdConfig()
        pass_threshold = cfg.get_pass_threshold("enterprise_case")
        assert cfg.get_decision(pass_threshold, "enterprise_case") == "pass"

    def test_get_decision_pass_with_content_type_enum(self):
        """get_decision should work with ContentType enum values."""
        cfg = ThresholdConfig()
        assert cfg.get_decision(9.5, ContentType.ENTERPRISE_CASE) == "pass"

    # -- get_decision: "revise" (test 4) -----------------------------------

    def test_get_decision_revise_for_mid_score(self):
        """Score between rejection and pass thresholds should return 'revise'."""
        cfg = ThresholdConfig()
        # enterprise_case: pass = 7.2, reject = 5.5 * 0.90 = 4.95
        # Score of 6.0 is between 4.95 and 7.2
        assert cfg.get_decision(6.0, "enterprise_case") == "revise"

    def test_get_decision_revise_at_exact_rejection_threshold(self):
        """Score exactly at rejection threshold should return 'revise' (>= comparison)."""
        cfg = ThresholdConfig()
        reject_threshold = cfg.get_rejection_threshold("enterprise_case")
        assert cfg.get_decision(reject_threshold, "enterprise_case") == "revise"

    # -- get_decision: "reject" (test 5) -----------------------------------

    def test_get_decision_reject_for_low_score(self):
        """Score below rejection threshold should return 'reject'."""
        cfg = ThresholdConfig()
        assert cfg.get_decision(2.0, "enterprise_case") == "reject"

    def test_get_decision_reject_just_below_threshold(self):
        """Score just below rejection threshold should return 'reject'."""
        cfg = ThresholdConfig()
        reject_threshold = cfg.get_rejection_threshold("enterprise_case")
        assert cfg.get_decision(reject_threshold - 0.01, "enterprise_case") == "reject"

    def test_get_decision_reject_zero_score(self):
        """Zero score should always return 'reject'."""
        cfg = ThresholdConfig()
        assert cfg.get_decision(0.0, "enterprise_case") == "reject"

    # -- get_decision with unknown type (uses 1.0 multiplier) --------------

    def test_get_decision_unknown_type_uses_base_thresholds(self):
        """Unknown content types use multiplier 1.0 (base thresholds)."""
        cfg = ThresholdConfig()
        # pass threshold = 8.0 * 1.0 = 8.0
        assert cfg.get_decision(8.5, "unknown_type") == "pass"
        assert cfg.get_decision(6.0, "unknown_type") == "revise"
        assert cfg.get_decision(4.0, "unknown_type") == "reject"

    # -- get_max_revisions (test 6) ----------------------------------------

    def test_get_max_revisions_enterprise_case(self):
        cfg = ThresholdConfig()
        result = cfg.get_max_revisions("enterprise_case")
        assert isinstance(result, int)
        assert result == 3

    def test_get_max_revisions_primary_source(self):
        cfg = ThresholdConfig()
        assert cfg.get_max_revisions("primary_source") == 4

    def test_get_max_revisions_community_content(self):
        cfg = ThresholdConfig()
        assert cfg.get_max_revisions("community_content") == 2

    def test_get_max_revisions_with_content_type_enum(self):
        cfg = ThresholdConfig()
        result = cfg.get_max_revisions(ContentType.AUTOMATION_CASE)
        assert isinstance(result, int)
        assert result == 3

    def test_get_max_revisions_unknown_type_returns_default(self):
        """Unknown content types should get a default of 3 revisions."""
        cfg = ThresholdConfig()
        assert cfg.get_max_revisions("unknown_type") == 3

    # -- Threshold accessors -----------------------------------------------

    def test_get_pass_threshold_enterprise_case(self):
        cfg = ThresholdConfig()
        # 8.0 * 0.90 = 7.2
        assert cfg.get_pass_threshold("enterprise_case") == pytest.approx(7.2)

    def test_get_pass_threshold_primary_source(self):
        cfg = ThresholdConfig()
        # 8.0 * 0.9375 = 7.5
        assert cfg.get_pass_threshold("primary_source") == pytest.approx(7.5)

    def test_get_revision_threshold_with_multiplier(self):
        cfg = ThresholdConfig()
        # 7.0 * 0.90 = 6.3
        assert cfg.get_revision_threshold("enterprise_case") == pytest.approx(6.3)

    def test_get_rejection_threshold_with_multiplier(self):
        cfg = ThresholdConfig()
        # 5.5 * 0.90 = 4.95
        assert cfg.get_rejection_threshold("enterprise_case") == pytest.approx(4.95)

    def test_get_auto_publish_threshold_is_constant(self):
        """Auto-publish threshold has no type adjustment."""
        cfg = ThresholdConfig()
        assert cfg.get_auto_publish_threshold() == 9.0

    def test_get_max_meta_iterations(self):
        cfg = ThresholdConfig()
        assert cfg.get_max_meta_iterations() == 3


# ===========================================================================
# 8. Settings
# ===========================================================================


class TestSettings:
    """Tests for Settings dataclass defaults and from_yaml."""

    def test_default_autonomy_level(self):
        settings = Settings()
        assert settings.autonomy_level == 3

    def test_default_llm_model_contains_claude(self):
        settings = Settings()
        assert "claude" in settings.llm_model

    def test_default_llm_model_exact(self):
        settings = Settings()
        assert settings.llm_model == "claude-opus-4-5-20251101"

    def test_default_timezone(self):
        settings = Settings()
        assert settings.timezone == "UTC"

    def test_default_log_level(self):
        settings = Settings()
        assert settings.log_level == "INFO"

    def test_default_min_hours_between_posts(self):
        settings = Settings()
        assert settings.min_hours_between_posts == 6

    def test_default_max_posts_per_day(self):
        settings = Settings()
        assert settings.max_posts_per_day == 2

    def test_default_thresholds_is_threshold_config(self):
        settings = Settings()
        assert isinstance(settings.thresholds, ThresholdConfig)

    def test_default_node_timeouts_has_expected_keys(self):
        settings = Settings()
        expected_keys = {"scout", "analyze", "write", "meta_evaluate", "humanize", "visualize", "qc"}
        assert set(settings.node_timeouts.keys()) == expected_keys

    def test_from_yaml_returns_settings_instance(self, tmp_path):
        """from_yaml with a nonexistent path should return defaults."""
        path = tmp_path / "nonexistent.yaml"
        settings = Settings.from_yaml(path)
        assert isinstance(settings, Settings)
        assert settings.autonomy_level == 3

    def test_from_yaml_loads_custom_values(self, tmp_path):
        """from_yaml should load values from a valid YAML file."""
        yaml_content = (
            "autonomy_level: 4\n"
            "llm_model: claude-test-model\n"
            "timezone: US/Eastern\n"
            "max_posts_per_day: 5\n"
            "thresholds:\n"
            "  min_score_to_proceed: 7.0\n"
            "  auto_publish_threshold: 8.5\n"
        )
        yaml_path = tmp_path / "settings.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")

        settings = Settings.from_yaml(yaml_path)
        assert settings.autonomy_level == 4
        assert settings.llm_model == "claude-test-model"
        assert settings.timezone == "US/Eastern"
        assert settings.max_posts_per_day == 5
        assert settings.thresholds.min_score_to_proceed == 7.0
        assert settings.thresholds.auto_publish_threshold == 8.5

    def test_from_yaml_invalid_yaml_raises(self, tmp_path):
        """from_yaml should raise ConfigurationError for malformed YAML."""
        from src.exceptions import ConfigurationError

        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("{{{{invalid yaml: [", encoding="utf-8")

        with pytest.raises(ConfigurationError, match="Failed to parse"):
            Settings.from_yaml(yaml_path)

    def test_from_yaml_preserves_threshold_defaults_when_not_specified(self, tmp_path):
        """Threshold values not in YAML should keep their defaults."""
        yaml_content = (
            "autonomy_level: 2\n"
            "thresholds:\n"
            "  min_score_to_proceed: 7.5\n"
        )
        yaml_path = tmp_path / "settings.yaml"
        yaml_path.write_text(yaml_content, encoding="utf-8")

        settings = Settings.from_yaml(yaml_path)
        assert settings.thresholds.min_score_to_proceed == 7.5
        # These should still be defaults
        assert settings.thresholds.auto_publish_threshold == 9.0
        assert settings.thresholds.rejection_threshold == 5.5


# ===========================================================================
# 9. load_type_context
# ===========================================================================


class TestLoadTypeContext:
    """Tests for load_type_context function."""

    EXPECTED_KEYS = {
        "extraction_focus",
        "required_fields",
        "preferred_templates",
        "hook_styles",
        "cta_style",
        "humanization_intensity",
        "tone_markers",
        "avoid_markers",
        "visual_formats",
        "color_scheme",
        "extra_criteria",
        "weight_adjustments",
        "pass_threshold",
    }

    @pytest.mark.parametrize(
        "content_type",
        [
            ContentType.ENTERPRISE_CASE,
            ContentType.PRIMARY_SOURCE,
            ContentType.AUTOMATION_CASE,
            ContentType.COMMUNITY_CONTENT,
            ContentType.TOOL_RELEASE,
        ],
    )
    def test_returns_dict_with_expected_keys(self, content_type):
        """Each ContentType should produce a dict with all expected keys."""
        context = load_type_context(content_type)
        assert isinstance(context, dict)
        assert set(context.keys()) == self.EXPECTED_KEYS

    def test_returns_dict_for_string_content_type(self):
        """load_type_context should accept a plain string as well."""
        context = load_type_context("enterprise_case")
        assert isinstance(context, dict)
        assert "extraction_focus" in context

    def test_unknown_type_returns_empty_dict(self):
        """An unrecognized content type should return an empty dict."""
        context = load_type_context("nonexistent_type")
        assert context == {}

    def test_enterprise_case_extraction_focus(self):
        context = load_type_context(ContentType.ENTERPRISE_CASE)
        assert "company" in context["extraction_focus"]
        assert "metrics" in context["extraction_focus"]

    def test_primary_source_humanization_intensity(self):
        context = load_type_context(ContentType.PRIMARY_SOURCE)
        assert context["humanization_intensity"] == "low"

    def test_automation_case_cta_style(self):
        context = load_type_context(ContentType.AUTOMATION_CASE)
        assert context["cta_style"] == "practical_action"

    def test_community_content_color_scheme(self):
        context = load_type_context(ContentType.COMMUNITY_CONTENT)
        assert context["color_scheme"] == "community_warm"

    def test_tool_release_pass_threshold(self):
        context = load_type_context(ContentType.TOOL_RELEASE)
        assert context["pass_threshold"] == 7.0


# ===========================================================================
# 10. get_domain_candidate_types
# ===========================================================================


class TestGetDomainCandidateTypes:
    """Tests for get_domain_candidate_types function."""

    def test_known_domain_returns_list(self):
        result = get_domain_candidate_types("arxiv.org")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_arxiv_returns_primary_source(self):
        result = get_domain_candidate_types("arxiv.org")
        assert "primary_source" in result

    def test_producthunt_returns_tool_release(self):
        result = get_domain_candidate_types("producthunt.com")
        assert "tool_release" in result

    def test_reddit_returns_multiple_types(self):
        result = get_domain_candidate_types("reddit.com")
        assert len(result) > 1
        assert "community_content" in result

    def test_gartner_returns_both_research_and_case(self):
        result = get_domain_candidate_types("gartner.com")
        assert "primary_source" in result
        assert "enterprise_case" in result

    def test_unknown_domain_returns_all_types(self):
        """Unknown domains should fall back to all five content types."""
        result = get_domain_candidate_types("example-unknown-domain.com")
        assert isinstance(result, list)
        assert len(result) == 5
        expected_types = {
            "enterprise_case",
            "primary_source",
            "automation_case",
            "community_content",
            "tool_release",
        }
        assert set(result) == expected_types

    def test_returns_new_list_not_reference(self):
        """Each call should return a fresh list, not a reference to the internal dict."""
        result1 = get_domain_candidate_types("arxiv.org")
        result2 = get_domain_candidate_types("arxiv.org")
        assert result1 == result2
        assert result1 is not result2


# ===========================================================================
# 11. get_url_type_hint
# ===========================================================================


class TestGetUrlTypeHint:
    """Tests for get_url_type_hint function."""

    def test_case_study_pattern(self):
        result = get_url_type_hint("https://example.com/case-study/acme-corp")
        assert result == "enterprise_case"

    def test_research_pattern(self):
        result = get_url_type_hint("https://mckinsey.com/research/ai-trends-2025")
        assert result == "primary_source"

    def test_workflow_pattern(self):
        result = get_url_type_hint("https://blog.example.com/workflow-automation-guide")
        assert result == "automation_case"

    def test_n8n_pattern(self):
        result = get_url_type_hint("https://community.n8n.io/cool-template")
        assert result == "automation_case"

    def test_release_pattern(self):
        result = get_url_type_hint("https://openai.com/blog/release-gpt5")
        assert result == "tool_release"

    def test_launch_pattern(self):
        result = get_url_type_hint("https://producthunt.com/posts/launch-of-new-tool")
        assert result == "tool_release"

    def test_no_match_returns_none(self):
        result = get_url_type_hint("https://example.com/about-us")
        assert result is None

    def test_case_insensitive_match(self):
        """URL matching should be case-insensitive."""
        result = get_url_type_hint("https://example.com/CASE-STUDY/Acme")
        assert result == "enterprise_case"

    def test_empty_url_returns_none(self):
        result = get_url_type_hint("")
        assert result is None
