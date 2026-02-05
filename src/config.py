"""
Centralized configuration loader for the LinkedIn Super Agent.

Loads settings from YAML files and environment variables, providing
sensible defaults when configuration files are absent.

Provides:
    - ThresholdConfig: Centralized threshold management for all agents (QC, meta-eval, etc.)
    - SourceThresholdConfig: Source-specific engagement thresholds with env var overrides
    - Settings: Global application settings loaded from YAML + env vars
    - load_type_context(): Type-specific pipeline context for each ContentType
    - domain_to_candidate_types: Domain-to-ContentType mapping for classification
    - url_pattern_type_hints: URL pattern hints for ContentType classification
    - get_settings(): Thread-safe singleton accessor for Settings
    - validate_env(): Startup validation of required environment variables

Architecture reference: architecture.md lines 2250-2362 (ThresholdConfig),
    2764-2838 (SourceThresholdConfig), 3209-3272 (domain/URL mappings),
    11358-11471 (load_type_context), 20580-20604 (autonomy config).
"""

import logging
import os
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

from src.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Load .env file (no-op if file does not exist)
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Project root directory (parent of src/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


# ===========================================================================
# AUTONOMY LEVELS
# ===========================================================================


class AutonomyLevel(IntEnum):
    """
    Autonomy levels for the agent system.

    Level 1: Human approves everything (posts, modifications, research)
    Level 2: Human approves posts only, auto-modifications allowed
    Level 3: Auto-publish high-score posts (>=9.0), human for rest
    Level 4: Full autonomy (human notified, not asked)
    """

    HUMAN_ALL = 1
    HUMAN_POSTS = 2
    AUTO_HIGH_SCORE = 3
    FULL_AUTONOMY = 4


# ===========================================================================
# SOURCE THRESHOLD CONFIGURATION
# Architecture reference: lines 2764-2838
# ===========================================================================


@dataclass
class SourceThresholdConfig:
    """
    Single source of truth for all source-specific engagement thresholds.

    Allows tuning via environment variables for A/B testing without
    code changes. See ``.env.example`` for available overrides.

    Usage::

        config = SourceThresholdConfig()
        min_score = config.get_min_score("hackernews")
    """

    # Engagement thresholds by source
    hackernews_min_score: int = 50
    twitter_min_engagement: int = 1000
    product_hunt_min_upvotes: int = 200
    github_min_stars_velocity: int = 100  # Stars gained in last 24h
    reddit_min_score: int = 100
    reddit_min_comments: int = 20
    youtube_min_views: int = 10000
    devto_min_reactions: int = 50
    medium_min_claps: int = 100

    # Research source thresholds
    arxiv_min_citations: int = 0  # New papers OK
    gartner_min_recency_days: int = 30
    mckinsey_min_recency_days: int = 90

    def __post_init__(self) -> None:
        """Override thresholds from environment variables if set."""
        env_overrides = {
            "HN_MIN_SCORE": ("hackernews_min_score", int),
            "PH_MIN_UPVOTES": ("product_hunt_min_upvotes", int),
            "GH_MIN_STARS_VELOCITY": ("github_min_stars_velocity", int),
            "REDDIT_MIN_COMMENTS": ("reddit_min_comments", int),
            "DEVTO_MIN_REACTIONS": ("devto_min_reactions", int),
        }
        for env_key, (attr_name, cast_fn) in env_overrides.items():
            env_val = os.environ.get(env_key)
            if env_val is not None:
                try:
                    setattr(self, attr_name, cast_fn(env_val))
                except (ValueError, TypeError) as exc:
                    raise ConfigurationError(
                        f"Invalid value for env var {env_key}='{env_val}': {exc}"
                    ) from exc

    def get_min_score(self, source: str) -> int:
        """
        Get minimum score/engagement threshold for a source.

        Args:
            source: Source identifier (e.g. ``"hackernews"``, ``"reddit"``).

        Returns:
            Minimum threshold value.

        Raises:
            ValueError: If source is unknown (fail-fast).
        """
        thresholds: Dict[str, int] = {
            "hackernews": self.hackernews_min_score,
            "twitter_x": self.twitter_min_engagement,
            "product_hunt": self.product_hunt_min_upvotes,
            "github_trending": self.github_min_stars_velocity,
            "reddit": self.reddit_min_score,
            "youtube": self.youtube_min_views,
            "devto": self.devto_min_reactions,
            "medium": self.medium_min_claps,
        }
        if source not in thresholds:
            raise ValueError(
                f"Unknown source '{source}'. Add it to SourceThresholdConfig. "
                f"Valid sources: {list(thresholds.keys())}"
            )
        return thresholds[source]


# ===========================================================================
# QUALITY THRESHOLD CONFIGURATION
# Architecture reference: lines 2250-2362
# ===========================================================================

# Late import to avoid circular dependency -- ContentType is needed for
# type_multipliers default_factory. We perform the import inside the
# dataclass __post_init__ and in functions that need it.  For the
# dataclass field defaults we use string keys and resolve at access time.


@dataclass
class ThresholdConfig:
    """
    Single source of truth for all quality thresholds.

    All QC, meta-agent, and auto-publish decisions derive from this
    configuration. Content-type-specific thresholds are computed via
    multipliers applied to the base thresholds.

    Usage::

        config = ThresholdConfig()
        threshold = config.get_pass_threshold(content_type)
        auto_threshold = config.get_auto_publish_threshold()
    """

    # -----------------------------------------------------------------
    # BASE THRESHOLDS
    # -----------------------------------------------------------------
    min_score_to_proceed: float = 8.0  # Minimum to pass QC
    auto_publish_threshold: float = 9.0  # Auto-publish without human review
    rejection_threshold: float = 5.5  # Below this = reject and restart
    revision_threshold: float = 7.0  # Below pass but above reject = revise

    # -----------------------------------------------------------------
    # CONTENT-TYPE MULTIPLIERS (not absolute values)
    # Applied to base thresholds for type-specific adjustments.
    # Keys are ContentType enum values (strings) for JSON/YAML compat.
    # -----------------------------------------------------------------
    type_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "enterprise_case": 0.90,       # 8.0 * 0.90 = 7.2 pass
        "primary_source": 0.9375,      # 8.0 * 0.9375 = 7.5 pass
        "automation_case": 0.90,       # 8.0 * 0.90 = 7.2 pass
        "community_content": 0.90,     # 8.0 * 0.90 = 7.2 pass
        "tool_release": 0.90,          # 8.0 * 0.90 = 7.2 pass
    })

    # Max revisions by content type
    max_revisions_by_type: Dict[str, int] = field(default_factory=lambda: {
        "enterprise_case": 3,
        "primary_source": 4,       # Research needs more refinement
        "automation_case": 3,
        "community_content": 2,    # Authenticity suffers from over-editing
        "tool_release": 3,
    })

    # Meta-agent iteration limits
    max_meta_iterations: int = 3

    # Maximum topic restarts before manual escalation
    max_reject_restarts: int = 2

    # Source thresholds (nested config)
    sources: SourceThresholdConfig = field(default_factory=SourceThresholdConfig)

    # -----------------------------------------------------------------
    # HELPER: resolve content_type to its string key
    # -----------------------------------------------------------------
    @staticmethod
    def _type_key(content_type: Any) -> str:
        """
        Convert a ContentType enum (or string) to the string key used in
        the multiplier/revision dicts.
        """
        if hasattr(content_type, "value"):
            return content_type.value
        return str(content_type)

    # -----------------------------------------------------------------
    # THRESHOLD ACCESSORS
    # -----------------------------------------------------------------

    def get_pass_threshold(self, content_type: Any) -> float:
        """Get effective pass threshold for content type."""
        key = self._type_key(content_type)
        multiplier = self.type_multipliers.get(key, 1.0)
        return self.min_score_to_proceed * multiplier

    def get_revision_threshold(self, content_type: Any) -> float:
        """Get threshold for revision (between reject and pass)."""
        key = self._type_key(content_type)
        multiplier = self.type_multipliers.get(key, 1.0)
        return self.revision_threshold * multiplier

    def get_rejection_threshold(self, content_type: Any) -> float:
        """Get threshold below which content is rejected."""
        key = self._type_key(content_type)
        multiplier = self.type_multipliers.get(key, 1.0)
        return self.rejection_threshold * multiplier

    def get_auto_publish_threshold(self) -> float:
        """
        Get threshold for auto-publishing.

        No type adjustment -- auto-publish requires universally high quality.
        """
        return self.auto_publish_threshold

    def get_decision(self, score: float, content_type: Any) -> str:
        """
        Get decision based on score and content type.

        Returns:
            ``"pass"`` | ``"revise"`` | ``"reject"``
        """
        pass_threshold = self.get_pass_threshold(content_type)
        reject_threshold = self.get_rejection_threshold(content_type)

        if score >= pass_threshold:
            return "pass"
        elif score >= reject_threshold:
            return "revise"
        else:
            return "reject"

    def get_max_revisions(self, content_type: Any) -> int:
        """Get max revisions for content type from centralized config."""
        key = self._type_key(content_type)
        return self.max_revisions_by_type.get(key, 3)

    def get_max_meta_iterations(self) -> int:
        """Get max meta-agent iterations."""
        return self.max_meta_iterations


# ---------------------------------------------------------------------------
# GLOBAL INSTANCES
# Import these in all modules that need threshold values.
# ---------------------------------------------------------------------------
THRESHOLD_CONFIG = ThresholdConfig()
SOURCE_THRESHOLD_CONFIG = THRESHOLD_CONFIG.sources


# ===========================================================================
# GLOBAL SETTINGS
# ===========================================================================


@dataclass
class Settings:
    """
    Global application settings.

    Loaded from ``config/settings.yaml`` when available, falling back to
    sensible defaults. Environment variables override YAML values for
    secrets and deployment-specific configuration.
    """

    # Autonomy level (1-4)
    autonomy_level: int = 3

    # LLM settings
    llm_model: str = "claude-opus-4-5-20251101"
    llm_thinking_mode: bool = True

    # Timezone
    timezone: str = "UTC"

    # Logging
    log_level: str = "INFO"
    log_dir: str = "logs"

    # Scheduling
    min_hours_between_posts: int = 6
    max_posts_per_day: int = 2

    # Thresholds
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)

    # Autonomy config (nested)
    autonomy_config: Dict[str, Any] = field(default_factory=lambda: {
        "default_level": 3,
        "auto_publish_threshold": 9.0,
        "notify_on_publish": True,
        "notify_on_modification": True,
        "content_type_levels": {},
        "auto_degradation": {
            "enabled": True,
            "consecutive_failures_threshold": 3,
            "degradation_duration_hours": 24,
        },
    })

    # Node timeouts (seconds)
    node_timeouts: Dict[str, int] = field(default_factory=lambda: {
        "scout": 120,
        "analyze": 90,
        "write": 60,
        "meta_evaluate": 45,
        "humanize": 45,
        "visualize": 180,
        "qc": 60,
    })

    @classmethod
    def from_yaml(cls, path: Optional[Path] = None) -> "Settings":
        """
        Load settings from a YAML file.

        If the file does not exist, returns an instance with all defaults.
        Environment variables override YAML values for specific keys.

        Args:
            path: Path to the YAML file. Defaults to
                ``<PROJECT_ROOT>/config/settings.yaml``.

        Returns:
            Populated Settings instance.

        Raises:
            ConfigurationError: If the YAML file exists but cannot be parsed.
        """
        path = path or PROJECT_ROOT / "config" / "settings.yaml"

        data: Dict[str, Any] = {}
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
            except yaml.YAMLError as exc:
                raise ConfigurationError(
                    f"Failed to parse settings YAML at {path}: {exc}"
                ) from exc

        # -----------------------------------------------------------------
        # Build ThresholdConfig from nested YAML section
        # -----------------------------------------------------------------
        threshold_data = data.get("thresholds", {})
        source_data = threshold_data.pop("sources", {})
        source_config = SourceThresholdConfig(**{
            k: v for k, v in source_data.items()
            if hasattr(SourceThresholdConfig, k)
        }) if source_data else SourceThresholdConfig()

        threshold_kwargs: Dict[str, Any] = {}
        for key in (
            "min_score_to_proceed",
            "auto_publish_threshold",
            "rejection_threshold",
            "revision_threshold",
            "max_meta_iterations",
            "max_reject_restarts",
        ):
            if key in threshold_data:
                threshold_kwargs[key] = threshold_data[key]
        if "type_multipliers" in threshold_data:
            threshold_kwargs["type_multipliers"] = threshold_data["type_multipliers"]
        if "max_revisions_by_type" in threshold_data:
            threshold_kwargs["max_revisions_by_type"] = threshold_data["max_revisions_by_type"]
        threshold_kwargs["sources"] = source_config
        thresholds = ThresholdConfig(**threshold_kwargs)

        # -----------------------------------------------------------------
        # Build node_timeouts (YAML + env var overrides)
        # -----------------------------------------------------------------
        node_timeouts = cls.__dataclass_fields__["node_timeouts"].default_factory()  # type: ignore[misc]
        yaml_timeouts = data.get("node_timeouts", {})
        node_timeouts.update(yaml_timeouts)

        # Environment variable overrides for node timeouts
        timeout_env_map = {
            "NODE_TIMEOUT_SCOUT": "scout",
            "NODE_TIMEOUT_ANALYZE": "analyze",
            "NODE_TIMEOUT_WRITE": "write",
            "NODE_TIMEOUT_META_EVALUATE": "meta_evaluate",
            "NODE_TIMEOUT_HUMANIZE": "humanize",
            "NODE_TIMEOUT_VISUALIZE": "visualize",
            "NODE_TIMEOUT_QC": "qc",
        }
        for env_key, timeout_key in timeout_env_map.items():
            env_val = os.environ.get(env_key)
            if env_val is not None:
                try:
                    node_timeouts[timeout_key] = int(env_val)
                except (ValueError, TypeError):
                    logger.warning(
                        "Invalid value for %s='%s', using default", env_key, env_val
                    )

        # -----------------------------------------------------------------
        # Build autonomy config
        # -----------------------------------------------------------------
        default_autonomy = cls.__dataclass_fields__["autonomy_config"].default_factory()  # type: ignore[misc]
        yaml_autonomy = data.get("autonomy", {})
        if yaml_autonomy:
            # Merge nested dicts
            for key, value in yaml_autonomy.items():
                if isinstance(value, dict) and isinstance(default_autonomy.get(key), dict):
                    default_autonomy[key].update(value)
                else:
                    default_autonomy[key] = value

        # -----------------------------------------------------------------
        # Assemble the Settings object
        # -----------------------------------------------------------------
        return cls(
            autonomy_level=data.get("autonomy_level", data.get("autonomy", {}).get("default_level", 3)),
            llm_model=data.get("llm_model", "claude-opus-4-5-20251101"),
            llm_thinking_mode=data.get("llm_thinking_mode", True),
            timezone=data.get("timezone", "UTC"),
            log_level=data.get("log_level", "INFO"),
            log_dir=data.get("log_dir", "logs"),
            min_hours_between_posts=data.get("min_hours_between_posts", 6),
            max_posts_per_day=data.get("max_posts_per_day", 2),
            thresholds=thresholds,
            autonomy_config=default_autonomy,
            node_timeouts=node_timeouts,
        )


# ===========================================================================
# SINGLETON SETTINGS ACCESSOR
# ===========================================================================

_settings_instance: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global Settings singleton.

    On first call, loads from ``config/settings.yaml`` (or defaults).
    Subsequent calls return the cached instance.

    Returns:
        The global Settings instance.
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings.from_yaml()
    return _settings_instance


def reset_settings() -> None:
    """
    Reset the cached Settings singleton.

    Useful for testing or when configuration files have been updated
    at runtime.
    """
    global _settings_instance
    _settings_instance = None


# ===========================================================================
# ENVIRONMENT VARIABLE VALIDATION
# ===========================================================================

# Required environment variables for the system to function
REQUIRED_ENV_VARS: List[str] = [
    "SUPABASE_URL",
    "SUPABASE_SERVICE_KEY",
    "ANTHROPIC_API_KEY",
]

# Optional but recommended environment variables
OPTIONAL_ENV_VARS: List[str] = [
    "PERPLEXITY_API_KEY",
    "LINKEDIN_EMAIL",
    "LINKEDIN_PASSWORD",
    "LINKEDIN_TOTP_SECRET",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "LAOZHANG_API_KEY",
    "TWITTER_BEARER_TOKEN",
]


def validate_env(strict: bool = True) -> Dict[str, bool]:
    """
    Validate that required environment variables are set.

    Args:
        strict: If ``True``, raise ``ConfigurationError`` when any required
            variable is missing. If ``False``, return the status dict
            without raising.

    Returns:
        Dict mapping variable name to presence status (``True`` if set).

    Raises:
        ConfigurationError: If ``strict=True`` and required vars are missing.
    """
    status: Dict[str, bool] = {}
    missing: List[str] = []

    for var in REQUIRED_ENV_VARS:
        present = bool(os.environ.get(var))
        status[var] = present
        if not present:
            missing.append(var)

    for var in OPTIONAL_ENV_VARS:
        status[var] = bool(os.environ.get(var))

    if strict and missing:
        raise ConfigurationError(
            f"Missing required environment variables: {missing}. "
            f"Copy .env.example to .env and fill in the values."
        )

    return status


# ===========================================================================
# DOMAIN-TO-CANDIDATE-TYPES MAPPING
# Architecture reference: lines 3209-3257
# ===========================================================================

# NOTE: Values are string lists matching ContentType.value. We use strings
# to avoid importing models at module level (which may not exist yet during
# early bootstrapping). Consumers should convert to ContentType enums.

DOMAIN_TO_CANDIDATE_TYPES: Dict[str, List[str]] = {
    # Research-focused domains
    "arxiv.org": ["primary_source"],
    "papers.ssrn.com": ["primary_source"],
    "semanticscholar.org": ["primary_source"],

    # Consultancy (publish both research AND case studies)
    "gartner.com": ["primary_source", "enterprise_case"],
    "mckinsey.com": ["primary_source", "enterprise_case"],
    "deloitte.com": ["primary_source", "enterprise_case"],
    "bcg.com": ["primary_source", "enterprise_case"],

    # Product launches
    "producthunt.com": ["tool_release"],
    "techcrunch.com": ["tool_release", "enterprise_case"],

    # Community (can surface any type)
    "reddit.com": ["community_content", "automation_case", "enterprise_case"],
    "youtube.com": ["community_content", "automation_case"],
    "news.ycombinator.com": ["community_content", "enterprise_case", "tool_release"],

    # Mixed content platforms
    "medium.com": ["automation_case", "enterprise_case", "primary_source"],
    "dev.to": ["automation_case", "community_content"],

    # Substack (general + specific high-value newsletters)
    "substack.com": ["primary_source", "community_content"],
    "simonwillison.substack.com": ["primary_source"],
    "lethain.substack.com": ["primary_source"],
    "thealgorithmicbridge.substack.com": ["primary_source"],
    "oneusefulthing.substack.com": ["primary_source"],
}


def get_domain_candidate_types(domain: str) -> List[str]:
    """
    Get candidate ContentType values for a given domain.

    Args:
        domain: Domain string (e.g. ``"arxiv.org"``).

    Returns:
        List of ContentType value strings. Returns all types if domain
        is unknown (intentional fallback with logging).
    """
    if domain in DOMAIN_TO_CANDIDATE_TYPES:
        return list(DOMAIN_TO_CANDIDATE_TYPES[domain])

    logger.warning(
        "Unknown domain '%s' -- falling back to all content types for LLM classification",
        domain,
    )
    return [
        "enterprise_case",
        "primary_source",
        "automation_case",
        "community_content",
        "tool_release",
    ]


# ===========================================================================
# URL PATTERN TYPE HINTS
# Architecture reference: lines 3260-3272
# ===========================================================================

URL_PATTERN_TYPE_HINTS: Dict[str, str] = {
    "case-study": "enterprise_case",
    "case-studies": "enterprise_case",
    "customer-story": "enterprise_case",
    "research": "primary_source",
    "insights": "primary_source",
    "n8n": "automation_case",
    "workflow": "automation_case",
    "agent": "automation_case",
    "release": "tool_release",
    "launch": "tool_release",
    "announce": "tool_release",
}


def get_url_type_hint(url: str) -> Optional[str]:
    """
    Check a URL against known patterns to get a ContentType hint.

    Args:
        url: Full URL string to check.

    Returns:
        ContentType value string if a pattern matches, ``None`` otherwise.
    """
    url_lower = url.lower()
    for pattern, content_type_value in URL_PATTERN_TYPE_HINTS.items():
        if pattern in url_lower:
            return content_type_value
    return None


# ===========================================================================
# TYPE-SPECIFIC CONTEXT LOADER
# Architecture reference: lines 11358-11471
# ===========================================================================


def load_type_context(content_type: Any) -> Dict[str, Any]:
    """
    Load all type-specific configurations when ContentType is determined.

    This context flows through the entire pipeline, providing per-type
    settings for Analyzer, Writer, Humanizer, Visual Creator, and QC Agent.

    Args:
        content_type: A ContentType enum member or its string value
            (e.g. ``"enterprise_case"``).

    Returns:
        Dict with keys: ``extraction_focus``, ``required_fields``,
        ``preferred_templates``, ``hook_styles``, ``cta_style``,
        ``humanization_intensity``, ``tone_markers``, ``avoid_markers``,
        ``visual_formats``, ``color_scheme``, ``extra_criteria``,
        ``weight_adjustments``, ``pass_threshold``.
        Returns empty dict if content_type is unrecognized.
    """
    # Resolve enum to string value
    type_key = content_type.value if hasattr(content_type, "value") else str(content_type)

    # Attempt to load hook styles from models module.
    # Falls back to empty list if models module is not yet available.
    hook_styles: Dict[str, List[str]] = {}
    try:
        from src.models import get_hook_styles_for_type as _get_hooks
        from src.models import ContentType as CT

        ct_enum = CT(type_key)
        for ct in CT:
            hooks = _get_hooks(ct)
            hook_styles[ct.value] = [h.value if hasattr(h, "value") else str(h) for h in hooks]
    except (ImportError, ValueError):
        # models.py not yet created or ContentType value invalid.
        # Use hardcoded fallback hook styles from architecture spec.
        hook_styles = _FALLBACK_HOOK_STYLES

    TYPE_CONTEXTS: Dict[str, Dict[str, Any]] = {
        "enterprise_case": {
            # Analyzer config
            "extraction_focus": [
                "company", "industry", "problem", "solution",
                "metrics", "timeline", "lessons",
            ],
            "required_fields": ["company", "metrics", "problem_statement"],

            # Writer config
            "preferred_templates": ["METRICS_HERO", "LESSONS_LEARNED", "HOW_THEY_DID_IT"],
            "hook_styles": hook_styles.get("enterprise_case", [
                "metrics", "lessons_learned", "problem_solution",
            ]),
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

        "primary_source": {
            "extraction_focus": [
                "authors", "thesis", "methodology", "findings",
                "implications", "counterintuitive",
            ],
            "required_fields": ["thesis", "key_findings", "authors"],

            "preferred_templates": ["RESEARCH_INSIGHT", "CONTRARIAN_TAKE", "FUTURE_PREDICTION"],
            "hook_styles": hook_styles.get("primary_source", [
                "contrarian", "question", "surprising_stat",
                "simplified_explainer", "debate_starter",
            ]),
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

        "automation_case": {
            "extraction_focus": [
                "task_automated", "tools_used", "workflow_steps",
                "time_saved", "code_available",
            ],
            "required_fields": ["task_automated", "tools_used", "workflow_steps"],

            "preferred_templates": ["HOW_TO_GUIDE", "TOOL_STACK_REVEAL", "AUTOMATION_STORY"],
            "hook_styles": hook_styles.get("automation_case", [
                "how_to", "time_saved", "before_after",
                "results_story", "tool_comparison",
            ]),
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

        "community_content": {
            "extraction_focus": [
                "platform", "author_credibility", "key_insights",
                "engagement_signals", "code_examples",
            ],
            "required_fields": ["platform", "key_insights"],

            "preferred_templates": ["COMMUNITY_SPOTLIGHT", "DISCUSSION_SUMMARY", "PERSONAL_STORY"],
            "hook_styles": hook_styles.get("community_content", [
                "relatable", "community_reference", "personal",
                "curated_insights", "hot_take_response", "practitioner_wisdom",
            ]),
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

        "tool_release": {
            "extraction_focus": [
                "tool_name", "company", "key_features",
                "pricing", "demo_url", "competing_tools",
            ],
            "required_fields": ["tool_name", "key_features"],

            "preferred_templates": ["PRODUCT_LAUNCH", "TOOL_COMPARISON", "FIRST_LOOK"],
            "hook_styles": hook_styles.get("tool_release", [
                "news_breaking", "feature_highlight", "comparison",
                "first_look", "implications",
            ]),
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

    context = TYPE_CONTEXTS.get(type_key, {})
    if not context:
        logger.warning(
            "No type context found for content_type='%s'. "
            "Valid types: %s",
            type_key,
            list(TYPE_CONTEXTS.keys()),
        )
    return context


# ---------------------------------------------------------------------------
# Fallback hook styles (used when src.models is not yet available)
# Matches architecture.md lines 4863-4898
# ---------------------------------------------------------------------------
_FALLBACK_HOOK_STYLES: Dict[str, List[str]] = {
    "enterprise_case": ["metrics", "lessons_learned", "problem_solution"],
    "primary_source": [
        "contrarian", "question", "surprising_stat",
        "simplified_explainer", "debate_starter",
    ],
    "automation_case": [
        "how_to", "time_saved", "before_after",
        "results_story", "tool_comparison",
    ],
    "community_content": [
        "relatable", "community_reference", "personal",
        "curated_insights", "hot_take_response", "practitioner_wisdom",
    ],
    "tool_release": [
        "news_breaking", "feature_highlight", "comparison",
        "first_look", "implications",
    ],
}


# ===========================================================================
# PUBLIC API
# ===========================================================================

__all__ = [
    # Configuration classes
    "ThresholdConfig",
    "SourceThresholdConfig",
    "Settings",
    "AutonomyLevel",
    # Global instances
    "THRESHOLD_CONFIG",
    "SOURCE_THRESHOLD_CONFIG",
    # Settings accessor
    "get_settings",
    "reset_settings",
    # Environment validation
    "validate_env",
    "REQUIRED_ENV_VARS",
    "OPTIONAL_ENV_VARS",
    # Domain / URL mappings
    "DOMAIN_TO_CANDIDATE_TYPES",
    "URL_PATTERN_TYPE_HINTS",
    "get_domain_candidate_types",
    "get_url_type_hint",
    # Type context
    "load_type_context",
    # Constants
    "PROJECT_ROOT",
]
