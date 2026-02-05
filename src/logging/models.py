"""Logging data models: LogLevel, LogComponent, LogEntry."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class LogLevel(Enum):
    """Log levels with numeric values for severity comparison.

    Uses integer values so that severity comparison works correctly.
    String comparison would fail (e.g., "debug" > "critical" lexicographically).
    """

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    @property
    def name_str(self) -> str:
        """Get lowercase name for display/serialization."""
        return self.name.lower()


class LogComponent(Enum):
    """All system components that can produce logs."""

    # Core agents
    ORCHESTRATOR = "orchestrator"
    TREND_SCOUT = "trend_scout"
    ANALYZER = "analyzer"
    WRITER = "writer"
    HUMANIZER = "humanizer"
    VISUAL_CREATOR = "visual_creator"
    QC_AGENT = "qc_agent"
    META_AGENT = "meta_agent"
    EVALUATOR = "evaluator"

    # Supporting components
    VALIDATOR = "validator"
    LEARNER = "learner"
    NANO_BANANA = "nano_banana"
    PERPLEXITY = "perplexity"
    HUMAN_APPROVAL = "human_approval"
    AB_TEST = "ab_test"
    STARTUP = "startup"
    CONFIG = "config"

    # Infrastructure
    MODIFICATION_SAFETY = "modification_safety"
    AUTHOR_PROFILE = "author_profile"
    SCHEDULER = "scheduler"
    LINKEDIN_CLIENT = "linkedin_client"
    ANALYTICS = "analytics"
    TELEGRAM = "telegram"

    # Additional components
    CIRCUIT_BREAKER = "circuit_breaker"
    PIPELINE_RECOVERY = "pipeline_recovery"
    CODE_GENERATION = "code_generation"
    ROLLBACK_MANAGER = "rollback_manager"
    RESEARCH_AGENT = "research_agent"
    CODE_EVOLUTION = "code_evolution"
    SELF_MODIFICATION = "self_modification"
    DATABASE = "database"
    PHOTO_SELECTOR = "photo_selector"


@dataclass
class LogEntry:
    """Structured log entry.

    Represents a single log event with context, optional error details,
    and performance timing. Supports serialization to JSON, dict, and
    human-readable text formats.
    """

    # Required fields
    timestamp: datetime
    level: LogLevel
    component: LogComponent
    message: str

    # Context
    run_id: Optional[str] = None
    post_id: Optional[str] = None

    # Additional data
    data: Dict[str, Any] = field(default_factory=dict)

    # Error details
    error_type: Optional[str] = None
    error_traceback: Optional[str] = None

    # Performance
    duration_ms: Optional[int] = None

    def to_json(self) -> str:
        """Serialize to JSON string for file logging."""
        return json.dumps(
            {
                "timestamp": self.timestamp.isoformat(),
                "level": self.level.value,
                "level_name": self.level.name_str,
                "component": self.component.value,
                "message": self.message,
                "run_id": self.run_id,
                "post_id": self.post_id,
                "data": self.data,
                "error_type": self.error_type,
                "error_traceback": self.error_traceback,
                "duration_ms": self.duration_ms,
            },
            ensure_ascii=False,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for Supabase insertion."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "level_name": self.level.name_str,
            "component": self.component.value,
            "message": self.message,
            "run_id": self.run_id,
            "post_id": self.post_id,
            "data": self.data,
            "error_type": self.error_type,
            "error_traceback": self.error_traceback,
            "duration_ms": self.duration_ms,
        }

    def to_readable(self) -> str:
        """Human-readable format for Telegram/console output."""
        time_str = self.timestamp.strftime("%H:%M:%S")
        level_indicators = {
            LogLevel.DEBUG: "[DEBUG]",
            LogLevel.INFO: "[INFO]",
            LogLevel.WARNING: "[WARN]",
            LogLevel.ERROR: "[ERROR]",
            LogLevel.CRITICAL: "[CRIT]",
        }
        indicator = level_indicators.get(self.level, "[???]")
        msg = f"{indicator} [{time_str}] [{self.component.value}] {self.message}"
        if self.duration_ms:
            msg += f" ({self.duration_ms}ms)"
        return msg
