"""
Custom exception classes for the LinkedIn Super Agent system.

This module defines all exception classes used throughout the codebase.
Exceptions follow the fail-fast philosophy: no fallbacks, surface errors
immediately with clear context for debugging.

Hierarchy:
    Exception
    +-- AgentBaseError (base for all agent-specific errors)
    |   +-- LinkedInRateLimitError
    |   +-- LinkedInSessionExpiredError
    |   +-- LinkedInAPIError
    |   +-- ImageGenerationError
    |   +-- SchedulingConflictError
    |   +-- TrendScoutError
    |   +-- TopicSelectionError
    |   +-- AnalyzerError
    |   +-- WriterError
    |   +-- VisualizerError
    +-- ValidationError (ValueError)
    |   +-- MetadataTypeMismatchError
    |   +-- ExtractionTypeMismatchError
    |   +-- BoundaryValidationError
    +-- DatabaseError
    +-- ConfigurationError
    |   +-- ConfigurationBackupError
    |   +-- ConfigurationCorruptedError
    |   +-- ConfigurationAccessError
    |   +-- ConfigurationWriteError
    +-- SecurityError
    +-- RetryExhaustedError
    +-- CircuitOpenError
    +-- NodeTimeoutError
"""

from typing import List


# =============================================================================
# BASE EXCEPTION
# =============================================================================


class AgentBaseError(Exception):
    """Base exception for all agent-related errors."""

    pass


# =============================================================================
# CORE EXCEPTIONS
# =============================================================================


class ValidationError(ValueError):
    """Raised when input validation fails."""

    pass


class DatabaseError(Exception):
    """Raised when database operations fail."""

    pass


class ConfigurationError(Exception):
    """Raised when system configuration is invalid."""

    pass


class SecurityError(Exception):
    """Raised when a security validation fails (e.g., risk level spoofing)."""

    pass


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted.

    Attributes:
        operation: Name of the operation that was retried.
        attempts: Total number of attempts made.
        last_error: The last exception raised before giving up.
    """

    def __init__(self, operation: str, attempts: int, last_error: Exception):
        self.operation = operation
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"{operation} failed after {attempts} attempts. "
            f"Last error: {last_error}"
        )


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and request is blocked."""

    pass


class NodeTimeoutError(Exception):
    """Raised when a pipeline node exceeds its timeout.

    Attributes:
        node_name: Name of the node that timed out.
        timeout: Timeout duration in seconds.
    """

    def __init__(self, node_name: str, timeout: int):
        self.node_name = node_name
        self.timeout = timeout
        super().__init__(f"Node '{node_name}' timed out after {timeout} seconds")


# =============================================================================
# LINKEDIN API EXCEPTIONS
# =============================================================================


class LinkedInRateLimitError(AgentBaseError):
    """Raised when LinkedIn rate limit is hit."""

    pass


class LinkedInSessionExpiredError(AgentBaseError):
    """Raised when LinkedIn session expires."""

    pass


class LinkedInAPIError(AgentBaseError):
    """Raised for general LinkedIn API errors."""

    pass


# =============================================================================
# AGENT PIPELINE EXCEPTIONS
# =============================================================================


class TrendScoutError(AgentBaseError):
    """Raised when trend scout fails to find topics."""

    pass


class TopicSelectionError(AgentBaseError):
    """Raised when topic selection fails."""

    pass


class AnalyzerError(AgentBaseError):
    """Raised when analyzer fails to extract required data."""

    pass


class WriterError(AgentBaseError):
    """Raised when writer fails to generate draft."""

    pass


class VisualizerError(AgentBaseError):
    """Raised when visual creator fails to generate assets."""

    pass


class ImageGenerationError(AgentBaseError):
    """Raised when image generation fails."""

    pass


class SchedulingConflictError(AgentBaseError):
    """Raised when scheduling slot is unavailable."""

    pass


# =============================================================================
# VALIDATION EXCEPTIONS
# =============================================================================


class MetadataTypeMismatchError(ValidationError):
    """Raised when metadata.type doesn't match topic.content_type."""

    pass


class ExtractionTypeMismatchError(ValidationError):
    """Raised when extraction.type doesn't match content_type."""

    pass


class BoundaryValidationError(ValidationError):
    """Raised when data fails validation at an agent boundary.

    Attributes:
        source_agent: Name of the agent that produced the data.
        target_agent: Name of the agent that rejected the data.
        issues: List of validation issues found.
    """

    def __init__(self, source_agent: str, target_agent: str, issues: List[str]):
        self.source_agent = source_agent
        self.target_agent = target_agent
        self.issues = issues
        super().__init__(
            f"Validation failed at {source_agent} -> {target_agent} "
            f"boundary: {issues}"
        )


# =============================================================================
# CONFIGURATION EXCEPTIONS
# =============================================================================


class ConfigurationBackupError(ConfigurationError):
    """Raised when configuration backup fails."""

    pass


class ConfigurationCorruptedError(ConfigurationError):
    """Raised when configuration file is corrupted."""

    pass


class ConfigurationAccessError(ConfigurationError):
    """Raised when configuration file cannot be accessed."""

    pass


class ConfigurationWriteError(ConfigurationError):
    """Raised when configuration file cannot be written."""

    pass


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Base
    "AgentBaseError",
    # Core
    "ValidationError",
    "DatabaseError",
    "ConfigurationError",
    "SecurityError",
    "RetryExhaustedError",
    "CircuitOpenError",
    "NodeTimeoutError",
    # LinkedIn API
    "LinkedInRateLimitError",
    "LinkedInSessionExpiredError",
    "LinkedInAPIError",
    # Agent Pipeline
    "TrendScoutError",
    "TopicSelectionError",
    "AnalyzerError",
    "WriterError",
    "VisualizerError",
    "ImageGenerationError",
    "SchedulingConflictError",
    # Validation
    "MetadataTypeMismatchError",
    "ExtractionTypeMismatchError",
    "BoundaryValidationError",
    # Configuration
    "ConfigurationBackupError",
    "ConfigurationCorruptedError",
    "ConfigurationAccessError",
    "ConfigurationWriteError",
]
