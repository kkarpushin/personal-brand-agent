"""
Shared utility functions used throughout the LinkedIn Super Agent codebase.

Provides:
    - utc_now(): Timezone-aware UTC datetime (for Supabase TIMESTAMPTZ columns)
    - generate_id(): UUID4 string generator (for database primary keys)
    - ensure_utc(dt): Convert any datetime to timezone-aware UTC
    - @with_retry: Decorator with exponential backoff for transient failures
"""

from datetime import datetime, timezone
import uuid
import asyncio
import logging
import time as time_module
from functools import wraps
from typing import Callable, TypeVar, Any, Tuple, Type, Optional

from src.exceptions import RetryExhaustedError

# ---------------------------------------------------------------------------
# Type variable for generic return types in the retry decorator
# ---------------------------------------------------------------------------
T = TypeVar("T")


# ===========================================================================
# TIMEZONE UTILITIES
# All timestamps stored in Supabase must be timezone-aware (TIMESTAMPTZ)
# ===========================================================================


def utc_now() -> datetime:
    """
    Get current UTC time as timezone-aware datetime.

    ALWAYS use this instead of ``datetime.now()`` or ``datetime.utcnow()``
    for Supabase compatibility (TIMESTAMPTZ columns).

    Returns:
        Timezone-aware datetime in UTC.
    """
    return datetime.now(timezone.utc)


def generate_id() -> str:
    """
    Generate a unique ID for database records.

    Uses UUID4 which is suitable for:
    - Database primary keys
    - Unique identifiers for modifications, scheduled posts, etc.

    Returns:
        A unique UUID string (compatible with Supabase UUID type).
    """
    return str(uuid.uuid4())


def ensure_utc(dt: datetime) -> datetime:
    """
    Ensure a datetime is timezone-aware in UTC.

    Args:
        dt: Datetime to convert (naive or aware).

    Returns:
        Timezone-aware datetime in UTC.
    """
    if dt.tzinfo is None:
        # Assume naive datetime is UTC
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ===========================================================================
# RETRY DECORATOR WITH EXPONENTIAL BACKOFF
# Compliant with fail-fast philosophy: retries are for transient failures
# (rate limits, timeouts). Eventually raises if all attempts fail.
# ===========================================================================


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 2.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    operation_name: Optional[str] = None,
) -> Callable:
    """
    Decorator for retry logic with exponential backoff.

    COMPLIANT with fail-fast philosophy:
    - Retries are for transient failures (rate limits, timeouts).
    - Eventually raises ``RetryExhaustedError`` if all attempts fail.
    - Logs each retry attempt for debugging.

    Works with both synchronous and asynchronous functions. The decorator
    automatically detects whether the wrapped function is a coroutine and
    applies the appropriate wrapper.

    Args:
        max_attempts: Maximum number of attempts (default ``3``).
        base_delay: Initial delay in seconds before the first retry
            (default ``2.0``). Subsequent delays grow exponentially:
            ``base_delay * (2 ** attempt)``.
        retryable_exceptions: Tuple of exception types that should trigger
            a retry. Any exception **not** in this tuple will propagate
            immediately without retrying.
        operation_name: Human-readable name used in log messages. If
            ``None``, the wrapped function's ``__name__`` is used.

    Raises:
        RetryExhaustedError: When all retry attempts have been exhausted.
            The original exception is chained as the ``last_error``
            attribute and as the ``__cause__``.

    Usage::

        @with_retry(max_attempts=3, base_delay=2.0)
        async def call_claude(prompt: str) -> str:
            return await llm.generate(prompt)

        @with_retry(
            max_attempts=5,
            base_delay=1.0,
            retryable_exceptions=(RateLimitError, TimeoutError),
        )
        def fetch_data(url: str) -> dict:
            return requests.get(url).json()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        op_name = operation_name or func.__name__

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            last_error: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_error = e
                    if attempt < max_attempts:
                        delay = base_delay * (2 ** (attempt - 1))
                        logging.warning(
                            "[RETRY] %s attempt %d/%d failed: %s. "
                            "Retrying in %.1fs...",
                            op_name,
                            attempt,
                            max_attempts,
                            e,
                            delay,
                        )
                        await asyncio.sleep(delay)
                    else:
                        logging.error(
                            "[RETRY EXHAUSTED] %s failed after %d attempts: %s",
                            op_name,
                            max_attempts,
                            e,
                        )
            raise RetryExhaustedError(
                op_name, max_attempts, last_error  # type: ignore[arg-type]
            )

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            last_error: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_error = e
                    if attempt < max_attempts:
                        delay = base_delay * (2 ** (attempt - 1))
                        logging.warning(
                            "[RETRY] %s attempt %d/%d failed: %s. "
                            "Retrying in %.1fs...",
                            op_name,
                            attempt,
                            max_attempts,
                            e,
                            delay,
                        )
                        time_module.sleep(delay)
                    else:
                        logging.error(
                            "[RETRY EXHAUSTED] %s failed after %d attempts: %s",
                            op_name,
                            max_attempts,
                            e,
                        )
            raise RetryExhaustedError(
                op_name, max_attempts, last_error  # type: ignore[arg-type]
            )

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator
