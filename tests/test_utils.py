"""
Tests for src.utils module.

Covers:
    - utc_now(): timezone-aware UTC datetime
    - generate_id(): UUID4 string generation
    - ensure_utc(): naive/aware datetime UTC conversion
    - with_retry(): exponential backoff decorator for sync and async functions
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import patch, AsyncMock
from uuid import UUID

import pytest

from src.exceptions import RetryExhaustedError
from src.utils import utc_now, generate_id, ensure_utc, with_retry


# ===========================================================================
# utc_now()
# ===========================================================================


def test_utc_now_returns_timezone_aware_utc():
    """utc_now() must return a datetime whose tzinfo is UTC."""
    result = utc_now()
    assert result.tzinfo is not None
    assert result.tzinfo == timezone.utc


def test_utc_now_returns_current_time():
    """utc_now() must return a time within 2 seconds of datetime.now(utc)."""
    before = datetime.now(timezone.utc)
    result = utc_now()
    after = datetime.now(timezone.utc)
    assert before <= result <= after
    assert (after - before) < timedelta(seconds=2)


# ===========================================================================
# generate_id()
# ===========================================================================


def test_generate_id_returns_valid_uuid4_string():
    """generate_id() must return a string that parses as a valid UUID4."""
    result = generate_id()
    assert isinstance(result, str)
    parsed = UUID(result)
    # UUID version 4
    assert parsed.version == 4


def test_generate_id_returns_unique_values():
    """Successive calls to generate_id() must produce distinct values."""
    ids = {generate_id() for _ in range(100)}
    assert len(ids) == 100


# ===========================================================================
# ensure_utc()
# ===========================================================================


def test_ensure_utc_naive_datetime_adds_utc():
    """A naive (tzinfo=None) datetime gets UTC attached via replace."""
    naive = datetime(2025, 6, 15, 12, 0, 0)
    assert naive.tzinfo is None

    result = ensure_utc(naive)

    assert result.tzinfo == timezone.utc
    # The date/time components must be unchanged (not shifted).
    assert result.year == 2025
    assert result.month == 6
    assert result.day == 15
    assert result.hour == 12
    assert result.minute == 0
    assert result.second == 0


def test_ensure_utc_already_utc_returns_same_value():
    """A UTC-aware datetime is returned unchanged."""
    aware = datetime(2025, 1, 1, 8, 30, 0, tzinfo=timezone.utc)
    result = ensure_utc(aware)
    assert result == aware
    assert result.tzinfo == timezone.utc


def test_ensure_utc_non_utc_aware_converts_to_utc():
    """A timezone-aware datetime in a non-UTC zone is converted to UTC."""
    # UTC+5
    plus_five = timezone(timedelta(hours=5))
    dt_plus5 = datetime(2025, 6, 15, 17, 0, 0, tzinfo=plus_five)

    result = ensure_utc(dt_plus5)

    assert result.tzinfo == timezone.utc
    # 17:00 UTC+5 == 12:00 UTC
    assert result.hour == 12
    assert result.day == 15


# ===========================================================================
# with_retry() -- synchronous functions
# ===========================================================================


@patch("src.utils.time_module.sleep")
def test_with_retry_sync_succeeds_first_try(mock_sleep):
    """Sync function that succeeds immediately is not retried."""

    @with_retry(max_attempts=3, base_delay=2.0)
    def succeed():
        return "ok"

    result = succeed()

    assert result == "ok"
    mock_sleep.assert_not_called()


@patch("src.utils.time_module.sleep")
def test_with_retry_sync_retries_and_succeeds_second_try(mock_sleep):
    """Sync function that fails once then succeeds is retried exactly once."""
    call_count = 0

    @with_retry(max_attempts=3, base_delay=2.0)
    def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("transient")
        return "recovered"

    result = flaky()

    assert result == "recovered"
    assert call_count == 2
    # Delay for attempt 1 failure: base_delay * 2^(1-1) = 2.0 * 1 = 2.0
    mock_sleep.assert_called_once_with(2.0)


@patch("src.utils.time_module.sleep")
def test_with_retry_sync_exhausts_retries(mock_sleep):
    """Sync function that always fails raises RetryExhaustedError after max_attempts."""

    @with_retry(max_attempts=3, base_delay=1.0, operation_name="test_op")
    def always_fail():
        raise RuntimeError("permanent")

    with pytest.raises(RetryExhaustedError) as exc_info:
        always_fail()

    err = exc_info.value
    assert err.operation == "test_op"
    assert err.attempts == 3
    assert isinstance(err.last_error, RuntimeError)
    assert str(err.last_error) == "permanent"

    # Should have slept twice (after attempt 1 and attempt 2, not after last).
    assert mock_sleep.call_count == 2
    # Attempt 1 delay: 1.0 * 2^0 = 1.0
    # Attempt 2 delay: 1.0 * 2^1 = 2.0
    mock_sleep.assert_any_call(1.0)
    mock_sleep.assert_any_call(2.0)


# ===========================================================================
# with_retry() -- asynchronous functions
# ===========================================================================


@pytest.mark.asyncio
async def test_with_retry_async_succeeds_first_try():
    """Async function that succeeds immediately is not retried."""
    with patch("src.utils.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:

        @with_retry(max_attempts=3, base_delay=2.0)
        async def succeed():
            return "ok"

        result = await succeed()

        assert result == "ok"
        mock_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_with_retry_async_retries_and_succeeds_second_try():
    """Async function that fails once then succeeds is retried exactly once."""
    with patch("src.utils.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        call_count = 0

        @with_retry(max_attempts=3, base_delay=2.0)
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("transient")
            return "recovered"

        result = await flaky()

        assert result == "recovered"
        assert call_count == 2
        # Delay for attempt 1 failure: 2.0 * 2^0 = 2.0
        mock_sleep.assert_called_once_with(2.0)


@pytest.mark.asyncio
async def test_with_retry_async_exhausts_retries():
    """Async function that always fails raises RetryExhaustedError after max_attempts."""
    with patch("src.utils.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:

        @with_retry(max_attempts=3, base_delay=1.0, operation_name="async_op")
        async def always_fail():
            raise RuntimeError("permanent")

        with pytest.raises(RetryExhaustedError) as exc_info:
            await always_fail()

        err = exc_info.value
        assert err.operation == "async_op"
        assert err.attempts == 3
        assert isinstance(err.last_error, RuntimeError)
        assert str(err.last_error) == "permanent"

        # Slept after attempt 1 and attempt 2 (not after the final attempt).
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)


# ===========================================================================
# with_retry() -- edge cases
# ===========================================================================


@patch("src.utils.time_module.sleep")
def test_with_retry_respects_retryable_exceptions(mock_sleep):
    """Non-retryable exceptions propagate immediately without retrying."""

    @with_retry(
        max_attempts=3,
        base_delay=1.0,
        retryable_exceptions=(ValueError,),
    )
    def raise_type_error():
        raise TypeError("not retryable")

    with pytest.raises(TypeError, match="not retryable"):
        raise_type_error()

    # No sleep should have been called because the exception was not retryable.
    mock_sleep.assert_not_called()


def test_with_retry_preserves_function_name():
    """The decorated function preserves the original __name__ via functools.wraps."""

    @with_retry(max_attempts=2)
    def my_special_function():
        pass

    assert my_special_function.__name__ == "my_special_function"


def test_with_retry_preserves_async_function_name():
    """The decorated async function preserves the original __name__ via functools.wraps."""

    @with_retry(max_attempts=2)
    async def my_async_function():
        pass

    assert my_async_function.__name__ == "my_async_function"
