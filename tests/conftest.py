"""Shared fixtures for LinkedIn Super Agent test suite."""

import asyncio
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Ensure we don't hit real APIs during tests
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _block_env_keys(monkeypatch):
    """Clear all API keys so tests never hit real services."""
    keys = [
        "SUPABASE_URL",
        "SUPABASE_SERVICE_KEY",
        "ANTHROPIC_API_KEY",
        "PERPLEXITY_API_KEY",
        "LINKEDIN_EMAIL",
        "LINKEDIN_PASSWORD",
        "LINKEDIN_TOTP_SECRET",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "LAOZHANG_API_KEY",
        "TWITTER_BEARER_TOKEN",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Async event loop
# ---------------------------------------------------------------------------
@pytest.fixture
def event_loop():
    """Create a new event loop for each test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Common datetime fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_utc_now():
    """A fixed UTC datetime for deterministic tests."""
    return datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Mock Supabase client
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_supabase_client():
    """A mock Supabase async client."""
    client = AsyncMock()
    # table().select().execute() chain
    table_mock = MagicMock()
    table_mock.select.return_value = table_mock
    table_mock.insert.return_value = table_mock
    table_mock.update.return_value = table_mock
    table_mock.delete.return_value = table_mock
    table_mock.eq.return_value = table_mock
    table_mock.gte.return_value = table_mock
    table_mock.lte.return_value = table_mock
    table_mock.order.return_value = table_mock
    table_mock.limit.return_value = table_mock
    table_mock.range.return_value = table_mock
    table_mock.single.return_value = table_mock

    async def mock_execute():
        return MagicMock(data=[], count=0)

    table_mock.execute = mock_execute
    client.table.return_value = table_mock
    return client
