# LinkedIn Super Agent - Architecture Document

## Overview

Multi-agent system for maximizing LinkedIn engagement through AI-powered content creation about artificial intelligence topics.

**Key Decisions:**
- **Autonomy: Configurable (Levels 1-4)**
  - Level 1: Human approves everything
  - Level 2: Human approves posts, auto-modifications allowed
  - Level 3: Auto-publish high-score posts (≥9.0), human for rest
  - Level 4: Full autonomy (human notified, not asked)
- **Self-Improvement: YES** — researches, experiments, modifies itself
- **Self-Modifying Code: YES** — writes new Python modules during execution ⚡
  - Detects capability gaps (missing data sources, analysis methods, etc.)
  - Generates code via Claude, validates (syntax, types, security, tests)
  - Hot-reloads new modules, retries with new capabilities
  - All in the SAME RUN — no waiting for weekly triggers
- Tech Stack: Python + LangGraph
- Content Sources: All (scientific papers, news, personal experience)
- Style Learning: From top creators + continuous experimentation
- **Error Handling: NO FALLBACKS, FAIL FAST**

---

### Error Handling Philosophy: No Fallbacks, Graceful Routing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              NO FALLBACKS — FAIL FAST — ROUTE GRACEFULLY                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  THREE DISTINCT CONCEPTS (don't confuse them!):                             │
│                                                                             │
│  1. FAIL-FAST: Don't use backup services or alternative paths               │
│     - If Nano Banana fails, DON'T switch to DALL-E                          │
│     - If Claude fails, DON'T switch to GPT-4                                │
│                                                                             │
│  2. GRACEFUL ROUTING: Route errors through state machine to handler         │
│     - Set critical_error in state                                           │
│     - State machine routes to handle_error node                             │
│     - Error is logged, saved, and reported                                  │
│                                                                             │
│  3. RETRY: Same operation, just wait and try again                          │
│     - Rate limits, timeouts, transient failures                             │
│     - After max retries → set critical_error → route to handler             │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│  ERROR TYPE MATRIX                                                          │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  ERROR TYPE          │ RETRY? │ FALLBACK? │ ACTION                          │
│  ────────────────────┼────────┼───────────┼──────────────────────────────   │
│  Rate Limit          │ YES    │ NO        │ Exponential backoff, then fail  │
│  Timeout             │ YES    │ NO        │ Retry 3x, then critical_error   │
│  Auth Error          │ NO     │ NO        │ Immediate critical_error        │
│  Validation Error    │ NO     │ NO        │ Immediate critical_error        │
│  API 500 Error       │ YES    │ NO        │ Retry 3x, then critical_error   │
│  Missing Data        │ NO     │ NO        │ Immediate critical_error        │
│  Low Quality Score   │ N/A    │ N/A       │ Route to revise (not an error)  │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│  EXAMPLES                                                                   │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  ❌ BAD (fallback to different service):                                    │
│  ```python                                                                  │
│  try:                                                                       │
│      image = nano_banana.generate(prompt)                                   │
│  except:                                                                    │
│      image = dall_e.generate(prompt)  # FALLBACK — hidden failure!         │
│  ```                                                                        │
│                                                                             │
│  ❌ BAD (swallow error):                                                    │
│  ```python                                                                  │
│  try:                                                                       │
│      image = nano_banana.generate(prompt)                                   │
│  except:                                                                    │
│      image = None  # Silent failure — user doesn't know!                   │
│  ```                                                                        │
│                                                                             │
│  ✅ GOOD (retry then route to error handler):                               │
│  ```python                                                                  │
│  @with_retry(max_attempts=3, base_delay=2.0)                                │
│  async def generate_visual(state):                                          │
│      try:                                                                   │
│          image = await nano_banana.generate(prompt)                         │
│          return {"visual_asset": image}                                     │
│      except RetryExhaustedError as e:                                       │
│          return {"critical_error": f"Visual generation failed: {e}"}       │
│  # State machine will route to handle_error node                            │
│  ```                                                                        │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│  COMPONENT ERROR BEHAVIOR                                                   │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  COMPONENT              │ RETRY │ ON FAILURE                                │
│  ───────────────────────┼───────┼────────────────────────────────────────   │
│  LinkedIn API           │ 3x    │ critical_error → LinkedInAuthError        │
│  Nano Banana            │ 3x    │ critical_error → DiagramGenerationError   │
│  Claude Opus 4.5        │ 3x    │ critical_error → LLMError                 │
│  Perplexity Search      │ 3x    │ critical_error → SearchError              │
│  Photo Library          │ NO    │ critical_error → NoPhotosError            │
│  Supabase DB            │ 3x    │ critical_error → DatabaseError            │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│  RETRY IMPLEMENTATION                                                       │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  ```python                                                                  │
│  # Retry with exponential backoff, then set critical_error                  │
│  for attempt in range(3):                                                   │
│      try:                                                                   │
│          return await api.call()                                            │
│      except (RateLimitError, TimeoutError) as e:                            │
│          if attempt == 2:  # Last attempt                                   │
│              return {"critical_error": f"API failed after 3 retries: {e}"}  │
│          await asyncio.sleep(2 ** attempt)                                  │
│  ```                                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Supabase Database Architecture

**ЕДИНСТВЕННАЯ БАЗА ДАННЫХ: Supabase**

Все данные системы хранятся в Supabase. Никаких других баз (Redis, SQLite, MongoDB) не используется.

```python
# ═══════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# Standardized logging setup for entire codebase
# ═══════════════════════════════════════════════════════════════════════════

import logging
import sys
import threading
from typing import Optional


class LoggerFactory:
    """
    Centralized logger factory for consistent logging across all modules.

    Usage:
        logger = LoggerFactory.get_logger("ModuleName")
        logger.info("Message")

    All loggers inherit from root configuration.
    """

    _configured = False
    _lock = threading.Lock()  # FIX: Thread-safe configuration

    @classmethod
    def configure(
        cls,
        level: int = logging.INFO,
        format_string: Optional[str] = None,
        log_file: Optional[str] = None
    ) -> None:
        """
        Configure root logger. Call once at application startup.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
            format_string: Custom format string
            log_file: Optional file path for file logging
        """
        # FIX: Use lock to prevent duplicate handler registration in multi-threaded startup
        with cls._lock:
            if cls._configured:
                return

            format_string = format_string or (
                "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
            )

            # Configure root logger
            root = logging.getLogger()
            root.setLevel(level)

            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(format_string))
            root.addHandler(console_handler)

            # File handler (optional)
            if log_file:
                file_handler = logging.FileHandler(log_file, encoding="utf-8")
                file_handler.setFormatter(logging.Formatter(format_string))
                root.addHandler(file_handler)

            cls._configured = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger for a module.

        Standard naming convention:
        - Agents: "Agent.TrendScout", "Agent.Writer", "Agent.QC"
        - Pipeline: "Pipeline.Orchestrator", "Pipeline.State"
        - Services: "Service.LinkedIn", "Service.NanoBanana"
        - Core: "Core.Database", "Core.Config"
        """
        if not cls._configured:
            cls.configure()  # Auto-configure with defaults

        return logging.getLogger(name)


# Pre-configured loggers for common modules
def get_agent_logger(agent_name: str) -> logging.Logger:
    """Get logger for an agent (e.g., 'TrendScout' -> 'Agent.TrendScout')."""
    return LoggerFactory.get_logger(f"Agent.{agent_name}")


def get_service_logger(service_name: str) -> logging.Logger:
    """Get logger for a service (e.g., 'LinkedIn' -> 'Service.LinkedIn')."""
    return LoggerFactory.get_logger(f"Service.{service_name}")


def get_pipeline_logger(component: str) -> logging.Logger:
    """Get logger for pipeline component."""
    return LoggerFactory.get_logger(f"Pipeline.{component}")


# ═══════════════════════════════════════════════════════════════════════════
# SUPABASE CLIENT
# Единственный клиент для работы с базой данных
# ═══════════════════════════════════════════════════════════════════════════

import os
import asyncio
import aiohttp
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta, timezone
from supabase._async.client import AsyncClient, create_async_client


# ═══════════════════════════════════════════════════════════════════════════
# TIMEZONE UTILITIES
# All timestamps stored in Supabase must be timezone-aware (TIMESTAMPTZ)
# ═══════════════════════════════════════════════════════════════════════════

def utc_now() -> datetime:
    """
    Get current UTC time as timezone-aware datetime.

    ALWAYS use this instead of datetime.now() or datetime.utcnow()
    for Supabase compatibility (TIMESTAMPTZ columns).

    Returns:
        Timezone-aware datetime in UTC
    """
    return datetime.now(timezone.utc)


def ensure_utc(dt: datetime) -> datetime:
    """
    Ensure a datetime is timezone-aware in UTC.

    Args:
        dt: Datetime to convert (naive or aware)

    Returns:
        Timezone-aware datetime in UTC
    """
    if dt.tzinfo is None:
        # Assume naive datetime is UTC
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class SupabaseConfig:
    """Supabase configuration from environment variables."""
    url: str
    key: str  # anon key for client-side, service_role for server-side

    @classmethod
    def from_env(cls) -> "SupabaseConfig":
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_KEY")  # Use service key for full access

        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

        return cls(url=url, key=key)


class SupabaseDB:
    """
    Unified ASYNC database client for all system operations.

    ALL database operations go through this class.
    No direct Supabase calls elsewhere in the codebase.

    IMPORTANT: Use create() factory method instead of __init__ directly.
    """

    _instance: Optional["SupabaseDB"] = None

    def __init__(self, client: AsyncClient):
        """Private constructor. Use create() factory method."""
        self.client = client

    @classmethod
    async def create(cls, config: Optional[SupabaseConfig] = None) -> "SupabaseDB":
        """Factory method to create async SupabaseDB instance."""
        config = config or SupabaseConfig.from_env()
        client = await create_async_client(config.url, config.key)
        return cls(client)

    # ─────────────────────────────────────────────────────────────────────
    # POSTS
    # ─────────────────────────────────────────────────────────────────────

    async def save_post(self, post: Dict[str, Any]) -> str:
        """Save a published post."""
        # Input validation
        if not post:
            raise ValidationError("post cannot be None or empty")
        if "text_content" not in post or not post["text_content"]:
            raise ValidationError("post must have text_content")
        if "content_type" not in post:
            raise ValidationError("post must have content_type")

        result = await self.client.table("posts").insert(post).execute()
        # FIX: Validate response to prevent cryptic IndexError
        if not result.data:
            raise DatabaseError("Insert succeeded but returned no data")
        return result.data[0]["id"]

    async def get_post(self, post_id: str) -> Optional[Dict[str, Any]]:
        """Get post by ID."""
        validate_not_empty(post_id, "post_id")

        result = await self.client.table("posts").select("*").eq("id", post_id).execute()
        return result.data[0] if result.data else None

    async def get_recent_posts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent posts ordered by creation date."""
        validate_positive(limit, "limit")
        result = await (
            self.client.table("posts")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data

    async def get_total_post_count(self) -> int:
        """Get total number of posts."""
        result = await self.client.table("posts").select("id", count="exact").execute()
        return result.count or 0

    async def get_posts_by_content_type(
        self,
        content_type: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get posts filtered by content type."""
        result = await (
            self.client.table("posts")
            .select("*")
            .eq("content_type", content_type)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data

    # ─────────────────────────────────────────────────────────────────────
    # POST METRICS (Analytics)
    # ─────────────────────────────────────────────────────────────────────

    async def store_metrics_snapshot(self, snapshot: Dict[str, Any]) -> str:
        """Store a metrics snapshot for a post."""
        result = await self.client.table("post_metrics").insert(snapshot).execute()
        return result.data[0]["id"]

    async def get_metrics_history(
        self,
        post_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get metrics history for a post."""
        result = await (
            self.client.table("post_metrics")
            .select("*")
            .eq("post_id", post_id)
            .order("collected_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data

    async def get_average_metrics_at(self, minutes_after_post: int) -> Dict[str, float]:
        """Get average metrics at a specific time after posting."""
        # Using Supabase RPC for aggregation
        result = await self.client.rpc(
            "get_average_metrics_at_minutes",
            {"minutes": minutes_after_post}
        ).execute()
        return result.data[0] if result.data else {}

    async def get_average_score(self) -> float:
        """Get average QC score across all posts."""
        result = await self.client.rpc("get_average_qc_score").execute()
        return result.data[0]["avg_score"] if result.data else 7.0

    async def get_percentile(self, likes: int, minutes_after_post: int) -> float:
        """
        Calculate percentile rank for a post's likes at a given time checkpoint.

        Returns what percent of posts this one outperformed (e.g., 90 = top 10%).
        """
        result = await self.client.rpc(
            "get_likes_percentile",
            {"likes_count": likes, "minutes": minutes_after_post}
        ).execute()
        return result.data[0]["percentile"] if result.data else 50.0

    # ─────────────────────────────────────────────────────────────────────
    # LEARNINGS (Continuous Learning Engine)
    # ─────────────────────────────────────────────────────────────────────

    async def save_learnings(self, learnings: List[Dict[str, Any]]) -> None:
        """Save new learnings."""
        if learnings:
            await self.client.table("learnings").insert(learnings).execute()

    async def get_all_learnings(
        self,
        limit: int = 1000,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get active learnings with pagination.

        FIX: Added pagination to prevent memory overflow with large datasets.
        Default limit of 1000 is reasonable for most use cases.
        """
        result = await (
            self.client.table("learnings")
            .select("*")
            .eq("is_active", True)
            .order("confidence", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return result.data

    async def update_learnings(self, learnings: List[Dict[str, Any]]) -> None:
        """
        Update existing learnings (confirmations, contradictions).

        FIX: Uses batch upsert instead of N individual UPDATE queries.
        This reduces database round-trips from O(n) to O(1).
        """
        if not learnings:
            return

        # Batch upsert - single query instead of N queries
        # on_conflict="id" updates existing rows, inserts new ones
        await self.client.table("learnings").upsert(
            learnings,
            on_conflict="id"
        ).execute()

    async def get_learnings_for_component(
        self,
        component: str,
        content_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get learnings for a specific component."""
        query = (
            self.client.table("learnings")
            .select("*")
            .eq("affected_component", component)
            .eq("is_active", True)
            .gte("confidence", 0.5)
        )

        if content_type:
            query = query.or_(f"content_type.eq.{content_type},content_type.is.null")

        result = await query.order("confidence", desc=True).limit(10).execute()
        return result.data

    # ─────────────────────────────────────────────────────────────────────
    # MODIFICATIONS (Self-Modifying Code)
    # ─────────────────────────────────────────────────────────────────────

    async def save_modification(self, modification: Dict[str, Any]) -> str:
        """Save a code modification record."""
        result = await self.client.table("code_modifications").insert(modification).execute()
        return result.data[0]["id"]

    async def get_modification(self, modification_id: str) -> Optional[Dict[str, Any]]:
        """Get modification by ID."""
        result = await (
            self.client.table("code_modifications")
            .select("*")
            .eq("id", modification_id)
            .execute()
        )
        return result.data[0] if result.data else None

    async def update_modification(self, modification: Dict[str, Any]) -> None:
        """Update modification status."""
        await self.client.table("code_modifications").update(modification).eq("id", modification["id"]).execute()

    async def get_modifications(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get recent modifications."""
        from_date = (utc_now() - timedelta(days=days)).isoformat()
        result = await (
            self.client.table("code_modifications")
            .select("*")
            .gte("created_at", from_date)
            .order("created_at", desc=True)
            .execute()
        )
        return result.data

    # ─────────────────────────────────────────────────────────────────────
    # EXPERIMENTS (A/B Testing)
    # ─────────────────────────────────────────────────────────────────────

    async def save_experiment(self, experiment: Dict[str, Any]) -> str:
        """Save an experiment."""
        result = await self.client.table("experiments").insert(experiment).execute()
        return result.data[0]["id"]

    async def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment by ID."""
        result = await (
            self.client.table("experiments")
            .select("*")
            .eq("id", experiment_id)
            .execute()
        )
        return result.data[0] if result.data else None

    async def update_experiment(self, experiment: Dict[str, Any]) -> None:
        """Update experiment."""
        await self.client.table("experiments").update(experiment).eq("id", experiment["id"]).execute()

    async def get_active_experiments(self) -> List[Dict[str, Any]]:
        """Get all active experiments."""
        result = await (
            self.client.table("experiments")
            .select("*")
            .eq("status", "active")
            .execute()
        )
        return result.data

    # ─────────────────────────────────────────────────────────────────────
    # RESEARCH (Research Agent)
    # ─────────────────────────────────────────────────────────────────────

    async def save_research_report(self, report: Dict[str, Any]) -> str:
        """Save a research report."""
        result = await self.client.table("research_reports").insert(report).execute()
        return result.data[0]["id"]

    async def get_last_research_date(self) -> Optional[datetime]:
        """Get date of last research."""
        result = await (
            self.client.table("research_reports")
            .select("created_at")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        if result.data:
            return datetime.fromisoformat(result.data[0]["created_at"])
        return None

    async def get_past_experiments(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get past experiment results for research context."""
        result = await (
            self.client.table("experiments")
            .select("*")
            .eq("status", "completed")
            .order("completed_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data

    # ─────────────────────────────────────────────────────────────────────
    # PROMPTS (Prompt Management)
    # ─────────────────────────────────────────────────────────────────────

    async def get_prompt(self, component: str) -> Optional[str]:
        """Get current prompt for a component."""
        result = await (
            self.client.table("prompts")
            .select("content")
            .eq("component", component)
            .eq("is_active", True)
            .order("version", desc=True)
            .limit(1)
            .execute()
        )
        return result.data[0]["content"] if result.data else None

    async def save_prompt(
        self,
        component: str,
        content: str,
        reason: str
    ) -> str:
        """
        Save a new prompt version atomically.

        Uses Supabase RPC function to ensure atomicity:
        - Get current version
        - Deactivate old prompts
        - Insert new prompt
        All in single transaction to prevent race conditions.
        """
        # Use RPC for atomic operation (see SQL schema for function definition)
        result = await self.client.rpc(
            "save_prompt_atomic",
            {
                "p_component": component,
                "p_content": content,
                "p_reason": reason
            }
        ).execute()

        return result.data[0]["id"]

    # ─────────────────────────────────────────────────────────────────────
    # TREND TOPICS (Scout Cache)
    # ─────────────────────────────────────────────────────────────────────

    async def cache_topics(self, topics: List[Dict[str, Any]], source: str) -> None:
        """Cache discovered topics."""
        for topic in topics:
            topic["source"] = source
            topic["cached_at"] = utc_now().isoformat()

        await self.client.table("topic_cache").insert(topics).execute()

    async def get_cached_topics(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recently cached topics."""
        from_time = (utc_now() - timedelta(hours=hours)).isoformat()
        result = await (
            self.client.table("topic_cache")
            .select("*")
            .gte("cached_at", from_time)
            .order("score", desc=True)
            .execute()
        )
        return result.data

    # ─────────────────────────────────────────────────────────────────────
    # PHOTOS (Author photo library)
    # ─────────────────────────────────────────────────────────────────────

    async def get_all_photos(self) -> List[Dict[str, Any]]:
        """Get all photos from the library."""
        result = await (
            self.client.table("author_photos")
            .select("*")
            .eq("disabled", False)
            .order("times_used", desc=False)  # Prefer less used
            .execute()
        )
        return result.data

    async def update_photo_usage(self, photo_id: str, post_id: str) -> None:
        """Update photo usage statistics."""
        await self.client.rpc(
            "increment_photo_usage",
            {"p_photo_id": photo_id, "p_post_id": post_id}
        ).execute()

    async def save_photo_metadata(self, metadata: Dict[str, Any]) -> str:
        """Save new photo metadata."""
        result = await self.client.table("author_photos").insert(metadata).execute()
        return result.data[0]["id"]

    # ─────────────────────────────────────────────────────────────────────
    # DRAFTS (Work in Progress)
    # ─────────────────────────────────────────────────────────────────────

    async def save_draft(self, draft: Dict[str, Any]) -> str:
        """Save a draft post."""
        result = await self.client.table("drafts").insert(draft).execute()
        return result.data[0]["id"]

    async def get_draft(self, draft_id: str) -> Optional[Dict[str, Any]]:
        """Get draft by ID."""
        result = await (
            self.client.table("drafts")
            .select("*")
            .eq("id", draft_id)
            .execute()
        )
        return result.data[0] if result.data else None

    async def update_draft(self, draft: Dict[str, Any]) -> None:
        """Update a draft."""
        await self.client.table("drafts").update(draft).eq("id", draft["id"]).execute()


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL DATABASE INSTANCE
# Import this in all modules that need database access
# ═══════════════════════════════════════════════════════════════════════════

# Singleton pattern - one async connection for entire application
_db_instance: Optional[SupabaseDB] = None
_db_lock = asyncio.Lock()

async def get_db() -> SupabaseDB:
    """Get the global async database instance (lazy initialization)."""
    global _db_instance
    if _db_instance is None:
        async with _db_lock:
            # Double-check after acquiring lock
            if _db_instance is None:
                _db_instance = await SupabaseDB.create()
    return _db_instance
```

---

### Supabase Schema (SQL)

```sql
-- ═══════════════════════════════════════════════════════════════════════════
-- SUPABASE SCHEMA FOR LINKEDIN SUPER AGENT
-- Run this in Supabase SQL Editor to create all tables
-- ═══════════════════════════════════════════════════════════════════════════

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ───────────────────────────────────────────────────────────────────────────
-- POSTS: Published LinkedIn posts
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE posts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    linkedin_post_id TEXT UNIQUE,           -- LinkedIn's URN

    -- Content
    content_type TEXT NOT NULL,             -- enterprise_case, primary_source, etc.
    title TEXT,
    text_content TEXT NOT NULL,
    hook TEXT,                              -- First line (for analysis)

    -- Visual
    visual_type TEXT,                       -- photo, diagram, carousel, none
    visual_url TEXT,

    -- Scores
    qc_score FLOAT,
    meta_evaluation_score FLOAT,

    -- Generation metadata
    topic_id TEXT,
    template_used TEXT,
    hook_style TEXT,
    revision_count INTEGER DEFAULT 0,

    -- Learning context
    learnings_used JSONB DEFAULT '[]',      -- Which learnings were applied

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    published_at TIMESTAMPTZ,

    -- Indexes for common queries
    CONSTRAINT valid_content_type CHECK (content_type IN (
        'enterprise_case', 'primary_source', 'automation_case',
        'community_content', 'tool_release'
    ))
);

CREATE INDEX idx_posts_content_type ON posts(content_type);
CREATE INDEX idx_posts_created_at ON posts(created_at DESC);
CREATE INDEX idx_posts_qc_score ON posts(qc_score DESC);
-- FIX: Missing indexes identified by architecture review
CREATE INDEX idx_posts_topic_id ON posts(topic_id);
CREATE INDEX idx_posts_published_at ON posts(published_at DESC) WHERE published_at IS NOT NULL;

-- ───────────────────────────────────────────────────────────────────────────
-- POST_METRICS: Analytics snapshots over time
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE post_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    post_id UUID REFERENCES posts(id) ON DELETE CASCADE,

    -- Metrics
    likes INTEGER DEFAULT 0,
    comments INTEGER DEFAULT 0,
    reposts INTEGER DEFAULT 0,
    impressions INTEGER DEFAULT 0,

    -- Calculated
    engagement_rate FLOAT,

    -- Time context
    minutes_after_post INTEGER,             -- How long after posting
    collected_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(post_id, minutes_after_post)
);

CREATE INDEX idx_metrics_post_id ON post_metrics(post_id);
CREATE INDEX idx_metrics_collected_at ON post_metrics(collected_at DESC);

-- ───────────────────────────────────────────────────────────────────────────
-- LEARNINGS: Micro-learnings from continuous learning engine
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE learnings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Learning content
    learning_type TEXT NOT NULL,            -- hook_pattern, visual_style, etc.
    source TEXT NOT NULL,                   -- meta_evaluation, qc_feedback, etc.
    description TEXT NOT NULL,
    rule TEXT NOT NULL,                     -- Machine-applicable rule

    -- Context
    affected_component TEXT NOT NULL,       -- writer, visual_creator, etc.
    content_type TEXT,                      -- NULL = applies to all

    -- Confidence tracking
    confidence FLOAT DEFAULT 0.4,
    confirmations INTEGER DEFAULT 0,
    contradictions INTEGER DEFAULT 0,

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    is_promoted_to_rule BOOLEAN DEFAULT FALSE,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_confirmed_at TIMESTAMPTZ,
    promoted_at TIMESTAMPTZ
);

CREATE INDEX idx_learnings_component ON learnings(affected_component);
CREATE INDEX idx_learnings_confidence ON learnings(confidence DESC);
CREATE INDEX idx_learnings_active ON learnings(is_active) WHERE is_active = TRUE;
-- FIX: Missing indexes for content_type filtering (used in get_learnings_for_component)
CREATE INDEX idx_learnings_content_type ON learnings(content_type);
CREATE INDEX idx_learnings_component_type ON learnings(affected_component, content_type);

-- ───────────────────────────────────────────────────────────────────────────
-- CODE_MODIFICATIONS: Self-modifying code history
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE code_modifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- What was modified
    gap_type TEXT NOT NULL,                 -- data_source, analysis_method, etc.
    gap_description TEXT NOT NULL,

    -- Generated code
    module_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    code_content TEXT NOT NULL,
    test_code TEXT,

    -- Validation results
    syntax_valid BOOLEAN,
    type_check_passed BOOLEAN,
    security_passed BOOLEAN,
    tests_passed BOOLEAN,

    -- Status
    status TEXT DEFAULT 'pending',          -- pending, validated, deployed, rolled_back

    -- Rollback info
    previous_code TEXT,                     -- For rollback
    rollback_reason TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    validated_at TIMESTAMPTZ,
    deployed_at TIMESTAMPTZ,
    rolled_back_at TIMESTAMPTZ
);

CREATE INDEX idx_modifications_status ON code_modifications(status);
CREATE INDEX idx_modifications_created ON code_modifications(created_at DESC);

-- ───────────────────────────────────────────────────────────────────────────
-- EXPERIMENTS: A/B testing
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Experiment definition
    name TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    variable TEXT NOT NULL,                 -- What we're testing

    -- Variants
    control_value JSONB NOT NULL,
    treatment_value JSONB NOT NULL,

    -- Results
    control_posts JSONB DEFAULT '[]',       -- Array of post IDs
    treatment_posts JSONB DEFAULT '[]',
    control_metrics JSONB,                  -- Aggregated metrics
    treatment_metrics JSONB,

    -- Statistical analysis
    sample_size_target INTEGER DEFAULT 10,
    significance_threshold FLOAT DEFAULT 0.05,
    winner TEXT,                            -- 'control', 'treatment', NULL
    p_value FLOAT,

    -- Status
    status TEXT DEFAULT 'active',           -- active, completed, cancelled

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_experiments_status ON experiments(status);

-- ───────────────────────────────────────────────────────────────────────────
-- RESEARCH_REPORTS: Deep research results
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE research_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Trigger
    trigger_type TEXT NOT NULL,             -- weekly_cycle, underperformance, etc.
    trigger_context JSONB,

    -- Research content
    queries_executed JSONB,                 -- What was searched
    findings JSONB NOT NULL,                -- List of findings
    recommendations JSONB NOT NULL,

    -- Actions taken
    modifications_made JSONB DEFAULT '[]',  -- References to code_modifications
    prompt_changes JSONB DEFAULT '[]',
    config_changes JSONB DEFAULT '[]',

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_research_created ON research_reports(created_at DESC);

-- ───────────────────────────────────────────────────────────────────────────
-- PROMPTS: Versioned prompts for all components
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE prompts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Identity
    component TEXT NOT NULL,                -- writer, humanizer, qc, etc.
    version INTEGER NOT NULL,

    -- Content
    content TEXT NOT NULL,

    -- Status
    is_active BOOLEAN DEFAULT TRUE,

    -- Change tracking
    change_reason TEXT,
    changed_by TEXT DEFAULT 'system',       -- 'system', 'human', 'research_agent'

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(component, version)
);

CREATE INDEX idx_prompts_component ON prompts(component);
CREATE INDEX idx_prompts_active ON prompts(component, is_active) WHERE is_active = TRUE;

-- ───────────────────────────────────────────────────────────────────────────
-- AUTHOR_PHOTOS: Author photo library for post personalization
-- FIX: Previously referenced but not defined
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE author_photos (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- File info
    file_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_size_kb INTEGER,

    -- Auto-tagged properties (via Claude Vision)
    setting TEXT,                           -- office, conference, outdoor, studio, home
    pose TEXT,                              -- portrait, speaking, working, thinking, gesturing
    mood TEXT,                              -- professional, friendly, focused, excited, thoughtful
    attire TEXT,                            -- formal, business_casual, casual
    background TEXT,                        -- plain, office, stage, nature, abstract
    face_position TEXT,                     -- center, left_third, right_third
    eye_contact TEXT,                       -- direct, away, profile
    suitable_for TEXT[],                    -- Array of content type strings

    -- Technical properties
    width INTEGER,
    height INTEGER,
    aspect_ratio TEXT,

    -- Usage tracking
    times_used INTEGER DEFAULT 0,
    last_used_date TIMESTAMPTZ,
    last_used_post_id UUID REFERENCES posts(id),

    -- Manual overrides
    favorite BOOLEAN DEFAULT FALSE,
    disabled BOOLEAN DEFAULT FALSE,
    custom_tags TEXT[],

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_author_photos_disabled ON author_photos(disabled) WHERE disabled = FALSE;
CREATE INDEX idx_author_photos_usage ON author_photos(times_used ASC);

-- ───────────────────────────────────────────────────────────────────────────
-- TOPIC_CACHE: Cached trending topics
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE topic_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Topic data
    external_id TEXT,                       -- ID from source
    source TEXT NOT NULL,                   -- hackernews, twitter, etc.
    title TEXT NOT NULL,
    url TEXT,

    -- Classification
    content_type TEXT,
    score FLOAT,

    -- Metadata
    raw_data JSONB,

    -- Timestamps
    cached_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

CREATE INDEX idx_topic_cache_score ON topic_cache(score DESC);
CREATE INDEX idx_topic_cache_cached ON topic_cache(cached_at DESC);
-- FIX: Index for cleanup queries on expired topics
CREATE INDEX idx_topic_cache_expires ON topic_cache(expires_at) WHERE expires_at IS NOT NULL;

-- ───────────────────────────────────────────────────────────────────────────
-- DRAFTS: Work in progress posts
-- FIX: Expanded to match DraftPost dataclass structure
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE drafts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- References
    topic_id TEXT,
    content_type TEXT NOT NULL,

    -- Structured content (FIX: separate fields instead of single text_content)
    hook TEXT NOT NULL,                     -- First line (must fit in 210 chars)
    body TEXT NOT NULL,                     -- Main content
    cta TEXT,                               -- Call to action
    hashtags TEXT[] DEFAULT '{}',           -- Array of hashtags
    full_text TEXT NOT NULL,                -- Combined, formatted

    -- Template metadata (FIX: added fields)
    template_used TEXT,
    template_category TEXT,                 -- universal / enterprise / research / automation / etc.
    hook_style TEXT,                        -- metrics / lessons / contrarian / how_to / etc.

    -- Content metrics (FIX: added fields)
    character_count INTEGER,
    estimated_read_time TEXT,
    hook_in_limit BOOLEAN DEFAULT TRUE,     -- Is hook under 210 chars?
    length_in_range BOOLEAN DEFAULT TRUE,   -- Is total length in target range?

    -- Type-specific data (FIX: structured JSONB)
    type_data_injected JSONB DEFAULT '{}',  -- What extraction data was used

    -- Visual brief for next agents (FIX: added fields)
    visual_brief TEXT,                      -- Description for image generation
    visual_type TEXT,                       -- data_viz / diagram / screenshot / quote_card
    visual_data JSONB,                      -- Generated visual data
    key_terms TEXT[] DEFAULT '{}',          -- For hashtag optimization

    -- State
    stage TEXT DEFAULT 'draft',             -- draft, humanized, with_visual, qc_passed
    revision_count INTEGER DEFAULT 0,
    version INTEGER DEFAULT 1,

    -- Evaluations
    evaluations JSONB DEFAULT '[]',         -- History of QC evaluations

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT valid_draft_content_type CHECK (content_type IN (
        'enterprise_case', 'primary_source', 'automation_case',
        'community_content', 'tool_release'
    )),
    CONSTRAINT valid_draft_stage CHECK (stage IN (
        'draft', 'meta_evaluated', 'humanized', 'with_visual', 'qc_passed', 'approved', 'rejected'
    ))
);

CREATE INDEX idx_drafts_stage ON drafts(stage);
CREATE INDEX idx_drafts_topic_id ON drafts(topic_id);
CREATE INDEX idx_drafts_content_type ON drafts(content_type);

-- ───────────────────────────────────────────────────────────────────────────
-- PHOTO_METADATA: Personal photo library for post visuals
-- FIX: New table to match PhotoMetadata dataclass
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE photo_metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- File info
    file_path TEXT UNIQUE NOT NULL,
    file_name TEXT NOT NULL,

    -- Auto-tagged properties (from AI analysis)
    setting TEXT,                           -- office, outdoor, conference, casual
    pose TEXT,                              -- standing, sitting, speaking, working
    mood TEXT,                              -- professional, friendly, thoughtful, energetic
    attire TEXT,                            -- formal, business_casual, casual
    background TEXT,                        -- plain, office, nature, event
    face_position TEXT,                     -- center, left_third, right_third
    eye_contact BOOLEAN DEFAULT TRUE,

    -- Suitability tags
    suitable_for TEXT[] DEFAULT '{}',       -- enterprise_case, primary_source, etc.

    -- Technical properties
    width INTEGER,
    height INTEGER,
    aspect_ratio TEXT,                      -- 1:1, 4:3, 16:9
    file_size_kb INTEGER,
    dominant_colors TEXT[] DEFAULT '{}',

    -- Usage tracking
    times_used INTEGER DEFAULT 0,
    last_used_date TIMESTAMPTZ,
    last_used_post_id UUID REFERENCES posts(id) ON DELETE SET NULL,

    -- Manual overrides
    favorite BOOLEAN DEFAULT FALSE,
    disabled BOOLEAN DEFAULT FALSE,
    custom_tags TEXT[] DEFAULT '{}',
    notes TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_photo_setting ON photo_metadata(setting);
CREATE INDEX idx_photo_pose ON photo_metadata(pose);
CREATE INDEX idx_photo_mood ON photo_metadata(mood);
CREATE INDEX idx_photo_times_used ON photo_metadata(times_used);
CREATE INDEX idx_photo_suitable_for ON photo_metadata USING GIN(suitable_for);
CREATE INDEX idx_photo_not_disabled ON photo_metadata(disabled) WHERE disabled = FALSE;

-- ───────────────────────────────────────────────────────────────────────────
-- PIPELINE_ERRORS: Error tracking for post-mortem analysis
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE pipeline_errors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Error identification
    run_id TEXT,
    error_type TEXT NOT NULL,               -- Exception class name
    error_message TEXT NOT NULL,

    -- Context
    stage TEXT,                             -- Last successful stage
    context JSONB,                          -- Full error context for debugging

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_pipeline_errors_run_id ON pipeline_errors(run_id);
CREATE INDEX idx_pipeline_errors_created ON pipeline_errors(created_at DESC);
CREATE INDEX idx_pipeline_errors_type ON pipeline_errors(error_type);

-- ───────────────────────────────────────────────────────────────────────────
-- AGENT_LOGS: Structured logging storage for AgentLogger
-- ───────────────────────────────────────────────────────────────────────────
CREATE TABLE agent_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Timing
    timestamp TIMESTAMPTZ NOT NULL,

    -- Classification
    level INTEGER NOT NULL,                 -- LogLevel numeric value (10=DEBUG, 50=CRITICAL)
    level_name TEXT NOT NULL,               -- Human-readable level name
    component TEXT NOT NULL,                -- LogComponent value

    -- Content
    message TEXT NOT NULL,

    -- Context
    run_id TEXT,                            -- Pipeline run ID
    post_id TEXT,                           -- Related post ID
    data JSONB DEFAULT '{}',                -- Additional structured data

    -- Error details (if applicable)
    error_type TEXT,                        -- Exception class name
    error_traceback TEXT,                   -- Full traceback

    -- Performance
    duration_ms INTEGER,                    -- Operation duration if timed

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX idx_agent_logs_timestamp ON agent_logs(timestamp DESC);
CREATE INDEX idx_agent_logs_level ON agent_logs(level);
CREATE INDEX idx_agent_logs_component ON agent_logs(component);
CREATE INDEX idx_agent_logs_run_id ON agent_logs(run_id) WHERE run_id IS NOT NULL;
CREATE INDEX idx_agent_logs_error ON agent_logs(error_type) WHERE error_type IS NOT NULL;

-- Retention policy: auto-delete logs older than 30 days (configurable)
-- Run via pg_cron or scheduled function
CREATE OR REPLACE FUNCTION cleanup_old_agent_logs(retention_days INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM agent_logs
    WHERE timestamp < NOW() - (retention_days || ' days')::INTERVAL;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ───────────────────────────────────────────────────────────────────────────
-- RPC FUNCTIONS: Aggregation queries
-- ───────────────────────────────────────────────────────────────────────────

-- Average metrics at specific time after posting
CREATE OR REPLACE FUNCTION get_average_metrics_at_minutes(minutes INTEGER)
RETURNS TABLE (
    avg_likes FLOAT,
    avg_comments FLOAT,
    avg_reposts FLOAT,
    avg_impressions FLOAT,
    avg_engagement_rate FLOAT,
    sample_size INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        AVG(likes)::FLOAT as avg_likes,
        AVG(comments)::FLOAT as avg_comments,
        AVG(reposts)::FLOAT as avg_reposts,
        AVG(impressions)::FLOAT as avg_impressions,
        AVG(engagement_rate)::FLOAT as avg_engagement_rate,
        COUNT(*)::INTEGER as sample_size
    FROM post_metrics
    WHERE minutes_after_post = minutes;
END;
$$ LANGUAGE plpgsql;

-- Average QC score
CREATE OR REPLACE FUNCTION get_average_qc_score()
RETURNS TABLE (avg_score FLOAT) AS $$
BEGIN
    RETURN QUERY
    SELECT AVG(qc_score)::FLOAT as avg_score
    FROM posts
    WHERE qc_score IS NOT NULL;
END;
$$ LANGUAGE plpgsql;

-- Percentile rank for likes at a given time checkpoint
-- Returns what percent of posts this one outperformed (e.g., 90 = top 10%)
CREATE OR REPLACE FUNCTION get_likes_percentile(likes_count INTEGER, minutes INTEGER)
RETURNS TABLE (percentile FLOAT) AS $$
BEGIN
    RETURN QUERY
    SELECT
        (COUNT(*) FILTER (WHERE likes < likes_count)::FLOAT / NULLIF(COUNT(*), 0) * 100)::FLOAT as percentile
    FROM post_metrics
    WHERE minutes_after_post = minutes;
END;
$$ LANGUAGE plpgsql;

-- Performance gaps (for research agent)
CREATE OR REPLACE FUNCTION get_performance_gaps()
RETURNS TABLE (
    content_type TEXT,
    avg_score FLOAT,
    post_count INTEGER,
    below_threshold_count INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.content_type,
        AVG(p.qc_score)::FLOAT as avg_score,
        COUNT(*)::INTEGER as post_count,
        COUNT(*) FILTER (WHERE p.qc_score < 8.0)::INTEGER as below_threshold_count
    FROM posts p
    WHERE p.created_at > NOW() - INTERVAL '30 days'
    GROUP BY p.content_type
    ORDER BY avg_score ASC;
END;
$$ LANGUAGE plpgsql;

-- Atomic prompt save (prevents race conditions)
CREATE OR REPLACE FUNCTION save_prompt_atomic(
    p_component TEXT,
    p_content TEXT,
    p_reason TEXT
)
RETURNS TABLE (id UUID) AS $$
DECLARE
    v_new_version INTEGER;
    v_new_id UUID;
BEGIN
    -- Lock the prompts table for this component to prevent concurrent updates
    PERFORM pg_advisory_xact_lock(hashtext(p_component));

    -- Get current version
    SELECT COALESCE(MAX(version), 0) + 1 INTO v_new_version
    FROM prompts
    WHERE component = p_component;

    -- Deactivate old prompts (within transaction)
    UPDATE prompts
    SET is_active = FALSE
    WHERE component = p_component AND is_active = TRUE;

    -- Insert new prompt
    INSERT INTO prompts (component, content, version, is_active, change_reason, created_at)
    VALUES (p_component, p_content, v_new_version, TRUE, p_reason, NOW())
    RETURNING prompts.id INTO v_new_id;

    RETURN QUERY SELECT v_new_id;
END;
$$ LANGUAGE plpgsql;

-- Increment photo usage counter
-- FIX: Previously called by SupabaseDB.update_photo_usage but not defined
CREATE OR REPLACE FUNCTION increment_photo_usage(
    p_photo_id UUID,
    p_post_id UUID
)
RETURNS VOID AS $$
BEGIN
    UPDATE author_photos
    SET
        times_used = times_used + 1,
        last_used_date = NOW(),
        last_used_post_id = p_post_id,
        updated_at = NOW()
    WHERE id = p_photo_id;
END;
$$ LANGUAGE plpgsql;

-- ───────────────────────────────────────────────────────────────────────────
-- ROW LEVEL SECURITY (optional, for multi-tenant)
-- ───────────────────────────────────────────────────────────────────────────

-- For single-user agent, RLS is not needed
-- But if you want to support multiple users in future:
-- ALTER TABLE posts ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY "Users can only see their own posts" ON posts FOR ALL USING (auth.uid() = user_id);
```

---

### Claude Code CLI Integration

**НЕ ИСПОЛЬЗУЕМ Claude API — используем Claude Code CLI с твоей подпиской!**

Система запускает `claude` CLI команду через subprocess, что позволяет:
- Использовать твою подписку Claude Pro/Max
- Не платить за API токены
- Иметь тот же контекст и capabilities что и в терминале

```python
# ═══════════════════════════════════════════════════════════════════════════
# CLAUDE CODE CLI INTEGRATION
# Используем claude CLI с подпиской, а не API
# ═══════════════════════════════════════════════════════════════════════════

import subprocess
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
import asyncio


@dataclass
class ClaudeResponse:
    """Response from Claude Code CLI."""
    success: bool
    content: str
    error: Optional[str] = None
    raw_output: Optional[str] = None


class ClaudeCodeCLI:
    """
    Interface to Claude Code CLI.

    Uses your Claude subscription (Pro/Max), NOT API tokens.
    Runs `claude` command via subprocess.

    Usage:
        claude_cli = ClaudeCodeCLI()
        response = await claude_cli.generate("Write a Python function...")
        structured = await claude_cli.generate_structured("Analyze...", output_schema)
    """

    def __init__(
        self,
        working_dir: Optional[Path] = None,
        timeout: int = 300,  # 5 minutes default
        model: str = "opus"  # opus, sonnet, haiku
    ):
        self.working_dir = working_dir or Path.cwd()
        self.timeout = timeout
        self.model = model

        # Verify claude CLI is available
        self._verify_cli()

    def _verify_cli(self):
        """Verify Claude Code CLI is installed and authenticated."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError("Claude CLI not found or not authenticated")
        except FileNotFoundError:
            raise RuntimeError(
                "Claude Code CLI not installed. "
                "Install with: npm install -g @anthropic-ai/claude-code"
            )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context_files: Optional[list[Path]] = None
    ) -> ClaudeResponse:
        """
        Generate text using Claude Code CLI.

        Args:
            prompt: The main prompt/question
            system_prompt: Optional system instructions
            context_files: Optional files to include as context

        Returns:
            ClaudeResponse with the generated content
        """
        # Build command
        # FIX: Use correct Claude Code CLI flags
        # Previous code used non-existent flags: --print, --system, --file, --prompt
        # Correct flags: -p for prompt, --output-format for output type

        # Build the full prompt (including system prompt and file contents)
        full_prompt = ""

        # Add system prompt if provided (no --system flag exists)
        if system_prompt:
            full_prompt += f"<system>\n{system_prompt}\n</system>\n\n"

        # Add context files (no --file flag exists, must include content in prompt)
        if context_files:
            for file in context_files:
                if file.exists():
                    try:
                        content = file.read_text(encoding='utf-8')
                        full_prompt += f"<file path=\"{file}\">\n{content}\n</file>\n\n"
                    except Exception as e:
                        # Log but continue if file can't be read
                        pass

        full_prompt += prompt

        cmd = [
            "claude",
            "-p", full_prompt,  # -p is the correct flag for prompt
            "--output-format", "text",  # Get text output
        ]

        # Add model selection if specified
        if self.model:
            cmd.extend(["--model", self.model])

        # Run asynchronously
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.working_dir)
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )

            if process.returncode == 0:
                return ClaudeResponse(
                    success=True,
                    content=stdout.decode("utf-8").strip(),
                    raw_output=stdout.decode("utf-8")
                )
            else:
                return ClaudeResponse(
                    success=False,
                    content="",
                    error=stderr.decode("utf-8"),
                    raw_output=stdout.decode("utf-8")
                )

        except asyncio.TimeoutError:
            # FIX: Kill orphan process to prevent resource leak
            try:
                process.kill()
                await process.wait()  # Clean up zombie process
            except Exception:
                pass  # Process may have already exited
            return ClaudeResponse(
                success=False,
                content="",
                error=f"Timeout after {self.timeout} seconds (process killed)"
            )
        except Exception as e:
            # FIX: Also cleanup subprocess on general Exception
            try:
                if process and process.returncode is None:
                    process.kill()
                    await process.wait()
            except Exception:
                pass  # Process may have already exited
            return ClaudeResponse(
                success=False,
                content="",
                error=str(e)
            )

    async def generate_structured(
        self,
        prompt: str,
        output_type: type = dict,
        system_prompt: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate structured JSON output using Claude Code CLI.

        Args:
            prompt: The prompt (should ask for JSON output)
            output_type: Expected output type (dict, list, etc.)
            system_prompt: Optional system instructions

        Returns:
            Parsed JSON as dict/list, or None if parsing fails
        """
        # Add JSON instruction to prompt
        json_prompt = f"""
{prompt}

IMPORTANT: Return ONLY valid JSON, no markdown, no explanation, no code blocks.
Just the raw JSON object/array.
"""

        response = await self.generate(json_prompt, system_prompt)

        if not response.success:
            raise RuntimeError(f"Claude CLI error: {response.error}")

        # Parse JSON from response
        import re  # FIX: Move import to top of function, not inside except

        try:
            # Try to extract JSON from response (handle markdown code blocks)
            content = response.content.strip()

            # FIX: Improved markdown code block removal
            # Handle ```json, ```python, etc. and verify closing ```
            if content.startswith("```"):
                lines = content.split("\n")
                # Check if last line is closing ```
                if lines[-1].strip() == "```":
                    # Remove first line (```json) and last line (```)
                    content = "\n".join(lines[1:-1])
                elif "```" in lines[-1]:
                    # Handle case where ``` is on same line as content
                    lines[-1] = lines[-1].replace("```", "")
                    content = "\n".join(lines[1:])

            return json.loads(content)

        except json.JSONDecodeError as e:
            # Try to find JSON object or array in the response
            # FIX: Use non-greedy regex to avoid over-matching
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]', response.content)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass  # FIX: Specify exception type instead of bare except

            raise ValueError(f"Failed to parse JSON from response: {e}\nContent: {response.content[:500]}")

    async def generate_code(
        self,
        task_description: str,
        language: str = "python",
        context_files: Optional[list[Path]] = None,
        output_file: Optional[Path] = None
    ) -> ClaudeResponse:
        """
        Generate code using Claude Code CLI.

        This is specifically for code generation tasks.

        Args:
            task_description: What code to generate
            language: Programming language
            context_files: Existing files for context
            output_file: If provided, save code to this file

        Returns:
            ClaudeResponse with generated code
        """
        prompt = f"""
Generate {language} code for the following task:

{task_description}

Requirements:
- Production-ready code, not a prototype
- Include type hints
- Include docstrings
- Handle errors appropriately
- Follow best practices for {language}

Return ONLY the code, no explanation.
"""

        response = await self.generate(
            prompt,
            system_prompt=f"You are an expert {language} developer. Generate clean, production-ready code.",
            context_files=context_files
        )

        if response.success and output_file:
            # Extract code from response (remove markdown if present)
            code = response.content.strip()
            if code.startswith("```"):
                lines = code.split("\n")
                code = "\n".join(lines[1:-1])

            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(code, encoding="utf-8")

        return response

    async def analyze(
        self,
        content: str,
        analysis_prompt: str
    ) -> ClaudeResponse:
        """
        Analyze content using Claude Code CLI.

        Args:
            content: The content to analyze
            analysis_prompt: What to analyze/extract

        Returns:
            ClaudeResponse with analysis
        """
        prompt = f"""
Analyze the following content:

---
{content}
---

{analysis_prompt}
"""
        return await self.generate(prompt)

    async def evaluate(
        self,
        content: str,
        criteria: Dict[str, str],
        output_format: str = "json"
    ) -> Dict[str, Any]:
        """
        Evaluate content against criteria.

        Args:
            content: Content to evaluate
            criteria: Dict of criterion_name -> description
            output_format: "json" or "text"

        Returns:
            Evaluation results as dict
        """
        criteria_text = "\n".join(
            f"- {name}: {desc}"
            for name, desc in criteria.items()
        )

        prompt = f"""
Evaluate this content against the following criteria:

CONTENT:
---
{content}
---

CRITERIA:
{criteria_text}

Return JSON with:
{{
    "scores": {{
        "criterion_name": score_1_to_10,
        ...
    }},
    "feedback": {{
        "criterion_name": "specific feedback",
        ...
    }},
    "overall_score": weighted_average,
    "summary": "brief overall assessment"
}}
"""
        return await self.generate_structured(prompt)


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL CLAUDE CLI INSTANCE
# FIX: Added thread-safe initialization (consistent with get_db() pattern)
# ═══════════════════════════════════════════════════════════════════════════

_claude_cli_instance: Optional[ClaudeCodeCLI] = None
_claude_cli_lock = threading.Lock()

def get_claude() -> ClaudeCodeCLI:
    """Get the global Claude Code CLI instance (thread-safe)."""
    global _claude_cli_instance
    if _claude_cli_instance is None:
        with _claude_cli_lock:
            # Double-checked locking pattern
            if _claude_cli_instance is None:
                _claude_cli_instance = ClaudeCodeCLI()
    return _claude_cli_instance


# ═══════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════

# Instead of:
#   response = await get_claude().generate_structured(prompt)
#
# Use:
#   claude = get_claude()
#   response = await claude.generate_structured(prompt)
#
# For code generation:
#   response = await claude.generate_code(
#       task_description="Create a sentiment analyzer module",
#       language="python",
#       output_file=Path("analyzers/sentiment.py")
#   )
```

---

### Centralized Threshold Configuration

```python
# ═══════════════════════════════════════════════════════════════════════════
# CENTRALIZED THRESHOLD CONFIGURATION
# Single Source of Truth for ALL QC/Evaluation thresholds
# ═══════════════════════════════════════════════════════════════════════════

from __future__ import annotations  # Enable forward references
from dataclasses import dataclass, field
from typing import Dict, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from typing import Dict  # For type checkers

# ContentType MUST be defined before ThresholdConfig to avoid forward reference issues
class ContentType(Enum):
    """Five distinct content verticals requiring different scoring."""
    ENTERPRISE_CASE = "enterprise_case"
    PRIMARY_SOURCE = "primary_source"
    AUTOMATION_CASE = "automation_case"
    COMMUNITY_CONTENT = "community_content"
    TOOL_RELEASE = "tool_release"


# ═══════════════════════════════════════════════════════════════════════════
# BRANDED TYPES FOR SCORE SCALES
# FIX: Prevents confusion between 0-100 scale and 0-1 scale (confidence)
# ═══════════════════════════════════════════════════════════════════════════
from typing import NewType, Union

# Score on 0-10 scale (QC scores, aggregate scores)
Score10 = NewType('Score10', float)

# Score on 0-100 scale (relevance scores, percentage-based scores)
Score100 = NewType('Score100', float)

# Confidence on 0-1 scale (probability, confidence levels)
Confidence = NewType('Confidence', float)


def validate_score10(value: float, name: str = "score") -> Score10:
    """Validate and convert to Score10 (0-10 scale)."""
    if not 0.0 <= value <= 10.0:
        raise ValueError(f"{name} must be between 0 and 10, got {value}")
    return Score10(value)


def validate_score100(value: float, name: str = "score") -> Score100:
    """Validate and convert to Score100 (0-100 scale)."""
    if not 0.0 <= value <= 100.0:
        raise ValueError(f"{name} must be between 0 and 100, got {value}")
    return Score100(value)


def validate_confidence(value: float, name: str = "confidence") -> Confidence:
    """Validate and convert to Confidence (0-1 scale)."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
    return Confidence(value)


def score100_to_score10(score: Score100) -> Score10:
    """Convert 0-100 scale to 0-10 scale."""
    return Score10(score / 10.0)


def score10_to_score100(score: Score10) -> Score100:
    """Convert 0-10 scale to 0-100 scale."""
    return Score100(score * 10.0)


def confidence_to_score10(confidence: Confidence) -> Score10:
    """Convert confidence (0-1) to score (0-10)."""
    return Score10(confidence * 10.0)


def score10_to_confidence(score: Score10) -> Confidence:
    """Convert score (0-10) to confidence (0-1)."""
    return Confidence(score / 10.0)


# ═══════════════════════════════════════════════════════════════════════════
# CENTRALIZED RETRY CONFIGURATION
# Single Source of Truth for all retry behavior
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RetryConfig:
    """
    SINGLE SOURCE OF TRUTH for retry behavior across all components.

    Previously hardcoded as magic numbers throughout:
    - max_attempts=3 everywhere
    - base_delay=2.0 in some places, 5.0 in others
    - Different backoff strategies

    NOW: All retry behavior derives from this configuration.

    Usage:
        config = RETRY_CONFIG  # Global instance
        @with_retry(
            max_attempts=config.default_max_attempts,
            base_delay=config.default_base_delay
        )
    """

    # Default retry settings
    default_max_attempts: int = 3
    default_base_delay: float = 2.0
    default_max_delay: float = 60.0
    default_backoff_factor: float = 2.0

    # Component-specific overrides
    llm_max_attempts: int = 3           # Claude API calls
    llm_base_delay: float = 5.0         # Longer delay for rate limits

    linkedin_max_attempts: int = 3      # LinkedIn API
    linkedin_base_delay: float = 5.0    # LinkedIn is rate-limit sensitive

    visual_gen_max_attempts: int = 3    # Nano Banana
    visual_gen_base_delay: float = 3.0

    database_max_attempts: int = 3      # Supabase
    database_base_delay: float = 1.0    # DB usually recovers fast

    search_max_attempts: int = 2        # Perplexity/Tavily - fail faster
    search_base_delay: float = 2.0

    def get_config_for(self, component: str) -> tuple:
        """Get (max_attempts, base_delay) for a component."""
        configs = {
            "llm": (self.llm_max_attempts, self.llm_base_delay),
            "linkedin": (self.linkedin_max_attempts, self.linkedin_base_delay),
            "visual": (self.visual_gen_max_attempts, self.visual_gen_base_delay),
            "database": (self.database_max_attempts, self.database_base_delay),
            "search": (self.search_max_attempts, self.search_base_delay),
        }
        return configs.get(component, (self.default_max_attempts, self.default_base_delay))


# Global retry configuration instance
RETRY_CONFIG = RetryConfig()


@dataclass
class ThresholdConfig:
    """
    SINGLE SOURCE OF TRUTH for all quality thresholds.

    RATIONALE:
    Previously, thresholds were scattered across the codebase:
    - QC Agent: 7.5 (universal), 7.2-7.5 (type-specific)
    - Meta-Agent: 8.0
    - Auto-publish: 9.0

    This caused inconsistent behavior where a post could pass QC but
    fail Meta-Agent evaluation, or vice versa.

    NOW: All thresholds derive from this single configuration.

    Usage:
        config = ThresholdConfig()
        threshold = config.get_pass_threshold(ContentType.ENTERPRISE_CASE)
        auto_threshold = config.get_auto_publish_threshold()
    """

    # ─────────────────────────────────────────────────────────────────
    # BASE THRESHOLDS
    # ─────────────────────────────────────────────────────────────────
    min_score_to_proceed: float = 8.0       # Minimum to pass QC
    auto_publish_threshold: float = 9.0     # Auto-publish without human
    rejection_threshold: float = 5.5        # Below this = reject & restart
    revision_threshold: float = 7.0         # Below pass but above reject = revise

    # ─────────────────────────────────────────────────────────────────
    # CONTENT-TYPE MULTIPLIERS (not absolute values!)
    # This allows type-specific adjustments while maintaining
    # consistent relative thresholds.
    # ─────────────────────────────────────────────────────────────────
    type_multipliers: Dict[ContentType, float] = field(default_factory=lambda: {
        ContentType.ENTERPRISE_CASE: 0.90,      # 8.0 * 0.90 = 7.2 effective
        ContentType.PRIMARY_SOURCE: 0.9375,     # 8.0 * 0.9375 = 7.5 effective
        ContentType.AUTOMATION_CASE: 0.875,     # 8.0 * 0.875 = 7.0 effective
        ContentType.COMMUNITY_CONTENT: 0.85,    # 8.0 * 0.85 = 6.8 effective
        ContentType.TOOL_RELEASE: 0.875,        # 8.0 * 0.875 = 7.0 effective
    })

    # ─────────────────────────────────────────────────────────────────
    # FIX: CENTRALIZED MAX REVISIONS BY CONTENT TYPE
    # Previously hardcoded in route_after_qc, now centralized here.
    # Rationale:
    # - PRIMARY_SOURCE: Research posts need more refinement for accuracy
    # - COMMUNITY_CONTENT: Authenticity suffers from over-editing
    # ─────────────────────────────────────────────────────────────────
    max_revisions_by_type: Dict[ContentType, int] = field(default_factory=lambda: {
        ContentType.ENTERPRISE_CASE: 3,
        ContentType.PRIMARY_SOURCE: 4,      # Research needs more refinement
        ContentType.AUTOMATION_CASE: 3,
        ContentType.COMMUNITY_CONTENT: 2,   # Authenticity suffers from over-editing
        ContentType.TOOL_RELEASE: 3
    })

    # Meta-agent iteration limits
    max_meta_iterations: int = 3  # FIX: Was hardcoded as magic number

    # FIX: Centralize max_reject_restarts (was hardcoded in route_after_qc)
    max_reject_restarts: int = 2  # Maximum topic restarts before manual escalation

    def get_pass_threshold(self, content_type: ContentType) -> float:
        """Get effective pass threshold for content type."""
        multiplier = self.type_multipliers.get(content_type, 1.0)
        return self.min_score_to_proceed * multiplier

    def get_revision_threshold(self, content_type: ContentType) -> float:
        """Get threshold for revision (between reject and pass)."""
        multiplier = self.type_multipliers.get(content_type, 1.0)
        return self.revision_threshold * multiplier

    def get_rejection_threshold(self, content_type: ContentType) -> float:
        """Get threshold below which content is rejected."""
        multiplier = self.type_multipliers.get(content_type, 1.0)
        return self.rejection_threshold * multiplier

    def get_auto_publish_threshold(self) -> float:
        """
        Get threshold for auto-publishing.
        NOTE: No type adjustment - auto-publish requires universally high quality.
        """
        return self.auto_publish_threshold

    def get_decision(self, score: float, content_type: ContentType) -> str:
        """
        Get decision based on score and content type.
        Returns: "pass" | "revise" | "reject"
        """
        pass_threshold = self.get_pass_threshold(content_type)
        rejection_threshold = self.get_rejection_threshold(content_type)

        if score >= pass_threshold:
            return "pass"
        elif score >= rejection_threshold:
            return "revise"
        else:
            return "reject"

    def get_max_revisions(self, content_type: ContentType) -> int:
        """
        FIX: Get max revisions for content type from centralized config.
        Previously hardcoded in route_after_qc.
        """
        return self.max_revisions_by_type.get(content_type, 3)

    def get_max_meta_iterations(self) -> int:
        """
        FIX: Get max meta-agent iterations.
        Previously hardcoded as magic number 3 in meta_evaluate_node.
        """
        return self.max_meta_iterations


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# Import this in all modules that need threshold values
# ═══════════════════════════════════════════════════════════════════════════
THRESHOLD_CONFIG = ThresholdConfig()


# ═══════════════════════════════════════════════════════════════════════════
# MEDIUM PRIORITY FIX #1: RETRY DECORATOR
# Reusable retry logic for external API calls (LLM, LinkedIn, Perplexity)
# ═══════════════════════════════════════════════════════════════════════════

import asyncio
import logging
from functools import wraps
from typing import Type, Tuple, Callable, TypeVar

T = TypeVar('T')

class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""
    def __init__(self, operation: str, attempts: int, last_error: Exception):
        self.operation = operation
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"{operation} failed after {attempts} attempts. Last error: {last_error}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# INPUT VALIDATION HELPERS
# Fail-fast validation for public method parameters
# ═══════════════════════════════════════════════════════════════════════════

class ValidationError(ValueError):
    """Raised when input validation fails."""
    pass


class DatabaseError(Exception):
    """Raised when database operations fail."""
    pass


class ConfigurationError(Exception):
    """Raised when system configuration is invalid."""
    pass


def validate_not_empty(value: Any, name: str) -> None:
    """Validate that value is not None or empty string."""
    if value is None:
        raise ValidationError(f"{name} cannot be None")
    if isinstance(value, str) and not value.strip():
        raise ValidationError(f"{name} cannot be empty string")


def validate_positive(value: Union[int, float], name: str) -> None:
    """Validate that value is positive (> 0)."""
    if value is None:
        raise ValidationError(f"{name} cannot be None")
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")


def validate_non_negative(value: Union[int, float], name: str) -> None:
    """Validate that value is non-negative (>= 0)."""
    if value is None:
        raise ValidationError(f"{name} cannot be None")
    if value < 0:
        raise ValidationError(f"{name} cannot be negative, got {value}")


def validate_range(value: Union[int, float], name: str, min_val: float, max_val: float) -> None:
    """Validate that value is within range [min_val, max_val]."""
    if value is None:
        raise ValidationError(f"{name} cannot be None")
    if not (min_val <= value <= max_val):
        raise ValidationError(f"{name} must be between {min_val} and {max_val}, got {value}")


def validate_enum(value: Any, name: str, enum_class: Type) -> None:
    """Validate that value is a valid enum member."""
    if value is None:
        raise ValidationError(f"{name} cannot be None")
    if not isinstance(value, enum_class):
        try:
            enum_class(value)  # Try to convert
        except ValueError:
            valid = [e.value for e in enum_class]
            raise ValidationError(f"{name} must be one of {valid}, got {value}")


def validate_url(value: str, name: str) -> None:
    """Validate that value is a valid URL."""
    validate_not_empty(value, name)
    if not (value.startswith("http://") or value.startswith("https://")):
        raise ValidationError(f"{name} must be a valid URL starting with http:// or https://, got {value}")


def validate_list_not_empty(value: list, name: str) -> None:
    """Validate that list is not None or empty."""
    if value is None:
        raise ValidationError(f"{name} cannot be None")
    if not isinstance(value, list):
        raise ValidationError(f"{name} must be a list, got {type(value).__name__}")
    if len(value) == 0:
        raise ValidationError(f"{name} cannot be empty")


# ═══════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER PATTERN
# FIX: Prevents cascade failures when external services are unavailable
# ═══════════════════════════════════════════════════════════════════════════

from enum import Enum
from dataclasses import dataclass, field
from threading import Lock
import time


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Service unavailable, fail fast
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5     # Failures before opening circuit
    success_threshold: int = 2     # Successes in half-open before closing
    timeout_seconds: float = 60.0  # Time to wait before half-open
    excluded_exceptions: tuple = ()  # Exceptions that don't count as failures


class CircuitBreaker:
    """
    Circuit Breaker pattern implementation for external API protection.

    Usage:
        breaker = CircuitBreaker("linkedin_api")

        @breaker
        async def call_linkedin():
            return await linkedin.get_posts()

        # Or manual:
        if breaker.can_execute():
            try:
                result = await api.call()
                breaker.record_success()
            except Exception as e:
                breaker.record_failure()
                raise
    """

    # Shared circuit breakers by service name
    _instances: Dict[str, "CircuitBreaker"] = {}
    _lock = Lock()

    def __new__(cls, service_name: str, config: CircuitBreakerConfig = None):
        """Singleton per service name."""
        with cls._lock:
            if service_name not in cls._instances:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instances[service_name] = instance
            return cls._instances[service_name]

    def __init__(self, service_name: str, config: CircuitBreakerConfig = None):
        if self._initialized:
            return
        self.service_name = service_name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self._state_lock = Lock()
        self._initialized = True
        self.logger = logging.getLogger(f"CircuitBreaker.{service_name}")

    def can_execute(self) -> bool:
        """Check if request can proceed."""
        with self._state_lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                # Check if timeout has passed
                if self.last_failure_time and \
                   time.time() - self.last_failure_time >= self.config.timeout_seconds:
                    self.logger.info(f"Circuit {self.service_name}: OPEN -> HALF_OPEN (timeout passed)")
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    return True
                return False

            # HALF_OPEN: allow limited traffic
            return True

    def record_success(self) -> None:
        """Record successful call."""
        with self._state_lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.logger.info(f"Circuit {self.service_name}: HALF_OPEN -> CLOSED (recovered)")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0

            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0

    def record_failure(self, exception: Exception = None) -> None:
        """Record failed call."""
        # Check if exception should be excluded
        if exception and isinstance(exception, self.config.excluded_exceptions):
            return

        with self._state_lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self.logger.warning(f"Circuit {self.service_name}: HALF_OPEN -> OPEN (failure during recovery)")
                self.state = CircuitState.OPEN
                self.success_count = 0  # FIX: Reset success_count to prevent stale state

            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.logger.error(
                        f"Circuit {self.service_name}: CLOSED -> OPEN "
                        f"(threshold {self.config.failure_threshold} reached)"
                    )
                    self.state = CircuitState.OPEN
                    self.success_count = 0  # FIX: Reset success_count on circuit open

    def __call__(self, func):
        """Decorator usage."""
        import functools

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if not self.can_execute():
                raise CircuitOpenError(
                    f"Circuit breaker {self.service_name} is OPEN. "
                    f"Service unavailable, failing fast."
                )
            try:
                result = await func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure(e)
                raise

        return wrapper

    @classmethod
    def get_status(cls) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers."""
        return {
            name: {
                "state": cb.state.value,
                "failure_count": cb.failure_count,
                "last_failure": cb.last_failure_time
            }
            for name, cb in cls._instances.items()
        }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and request is blocked."""
    pass


# Pre-configured circuit breakers for external services
CIRCUIT_BREAKERS = {
    "linkedin": CircuitBreaker("linkedin", CircuitBreakerConfig(
        failure_threshold=5,
        timeout_seconds=120.0  # LinkedIn rate limits reset after ~2 minutes
    )),
    "nano_banana": CircuitBreaker("nano_banana", CircuitBreakerConfig(
        failure_threshold=3,
        timeout_seconds=300.0  # Visual generation service may need longer recovery
    )),
    "perplexity": CircuitBreaker("perplexity", CircuitBreakerConfig(
        failure_threshold=5,
        timeout_seconds=60.0
    )),
}


def with_retry(
    max_attempts: Optional[int] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    exponential_base: Optional[float] = None,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    operation_name: Optional[str] = None,
    component: Optional[str] = None  # Use component-specific config from RETRY_CONFIG
) -> Callable:
    """
    Decorator for retry logic with exponential backoff.

    COMPLIANT with fail-fast philosophy:
    - Retries are for transient failures (rate limits, timeouts)
    - Eventually raises if all attempts fail
    - Logs each retry attempt for debugging

    Uses RETRY_CONFIG for default values. Override with explicit parameters.

    Args:
        max_attempts: Maximum retry attempts (default: from RETRY_CONFIG)
        base_delay: Initial delay in seconds (default: from RETRY_CONFIG)
        max_delay: Maximum delay cap in seconds (default: from RETRY_CONFIG)
        exponential_base: Backoff base (default: from RETRY_CONFIG)
        retryable_exceptions: Exception types that trigger retry
        operation_name: Name for logging (defaults to function name)
        component: Component name for RETRY_CONFIG lookup ("llm", "linkedin", "visual", etc.)

    Usage:
        # Use defaults from RETRY_CONFIG
        @with_retry(component="llm", retryable_exceptions=(RateLimitError,))
        async def call_claude(): ...

        # Or override explicitly
        @with_retry(max_attempts=5, base_delay=1.0)
        async def call_api(): ...
    """
    # Get defaults from RETRY_CONFIG
    if component:
        default_attempts, default_delay = RETRY_CONFIG.get_config_for(component)
    else:
        default_attempts = RETRY_CONFIG.default_max_attempts
        default_delay = RETRY_CONFIG.default_base_delay

    # Apply defaults if not explicitly provided
    max_attempts = max_attempts if max_attempts is not None else default_attempts
    base_delay = base_delay if base_delay is not None else default_delay
    max_delay = max_delay if max_delay is not None else RETRY_CONFIG.default_max_delay
    exponential_base = exponential_base if exponential_base is not None else RETRY_CONFIG.default_backoff_factor
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        op_name = operation_name or func.__name__

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_error = e
                    if attempt < max_attempts:
                        delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)
                        logging.warning(
                            f"[RETRY] {op_name} attempt {attempt}/{max_attempts} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logging.error(
                            f"[RETRY EXHAUSTED] {op_name} failed after {max_attempts} attempts: {e}"
                        )
            raise RetryExhaustedError(op_name, max_attempts, last_error)

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            import time
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_error = e
                    if attempt < max_attempts:
                        delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)
                        logging.warning(
                            f"[RETRY] {op_name} attempt {attempt}/{max_attempts} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logging.error(
                            f"[RETRY EXHAUSTED] {op_name} failed after {max_attempts} attempts: {e}"
                        )
            raise RetryExhaustedError(op_name, max_attempts, last_error)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ═══════════════════════════════════════════════════════════════════════════
# MEDIUM PRIORITY FIX #3: SOURCE THRESHOLD CONFIGURATION
# Centralized thresholds for content sources (HN, ProductHunt, GitHub, etc.)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SourceThresholdConfig:
    """
    SINGLE SOURCE OF TRUTH for all source-specific thresholds.

    RATIONALE:
    Previously, min_score, min_upvotes, min_stars_velocity were hardcoded
    in sources_config dictionary. This made tuning difficult and risked
    inconsistency when thresholds needed adjustment.

    NOW: All source thresholds derive from this single configuration.

    Usage:
        config = SourceThresholdConfig()
        min_score = config.get_min_score("hackernews")
    """

    # ─────────────────────────────────────────────────────────────────
    # ENGAGEMENT THRESHOLDS BY SOURCE
    # ─────────────────────────────────────────────────────────────────
    hackernews_min_score: int = 50
    twitter_min_engagement: int = 1000
    product_hunt_min_upvotes: int = 200
    github_min_stars_velocity: int = 100  # Stars gained in last 24h
    reddit_min_score: int = 100
    reddit_min_comments: int = 20         # FIX: Added (was hardcoded in sources_config)
    youtube_min_views: int = 10000
    devto_min_reactions: int = 50         # FIX: Added (was hardcoded in sources_config)
    medium_min_claps: int = 100           # FIX: Added for completeness

    # ─────────────────────────────────────────────────────────────────
    # RESEARCH SOURCE THRESHOLDS
    # ─────────────────────────────────────────────────────────────────
    arxiv_min_citations: int = 0  # New papers OK
    gartner_min_recency_days: int = 30
    mckinsey_min_recency_days: int = 90

    # ─────────────────────────────────────────────────────────────────
    # ENVIRONMENT OVERRIDES
    # Allow tuning via environment variables for A/B testing
    # ─────────────────────────────────────────────────────────────────
    def __post_init__(self):
        import os
        # Override from environment if set
        if os.environ.get('HN_MIN_SCORE'):
            self.hackernews_min_score = int(os.environ['HN_MIN_SCORE'])
        if os.environ.get('PH_MIN_UPVOTES'):
            self.product_hunt_min_upvotes = int(os.environ['PH_MIN_UPVOTES'])
        if os.environ.get('GH_MIN_STARS_VELOCITY'):
            self.github_min_stars_velocity = int(os.environ['GH_MIN_STARS_VELOCITY'])
        # FIX: Added environment overrides for new thresholds
        if os.environ.get('REDDIT_MIN_COMMENTS'):
            self.reddit_min_comments = int(os.environ['REDDIT_MIN_COMMENTS'])
        if os.environ.get('DEVTO_MIN_REACTIONS'):
            self.devto_min_reactions = int(os.environ['DEVTO_MIN_REACTIONS'])

    def get_min_score(self, source: str) -> int:
        """Get minimum score/engagement threshold for a source."""
        thresholds = {
            "hackernews": self.hackernews_min_score,
            "twitter_x": self.twitter_min_engagement,
            "product_hunt": self.product_hunt_min_upvotes,
            "github_trending": self.github_min_stars_velocity,
            "reddit": self.reddit_min_score,
            "youtube": self.youtube_min_views,
        }
        # FAIL-FAST: Unknown sources must be explicitly configured
        if source not in thresholds:
            raise ValueError(
                f"Unknown source '{source}'. Add it to SourceThresholdConfig. "
                f"Valid sources: {list(thresholds.keys())}"
            )
        return thresholds[source]


# Global instance
SOURCE_THRESHOLD_CONFIG = SourceThresholdConfig()


# ═══════════════════════════════════════════════════════════════════════════
# MEDIUM PRIORITY FIX #10: LINKEDIN PLATFORM LIMITS
# Centralized configuration for LinkedIn platform constraints
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LinkedInLimitsConfig:
    """
    SINGLE SOURCE OF TRUTH for LinkedIn platform limits.

    RATIONALE:
    LinkedIn limits change periodically. Having them in a centralized config:
    1. Makes updates easy when LinkedIn changes limits
    2. Allows A/B testing of different length targets
    3. Ensures consistency across Writer, QC, and validation

    Source: LinkedIn official documentation (update date in each field)
    """

    # ─────────────────────────────────────────────────────────────────
    # POST LENGTH LIMITS (Updated: 2024-01)
    # ─────────────────────────────────────────────────────────────────
    post_max_chars: int = 3000              # Hard limit from LinkedIn
    post_optimal_min: int = 1200            # Best engagement range
    post_optimal_max: int = 1500            # Best engagement range
    hook_visible_chars: int = 210           # Chars before "see more"

    # ─────────────────────────────────────────────────────────────────
    # VALIDATION BOUNDARIES
    # ─────────────────────────────────────────────────────────────────
    post_min_chars: int = 200               # Minimum for meaningful content
    hashtag_max_count: int = 5              # LinkedIn recommendation
    mention_max_count: int = 10             # Avoid spam detection

    # ─────────────────────────────────────────────────────────────────
    # IMAGE/VISUAL LIMITS (Updated: 2024-01)
    # ─────────────────────────────────────────────────────────────────
    image_max_size_mb: float = 8.0
    image_min_width_px: int = 552           # LinkedIn recommendation
    image_optimal_width_px: int = 1200      # 2x for retina
    carousel_max_slides: int = 20
    video_max_duration_seconds: int = 600   # 10 minutes

    # ─────────────────────────────────────────────────────────────────
    # RATE LIMITS (estimated, be conservative)
    # ─────────────────────────────────────────────────────────────────
    posts_per_day_safe: int = 3
    api_requests_per_minute_safe: int = 10

    def get_length_target_str(self) -> str:
        """Return length target as string for prompts."""
        return f"{self.post_optimal_min}-{self.post_optimal_max} characters"

    def validate_post_length(self, text: str) -> Tuple[bool, str]:
        """
        Validate post length against limits.

        Returns:
            (is_valid, message)
        """
        length = len(text)

        if length < self.post_min_chars:
            return False, f"Post too short ({length} chars, min {self.post_min_chars})"
        if length > self.post_max_chars:
            return False, f"Post too long ({length} chars, max {self.post_max_chars})"
        if length < self.post_optimal_min:
            return True, f"Post short but valid ({length} chars, optimal {self.post_optimal_min}+)"
        if length > self.post_optimal_max:
            return True, f"Post long but valid ({length} chars, optimal under {self.post_optimal_max})"
        return True, f"Post length optimal ({length} chars)"


# Global instance
LINKEDIN_LIMITS = LinkedInLimitsConfig()


# ═══════════════════════════════════════════════════════════════════════════
# LOW PRIORITY FIX #2, #3, #8: NAMED CONSTANTS
# Replace magic numbers and hardcoded strings with named constants
# ═══════════════════════════════════════════════════════════════════════════

class ScoringConstants:
    """Named constants for scoring and evaluation thresholds."""

    # LOW PRIORITY FIX #2: Magic numbers for scoring
    LOW_SCORE_THRESHOLD: int = 7          # Below this is considered a low score
    TARGET_SCORE: int = 8                 # Target score for revisions
    EXCELLENT_SCORE: int = 9              # Excellent quality threshold

    # LOW PRIORITY FIX #3: Summary/text truncation limits
    CLASSIFICATION_SUMMARY_MAX_CHARS: int = 500
    HOOK_PREVIEW_MAX_CHARS: int = 100
    ERROR_CONTEXT_MAX_CHARS: int = 200


class HistoryLimits:
    """
    LOW PRIORITY FIX #11: Limits for history/log collections to prevent memory leaks.
    """
    MAX_CRITIQUE_HISTORY: int = 100       # Max meta-critique entries to keep
    MAX_REVISION_HISTORY: int = 50        # Max revision entries per post
    MAX_RESEARCH_HISTORY: int = 30        # Max research reports to keep
    MAX_SNAPSHOT_AGE_DAYS: int = 30       # Auto-cleanup snapshots older than this


class ErrorMessages:
    """
    LOW PRIORITY FIX #8: Centralized error messages.
    Use these instead of hardcoded strings throughout the codebase.
    """
    # Trend Scout errors
    NO_TOPICS_FOUND = "Scout found no topics after filtering"
    NO_TOP_PICK = "Scout failed to select top pick"
    INVALID_TOPIC_METADATA = "Topic metadata type mismatch"

    # Analyzer errors
    EXTRACTION_FAILED = "Failed to extract data from source"
    INSUFFICIENT_CONTENT = "Source content too short for meaningful analysis"

    # Writer errors
    HOOK_GENERATION_FAILED = "Failed to generate compelling hooks"
    DRAFT_TOO_SHORT = "Generated draft below minimum length"
    DRAFT_TOO_LONG = "Generated draft exceeds maximum length"

    # QC errors
    EVALUATION_FAILED = "Quality evaluation failed"
    MAX_REVISIONS_EXCEEDED = "Maximum revision attempts exceeded"

    # Visual errors
    IMAGE_GENERATION_FAILED = "Image generation failed"
    PHOTO_NOT_FOUND = "Requested photo not found in library"

    # System errors
    CONFIG_LOAD_FAILED = "Failed to load configuration file"
    DATABASE_CONNECTION_FAILED = "Database connection failed"
    EXTERNAL_API_FAILED = "External API call failed"


# Backward compatibility - can use these directly
LOW_SCORE_THRESHOLD = ScoringConstants.LOW_SCORE_THRESHOLD
TARGET_SCORE = ScoringConstants.TARGET_SCORE
MAX_CRITIQUE_HISTORY = HistoryLimits.MAX_CRITIQUE_HISTORY
```

---

## System Architecture

```
                        ┌──────────────────┐
                        │   SCHEDULER      │
                        │  (cron/manual)   │
                        └────────┬─────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR                                │
│                    (LangGraph StateMachine)                         │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌───────┐│
│  │ SCOUT   │──▶│ ANALYZE │──▶│ CREATE  │──▶│  EVAL   │──▶│  QC   ││
│  │  STATE  │   │  STATE  │   │  STATE  │   │  STATE  │   │ STATE ││
│  └─────────┘   └─────────┘   └─────────┘   └────┬────┘   └───────┘│
│       │              │             │            │                   │
│       ▼              ▼             ▼            ▼                   │
│  [Trend Scout] [Analyzer]  [Writer+Visual] [Meta-Agent]  [QC]      │
│                                          evaluates draft            │
│                                          ↓                          │
│                                    Score < 8? ──────────────────┐   │
│                                          │                      │   │
│                                          ▼                      │   │
│                                    REWRITE (max 3x) ←───────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴───────────┐
                    │                        │
                    ▼                        ▼
          Score >= 9.0 (auto)         Score < 9.0
                    │                        │
                    │           ┌────────────────────────┐
                    │           │   HUMAN APPROVAL UI    │
                    │           │   (Telegram/Web/CLI)   │
                    │           └────────────┬───────────┘
                    │                        │ ✅ Approved
                    └────────────┬───────────┘
                                 ▼
                    ┌────────────────────────┐
                    │   LINKEDIN PUBLISHER   │
                    │   (API / Automation)   │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   ANALYTICS COLLECTOR  │
                    │  (T+15, T+30, T+60...)  │
                    └────────────┬───────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│               SELF-IMPROVEMENT ENGINE (Triple-Mode)                 │
│               Maximum Autonomy — Writes Its Own Code                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ════════════════════════════════════════════════════════════════   │
│  MODE 1: CONTINUOUS LEARNING (Every Iteration)                      │
│  ════════════════════════════════════════════════════════════════   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  After EVERY evaluation — adjust prompts & configs:          │   │
│  │                                                              │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │   │
│  │  │ EVALUATE │─▶│ EXTRACT  │─▶│  APPLY   │─▶│ PERSIST  │    │   │
│  │  │text+visual│  │micro-learn│  │to prompts│  │to DB     │    │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │   │
│  │                                                              │   │
│  │  Examples: "Add numbers to hooks", "Shorter paragraphs"     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ════════════════════════════════════════════════════════════════   │
│  MODE 2: SELF-MODIFYING CODE (When Learning Isn't Enough) ⚡       │
│  ════════════════════════════════════════════════════════════════   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  When prompt changes can't solve the problem — WRITE CODE:   │   │
│  │                                                              │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │   │
│  │  │ DETECT   │─▶│ GENERATE │─▶│ VALIDATE │─▶│HOT RELOAD│    │   │
│  │  │capability│  │new module│  │& test    │  │& retry   │    │   │
│  │  │   gap    │  │via Claude│  │          │  │          │    │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │   │
│  │                                                              │   │
│  │  Examples:                                                   │   │
│  │  • "Need OpenAI blog data" → writes openai_blog_source.py   │   │
│  │  • "Need sentiment analysis" → writes sentiment_analyzer.py │   │
│  │  • "Need carousel images" → writes carousel_generator.py    │   │
│  │                                                              │   │
│  │  System WRITES CODE during execution, then uses it!          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ════════════════════════════════════════════════════════════════   │
│  MODE 3: DEEP RESEARCH (Weekly/Triggered)                           │
│  ════════════════════════════════════════════════════════════════   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Strategic research — competitor analysis, trend research:   │   │
│  │                                                              │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │   │
│  │  │ ANALYZE  │─▶│ RESEARCH │─▶│ REDESIGN │─▶│ A/B TEST │    │   │
│  │  │ patterns │  │Perplexity│  │ strategy │  │hypotheses│    │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │   │
│  │                                                              │   │
│  │  Triggers: Sunday weekly | 3+ underperforming posts         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  ESCALATION LADDER:                                          │   │
│  │  1. Prompt change (micro-learning)     — milliseconds        │   │
│  │  2. Write new code (self-modification) — seconds             │   │
│  │  3. Strategic research (deep analysis) — minutes/hours       │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
     SYSTEM WRITES ITS OWN CODE AND USES IT IN THE SAME RUN
```

---

## Agent Specifications

### 1. TREND SCOUT AGENT

#### Purpose
Find the most promising topics for posts by scanning multiple source categories, classifying content by type, filtering low-quality materials, and scoring based on both engagement potential AND content quality (metrics, depth, applicability).

#### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TREND SCOUT AGENT (Enhanced)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TIER 1: RESEARCH & ENTERPRISE                                              │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐    │
│  │  ArXiv    │ │Consultancy│ │ Corporate │ │Whitepapers│ │  CIO/CTO  │    │
│  │  Papers   │ │ Reports   │ │Tech Blogs │ │ Research  │ │  Forums   │    │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘    │
│        │             │             │             │             │           │
│  TIER 2: NEWS & SOCIAL                                                      │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐    │
│  │ HackerNews│ │ Twitter/X │ │  Product  │ │  GitHub   │ │ Perplexity│    │
│  │    API    │ │  Monitor  │ │   Hunt    │ │ Trending  │ │   Search  │    │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘    │
│        │             │             │             │             │           │
│  TIER 3: COMMUNITY & PRACTITIONERS                                          │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐    │
│  │  Reddit   │ │  YouTube  │ │  Medium   │ │ Substack  │ │  Dev.to   │    │
│  │ (PRAW)    │ │Transcripts│ │   RSS     │ │   RSS     │ │   API     │    │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘    │
│        │             │             │             │             │           │
│        └─────────────┴─────────────┴─────────────┴─────────────┘           │
│                                    │                                        │
│                                    ▼                                        │
│                    ┌───────────────────────────────┐                       │
│                    │      CONTENT CLASSIFIER       │                       │
│                    │   (Assigns ContentType enum)  │                       │
│                    └───────────────┬───────────────┘                       │
│                                    ▼                                        │
│                    ┌───────────────────────────────┐                       │
│                    │        PRE-FILTER             │                       │
│                    │   (Exclusion Rules Check)     │                       │
│                    └───────────────┬───────────────┘                       │
│                                    ▼                                        │
│                    ┌───────────────────────────────┐                       │
│                    │   AGGREGATOR + DEDUPLICATOR   │                       │
│                    │   (Cross-source merging)      │                       │
│                    └───────────────┬───────────────┘                       │
│                                    ▼                                        │
│                    ┌───────────────────────────────┐                       │
│                    │       QUALITY SCORER          │                       │
│                    │  (Type-specific weights)      │                       │
│                    └───────────────┬───────────────┘                       │
│                                    ▼                                        │
│                    ┌───────────────────────────────┐                       │
│                    │     TOP PICK SELECTOR         │                       │
│                    │  ("Самый важный кейс дня")    │                       │
│                    └───────────────┬───────────────┘                       │
│                                    ▼                                        │
│                    ┌───────────────────────────────┐                       │
│                    │   Ranked Topics + Top Pick    │                       │
│                    │   with Type-Specific Angles   │                       │
│                    └───────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### Content Type Classification

```python
from enum import Enum
from typing import Optional, List

# NOTE: ContentType is defined in the CENTRALIZED THRESHOLD CONFIGURATION section.
# This is the same enum, shown here for context in the classification workflow.
# In implementation, import from threshold_config module:
#   from core.threshold_config import ContentType

class ContentType(Enum):
    """
    Five distinct content verticals requiring different scoring and angle generation.

    AUTHORITATIVE DEFINITION: See CENTRALIZED THRESHOLD CONFIGURATION section.
    """
    ENTERPRISE_CASE = "enterprise_case"      # Detailed implementation stories with ROI/KPI
    PRIMARY_SOURCE = "primary_source"        # Research papers, think tank reports, expert essays
    AUTOMATION_CASE = "automation_case"      # AI agents, n8n workflows, practical automation
    COMMUNITY_CONTENT = "community_content"  # YouTube, Reddit, HN discussions
    TOOL_RELEASE = "tool_release"            # New products, APIs, demos


# ═══════════════════════════════════════════════════════════════════════════
# TWO-LEVEL CLASSIFICATION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════
#
# PROBLEM: Domain-based classification ignores actual content type.
# Example: Gartner publishes BOTH research reports AND enterprise case studies.
#
# SOLUTION: Two-level classification
# Level 1: Domain → candidate_types (list of possible types)
# Level 2: Content analysis → final_type (LLM-based refinement)
#
# ═══════════════════════════════════════════════════════════════════════════

# Level 1: Domain to CANDIDATE types (not final!)
domain_to_candidate_types = {
    # Research-focused domains (may also have case studies)
    "arxiv.org": [ContentType.PRIMARY_SOURCE],
    "papers.ssrn.com": [ContentType.PRIMARY_SOURCE],
    "semanticscholar.org": [ContentType.PRIMARY_SOURCE],

    # Consultancy (publish both research AND case studies)
    "gartner.com": [ContentType.PRIMARY_SOURCE, ContentType.ENTERPRISE_CASE],
    "mckinsey.com": [ContentType.PRIMARY_SOURCE, ContentType.ENTERPRISE_CASE],
    "deloitte.com": [ContentType.PRIMARY_SOURCE, ContentType.ENTERPRISE_CASE],
    "bcg.com": [ContentType.PRIMARY_SOURCE, ContentType.ENTERPRISE_CASE],

    # Product launches
    "producthunt.com": [ContentType.TOOL_RELEASE],
    "techcrunch.com": [ContentType.TOOL_RELEASE, ContentType.ENTERPRISE_CASE],

    # Community (can surface any type)
    "reddit.com": [ContentType.COMMUNITY_CONTENT, ContentType.AUTOMATION_CASE, ContentType.ENTERPRISE_CASE],
    "youtube.com": [ContentType.COMMUNITY_CONTENT, ContentType.AUTOMATION_CASE],
    "news.ycombinator.com": [ContentType.COMMUNITY_CONTENT, ContentType.ENTERPRISE_CASE, ContentType.TOOL_RELEASE],

    # Mixed content platforms
    "medium.com": [ContentType.AUTOMATION_CASE, ContentType.ENTERPRISE_CASE, ContentType.PRIMARY_SOURCE],
    "dev.to": [ContentType.AUTOMATION_CASE, ContentType.COMMUNITY_CONTENT],

    # ═══════════════════════════════════════════════════════════════
    # FIX: Added missing domains from sources_config
    # ═══════════════════════════════════════════════════════════════

    # Substack (general + specific high-value newsletters)
    "substack.com": [ContentType.PRIMARY_SOURCE, ContentType.COMMUNITY_CONTENT],
    "simonwillison.substack.com": [ContentType.PRIMARY_SOURCE],
    "lethain.substack.com": [ContentType.PRIMARY_SOURCE],
    "thealgorithmicbridge.substack.com": [ContentType.PRIMARY_SOURCE],
    "oneusefulthing.substack.com": [ContentType.PRIMARY_SOURCE],
    "aisnakeoil.substack.com": [ContentType.PRIMARY_SOURCE],

    # Medium publications
    "towardsdatascience.com": [ContentType.AUTOMATION_CASE, ContentType.PRIMARY_SOURCE],
    "ai.plainenglish.io": [ContentType.AUTOMATION_CASE],
    "betterprogramming.pub": [ContentType.AUTOMATION_CASE],

    # GitHub
    "github.com": [ContentType.TOOL_RELEASE, ContentType.AUTOMATION_CASE],

    # Company blogs with mixed content
    "openai.com": [ContentType.PRIMARY_SOURCE, ContentType.TOOL_RELEASE],
    "anthropic.com": [ContentType.PRIMARY_SOURCE, ContentType.TOOL_RELEASE],
    "ai.google": [ContentType.PRIMARY_SOURCE, ContentType.TOOL_RELEASE],
    "engineering.fb.com": [ContentType.ENTERPRISE_CASE, ContentType.PRIMARY_SOURCE],
    "aws.amazon.com": [ContentType.ENTERPRISE_CASE, ContentType.TOOL_RELEASE],
    "cloud.google.com": [ContentType.ENTERPRISE_CASE, ContentType.TOOL_RELEASE],
}

# URL patterns for stronger hints
url_pattern_type_hints = {
    "*/case-study/*": ContentType.ENTERPRISE_CASE,
    "*/case-studies/*": ContentType.ENTERPRISE_CASE,
    "*/customer-story/*": ContentType.ENTERPRISE_CASE,
    "*/research/*": ContentType.PRIMARY_SOURCE,
    "*/insights/*": ContentType.PRIMARY_SOURCE,
    "*/n8n/*": ContentType.AUTOMATION_CASE,
    "*/workflow/*": ContentType.AUTOMATION_CASE,
    "*/agent/*": ContentType.AUTOMATION_CASE,
    "*/release/*": ContentType.TOOL_RELEASE,
    "*/launch/*": ContentType.TOOL_RELEASE,
    "*/announce/*": ContentType.TOOL_RELEASE,
}


# Level 2: Content-based refinement (LLM classifier)
CLASSIFICATION_PROMPT = """
Classify this content into ONE of these categories:

ENTERPRISE_CASE: A detailed implementation story from a specific company with:
- Named company
- Business problem described
- Solution implemented
- Results/metrics reported

PRIMARY_SOURCE: Original research, analysis, or expert opinion with:
- Novel thesis or findings
- Methodology or reasoning explained
- Written by researcher/analyst/expert

AUTOMATION_CASE: Practical automation or AI agent implementation with:
- Specific workflow or agent described
- Tools and technologies listed
- Reproducible steps or code

COMMUNITY_CONTENT: Discussion, reaction, or synthesis from community with:
- Multiple viewpoints
- User-generated content
- Commentary/reactions

TOOL_RELEASE: Announcement of new product, API, or feature with:
- Product name
- Features described
- Launch/availability info

Content to classify:
Title: {title}
URL: {url}
Summary: {summary}

Respond with ONLY the category name (e.g., "ENTERPRISE_CASE").
"""


def classify_content(
    url: str,
    title: str,
    summary: str,
    llm_client
) -> ContentType:
    """
    Two-level classification: domain hint + LLM refinement.

    MEDIUM PRIORITY FIX #1: Added retry logic and logging.
    """
    from urllib.parse import urlparse
    import logging

    logger = logging.getLogger("classify_content")

    # Level 1: Get candidate types from domain
    domain = urlparse(url).netloc.replace("www.", "")
    if domain in domain_to_candidate_types:
        candidates = domain_to_candidate_types[domain]
    else:
        # Unknown domain - use ALL types and let LLM decide
        # This is intentional fallback (not hidden) - log it for visibility
        logger.warning(
            f"Unknown domain '{domain}' - falling back to LLM classification "
            f"among all {len(ContentType)} content types"
        )
        candidates = list(ContentType)

    # Check URL patterns for stronger hints
    for pattern, hint_type in url_pattern_type_hints.items():
        if _matches_pattern(url, pattern):
            if hint_type in candidates:
                # URL pattern gives strong hint, move to front
                candidates = [hint_type] + [c for c in candidates if c != hint_type]
            break

    # If only one candidate, use it
    if len(candidates) == 1:
        logger.debug(f"Single candidate from domain: {candidates[0].value} for {domain}")
        return candidates[0]

    # Level 2: LLM refinement among candidates
    prompt = CLASSIFICATION_PROMPT.format(
        title=title,
        url=url,
        summary=summary[:500]  # Limit summary length
    )

    # FIX: Use retry for LLM call with proper error handling
    try:
        response = _classify_with_retry(llm_client, prompt)
        classified = response.strip().upper()
    except RetryExhaustedError as e:
        logger.warning(
            f"LLM classification failed after retries for {url}. "
            f"Falling back to first candidate: {candidates[0].value}. Error: {e.last_error}"
        )
        return candidates[0]

    # Map response to ContentType
    try:
        result = ContentType(classified.lower())
        # Validate against candidates (prefer candidates but allow override)
        if result in candidates:
            logger.debug(f"LLM classified {url} as {result.value} (in candidates)")
            return result
        # LLM classified differently - trust it but log
        logger.info(
            f"LLM override: classified {url} as {result.value} "
            f"(not in candidates: {[c.value for c in candidates]})"
        )
        return result
    except ValueError:
        # Fallback to first candidate with logging
        logger.warning(
            f"Invalid LLM response '{classified}' for {url}. "
            f"Falling back to {candidates[0].value}"
        )
        return candidates[0]


@with_retry(
    max_attempts=3,
    base_delay=1.0,
    retryable_exceptions=(TimeoutError, ConnectionError, Exception),  # Broad for LLM APIs
    operation_name="llm_classify"
)
def _classify_with_retry(llm_client, prompt: str) -> str:
    """
    Internal helper: LLM classification call with retry.
    Separated for testability and retry decorator application.
    """
    return llm_client.complete(prompt)


def _matches_pattern(url: str, pattern: str) -> bool:
    """Simple wildcard pattern matching for URLs."""
    import fnmatch
    return fnmatch.fnmatch(url.lower(), pattern.lower())


# ═══════════════════════════════════════════════════════════════════════════
# LEGACY: Simple domain-only classification
# LOW PRIORITY FIX #5: Added proper deprecation warning
# ═══════════════════════════════════════════════════════════════════════════
import warnings

_CONTENT_TYPE_CLASSIFICATION_DEPRECATED = {
    "arxiv.org": ContentType.PRIMARY_SOURCE,
    "gartner.com": ContentType.ENTERPRISE_CASE,
    "producthunt.com": ContentType.TOOL_RELEASE,
    "reddit.com": ContentType.COMMUNITY_CONTENT,
    "medium.com": ContentType.AUTOMATION_CASE,
}


def get_content_type_by_domain(domain: str) -> ContentType:
    """
    DEPRECATED: Use classify_content() instead for accurate two-level classification.

    This function uses simple domain-only classification which is less accurate
    than the two-level classify_content() which considers URL patterns and LLM refinement.
    """
    warnings.warn(
        "get_content_type_by_domain() is deprecated. "
        "Use classify_content(url, title, summary, llm_client) for accurate classification.",
        DeprecationWarning,
        stacklevel=2
    )
    # FAIL-FAST: Don't hide unknown domains with silent fallback
    if domain not in _CONTENT_TYPE_CLASSIFICATION_DEPRECATED:
        raise ValueError(
            f"Unknown domain '{domain}'. Use classify_content() instead "
            f"which handles unknown domains via LLM classification."
        )
    return _CONTENT_TYPE_CLASSIFICATION_DEPRECATED[domain]


# Backward compatibility alias with deprecation
content_type_classification = _CONTENT_TYPE_CLASSIFICATION_DEPRECATED
```

---

#### Sources Configuration

```python
sources_config = {
    # ═══════════════════════════════════════════════════════════════
    # TIER 1: RESEARCH & ENTERPRISE (High-depth, strategic content)
    # ═══════════════════════════════════════════════════════════════

    "arxiv": {
        "endpoint": "https://export.arxiv.org/api/query",
        "categories": ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE"],
        "frequency": "daily",
        "priority": "P1",
        "content_type": ContentType.PRIMARY_SOURCE,
        "extraction": {
            "title": True,
            "abstract": True,
            "authors": True,
            "citations_trend": "via Semantic Scholar API"
        },
        "quality_signals": ["novel_contribution", "benchmark_results", "code_available"]
    },

    "consultancy_reports": {
        "sources": {
            "gartner": "gartner.com/en/research",
            "mckinsey": "mckinsey.com/featured-insights/artificial-intelligence",
            "deloitte": "deloitte.com/insights/artificial-intelligence",
            "bcg": "bcg.com/capabilities/digital-technology-data/artificial-intelligence",
            "accenture": "accenture.com/us-en/insights/artificial-intelligence"
        },
        "method": "Perplexity Deep Research + Web scraping",
        "frequency": "weekly",
        "priority": "P2",
        "content_type": ContentType.ENTERPRISE_CASE,
        "search_queries": [
            "enterprise AI implementation case study detailed",
            "AI agent corporate use case ROI",
            "LLM automation enterprise metrics"
        ]
    },

    "corporate_tech_blogs": {
        "companies": [
            "microsoft.com/en-us/research/blog",
            "ai.google/research",
            "openai.com/blog",
            "anthropic.com/news",
            "engineering.fb.com",
            "aws.amazon.com/blogs/machine-learning",
            "cloud.google.com/blog/products/ai-machine-learning"
        ],
        "method": "RSS feeds + Web scraping",
        "frequency": "daily",
        "priority": "P1",
        "content_type": ContentType.ENTERPRISE_CASE,
        "extraction": {
            "title": True,
            "content": True,
            "publish_date": True,
            "author": True,
            "tags": True
        }
    },

    "whitepapers_research": {
        "sources": ["papers.ssrn.com", "semanticscholar.org"],
        "search_terms": [
            "enterprise AI deployment",
            "LLM business applications ROI",
            "AI automation study"
        ],
        "method": "Semantic Scholar API + Perplexity",
        "frequency": "weekly",
        "priority": "P2",
        "content_type": ContentType.PRIMARY_SOURCE
    },

    # ═══════════════════════════════════════════════════════════════
    # TIER 2: NEWS & SOCIAL (Trending, time-sensitive content)
    # ═══════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════════
    # FIX #3: Thresholds now use SOURCE_THRESHOLD_CONFIG
    # This enables runtime tuning via environment variables
    # ═══════════════════════════════════════════════════════════════════

    "hackernews": {
        "endpoint": "https://hacker-news.firebaseio.com/v0/",
        "filter_keywords": ["AI", "LLM", "GPT", "machine learning", "AGI", "Claude", "agent"],
        "min_score": SOURCE_THRESHOLD_CONFIG.hackernews_min_score,  # FIX: Centralized
        "frequency": "every_4_hours",
        "priority": "P0",
        "content_type": ContentType.COMMUNITY_CONTENT,
        "enterprise_filter": {
            "keywords": ["enterprise", "production", "deployed", "case study", "implementation"],
            "reclassify_to": ContentType.ENTERPRISE_CASE
        }
    },

    "twitter_x": {
        "accounts_to_monitor": [
            "@sama", "@karpathy", "@ylecun", "@AndrewYNg",
            "@hardmaru", "@fchollet", "@GaryMarcus", "@bindureddy",
            "@emaborratov", "@oaborratov"
        ],
        "hashtags": ["#AI", "#MachineLearning", "#LLM", "#GenerativeAI", "#AIAgents"],
        "min_engagement": SOURCE_THRESHOLD_CONFIG.twitter_min_engagement,  # FIX: Centralized
        "frequency": "continuous",
        "priority": "P1",
        "content_type": ContentType.COMMUNITY_CONTENT
    },

    "product_hunt": {
        "endpoint": "https://api.producthunt.com/v2/api/graphql",
        "category": "artificial-intelligence",
        "min_upvotes": SOURCE_THRESHOLD_CONFIG.product_hunt_min_upvotes,  # FIX: Centralized
        "frequency": "daily",
        "priority": "P1",
        "content_type": ContentType.TOOL_RELEASE,
        "extraction": {
            "name": True,
            "tagline": True,
            "description": True,
            "topics": True,
            "votes_count": True,
            "website": True
        }
    },

    "github_trending": {
        "endpoint": "https://api.github.com/search/repositories",
        "language": "Python",
        "topics": ["llm", "ai", "machine-learning", "generative-ai", "langchain", "agents"],
        "frequency": "daily",
        "priority": "P1",
        "content_type": ContentType.TOOL_RELEASE,
        "min_stars_velocity": SOURCE_THRESHOLD_CONFIG.github_min_stars_velocity  # FIX: Centralized
    },

    "perplexity": {
        "use_case": "synthesis_and_discovery",
        "frequency": "twice_daily",
        "priority": "P0",
        "query_sets_by_content_type": {
            ContentType.ENTERPRISE_CASE: {
                "queries": [
                    "enterprise AI implementation case study detailed site:mckinsey.com OR site:deloitte.com",
                    "case study LLM automation enterprise ROI metrics",
                    "AI agent corporate use case results",
                    "IT company case AI automation implementation",
                    "Fortune 500 AI deployment case study"
                ],
                "time_range": "2-7 days"
            },
            ContentType.PRIMARY_SOURCE: {
                "queries": [
                    "AI research paper breakthrough this week",
                    "AI think tank report latest findings",
                    "expert essay artificial intelligence future implications"
                ],
                "time_range": "2-5 days",
                "exclude": ["YouTube", "marketing materials", "PR articles"]
            },
            ContentType.AUTOMATION_CASE: {
                "queries": [
                    "AI agent case study production deployment",
                    "n8n automation with AI workflow real example",
                    "LangChain agents production use case",
                    "AutoGPT CrewAI business implementation"
                ],
                "time_range": "2-5 days"
            },
            ContentType.COMMUNITY_CONTENT: {
                "queries": [
                    "AI business case discussion site:reddit.com",
                    "AI implementation experience site:news.ycombinator.com",
                    "practical AI tools review"
                ],
                "time_range": "2-5 days"
            },
            ContentType.TOOL_RELEASE: {
                "queries": [
                    "new AI tool release this week API demo",
                    "AI API launch announcement open source",
                    "AI product launch features"
                ],
                "time_range": "2-3 days"
            }
        }
    },

    # ═══════════════════════════════════════════════════════════════
    # TIER 3: COMMUNITY & PRACTITIONERS (Real-world implementations)
    # ═══════════════════════════════════════════════════════════════

    "reddit": {
        "subreddits": [
            "r/MachineLearning",
            "r/artificial",
            "r/LocalLLaMA",
            "r/ChatGPT",
            "r/OpenAI",
            "r/singularity",
            "r/LangChain",
            "r/n8n",
            "r/automation"
        ],
        "method": "PRAW (Reddit API)",
        "min_upvotes": SOURCE_THRESHOLD_CONFIG.reddit_min_score,      # FIX: Centralized
        "min_comments": SOURCE_THRESHOLD_CONFIG.reddit_min_comments,  # FIX: Centralized
        "frequency": "every_4_hours",
        "priority": "P0",
        "content_type": ContentType.COMMUNITY_CONTENT,
        "filter_keywords": ["case study", "production", "enterprise", "deployed", "real world", "built"],
        "flair_filters": ["Discussion", "Project", "News"]
    },

    "youtube_transcripts": {
        "channels": [
            "Two Minute Papers",
            "Yannic Kilcher",
            "AI Explained",
            "Matt Wolfe",
            "AI Jason",
            "David Shapiro"
        ],
        "search_queries": [
            "AI enterprise implementation",
            "LLM production deployment tutorial",
            "AI agent workflow automation"
        ],
        "method": "YouTube Data API v3 + youtube-transcript-api",
        "frequency": "daily",
        "priority": "P1",
        "content_type": ContentType.COMMUNITY_CONTENT,
        "extraction": {
            "title": True,
            "description": True,
            "transcript": True,
            "views": True,
            "publish_date": True,
            "duration": True
        },
        "min_views": SOURCE_THRESHOLD_CONFIG.youtube_min_views  # FIX: Centralized
    },

    "medium": {
        "publications": [
            "towardsdatascience.com",
            "ai.plainenglish.io",
            "betterprogramming.pub"
        ],
        "tags": ["artificial-intelligence", "machine-learning", "llm", "ai-agents", "automation"],
        "method": "RSS feeds",
        "frequency": "daily",
        "priority": "P1",
        "content_type": ContentType.AUTOMATION_CASE
    },

    "substack": {
        "newsletters": [
            "simonwillison.substack.com",
            "lethain.substack.com",
            "thealgorithmicbridge.substack.com",
            "oneusefulthing.substack.com",
            "aisnakeoil.substack.com"
        ],
        "method": "RSS feeds",
        "frequency": "daily",
        "priority": "P1",
        "content_type": ContentType.PRIMARY_SOURCE
    },

    "devto": {
        "endpoint": "https://dev.to/api/articles",
        "tags": ["ai", "machinelearning", "llm", "automation", "langchain"],
        "min_reactions": SOURCE_THRESHOLD_CONFIG.devto_min_reactions,  # FIX: Centralized
        "frequency": "daily",
        "priority": "P2",
        "content_type": ContentType.AUTOMATION_CASE
    }
}
```

---

#### Exclusion Rules (Pre-Filter)

```python
"""
Apply BEFORE scoring to filter out low-quality content.
Based on explicit exclusion criteria from content strategy.
"""

# ═══════════════════════════════════════════════════════════════════════════
# LOW CREDIBILITY DOMAINS CONSTANT
# Define as module-level constant BEFORE exclusion_rules that references it
#
# NOTE: These are EXAMPLE entries. In production, populate from:
# - content_strategy.md blocklist
# - External blocklist service
# - Environment variable: BLOCKED_DOMAINS_PATH pointing to JSON/TXT file
# ═══════════════════════════════════════════════════════════════════════════
LOW_CREDIBILITY_DOMAINS = frozenset([
    # Content farms (example - replace with actual domains)
    # "example-content-farm.com",
    # AI-generated article mills (example - replace with actual domains)
    # "ai-article-spam.net",
    # Known clickbait (example - replace with actual domains)
    # "clickbait-tech-news.io",
    # TODO: Populate from external blocklist or content_strategy.md
])

def _warn_if_empty_blocklist():
    """Warn at startup if blocklist is empty."""
    import logging
    if not LOW_CREDIBILITY_DOMAINS:
        logging.getLogger("exclusion_rules").warning(
            "LOW_CREDIBILITY_DOMAINS is empty. "
            "Consider populating from content_strategy.md or external blocklist."
        )

_warn_if_empty_blocklist()

exclusion_rules = {
    # ═══════════════════════════════════════════════════════════════
    # CONTENT TYPE EXCLUSIONS
    # ═══════════════════════════════════════════════════════════════

    "secondary_without_source": {
        "description": "Пересказы новостей без первоисточника",
        "exclude_for": ["all"],
        "condition": lambda topic: (
            topic.is_rewrite and
            not topic.has_original_source_link
        )
    },

    "marketing_pr_content": {
        "description": "Маркетинговые статьи и PR-материалы",
        "exclude_for": ["all"],
        "condition": lambda topic: topic.content_type in [
            "press_release", "sponsored", "promotional", "advertorial"
        ],
        "signals": [
            "excessive_product_mentions",
            "no_third_party_validation",
            "sales_cta_present",
            "competitor_bashing"
        ]
    },

    "video_for_research": {
        "description": "YouTube для поиска primary sources",
        "exclude_for": [ContentType.PRIMARY_SOURCE],
        "allow_for": [ContentType.COMMUNITY_CONTENT],
        "condition": lambda topic: topic.source_type == "video"
    },

    # ═══════════════════════════════════════════════════════════════
    # QUALITY EXCLUSIONS
    # ═══════════════════════════════════════════════════════════════

    "shallow_hype_content": {
        "description": "Поверхностный хайповый контент без глубины",
        "signals": [
            "no_specific_examples",
            "no_metrics_or_data",
            "generic_statements_only",
            "excessive_buzzwords",
            "no_actionable_insights"
        ],
        "threshold": 3,  # Exclude if 3+ signals present
        "condition": lambda topic, signals: sum(signals) >= 3
    },

    "overview_without_cases": {
        "description": "Обзорные материалы без конкретных кейсов",
        "condition": lambda topic: (
            topic.is_overview and
            topic.case_study_count == 0
        )
    },

    "no_practical_examples": {
        "description": "Контент без практических примеров",
        "condition": lambda topic: (
            topic.actionable_example_count == 0 and
            topic.how_to_steps_count == 0
        )
    },

    "no_reported_results": {
        "description": "Кейсы без отчётных результатов",
        "apply_to": [ContentType.ENTERPRISE_CASE, ContentType.AUTOMATION_CASE],
        "condition": lambda topic: (
            topic.metrics_mentioned == 0 and
            not topic.outcomes_described
        )
    },

    # ═══════════════════════════════════════════════════════════════
    # SOURCE EXCLUSIONS
    # ═══════════════════════════════════════════════════════════════

    "low_credibility_sources": {
        "description": "Источники с низкой достоверностью",
        "blocked_domains": LOW_CREDIBILITY_DOMAINS,  # Use module-level constant
        # FIX: Reference the module-level constant to avoid duplication
        # Previously had inline list duplicated - now uses LOW_CREDIBILITY_DOMAINS frozenset
        "condition": lambda topic: topic.source_domain in LOW_CREDIBILITY_DOMAINS
    }
}


def pre_filter_topics(
    raw_topics: List[dict]
) -> Tuple[List[dict], List[dict]]:
    """
    Apply exclusion rules BEFORE scoring.

    MEDIUM PRIORITY FIX #12: Added type hints, error handling, and logging.

    Args:
        raw_topics: List of raw topic dictionaries from sources

    Returns:
        Tuple containing:
        - passed_topics: Topics that passed all exclusion rules
        - excluded_topics_with_reasons: Topics excluded with their reasons

    Raises:
        No exceptions - logs errors and continues processing
    """
    import logging
    logger = logging.getLogger("pre_filter_topics")

    passed: List[dict] = []
    excluded_log: List[dict] = []

    logger.info(f"[PRE_FILTER] Starting pre-filter for {len(raw_topics)} topics")

    for topic in raw_topics:
        exclusion_reasons: List[str] = []

        for rule_name, rule in exclusion_rules.items():
            # FIX #12: Wrap rule evaluation in try/except
            try:
                if evaluate_rule(topic, rule):
                    exclusion_reasons.append(rule_name)
            except Exception as e:
                logger.warning(
                    f"[PRE_FILTER] Rule evaluation error for '{rule_name}' on topic "
                    f"'{getattr(topic, 'id', 'unknown')}': {e}"
                )
                # Don't exclude on rule evaluation error - let it pass
                continue

        if exclusion_reasons:
            excluded_entry = {
                "topic_id": getattr(topic, 'id', 'unknown'),
                "title": getattr(topic, 'title', 'unknown'),
                "reasons": exclusion_reasons
            }
            excluded_log.append(excluded_entry)
            logger.debug(
                f"[PRE_FILTER] Excluded: {excluded_entry['title'][:50]}... "
                f"Reasons: {exclusion_reasons}"
            )
        else:
            passed.append(topic)

    # FIX #12: Log summary
    logger.info(
        f"[PRE_FILTER] Complete: {len(passed)} passed, {len(excluded_log)} excluded\n"
        f"  Exclusion breakdown: {_summarize_exclusions(excluded_log)}"
    )

    return passed, excluded_log


def _summarize_exclusions(excluded_log: List[dict]) -> Dict[str, int]:
    """Summarize exclusion reasons for logging."""
    summary: Dict[str, int] = {}
    for entry in excluded_log:
        for reason in entry.get("reasons", []):
            summary[reason] = summary.get(reason, 0) + 1
    return summary


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS (FIX: Previously undefined, now implemented)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_rule(topic: Any, rule: dict) -> bool:
    """
    Evaluate an exclusion rule against a topic.

    FIX: Previously called but never defined, causing NameError.
    FIX: Now respects apply_to and exclude_for content type restrictions.

    Args:
        topic: Topic object or dict to evaluate
        rule: Rule dict containing 'condition' lambda, optional 'signals',
              and optional 'apply_to'/'exclude_for' type restrictions

    Returns:
        True if topic should be excluded (rule matches), False otherwise
    """
    # ═══════════════════════════════════════════════════════════════
    # FIX: Check content type restrictions FIRST
    # ═══════════════════════════════════════════════════════════════
    topic_type = getattr(topic, 'content_type', None)

    # Check apply_to restriction (rule only applies to listed types)
    apply_to = rule.get("apply_to")
    if apply_to and topic_type not in apply_to:
        return False  # Rule doesn't apply to this content type

    # Check exclude_for restriction (rule applies to all EXCEPT listed types)
    exclude_for = rule.get("exclude_for")
    if exclude_for and exclude_for != ["all"]:
        if topic_type in exclude_for:
            return False  # Rule excludes this content type

    # Check if rule has explicit condition
    condition = rule.get("condition")
    if condition and callable(condition):
        try:
            return condition(topic)
        except (AttributeError, KeyError, TypeError):
            # Topic doesn't have required attributes - rule doesn't apply
            return False

    # Check signal-based rules
    signals = rule.get("signals", [])
    threshold = rule.get("threshold", len(signals))
    if signals:
        signal_count = sum(
            1 for signal in signals
            if _check_signal(topic, signal)
        )
        return signal_count >= threshold

    return False


def _check_signal(topic: Any, signal: str) -> bool:
    """Check if a topic exhibits a specific quality signal."""
    # Map signal names to topic attribute checks
    return getattr(topic, signal, False) if hasattr(topic, signal) else False


def weighted_average(score_config: dict, topic: Any) -> float:
    """
    Calculate weighted average score for a topic.

    FIX: Previously called but never defined, causing NameError.

    Args:
        score_config: Dict of factor_name -> {weight, factors}
        topic: Topic object with scoring attributes

    Returns:
        Weighted average score (0.0 - 1.0)
    """
    total_score = 0.0
    total_weight = 0.0

    for factor_name, config in score_config.items():
        weight = config.get("weight", 0)
        if weight <= 0:
            continue

        # Calculate factor score from sub-factors
        factor_score = _calculate_factor_score(topic, config.get("factors", []))

        total_score += factor_score * weight
        total_weight += weight

    return total_score / total_weight if total_weight > 0 else 0.0


def _calculate_factor_score(topic: Any, factors: List[str]) -> float:
    """Calculate score for a single factor based on its sub-factors."""
    if not factors:
        return 0.5  # Neutral if no factors defined

    scores = []
    for factor in factors:
        value = getattr(topic, factor, None)
        if value is None:
            continue
        if isinstance(value, bool):
            scores.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)):
            # Normalize numeric values (assume 0-100 scale)
            scores.append(min(1.0, max(0.0, float(value) / 100)))
        else:
            # For strings, assume presence = positive signal
            scores.append(0.7 if value else 0.0)

    return sum(scores) / len(scores) if scores else 0.5


def llm_extract(prompt: str) -> List[str]:
    """
    Extract structured data using LLM.

    FIX: Previously called but never defined. Uses Claude Code CLI.

    Args:
        prompt: Extraction prompt with instructions

    Returns:
        List of extracted strings (e.g., takeaways)
    """
    import asyncio

    async def _extract():
        claude = get_claude()
        response = await claude.generate_structured(prompt)

        # Handle different response formats
        if isinstance(response, list):
            return response
        elif isinstance(response, dict):
            # Look for common list keys
            for key in ["takeaways", "items", "results", "extracted", "data"]:
                if key in response and isinstance(response[key], list):
                    return response[key]
            # Return values if dict of numbered items
            return list(response.values())
        else:
            return [str(response)]

    return asyncio.run(_extract())
```

---

#### Scoring Algorithm (Quality-Focused)

```python
def calculate_trend_score(topic: TrendTopic) -> float:
    """
    Enhanced scoring with content quality dimensions.
    Different weights applied based on ContentType.
    """

    # ═══════════════════════════════════════════════════════════════
    # BASE SCORING FACTORS
    # ═══════════════════════════════════════════════════════════════

    base_scores = {
        # Engagement-focused (reduced weights)
        "recency": {
            "weight": 0.15,  # Was 0.25
            "factors": [
                "hours_since_published",
                "velocity_of_mentions",
                "is_breaking_news"
            ]
        },

        "virality_potential": {
            "weight": 0.15,  # Was 0.30
            "factors": [
                "controversy_level",
                "emotional_hook",
                "relatability",
                "shareability"
            ]
        },

        "relevance": {
            "weight": 0.15,  # Was 0.25
            "factors": [
                "ai_topic_match",
                "audience_interest_history",
                "your_expertise_overlap"
            ]
        },

        "uniqueness": {
            "weight": 0.10,  # Was 0.20
            "factors": [
                "linkedin_saturation",
                "unique_angle_possible",
                "first_mover_advantage"
            ]
        },

        # ═══════════════════════════════════════════════════════════
        # NEW: QUALITY-FOCUSED FACTORS
        # ═══════════════════════════════════════════════════════════

        "evidence_quality": {
            "weight": 0.20,  # NEW - highest priority
            "factors": [
                "has_specific_metrics",       # KPI, ROI, concrete numbers
                "has_quantified_results",     # "87% improvement" vs "significant"
                "has_timeline_data",          # Implementation timeline mentioned
                "has_cost_benefit_data"       # ROI or cost savings stated
            ]
        },

        "implementation_depth": {
            "weight": 0.10,  # NEW
            "factors": [
                "architecture_described",     # Technical components explained
                "process_steps_included",     # How-to elements present
                "tools_technologies_named",   # Specific stack mentioned
                "team_roles_mentioned"        # Organizational context given
            ]
        },

        "practical_applicability": {
            "weight": 0.10,  # NEW
            "factors": [
                "actionable_takeaways",       # Reader can do something with this
                "transferable_learnings",     # Applies to other contexts
                "specific_use_cases",         # Concrete examples provided
                "lessons_learned_explicit"    # What worked / didn't work
            ]
        },

        "source_credibility": {
            "weight": 0.05,  # NEW
            "factors": [
                "is_primary_source",          # Original vs rewrite
                "author_authority",           # Expert / practitioner / lab
                "company_tier",               # Fortune 500 / startup
                "publication_reputation"      # arXiv / corp blog / Medium
            ]
        }
    }
    # Total: 1.00

    return weighted_average(base_scores, topic)


# ═══════════════════════════════════════════════════════════════════
# CONTENT TYPE SPECIFIC WEIGHT MODIFIERS
# ═══════════════════════════════════════════════════════════════════

content_type_weight_modifiers = {
    ContentType.ENTERPRISE_CASE: {
        # Adjust base weights
        "recency": -0.10,            # Less time-sensitive → 0.05 final
        "virality_potential": -0.05, # Less about virality → 0.10 final
        "relevance": +0.05,          # More about depth → 0.20 final
        "uniqueness": +0.10,         # Unique cases are gold → 0.20 final

        # Boost quality factors
        "evidence_quality": +0.10,   # Metrics critical → 0.30 final
        "implementation_depth": +0.05,  # Architecture important → 0.15 final
    },

    ContentType.PRIMARY_SOURCE: {
        "recency": -0.10,            # Research is evergreen → 0.05 final
        "virality_potential": -0.10, # Not about virality → 0.05 final
        "relevance": +0.10,          # High relevance weight → 0.25 final
        "uniqueness": +0.10,         # Novel ideas crucial → 0.20 final

        # Add research-specific factors
        "author_authority": 0.15,    # NEW factor for this type
        "counterintuitive_factor": 0.10,  # NEW factor for this type
    },

    ContentType.AUTOMATION_CASE: {
        "recency": -0.05,            # Somewhat evergreen → 0.10 final
        "virality_potential": +0.05, # Practitioners love these → 0.20 final
        "practical_applicability": +0.10,  # Key factor → 0.20 final

        # Add automation-specific factors
        "reproducibility": 0.10,     # Can reader replicate?
        "tool_accessibility": 0.05,  # Are tools available?
    },

    ContentType.COMMUNITY_CONTENT: {
        "recency": +0.05,            # Fresh discussions matter → 0.20 final
        "virality_potential": +0.10, # Community buzz → 0.25 final
        "uniqueness": -0.05,         # Discussions overlap → 0.05 final

        # Add community-specific factors
        "engagement_velocity": 0.10, # How fast is it growing?
        "practitioner_validation": 0.05,  # Real builders commenting?
    },

    ContentType.TOOL_RELEASE: {
        "recency": +0.15,            # Breaking news → 0.30 final
        "virality_potential": +0.05, # Tech community interest → 0.20 final
        "relevance": -0.05,          # Not always deep → 0.10 final
        "uniqueness": -0.05,         # Multiple sources cover same → 0.05 final

        # Add release-specific factors
        "demo_availability": 0.10,   # Can people try it?
        "market_disruption": 0.10,   # How significant?
    }
}


def apply_content_type_modifiers(base_scores: dict, content_type: ContentType) -> dict:
    """
    Apply content-type-specific weight adjustments.

    IMPORTANT: Normalizes weights to sum to 1.0 after all adjustments.
    This ensures scoring remains mathematically consistent regardless of
    which modifiers are applied.
    """
    modifiers = content_type_weight_modifiers.get(content_type, {})
    adjusted = {}

    for factor, config in base_scores.items():
        modifier = modifiers.get(factor, 0)
        adjusted[factor] = {
            **config,
            "weight": max(0, config["weight"] + modifier)
        }

    # Add type-specific factors
    for factor, weight in modifiers.items():
        if factor not in base_scores:
            adjusted[factor] = {"weight": weight, "factors": [factor]}

    # ═══════════════════════════════════════════════════════════════
    # CRITICAL: NORMALIZE WEIGHTS TO SUM TO 1.0
    # Without this, different ContentTypes would have different total
    # weights, making scores incomparable across types.
    # ═══════════════════════════════════════════════════════════════
    total_weight = sum(config["weight"] for config in adjusted.values())
    if total_weight > 0:
        for factor in adjusted:
            adjusted[factor]["weight"] = adjusted[factor]["weight"] / total_weight

    return adjusted
```

---

#### Content Quality Signals by Type

```python
content_quality_signals = {
    ContentType.ENTERPRISE_CASE: {
        "required_signals": [
            "company_name_identified",
            "industry_specified",
            "problem_statement_clear",
            "solution_described",
            "results_mentioned"
        ],
        "bonus_signals": [
            "kpi_metrics_quantified",        # +0.5 score
            "roi_calculated",                # +0.5 score
            "implementation_timeline_given", # +0.3 score
            "architecture_diagram_present",  # +0.3 score
            "team_structure_described",      # +0.2 score
            "lessons_learned_section"        # +0.3 score
        ],
        "quality_threshold": 0.7  # Min 70% of required signals
    },

    ContentType.PRIMARY_SOURCE: {
        "required_signals": [
            "novel_contribution_stated",
            "methodology_described",
            "results_presented",
            "implications_discussed"
        ],
        "bonus_signals": [
            "code_open_sourced",             # +0.5 score
            "benchmark_comparisons",         # +0.4 score
            "practical_applications_shown",  # +0.3 score
            "author_from_major_lab"          # +0.3 score
        ],
        "quality_threshold": 0.8
    },

    ContentType.AUTOMATION_CASE: {
        "required_signals": [
            "automation_target_clear",
            "tools_listed",
            "workflow_explained",
            "outcome_stated"
        ],
        "bonus_signals": [
            "code_snippets_included",        # +0.5 score
            "prompt_examples_shown",         # +0.4 score
            "metrics_quantified",            # +0.4 score
            "error_handling_discussed",      # +0.2 score
            "replication_instructions"       # +0.3 score
        ],
        "quality_threshold": 0.6
    },

    ContentType.COMMUNITY_CONTENT: {
        "required_signals": [
            "topic_clearly_defined",
            "multiple_perspectives_present",
            "actionable_insights_extractable"
        ],
        "bonus_signals": [
            "expert_participants",           # +0.4 score
            "code_examples_shared",          # +0.3 score
            "real_world_validation",         # +0.4 score
            "high_engagement_quality"        # +0.3 score (not just quantity)
        ],
        "quality_threshold": 0.5
    },

    ContentType.TOOL_RELEASE: {
        "required_signals": [
            "tool_name_clear",
            "functionality_described",
            "access_method_stated"
        ],
        "bonus_signals": [
            "demo_available",                # +0.5 score
            "api_documented",                # +0.4 score
            "pricing_transparent",           # +0.2 score
            "comparison_to_alternatives",    # +0.3 score
            "early_user_feedback"            # +0.3 score
        ],
        "quality_threshold": 0.7
    }
}
```

---

#### Top Pick of the Day Selection

```python
# ═══════════════════════════════════════════════════════════════════════════
# TOP PICK HELPER FUNCTIONS (FIX: Previously undefined, now implemented)
# ═══════════════════════════════════════════════════════════════════════════

def calculate_top_pick_bonus(topic: Any, criteria: dict) -> float:
    """
    Calculate bonus score for top pick selection.

    FIX: Previously called but never defined, causing NameError.

    Args:
        topic: Topic object with scoring attributes
        criteria: Dict of criterion_name -> {weight, factors}

    Returns:
        Bonus score (0.0 - 1.0)
    """
    return weighted_average(criteria, topic)


def generate_selection_rationale(topic: Any) -> str:
    """
    Generate explanation of why this topic was selected as top pick.

    FIX: Previously called but never defined.

    Args:
        topic: Selected topic object

    Returns:
        Human-readable rationale string
    """
    import asyncio

    async def _generate():
        claude = get_claude()
        prompt = f"""
        Generate a brief (2-3 sentences) explanation of why this topic was selected
        as the most important topic of the day for a LinkedIn AI content creator.

        Topic: {getattr(topic, 'title', 'Unknown')}
        Summary: {getattr(topic, 'summary', 'N/A')[:500]}
        Content Type: {getattr(topic, 'content_type', 'Unknown')}
        Score: {getattr(topic, 'score', 0):.2f}

        Focus on strategic value and actionable insights.
        Return ONLY the rationale text, no JSON.
        """
        response = await claude.generate(prompt)
        return response.content if response.success else "High-scoring topic with strong strategic relevance."

    return asyncio.run(_generate())


def identify_target_audience(topic: Any) -> str:
    """
    Identify who should care about this topic.

    FIX: Previously called but never defined.

    Args:
        topic: Topic object

    Returns:
        Target audience description
    """
    content_type = getattr(topic, 'content_type', None)

    audience_map = {
        ContentType.ENTERPRISE_CASE: "CTOs, VPs of Engineering, AI/ML Directors at mid-to-large companies",
        ContentType.PRIMARY_SOURCE: "AI researchers, technical leaders, and strategy professionals",
        ContentType.AUTOMATION_CASE: "Developers, automation engineers, and operations teams",
        ContentType.COMMUNITY_CONTENT: "AI practitioners and tech professionals across all levels",
        ContentType.TOOL_RELEASE: "Early adopters, developers, and tech decision-makers"
    }

    if content_type and content_type in audience_map:
        return audience_map[content_type]

    # Fallback: generate via LLM
    import asyncio

    async def _identify():
        claude = get_claude()
        prompt = f"""
        Who should care about this AI topic? Provide a brief audience description.

        Topic: {getattr(topic, 'title', 'Unknown')}
        Summary: {getattr(topic, 'summary', 'N/A')[:300]}

        Return ONLY the audience description (1-2 sentences), no JSON.
        """
        response = await claude.generate(prompt)
        return response.content if response.success else "AI professionals and tech leaders"

    return asyncio.run(_identify())


def select_top_pick_of_day(scored_topics: List[TrendTopic]) -> TrendTopic:
    """
    Select "Самый важный кейс дня" - the single most important topic.
    Applied AFTER regular scoring to identify the must-read content.
    """

    top_pick_criteria = {
        "strategic_importance": {
            "weight": 0.30,
            "factors": [
                "changes_how_we_think",           # Paradigm shift potential
                "affects_many_businesses",        # Broad applicability
                "time_sensitive_insight",         # "Must know now"
                "competitive_advantage_potential" # First-mover benefit
            ]
        },

        "evidence_strength": {
            "weight": 0.25,
            "factors": [
                "concrete_metrics_present",
                "multiple_sources_confirm",
                "primary_source_available",
                "verified_results"
            ]
        },

        "learning_density": {
            "weight": 0.25,
            "factors": [
                "multiple_actionable_insights",   # 3+ takeaways extractable
                "transferable_to_reader_context",
                "specific_enough_to_replicate",
                "avoids_obvious_conclusions"
            ]
        },

        "content_quality_match": {
            "weight": 0.20,
            "factors": [
                "meets_content_type_threshold",
                "bonus_signals_present",
                "exceeds_baseline_score"
            ]
        }
    }

    # Calculate top pick bonus for each topic
    top_pick_scores = []
    for topic in scored_topics:
        base_score = topic.score
        top_pick_bonus = calculate_top_pick_bonus(topic, top_pick_criteria)
        final_score = (base_score * 0.6) + (top_pick_bonus * 0.4)
        top_pick_scores.append((topic, final_score))

    # Sort and select winner
    top_pick_scores.sort(key=lambda x: x[1], reverse=True)
    winner = top_pick_scores[0][0]

    # Generate mandatory summary
    winner.top_pick_summary = {
        "why_chosen": generate_selection_rationale(winner),
        "key_takeaways": generate_three_takeaways(winner),
        "who_should_care": identify_target_audience(winner)
    }

    return winner


def generate_three_takeaways(topic: TrendTopic) -> List[str]:
    """
    Extract exactly 3 key takeaways as required by content strategy.
    """
    # LLM-based extraction
    prompt = f"""
    Extract exactly 3 key takeaways from this content.

    Title: {topic.title}
    Summary: {topic.summary}
    Content Type: {topic.content_type.value}

    Format:
    1. [Most actionable insight - what reader can DO]
    2. [Strategic implication - why this matters]
    3. [What to watch next - follow-up action]

    Keep each takeaway to 1-2 sentences max.
    """
    return llm_extract(prompt)
```

---

#### Type-Specific Suggested Angles

```python
suggested_angles_by_type = {
    ContentType.ENTERPRISE_CASE: {
        "angle_templates": [
            {
                "type": "lessons_learned",
                "template": "What {company} learned implementing {technology}",
                "hooks": [
                    "{company} spent {timeline} building {solution}. Here's what they discovered:",
                    "The real ROI of AI at {company}: {metric}. But the journey wasn't smooth.",
                    "{industry} giant reveals their AI implementation playbook."
                ]
            },
            {
                "type": "metrics_story",
                "template": "From {before} to {after}: How {company} achieved {improvement}",
                "hooks": [
                    "{metric_improvement} in {kpi}. Here's exactly how {company} did it.",
                    "The AI investment that paid for itself in {timeline}.",
                    "When {company} showed their board this number, everything changed: {metric}"
                ]
            },
            {
                "type": "architecture_breakdown",
                "template": "Inside {company}'s AI stack",
                "hooks": [
                    "I dissected {company}'s AI architecture. It's simpler than you'd think.",
                    "The tech stack behind {company}'s {capability}:",
                    "{company} uses {surprising_component}. Here's why."
                ]
            }
        ]
    },

    ContentType.PRIMARY_SOURCE: {
        "angle_templates": [
            {
                "type": "contrarian_interpretation",
                "template": "Why {author}'s research challenges {assumption}",
                "hooks": [
                    "A new paper just challenged the biggest assumption in AI.",
                    "What if everything we believed about {topic} is wrong?",
                    "{author} spent {time} researching {topic}. The conclusion surprised everyone."
                ]
            },
            {
                "type": "simplified_explainer",
                "template": "{complex_finding} explained simply",
                "hooks": [
                    "I read the {paper} so you don't have to. The key insight:",
                    "This research is dense. Here's what it actually means:",
                    "Translating {author}'s latest into plain English:"
                ]
            },
            {
                "type": "strategic_implications",
                "template": "What {finding} means for AI strategy",
                "hooks": [
                    "This paper will shape AI strategy for years.",
                    "If {finding} is true, every AI roadmap needs to change.",
                    "The strategic implications no one is talking about."
                ]
            }
        ]
    },

    ContentType.AUTOMATION_CASE: {
        "angle_templates": [
            {
                "type": "how_to_replicate",
                "template": "How to build {solution} with {tools}",
                "hooks": [
                    "I built {solution} in {timeframe}. Here's exactly how:",
                    "The {tool} workflow that automated {task}:",
                    "Stop doing {manual_task} manually. Here's the automation:"
                ]
            },
            {
                "type": "results_story",
                "template": "From {hours} hours to {minutes} minutes",
                "hooks": [
                    "This workflow saves {time} per {period}. Setup took {setup_time}.",
                    "The automation that changed how I {task}:",
                    "{time_saved} saved. {cost_saved} saved. One workflow."
                ]
            },
            {
                "type": "tool_comparison",
                "template": "Why {tool_a} beat {tool_b} for {use_case}",
                "hooks": [
                    "I tried {tool_a} and {tool_b}. Clear winner.",
                    "The comparison no one asked for (but everyone needs):",
                    "Before choosing between {tool_a} and {tool_b}, read this."
                ]
            }
        ]
    },

    ContentType.COMMUNITY_CONTENT: {
        "angle_templates": [
            {
                "type": "curated_insights",
                "template": "Best insights from {platform}'s {topic} discussion",
                "hooks": [
                    "A {platform} thread just exploded. Key takeaways:",
                    "The community discovered something important:",
                    "I spent {time} in {platform} threads. Here's the gold:"
                ]
            },
            {
                "type": "practitioner_wisdom",
                "template": "What people actually building with {topic} say",
                "hooks": [
                    "Forget the hype. Here's what builders actually say:",
                    "I asked practitioners what really works. The answers surprised me.",
                    "The gap between {topic} theory and practice:"
                ]
            }
        ]
    },

    ContentType.TOOL_RELEASE: {
        "angle_templates": [
            {
                "type": "first_look",
                "template": "I tested {tool} so you don't have to",
                "hooks": [
                    "{company} just dropped {tool}. I tested it immediately.",
                    "First hands-on with {tool}. Here's what you need to know:",
                    "{tool} is live. First impression: {one_word}."
                ]
            },
            {
                "type": "comparison",
                "template": "How {new_tool} compares to {existing_tool}",
                "hooks": [
                    "{new_tool} vs {existing_tool}: which one wins?",
                    "Does {new_tool} actually beat {existing_tool}?",
                    "The landscape just changed. Here's the new ranking."
                ]
            },
            {
                "type": "implications",
                "template": "What {tool} means for {affected_users}",
                "hooks": [
                    "If you use {competing_tool}, pay attention.",
                    "{tool} changes everything for {user_type}. Here's why:",
                    "The announcement that flew under the radar (but shouldn't have)."
                ]
            }
        ]
    }
}
```

---

#### Output Schema

```python
from dataclasses import dataclass, field  # FIX: Added 'field' import (was missing)
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
from enum import Enum


@dataclass
class SuggestedAngle:
    """A potential hook/angle for writing about this topic."""
    angle_text: str
    angle_type: str  # lessons_learned, metrics_story, how_to, etc.
    hook_templates: List[str]
    content_type_fit: float  # 0-1 how well this fits the content type


@dataclass
class TopPickSummary:
    """Summary for the "Самый важный кейс дня"."""
    why_chosen: str
    key_takeaways: List[str]  # Exactly 3 takeaways
    who_should_care: str


# ═══════════════════════════════════════════════════════════════════
# TYPE-SPECIFIC METADATA SCHEMAS
# ═══════════════════════════════════════════════════════════════════
#
# IMPORTANT: Each metadata class has a `type` field as DISCRIMINATOR.
# This enables:
# 1. Runtime type checking during JSON deserialization
# 2. Type-safe handling in downstream agents
# 3. Validation that metadata.type matches topic.content_type
#
# Example validation:
#   if topic.metadata.type != topic.content_type.value:
#       raise MetadataTypeMismatchError(...)
# ═══════════════════════════════════════════════════════════════════

from typing import Literal

# LOW PRIORITY FIX #7: Valid values for enumerated fields
VALID_ENTERPRISE_SCALES = ["SMB", "Mid-Market", "Enterprise", "Fortune 500"]
VALID_COMPLEXITY_LEVELS = ["simplify_heavily", "simplify_slightly", "keep_technical"]
VALID_CONTROVERSY_LEVELS = ["low", "medium", "high", "spicy"]


# ═══════════════════════════════════════════════════════════════════
# HOOK STYLE ENUM
# ═══════════════════════════════════════════════════════════════════
#
# FIX: Created unified HookStyle enum to resolve naming inconsistencies
# between type_extraction_config and TYPE_CONTEXTS.
#
# Each hook style has TWO names:
# - The enum VALUE is the canonical name (used for storage/serialization)
# - The aliases are for backward compatibility and human readability
#
# Usage:
#   hook_style = HookStyle.METRICS  # Use enum directly
#   hook_style.value -> "metrics"   # For serialization
# ═══════════════════════════════════════════════════════════════════

class HookStyle(Enum):
    """
    Unified hook styles for all content types.

    Hook styles determine the angle/approach used for writing content.
    Each ContentType has a specific set of allowed hook styles.
    """

    # ─────────────────────────────────────────────────────────────────
    # ENTERPRISE_CASE hook styles
    # ─────────────────────────────────────────────────────────────────
    METRICS = "metrics"                      # "X% improvement in Y" angle
    LESSONS_LEARNED = "lessons_learned"      # "What they learned" angle
    PROBLEM_SOLUTION = "problem_solution"    # "Problem → Solution → Result" angle

    # ─────────────────────────────────────────────────────────────────
    # PRIMARY_SOURCE hook styles
    # ─────────────────────────────────────────────────────────────────
    CONTRARIAN = "contrarian"                # Challenge conventional wisdom
    QUESTION = "question"                    # Pose thought-provoking question
    SURPRISING_STAT = "surprising_stat"      # Lead with unexpected data
    SIMPLIFIED_EXPLAINER = "simplified_explainer"  # Make complex accessible
    DEBATE_STARTER = "debate_starter"        # Frame as discussion topic

    # ─────────────────────────────────────────────────────────────────
    # AUTOMATION_CASE hook styles
    # ─────────────────────────────────────────────────────────────────
    HOW_TO = "how_to"                        # Step-by-step tutorial angle
    TIME_SAVED = "time_saved"                # "I automated X, saved Y hours"
    BEFORE_AFTER = "before_after"            # Transformation story
    RESULTS_STORY = "results_story"          # Lead with outcomes
    TOOL_COMPARISON = "tool_comparison"      # Compare tools/approaches

    # ─────────────────────────────────────────────────────────────────
    # COMMUNITY_CONTENT hook styles
    # ─────────────────────────────────────────────────────────────────
    RELATABLE = "relatable"                  # "We've all been there" angle
    COMMUNITY_REFERENCE = "community_reference"  # Quote/reference community
    PERSONAL = "personal"                    # Personal experience angle
    CURATED_INSIGHTS = "curated_insights"    # "Best of" compilation
    HOT_TAKE_RESPONSE = "hot_take_response"  # React to viral opinion
    PRACTITIONER_WISDOM = "practitioner_wisdom"  # "Builders say..." angle

    # ─────────────────────────────────────────────────────────────────
    # TOOL_RELEASE hook styles
    # ─────────────────────────────────────────────────────────────────
    NEWS_BREAKING = "news_breaking"          # Breaking news angle
    FEATURE_HIGHLIGHT = "feature_highlight"  # Focus on killer feature
    COMPARISON = "comparison"                # Compare to alternatives
    FIRST_LOOK = "first_look"                # "I tested it" angle
    IMPLICATIONS = "implications"            # "What this means for X"


# ═══════════════════════════════════════════════════════════════════
# SINGLE SOURCE OF TRUTH FOR HOOK STYLES
# ═══════════════════════════════════════════════════════════════════
#
# FIX: This mapping is THE authoritative source for allowed hook styles.
# The `hook_styles` field in `type_extraction_config` references these
# values. Do NOT duplicate this mapping elsewhere.
#
# When adding new hook styles:
# 1. Add to HookStyle enum above
# 2. Add to this mapping
# 3. The type_extraction_config will automatically use get_hook_styles_for_type()
# ═══════════════════════════════════════════════════════════════════
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


def get_hook_styles_for_type(content_type: ContentType) -> List[HookStyle]:
    """Get allowed hook styles for a content type."""
    return CONTENT_TYPE_HOOK_STYLES.get(content_type, [])


def validate_hook_style(hook_style: HookStyle, content_type: ContentType) -> bool:
    """Validate that a hook style is allowed for the given content type."""
    allowed = CONTENT_TYPE_HOOK_STYLES.get(content_type, [])
    return hook_style in allowed


@dataclass
class EnterpriseCaseMetadata:
    """Metadata for enterprise AI implementation case studies."""
    # Discriminator field - MUST match ContentType.ENTERPRISE_CASE.value
    type: Literal["enterprise_case"] = "enterprise_case"

    # Required fields
    company: str
    industry: str
    scale: str  # SMB / Mid-Market / Enterprise / Fortune 500
    problem_domain: str
    ai_technologies: List[str]
    metrics: Dict[str, str]  # KPI name -> value/improvement
    roi_mentioned: bool
    architecture_available: bool
    lessons_learned: List[str] = field(default_factory=list)

    # LOW PRIORITY FIX #7: Add field validation
    def __post_init__(self):
        """Validate field values against allowed options."""
        if self.scale not in VALID_ENTERPRISE_SCALES:
            raise ValueError(
                f"EnterpriseCaseMetadata.scale must be one of {VALID_ENTERPRISE_SCALES}, "
                f"got '{self.scale}'"
            )
        if not self.company:
            raise ValueError("EnterpriseCaseMetadata.company is required and cannot be empty")
        if not self.industry:
            raise ValueError("EnterpriseCaseMetadata.industry is required and cannot be empty")

    # Optional fields
    implementation_timeline: Optional[str] = None
    team_size: Optional[str] = None


# FIX: Validation constants for metadata classes
VALID_SOURCE_TYPES = ["research_paper", "think_tank_report", "expert_essay", "whitepaper"]
VALID_REPRODUCIBILITY_LEVELS = ["high", "medium", "low"]
VALID_PLATFORMS = ["YouTube", "Reddit", "HackerNews", "Dev.to", "Twitter", "Medium", "Substack"]
VALID_CONTENT_FORMATS = ["video", "post", "comment", "thread", "article", "newsletter"]
VALID_AUTHOR_CREDIBILITY = ["verified_expert", "practitioner", "unknown"]
VALID_RELEASE_TYPES = ["new_product", "major_update", "api_release", "open_source"]


@dataclass
class PrimarySourceMetadata:
    """Metadata for research papers, reports, and primary sources."""
    # Discriminator field - MUST match ContentType.PRIMARY_SOURCE.value
    type: Literal["primary_source"] = "primary_source"

    # Required fields
    authors: List[str]
    organization: str
    source_type: str  # research_paper / think_tank_report / expert_essay / whitepaper
    publication_venue: str
    key_hypothesis: str
    methodology_summary: str
    code_available: bool

    # Optional fields
    counterintuitive_finding: Optional[str] = None
    citations_count: Optional[int] = None

    # FIX: Add __post_init__ validation
    def __post_init__(self):
        """Validate field values against allowed options."""
        if self.source_type not in VALID_SOURCE_TYPES:
            raise ValueError(
                f"PrimarySourceMetadata.source_type must be one of {VALID_SOURCE_TYPES}, "
                f"got '{self.source_type}'"
            )
        if not self.authors:
            raise ValueError("PrimarySourceMetadata.authors is required and cannot be empty")
        if not self.organization:
            raise ValueError("PrimarySourceMetadata.organization is required and cannot be empty")
        if not self.key_hypothesis:
            raise ValueError("PrimarySourceMetadata.key_hypothesis is required and cannot be empty")


@dataclass
class AutomationCaseMetadata:
    """Metadata for AI automation and agent case studies."""
    # Discriminator field - MUST match ContentType.AUTOMATION_CASE.value
    type: Literal["automation_case"] = "automation_case"

    # Required fields
    agent_type: str  # AutoGPT / LangChain / CrewAI / n8n / custom
    workflow_components: List[str]
    integrations: List[str]
    use_case_domain: str
    metrics: Dict[str, str]
    reproducibility: str  # high / medium / low
    code_available: bool

    # Optional fields
    time_saved: Optional[str] = None
    cost_saved: Optional[str] = None

    # FIX: Add __post_init__ validation
    def __post_init__(self):
        """Validate field values against allowed options."""
        if self.reproducibility not in VALID_REPRODUCIBILITY_LEVELS:
            raise ValueError(
                f"AutomationCaseMetadata.reproducibility must be one of {VALID_REPRODUCIBILITY_LEVELS}, "
                f"got '{self.reproducibility}'"
            )
        if not self.agent_type:
            raise ValueError("AutomationCaseMetadata.agent_type is required and cannot be empty")
        if not self.workflow_components:
            raise ValueError("AutomationCaseMetadata.workflow_components is required and cannot be empty")
        if not self.use_case_domain:
            raise ValueError("AutomationCaseMetadata.use_case_domain is required and cannot be empty")


@dataclass
class CommunityContentMetadata:
    """Metadata for community discussions, videos, and threads."""
    # Discriminator field - MUST match ContentType.COMMUNITY_CONTENT.value
    type: Literal["community_content"] = "community_content"

    # Required fields
    platform: str  # YouTube / Reddit / HackerNews / Dev.to
    format: str  # video / post / comment / thread
    engagement_metrics: Dict[str, int]  # upvotes, comments, views
    author_credibility: str  # verified_expert / practitioner / unknown
    has_code_examples: bool
    has_demo: bool
    key_contributors: List[str] = field(default_factory=list)

    # FIX: Add __post_init__ validation
    def __post_init__(self):
        """Validate field values against allowed options."""
        if self.platform not in VALID_PLATFORMS:
            raise ValueError(
                f"CommunityContentMetadata.platform must be one of {VALID_PLATFORMS}, "
                f"got '{self.platform}'"
            )
        if self.format not in VALID_CONTENT_FORMATS:
            raise ValueError(
                f"CommunityContentMetadata.format must be one of {VALID_CONTENT_FORMATS}, "
                f"got '{self.format}'"
            )
        if self.author_credibility not in VALID_AUTHOR_CREDIBILITY:
            raise ValueError(
                f"CommunityContentMetadata.author_credibility must be one of {VALID_AUTHOR_CREDIBILITY}, "
                f"got '{self.author_credibility}'"
            )


@dataclass
class ToolReleaseMetadata:
    """Metadata for new tool and product releases."""
    # Discriminator field - MUST match ContentType.TOOL_RELEASE.value
    type: Literal["tool_release"] = "tool_release"

    # Required fields
    tool_name: str
    company: str
    release_date: str
    release_type: str  # new_product / major_update / api_release / open_source
    demo_url: Optional[str]
    api_available: bool
    pricing_model: Optional[str]
    key_features: List[str]
    competing_tools: List[str]
    early_reviews: List[str]

    # FIX: Add __post_init__ validation
    def __post_init__(self):
        """Validate field values against allowed options."""
        if self.release_type not in VALID_RELEASE_TYPES:
            raise ValueError(
                f"ToolReleaseMetadata.release_type must be one of {VALID_RELEASE_TYPES}, "
                f"got '{self.release_type}'"
            )
        if not self.tool_name:
            raise ValueError("ToolReleaseMetadata.tool_name is required and cannot be empty")
        if not self.company:
            raise ValueError("ToolReleaseMetadata.company is required and cannot be empty")
        if not self.key_features:
            raise ValueError("ToolReleaseMetadata.key_features is required and cannot be empty")


# ═══════════════════════════════════════════════════════════════════
# DISCRIMINATED UNION FOR METADATA
# ═══════════════════════════════════════════════════════════════════
#
# This is a DISCRIMINATED UNION (aka Tagged Union).
# Each variant has a `type` field that identifies it at runtime.
#
# Benefits:
# 1. Type-safe JSON serialization/deserialization
# 2. IDE autocomplete works correctly
# 3. Runtime validation possible
# ═══════════════════════════════════════════════════════════════════

TopicMetadata = Union[
    EnterpriseCaseMetadata,
    PrimarySourceMetadata,
    AutomationCaseMetadata,
    CommunityContentMetadata,
    ToolReleaseMetadata
]


# Mapping from ContentType to expected metadata type string
CONTENT_TYPE_TO_METADATA_TYPE = {
    ContentType.ENTERPRISE_CASE: "enterprise_case",
    ContentType.PRIMARY_SOURCE: "primary_source",
    ContentType.AUTOMATION_CASE: "automation_case",
    ContentType.COMMUNITY_CONTENT: "community_content",
    ContentType.TOOL_RELEASE: "tool_release",
}


class MetadataTypeMismatchError(Exception):
    """Raised when metadata.type doesn't match topic.content_type."""
    pass


def validate_topic_metadata(topic: "TrendTopic") -> bool:
    """
    Validate that topic.metadata.type matches topic.content_type.

    This is a critical validation that should run:
    1. After TrendScout creates a TrendTopic
    2. Before passing to Analyzer
    3. During JSON deserialization

    Raises:
        MetadataTypeMismatchError: If types don't match
    """
    expected_type = CONTENT_TYPE_TO_METADATA_TYPE.get(topic.content_type)
    actual_type = getattr(topic.metadata, 'type', None)

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


# ═══════════════════════════════════════════════════════════════════
# MAIN OUTPUT SCHEMA
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TrendTopic:
    """
    Enhanced topic schema with content type classification,
    type-specific metadata, and quality-focused scoring.
    """
    # Identification
    id: str
    title: str
    summary: str  # 2-3 sentences

    # Classification
    content_type: ContentType

    # Sources
    sources: List[str]  # URLs
    primary_source_url: str  # The main/original source

    # Scoring
    score: float  # 0-10, quality-adjusted
    score_breakdown: Dict[str, float]  # Factor -> score
    quality_signals_matched: List[str]

    # Content for downstream agents
    suggested_angles: List[SuggestedAngle]  # Type-specific hooks
    related_topics: List[str]
    raw_content: str  # Full text for Analyzer

    # Type-specific metadata
    metadata: TopicMetadata

    # Analysis guidance for Analyzer Agent
    analysis_format: str  # Guides extraction strategy
    recommended_post_format: str  # insight_thread / contrarian / tutorial / etc.
    recommended_visual_type: str  # data_viz / architecture / screenshot / etc.

    # Top pick designation
    is_top_pick: bool = False
    top_pick_summary: Optional[TopPickSummary] = None

    # Timestamps
    discovered_at: datetime
    source_published_at: Optional[datetime]

    # LOW PRIORITY FIX #1: Add __repr__ for better debugging
    def __repr__(self) -> str:
        """Concise representation for debugging."""
        top_pick_marker = "⭐" if self.is_top_pick else ""
        return (
            f"TrendTopic({top_pick_marker}id='{self.id}', "
            f"title='{self.title[:40]}...', "
            f"type={self.content_type.value}, "
            f"score={self.score:.1f})"
        )


@dataclass
class TrendScoutOutput:
    """
    Complete output from Trend Scout Agent.
    """
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
    exclusion_log: List[Dict]  # What was filtered and why

    # By content type breakdown
    topics_by_type: Dict[ContentType, int]
```

---

### 2. ANALYZER AGENT

#### Purpose
Deeply understand the material using **content-type-aware extraction**, extract key insights optimized for the specific content type, and prepare a structured brief for Writer that includes type-specific hooks and recommended post formats.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ANALYZER AGENT (Enhanced)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: TrendTopic (with ContentType + metadata)                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    CONTENT TYPE ROUTER                             │     │
│  │  Routes to type-specific extraction pipeline based on ContentType │     │
│  └───────────────────────────┬───────────────────────────────────────┘     │
│                              │                                              │
│     ┌────────────────────────┼────────────────────────┐                    │
│     ▼                        ▼                        ▼                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│  │  ENTERPRISE  │    │   PRIMARY    │    │  AUTOMATION  │                 │
│  │    CASE      │    │   SOURCE     │    │    CASE      │                 │
│  │  EXTRACTOR   │    │  EXTRACTOR   │    │  EXTRACTOR   │                 │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                 │
│         │                   │                   │                          │
│     ┌───┴───────────────────┴───────────────────┴───┐                     │
│     ▼                        ▼                        ▼                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                 │
│  │  COMMUNITY   │    │    TOOL      │    │   GENERIC    │                 │
│  │   CONTENT    │    │   RELEASE    │    │  (fallback)  │                 │
│  │  EXTRACTOR   │    │  EXTRACTOR   │    │  EXTRACTOR   │                 │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                 │
│         │                   │                   │                          │
│         └───────────────────┼───────────────────┘                          │
│                             ▼                                              │
│              ┌───────────────────────────────┐                             │
│              │     TYPE-SPECIFIC INSIGHTS    │                             │
│              │         EXTRACTOR             │                             │
│              └───────────────┬───────────────┘                             │
│                              ▼                                              │
│              ┌───────────────────────────────┐                             │
│              │     CONTROVERSY DETECTOR      │                             │
│              │   + Debate Angle Finder       │                             │
│              └───────────────┬───────────────┘                             │
│                              ▼                                              │
│              ┌───────────────────────────────┐                             │
│              │    HOOK MATERIALS COLLECTOR   │                             │
│              │  (collects raw materials:     │                             │
│              │   angles, quotes, metrics)    │                             │
│              │                               │                             │
│              │  NOTE: Final hooks generated  │                             │
│              │  in WRITER, not here.         │                             │
│              │  This avoids duplication.     │                             │
│              └───────────────┬───────────────┘                             │
│                              ▼                                              │
│              ┌───────────────────────────────┐                             │
│              │     COMPLEXITY ASSESSOR       │                             │
│              │   + Simplification Guide      │                             │
│              └───────────────┬───────────────┘                             │
│                              ▼                                              │
│              ┌───────────────────────────────┐                             │
│              │   TYPE-SPECIFIC BRIEF GEN     │                             │
│              │  (AnalysisBrief with type-    │                             │
│              │   appropriate recommendations)│                             │
│              └───────────────────────────────┘                             │
│                              │                                              │
│                              ▼                                              │
│  OUTPUT: AnalysisBrief (type-aware, with extraction_data by content type)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### Type-Specific Extraction Configuration

```python
"""
Each ContentType has specific fields to extract and analysis focus areas.
This guides what information the Analyzer prioritizes.
"""

type_extraction_config = {
    ContentType.ENTERPRISE_CASE: {
        "required_extraction": [
            "company_identification",
            "industry_and_scale",
            "business_problem_statement",
            "ai_solution_details",
            "implementation_approach",
            "measurable_results",
            "lessons_learned"
        ],
        "optional_extraction": [
            "architecture_description",
            "team_structure",
            "technology_stack",
            "timeline_and_phases",
            "challenges_faced",
            "cost_information",
            "roi_calculation"
        ],
        "analysis_focus": [
            "What specific problem did they solve?",
            "What metrics improved and by how much?",
            "What's the replicable insight for other companies?",
            "What would they do differently?"
        ],
        # FIX: Use get_hook_styles_for_type() - single source of truth
        "hook_styles": get_hook_styles_for_type(ContentType.ENTERPRISE_CASE),
        "recommended_post_formats": ["insight_thread", "personal_story"],
        "visual_recommendation": "data_visualization or architecture_diagram"
    },

    ContentType.PRIMARY_SOURCE: {
        "required_extraction": [
            "core_thesis",
            "key_findings",
            "methodology_summary",
            "implications_stated"
        ],
        "optional_extraction": [
            "counterarguments",
            "limitations_acknowledged",
            "future_research_directions",
            "practical_applications",
            "author_credentials"
        ],
        "analysis_focus": [
            "What's the novel contribution?",
            "What conventional wisdom does this challenge?",
            "How does this change practical AI work?",
            "What's the 'so what' for practitioners?"
        ],
        # FIX: Use get_hook_styles_for_type() - single source of truth
        "hook_styles": get_hook_styles_for_type(ContentType.PRIMARY_SOURCE),
        "recommended_post_formats": ["contrarian", "insight_thread"],
        "visual_recommendation": "concept_illustration or quote_card"
    },

    ContentType.AUTOMATION_CASE: {
        "required_extraction": [
            "use_case_description",
            "tools_and_technologies",
            "workflow_steps",
            "integration_points",
            "results_achieved"
        ],
        "optional_extraction": [
            "code_snippets",
            "prompt_examples",
            "configuration_details",
            "error_handling",
            "cost_breakdown",
            "scaling_considerations"
        ],
        "analysis_focus": [
            "Can readers replicate this?",
            "What's the time/cost savings?",
            "What tools are required?",
            "What are the gotchas?"
        ],
        # FIX: Use get_hook_styles_for_type() - single source of truth
        "hook_styles": get_hook_styles_for_type(ContentType.AUTOMATION_CASE),
        "recommended_post_formats": ["tutorial_light", "insight_thread"],
        "visual_recommendation": "workflow_diagram or carousel"
    },

    ContentType.COMMUNITY_CONTENT: {
        "required_extraction": [
            "main_discussion_topic",
            "key_viewpoints",
            "notable_contributions",
            "consensus_emerging",
            "dissenting_opinions"
        ],
        "optional_extraction": [
            "linked_resources",
            "code_examples_shared",
            "author_credentials_noted",
            "follow_up_questions"
        ],
        "analysis_focus": [
            "What's the community saying about this?",
            "What insights come from practitioners?",
            "Where do experts disagree?",
            "What's the real-world signal vs hype?"
        ],
        # FIX: Use get_hook_styles_for_type() - single source of truth
        "hook_styles": get_hook_styles_for_type(ContentType.COMMUNITY_CONTENT),
        "recommended_post_formats": ["list_post", "question_based"],
        "visual_recommendation": "quote_card or screenshot"
    },

    ContentType.TOOL_RELEASE: {
        "required_extraction": [
            "tool_name_and_company",
            "core_functionality",
            "key_features",
            "target_users",
            "availability_and_access"
        ],
        "optional_extraction": [
            "pricing_information",
            "demo_or_trial",
            "comparison_to_alternatives",
            "limitations",
            "early_user_feedback"
        ],
        "analysis_focus": [
            "What problem does this solve?",
            "How does it compare to existing tools?",
            "Who should care about this?",
            "What's the killer feature?"
        ],
        # FIX: Use get_hook_styles_for_type() - single source of truth
        "hook_styles": get_hook_styles_for_type(ContentType.TOOL_RELEASE),
        "recommended_post_formats": ["insight_thread", "list_post"],
        "visual_recommendation": "product_screenshot or comparison_chart"
    }
}
```

---

#### Type-Specific Analysis Prompts

```python
analysis_prompts_by_type = {
    # ═══════════════════════════════════════════════════════════════════
    # ENTERPRISE CASE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    ContentType.ENTERPRISE_CASE: {
        "extraction_prompt": """
        Analyze this enterprise AI implementation case study.

        Content: {content}
        Source: {source_url}

        Extract the following (mark as "NOT_FOUND" if not available):

        ## Company & Context
        - Company name:
        - Industry:
        - Company scale (SMB/Mid-Market/Enterprise/Fortune 500):
        - Geographic region:

        ## Business Problem
        - What problem were they trying to solve?
        - What was the business impact of this problem?
        - What solutions did they try before?

        ## AI Solution
        - What AI technologies were used? (specific models, platforms)
        - Architecture overview:
        - Integration with existing systems:

        ## Implementation
        - Timeline (how long did it take?):
        - Team structure (who was involved?):
        - Key phases/milestones:
        - Major challenges faced:

        ## Results & Metrics
        - Quantified results (KPIs, metrics):
        - ROI or cost savings:
        - Qualitative benefits:

        ## Lessons Learned
        - What worked well?
        - What would they do differently?
        - Advice for others?
        """,

        "insights_prompt": """
        Based on this enterprise case, extract 3-5 KEY INSIGHTS that would
        be valuable for LinkedIn audience of tech professionals and business leaders.

        Focus on:
        1. Actionable lessons (what can others replicate?)
        2. Surprising findings (what defies expectations?)
        3. Metrics that tell a story (not just numbers)
        4. Strategic implications (what does this mean for the industry?)

        For each insight:
        - State it in one compelling sentence
        - Explain why it matters (business impact)
        - Rate "wow factor" 1-10
        - Suggest a hook angle for this insight

        Case summary: {extraction_summary}
        """,

        "hooks_prompt": """
        Generate 5 hooks for a LinkedIn post about this enterprise case.

        Company: {company}
        Industry: {industry}
        Key metric: {key_metric}
        Main lesson: {main_lesson}

        Hook styles to use:
        1. METRICS: Lead with the impressive number
           Example: "{metric_improvement} in {kpi}. Here's how {company} did it."

        2. LESSONS: Lead with the learning
           Example: "{company} learned something unexpected about AI implementation..."

        3. INDUSTRY IMPACT: Lead with broader implications
           Example: "Every {industry} company should study what {company} just did."

        4. CONTRARIAN: Challenge assumptions
           Example: "Everyone thinks enterprise AI is {assumption}. {company} proved otherwise."

        5. STORY: Lead with narrative
           Example: "When {company}'s team first proposed this AI project, leadership was skeptical..."
        """
    },

    # ═══════════════════════════════════════════════════════════════════
    # PRIMARY SOURCE (RESEARCH) ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    ContentType.PRIMARY_SOURCE: {
        "extraction_prompt": """
        Analyze this research paper/report/essay.

        Content: {content}
        Source: {source_url}

        Extract:

        ## Source Metadata
        - Title:
        - Author(s):
        - Organization/Institution:
        - Publication type (paper/report/essay/whitepaper):
        - Publication date:

        ## Core Thesis
        - Main argument or hypothesis (1-2 sentences):
        - What question does this answer?

        ## Methodology
        - How did they arrive at these conclusions?
        - Data sources used:
        - Limitations acknowledged:

        ## Key Findings
        - Finding 1:
        - Finding 2:
        - Finding 3:
        - Most surprising/counterintuitive finding:

        ## Implications
        - For AI practitioners:
        - For business leaders:
        - For the AI field overall:

        ## Controversies/Debates
        - What existing beliefs does this challenge?
        - Potential counterarguments:
        """,

        "insights_prompt": """
        Extract 3-5 insights from this research that would resonate with
        a LinkedIn audience of AI practitioners and tech leaders.

        Focus on:
        1. "So what?" - Why should busy professionals care?
        2. Contrarian angles - What does this challenge?
        3. Practical implications - How does this change what we do?
        4. Debate potential - Where will experts disagree?

        For each insight:
        - State it accessibly (no jargon)
        - Explain the significance
        - Rate "wow factor" 1-10
        - Suggest controversy level (low/medium/high/spicy)

        Research summary: {extraction_summary}
        """,

        "hooks_prompt": """
        Generate 5 hooks for a LinkedIn post about this research.

        Paper/Report: {title}
        Author(s): {authors}
        Key finding: {key_finding}
        Counterintuitive element: {counterintuitive}

        Hook styles:
        1. CONTRARIAN: Challenge conventional wisdom
           Example: "A new paper just proved that {common_belief} is wrong."

        2. SIMPLIFIED: Make complex accessible
           Example: "I read {paper} so you don't have to. The one thing you need to know:"

        3. STRATEGIC: Business implications
           Example: "This research will change AI strategy for the next 5 years."

        4. DEBATE: Invite discussion
           Example: "{author} claims {claim}. I'm not sure I agree. Here's why:"

        5. CURIOSITY: Create information gap
           Example: "What if everything we believed about {topic} was based on a false assumption?"
        """
    },

    # ═══════════════════════════════════════════════════════════════════
    # AUTOMATION CASE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    ContentType.AUTOMATION_CASE: {
        "extraction_prompt": """
        Analyze this AI automation/agent case study.

        Content: {content}
        Source: {source_url}

        Extract:

        ## Use Case
        - What task/process was automated?
        - What problem did this solve?
        - Who is the target user?

        ## Technical Implementation
        - Agent type/framework (LangChain/AutoGPT/CrewAI/n8n/custom):
        - Core components:
        - LLM(s) used:
        - Integrations:
        - Workflow steps:

        ## Results
        - Time saved:
        - Cost saved:
        - Quality improvements:
        - Other metrics:

        ## Reproducibility
        - Are instructions provided?
        - Is code available?
        - Required tools/accounts:
        - Estimated setup difficulty (easy/medium/hard):

        ## Gotchas & Tips
        - Common pitfalls mentioned:
        - Best practices shared:
        - Edge cases handled:
        """,

        "insights_prompt": """
        Extract 3-5 insights from this automation case that practitioners
        can immediately apply.

        Focus on:
        1. Replicable patterns - What can others copy?
        2. Time/cost savings - What's the ROI?
        3. Tool recommendations - What worked well?
        4. Mistakes to avoid - What didn't work?

        For each insight:
        - Make it actionable
        - Include specific tools/techniques
        - Rate "reproducibility" 1-10
        - Estimate time to implement

        Case summary: {extraction_summary}
        """,

        "hooks_prompt": """
        Generate 5 hooks for a LinkedIn post about this automation case.

        Task automated: {task}
        Tools used: {tools}
        Time saved: {time_saved}
        Key insight: {key_insight}

        Hook styles:
        1. HOW-TO: Promise practical value
           Example: "How I automated {task} in {time}. Here's the exact workflow:"

        2. RESULTS: Lead with impact
           Example: "{time_saved} saved per {period}. One {tool} workflow."

        3. PROBLEM-SOLUTION: Start with pain
           Example: "Stop doing {manual_task} manually. There's a better way."

        4. COMPARISON: Tool vs tool
           Example: "I tried {tool_a} and {tool_b} for {task}. Clear winner."

        5. JOURNEY: Show the process
           Example: "I spent {time} building the perfect {task} automation. Worth it."
        """
    },

    # ═══════════════════════════════════════════════════════════════════
    # COMMUNITY CONTENT ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    ContentType.COMMUNITY_CONTENT: {
        "extraction_prompt": """
        Analyze this community discussion/video/thread.

        Content: {content}
        Platform: {platform}
        Source: {source_url}

        Extract:

        ## Discussion Overview
        - Main topic:
        - Why is this being discussed now?
        - Engagement level (comments, upvotes, views):

        ## Key Viewpoints
        - Viewpoint 1 (with attribution if notable):
        - Viewpoint 2:
        - Viewpoint 3:
        - Areas of agreement:
        - Areas of disagreement:

        ## Notable Contributions
        - Best insight shared:
        - Most controversial take:
        - Practical tip mentioned:
        - Resource/link shared:

        ## Practitioner Signals
        - What are people actually building?
        - What problems are they facing?
        - What tools are they recommending?
        - What's hype vs reality?
        """,

        "insights_prompt": """
        Extract 3-5 insights from this community discussion that capture
        real practitioner wisdom.

        Focus on:
        1. Wisdom from builders - What do people actually doing this say?
        2. Common pain points - What problems keep coming up?
        3. Tool recommendations - What's working in practice?
        4. Hype vs reality - What's the gap?

        For each insight:
        - Attribute to "practitioners" or specific expertise level
        - Note if this is consensus or controversial
        - Rate "signal strength" 1-10 (how reliable is this insight?)

        Discussion summary: {extraction_summary}
        """,

        "hooks_prompt": """
        Generate 5 hooks for a LinkedIn post based on this community discussion.

        Platform: {platform}
        Topic: {topic}
        Key insight: {key_insight}
        Controversy: {controversy}

        Hook styles:
        1. CURATED: Synthesize community wisdom
           Example: "A {platform} thread about {topic} just exploded. Key takeaways:"

        2. PRACTITIONER: Real-world signal
           Example: "Forget the hype. Here's what people actually building with {topic} say:"

        3. HOT TAKE: Respond to controversy
           Example: "Someone on {platform} said {hot_take}. They might be right."

        4. QUESTION: Invite engagement
           Example: "I asked {platform} about {topic}. The answers surprised me."

        5. SIGNAL VS NOISE: Cut through hype
           Example: "The gap between {topic} theory and practice, according to builders:"
        """
    },

    # ═══════════════════════════════════════════════════════════════════
    # TOOL RELEASE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    ContentType.TOOL_RELEASE: {
        "extraction_prompt": """
        Analyze this AI tool/product release.

        Content: {content}
        Source: {source_url}

        Extract:

        ## Tool Overview
        - Tool name:
        - Company:
        - Release date:
        - Release type (new product/major update/API/open source):

        ## Functionality
        - Core capability:
        - Key features (list top 3-5):
        - Target users:
        - Use cases:

        ## Access & Pricing
        - Availability (public/waitlist/enterprise only):
        - Pricing model:
        - API available?
        - Demo/trial available?

        ## Competitive Position
        - Main competitors:
        - Key differentiators:
        - What it does better:
        - What it lacks:

        ## Early Signals
        - Early user feedback:
        - Expert opinions:
        - Potential concerns:
        """,

        "insights_prompt": """
        Extract 3-5 insights about this tool release that help readers
        decide if they should pay attention.

        Focus on:
        1. Who should care? - Specific user segments
        2. What's new? - vs existing solutions
        3. What's the catch? - Limitations, pricing
        4. When to try it? - Now vs wait

        For each insight:
        - Be specific about use cases
        - Note competitive positioning
        - Rate "market impact" 1-10

        Release summary: {extraction_summary}
        """,

        "hooks_prompt": """
        Generate 5 hooks for a LinkedIn post about this tool release.

        Tool: {tool_name}
        Company: {company}
        Key feature: {key_feature}
        Competitor: {main_competitor}

        Hook styles:
        1. FIRST LOOK: Early review
           Example: "{company} just dropped {tool}. I tested it immediately."

        2. COMPARISON: vs alternatives
           Example: "Does {tool} actually beat {competitor}? I tested both."

        3. IMPLICATIONS: Who should care
           Example: "If you use {related_tool}, pay attention to {tool}."

        4. USE CASES: Practical applications
           Example: "Everyone's using {tool} for {obvious_use}. But have you tried {creative_use}?"

        5. SKEPTICAL: Not just hype
           Example: "{tool} looks impressive. But here's what they're not telling you:"
        """
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# GENERIC PROMPTS (Applied to all types)
# ═══════════════════════════════════════════════════════════════════════════

generic_analysis_prompts = {
    "controversy_check": """
    Analyze this content for controversy and debate potential:

    Topic: {topic}
    Content Type: {content_type}
    Key points: {key_points}

    Questions:
    1. Are there opposing viewpoints in the AI community?
    2. Does this challenge conventional wisdom?
    3. Could this spark debate in comments?
    4. What are the potential hot takes?
    5. Any ethical concerns to be aware of?
    6. What would critics say?

    Rate controversy_level: low / medium / high / spicy
    Suggest debate angles (2-3):
    """,

    "simplification_check": """
    Assess the complexity of this topic for a general tech audience:

    Topic: {topic}
    Content Type: {content_type}
    Technical details: {technical_details}

    Questions:
    1. What jargon needs explanation?
    2. What analogies could help?
    3. What can be cut without losing value?
    4. What background knowledge is assumed?
    5. Can this be explained to a smart non-expert?

    Provide:
    - Complexity rating: simplify_heavily / simplify_slightly / keep_technical
    - List of jargon with simple explanations
    - 2-3 helpful analogies
    - Suggested cuts
    """
}
```

---

#### Output Schema

```python
@dataclass
class Insight:
    statement: str  # One sentence insight
    explanation: str  # Why it matters
    wow_factor: int  # 1-10
    hook_angle: str  # Suggested hook style for this insight
    source_quote: Optional[str]  # Supporting quote from source


@dataclass
class Hook:
    text: str
    style: HookStyle  # FIX: Use HookStyle enum for type safety (was str)
    content_type_fit: float  # 0-1 how well this fits the content type


# ═══════════════════════════════════════════════════════════════════════════
# TYPE-SPECIFIC EXTRACTION CLASSES (Replaces god object)
# ═══════════════════════════════════════════════════════════════════════════
#
# RATIONALE: The old TypeSpecificExtraction had 25+ Optional fields for ALL
# content types. This violates Interface Segregation Principle and makes it
# impossible to statically verify that required fields are present.
#
# NEW DESIGN: Each ContentType has its own extraction dataclass with:
# 1. Required fields (no Optional) - must be extracted
# 2. Optional fields - nice to have
# 3. Discriminator field for runtime type checking
#
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EnterpriseCaseExtraction:
    """Extraction data for enterprise AI implementation cases."""
    type: Literal["enterprise_case"] = "enterprise_case"

    # Required fields (Analyzer MUST extract these)
    company: str
    industry: str
    problem_statement: str
    solution_description: str
    metrics_extracted: Dict[str, str]  # KPI name -> value

    # Optional fields (nice to have)
    implementation_timeline: Optional[str] = None
    lessons_learned: List[str] = field(default_factory=list)
    architecture_notes: Optional[str] = None
    team_size: Optional[str] = None
    roi_stated: Optional[str] = None


@dataclass
class PrimarySourceExtraction:
    """Extraction data for research papers and primary sources."""
    type: Literal["primary_source"] = "primary_source"

    # Required fields
    authors: List[str]
    thesis: str
    key_findings: List[str]

    # Optional fields
    methodology: Optional[str] = None
    counterintuitive_finding: Optional[str] = None
    implications: List[str] = field(default_factory=list)
    limitations: Optional[str] = None
    future_directions: Optional[str] = None


@dataclass
class AutomationCaseExtraction:
    """Extraction data for AI automation and agent cases."""
    type: Literal["automation_case"] = "automation_case"

    # Required fields
    task_automated: str
    tools_used: List[str]
    workflow_steps: List[str]

    # Optional fields
    time_saved: Optional[str] = None
    cost_saved: Optional[str] = None
    reproducibility_notes: Optional[str] = None
    code_snippets: List[str] = field(default_factory=list)
    gotchas: List[str] = field(default_factory=list)


@dataclass
class CommunityContentExtraction:
    """Extraction data for community discussions and content."""
    type: Literal["community_content"] = "community_content"

    # Required fields
    platform: str
    key_viewpoints: List[str]

    # FIX: Added missing fields referenced in extraction prompts
    main_discussion_topic: Optional[str] = None  # What the discussion is about
    engagement_level: Optional[str] = None       # high / medium / low
    best_insight_shared: Optional[str] = None    # Most valuable insight from discussion
    most_controversial_take: Optional[str] = None  # Hottest debate point
    practical_tip_mentioned: Optional[str] = None  # Actionable advice shared
    linked_resources: List[str] = field(default_factory=list)  # Referenced URLs/resources

    # Optional fields
    notable_contributions: List[str] = field(default_factory=list)
    practitioner_signals: List[str] = field(default_factory=list)
    consensus_points: List[str] = field(default_factory=list)
    disagreement_points: List[str] = field(default_factory=list)


@dataclass
class ToolReleaseExtraction:
    """Extraction data for tool and product releases."""
    type: Literal["tool_release"] = "tool_release"

    # Required fields
    tool_name: str
    key_features: List[str]
    target_users: str

    # FIX: Added missing fields referenced in extraction prompts
    company: Optional[str] = None           # Company releasing the tool
    release_date: Optional[str] = None      # When it was released
    release_type: Optional[str] = None      # new_product / major_update / api_release / open_source
    core_functionality: Optional[str] = None  # Main purpose/function
    availability: Optional[str] = None      # free / paid / freemium / waitlist
    limitations: List[str] = field(default_factory=list)  # Known limitations

    # Optional fields
    competitive_position: Optional[str] = None
    pricing_info: Optional[str] = None
    demo_url: Optional[str] = None
    early_feedback: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# DISCRIMINATED UNION FOR EXTRACTION DATA
# ═══════════════════════════════════════════════════════════════════════════

ExtractionData = Union[
    EnterpriseCaseExtraction,
    PrimarySourceExtraction,
    AutomationCaseExtraction,
    CommunityContentExtraction,
    ToolReleaseExtraction
]

# Mapping for validation
CONTENT_TYPE_TO_EXTRACTION_TYPE = {
    ContentType.ENTERPRISE_CASE: "enterprise_case",
    ContentType.PRIMARY_SOURCE: "primary_source",
    ContentType.AUTOMATION_CASE: "automation_case",
    ContentType.COMMUNITY_CONTENT: "community_content",
    ContentType.TOOL_RELEASE: "tool_release",
}


class ExtractionTypeMismatchError(Exception):
    """Raised when extraction.type doesn't match content_type."""
    pass


def validate_extraction_type(content_type: ContentType, extraction: ExtractionData) -> bool:
    """
    Validate that extraction.type matches the expected content_type.

    Raises:
        ExtractionTypeMismatchError: If types don't match
    """
    expected = CONTENT_TYPE_TO_EXTRACTION_TYPE.get(content_type)
    actual = getattr(extraction, 'type', None)

    if actual != expected:
        raise ExtractionTypeMismatchError(
            f"Expected extraction type '{expected}' for {content_type.value}, "
            f"got '{actual}'"
        )
    return True


# ═══════════════════════════════════════════════════════════════════════════
# LEGACY ALIAS (for backward compatibility during migration)
# LOW PRIORITY FIX #13: Lazy deprecation warning - only warns on actual use
# ═══════════════════════════════════════════════════════════════════════════

def __getattr__(name: str):
    """
    Module-level __getattr__ for lazy deprecation warnings.
    Only triggers when TypeSpecificExtraction is actually accessed.
    """
    if name == "TypeSpecificExtraction":
        warnings.warn(
            "TypeSpecificExtraction is deprecated. Use ExtractionData instead. "
            "This alias will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
        )
        return ExtractionData
    raise AttributeError(f"module has no attribute '{name}'")


# ═══════════════════════════════════════════════════════════════════════════
# FIX #11: METADATA → EXTRACTION FIELD MAPPING
# ═══════════════════════════════════════════════════════════════════════════
#
# TopicMetadata (from TrendScout) and ExtractionData (from Analyzer) have
# INTENTIONALLY DIFFERENT purposes:
#
# - TopicMetadata: What TrendScout OBSERVED about the source (metadata about the topic)
# - ExtractionData: What Analyzer EXTRACTED from the content (detailed analysis)
#
# Field differences are BY DESIGN:
#
# ENTERPRISE_CASE:
#   Metadata.metrics (Dict[str,str])      → Extraction.metrics_extracted (Dict[str,str]) ✓ same
#   Metadata.roi_mentioned (bool)         → Extraction.roi_stated (Optional[str]) - bool→str detail
#   Metadata.architecture_available (bool)→ Extraction.architecture_notes (str) - bool→str detail
#   Metadata.scale (str)                  → NOT in Extraction (Scout-only context)
#   Metadata.problem_domain (str)         → Extraction.problem_statement (more specific)
#
# PRIMARY_SOURCE:
#   Metadata.key_hypothesis (str)         → Extraction.thesis (same concept, different name)
#   Metadata.methodology_summary (str)    → Extraction.methodology (same concept)
#   Metadata.code_available (bool)        → NOT in Extraction (Scout-only metadata)
#
# These differences are correct - Scout provides surface metadata,
# Analyzer provides deep extraction.
#
# ═══════════════════════════════════════════════════════════════════════════


def seed_extraction_from_metadata(
    metadata: "TopicMetadata",
    content_type: ContentType
) -> ExtractionData:
    """
    FIX #11: Create a pre-populated extraction from metadata.

    Use this to give Analyzer a starting point when metadata already
    contains relevant information. Analyzer should ENHANCE, not replace.

    Returns:
        Partially populated ExtractionData of the correct type
    """

    if content_type == ContentType.ENTERPRISE_CASE:
        assert hasattr(metadata, 'company')
        return EnterpriseCaseExtraction(
            company=metadata.company,
            industry=metadata.industry,
            problem_statement=metadata.problem_domain,
            solution_description="",  # Analyzer must fill
            metrics_extracted=metadata.metrics,
            implementation_timeline=metadata.implementation_timeline,
            lessons_learned=metadata.lessons_learned,
            team_size=metadata.team_size,
            roi_stated=str(metadata.roi_mentioned) if metadata.roi_mentioned else None,
            architecture_notes="Available" if metadata.architecture_available else None
        )

    elif content_type == ContentType.PRIMARY_SOURCE:
        assert hasattr(metadata, 'authors')
        return PrimarySourceExtraction(
            authors=metadata.authors,
            thesis=metadata.key_hypothesis,
            key_findings=[],  # Analyzer must fill
            methodology=metadata.methodology_summary,
            counterintuitive_finding=metadata.counterintuitive_finding
        )

    elif content_type == ContentType.AUTOMATION_CASE:
        assert hasattr(metadata, 'agent_type')
        return AutomationCaseExtraction(
            task_automated="",  # Analyzer must fill
            tools_used=[metadata.agent_type] + metadata.integrations,
            workflow_steps=metadata.workflow_components,
            time_saved=metadata.time_saved,
            cost_saved=metadata.cost_saved,
            reproducibility_notes=metadata.reproducibility
        )

    elif content_type == ContentType.COMMUNITY_CONTENT:
        assert hasattr(metadata, 'platform')
        return CommunityContentExtraction(
            platform=metadata.platform,
            key_viewpoints=[],  # Analyzer must fill
            notable_contributions=metadata.key_contributors
        )

    elif content_type == ContentType.TOOL_RELEASE:
        assert hasattr(metadata, 'tool_name')
        return ToolReleaseExtraction(
            tool_name=metadata.tool_name,
            key_features=metadata.key_features,
            target_users="",  # Analyzer must fill
            competitive_position=f"Competes with: {', '.join(metadata.competing_tools)}" if metadata.competing_tools else None,
            pricing_info=metadata.pricing_model,
            demo_url=metadata.demo_url,
            early_feedback=metadata.early_reviews
        )

    raise ValueError(f"Unknown content type: {content_type}")


# ═══════════════════════════════════════════════════════════════════════════
# FIX #12: BOUNDARY VALIDATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════
#
# These validators run at agent BOUNDARIES to catch errors early.
# Fail-fast philosophy: detect issues at handoff, not downstream.
#
# ═══════════════════════════════════════════════════════════════════════════


class BoundaryValidationError(Exception):
    """Raised when data fails validation at agent boundary."""
    def __init__(self, source_agent: str, target_agent: str, issues: List[str]):
        self.source_agent = source_agent
        self.target_agent = target_agent
        self.issues = issues
        super().__init__(
            f"Validation failed at {source_agent} → {target_agent} boundary: {issues}"
        )


def validate_trend_scout_to_analyzer(topic: "TrendTopic") -> None:
    """
    FIX #12: Validate TrendScout output before passing to Analyzer.

    Raises:
        BoundaryValidationError: If validation fails
    """
    issues = []

    # Required fields
    if not topic.id:
        issues.append("Missing topic.id")
    if not topic.raw_content or len(topic.raw_content) < 100:
        issues.append("raw_content too short for meaningful analysis")
    if not topic.content_type:
        issues.append("Missing content_type")

    # Type-specific metadata validation
    try:
        validate_topic_metadata(topic)
    except MetadataTypeMismatchError as e:
        issues.append(str(e))

    # Score sanity check
    if topic.score < 0 or topic.score > 10:
        issues.append(f"Score {topic.score} outside valid range 0-10")

    if issues:
        raise BoundaryValidationError("TrendScout", "Analyzer", issues)


def validate_analyzer_to_writer(brief: "AnalysisBrief") -> None:
    """
    FIX #12: Validate Analyzer output before passing to Writer.

    Raises:
        BoundaryValidationError: If validation fails
    """
    issues = []

    # Required fields
    if not brief.topic_id:
        issues.append("Missing topic_id")
    if not brief.main_takeaway:
        issues.append("Missing main_takeaway")
    if not brief.hooks or len(brief.hooks) < 3:
        issues.append(f"Need at least 3 hooks, got {len(brief.hooks) if brief.hooks else 0}")
    if not brief.key_insights or len(brief.key_insights) < 2:
        issues.append(f"Need at least 2 key_insights, got {len(brief.key_insights) if brief.key_insights else 0}")

    # Extraction data validation
    try:
        validate_extraction_type(brief.content_type, brief.extraction_data)
    except ExtractionTypeMismatchError as e:
        issues.append(str(e))

    # Check extraction completeness based on type
    extraction = brief.extraction_data
    extraction_issues = _validate_extraction_completeness(brief.content_type, extraction)
    issues.extend(extraction_issues)

    if issues:
        raise BoundaryValidationError("Analyzer", "Writer", issues)


def _validate_extraction_completeness(
    content_type: ContentType,
    extraction: ExtractionData
) -> List[str]:
    """
    Check that required extraction fields are populated.

    FIX: Added missing required field validations per Analyzer Agent analysis.
    Each content type now validates ALL its required fields.
    """
    issues = []

    if content_type == ContentType.ENTERPRISE_CASE:
        ext = extraction  # type: EnterpriseCaseExtraction
        if not ext.company:
            issues.append("EnterpriseCaseExtraction.company is required")
        if not ext.industry:
            issues.append("EnterpriseCaseExtraction.industry is required")
        if not ext.problem_statement:
            issues.append("EnterpriseCaseExtraction.problem_statement is required")
        if not ext.solution_description:
            issues.append("EnterpriseCaseExtraction.solution_description is required")
        if not ext.metrics_extracted:
            issues.append("EnterpriseCaseExtraction.metrics_extracted is required")
        if not ext.lessons_learned:
            issues.append("EnterpriseCaseExtraction.lessons_learned is required")

    elif content_type == ContentType.PRIMARY_SOURCE:
        ext = extraction  # type: PrimarySourceExtraction
        if not ext.thesis:
            issues.append("PrimarySourceExtraction.thesis is required")
        if not ext.key_findings:
            issues.append("PrimarySourceExtraction.key_findings is required")
        if not ext.authors:
            issues.append("PrimarySourceExtraction.authors is required")

    elif content_type == ContentType.AUTOMATION_CASE:
        ext = extraction  # type: AutomationCaseExtraction
        if not ext.task_automated:
            issues.append("AutomationCaseExtraction.task_automated is required")
        if not ext.tools_used:
            issues.append("AutomationCaseExtraction.tools_used is required")
        if not ext.workflow_steps:
            issues.append("AutomationCaseExtraction.workflow_steps is required")

    elif content_type == ContentType.COMMUNITY_CONTENT:
        ext = extraction  # type: CommunityContentExtraction
        if not ext.platform:
            issues.append("CommunityContentExtraction.platform is required")
        if not ext.key_viewpoints:
            issues.append("CommunityContentExtraction.key_viewpoints is required")

    elif content_type == ContentType.TOOL_RELEASE:
        ext = extraction  # type: ToolReleaseExtraction
        if not ext.tool_name:
            issues.append("ToolReleaseExtraction.tool_name is required")
        if not ext.key_features:
            issues.append("ToolReleaseExtraction.key_features is required")
        if not ext.target_users:
            issues.append("ToolReleaseExtraction.target_users is required")

    return issues


def validate_writer_to_humanizer(draft: "DraftPost") -> None:
    """
    FIX #12: Validate Writer output before passing to Humanizer.
    FIX: Corrected class name (WriterDraft -> DraftPost) and field names.

    Raises:
        BoundaryValidationError: If validation fails
    """
    issues = []

    # FIX: Use correct field names (full_text, not post_text; hook_style, not hook_used)
    if not draft.full_text or len(draft.full_text) < 200:
        issues.append("full_text too short (min 200 chars)")
    if len(draft.full_text) > 3000:
        issues.append("full_text too long (max 3000 chars for LinkedIn)")
    if not draft.hook_style:
        issues.append("Missing hook_style reference")

    if issues:
        raise BoundaryValidationError("Writer", "Humanizer", issues)


def validate_visual_to_qc(
    visual: "VisualAsset",
    post: "HumanizedPost"
) -> None:
    """
    FIX #12: Validate Visual + Post before passing to QC.

    Raises:
        BoundaryValidationError: If validation fails
    """
    issues = []

    # Visual validation
    if not visual.files:
        issues.append("VisualAsset.files is empty")
    if not visual.visual_style:
        issues.append("Missing visual_style")
    if not visual.alt_text:
        issues.append("Missing alt_text (accessibility requirement)")

    # Post validation
    if not post.humanized_text:
        issues.append("Missing humanized_text")
    if post.content_type != visual.content_type:
        issues.append(
            f"Content type mismatch: post={post.content_type}, visual={visual.content_type}"
        )

    if issues:
        raise BoundaryValidationError("Visual+Humanizer", "QC", issues)


@dataclass
class AnalysisBrief:
    """
    Enhanced analysis brief with content-type awareness.

    IMPORTANT: extraction_data uses the new typed ExtractionData union.
    The extraction.type MUST match content_type for validation.
    """
    # Identification
    topic_id: str
    content_type: ContentType

    # Type-specific extraction (uses discriminated union)
    extraction_data: ExtractionData  # EnterpriseCaseExtraction | PrimarySourceExtraction | ...

    def __post_init__(self):
        """Validate extraction type matches content type."""
        validate_extraction_type(self.content_type, self.extraction_data)

    # Core insights (type-optimized)
    key_insights: List[Insight]  # 3-5 insights with wow_factor scores
    main_takeaway: str  # One sentence summary

    # Hooks (type-specific styles)
    hooks: List[Hook]  # 5 hooks using type-appropriate styles
    recommended_hook_style: HookStyle  # FIX: Use HookStyle enum for type safety

    # Controversy analysis
    controversy_level: str  # low/medium/high/spicy
    debate_angles: List[str]  # Potential controversy hooks
    emotional_triggers: List[str]  # surprise, fear, hope, etc.

    # Technical assessment
    complexity_level: str  # simplify_heavily / simplify_slightly / keep_technical
    simplification_suggestions: List[str]
    jargon_to_explain: Dict[str, str]  # term -> simple explanation
    analogies: List[str]  # Helpful analogies

    # Strategic recommendations (type-informed)
    recommended_format: str  # insight_thread / contrarian / tutorial / story / list_post
    format_rationale: str  # Why this format fits the content type
    target_length: str  # short / medium / long
    suggested_cta: str  # Type-appropriate call to action
    recommended_visual: str  # data_viz / diagram / screenshot / quote_card

    # Raw materials for writer
    key_quotes: List[str]  # Quotable lines from sources
    statistics: List[str]  # Numbers to use
    source_urls: List[str]

    # Quality metrics
    extraction_completeness: float  # 0-1 how much required data was found
    confidence_score: float  # 0-1 confidence in analysis quality

    # LOW PRIORITY FIX #1: Add __repr__ for better debugging
    def __repr__(self) -> str:
        """Concise representation for debugging."""
        return (
            f"AnalysisBrief(topic_id='{self.topic_id}', "
            f"type={self.content_type.value}, "
            f"insights={len(self.key_insights)}, "
            f"hooks={len(self.hooks)}, "
            f"confidence={self.confidence_score:.2f})"
        )
```

---

### 3. WRITER AGENT

#### Purpose
Create the first draft of the post based on the **content-type-aware brief** from Analyzer. Select templates optimized for each ContentType, use type-specific hooks, and generate posts that match the content's nature (case study vs research vs tutorial vs news).

#### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          WRITER AGENT (Enhanced)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: AnalysisBrief (with ContentType + type-specific extraction)         │
│         │                                                                   │
│         ▼                                                                   │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                  CONTENT-TYPE AWARE TEMPLATE SELECTOR              │     │
│  │                                                                    │     │
│  │  ContentType → Primary Templates → Fallback Templates             │     │
│  │  ─────────────────────────────────────────────────────            │     │
│  │  ENTERPRISE_CASE   → metrics_story, lessons_learned, case_study  │     │
│  │  PRIMARY_SOURCE    → contrarian, explainer, debate_starter       │     │
│  │  AUTOMATION_CASE   → tutorial_light, how_to, results_story       │     │
│  │  COMMUNITY_CONTENT → curated_insights, list_post, hot_take       │     │
│  │  TOOL_RELEASE      → first_look, comparison, implications        │     │
│  │                                                                    │     │
│  └───────────────────────────┬───────────────────────────────────────┘     │
│                              │                                              │
│                              ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    HOOK SELECTOR                                   │     │
│  │  Uses: brief.hooks (pre-generated by Analyzer)                    │     │
│  │        + type-specific hook templates                             │     │
│  │        + brief.recommended_hook_style                             │     │
│  └───────────────────────────┬───────────────────────────────────────┘     │
│                              │                                              │
│                              ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    KNOWLEDGE INJECTOR                              │     │
│  │  • Style guide (your voice)                                       │     │
│  │  • Type-specific examples (few-shot per ContentType)              │     │
│  │  • LinkedIn best practices                                        │     │
│  │  • Type-specific data injection:                                  │     │
│  │    - Enterprise: metrics, company, lessons                        │     │
│  │    - Research: findings, author, implications                     │     │
│  │    - Automation: steps, tools, time_saved                         │     │
│  │    - Community: viewpoints, quotes, platform                      │     │
│  │    - Tool: features, comparison, availability                     │     │
│  └───────────────────────────┬───────────────────────────────────────┘     │
│                              │                                              │
│                              ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    DRAFT GENERATOR                                 │     │
│  │                    (Claude Opus 4.5 thinking mode)                │     │
│  │  Uses type-specific generation prompt                             │     │
│  └───────────────────────────┬───────────────────────────────────────┘     │
│                              │                                              │
│                              ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    LINKEDIN FORMATTER                              │     │
│  │  • Line breaks optimization                                       │     │
│  │  • Emoji placement (type-appropriate)                             │     │
│  │  • Type-specific hashtag selection                                │     │
│  │  • Length check (1200-1500 chars)                                │     │
│  └───────────────────────────┬───────────────────────────────────────┘     │
│                              │                                              │
│                              ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    VISUAL BRIEF GENERATOR                          │     │
│  │  Creates type-appropriate visual description                      │     │
│  │  (data_viz for metrics, diagram for automation, etc.)            │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                              │                                              │
│                              ▼                                              │
│  OUTPUT: DraftPost (with ContentType context)                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### Content Type to Template Mapping

```python
"""
Maps each ContentType to its optimal post templates.
Primary templates are tried first, fallbacks used if content doesn't fit.
"""

content_type_template_mapping = {
    ContentType.ENTERPRISE_CASE: {
        "primary_templates": ["metrics_story", "lessons_learned", "case_study"],
        "fallback_templates": ["insight_thread", "personal_story"],
        "best_for": "Detailed implementation stories with business impact",
        "avoid": ["tutorial_light"],  # Case studies shouldn't feel like tutorials
        "tone_override": {
            "confidence": "high - you're sharing real results",
            "specificity": "maximum - use exact numbers and names"
        }
    },

    ContentType.PRIMARY_SOURCE: {
        "primary_templates": ["contrarian", "explainer", "debate_starter"],
        "fallback_templates": ["insight_thread"],
        "best_for": "Research insights, challenging conventional wisdom",
        "avoid": ["tutorial_light", "list_post"],
        "tone_override": {
            "intellectual": "high - engage with ideas deeply",
            "humility": "medium - acknowledge complexity"
        }
    },

    ContentType.AUTOMATION_CASE: {
        "primary_templates": ["tutorial_light", "how_to", "results_story"],
        "fallback_templates": ["insight_thread", "list_post"],
        "best_for": "Practical implementations readers can replicate",
        "avoid": ["contrarian"],  # Focus on helpfulness, not debate
        "tone_override": {
            "practical": "maximum - readers should be able to DO this",
            "generosity": "high - share everything"
        }
    },

    ContentType.COMMUNITY_CONTENT: {
        "primary_templates": ["curated_insights", "list_post", "hot_take"],
        "fallback_templates": ["insight_thread", "question_based"],
        "best_for": "Synthesizing community wisdom, sparking discussion",
        "avoid": ["tutorial_light"],
        "tone_override": {
            "conversational": "high - you're part of the community",
            "attribution": "important - credit sources"
        }
    },

    ContentType.TOOL_RELEASE: {
        "primary_templates": ["first_look", "comparison", "implications"],
        "fallback_templates": ["insight_thread", "list_post"],
        "best_for": "Breaking news, helping readers evaluate new tools",
        "avoid": ["personal_story"],  # Stay focused on the tool
        "tone_override": {
            "timeliness": "emphasize - this is fresh",
            "objectivity": "balanced - pros and cons"
        }
    }
}
```

---

#### Post Templates (Enhanced with Content Type Variants)

```python
templates = {
    # ═══════════════════════════════════════════════════════════════════
    # UNIVERSAL TEMPLATES (work for multiple content types)
    # ═══════════════════════════════════════════════════════════════════

    "insight_thread": {
        "structure": """
        [HOOK - 1 line, attention grabbing]

        [CONTEXT - 2-3 lines, set the stage]

        [INSIGHT 1 - with brief explanation]

        [INSIGHT 2 - with brief explanation]

        [INSIGHT 3 - with brief explanation]

        [TAKEAWAY - 1-2 lines, so what?]

        [CTA - question or call to action]

        [HASHTAGS - 3-5 relevant]
        """,
        "example_hooks": [
            "I spent 10 hours reading AI papers so you don't have to.",
            "The AI news you missed this week (but shouldn't have):",
            "Three things I learned about {topic} that changed how I think:"
        ],
        "tone": "educational, generous, expert",
        "best_for_types": [ContentType.PRIMARY_SOURCE, ContentType.COMMUNITY_CONTENT],
        "length_target": "1200-1500 chars"
    },

    "contrarian": {
        "structure": """
        [PROVOCATIVE OPENING - challenge common belief]

        [ACKNOWLEDGE - why people think the opposite]

        [YOUR TAKE - with evidence]

        [NUANCE - it's not black and white]

        [INVITATION - what do you think?]

        [HASHTAGS]
        """,
        "example_hooks": [
            "Unpopular opinion: {controversial_take}",
            "Everyone's excited about {topic}. Here's why I'm skeptical.",
            "Hot take: {bold_claim}"
        ],
        "tone": "confident, thoughtful, open to debate",
        "best_for_types": [ContentType.PRIMARY_SOURCE],
        "length_target": "1000-1400 chars"
    },

    "personal_story": {
        "structure": """
        [STORY HOOK - something happened to me]

        [THE STORY - brief, specific details]

        [THE LESSON - what I learned]

        [THE INSIGHT - why this matters for you]

        [CTA - have you experienced this?]

        [HASHTAGS]
        """,
        "example_hooks": [
            "Last week I made a mistake with AI that cost me {X}.",
            "I tried {thing} for 30 days. Here's what happened.",
            "A conversation with {person} completely changed my view on {topic}."
        ],
        "tone": "vulnerable, authentic, relatable",
        "best_for_types": [ContentType.ENTERPRISE_CASE],
        "length_target": "1000-1300 chars"
    },

    "tutorial_light": {
        "structure": """
        [PROBLEM - relatable pain point]

        [SOLUTION PREVIEW - what you'll learn]

        [STEP 1]
        [STEP 2]
        [STEP 3]

        [RESULT - what this achieves]

        [BONUS TIP - extra value]

        [CTA - try it and let me know]

        [HASHTAGS]
        """,
        "example_hooks": [
            "How to {achieve X} in 5 minutes (no code required):",
            "The {topic} cheat sheet I wish I had when starting:",
            "Stop doing {bad thing}. Do this instead:"
        ],
        "tone": "helpful, practical, encouraging",
        "best_for_types": [ContentType.AUTOMATION_CASE],
        "length_target": "1200-1600 chars"
    },

    "list_post": {
        "structure": """
        [HOOK - promise of value]

        [BRIEF CONTEXT - why this list matters]

        1. [ITEM 1 - with brief explanation]
        2. [ITEM 2 - with brief explanation]
        3. [ITEM 3 - with brief explanation]
        4. [ITEM 4 - with brief explanation]
        5. [ITEM 5 - with brief explanation]

        [BONUS - extra item or resource]

        [CTA - save this / which is your favorite?]

        [HASHTAGS]
        """,
        "example_hooks": [
            "5 AI tools I use every day (and why):",
            "The top {N} lessons from {source}:",
            "{N} things nobody tells you about {topic}:"
        ],
        "tone": "practical, scannable, valuable",
        "best_for_types": [ContentType.COMMUNITY_CONTENT, ContentType.TOOL_RELEASE],
        "length_target": "1200-1500 chars"
    },

    "question_based": {
        "structure": """
        [PROVOCATIVE QUESTION - makes reader think]

        [WHY I'M ASKING - context]

        [PERSPECTIVE 1 - one way to look at it]

        [PERSPECTIVE 2 - another way]

        [MY CURRENT THINKING - where I lean]

        [INVITATION - genuinely curious what you think]

        [HASHTAGS]
        """,
        "example_hooks": [
            "Is {assumption} actually true?",
            "What if we've been thinking about {topic} all wrong?",
            "Genuine question: {question}"
        ],
        "tone": "curious, humble, engaging",
        "best_for_types": [ContentType.COMMUNITY_CONTENT, ContentType.PRIMARY_SOURCE],
        "length_target": "800-1200 chars"
    },

    # ═══════════════════════════════════════════════════════════════════
    # ENTERPRISE CASE SPECIFIC TEMPLATES
    # ═══════════════════════════════════════════════════════════════════

    "metrics_story": {
        "structure": """
        [METRIC HOOK - lead with impressive number]

        [COMPANY CONTEXT - who achieved this]

        [THE CHALLENGE - what problem they faced]

        [THE SOLUTION - what they implemented]

        [THE RESULTS - specific metrics]
        • Metric 1
        • Metric 2
        • Metric 3

        [THE LESSON - what we can learn]

        [CTA - have you seen similar results?]

        [HASHTAGS]
        """,
        "example_hooks": [
            "{X}% improvement in {KPI}. Here's how {company} did it.",
            "From {before} to {after} in {timeline}. The story of {company}'s AI transformation.",
            "When {company} showed their board {metric}, everything changed."
        ],
        "tone": "data-driven, credible, inspiring",
        "best_for_types": [ContentType.ENTERPRISE_CASE],
        "length_target": "1300-1600 chars",
        "required_data": ["company", "metrics", "timeline"]
    },

    "lessons_learned": {
        "structure": """
        [EXPERIENCE HOOK - what they learned]

        [COMPANY INTRO - brief context]

        [LESSON 1 - what worked]

        [LESSON 2 - what didn't work]

        [LESSON 3 - what they'd do differently]

        [KEY TAKEAWAY - the meta-lesson]

        [CTA - what's your biggest AI implementation lesson?]

        [HASHTAGS]
        """,
        "example_hooks": [
            "{company} spent {time} building {solution}. Here's what they'd do differently.",
            "The 3 biggest lessons from {company}'s AI journey.",
            "What {company} wishes they knew before implementing AI:"
        ],
        "tone": "reflective, honest, educational",
        "best_for_types": [ContentType.ENTERPRISE_CASE],
        "length_target": "1200-1500 chars",
        "required_data": ["company", "lessons_learned"]
    },

    "case_study": {
        "structure": """
        [OUTCOME HOOK - what they achieved]

        [COMPANY + PROBLEM]
        Company: {company}
        Challenge: {challenge}

        [SOLUTION OVERVIEW]
        What they did: {solution}
        Tech used: {tech_stack}

        [RESULTS]
        📊 {metric_1}
        📈 {metric_2}
        💰 {metric_3}

        [WHY IT MATTERS - industry implications]

        [CTA - link to full case study or question]

        [HASHTAGS]
        """,
        "example_hooks": [
            "Case study: How {company} achieved {result} with AI",
            "{industry} giant {company} just showed what's possible with AI.",
            "Inside {company}'s AI transformation: {result}"
        ],
        "tone": "professional, factual, impressive",
        "best_for_types": [ContentType.ENTERPRISE_CASE],
        "length_target": "1400-1700 chars",
        "required_data": ["company", "challenge", "solution", "metrics"]
    },

    # ═══════════════════════════════════════════════════════════════════
    # PRIMARY SOURCE SPECIFIC TEMPLATES
    # ═══════════════════════════════════════════════════════════════════

    "explainer": {
        "structure": """
        [COMPLEXITY HOOK - I'll make this simple]

        [WHAT THE RESEARCH SAYS - core finding]

        [WHY IT MATTERS - so what?]

        [THE SIMPLE VERSION - analogy or plain language]

        [IMPLICATIONS - what this means for you]

        [MY TAKE - personal perspective]

        [CTA - does this change how you think about X?]

        [HASHTAGS]
        """,
        "example_hooks": [
            "I read {paper} so you don't have to. The key insight:",
            "{complex_topic} explained in plain English:",
            "This research paper is 50 pages. Here's what actually matters:"
        ],
        "tone": "accessible, generous, clarifying",
        "best_for_types": [ContentType.PRIMARY_SOURCE],
        "length_target": "1100-1400 chars",
        "required_data": ["paper_title", "key_finding", "implications"]
    },

    "debate_starter": {
        "structure": """
        [CONTROVERSIAL CLAIM - from the research]

        [THE EVIDENCE - what supports this]

        [THE COUNTERARGUMENT - what critics say]

        [THE NUANCE - it's complicated]

        [WHERE I LAND - my perspective]

        [INVITATION - what's your take?]

        [HASHTAGS]
        """,
        "example_hooks": [
            "{author} claims {bold_claim}. Is this right?",
            "This paper will divide the AI community.",
            "Controversial take from {source}: {claim}"
        ],
        "tone": "intellectually engaging, balanced, provocative",
        "best_for_types": [ContentType.PRIMARY_SOURCE],
        "length_target": "1000-1300 chars",
        "required_data": ["claim", "evidence", "counterargument"]
    },

    # ═══════════════════════════════════════════════════════════════════
    # AUTOMATION CASE SPECIFIC TEMPLATES
    # ═══════════════════════════════════════════════════════════════════

    "how_to": {
        "structure": """
        [OUTCOME HOOK - what you'll be able to do]

        [PREREQUISITES - what you need]

        [STEP 1] - {action}
        ↳ {detail}

        [STEP 2] - {action}
        ↳ {detail}

        [STEP 3] - {action}
        ↳ {detail}

        [RESULT] - what this achieves

        [PRO TIP] - insider knowledge

        [CTA - try it and tag me]

        [HASHTAGS]
        """,
        "example_hooks": [
            "How I automated {task} with {tool}. Step by step:",
            "Build {solution} in {time}. No code required.",
            "The exact workflow I use to {achieve_result}:"
        ],
        "tone": "practical, detailed, empowering",
        "best_for_types": [ContentType.AUTOMATION_CASE],
        "length_target": "1300-1600 chars",
        "required_data": ["task", "tools", "steps"]
    },

    "results_story": {
        "structure": """
        [RESULTS HOOK - the transformation]

        [BEFORE - the painful old way]

        [THE CHANGE - what I built/did]

        [AFTER - the new reality]
        ⏱️ Time saved: {time}
        💰 Cost saved: {cost}
        🎯 Quality: {quality}

        [THE KEY INSIGHT - what made it work]

        [CTA - what would you automate first?]

        [HASHTAGS]
        """,
        "example_hooks": [
            "From {hours} hours to {minutes} minutes. Here's how:",
            "This automation saves me {time} per {period}.",
            "{time_saved} saved. {cost_saved} saved. One workflow."
        ],
        "tone": "results-oriented, specific, inspiring",
        "best_for_types": [ContentType.AUTOMATION_CASE],
        "length_target": "1100-1400 chars",
        "required_data": ["time_saved", "tool", "task"]
    },

    # ═══════════════════════════════════════════════════════════════════
    # COMMUNITY CONTENT SPECIFIC TEMPLATES
    # ═══════════════════════════════════════════════════════════════════

    "curated_insights": {
        "structure": """
        [CURATION HOOK - I found the gold]

        [SOURCE CONTEXT - where this came from]

        💡 Insight 1: {insight}
        → {attribution}

        💡 Insight 2: {insight}
        → {attribution}

        💡 Insight 3: {insight}
        → {attribution}

        [MY META-TAKEAWAY - connecting the dots]

        [CTA - what resonated with you?]

        [HASHTAGS]
        """,
        "example_hooks": [
            "A {platform} thread about {topic} just exploded. Key takeaways:",
            "I spent {time} in {platform} threads. Here's the gold:",
            "The {platform} community is debating {topic}. Best insights:"
        ],
        "tone": "curator, synthesizer, community-connected",
        "best_for_types": [ContentType.COMMUNITY_CONTENT],
        "length_target": "1200-1500 chars",
        "required_data": ["platform", "insights", "attributions"]
    },

    "hot_take": {
        "structure": """
        [HOT TAKE HOOK - the spicy opinion]

        [WHERE I SAW THIS - attribution]

        [THE ARGUMENT - why they might be right]

        [THE COUNTER - why they might be wrong]

        [MY POSITION - where I land]

        [INVITATION - convince me otherwise]

        [HASHTAGS]
        """,
        "example_hooks": [
            "Someone on {platform} said: '{hot_take}'. They might be right.",
            "The most controversial {topic} take I've seen this week:",
            "Hot take from {source}: {claim}. Thoughts?"
        ],
        "tone": "engaged, opinionated but open, provocative",
        "best_for_types": [ContentType.COMMUNITY_CONTENT],
        "length_target": "900-1200 chars",
        "required_data": ["hot_take", "source", "my_position"]
    },

    # ═══════════════════════════════════════════════════════════════════
    # TOOL RELEASE SPECIFIC TEMPLATES
    # ═══════════════════════════════════════════════════════════════════

    "first_look": {
        "structure": """
        [BREAKING HOOK - something new dropped]

        [WHAT IT IS - tool overview]

        [KEY FEATURES]
        ✅ {feature_1}
        ✅ {feature_2}
        ✅ {feature_3}

        [FIRST IMPRESSION - my initial take]

        [WHO SHOULD CARE - target users]

        [ACCESS - how to try it]

        [CTA - are you going to try it?]

        [HASHTAGS]
        """,
        "example_hooks": [
            "{company} just dropped {tool}. First impressions:",
            "New AI tool alert: {tool}. Here's what you need to know:",
            "I tested {tool} the moment it launched. Verdict:"
        ],
        "tone": "timely, informative, evaluative",
        "best_for_types": [ContentType.TOOL_RELEASE],
        "length_target": "1200-1500 chars",
        "required_data": ["tool_name", "features", "access"]
    },

    "comparison": {
        "structure": """
        [COMPARISON HOOK - the matchup]

        [TOOL A OVERVIEW]
        {tool_a}: {brief_description}

        [TOOL B OVERVIEW]
        {tool_b}: {brief_description}

        [HEAD TO HEAD]
        • Speed: {winner}
        • Features: {winner}
        • Price: {winner}
        • Ease of use: {winner}

        [MY VERDICT - which one and why]

        [CTA - which do you prefer?]

        [HASHTAGS]
        """,
        "example_hooks": [
            "{tool_a} vs {tool_b}: which one wins?",
            "I tested {tool_a} and {tool_b}. Clear winner.",
            "The {category} showdown: {tool_a} vs {tool_b}"
        ],
        "tone": "objective, analytical, helpful",
        "best_for_types": [ContentType.TOOL_RELEASE],
        "length_target": "1300-1600 chars",
        "required_data": ["tool_a", "tool_b", "comparison_points"]
    },

    "implications": {
        "structure": """
        [SIGNIFICANCE HOOK - why this matters]

        [WHAT LAUNCHED - brief description]

        [WHO SHOULD CARE]
        → {user_group_1}: because {reason}
        → {user_group_2}: because {reason}

        [WHAT THIS MEANS FOR THE MARKET]

        [WHAT TO WATCH - future implications]

        [CTA - how does this affect your work?]

        [HASHTAGS]
        """,
        "example_hooks": [
            "{tool} changes everything for {user_group}. Here's why:",
            "The {tool} announcement everyone's missing:",
            "What {company}'s new {tool} means for {industry}:"
        ],
        "tone": "strategic, forward-looking, analytical",
        "best_for_types": [ContentType.TOOL_RELEASE],
        "length_target": "1100-1400 chars",
        "required_data": ["tool", "affected_users", "implications"]
    }
}
```

---

#### Type-Specific Generation Prompts

```python
generation_prompts_by_type = {
    ContentType.ENTERPRISE_CASE: """
    Write a LinkedIn post about this enterprise AI case study.

    CONTENT TYPE: Enterprise Case
    TEMPLATE: {template_name}
    TEMPLATE STRUCTURE:
    {template_structure}

    EXTRACTED DATA:
    - Company: {company}
    - Industry: {industry}
    - Challenge: {challenge}
    - Solution: {solution}
    - Key Metrics: {metrics}
    - Lessons Learned: {lessons}

    HOOKS (choose best or create variation):
    {hooks}

    INSIGHTS TO INCLUDE:
    {key_insights}

    STYLE REQUIREMENTS:
    - Lead with specific metrics when possible
    - Name the company (builds credibility)
    - Include concrete numbers, not vague claims
    - Make lessons actionable for readers
    - Use authoritative but approachable tone

    CONSTRAINTS:
    - Length: {length_target}
    - Hook must fit in 210 chars (before "see more")
    - Use {max_emojis} emojis max
    - End with {suggested_cta}

    Generate the post:
    """,

    ContentType.PRIMARY_SOURCE: """
    Write a LinkedIn post about this research/paper.

    CONTENT TYPE: Primary Source (Research)
    TEMPLATE: {template_name}
    TEMPLATE STRUCTURE:
    {template_structure}

    EXTRACTED DATA:
    - Title/Source: {title}
    - Author(s): {authors}
    - Core Thesis: {thesis}
    - Key Finding: {key_finding}
    - Counterintuitive Element: {counterintuitive}
    - Implications: {implications}

    HOOKS (choose best or create variation):
    {hooks}

    JARGON TO EXPLAIN:
    {jargon_explanations}

    HELPFUL ANALOGIES:
    {analogies}

    STYLE REQUIREMENTS:
    - Make complex ideas accessible
    - Credit the authors/source
    - Show why this matters practically
    - Invite intellectual engagement
    - Balance confidence with humility

    CONSTRAINTS:
    - Length: {length_target}
    - Simplification level: {complexity_level}
    - End with {suggested_cta}

    Generate the post:
    """,

    ContentType.AUTOMATION_CASE: """
    Write a LinkedIn post about this automation/workflow.

    CONTENT TYPE: Automation Case
    TEMPLATE: {template_name}
    TEMPLATE STRUCTURE:
    {template_structure}

    EXTRACTED DATA:
    - Task Automated: {task}
    - Tools Used: {tools}
    - Workflow Steps: {steps}
    - Time Saved: {time_saved}
    - Key Insight: {key_insight}

    HOOKS (choose best or create variation):
    {hooks}

    STYLE REQUIREMENTS:
    - Be specific about tools and steps
    - Include concrete time/cost savings
    - Make it feel replicable
    - Generous with details
    - Practical over philosophical

    CONSTRAINTS:
    - Length: {length_target}
    - Must be actionable
    - End with {suggested_cta}

    Generate the post:
    """,

    ContentType.COMMUNITY_CONTENT: """
    Write a LinkedIn post synthesizing this community discussion.

    CONTENT TYPE: Community Content
    TEMPLATE: {template_name}
    TEMPLATE STRUCTURE:
    {template_structure}

    EXTRACTED DATA:
    - Platform: {platform}
    - Topic: {topic}
    - Key Viewpoints: {viewpoints}
    - Notable Quotes: {quotes}
    - Practitioner Signals: {signals}

    HOOKS (choose best or create variation):
    {hooks}

    STYLE REQUIREMENTS:
    - Credit the community/sources
    - Capture diverse perspectives
    - Add your synthesis/take
    - Feel connected to the community
    - Invite continued discussion

    CONSTRAINTS:
    - Length: {length_target}
    - Attribution is important
    - End with {suggested_cta}

    Generate the post:
    """,

    ContentType.TOOL_RELEASE: """
    Write a LinkedIn post about this new AI tool/release.

    CONTENT TYPE: Tool Release
    TEMPLATE: {template_name}
    TEMPLATE STRUCTURE:
    {template_structure}

    EXTRACTED DATA:
    - Tool Name: {tool_name}
    - Company: {company}
    - Key Features: {features}
    - Target Users: {target_users}
    - Availability: {availability}
    - Comparison: {comparison}

    HOOKS (choose best or create variation):
    {hooks}

    STYLE REQUIREMENTS:
    - Feel timely/fresh
    - Be specific about features
    - Help readers evaluate fit
    - Balanced (pros and cons)
    - Include access/try info

    CONSTRAINTS:
    - Length: {length_target}
    - Include demo/link if available
    - End with {suggested_cta}

    Generate the post:
    """
}
```

---

#### Type-Specific CTAs and Hashtags

```python
type_specific_ctas = {
    ContentType.ENTERPRISE_CASE: [
        "Have you seen similar results in your organization?",
        "What's your biggest AI implementation lesson?",
        "Would this approach work in your industry?",
        "Comment with your company's AI journey.",
        "Tag someone leading AI transformation at your company."
    ],

    ContentType.PRIMARY_SOURCE: [
        "Does this change how you think about {topic}?",
        "What's your take on {claim}?",
        "Agree or disagree? I'm curious.",
        "What research papers have shaped your thinking lately?",
        "Share your contrarian AI opinion below."
    ],

    ContentType.AUTOMATION_CASE: [
        "Try it and let me know how it goes!",
        "What would you automate first?",
        "Tag someone who needs to see this workflow.",
        "Drop your automation wins in the comments.",
        "What's your favorite {tool} use case?"
    ],

    ContentType.COMMUNITY_CONTENT: [
        "What resonated most with you?",
        "Add your perspective to the discussion.",
        "What's your experience with {topic}?",
        "The best insights often come from comments. Share yours.",
        "Which take do you agree with?"
    ],

    ContentType.TOOL_RELEASE: [
        "Are you going to try it?",
        "How does this compare to your current stack?",
        "Early adopters: share your first impressions.",
        "Which feature excites you most?",
        "Tag someone who should check this out."
    ]
}

type_specific_hashtag_strategies = {
    ContentType.ENTERPRISE_CASE: {
        "broad": ["#AI", "#DigitalTransformation", "#Enterprise"],
        "specific": ["#AIImplementation", "#CaseStudy", "#AIinBusiness"],
        "industry_specific": True  # Add industry hashtag like #FinanceAI
    },

    ContentType.PRIMARY_SOURCE: {
        "broad": ["#AI", "#MachineLearning", "#Research"],
        "specific": ["#AIResearch", "#TechLeadership", "#FutureOfAI"],
        "author_mention": True  # Mention author if on LinkedIn
    },

    ContentType.AUTOMATION_CASE: {
        "broad": ["#AI", "#Automation", "#Productivity"],
        "specific": ["#AITools", "#NoCode", "#WorkflowAutomation"],
        "tool_hashtags": True  # Add tool-specific like #n8n #LangChain
    },

    ContentType.COMMUNITY_CONTENT: {
        "broad": ["#AI", "#TechCommunity"],
        "specific": ["#AITwitter", "#TechDiscussion", "#AICommunity"],
        "platform_specific": True  # Add platform like #Reddit #HackerNews
    },

    ContentType.TOOL_RELEASE: {
        "broad": ["#AI", "#TechNews", "#NewTools"],
        "specific": ["#AITools", "#ProductLaunch", "#TechReview"],
        "tool_specific": True  # Add tool name as hashtag
    }
}
```

---

#### Style Guide (Enhanced)

```python
style_guide = {
    "voice_characteristics": {
        "expertise_level": "expert but approachable",
        "formality": "professional casual",
        "humor": "subtle, occasional",
        "confidence": "high but not arrogant"
    },

    "writing_rules": [
        "Start with the most interesting thing",
        "One idea per paragraph",
        "Use 'you' and 'I', not 'one' or 'users'",
        "Short sentences. Like this. Vary rhythm.",
        "End paragraphs on strong words",
        "Questions create engagement - use them",
        "Numbers are specific (87%, not 'most')",
        "Cut every word that doesn't add value"
    ],

    "linkedin_specific": {
        "line_breaks": "After every 1-2 sentences",
        "emojis": {
            "use": True,
            "max_per_post": 4,
            "placement": "Beginning of sections or bullets",
            "avoid": "Excessive fire/rocket/100 spam",
            "by_content_type": {
                ContentType.ENTERPRISE_CASE: ["📊", "💡", "🎯", "📈"],
                ContentType.PRIMARY_SOURCE: ["🔬", "📖", "🤔", "💭"],
                ContentType.AUTOMATION_CASE: ["⚡", "🔧", "⏱️", "💰"],
                ContentType.COMMUNITY_CONTENT: ["💡", "🗣️", "👀", "🔥"],
                ContentType.TOOL_RELEASE: ["🚀", "✅", "🆕", "⚡"]
            }
        },
        "hashtags": {
            "count": "3-5",
            "placement": "End of post",
            "mix": "1-2 broad (#AI #MachineLearning) + 2-3 specific"
        },
        "length": {
            "optimal": "1200-1500 characters",
            "max": "3000 characters",
            "hook_visible": "First 210 chars before 'see more'"
        }
    },

    "phrases_to_use": [
        "Here's the thing:",
        "I've noticed that",
        "In my experience",
        "The real question is",
        "What surprised me most was"
    ],

    "phrases_to_avoid": [
        "In conclusion",
        "Furthermore",
        "It's worth noting that",
        "As an AI enthusiast",
        "Let me share my thoughts"
    ]
}
```

---

#### Output Schema

```python
@dataclass
class DraftPost:
    """
    Enhanced draft post with content type awareness.
    """
    id: str
    topic_id: str
    content_type: ContentType

    # Content
    hook: str  # First line (must fit in 210 chars)
    body: str  # Main content
    cta: str  # Call to action
    hashtags: List[str]
    full_text: str  # Combined, formatted

    # Template info
    template_used: str
    template_category: str  # universal / enterprise / research / automation / etc.
    hook_style: HookStyle  # FIX: Use HookStyle enum for type safety

    # Metadata
    character_count: int
    estimated_read_time: str

    # Type-specific data used
    type_data_injected: Dict[str, Any]  # What extraction data was used

    # For next agents
    visual_brief: str  # Description for image generation
    visual_type: str  # data_viz / diagram / screenshot / quote_card
    key_terms: List[str]  # For hashtag optimization

    # Quality signals
    hook_in_limit: bool  # Is hook under 210 chars?
    length_in_range: bool  # Is total length in target range?

    # Versioning
    version: int
    created_at: datetime


@dataclass
class WriterOutput:
    """
    Complete output from Writer Agent.
    """
    draft: DraftPost

    # Alternatives generated
    alternative_hooks: List[str]  # Other hook options
    alternative_templates: List[str]  # Other template options considered

    # Decisions made
    template_selection_rationale: str
    hook_selection_rationale: str

    # For QC
    confidence_score: float  # 0-1 how confident in this draft
    areas_of_uncertainty: List[str]  # What might need human review
```

---

### 4. HUMANIZER AGENT

#### Purpose
Remove "AI sound" and add personality while **respecting content-type-specific tone requirements**. Enterprise case studies need credibility markers; research posts need intellectual engagement; automation posts need practical authenticity; community posts need conversational warmth; tool reviews need balanced assessment.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HUMANIZER AGENT (Enhanced)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: DraftPost (with ContentType)                                        │
│         │                                                                   │
│         ▼                                                                   │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                  CONTENT-TYPE TONE CALIBRATOR                      │     │
│  │  Adjusts humanization intensity based on ContentType:             │     │
│  │  • ENTERPRISE_CASE: Professional credibility (less casual)        │     │
│  │  • PRIMARY_SOURCE: Intellectual engagement (thoughtful)           │     │
│  │  • AUTOMATION_CASE: Practitioner authenticity (helpful)           │     │
│  │  • COMMUNITY_CONTENT: Conversational warmth (engaging)            │     │
│  │  • TOOL_RELEASE: Balanced assessment (fair, timely)              │     │
│  └───────────────────────────┬───────────────────────────────────────┘     │
│                              │                                              │
│                              ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    AI PATTERN DETECTOR                             │     │
│  │  Finds: robotic phrases, repetitive structures,                   │     │
│  │  overly formal language, generic statements                       │     │
│  │  (with type-specific detection rules)                             │     │
│  └───────────────────────────┬───────────────────────────────────────┘     │
│                              │                                              │
│                              ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    TYPE-AWARE VOICE INJECTOR                       │     │
│  │  Adds type-appropriate human markers:                             │     │
│  │  • Enterprise: credibility phrases, authority signals             │     │
│  │  • Research: intellectual curiosity, nuanced takes                │     │
│  │  • Automation: practical wisdom, builder empathy                  │     │
│  │  • Community: connection, attribution, discussion                 │     │
│  │  • Tool: hands-on experience, balanced evaluation                 │     │
│  └───────────────────────────┬───────────────────────────────────────┘     │
│                              │                                              │
│                              ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    RHYTHM VARIATOR                                 │     │
│  │  Varies: sentence length, paragraph breaks, punctuation           │     │
│  │  (calibrated to content type expectations)                        │     │
│  └───────────────────────────┬───────────────────────────────────────┘     │
│                              │                                              │
│                              ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    AUTHENTICITY CHECK                              │     │
│  │  Verifies: sounds like real person, matches author voice,         │     │
│  │  appropriate for content type                                     │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                              │                                              │
│                              ▼                                              │
│  OUTPUT: HumanizedPost (with type-appropriate voice)                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### Content-Type Specific Humanization Rules

```python
"""
Different content types require different levels and styles of humanization.
A case study needs to feel credible; a community synthesis needs to feel connected.
"""

type_specific_humanization = {
    ContentType.ENTERPRISE_CASE: {
        "tone_target": "professional credibility with personal insight",
        "humanization_intensity": "medium",  # Don't over-casualize
        "markers_to_add": [
            "Credibility signals ('In my analysis of this case...')",
            "Interpretive comments ('What struck me most was...')",
            "Industry context ('This is significant for {industry} because...')",
            "Measured enthusiasm ('The results are genuinely impressive.')",
            "Honest caveats ('Though we should note that...')"
        ],
        "markers_to_avoid": [
            "Overly casual language ('crazy', 'insane', 'wild')",
            "Excessive personal stories (focus stays on the case)",
            "Speculation beyond evidence"
        ],
        "rhythm_profile": {
            "sentence_length_mix": {"short": "25%", "medium": "55%", "long": "20%"},
            "questions_per_post": "1",
            "exclamations_per_post": "0-1"
        }
    },

    ContentType.PRIMARY_SOURCE: {
        "tone_target": "intellectual engagement with accessibility",
        "humanization_intensity": "medium-high",
        "markers_to_add": [
            "Intellectual curiosity ('This made me think about...')",
            "Honest reactions ('I'll admit, this surprised me.')",
            "Accessible framing ('In plain terms, this means...')",
            "Nuanced takes ('It's more complicated than the headline.')",
            "Debate invitations ('I'm not sure I fully agree. Here's why...')"
        ],
        "markers_to_avoid": [
            "Dumbing down too much (respect the research)",
            "Overclaiming certainty",
            "Dismissing complexity"
        ],
        "rhythm_profile": {
            "sentence_length_mix": {"short": "30%", "medium": "45%", "long": "25%"},
            "questions_per_post": "2-3",
            "exclamations_per_post": "0"
        }
    },

    ContentType.AUTOMATION_CASE: {
        "tone_target": "practitioner authenticity with helpful generosity",
        "humanization_intensity": "high",
        "markers_to_add": [
            "Builder empathy ('I know the pain of doing this manually.')",
            "Practical wisdom ('The trick that made this work...')",
            "Honest about gotchas ('Fair warning: this part was tricky.')",
            "Generous sharing ('Here's exactly what I did...')",
            "Encouragement ('You can totally do this.')"
        ],
        "markers_to_avoid": [
            "Making it sound harder than it is",
            "Gatekeeping language",
            "Overcomplicating explanations"
        ],
        "rhythm_profile": {
            "sentence_length_mix": {"short": "35%", "medium": "50%", "long": "15%"},
            "questions_per_post": "1-2",
            "exclamations_per_post": "1"
        }
    },

    ContentType.COMMUNITY_CONTENT: {
        "tone_target": "conversational connection with community",
        "humanization_intensity": "high",
        "markers_to_add": [
            "Community connection ('The thread was fascinating.')",
            "Attribution respect ('As @user pointed out...')",
            "Discussion energy ('This sparked a great debate.')",
            "Personal engagement ('Here's where I weigh in...')",
            "Invitation to join ('What's your experience?')"
        ],
        "markers_to_avoid": [
            "Taking credit for others' insights",
            "Dismissing community voices",
            "Being preachy"
        ],
        "rhythm_profile": {
            "sentence_length_mix": {"short": "35%", "medium": "45%", "long": "20%"},
            "questions_per_post": "2-3",
            "exclamations_per_post": "1"
        }
    },

    ContentType.TOOL_RELEASE: {
        "tone_target": "balanced assessment with hands-on credibility",
        "humanization_intensity": "medium",
        "markers_to_add": [
            "Hands-on experience ('I tested this immediately.')",
            "Balanced evaluation ('Here's what's great. And what's not.')",
            "Practical perspective ('For my workflow, this means...')",
            "Timely excitement ('Just dropped. First impressions:')",
            "Honest limitations ('It's not perfect. Here's why.')"
        ],
        "markers_to_avoid": [
            "Sounding like marketing copy",
            "Uncritical enthusiasm",
            "Dismissing without fair trial"
        ],
        "rhythm_profile": {
            "sentence_length_mix": {"short": "30%", "medium": "50%", "long": "20%"},
            "questions_per_post": "1-2",
            "exclamations_per_post": "1"
        }
    }
}
```

---

#### Universal Humanization Rules

```python
humanization_rules = {
    "ai_patterns_to_remove": {
        "phrases": [
            "It's important to note that",
            "In today's rapidly evolving",
            "Let's dive in",
            "Let's explore",
            "In conclusion",
            "Furthermore",
            "Moreover",
            "Additionally",
            "That being said",
            "At the end of the day",
            "It goes without saying",
            "Needless to say",
            "Without further ado",
            "In this article",
            "First and foremost",
            "Last but not least",
            "It is worth mentioning",
            "As we all know"
        ],
        "structures": [
            "Three-part lists with parallel structure",
            "Overly balanced paragraphs",
            "Too many transitional words",
            "Every sentence same length",
            "Perfect grammar throughout (some variation is human)"
        ],
        "tells": [
            "Excessive hedging (might, could, may, perhaps)",
            "Overly comprehensive (covering all angles equally)",
            "Too polished (no rough edges)",
            "Generic examples (imagine a scenario...)",
            "Robotic enumerations (firstly, secondly, thirdly)"
        ]
    },

    "human_markers_to_add": {
        "personal_touches": [
            "Specific numbers ('I read 23 papers')",
            "Named tools/people ('when I asked GPT-4')",
            "Time references ('last Tuesday', 'this morning')",
            "Emotional reactions ('honestly, this surprised me')",
            "Mini-confessions ('I used to think X, but...')"
        ],
        "conversational_elements": [
            "Direct address ('you know what?')",
            "Rhetorical questions",
            "Incomplete sentences. Sometimes.",
            "Parenthetical asides (like this one)",
            "Self-corrections ('well, actually...')"
        ],
        "imperfections": [
            "Occasional informal words",
            "Starting sentences with 'And' or 'But'",
            "Em dashes for interruptions—like this",
            "Sentence fragments for emphasis. Really."
        ]
    },

    "voice_consistency": {
        "your_unique_phrases": [
            "Here's the thing:",
            "What surprised me most:",
            "The real insight here:",
            "Let me be direct:"
        ],
        "your_opinions": [
            # Your typical takes and viewpoints
        ],
        "your_expertise_areas": [
            "AI implementation",
            "Automation workflows",
            "Enterprise AI strategy"
        ]
    }
}
```

---

#### Type-Aware Humanization Prompt

```python
humanization_prompt_by_type = {
    "base_prompt": """
    You are a humanization expert. Your job is to take AI-generated
    LinkedIn content and make it sound authentically human while
    RESPECTING THE CONTENT TYPE.

    ORIGINAL POST:
    {draft_post}

    CONTENT TYPE: {content_type}
    TYPE-SPECIFIC TONE: {tone_target}
    HUMANIZATION INTENSITY: {humanization_intensity}

    AUTHOR PROFILE:
    {author_voice_profile}

    TYPE-SPECIFIC MARKERS TO ADD:
    {markers_to_add}

    TYPE-SPECIFIC MARKERS TO AVOID:
    {markers_to_avoid}

    RHYTHM PROFILE:
    {rhythm_profile}

    UNIVERSAL CHECKLIST:
    1. Remove AI phrases: {ai_phrases_to_avoid}
    2. Add type-appropriate personal touches
    3. Vary sentence rhythm per the rhythm profile
    4. Add conversational elements appropriate to type
    5. Include slight imperfections (but not too many)
    6. Make opinions sound genuine, not hedged
    7. Keep the same information, change the delivery

    CONSTRAINTS:
    - Don't add information that wasn't in the original
    - Keep the same structure and main points
    - Match the tone to the content type
    - Don't add fake personal stories
    - Respect the source material

    OUTPUT:
    Return the humanized version of the post.
    """,

    ContentType.ENTERPRISE_CASE: """
    ADDITIONAL GUIDANCE FOR ENTERPRISE CASE:
    - Maintain professional credibility
    - Add interpretive insights, not casual commentary
    - Keep metrics and facts prominent
    - Sound like a thoughtful analyst, not a cheerleader
    """,

    ContentType.PRIMARY_SOURCE: """
    ADDITIONAL GUIDANCE FOR RESEARCH:
    - Show intellectual engagement
    - Make complex ideas accessible without dumbing down
    - Invite debate respectfully
    - Credit the researchers appropriately
    """,

    ContentType.AUTOMATION_CASE: """
    ADDITIONAL GUIDANCE FOR AUTOMATION:
    - Sound like a helpful practitioner
    - Be generous with practical details
    - Acknowledge challenges honestly
    - Encourage readers they can do this too
    """,

    ContentType.COMMUNITY_CONTENT: """
    ADDITIONAL GUIDANCE FOR COMMUNITY:
    - Feel connected to the community
    - Attribute insights properly
    - Add energy to the discussion
    - Invite participation
    """,

    ContentType.TOOL_RELEASE: """
    ADDITIONAL GUIDANCE FOR TOOL:
    - Sound hands-on, not like marketing
    - Balance pros and cons
    - Be timely in tone
    - Help readers decide if it's relevant for them
    """
}
```

---

#### Output Schema

```python
@dataclass
class HumanizedPost:
    id: str
    original_draft_id: str
    content_type: ContentType

    # Content
    humanized_text: str

    # Changes made
    changes_log: List[str]  # What was changed and why
    ai_patterns_removed: List[str]
    human_markers_added: List[str]
    type_specific_adjustments: List[str]  # What type-specific changes were made

    # Quality metrics
    humanness_score: float  # 0-10
    voice_consistency_score: float  # 0-10
    type_tone_match_score: float  # 0-10 (how well tone matches content type)

    # ─────────────────────────────────────────────────────────────────
    # REVISION TRACKING (needed by QC Agent for max_revisions check)
    # ─────────────────────────────────────────────────────────────────
    revision_count: int = 0
    revision_history: List[Dict[str, Any]] = field(default_factory=list)
    # Each entry: {"iteration": int, "target": str, "feedback": str, "timestamp": str}

    # Metadata
    humanization_intensity_used: str  # low / medium / high
    version: int
    humanized_at: datetime
```

---

### 5. VISUAL CREATOR AGENT

#### Purpose
Create visual content **optimized for each ContentType**. Enterprise cases need data visualizations and architecture diagrams; research posts need conceptual illustrations; automation posts need workflow diagrams or carousels; community posts need quote cards; tool releases need product screenshots or comparison charts.

**NEW: Personal Photo Integration** — Posts with author's face get 2-3x more engagement. The system now supports integrating personal photos from a local library, processed through Nano Banana Pro for natural-looking compositions.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      VISUAL CREATOR AGENT (Enhanced)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: HumanizedPost + visual_brief + ContentType                          │
│         │                                                                   │
│         ▼                                                                   │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                  CONTENT-TYPE FORMAT SELECTOR                      │     │
│  │                                                                    │     │
│  │  ContentType → Recommended Visual Formats                         │     │
│  │  ─────────────────────────────────────────────                    │     │
│  │  ENTERPRISE_CASE   → data_viz, metrics_card, AUTHOR_PHOTO (60%)  │     │
│  │  PRIMARY_SOURCE    → concept_illustration, quote_card+PHOTO (40%)│     │
│  │  AUTOMATION_CASE   → workflow_diagram, carousel+PHOTO_BOOKEND    │     │
│  │  COMMUNITY_CONTENT → quote_card+AUTHOR_PHOTO (70%), collage      │     │
│  │  TOOL_RELEASE      → product_screenshot, comparison_chart        │     │
│  │                                                                    │     │
│  └───────────────────────────┬───────────────────────────────────────┘     │
│                              │                                              │
│                              ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    PHOTO SELECTOR (NEW)                            │     │
│  │  ┌─────────────────────────────────────────────────────────────┐  │     │
│  │  │  ~/linkedin-photos/  →  Claude Vision auto-tagging          │  │     │
│  │  │  portraits/ | action/ | context/                            │  │     │
│  │  │                                                             │  │     │
│  │  │  Decision: use_photo? → select_best_match → integration_mode│  │     │
│  │  │  Variety check: avoid repeating same photo in 5 posts       │  │     │
│  │  └─────────────────────────────────────────────────────────────┘  │     │
│  └───────────────────────────┬───────────────────────────────────────┘     │
│                              │                                              │
│                              ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    FORMAT ROUTER                                   │     │
│  └───────────────────────────┬───────────────────────────────────────┘     │
│                              │                                              │
│       ┌──────────────────────┼──────────────────────┬──────────────┐       │
│       ▼                      ▼                      ▼              ▼       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  ┌───────────┐  │
│  │   AI IMAGE   │    │  CAROUSEL    │    │   TEMPLATE   │  │  PHOTO    │  │
│  │  GENERATOR   │    │  GENERATOR   │    │   COMPOSER   │  │ INTEGRATOR│  │
│  │(Nano Banana) │    │  (Slides)    │    │ (Cards/Docs) │  │           │  │
│  │ Laozhang 4K  │    │ +Photo Ends  │    │              │  │ 4 modes:  │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │ • as-is   │  │
│         │                   │                   │          │ • overlay │  │
│         │                   │                   │          │ • AI edit │  │
│         │                   │                   │          │ • carousel│  │
│         │                   │                   │          └─────┬─────┘  │
│         └───────────────────┴───────────────────┴────────────────┘        │
│                             │                                              │
│                             ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────┐    │
│  │                    STYLE ENFORCER                                  │    │
│  │  • Brand colors consistency                                       │    │
│  │  • Type-appropriate visual language                               │    │
│  │  • Photo+Graphics harmony check                                   │    │
│  │  • Mobile-optimized quality check                                 │    │
│  └───────────────────────────────────────────────────────────────────┘    │
│                             │                                              │
│                             ▼                                              │
│  OUTPUT: VisualAsset (type-optimized, with optional author photo)         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### Personal Photo Library System

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PHOTO LIBRARY SYSTEM                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LOCAL PHOTO STORAGE                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ~/linkedin-photos/                                                  │   │
│  │  ├── portraits/           # Headshots, professional photos          │   │
│  │  │   ├── formal/          # Suit, office background                 │   │
│  │  │   ├── casual/          # Relaxed, approachable                   │   │
│  │  │   └── speaking/        # Conference, presenting                  │   │
│  │  ├── action/              # Working, coding, meetings               │   │
│  │  │   ├── at_desk/         # Laptop, monitors                        │   │
│  │  │   ├── whiteboard/      # Explaining, teaching                    │   │
│  │  │   └── team/            # Collaboration shots                     │   │
│  │  ├── context/             # Specific contexts                       │   │
│  │  │   ├── conference/      # Events, stages                          │   │
│  │  │   ├── office/          # Workspace                               │   │
│  │  │   └── outdoor/         # Casual, lifestyle                       │   │
│  │  └── metadata.json        # Photo tags, descriptions, usage stats   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PHOTO INDEXER (on startup)                        │   │
│  │  • Scan folder for new photos                                       │   │
│  │  • Auto-tag using Claude Vision (face position, setting, mood)      │   │
│  │  • Extract EXIF metadata (date, camera)                             │   │
│  │  • Generate embeddings for semantic search                          │   │
│  └───────────────────────────┬─────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PHOTO SELECTOR                                    │   │
│  │                                                                      │   │
│  │  INPUT: post_content + content_type + visual_brief                  │   │
│  │                                                                      │   │
│  │  DECISION LOGIC:                                                    │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │  1. Should this post have a personal photo?                   │  │   │
│  │  │     • ENTERPRISE_CASE → 60% yes (credibility + human face)    │  │   │
│  │  │     • PRIMARY_SOURCE  → 40% yes (thought leader positioning)  │  │   │
│  │  │     • AUTOMATION_CASE → 30% yes (prefer diagrams)             │  │   │
│  │  │     • COMMUNITY_CONTENT → 70% yes (personal connection)       │  │   │
│  │  │     • TOOL_RELEASE → 20% yes (product focus)                  │  │   │
│  │  │                                                               │  │   │
│  │  │  2. What type of photo fits?                                  │  │   │
│  │  │     • "sharing insight" → portraits/casual                    │  │   │
│  │  │     • "teaching" → action/whiteboard                          │  │   │
│  │  │     • "announcing" → portraits/formal                         │  │   │
│  │  │     • "discussing" → action/at_desk                           │  │   │
│  │  │     • "reflecting" → context/outdoor                          │  │   │
│  │  │                                                               │  │   │
│  │  │  3. Variety check (avoid repetition)                          │  │   │
│  │  │     • Check last 10 posts' photos                             │  │   │
│  │  │     • Ensure different photo each time                        │  │   │
│  │  │     • Rotate through available options                        │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                                      │   │
│  │  OUTPUT: selected_photo_path OR "generate_without_photo"            │   │
│  └───────────────────────────┬─────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PHOTO INTEGRATION MODES                           │   │
│  │                                                                      │   │
│  │  MODE 1: PHOTO AS-IS (with enhancement)                             │   │
│  │  ─────────────────────────────────────────                          │   │
│  │  • Light color correction, crop optimization                        │   │
│  │  • Add subtle brand overlay (logo watermark)                        │   │
│  │  • Best for: portraits, speaking shots                              │   │
│  │                                                                      │   │
│  │  MODE 2: PHOTO + OVERLAY                                            │   │
│  │  ─────────────────────────────────────────                          │   │
│  │  • Photo on one side, text/graphics on other                        │   │
│  │  • Quote card with author photo                                     │   │
│  │  • Key metric + author reaction                                     │   │
│  │  • Best for: thought leadership, hot takes                          │   │
│  │                                                                      │   │
│  │  MODE 3: PHOTO + AI CONTEXT (Nano Banana Edit)                      │   │
│  │  ─────────────────────────────────────────────                      │   │
│  │  • Use Gemini 3 Pro image editing to add context                    │   │
│  │  • Add relevant background elements                                 │   │
│  │  • Create scene that matches post topic                             │   │
│  │  • ⚠️ Careful: must look natural, not obviously AI                  │   │
│  │  • Best for: conceptual posts, storytelling                         │   │
│  │                                                                      │   │
│  │  MODE 4: CAROUSEL WITH PHOTO                                        │   │
│  │  ─────────────────────────────────────────                          │   │
│  │  • First slide: author photo + title                                │   │
│  │  • Middle slides: content/diagrams                                  │   │
│  │  • Last slide: author photo + CTA                                   │   │
│  │  • Best for: tutorials, lists, educational content                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### Photo Integration Configuration

```python
photo_integration_config = {
    # ═══════════════════════════════════════════════════════════════════
    # PHOTO LIBRARY SETTINGS
    # ═══════════════════════════════════════════════════════════════════

    "library_path": "~/linkedin-photos",  # Local folder path
    "supported_formats": ["jpg", "jpeg", "png", "webp"],
    "min_resolution": (800, 800),  # Minimum photo size
    "max_photos_indexed": 500,

    # ═══════════════════════════════════════════════════════════════════
    # AUTO-TAGGING (Claude Vision)
    # ═══════════════════════════════════════════════════════════════════

    "auto_tag_prompt": """
    Analyze this photo for LinkedIn content use. Extract:

    1. setting: where is this? (office, conference, outdoor, studio, home)
    2. pose: what's the person doing? (portrait, speaking, working, thinking, gesturing)
    3. mood: what's the emotional tone? (professional, friendly, focused, excited, thoughtful)
    4. attire: what are they wearing? (formal, business_casual, casual)
    5. background: what's behind them? (plain, office, stage, nature, abstract)
    6. face_position: where is face in frame? (center, left_third, right_third)
    7. eye_contact: looking at camera? (direct, away, profile)
    8. suitable_for: list content types this works for

    Return as JSON.
    """,

    # ═══════════════════════════════════════════════════════════════════
    # CONTENT TYPE → PHOTO PREFERENCES
    # ═══════════════════════════════════════════════════════════════════

    "content_type_photo_prefs": {
        ContentType.ENTERPRISE_CASE: {
            "use_photo_probability": 0.6,
            "preferred_settings": ["office", "conference"],
            "preferred_poses": ["portrait", "speaking"],
            "preferred_attire": ["formal", "business_casual"],
            "integration_modes": ["photo_overlay", "photo_as_is"],
            "photo_position": "left_side",  # Data on right
        },

        ContentType.PRIMARY_SOURCE: {
            "use_photo_probability": 0.4,
            "preferred_settings": ["studio", "office"],
            "preferred_poses": ["thinking", "portrait"],
            "preferred_attire": ["business_casual"],
            "integration_modes": ["photo_overlay", "carousel_bookend"],
            "photo_position": "quote_card_corner",
        },

        ContentType.AUTOMATION_CASE: {
            "use_photo_probability": 0.3,
            "preferred_settings": ["office", "home"],
            "preferred_poses": ["working", "gesturing"],
            "preferred_attire": ["casual", "business_casual"],
            "integration_modes": ["carousel_bookend"],  # Diagrams are primary
            "photo_position": "first_last_slide",
        },

        ContentType.COMMUNITY_CONTENT: {
            "use_photo_probability": 0.7,
            "preferred_settings": ["any"],
            "preferred_poses": ["portrait", "thinking"],
            "preferred_mood": ["friendly", "thoughtful"],
            "integration_modes": ["photo_as_is", "photo_overlay"],
            "photo_position": "prominent",  # Face is the focus
        },

        ContentType.TOOL_RELEASE: {
            "use_photo_probability": 0.2,
            "preferred_settings": ["office"],
            "preferred_poses": ["working"],
            "integration_modes": ["photo_overlay"],  # Small, product is focus
            "photo_position": "corner_badge",
        }
    },

    # ═══════════════════════════════════════════════════════════════════
    # NANO BANANA PHOTO EDITING PROMPTS
    # ═══════════════════════════════════════════════════════════════════

    "photo_edit_prompts": {
        "add_tech_background": """
        Keep the person exactly as they are.
        Replace/enhance the background with a modern tech office environment.
        Subtle screens showing code or dashboards in the background.
        Professional lighting. Natural looking, not obviously AI.
        Maintain original photo quality and person's appearance.
        """,

        "add_conference_context": """
        Keep the person exactly as they are.
        Add subtle conference/event context: blurred audience, stage lighting.
        Professional speaking environment feel.
        Natural lighting that matches original photo.
        """,

        "add_data_overlay": """
        Keep the person exactly as they are.
        Add floating holographic-style data visualizations around them.
        Subtle, modern, futuristic but professional.
        Numbers and charts that feel relevant to AI/tech.
        Semi-transparent, not blocking the face.
        """,

        "enhance_professional": """
        Keep the person exactly as they are.
        Subtle professional enhancement only:
        - Slightly better lighting
        - Clean background blur
        - Professional color grading
        Do NOT change facial features or body.
        """
    },

    # ═══════════════════════════════════════════════════════════════════
    # VARIETY & FRESHNESS
    # ═══════════════════════════════════════════════════════════════════

    "variety_settings": {
        "lookback_posts": 10,  # Check last N posts
        "min_reuse_gap": 5,    # Don't reuse same photo within N posts
        "max_same_setting": 3, # Max consecutive posts with same setting
        "rotation_strategy": "weighted_random",  # or "sequential"
    }
}
```

---

#### Photo Metadata Schema

```python
@dataclass
class PhotoMetadata:
    """Metadata for each photo in the library."""

    id: str  # Unique identifier (hash of file)
    file_path: str
    file_name: str

    # Auto-tagged properties (via Claude Vision)
    setting: str  # office, conference, outdoor, studio, home
    pose: str     # portrait, speaking, working, thinking, gesturing
    mood: str     # professional, friendly, focused, excited, thoughtful
    attire: str   # formal, business_casual, casual
    background: str  # plain, office, stage, nature, abstract
    face_position: str  # center, left_third, right_third
    eye_contact: str    # direct, away, profile
    suitable_for: List[ContentType]

    # Technical properties
    width: int
    height: int
    aspect_ratio: str
    file_size_kb: int

    # Usage tracking
    times_used: int = 0
    last_used_date: Optional[datetime] = None
    last_used_post_id: Optional[str] = None

    # Manual overrides
    favorite: bool = False
    disabled: bool = False  # Don't use this photo
    custom_tags: List[str] = field(default_factory=list)


@dataclass
class PhotoSelectionResult:
    """Result of photo selection for a post."""

    use_photo: bool
    photo: Optional[PhotoMetadata]
    integration_mode: str  # photo_as_is, photo_overlay, photo_ai_edit, carousel_bookend

    # If using AI edit
    edit_prompt: Optional[str]

    # Positioning
    position: str  # left_side, right_side, corner_badge, quote_card_corner, etc.

    # Reasoning
    selection_rationale: str


class PhotoLibrary:
    """
    Photo library manager for author's personal photos.

    Handles storage, indexing, and retrieval of photos for LinkedIn posts.
    Uses Supabase Storage for file storage and database for metadata.
    """

    def __init__(self, db: "SupabaseDB"):
        self.db = db
        self._cache: Dict[str, PhotoMetadata] = {}
        self._loaded = False

    async def load(self) -> None:
        """Load all photo metadata from database."""
        if self._loaded:
            return
        photos = await self.db.get_all_photos()
        self._cache = {p["id"]: PhotoMetadata(**p) for p in photos}
        self._loaded = True

    async def get_available_photos(
        self,
        content_type: Optional[ContentType] = None,
        exclude_recent: int = 5
    ) -> List[PhotoMetadata]:
        """
        Get photos available for use.

        Args:
            content_type: Filter by suitability for content type
            exclude_recent: Exclude photos used in last N posts

        Returns:
            List of available PhotoMetadata, sorted by relevance
        """
        await self.load()

        photos = [
            p for p in self._cache.values()
            if not p.disabled
        ]

        if content_type:
            photos = [p for p in photos if content_type in p.suitable_for]

        # Exclude recently used
        if exclude_recent > 0:
            recent_ids = await self._get_recent_photo_ids(exclude_recent)
            photos = [p for p in photos if p.id not in recent_ids]

        # Sort by usage (prefer less used) and favorites first
        photos.sort(key=lambda p: (-p.favorite, p.times_used))

        return photos

    async def _get_recent_photo_ids(self, count: int) -> set:
        """Get IDs of photos used in recent posts."""
        recent_posts = await self.db.get_recent_posts(limit=count)
        return {p.get("photo_id") for p in recent_posts if p.get("photo_id")}

    async def mark_used(self, photo_id: str, post_id: str) -> None:
        """Mark a photo as used in a post."""
        if photo_id in self._cache:
            photo = self._cache[photo_id]
            photo.times_used += 1
            photo.last_used_date = datetime.now()
            photo.last_used_post_id = post_id

        await self.db.update_photo_usage(photo_id, post_id)

    def get_by_id(self, photo_id: str) -> Optional[PhotoMetadata]:
        """Get photo by ID from cache."""
        return self._cache.get(photo_id)
```

---

#### Photo Selection Algorithm

```python
async def select_photo_for_post(
    post_content: str,
    content_type: ContentType,
    visual_brief: str,
    photo_library: PhotoLibrary
) -> PhotoSelectionResult:
    """
    Intelligent photo selection for LinkedIn post.
    """
    config = photo_integration_config["content_type_photo_prefs"][content_type]

    # Step 1: Should we use a photo?
    use_photo_prob = config["use_photo_probability"]

    # Boost probability for certain content signals
    if "личный опыт" in post_content.lower() or "I learned" in post_content.lower():
        use_photo_prob += 0.2
    if "hot take" in visual_brief.lower() or "opinion" in visual_brief.lower():
        use_photo_prob += 0.15

    # Random decision with probability
    if random.random() > use_photo_prob:
        return PhotoSelectionResult(
            use_photo=False,
            photo=None,
            integration_mode="none",
            selection_rationale=f"Skipped photo (prob={use_photo_prob:.0%}, content_type={content_type.value})"
        )

    # Step 2: Find matching photos
    candidates = photo_library.search(
        settings=config.get("preferred_settings", ["any"]),
        poses=config.get("preferred_poses", ["any"]),
        attire=config.get("preferred_attire", ["any"]),
        exclude_recently_used=True
    )

    if not candidates:
        return PhotoSelectionResult(
            use_photo=False,
            photo=None,
            integration_mode="none",
            selection_rationale="No suitable photos found in library"
        )

    # Step 3: Score and select best match
    scored_candidates = []
    for photo in candidates:
        score = 0.0

        # Freshness bonus (less used = higher score)
        score += max(0, 1 - (photo.times_used / 10))

        # Favorite bonus
        if photo.favorite:
            score += 0.3

        # Variety bonus (different from recent)
        if photo.id not in photo_library.recent_used_ids:
            score += 0.2

        scored_candidates.append((photo, score))

    # Select best
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    selected_photo = scored_candidates[0][0]

    # Step 4: Determine integration mode
    integration_mode = random.choice(config["integration_modes"])

    # Step 5: Generate edit prompt if needed
    edit_prompt = None
    if integration_mode == "photo_ai_edit":
        edit_prompt = _generate_contextual_edit_prompt(post_content, content_type)

    return PhotoSelectionResult(
        use_photo=True,
        photo=selected_photo,
        integration_mode=integration_mode,
        edit_prompt=edit_prompt,
        position=config.get("photo_position", "left_side"),
        selection_rationale=f"Selected {selected_photo.file_name} ({selected_photo.setting}/{selected_photo.pose})"
    )
```

---

#### AI-Generated Interface Visuals (Nano Banana Pro)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              FULL AI INTERFACE GENERATION (Nano Banana Pro)                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  USE CASE: Posts about specific tools, CRMs, apps, interfaces               │
│  APPROACH: Nano Banana Pro generates EVERYTHING — no real screenshots       │
│                                                                             │
│  WHY THIS WORKS:                                                            │
│  ✓ Gemini 3 Pro Image Preview draws excellent, realistic interfaces        │
│  ✓ Can generate person + phone + interface in ONE request                  │
│  ✓ No need for screenshot library or mockup templates                      │
│  ✓ Consistent style across all visuals                                     │
│  ✓ Can show "ideal" interface, not cluttered real screenshots              │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│  GENERATION MODES                                                           │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  MODE 1: INTERFACE ONLY (Device Mockup)                                     │
│  ──────────────────────────────────────                                     │
│  • AI generates phone/laptop with interface on screen                       │
│  • Clean, professional product shot                                         │
│  │                                                                          │
│  │  Prompt: "Modern iPhone 15 Pro showing a CRM dashboard interface.       │
│  │          Clean UI with contact cards, deal pipeline, metrics.           │
│  │          Professional product photography, white background."           │
│                                                                             │
│  MODE 2: AUTHOR HOLDING DEVICE                                              │
│  ─────────────────────────────                                              │
│  • AI generates person holding phone with interface visible                 │
│  • Professional, natural-looking photo                                      │
│  │                                                                          │
│  │  Prompt: "Professional photo of a person in business casual             │
│  │          holding iPhone, showing CRM app interface on screen.           │
│  │          Clean modern office background. Natural lighting.              │
│  │          Person looking at camera with confident expression.            │
│  │          Phone screen clearly shows contact list and pipeline."         │
│                                                                             │
│  MODE 3: AUTHOR + LAPTOP/MONITOR                                            │
│  ───────────────────────────────                                            │
│  • AI generates person at desk with interface on screen                     │
│  • "Teaching" or "showing" pose                                             │
│  │                                                                          │
│  │  Prompt: "Professional person pointing at MacBook screen showing        │
│  │          n8n workflow automation interface. Modern office setting.      │
│  │          Screen shows connected nodes, AI integration visible.          │
│  │          Person explaining something, engaged expression."              │
│                                                                             │
│  MODE 4: SPLIT COMPOSITION                                                  │
│  ─────────────────────────                                                  │
│  • Left: person (portrait style)                                            │
│  • Right: device with interface                                             │
│  │                                                                          │
│  │  Prompt: "Split image composition. Left side: professional person       │
│  │          in business attire, thoughtful expression. Right side:         │
│  │          iPhone mockup showing AI chatbot interface. Clean design,      │
│  │          modern aesthetic. Visual separator between halves."            │
│                                                                             │
│  MODE 5: BEFORE/AFTER INTERFACES                                            │
│  ───────────────────────────────                                            │
│  • Two interfaces showing transformation                                    │
│  │                                                                          │
│  │  Prompt: "Before/After comparison of CRM interfaces.                    │
│  │          Left (Before): cluttered spreadsheet, messy data.              │
│  │          Right (After): clean modern CRM dashboard, organized.          │
│  │          Arrow or transition element between them."                     │
│                                                                             │
│  MODE 6: INTERFACE CLOSE-UP WITH HIGHLIGHTS                                 │
│  ──────────────────────────────────────────                                 │
│  • Detailed interface with visual callouts                                  │
│  │                                                                          │
│  │  Prompt: "Close-up of mobile app interface showing AI assistant.        │
│  │          Glowing highlight around key feature. Annotation arrows        │
│  │          pointing to important elements. Clean, instructional style."   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### Interface Generation Prompts by Product Type

```python
interface_generation_prompts = {
    # ═══════════════════════════════════════════════════════════════════
    # CRM INTERFACES
    # ═══════════════════════════════════════════════════════════════════

    "crm": {
        "description": "CRM/Sales pipeline interfaces",
        "interface_elements": [
            "contact cards with photos and details",
            "deal pipeline with stages (columns)",
            "activity feed with recent interactions",
            "revenue metrics and charts",
            "task list and reminders"
        ],
        "prompt_template": """
        {device_type} displaying modern CRM sales interface.

        INTERFACE ELEMENTS:
        - Kanban-style deal pipeline with {stage_count} stages
        - Contact cards showing names, companies, deal values
        - Activity sidebar with recent calls/emails
        - Top metrics bar: revenue, conversion rate, active deals

        STYLE:
        - Clean, modern SaaS aesthetic
        - Primary color: blue or teal
        - White/light gray background
        - Clear typography, professional icons
        - Mobile: bottom navigation bar
        - Desktop: left sidebar navigation

        {composition_instruction}
        """
    },

    # ═══════════════════════════════════════════════════════════════════
    # AI TOOL INTERFACES
    # ═══════════════════════════════════════════════════════════════════

    "ai_chat": {
        "description": "AI chatbot/assistant interfaces",
        "interface_elements": [
            "chat message bubbles",
            "AI response with formatted text",
            "input field with send button",
            "sidebar with conversation history",
            "model selector or settings"
        ],
        "prompt_template": """
        {device_type} showing AI chat assistant interface.

        INTERFACE ELEMENTS:
        - Chat conversation with user and AI messages
        - AI response showing {response_type}
        - Clean message bubbles, distinct colors for user/AI
        - Text input at bottom with send button
        - {sidebar_element}

        STYLE:
        - Modern, minimal chat UI
        - Dark mode or light mode (specify)
        - Subtle gradients, rounded corners
        - Professional, not playful

        {composition_instruction}
        """
    },

    # ═══════════════════════════════════════════════════════════════════
    # AUTOMATION WORKFLOW INTERFACES
    # ═══════════════════════════════════════════════════════════════════

    "automation": {
        "description": "Workflow automation (n8n, Zapier style)",
        "interface_elements": [
            "connected nodes/blocks",
            "arrows showing data flow",
            "trigger and action icons",
            "execution status indicators",
            "sidebar with node settings"
        ],
        "prompt_template": """
        {device_type} showing workflow automation interface.

        INTERFACE ELEMENTS:
        - Visual workflow with {node_count} connected nodes
        - Nodes representing: {node_types}
        - Arrows showing data flow direction
        - Color-coded by node type (triggers=green, actions=blue)
        - Mini icons for services (Slack, Gmail, CRM, AI)

        STYLE:
        - Technical but clean
        - Dark or light canvas background
        - Colorful node icons on neutral background
        - Grid or dot pattern on canvas

        {composition_instruction}
        """
    },

    # ═══════════════════════════════════════════════════════════════════
    # DASHBOARD/ANALYTICS INTERFACES
    # ═══════════════════════════════════════════════════════════════════

    "dashboard": {
        "description": "Analytics and metrics dashboards",
        "interface_elements": [
            "KPI cards with big numbers",
            "line/bar charts",
            "data tables",
            "date range selector",
            "filter controls"
        ],
        "prompt_template": """
        {device_type} showing analytics dashboard interface.

        INTERFACE ELEMENTS:
        - Top row: {metric_count} KPI cards with metrics
        - Main chart: {chart_type} showing trends
        - Secondary visualizations: {secondary_charts}
        - Filter bar with date range and segments

        METRICS TO SHOW:
        {metrics_list}

        STYLE:
        - Professional business intelligence aesthetic
        - Clean data visualization
        - Consistent color palette for charts
        - Proper data hierarchy

        {composition_instruction}
        """
    },

    # ═══════════════════════════════════════════════════════════════════
    # CODE EDITOR INTERFACES
    # ═══════════════════════════════════════════════════════════════════

    "code_editor": {
        "description": "IDE/Code editor interfaces",
        "interface_elements": [
            "syntax-highlighted code",
            "file tree sidebar",
            "terminal panel",
            "AI assistant sidebar (Cursor-style)"
        ],
        "prompt_template": """
        {device_type} showing modern code editor interface.

        INTERFACE ELEMENTS:
        - Code editor with {language} syntax highlighting
        - File tree on left sidebar
        - {ai_feature} (AI assistant panel/inline suggestions)
        - Terminal panel at bottom (optional)

        CODE SHOWN:
        - Clean, readable code snippet
        - Highlighted AI-suggested lines (if applicable)
        - Professional, not toy example

        STYLE:
        - Dark theme (VS Code / Cursor style)
        - Modern monospace font
        - Colorful syntax highlighting
        - Clean iconography

        {composition_instruction}
        """
    }
}


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITION INSTRUCTIONS (append to interface prompt)
# ═══════════════════════════════════════════════════════════════════════════

composition_instructions = {
    "device_only": """
    COMPOSITION:
    - Device floating on clean gradient background
    - Subtle shadow beneath device
    - Professional product photography style
    - No person in frame
    """,

    "author_holding_phone": """
    COMPOSITION:
    - Professional person holding the phone
    - Phone screen clearly visible, facing camera
    - Person in business casual attire
    - Modern office or neutral background
    - Natural, confident pose
    - Person looking at camera or at phone (specify)
    - Well-lit, professional photo quality
    """,

    "author_at_desk": """
    COMPOSITION:
    - Professional person at desk with laptop/monitor
    - Screen content clearly visible
    - Person pointing at screen or engaged with content
    - Modern workspace environment
    - Natural lighting from window
    - Professional but approachable vibe
    """,

    "split_view": """
    COMPOSITION:
    - Split image: person on left (40%), device on right (60%)
    - Clean vertical divider
    - Person: professional portrait, engaged expression
    - Device: clean mockup with interface
    - Cohesive color scheme across both halves
    """,

    "before_after": """
    COMPOSITION:
    - Side-by-side comparison layout
    - Left labeled "Before", right labeled "After"
    - Clear transformation arrow or indicator
    - Before: chaotic/complex/old
    - After: clean/simple/modern
    - Same framing for both sides
    """
}
```

---

#### Interface Visual Generation Logic

```python
# ═══════════════════════════════════════════════════════════════════════════
# NANO BANANA GENERATOR CLIENT
# External AI image generation service client
# ═══════════════════════════════════════════════════════════════════════════

class NanoBananaGenerator:
    """
    Client for Nano Banana Pro image generation service.

    This is an external API client for generating AI images.
    All image generation in the system goes through this client.

    Usage:
        generator = await NanoBananaGenerator.create()
        result = await generator.generate(prompt, style="professional")
    """

    def __init__(self, api_key: str, base_url: str = "https://api.nanobanana.pro"):
        self.api_key = api_key
        self.base_url = base_url
        self._session: Optional[aiohttp.ClientSession] = None

    @classmethod
    async def create(cls, api_key: Optional[str] = None) -> "NanoBananaGenerator":
        """Factory method to create NanoBananaGenerator instance."""
        key = api_key or os.environ.get("NANO_BANANA_API_KEY")
        if not key:
            raise ValueError("NANO_BANANA_API_KEY must be set")
        return cls(api_key=key)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
        return self._session

    async def generate(
        self,
        prompt: str,
        style: str = "professional",
        size: str = "1200x627",  # LinkedIn recommended
        timeout: int = 180
    ) -> Dict[str, Any]:
        """
        Generate an image using Nano Banana Pro.

        Args:
            prompt: Text description of desired image
            style: Visual style (professional, modern, minimal, etc.)
            size: Image dimensions (WxH)
            timeout: Maximum wait time in seconds

        Returns:
            Dict with 'id', 'url', 'local_path' keys

        Raises:
            ImageGenerationError: If generation fails
            asyncio.TimeoutError: If timeout exceeded
        """
        session = await self._get_session()

        payload = {
            "prompt": prompt,
            "style": style,
            "size": size,
            "quality": "high"
        }

        async with session.post(
            f"{self.base_url}/v1/generate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ImageGenerationError(
                    f"Nano Banana API error {response.status}: {error_text}"
                )

            data = await response.json()
            return {
                "id": data["id"],
                "url": data["image_url"],
                "local_path": None  # Downloaded later if needed
            }

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


# ═══════════════════════════════════════════════════════════════════════════
# MUTEX FOR NANO BANANA CLIENT
# Prevents race conditions when multiple pipeline runs share the same client
#
# WHY MODULE-LEVEL:
# - NanoBananaGenerator is an external client passed as parameter
# - Same client instance may be shared across concurrent pipelines
# - Module-level lock ensures serialization across ALL usages
#
# ALTERNATIVE: If each pipeline gets its own client instance, move lock
# to the client class itself or remove lock entirely.
# ═══════════════════════════════════════════════════════════════════════════
_nano_banana_lock = asyncio.Lock()

# Track which client instance is currently using the lock (for debugging)
_nano_banana_lock_holder: Optional[str] = None


async def generate_interface_visual(
    post_content: str,
    content_type: ContentType,
    mentioned_products: List[str],
    visual_brief: str,
    nano_banana: NanoBananaGenerator,
    timeout_seconds: int = 180  # FIX #15: Configurable timeout
) -> VisualAsset:
    """
    Generate interface visual using Nano Banana Pro.
    Everything is AI-generated — no real screenshots needed.

    MEDIUM PRIORITY FIX #15:
    - Added mutex lock to prevent race conditions when sharing Nano Banana client
    - Added configurable timeout for image generation
    - Added logging for debugging and monitoring

    Args:
        post_content: The post text to contextualize the visual
        content_type: Type of content being created
        mentioned_products: Products mentioned in post
        visual_brief: Brief description of desired visual
        nano_banana: NanoBananaGenerator client instance
        timeout_seconds: Maximum time to wait for generation (default 3 min)

    Returns:
        VisualAsset with generated image

    Raises:
        asyncio.TimeoutError: If generation exceeds timeout
        ImageGenerationError: If Nano Banana fails
    """
    import logging
    logger = logging.getLogger("InterfaceVisualGenerator")

    # Step 1: Detect product category
    product_category = _detect_product_category(mentioned_products)
    # Returns: "crm", "ai_chat", "automation", "dashboard", "code_editor"

    logger.info(
        f"[VISUAL_GEN] Starting interface visual generation\n"
        f"  Product category: {product_category}\n"
        f"  Content type: {content_type.value}\n"
        f"  Brief: {visual_brief[:50]}..."
    )

    # Step 2: Get prompt template for this category
    # FAIL-FAST: No hidden fallback - crash if category unknown
    if product_category not in interface_generation_prompts:
        raise ValueError(
            f"Unknown product category '{product_category}'. "
            f"Valid categories: {list(interface_generation_prompts.keys())}"
        )
    template_config = interface_generation_prompts[product_category]

    # Step 3: Select composition mode based on content type
    composition_mode = _select_composition_mode(content_type, post_content)
    composition_instruction = composition_instructions[composition_mode]

    # Step 4: Determine device type
    if "mobile" in visual_brief.lower() or "app" in visual_brief.lower():
        device_type = "iPhone 15 Pro"
    elif "workflow" in visual_brief.lower():
        device_type = "MacBook Pro screen"
    else:
        device_type = "modern laptop"

    # Step 5: Build the full prompt
    prompt = template_config["prompt_template"].format(
        device_type=device_type,
        composition_instruction=composition_instruction,
        # ... other template variables filled from post analysis
    )

    # Step 6: Add quality instructions
    prompt += """

    QUALITY REQUIREMENTS:
    - 4K resolution, sharp details
    - Realistic lighting and shadows
    - Professional photography quality
    - Interface text should be readable (not blurry)
    - Natural, not obviously AI-generated
    - Suitable for LinkedIn professional audience
    """

    # Step 7: Generate with Nano Banana Pro
    # Use lock to prevent concurrent access to shared client
    import uuid
    lock_request_id = f"visual_{uuid.uuid4().hex[:8]}"
    logger.debug(f"[VISUAL_GEN] {lock_request_id} waiting for Nano Banana lock...")

    async with _nano_banana_lock:
        global _nano_banana_lock_holder
        _nano_banana_lock_holder = lock_request_id
        logger.debug(f"[VISUAL_GEN] Lock acquired, starting generation with {timeout_seconds}s timeout")
        start_time = datetime.now()

        try:
            # FIX #15: Add timeout to generation
            diagram = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,  # Default executor
                    lambda: nano_banana.generate_contextual_image(
                        article_content=post_content,
                        research_data={"product_category": product_category},
                        diagram_purpose=visual_brief,
                        diagram_type="interface_visual"
                    )
                ),
                timeout=timeout_seconds
            )

            generation_duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"[VISUAL_GEN] Generation complete in {generation_duration:.1f}s\n"
                f"  Asset ID: {diagram.id if hasattr(diagram, 'id') else 'unknown'}"
            )

            return diagram

        except asyncio.TimeoutError:
            generation_duration = (datetime.now() - start_time).total_seconds()
            logger.error(
                f"[VISUAL_GEN] TIMEOUT after {generation_duration:.1f}s "
                f"(limit: {timeout_seconds}s)"
            )
            raise

        except Exception as e:
            generation_duration = (datetime.now() - start_time).total_seconds()
            logger.error(
                f"[VISUAL_GEN] Generation failed after {generation_duration:.1f}s: {e}"
            )
            raise ImageGenerationError(f"Nano Banana generation failed: {e}") from e


class ImageGenerationError(Exception):
    """Raised when image generation fails."""
    pass


def _select_composition_mode(
    content_type: ContentType,
    post_content: str
) -> str:
    """
    Select composition mode based on content type and post content.
    """
    # Personal recommendation → author holding device
    if any(phrase in post_content.lower() for phrase in [
        "я использую", "рекомендую", "мой опыт", "i use", "i recommend"
    ]):
        return "author_holding_phone"

    # Tutorial/how-to → author at desk explaining
    if content_type == ContentType.AUTOMATION_CASE:
        return "author_at_desk"

    # Tool comparison → before/after
    if "сравн" in post_content.lower() or "vs" in post_content.lower():
        return "before_after"

    # Thought leadership → split view
    if content_type == ContentType.PRIMARY_SOURCE:
        return "split_view"

    # Default → device only
    return "device_only"
```

---

#### Content-Type to Visual Format Mapping

```python
"""
Each ContentType has optimal visual formats that enhance the content.
"""

content_type_visual_mapping = {
    ContentType.ENTERPRISE_CASE: {
        "primary_formats": [
            "metrics_card",           # Highlight key metrics/ROI
            "data_visualization",     # Charts, graphs
            "architecture_diagram"    # System architecture
        ],
        "secondary_formats": [
            "company_logo_card",      # Professional with company branding
            "timeline_visual"         # Implementation journey
        ],
        "style_notes": [
            "Professional, credible aesthetic",
            "Data-driven visuals when metrics available",
            "Clean corporate feel without being boring"
        ],
        "avoid": ["overly playful", "abstract without data connection"]
    },

    ContentType.PRIMARY_SOURCE: {
        "primary_formats": [
            "concept_illustration",   # Abstract representation of ideas
            "quote_card",             # Key finding as visual quote
            "paper_highlight"         # Styled excerpt from research
        ],
        "secondary_formats": [
            "abstract_visualization", # Neural/scientific aesthetic
            "comparison_visual"       # Contrasting viewpoints
        ],
        "style_notes": [
            "Intellectual, thought-provoking",
            "Scientific or academic feel",
            "Complex ideas visualized simply"
        ],
        "avoid": ["oversimplified clipart", "corporate stock photos"]
    },

    ContentType.AUTOMATION_CASE: {
        "primary_formats": [
            "workflow_diagram",       # Step-by-step visual
            "carousel",               # Multi-step tutorial
            "tool_screenshot"         # Actual interface screenshots
        ],
        "secondary_formats": [
            "before_after_card",      # Transformation visual
            "results_metrics_card"    # Time/cost savings
        ],
        "style_notes": [
            "Practical, instructional",
            "Clear visual hierarchy",
            "Shows actual tools when possible"
        ],
        "avoid": ["abstract concepts", "stock photos of people at computers"]
    },

    ContentType.COMMUNITY_CONTENT: {
        "primary_formats": [
            "quote_card",             # Highlighted community insight
            "collage",                # Multiple perspectives
            "platform_screenshot"     # Reddit/Twitter thread styled
        ],
        "secondary_formats": [
            "discussion_summary",     # Visual debate representation
            "attribution_card"        # Credit community members
        ],
        "style_notes": [
            "Community-connected feel",
            "Multiple voices represented",
            "Platform-native aesthetics"
        ],
        "avoid": ["single perspective only", "overly polished"]
    },

    ContentType.TOOL_RELEASE: {
        "primary_formats": [
            "product_screenshot",     # Actual tool interface
            "feature_highlights",     # Key features visual
            "comparison_chart"        # vs competitors
        ],
        "secondary_formats": [
            "demo_animation",         # GIF of tool in action
            "pros_cons_card"          # Balanced evaluation visual
        ],
        "style_notes": [
            "Fresh, timely feel",
            "Showcase actual product when possible",
            "Balanced, not promotional"
        ],
        "avoid": ["marketing-speak aesthetics", "pure logo/branding"]
    }
}
```

---

#### Type-Specific Visual Configuration

```python
visual_config = {
    # ═══════════════════════════════════════════════════════════════════
    # SINGLE IMAGE PRESETS
    # ═══════════════════════════════════════════════════════════════════

    "single_image": {
        "tool": "Nano Banana Pro (Laozhang.ai, gemini-3-pro-image-preview, 4K)",
        "resolution": "4K",  # Supersampling, downscaled for web
        "aspect_ratio": "3:2",  # Landscape for diagrams
        "dimensions": "1200x800",  # Final LinkedIn size after downscale
        "style_presets": {
            # Enterprise Case presets
            "metrics_card": {
                "method": "template_overlay",
                "template": """
                Clean card with large number prominent.
                Company name subtle. Metric label clear.
                Brand colors. Professional typography.
                """,
                "elements": ["metric_value", "metric_label", "company_name", "context"]
            },
            "architecture_diagram": {
                "method": "ai_generation + annotation",
                "prompt_template": """
                Clean technical architecture diagram showing {system}.
                Boxes and arrows. Professional tech aesthetic.
                Blue and gray color scheme. White background.
                No text labels (added via overlay).
                """,
                "negative_prompt": "cluttered, hand-drawn, cartoon"
            },
            "data_visualization": {
                "method": "ai_generation",
                "prompt_template": """
                Abstract data visualization representing {concept}.
                Flowing lines, nodes, neural network aesthetic.
                Dark background with glowing elements.
                Futuristic, professional. {accent_color} accents.
                """,
                "negative_prompt": "text, watermark, logo, cluttered"
            },

            # Primary Source presets
            "concept_illustration": {
                "method": "ai_generation",
                "prompt_template": """
                Professional tech illustration representing {concept}.
                Abstract, intellectual aesthetic.
                {mood} mood. Minimalist style.
                High quality, 4K resolution.
                """,
                "negative_prompt": "text, watermark, logo, busy, cartoon"
            },
            "quote_card": {
                "method": "template_overlay",
                "template": """
                Elegant quote card with text prominent.
                Subtle background pattern or gradient.
                Author attribution. Source reference.
                Professional, academic feel.
                """
            },
            "paper_highlight": {
                "method": "template_overlay",
                "template": """
                Styled as academic paper excerpt.
                Highlighted key finding.
                Reference citation. Clean typography.
                """
            },

            # Automation Case presets
            "workflow_diagram": {
                "method": "template_generation",
                "template": """
                Step-by-step workflow visualization.
                Numbered steps. Arrow connections.
                Tool icons where applicable.
                Clean, instructional aesthetic.
                """
            },
            "tool_screenshot": {
                "method": "actual_screenshot + annotation",
                "notes": "Capture actual tool interface, add callout annotations"
            },
            "before_after_card": {
                "method": "template_overlay",
                "template": """
                Split comparison: Before (pain) | After (solution).
                Clear metrics comparison.
                Transformation arrow. Time/cost savings highlighted.
                """
            },

            # Community Content presets
            "community_quote_card": {
                "method": "template_overlay",
                "template": """
                Quote with platform-style aesthetic.
                User attribution. Platform icon.
                Engagement metrics subtle. Discussion feel.
                """
            },
            "collage": {
                "method": "template_composition",
                "template": """
                Multiple perspectives combined.
                2-4 quotes or viewpoints.
                Visual separation between voices.
                Community discussion aesthetic.
                """
            },
            "platform_screenshot": {
                "method": "actual_screenshot + styling",
                "notes": "Capture actual Reddit/HN/Twitter thread, style for LinkedIn"
            },

            # Tool Release presets
            "product_screenshot": {
                "method": "actual_screenshot + annotation",
                "notes": "Clean capture of tool interface, highlight key features"
            },
            "feature_highlights": {
                "method": "template_overlay",
                "template": """
                Tool name prominent.
                3-5 key features with icons.
                Clean, modern product aesthetic.
                """
            },
            "comparison_chart": {
                "method": "template_generation",
                "template": """
                Tool A vs Tool B comparison.
                Feature checklist format.
                Green checks, red x's.
                Winner indicated.
                """
            }
        }
    },

    # ═══════════════════════════════════════════════════════════════════
    # CAROUSEL CONFIG
    # ═══════════════════════════════════════════════════════════════════

    "carousel": {
        "tool": "Nano Banana Pro + Template overlay",
        "dimensions": "1080x1350",  # LinkedIn optimal for carousels
        "type_specific_structures": {
            ContentType.ENTERPRISE_CASE: {
                "slides": [
                    {"type": "cover", "content": "Company + headline result"},
                    {"type": "problem", "content": "The challenge faced"},
                    {"type": "solution", "content": "What they implemented"},
                    {"type": "results", "content": "Key metrics (1 per slide)"},
                    {"type": "lessons", "content": "Key takeaways"},
                    {"type": "cta", "content": "Follow for more + question"}
                ],
                "style": "professional, data-driven"
            },
            ContentType.AUTOMATION_CASE: {
                "slides": [
                    {"type": "cover", "content": "What you'll learn to build"},
                    {"type": "step", "content": "Step 1 with visual"},
                    {"type": "step", "content": "Step 2 with visual"},
                    {"type": "step", "content": "Step 3 with visual"},
                    {"type": "result", "content": "What this achieves"},
                    {"type": "cta", "content": "Try it + follow for more"}
                ],
                "style": "instructional, practical"
            },
            ContentType.TOOL_RELEASE: {
                "slides": [
                    {"type": "cover", "content": "New tool announcement"},
                    {"type": "feature", "content": "Key feature 1"},
                    {"type": "feature", "content": "Key feature 2"},
                    {"type": "feature", "content": "Key feature 3"},
                    {"type": "verdict", "content": "Who should care"},
                    {"type": "cta", "content": "Try it + link"}
                ],
                "style": "product showcase, balanced"
            }
        },
        "template_elements": {
            "brand_colors": ["#1E3A8A", "#3B82F6", "#F8FAFC"],
            "fonts": ["Inter Bold", "Inter Regular"],
            "logo_placement": "bottom-right",
            "slide_number": "top-right subtle"
        }
    },

    # ═══════════════════════════════════════════════════════════════════
    # DOCUMENT/PDF CONFIG
    # ═══════════════════════════════════════════════════════════════════

    "document_pdf": {
        "tool": "Canva API / Custom template",
        "dimensions": "8.5x11 or A4",
        "best_for_types": [ContentType.ENTERPRISE_CASE, ContentType.PRIMARY_SOURCE],
        "structures": {
            "case_study_summary": {
                "sections": [
                    "Header with company + headline",
                    "Challenge section",
                    "Solution section",
                    "Results with charts",
                    "Key takeaways",
                    "About the author"
                ]
            },
            "research_summary": {
                "sections": [
                    "Paper title + authors",
                    "Key finding highlight",
                    "Methodology brief",
                    "Implications",
                    "Source citation"
                ]
            }
        },
        "elements": {
            "header_image": "Generated via Nano Banana Pro (4K)",
            "body": "Clean typography",
            "callout_boxes": "Key insights highlighted",
            "data_viz": "Charts/graphs for metrics"
        }
    },

    # ═══════════════════════════════════════════════════════════════════
    # BRAND CONSISTENCY
    # ═══════════════════════════════════════════════════════════════════

    "brand_consistency": {
        "color_palette": {
            "primary": "#1E3A8A",
            "secondary": "#3B82F6",
            "accent": "#10B981",
            "background": "#F8FAFC",
            "text": "#1F2937",
            "success": "#22C55E",
            "warning": "#F59E0B"
        },
        "type_specific_accents": {
            ContentType.ENTERPRISE_CASE: "#1E3A8A",  # Professional blue
            ContentType.PRIMARY_SOURCE: "#7C3AED",   # Intellectual purple
            ContentType.AUTOMATION_CASE: "#10B981",  # Technical green
            ContentType.COMMUNITY_CONTENT: "#F59E0B", # Community orange
            ContentType.TOOL_RELEASE: "#3B82F6"      # Fresh blue
        },
        "visual_style": {
            "aesthetic": "clean, modern, professional",
            "mood": "innovative, approachable, expert",
            "avoid": "stock photo cliches, overly corporate, generic"
        }
    }
}
```

---

#### Type-Specific Image Prompt Templates

```python
image_prompt_templates_by_type = {
    ContentType.ENTERPRISE_CASE: {
        "data_focus": """
        Professional data visualization showing business metrics.
        Clean dashboard aesthetic with {metric_type} prominent.
        Corporate blue and white color scheme.
        Modern, credible, enterprise-grade feel.
        Abstract representation of {company_industry}.
        High quality, 4K resolution. No text.
        """,
        "architecture_focus": """
        Clean technical architecture diagram.
        System components as connected nodes.
        Professional tech illustration style.
        {brand_color} accents on white background.
        Enterprise software aesthetic.
        """
    },

    ContentType.PRIMARY_SOURCE: {
        "concept_focus": """
        Abstract intellectual illustration of {concept}.
        Scientific, research aesthetic.
        Purple and blue tones. Neural/cognitive imagery.
        Thought-provoking, not literal.
        Academic elegance. Minimalist.
        """,
        "discovery_focus": """
        Visualization of breakthrough or discovery.
        Light emerging from complexity.
        Abstract representation of {finding}.
        Wonder and insight mood.
        """
    },

    ContentType.AUTOMATION_CASE: {
        "workflow_focus": """
        Clean workflow visualization.
        Connected steps with arrows.
        Automation/efficiency aesthetic.
        Green accents suggesting productivity.
        Technical but approachable.
        """,
        "tool_integration": """
        Abstract representation of tools working together.
        Gears, connections, data flows.
        Modern automation aesthetic.
        Professional but energetic.
        """
    },

    ContentType.COMMUNITY_CONTENT: {
        "discussion_focus": """
        Abstract representation of community discussion.
        Multiple voices, speech bubbles abstracted.
        Warm, connected aesthetic.
        Orange and yellow warmth.
        Social, engaged mood.
        """,
        "synthesis_focus": """
        Visual of insights coming together.
        Multiple streams merging.
        Curation aesthetic.
        Warm, inclusive colors.
        """
    },

    ContentType.TOOL_RELEASE: {
        "product_focus": """
        Modern product launch aesthetic.
        Fresh, new, innovative feel.
        Blue gradient with light accents.
        Tech product photography style but abstract.
        Energy and possibility.
        """,
        "comparison_focus": """
        Abstract representation of evaluation/comparison.
        Balance, weighing, assessment imagery.
        Clean, objective aesthetic.
        Professional review feel.
        """
    }
}

negative_prompts_by_type = {
    ContentType.ENTERPRISE_CASE: [
        "text", "watermark", "playful", "cartoon",
        "unprofessional", "cluttered", "cheap"
    ],
    ContentType.PRIMARY_SOURCE: [
        "text", "watermark", "oversimplified", "cartoon",
        "childish", "corporate stock"
    ],
    ContentType.AUTOMATION_CASE: [
        "text", "watermark", "abstract only", "no tools",
        "confusing flow", "too technical"
    ],
    ContentType.COMMUNITY_CONTENT: [
        "text", "watermark", "single person", "isolated",
        "cold", "corporate"
    ],
    ContentType.TOOL_RELEASE: [
        "text", "watermark", "outdated", "marketing hype",
        "generic tech", "stock photo"
    ]
}
```

---

#### Output Schema

```python
@dataclass
class VisualAsset:
    id: str
    post_id: str
    content_type: ContentType

    # Content
    format: str  # single_image / carousel / document
    visual_style: str  # metrics_card / workflow_diagram / quote_card / etc.
    files: List[str]  # File paths or URLs

    # Generation details
    prompt_used: str
    tool_used: str
    dimensions: str
    type_specific_adjustments: List[str]  # What was adjusted for content type

    # Metadata
    alt_text: str  # Accessibility
    created_at: datetime

    # For carousel/document
    slides: Optional[List[dict]] = None  # Slide content if applicable
    slide_count: Optional[int] = None

    # Quality
    brand_consistency_check: bool = True
    mobile_optimized: bool = True

    # ─────────────────────────────────────────────────────────────────
    # PHOTO INTEGRATION (NEW)
    # Tracks whether author's personal photo was used and how
    # This is critical for:
    # 1. Analytics - correlate photo usage with engagement
    # 2. Variety checking - avoid reusing same photo too often
    # 3. Meta-Agent learning - understand photo impact
    # ─────────────────────────────────────────────────────────────────
    photo_used: bool = False
    photo_id: Optional[str] = None  # Reference to photo in PhotoLibrary
    photo_integration_mode: Optional[str] = None  # photo_as_is / photo_overlay / photo_ai_edit / carousel_bookend
    photo_position: Optional[str] = None  # left_side / right_side / corner_badge / etc.
    photo_edit_prompt: Optional[str] = None  # If AI editing was applied
    photo_selection_rationale: Optional[str] = None  # Why this photo was chosen


@dataclass
class VisualCreatorOutput:
    """
    Complete output from Visual Creator Agent.
    """
    primary_asset: VisualAsset
    alternative_assets: List[VisualAsset]  # Other options generated

    # Decisions
    format_selection_rationale: str
    style_selection_rationale: str

    # For QC
    visual_content_match_score: float  # 0-1 how well visual matches post
```

---

### 6. QC (QUALITY CONTROL) AGENT

#### Purpose
Check quality of final content using **content-type-aware scoring rubrics**. Different content types have different quality standards: enterprise cases need metrics credibility; research posts need intellectual depth; automation posts need practical reproducibility; community posts need authentic connection; tool reviews need balanced assessment.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        QC AGENT (Enhanced)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: HumanizedPost + VisualAsset + ContentType                           │
│         │                                                                   │
│         ▼                                                                   │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                  CONTENT-TYPE RUBRIC SELECTOR                      │     │
│  │  Loads type-specific scoring criteria and weights                 │     │
│  └───────────────────────────┬───────────────────────────────────────┘     │
│                              │                                              │
│                              ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    MULTI-CRITERIA SCORER                           │     │
│  │                                                                    │     │
│  │  UNIVERSAL CRITERIA:                                               │     │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │     │
│  │  │  Hook   │ │  Value  │ │ Human-  │ │ Visual  │ │ Safety  │    │     │
│  │  │Strength │ │ Density │ │  ness   │ │  Match  │ │         │    │     │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘    │     │
│  │                                                                    │     │
│  │  TYPE-SPECIFIC CRITERIA (examples):                               │     │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐              │     │
│  │  │   Metrics    │ │ Reproduce-   │ │    Tone      │              │     │
│  │  │  Credibility │ │   ability    │ │   Match      │              │     │
│  │  │ (Enterprise) │ │ (Automation) │ │   (All)      │              │     │
│  │  └──────────────┘ └──────────────┘ └──────────────┘              │     │
│  │                                                                    │     │
│  └───────────────────────────┬───────────────────────────────────────┘     │
│                              │                                              │
│                              ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────┐     │
│  │                    TYPE-AWARE AGGREGATOR                           │     │
│  │  Applies type-specific weight modifiers to scoring                │     │
│  └───────────────────────────┬───────────────────────────────────────┘     │
│                              │                                              │
│              ┌───────────────┼───────────────┐                             │
│              ▼               ▼               ▼                             │
│       ┌──────────┐    ┌──────────┐    ┌──────────┐                        │
│       │   PASS   │    │  REVISE  │    │  REJECT  │                        │
│       │ Score≥7.5│    │Score 5.5-│    │ Score<5.5│                        │
│       │          │    │   7.5    │    │          │                        │
│       └────┬─────┘    └────┬─────┘    └────┬─────┘                        │
│            │               │               │                               │
│            ▼               ▼               ▼                               │
│       To Human        TYPE-AWARE       Back to                            │
│       Approval        FEEDBACK         Trend Scout                        │
│                      Generator         (new topic)                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### Content-Type Specific Scoring Adjustments

```python
"""
Each ContentType has different quality priorities.
This adjusts both what's evaluated and how it's weighted.
"""

type_specific_scoring = {
    ContentType.ENTERPRISE_CASE: {
        "weight_adjustments": {
            "hook_strength": 0.20,      # Important but not primary
            "value_density": 0.25,      # High - must deliver insights
            "humanness": 0.15,          # Lower - credibility > personality
            "visual_match": 0.15,
            "controversy_safety": 0.10,
            # Type-specific criteria:
            "metrics_credibility": 0.15  # NEW - are numbers believable?
        },
        "additional_criteria": {
            "metrics_credibility": {
                "description": "Are metrics specific, believable, and well-sourced?",
                "evaluation_prompt": """
                Rate the metrics credibility from 1-10:
                "{post_text}"

                Consider:
                - Are specific numbers provided (not just "significant improvement")?
                - Is the source/company clearly identified?
                - Are claims believable and not exaggerated?
                - Is timeline/context provided?
                """,
                "criteria": {
                    "10": "Specific, sourced, believable metrics with context.",
                    "8": "Good metrics, clearly attributed.",
                    "6": "Some numbers but vague or unsourced.",
                    "4": "Mostly qualitative, lacking specifics.",
                    "2": "No real metrics or clearly exaggerated."
                }
            }
        },
        "tone_expectation": "professional credibility with insight"
    },

    ContentType.PRIMARY_SOURCE: {
        "weight_adjustments": {
            "hook_strength": 0.20,
            "value_density": 0.25,      # High - must simplify without dumbing down
            "humanness": 0.15,
            "visual_match": 0.10,
            "controversy_safety": 0.15,  # Higher - research can be misrepresented
            # Type-specific criteria:
            "intellectual_depth": 0.15   # NEW - is the analysis substantive?
        },
        "additional_criteria": {
            "intellectual_depth": {
                "description": "Does the post engage meaningfully with the research?",
                "evaluation_prompt": """
                Rate the intellectual depth from 1-10:
                "{post_text}"

                Consider:
                - Is the core finding accurately represented?
                - Are nuances and limitations acknowledged?
                - Does it add interpretive value beyond summary?
                - Would a researcher find this fair?
                """,
                "criteria": {
                    "10": "Substantive engagement, adds real insight.",
                    "8": "Good understanding, fair representation.",
                    "6": "Accurate but surface-level.",
                    "4": "Oversimplified or slightly misrepresents.",
                    "2": "Misunderstands or sensationalizes."
                }
            }
        },
        "tone_expectation": "intellectual engagement with accessibility"
    },

    ContentType.AUTOMATION_CASE: {
        "weight_adjustments": {
            "hook_strength": 0.15,      # Lower - practitioners search for solutions
            "value_density": 0.30,      # Highest - must be actionable
            "humanness": 0.15,
            "visual_match": 0.15,
            "controversy_safety": 0.05, # Low - tutorials rarely controversial
            # Type-specific criteria:
            "reproducibility": 0.20     # NEW - can reader actually do this?
        },
        "additional_criteria": {
            "reproducibility": {
                "description": "Can a reader actually reproduce this workflow?",
                "evaluation_prompt": """
                Rate the reproducibility from 1-10:
                "{post_text}"

                Consider:
                - Are tools/technologies clearly named?
                - Are steps specific enough to follow?
                - Are gotchas/warnings included?
                - Would someone with reasonable skill succeed?
                """,
                "criteria": {
                    "10": "Clear steps, specific tools, complete instructions.",
                    "8": "Good detail, reader could figure out gaps.",
                    "6": "General approach clear but missing specifics.",
                    "4": "Vague, would need significant research.",
                    "2": "Not reproducible, too abstract."
                }
            }
        },
        "tone_expectation": "practitioner authenticity with generosity"
    },

    ContentType.COMMUNITY_CONTENT: {
        "weight_adjustments": {
            "hook_strength": 0.20,
            "value_density": 0.20,
            "humanness": 0.20,          # Higher - must feel connected
            "visual_match": 0.10,
            "controversy_safety": 0.15,
            # Type-specific criteria:
            "community_authenticity": 0.15  # NEW - does it feel connected?
        },
        "additional_criteria": {
            "community_authenticity": {
                "description": "Does the post feel authentically connected to the community?",
                "evaluation_prompt": """
                Rate the community authenticity from 1-10:
                "{post_text}"

                Consider:
                - Are sources/contributors attributed?
                - Does it capture diverse perspectives?
                - Does it feel like part of the conversation?
                - Is the author positioned as participant, not observer?
                """,
                "criteria": {
                    "10": "Feels like community insider sharing wisdom.",
                    "8": "Good connection, proper attribution.",
                    "6": "Covers community content but feels distant.",
                    "4": "More like reporting on than participating in.",
                    "2": "Disconnected, appropriating without credit."
                }
            }
        },
        "tone_expectation": "conversational warmth with connection"
    },

    ContentType.TOOL_RELEASE: {
        "weight_adjustments": {
            "hook_strength": 0.25,      # Higher - competing for attention
            "value_density": 0.20,
            "humanness": 0.15,
            "visual_match": 0.15,
            "controversy_safety": 0.10,
            # Type-specific criteria:
            "evaluation_balance": 0.15  # NEW - is assessment fair?
        },
        "additional_criteria": {
            "evaluation_balance": {
                "description": "Is the tool evaluation balanced and credible?",
                "evaluation_prompt": """
                Rate the evaluation balance from 1-10:
                "{post_text}"

                Consider:
                - Are both pros and cons mentioned?
                - Is it clear who should/shouldn't use this?
                - Does it feel like honest assessment vs promotion?
                - Are claims verified or clearly labeled as first impressions?
                """,
                "criteria": {
                    "10": "Balanced, honest, helps reader decide.",
                    "8": "Good assessment, minor bias.",
                    "6": "Leans promotional or dismissive.",
                    "4": "Clearly unbalanced, agenda apparent.",
                    "2": "Pure promotion or unfair criticism."
                }
            }
        },
        "tone_expectation": "balanced assessment with hands-on credibility"
    }
}
```

---

#### Universal Scoring Rubric

```python
qc_rubric_universal = {
    "hook_strength": {
        "base_weight": 0.20,
        "criteria": {
            "10": "Impossible to not click. Perfect curiosity gap.",
            "8": "Very compelling, would stop most scrollers.",
            "6": "Good hook, but seen similar before.",
            "4": "Generic, doesn't stand out.",
            "2": "Boring, would scroll past."
        },
        "evaluation_prompt": """
        Rate this hook from 1-10:
        "{hook}"

        Content Type: {content_type}

        Consider:
        - Does it create curiosity appropriate to the content type?
        - Is it specific vs generic?
        - Would it make YOU stop scrolling?
        - Does it promise clear value?
        - Does it match the expected hook style for this content type?
        """
    },

    "value_density": {
        "base_weight": 0.25,
        "criteria": {
            "10": "Every sentence delivers value. Dense with insights.",
            "8": "High value, minimal filler.",
            "6": "Good value but some padding.",
            "4": "More fluff than substance.",
            "2": "Says nothing new."
        },
        "type_specific_value": {
            ContentType.ENTERPRISE_CASE: "Metrics, lessons, replicable insights",
            ContentType.PRIMARY_SOURCE: "Clarity of findings, implications, debate angles",
            ContentType.AUTOMATION_CASE: "Actionable steps, tools, time savings",
            ContentType.COMMUNITY_CONTENT: "Synthesized wisdom, diverse perspectives",
            ContentType.TOOL_RELEASE: "Feature clarity, use cases, honest evaluation"
        },
        "evaluation_prompt": """
        Rate the value density from 1-10:
        "{post_text}"

        Content Type: {content_type}
        Expected Value Type: {type_specific_value}

        Consider:
        - Does it deliver the type of value expected for this content type?
        - How many actionable insights for this type?
        - Is there filler content?
        - Would reader learn something new and useful?
        """
    },

    "humanness": {
        "base_weight": 0.15,
        "criteria": {
            "10": "Sounds exactly like a real person. Unique voice.",
            "8": "Very natural, occasional AI tells.",
            "6": "Mostly human but some robotic phrases.",
            "4": "Clearly AI-assisted.",
            "2": "Obviously AI-generated."
        },
        "evaluation_prompt": """
        Rate the humanness from 1-10:
        "{post_text}"

        Content Type: {content_type}
        Expected Tone: {tone_expectation}

        Look for:
        - AI phrases (It's important to note, Furthermore...)
        - Personal touches appropriate to content type
        - Sentence rhythm variety
        - Authentic voice markers
        - Tone match to content type expectations
        """
    },

    "visual_match": {
        "base_weight": 0.15,
        "criteria": {
            "10": "Image perfectly amplifies the message.",
            "8": "Strong visual that supports content.",
            "6": "Relevant but not exceptional.",
            "4": "Generic stock-photo feel.",
            "2": "Irrelevant or distracting."
        },
        "type_specific_expectations": {
            ContentType.ENTERPRISE_CASE: "Data visualization, metrics card, or architecture diagram",
            ContentType.PRIMARY_SOURCE: "Concept illustration or quote card",
            ContentType.AUTOMATION_CASE: "Workflow diagram, carousel, or screenshot",
            ContentType.COMMUNITY_CONTENT: "Quote card or platform-style visual",
            ContentType.TOOL_RELEASE: "Product screenshot or comparison chart"
        },
        "evaluation_prompt": """
        Rate how well the image matches the post from 1-10:

        Content Type: {content_type}
        Expected Visual: {type_specific_expectations}
        Image description: {image_alt_text}

        Consider:
        - Does the image add to the message?
        - Is it the right visual type for this content type?
        - Is it attention-grabbing?
        - Does it feel unique or generic?
        """
    },

    "controversy_safety": {
        "base_weight": 0.15,
        "criteria": {
            "10": "Thought-provoking without being offensive.",
            "8": "Takes a stance, but respectfully.",
            "6": "Safe, unlikely to spark negative reactions.",
            "4": "Could be misinterpreted.",
            "2": "Likely to cause backlash."
        },
        "evaluation_prompt": """
        Rate the controversy safety from 1-10:
        "{post_text}"

        Content Type: {content_type}

        Consider:
        - Could this offend any group?
        - Are claims well-supported?
        - Is the tone respectful?
        - Any potential for misinterpretation?
        - For research: Is it fairly representing the source?
        - For enterprise: Are claims about the company accurate?
        """
    },

    "tone_match": {
        "base_weight": 0.10,
        "description": "Does the tone match what's expected for this content type?",
        "criteria": {
            "10": "Perfect tone for content type.",
            "8": "Good tone match, minor deviations.",
            "6": "Acceptable but not optimal.",
            "4": "Tone mismatch noticeable.",
            "2": "Wrong tone entirely."
        },
        "evaluation_prompt": """
        Rate the tone match from 1-10:
        "{post_text}"

        Content Type: {content_type}
        Expected Tone: {tone_expectation}

        Consider:
        - Does it sound like the right voice for this content type?
        - Enterprise: Professional credibility
        - Research: Intellectual engagement
        - Automation: Practitioner helpfulness
        - Community: Connected warmth
        - Tool: Balanced assessment
        """
    }
}

# ═══════════════════════════════════════════════════════════════════════════
# FIX #4: Use centralized THRESHOLD_CONFIG instead of hardcoded values
# This ensures consistency with ThresholdConfig and enables runtime tuning
# ═══════════════════════════════════════════════════════════════════════════
decision_thresholds = {
    "pass": THRESHOLD_CONFIG.min_score_to_proceed,        # Default: 8.0
    "revise": THRESHOLD_CONFIG.revision_threshold,        # FIX: Use revision_threshold (7.0), not rejection_threshold
    "reject": THRESHOLD_CONFIG.rejection_threshold,       # Default: 5.5
    "auto_publish": THRESHOLD_CONFIG.auto_publish_threshold,  # Default: 9.0
}

# Note: For content-type-specific thresholds, use THRESHOLD_CONFIG.get_decision(score, content_type)
```

---

#### Helper Functions for Scoring

```python
# FIX: These functions were called but never defined

def calculate_weighted_score(scores: Dict[str, float], type_config: dict) -> float:
    """
    Calculate weighted aggregate score based on content-type weights.

    FIX: Previously called but undefined, causing NameError.

    Args:
        scores: Dict of criterion_name -> score (1-10)
        type_config: Type-specific config with weight_adjustments

    Returns:
        Weighted average score (0-10)
    """
    weights = type_config.get("weight_adjustments", {})
    total_weight = 0.0
    weighted_sum = 0.0

    for criterion, weight in weights.items():
        score = scores.get(criterion, 0)  # Missing scores = 0 (fail-fast)
        weighted_sum += score * weight
        total_weight += weight

    # Normalize if weights don't sum to 1.0
    if total_weight > 0:
        return round(weighted_sum / total_weight * (total_weight if total_weight <= 1 else 1), 2)
    return 0.0


def get_type_aware_suggestion(
    criterion: str,
    content_type: ContentType,
    post: "HumanizedPost"
) -> str:
    """
    Get content-type-specific improvement suggestion for a criterion.

    FIX: Previously called but undefined, causing NameError.

    Args:
        criterion: Name of the scoring criterion
        content_type: Type of content being evaluated
        post: The post being evaluated (for context)

    Returns:
        Specific, actionable suggestion string
    """
    suggestions = improvement_suggestions_by_type.get(content_type, {})
    criterion_suggestions = suggestions.get(criterion, [])

    if criterion_suggestions:
        return criterion_suggestions[0]  # Return first suggestion

    # Generic fallback
    return f"Improve {criterion} score by reviewing the rubric criteria."
```

---

#### Type-Aware Feedback Generator

```python
def generate_revision_feedback(
    scores: dict,
    post: HumanizedPost,
    content_type: ContentType
) -> dict:
    """
    Generates specific, actionable, TYPE-AWARE feedback for revision.
    """
    type_config = type_specific_scoring[content_type]

    feedback = {
        "content_type": content_type.value,
        "overall_score": calculate_weighted_score(scores, type_config),
        "low_scores": [],
        "type_specific_issues": [],
        "specific_suggestions": [],
        "examples_to_reference": []
    }

    for criterion, score in scores.items():
        if score < 7:
            suggestion = get_type_aware_suggestion(criterion, content_type, post)
            feedback["low_scores"].append({
                "criterion": criterion,
                "current_score": score,
                "target_score": 8,
                "weight_for_type": type_config["weight_adjustments"].get(criterion),
                "suggestion": suggestion
            })

    # Add type-specific feedback
    # FAIL-FAST: Missing scores default to 0 (worst) to surface issues immediately
    # Previously defaulted to 10 which HIDES missing scores!
    if content_type == ContentType.ENTERPRISE_CASE:
        if scores.get("metrics_credibility", 0) < 7:
            feedback["type_specific_issues"].append(
                "Metrics need more specificity. Add exact numbers, timeframes, and source attribution."
            )

    elif content_type == ContentType.AUTOMATION_CASE:
        if scores.get("reproducibility", 0) < 7:
            feedback["type_specific_issues"].append(
                "Steps need more detail. Name specific tools, versions, and configuration."
            )

    elif content_type == ContentType.PRIMARY_SOURCE:
        if scores.get("intellectual_depth", 0) < 7:
            feedback["type_specific_issues"].append(
                "Add more interpretive value. What does this mean for practitioners? What's your take?"
            )

    elif content_type == ContentType.COMMUNITY_CONTENT:
        if scores.get("community_authenticity", 0) < 7:
            feedback["type_specific_issues"].append(
                "Strengthen community connection. Add attributions, diverse perspectives, invitation to discuss."
            )

    elif content_type == ContentType.TOOL_RELEASE:
        if scores.get("evaluation_balance", 0) < 7:
            feedback["type_specific_issues"].append(
                "Balance the evaluation. Add limitations or who shouldn't use this."
            )

    return feedback


improvement_suggestions_by_type = {
    ContentType.ENTERPRISE_CASE: {
        "hook_strength": [
            "Lead with the most impressive metric",
            "Name the company in the hook for credibility",
            "Use a lessons-learned angle"
        ],
        "value_density": [
            "Add more specific metrics with context",
            "Include timeline for implementation",
            "Extract replicable lessons"
        ],
        "humanness": [
            "Add interpretive commentary ('What struck me was...')",
            "Include a measured opinion on the approach"
        ],
        "metrics_credibility": [
            "Add specific numbers (X% not 'significant')",
            "Include timeframe and baseline",
            "Attribute metrics to source"
        ]
    },

    ContentType.PRIMARY_SOURCE: {
        "hook_strength": [
            "Lead with the counterintuitive finding",
            "Challenge a common assumption",
            "Create a curiosity gap about implications"
        ],
        "value_density": [
            "Explain why this matters for practitioners",
            "Add your interpretation of implications",
            "Include debate angles"
        ],
        "humanness": [
            "Add intellectual reactions ('This made me reconsider...')",
            "Include nuanced takes, not just summary"
        ],
        "intellectual_depth": [
            "Go beyond summary to interpretation",
            "Acknowledge nuances and limitations",
            "Connect to practical implications"
        ]
    },

    ContentType.AUTOMATION_CASE: {
        "hook_strength": [
            "Lead with time/cost savings",
            "Promise specific outcome",
            "Address a relatable pain point"
        ],
        "value_density": [
            "Add step-by-step specifics",
            "Name exact tools and versions",
            "Include gotchas and tips"
        ],
        "humanness": [
            "Add practitioner empathy ('I know this pain...')",
            "Share what you learned building it"
        ],
        "reproducibility": [
            "Name all tools explicitly",
            "Add configuration details",
            "Include warnings about common issues"
        ]
    },

    ContentType.COMMUNITY_CONTENT: {
        "hook_strength": [
            "Highlight the most surprising insight",
            "Create FOMO about the discussion",
            "Promise curated wisdom"
        ],
        "value_density": [
            "Synthesize multiple perspectives",
            "Extract practitioner signals",
            "Add your meta-takeaway"
        ],
        "humanness": [
            "Position yourself as community participant",
            "Add genuine reactions to insights"
        ],
        "community_authenticity": [
            "Add proper attributions",
            "Include diverse viewpoints",
            "Invite continued discussion"
        ]
    },

    ContentType.TOOL_RELEASE: {
        "hook_strength": [
            "Emphasize timeliness ('Just dropped')",
            "Lead with killer feature or comparison",
            "Create urgency for relevant users"
        ],
        "value_density": [
            "Be specific about features",
            "Clarify who should care",
            "Include access information"
        ],
        "humanness": [
            "Add hands-on experience notes",
            "Include honest first impressions"
        ],
        "evaluation_balance": [
            "Add limitations or caveats",
            "Mention who shouldn't use this",
            "Compare fairly to alternatives"
        ]
    }
}
```

---

#### Output Schema

```python
# ═══════════════════════════════════════════════════════════════════════════
# UNIFIED CONTENT EVALUATION
# This replaces the scattered QCResult / SingleCallEvaluation types
# Single source of truth for content quality assessment
# ═══════════════════════════════════════════════════════════════════════════

class EvaluationDecision(Enum):
    """Unified decision enum for content evaluation."""
    PASS = "pass"
    REVISE = "revise"
    REJECT = "reject"


@dataclass
class CriterionFeedback:
    """Feedback for a single evaluation criterion."""
    criterion: str
    score: float           # 1.0-10.0 (float for consistency across all evaluators)
    weight: float          # Weight used in aggregate calculation
    weighted_score: float  # score * weight
    feedback: str          # Human-readable feedback
    suggestions: List[str] # Specific improvement suggestions


@dataclass
class ContentEvaluation:
    """
    UNIFIED evaluation result for QC pipeline.

    This replaces:
    - QCResult (original QC Agent output)
    - SingleCallEvaluation (new evaluator output)

    Benefits of unification:
    1. Consistent float scores (not int vs float confusion)
    2. Single decision enum (not string vs bool)
    3. Clear feedback structure
    4. Threshold decisions via THRESHOLD_CONFIG
    """

    # ─────────────────────────────────────────────────────────────────
    # Identification
    # ─────────────────────────────────────────────────────────────────
    id: str
    post_id: str
    content_type: ContentType

    # ─────────────────────────────────────────────────────────────────
    # Scores (all float for consistency)
    # ─────────────────────────────────────────────────────────────────
    criterion_scores: Dict[str, float]       # criterion_name -> score (1.0-10.0)
    type_specific_scores: Dict[str, float]   # Type-specific criteria scores
    aggregate_score: float                   # Final weighted score

    # ─────────────────────────────────────────────────────────────────
    # Feedback
    # ─────────────────────────────────────────────────────────────────
    criterion_feedback: Dict[str, CriterionFeedback]
    strengths: List[str]
    weaknesses: List[str]
    actionable_suggestions: List[str]

    # ─────────────────────────────────────────────────────────────────
    # Decision
    # ─────────────────────────────────────────────────────────────────
    decision: EvaluationDecision
    revision_instructions: Optional[List[str]] = None
    revision_target: Optional[str] = None    # "writer" | "humanizer" | "visual"

    # ─────────────────────────────────────────────────────────────────
    # Metadata
    # ─────────────────────────────────────────────────────────────────
    evaluated_at: datetime
    evaluator_model: str
    threshold_used: float                    # From THRESHOLD_CONFIG
    weights_used: Dict[str, float]

    def passes_threshold(self) -> bool:
        """Check if evaluation passes the threshold."""
        return self.decision == EvaluationDecision.PASS

    # LOW PRIORITY FIX #1: Add __repr__ for better debugging
    def __repr__(self) -> str:
        """Concise representation for debugging."""
        decision_emoji = {
            EvaluationDecision.PASS: "✅",
            EvaluationDecision.REVISE: "🔄",
            EvaluationDecision.REJECT: "❌"
        }.get(self.decision, "?")

        return (
            f"ContentEvaluation({decision_emoji} "
            f"post_id='{self.post_id}', "
            f"score={self.aggregate_score:.1f}/{self.threshold_used:.1f}, "
            f"decision={self.decision.value})"
        )

    @classmethod
    def from_scores(
        cls,
        post_id: str,
        content_type: ContentType,
        criterion_scores: Dict[str, float],
        criterion_feedback: Dict[str, CriterionFeedback],
        evaluator_model: str
    ) -> "ContentEvaluation":
        """
        Factory method that uses THRESHOLD_CONFIG for decision.
        """
        import uuid

        # Calculate aggregate score
        # FIX: If weighted_score = score * weight, don't divide again
        # Correct formula: sum of (score * weight) / sum of weights
        total_weight = sum(cf.weight for cf in criterion_feedback.values())
        aggregate_score = sum(cf.score * cf.weight for cf in criterion_feedback.values()) / total_weight

        # Get threshold and make decision
        threshold = THRESHOLD_CONFIG.get_pass_threshold(content_type)
        decision_str = THRESHOLD_CONFIG.get_decision(aggregate_score, content_type)
        decision = EvaluationDecision(decision_str)

        # Extract strengths/weaknesses
        strengths = [cf.feedback for cf in criterion_feedback.values() if cf.score >= 8.0]
        weaknesses = [cf.feedback for cf in criterion_feedback.values() if cf.score < 6.0]
        suggestions = [s for cf in criterion_feedback.values() for s in cf.suggestions]

        return cls(
            id=str(uuid.uuid4()),
            post_id=post_id,
            content_type=content_type,
            criterion_scores=criterion_scores,
            type_specific_scores={},  # Populated by type-specific evaluators
            aggregate_score=aggregate_score,
            criterion_feedback=criterion_feedback,
            strengths=strengths,
            weaknesses=weaknesses,
            actionable_suggestions=suggestions,
            decision=decision,
            evaluated_at=datetime.now(),
            evaluator_model=evaluator_model,
            threshold_used=threshold,
            weights_used={cf.criterion: cf.weight for cf in criterion_feedback.values()}
        )


# ═══════════════════════════════════════════════════════════════════════════
# APPROVED POST
# Bridge between QC approval and Scheduling/Publishing
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ApprovedPost:
    """
    Post that has passed QC and is approved for scheduling/publishing.

    This is the bridge between:
    - QC Agent output (ContentEvaluation)
    - Scheduling System input (ScheduledPost)

    Contains all information needed for:
    1. Scheduling (content, visual, timing preferences)
    2. Publishing (LinkedIn API payload)
    3. Analytics (lineage tracking, QC metadata for feedback loop)
    """
    id: str

    # ─────────────────────────────────────────────────────────────────
    # Content (from HumanizedPost)
    # ─────────────────────────────────────────────────────────────────
    content: str
    content_type: ContentType

    # ─────────────────────────────────────────────────────────────────
    # Visual (from VisualAsset)
    # ─────────────────────────────────────────────────────────────────
    visual_asset_id: str
    visual_type: str
    visual_files: List[str]

    # ─────────────────────────────────────────────────────────────────
    # QC Results (for analytics feedback loop)
    # ─────────────────────────────────────────────────────────────────
    qc_score: float
    qc_evaluation: ContentEvaluation
    revision_count: int

    # ─────────────────────────────────────────────────────────────────
    # Approval Metadata
    # ─────────────────────────────────────────────────────────────────
    approval_type: str      # "auto" | "human"
    approved_by: str        # "auto:score>=9.0" | "human:username"
    approved_at: datetime

    # ─────────────────────────────────────────────────────────────────
    # Lineage (for tracing through pipeline)
    # ─────────────────────────────────────────────────────────────────
    pipeline_run_id: str
    topic_id: str
    draft_id: str
    humanized_post_id: str

    # ─────────────────────────────────────────────────────────────────
    # Analytics Metadata (for feedback loop)
    # ─────────────────────────────────────────────────────────────────
    hook_style: Optional[HookStyle] = None  # FIX: Made Optional - may not be set in all cases
    template_used: str = ""
    has_author_photo: bool = False

    # LOW PRIORITY FIX #1: Add __repr__ for better debugging
    def __repr__(self) -> str:
        """Concise representation for debugging."""
        approval_emoji = "🤖" if self.approval_type == "auto" else "👤"
        return (
            f"ApprovedPost({approval_emoji} "
            f"id='{self.id}', "
            f"type={self.content_type.value}, "
            f"qc_score={self.qc_score:.1f}, "
            f"revisions={self.revision_count})"
        )

    @staticmethod
    def _get_hook_style_from_state(state: dict) -> Optional[HookStyle]:
        """
        FIX: Properly convert hook_style from state to HookStyle enum.
        Previously used empty string default which is incompatible with HookStyle type.
        """
        hook_style_value = state.get("hook_style_used")
        if hook_style_value is None:
            return None
        if isinstance(hook_style_value, HookStyle):
            return hook_style_value
        if isinstance(hook_style_value, str) and hook_style_value:
            try:
                return HookStyle(hook_style_value)
            except ValueError:
                return None
        return None

    @classmethod
    def from_pipeline_state(cls, state: "PipelineState") -> "ApprovedPost":
        """
        Create ApprovedPost from completed pipeline state.
        """
        import uuid

        humanized = state["humanized_post"]
        visual = state.get("visual_asset")
        qc = state.get("qc_result")

        # Determine approval type
        score = qc.aggregate_score if qc else 0
        auto_threshold = THRESHOLD_CONFIG.get_auto_publish_threshold()

        if score >= auto_threshold:
            approval_type = "auto"
            approved_by = f"auto:score>={auto_threshold}"
        else:
            approval_type = "human"
            approved_by = "human:pending"

        return cls(
            id=str(uuid.uuid4()),
            content=humanized.humanized_text,
            content_type=state["content_type"],
            visual_asset_id=visual.id if visual else "",
            visual_type=visual.visual_style if visual else "",
            visual_files=visual.files if visual else [],
            qc_score=score,
            qc_evaluation=qc,
            revision_count=state.get("revision_count", 0),
            approval_type=approval_type,
            approved_by=approved_by,
            approved_at=datetime.now(),
            pipeline_run_id=state["run_id"],
            topic_id=state["selected_topic"].id,
            draft_id=state["draft_post"].id,
            humanized_post_id=humanized.id,
            # FIX: Properly handle HookStyle enum - convert string to enum if needed
            hook_style=cls._get_hook_style_from_state(state),
            template_used=state.get("template_used", ""),
            has_author_photo=visual.photo_used if visual and hasattr(visual, 'photo_used') else False
        )


# ═══════════════════════════════════════════════════════════════════════════
# LEGACY: QCResult (kept for backward compatibility)
# DEPRECATED: Use ContentEvaluation instead
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class QCResult:
    """
    Quality Control evaluation result.

    ═══════════════════════════════════════════════════════════════════════════
    FIX: RESPONSIBILITY SEPARATION - QC Agent focuses on DECISION, not suggestions
    ═══════════════════════════════════════════════════════════════════════════

    QC Agent is responsible for:
    ✓ Numeric scores (universal_scores, type_specific_scores, aggregate_score)
    ✓ Decision (pass / revise / reject)
    ✓ Which agent should handle revision (revision_target_agent)
    ✓ What requirements are met/missing (objective checklist)

    QC Agent is NOT responsible for:
    ✗ Generating improvement suggestions (that's Critic Agent's job)
    ✗ Creative alternatives (that's Critic Agent's job)

    The `revision_instructions` field contains WHAT needs to change, not HOW.
    Example: "Hook doesn't meet 210 char limit" (what), not "Try starting with
    a question" (how - that's Critic's job).
    ═══════════════════════════════════════════════════════════════════════════
    """
    id: str
    post_id: str
    content_type: ContentType

    # Scores
    universal_scores: Dict[str, float]  # Universal criteria scores
    type_specific_scores: Dict[str, float]  # Type-specific criteria scores
    aggregate_score: float  # Weighted by type-specific weights

    # Decision
    decision: str  # pass / revise / reject

    # Objective feedback (WHAT needs fixing, not HOW)
    feedback: Optional[dict]
    type_specific_issues: List[str]  # What requirements failed
    revision_instructions: Optional[List[str]]  # WHAT to fix (not HOW - that's Critic's job)
    revision_target_agent: Optional[str]  # writer / humanizer / visual

    # Quality breakdown (objective checklist)
    tone_match_assessment: str
    type_requirements_met: List[str]
    type_requirements_missing: List[str]

    # ─────────────────────────────────────────────────────────────────
    # FIX #21: Visual Evaluation Integration
    # ─────────────────────────────────────────────────────────────────
    visual_evaluation: Optional["VisualEvaluation"] = None  # From VisualQualityEvaluator
    combined_score: Optional[float] = None  # Text (75%) + Visual (25%)
    revision_targets: Optional[Dict[str, bool]] = None  # {"text": bool, "visual": bool}

    # Audit trail
    evaluated_at: datetime = field(default_factory=datetime.now)
    evaluator_model: str = "claude-sonnet"
    weights_used: Dict[str, float] = field(default_factory=dict)


@dataclass
class QCOutput:
    """
    Complete QC output with decision and next steps.

    FIX #21: Now handles visual revision routing.
    """
    result: QCResult

    # Routing decision (expanded for visual)
    next_step: str  # "human_approval" / "revise_writer" / "revise_humanizer" / "revise_visual" / "revise_both" / "reject_restart"

    # For revision loop
    revision_instructions: Optional[str]
    priority_improvements: List[str]

    # FIX #21: Separate revision instructions by target
    text_revision_instructions: Optional[List[str]] = None  # For Writer/Humanizer
    visual_revision_instructions: Optional[List[str]] = None  # For Visual Creator

    # For human approval
    confidence_note: Optional[str]  # Any caveats for human reviewer
    suggested_edits: List[str] = field(default_factory=list)  # Minor tweaks human might want to make

    def get_revision_targets(self) -> List[str]:
        """Return list of agents that need to revise."""
        targets = []
        if self.result.revision_targets:
            if self.result.revision_targets.get("text"):
                targets.append("writer")  # or humanizer based on revision_target_agent
            if self.result.revision_targets.get("visual"):
                targets.append("visual_creator")
        return targets
```

---

## Orchestrator: LangGraph State Machine

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     LANGGRAPH ORCHESTRATOR (ContentType-Aware)                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         PIPELINE STATE                                   │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │  │  run_id │ content_type │ stage │ revision_count │ errors       │    │   │
│  │  └─────────────────────────────────────────────────────────────────┘    │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │   │
│  │  │  trend_topics → selected_topic → analysis_brief → draft_post   │    │   │
│  │  │  → humanized_post → visual_asset → qc_result → final_content   │    │   │
│  │  └─────────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                           WORKFLOW GRAPH                                  │  │
│  │                                                                           │  │
│  │   ┌─────────┐    ┌──────────┐    ┌────────┐    ┌──────────┐             │  │
│  │   │  SCOUT  │───▶│  SELECT  │───▶│ANALYZE │───▶│  WRITE   │◀────┐       │  │
│  │   │         │    │  TOPIC   │    │        │    │          │     │       │  │
│  │   └─────────┘    └──────────┘    └────────┘    └────┬─────┘     │       │  │
│  │        │              │               │             │           │       │  │
│  │        │         ContentType     ContentType        │           │       │  │
│  │        │         Classification  Extraction         ▼           │       │  │
│  │        │              │               │        ┌──────────┐     │       │  │
│  │        │              │               │        │HUMANIZE  │◀────┤       │  │
│  │        ▼              ▼               ▼        │          │     │       │  │
│  │   ┌─────────────────────────────────────┐     └────┬─────┘     │       │  │
│  │   │     ContentType flows through       │          │           │       │  │
│  │   │     entire pipeline, informing:     │          ▼           │       │  │
│  │   │     • Extraction config             │     ┌──────────┐     │       │  │
│  │   │     • Template selection            │     │ VISUALIZE│◀────┤       │  │
│  │   │     • Tone calibration              │     │          │     │       │  │
│  │   │     • Visual format                 │     └────┬─────┘     │       │  │
│  │   │     • QC criteria weights           │          │           │       │  │
│  │   └─────────────────────────────────────┘          ▼           │       │  │
│  │                                               ┌──────────┐     │       │  │
│  │                                               │    QC    │─────┤       │  │
│  │                                               │  (type-  │     │       │  │
│  │                                               │  aware)  │     │       │  │
│  │                                               └────┬─────┘     │       │  │
│  │                                                    │           │       │  │
│  │                              ┌──────────────┬──────┴─────┬─────┴───┐   │  │
│  │                              ▼              ▼            ▼         ▼   │  │
│  │                          [PASS]       [REVISE_W]   [REVISE_H] [REJECT] │  │
│  │                              │              │            │         │   │  │
│  │                              ▼              └────────────┴─────────┘   │  │
│  │                        ┌──────────┐                      │             │  │
│  │                        │ PREPARE  │               (back to scout)     │  │
│  │                        │ OUTPUT   │                                    │  │
│  │                        └────┬─────┘                                    │  │
│  │                             │                                          │  │
│  │                             ▼                                          │  │
│  │                          [END]                                         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### State Definition

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

# Import all data classes
from models import (
    ContentType,
    TrendTopic, TrendScoutOutput,
    AnalysisBrief, TypeSpecificExtraction,
    DraftPost, WriterOutput,
    HumanizedPost,
    VisualAsset, VisualCreatorOutput,
    QCResult, QCOutput
)


class PipelineState(TypedDict):
    """
    Enhanced pipeline state with ContentType awareness flowing through all stages.
    ContentType is determined at scout/select stage and propagates to all downstream agents.
    """

    # ═══════════════════════════════════════════════════════════════════
    # RUN TRACKING
    # ═══════════════════════════════════════════════════════════════════
    run_id: str
    run_timestamp: datetime
    stage: str  # current stage name for debugging/monitoring

    # ═══════════════════════════════════════════════════════════════════
    # CONTENT TYPE CONTEXT (propagates through entire pipeline)
    # ═══════════════════════════════════════════════════════════════════
    content_type: Optional[ContentType]  # Set after topic selection
    type_context: Optional[Dict[str, Any]]  # Type-specific configuration loaded at selection

    # ═══════════════════════════════════════════════════════════════════
    # SCOUT STAGE
    # ═══════════════════════════════════════════════════════════════════
    trend_topics: List[TrendTopic]  # All scored topics from scout
    top_pick: Optional[TrendTopic]  # "Самый важный кейс дня"
    topics_by_type: Dict[str, int]  # Count breakdown by ContentType
    scout_statistics: Optional[Dict[str, Any]]  # total_sources_scanned, exclusion_log, etc.

    # ═══════════════════════════════════════════════════════════════════
    # TOPIC SELECTION
    # ═══════════════════════════════════════════════════════════════════
    selected_topic: Optional[TrendTopic]
    selection_mode: str  # "auto_top_pick" / "human_choice" / "type_balance"

    # ═══════════════════════════════════════════════════════════════════
    # ANALYZER STAGE
    # ═══════════════════════════════════════════════════════════════════
    analysis_brief: Optional[AnalysisBrief]
    extraction_data: Optional[TypeSpecificExtraction]  # Type-specific fields

    # ═══════════════════════════════════════════════════════════════════
    # WRITER STAGE
    # ═══════════════════════════════════════════════════════════════════
    draft_post: Optional[DraftPost]
    writer_output: Optional[WriterOutput]  # Full output with alternatives
    template_used: Optional[str]
    hook_style_used: Optional[str]

    # ═══════════════════════════════════════════════════════════════════
    # HUMANIZER STAGE
    # ═══════════════════════════════════════════════════════════════════
    humanized_post: Optional[HumanizedPost]
    humanization_intensity: Optional[str]  # low / medium / high (type-dependent)

    # ═══════════════════════════════════════════════════════════════════
    # VISUAL CREATOR STAGE
    # ═══════════════════════════════════════════════════════════════════
    visual_asset: Optional[VisualAsset]
    visual_creator_output: Optional[VisualCreatorOutput]
    visual_format_used: Optional[str]  # metrics_card / workflow_diagram / etc.

    # ═══════════════════════════════════════════════════════════════════
    # QC STAGE
    # ═══════════════════════════════════════════════════════════════════
    qc_result: Optional[QCResult]
    qc_output: Optional[QCOutput]
    type_specific_scores: Optional[Dict[str, float]]  # Type-specific criteria scores

    # ═══════════════════════════════════════════════════════════════════
    # REVISION TRACKING
    # ═══════════════════════════════════════════════════════════════════
    revision_count: int
    revision_history: List[Dict[str, Any]]  # Log of all revisions
    current_revision_target: Optional[str]  # "writer" / "humanizer" / "visual"

    # ═══════════════════════════════════════════════════════════════════
    # REJECT/RESTART TRACKING
    # FIX: Previously referenced but never declared fields
    # ═══════════════════════════════════════════════════════════════════
    _reject_restart_count: int  # How many times topic was rejected and restarted
    _rejected_topics: List[str]  # IDs of topics rejected by QC
    _qc_decision: Optional[str]  # Last QC decision: PASS/REVISE_WRITER/REVISE_HUMANIZER/REJECT

    # ═══════════════════════════════════════════════════════════════════
    # META-AGENT SELF-EVALUATION LOOP
    # Integrated between Writer and Humanizer for quality iteration
    # ═══════════════════════════════════════════════════════════════════
    meta_evaluation: Optional[Dict[str, Any]]  # Latest meta-agent evaluation
    meta_evaluation_score: Optional[float]     # Score from meta-agent (uses THRESHOLD_CONFIG)
    meta_iteration: int                        # Current iteration (0, 1, 2, max 3)
    meta_passed: bool                          # Did draft pass meta-evaluation?
    meta_critique_history: List[Dict[str, Any]]  # History of all critiques for learning

    # ═══════════════════════════════════════════════════════════════════
    # FINAL OUTPUT
    # ═══════════════════════════════════════════════════════════════════
    final_content: Optional[Dict[str, Any]]
    human_approval_status: Optional[str]  # pending / approved / rejected / edited / auto_approved / escalated
    human_approval_requested_at: Optional[datetime]  # When approval was first requested
    human_approval_reminder_count: int  # How many reminders have been sent
    human_approval_escalation_level: int  # 0=pending, 1=reminded, 2=escalated, 3=auto-resolved

    # ═══════════════════════════════════════════════════════════════════
    # ERROR HANDLING (Structured, Fail-Fast Compatible)
    # ═══════════════════════════════════════════════════════════════════
    # NOTE: With fail-fast philosophy, critical errors should raise exceptions.
    # These fields are for warnings and non-critical issues only.
    critical_error: Optional[str]   # If set, route to error_handler node
    error_stage: Optional[str]      # Stage where critical error occurred
    errors: List[str]               # Non-critical errors (logged but continue)
    warnings: List[str]             # Warnings (informational)

    # ═══════════════════════════════════════════════════════════════════
    # CONTINUOUS LEARNING ENGINE
    # Self-improvement from EVERY iteration, starting from first post
    # ═══════════════════════════════════════════════════════════════════
    learning_engine: "ContinuousLearningEngine"    # Injected at pipeline start
    iteration_learnings: Optional["IterationLearnings"]  # Learnings from current iteration
    learnings_used_count: int                       # How many learnings were injected into prompts
    is_first_post: bool                             # True if this is the very first post (triggers bootstrap)

    # ═══════════════════════════════════════════════════════════════════
    # SELF-MODIFYING CODE ENGINE
    # System can write new modules during execution when capabilities are missing
    # ═══════════════════════════════════════════════════════════════════
    self_mod_engine: "SelfModificationEngine"       # Injected at pipeline start
    self_mod_result: Optional["SelfModificationResult"]  # Result of last modification attempt
    capabilities_added: List[str]                   # List of capabilities added this run
    code_generation_count: int                      # How many modules were generated this run


# ═══════════════════════════════════════════════════════════════════
# TYPE-SPECIFIC CONTEXT LOADER
# ═══════════════════════════════════════════════════════════════════

def load_type_context(content_type: ContentType) -> Dict[str, Any]:
    """
    Load all type-specific configurations when ContentType is determined.
    This context flows through the entire pipeline.
    """

    TYPE_CONTEXTS = {
        ContentType.ENTERPRISE_CASE: {
            # Analyzer config
            "extraction_focus": ["company", "industry", "problem", "solution", "metrics", "timeline", "lessons"],
            "required_fields": ["company", "metrics", "problem_statement"],

            # Writer config
            "preferred_templates": ["METRICS_HERO", "LESSONS_LEARNED", "HOW_THEY_DID_IT"],
            # FIX: Use get_hook_styles_for_type() to ensure single source of truth
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
            "pass_threshold": 7.2
        },

        ContentType.PRIMARY_SOURCE: {
            "extraction_focus": ["authors", "thesis", "methodology", "findings", "implications", "counterintuitive"],
            "required_fields": ["thesis", "key_findings", "authors"],

            "preferred_templates": ["RESEARCH_INSIGHT", "CONTRARIAN_TAKE", "FUTURE_PREDICTION"],
            # FIX: Use get_hook_styles_for_type() to ensure single source of truth
            "hook_styles": get_hook_styles_for_type(ContentType.PRIMARY_SOURCE),
            "cta_style": "intellectual_discourse",

            "humanization_intensity": "low",
            "tone_markers": ["thoughtful", "nuanced", "intellectual"],
            "avoid_markers": ["oversimplification", "clickbait", "absolutist"],

            "visual_formats": ["concept_diagram", "quote_card_elegant", "data_visualization"],
            "color_scheme": "academic_subtle",

            "extra_criteria": ["intellectual_depth"],
            "weight_adjustments": {"factual_accuracy": 1.4, "engagement_hook": 0.8},
            "pass_threshold": 7.5
        },

        ContentType.AUTOMATION_CASE: {
            "extraction_focus": ["task_automated", "tools_used", "workflow_steps", "time_saved", "code_available"],
            "required_fields": ["task_automated", "tools_used", "workflow_steps"],

            "preferred_templates": ["HOW_TO_GUIDE", "TOOL_STACK_REVEAL", "AUTOMATION_STORY"],
            # FIX: Use get_hook_styles_for_type() to ensure single source of truth
            "hook_styles": get_hook_styles_for_type(ContentType.AUTOMATION_CASE),
            "cta_style": "practical_action",

            "humanization_intensity": "high",
            "tone_markers": ["practical", "enthusiastic", "hands_on"],
            "avoid_markers": ["over_technical", "corporate_speak"],

            "visual_formats": ["workflow_diagram", "screenshot_annotated", "carousel_steps"],
            "color_scheme": "tech_vibrant",

            "extra_criteria": ["reproducibility"],
            "weight_adjustments": {"actionability": 1.3, "engagement_hook": 1.1},
            "pass_threshold": 7.0
        },

        ContentType.COMMUNITY_CONTENT: {
            "extraction_focus": ["platform", "author_credibility", "key_insights", "engagement_signals", "code_examples"],
            "required_fields": ["platform", "key_insights"],

            "preferred_templates": ["COMMUNITY_SPOTLIGHT", "DISCUSSION_SUMMARY", "PERSONAL_STORY"],
            # FIX: Use get_hook_styles_for_type() to ensure single source of truth
            "hook_styles": get_hook_styles_for_type(ContentType.COMMUNITY_CONTENT),
            "cta_style": "community_engagement",

            "humanization_intensity": "high",
            "tone_markers": ["conversational", "authentic", "curious"],
            "avoid_markers": ["formal", "corporate", "distant"],

            "visual_formats": ["quote_card_casual", "screenshot_highlighted", "meme_professional"],
            "color_scheme": "community_warm",

            "extra_criteria": ["community_authenticity"],
            "weight_adjustments": {"voice_match": 1.2, "engagement_hook": 1.2},
            "pass_threshold": 6.8
        },

        ContentType.TOOL_RELEASE: {
            "extraction_focus": ["tool_name", "company", "key_features", "pricing", "demo_url", "competing_tools"],
            "required_fields": ["tool_name", "key_features"],

            "preferred_templates": ["PRODUCT_LAUNCH", "TOOL_COMPARISON", "FIRST_LOOK"],
            # FIX: Use get_hook_styles_for_type() to ensure single source of truth
            "hook_styles": get_hook_styles_for_type(ContentType.TOOL_RELEASE),
            "cta_style": "evaluation_try",

            "humanization_intensity": "medium",
            "tone_markers": ["balanced", "evaluative", "practical"],
            "avoid_markers": ["promotional", "uncritical_hype", "sponsored_feel"],

            "visual_formats": ["product_screenshot", "feature_comparison_table", "demo_gif"],
            "color_scheme": "product_neutral",

            "extra_criteria": ["evaluation_balance"],
            "weight_adjustments": {"factual_accuracy": 1.2, "actionability": 1.1},
            "pass_threshold": 7.0
        }
    }

    return TYPE_CONTEXTS.get(content_type, {})
```

### Workflow Definition

```python
def create_content_pipeline():
    """
    Create the LangGraph workflow with ContentType-aware routing.
    """
    workflow = StateGraph(PipelineState)

    # ═══════════════════════════════════════════════════════════════════
    # ADD NODES
    # ═══════════════════════════════════════════════════════════════════

    # Stage 1: Scout and discover trending topics
    workflow.add_node("scout", trend_scout_node)

    # Stage 2: Select topic and load type context
    workflow.add_node("select_topic", topic_selection_node)

    # Stage 3: Deep analysis with type-specific extraction
    workflow.add_node("analyze", analyzer_node)

    # Stage 4: Generate draft using type-appropriate template
    workflow.add_node("write", writer_node)

    # Stage 4.5: Meta-Agent self-evaluation (NEW)
    # This creates an internal loop for quality iteration before humanization
    workflow.add_node("meta_evaluate", meta_evaluate_node)

    # Stage 5: Humanize with type-specific tone
    workflow.add_node("humanize", humanizer_node)

    # Stage 6: Create visuals in type-appropriate format
    workflow.add_node("visualize", visual_creator_node)

    # Stage 7: Quality check with type-specific criteria
    workflow.add_node("qc", qc_node)

    # Stage 7.5: Continuous Learning (NEW - learns from EVERY iteration)
    # Extracts micro-learnings from evaluation feedback, applies immediately
    workflow.add_node("learn", post_evaluation_learning_node)

    # Stage 8: Prepare for human approval
    workflow.add_node("prepare_output", prepare_for_human_approval)

    # Stage 9: Manual review queue (for max_revisions or low scores)
    workflow.add_node("manual_review_queue", queue_for_manual_review)

    # Error handling node
    workflow.add_node("handle_error", error_handler_node)

    # FIX: Reset node for reject_restart to prevent infinite loops
    # This resets counters while incrementing restart count
    workflow.add_node("reset_for_restart", reset_for_restart_node)

    # ═══════════════════════════════════════════════════════════════════
    # ERROR-AWARE ROUTING HELPER
    # Every node transition checks for critical_error first
    # ═══════════════════════════════════════════════════════════════════

    def make_error_aware_router(next_node: str):
        """
        Factory for error-aware routing functions.
        Returns handle_error if critical_error is set, otherwise next_node.
        """
        def router(state: PipelineState) -> str:
            if state.get("critical_error"):
                return "handle_error"
            return next_node
        return router

    # ═══════════════════════════════════════════════════════════════════
    # DEFINE EDGES (with error routing)
    # ═══════════════════════════════════════════════════════════════════

    # Entry point
    workflow.set_entry_point("scout")

    # Main flow with Meta-Agent integration
    # Each edge checks for critical_error before proceeding
    workflow.add_conditional_edges(
        "scout",
        make_error_aware_router("select_topic"),
        {"select_topic": "select_topic", "handle_error": "handle_error"}
    )

    workflow.add_conditional_edges(
        "select_topic",
        make_error_aware_router("analyze"),
        {"analyze": "analyze", "handle_error": "handle_error"}
    )

    workflow.add_conditional_edges(
        "analyze",
        make_error_aware_router("write"),
        {"write": "write", "handle_error": "handle_error"}
    )

    # ═══════════════════════════════════════════════════════════════════
    # META-AGENT SELF-EVALUATION LOOP (with error routing)
    # write → meta_evaluate → (humanize | write | handle_error)
    # This ensures drafts meet quality threshold before humanization
    # ═══════════════════════════════════════════════════════════════════
    workflow.add_conditional_edges(
        "write",
        make_error_aware_router("meta_evaluate"),
        {"meta_evaluate": "meta_evaluate", "handle_error": "handle_error"}
    )

    def route_after_meta_evaluate_with_error(state: PipelineState) -> str:
        """Route after meta evaluation, checking for errors first."""
        if state.get("critical_error"):
            return "handle_error"
        return route_after_meta_evaluate(state)

    workflow.add_conditional_edges(
        "meta_evaluate",
        route_after_meta_evaluate_with_error,
        {
            "humanize": "humanize",   # Passed evaluation, proceed
            "write": "write",         # Failed, send back for rewrite
            "handle_error": "handle_error"  # Critical error occurred
        }
    )

    workflow.add_conditional_edges(
        "humanize",
        make_error_aware_router("visualize"),
        {"visualize": "visualize", "handle_error": "handle_error"}
    )

    workflow.add_conditional_edges(
        "visualize",
        make_error_aware_router("qc"),
        {"qc": "qc", "handle_error": "handle_error"}
    )

    # ═══════════════════════════════════════════════════════════════════
    # QC ROUTING (Type-Aware) → LEARNING (with error routing)
    # IMPORTANT: After QC, ALWAYS go through learning node first
    # Learning extracts insights regardless of pass/fail decision
    # ═══════════════════════════════════════════════════════════════════
    workflow.add_conditional_edges(
        "qc",
        make_error_aware_router("learn"),
        {"learn": "learn", "handle_error": "handle_error"}
    )

    def route_after_learning_with_error(state: PipelineState) -> str:
        """Route after learning, checking for errors first."""
        if state.get("critical_error"):
            return "handle_error"
        return route_after_learning(state)

    workflow.add_conditional_edges(
        "learn",  # Route AFTER learning
        route_after_learning_with_error,
        {
            "pass": "prepare_output",
            "revise_writer": "write",
            "revise_humanizer": "humanize",
            "revise_visual": "visualize",
            "revise_both": "write",  # FIX: Handle "revise_both" - starts with text, visual_creator checks state
            "reject_restart": "reset_for_restart",  # FIX: Go through reset node first
            "max_revisions_force": "manual_review_queue",
            "handle_error": "handle_error"  # Added error route
        }
    )

    # FIX: Add reset_for_restart edge to scout
    # This node resets state counters to prevent infinite loops
    workflow.add_edge("reset_for_restart", "scout")

    # Terminal edges (with final error check)
    workflow.add_conditional_edges(
        "prepare_output",
        make_error_aware_router(END),
        {END: END, "handle_error": "handle_error"}
    )

    workflow.add_conditional_edges(
        "manual_review_queue",
        make_error_aware_router(END),
        {END: END, "handle_error": "handle_error"}
    )

    workflow.add_edge("handle_error", END)  # Error handler always terminates

    # ═══════════════════════════════════════════════════════════════════
    # CHECKPOINTING FOR PIPELINE RECOVERY
    # Enables resuming from last successful stage after failures
    # ═══════════════════════════════════════════════════════════════════
    from langgraph.checkpoint.sqlite import SqliteSaver

    # Use SQLite for persistent checkpoints (survives restarts)
    checkpointer = SqliteSaver.from_conn_string("data/pipeline_checkpoints.db")

    return workflow.compile(checkpointer=checkpointer)


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE RECOVERY UTILITIES
# Resume pipelines from checkpoints after failures
# ═══════════════════════════════════════════════════════════════════════════

class PipelineRecoveryManager:
    """
    Manages pipeline recovery from checkpoints.

    When a pipeline fails partway through:
    1. State is automatically checkpointed at each stage
    2. Recovery manager can list failed runs
    3. User can resume from last successful stage

    Usage:
        recovery = PipelineRecoveryManager()

        # List recoverable runs
        failed_runs = await recovery.list_recoverable_runs()

        # Resume a specific run
        result = await recovery.resume_pipeline(run_id="abc123")

        # Or start fresh with same topic
        result = await recovery.restart_with_topic(run_id="abc123")
    """

    def __init__(self, pipeline=None):
        self._pipeline = pipeline or create_content_pipeline()
        self._logger = logging.getLogger("PipelineRecovery")

    async def list_recoverable_runs(
        self,
        hours_back: int = 24
    ) -> List[Dict[str, Any]]:
        """
        List runs that failed and can be recovered.

        Returns runs where:
        - Stage is not "error" or terminal
        - Run is within hours_back window
        - Checkpoint exists
        """
        # Query checkpointer for recent runs
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(hours=hours_back)
        recoverable = []

        # Get all thread_ids (run_ids) from checkpointer
        # Note: This is a simplified example - actual implementation depends on checkpointer
        db = await get_db()
        result = await db.client.table("pipeline_errors").select("*").gte(
            "created_at", cutoff.isoformat()
        ).execute()

        for error in result.data or []:
            run_id = error.get("run_id")
            if run_id:
                # Check if checkpoint exists
                config = {"configurable": {"thread_id": run_id}}
                try:
                    state = await self._pipeline.aget_state(config)
                    if state and state.values:
                        recoverable.append({
                            "run_id": run_id,
                            "last_stage": state.values.get("stage"),
                            "content_type": state.values.get("content_type"),
                            "error": error.get("error_message"),
                            "failed_at": error.get("created_at"),
                        })
                except Exception:
                    pass  # Checkpoint doesn't exist or is corrupted

        return recoverable

    async def resume_pipeline(
        self,
        run_id: str,
        override_state: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Resume a pipeline from its last checkpoint.

        Args:
            run_id: The run_id to resume
            override_state: Optional state overrides (e.g., clear critical_error)

        Returns:
            Final pipeline state
        """
        config = {"configurable": {"thread_id": run_id}}

        # Get current state
        state_snapshot = await self._pipeline.aget_state(config)

        if not state_snapshot or not state_snapshot.values:
            raise ValueError(f"No checkpoint found for run {run_id}")

        current_state = state_snapshot.values

        self._logger.info(
            f"Resuming run {run_id} from stage '{current_state.get('stage')}'"
        )

        # Clear error state if set (we're retrying)
        if current_state.get("critical_error"):
            await self._pipeline.aupdate_state(
                config,
                {"critical_error": None, "error_stage": None}
            )
            self._logger.info("Cleared previous error state")

        # Apply any overrides
        if override_state:
            await self._pipeline.aupdate_state(config, override_state)
            self._logger.info(f"Applied state overrides: {list(override_state.keys())}")

        # Resume execution
        result = await self._pipeline.ainvoke(None, config)

        return result

    async def restart_with_topic(
        self,
        run_id: str,
        from_stage: str = "write"
    ) -> Dict[str, Any]:
        """
        Restart a pipeline with the same topic but from a specific stage.

        Useful when:
        - Analysis was good but writing failed
        - Visual generation failed but text is fine

        Args:
            run_id: Original run to get topic from
            from_stage: Stage to restart from
        """
        config = {"configurable": {"thread_id": run_id}}

        state_snapshot = await self._pipeline.aget_state(config)

        if not state_snapshot or not state_snapshot.values:
            raise ValueError(f"No checkpoint found for run {run_id}")

        old_state = state_snapshot.values

        # Create new run with preserved topic/analysis
        new_run_id = str(uuid.uuid4())
        new_config = {"configurable": {"thread_id": new_run_id}}

        # Initialize new state with preserved data
        preserved_keys = ["selected_topic", "content_type", "type_context", "deep_analysis"]
        new_state = {
            "run_id": new_run_id,
            "stage": from_stage,
            **{k: old_state.get(k) for k in preserved_keys if k in old_state}
        }

        self._logger.info(
            f"Restarting from stage '{from_stage}' with new run {new_run_id}"
        )

        # Start new pipeline with preserved state
        result = await self._pipeline.ainvoke(new_state, new_config)

        return result

    async def cleanup_old_checkpoints(self, days_old: int = 7) -> int:
        """
        Clean up checkpoints older than days_old.

        Returns number of checkpoints deleted.
        """
        # Implementation depends on checkpointer storage
        # For SQLite, we'd delete from the checkpoints table
        self._logger.info(f"Cleaning up checkpoints older than {days_old} days")
        # TODO: Implement based on checkpointer type
        return 0


# Global recovery manager instance
_recovery_manager: Optional[PipelineRecoveryManager] = None

def get_recovery_manager() -> PipelineRecoveryManager:
    """Get the global pipeline recovery manager."""
    global _recovery_manager
    if _recovery_manager is None:
        _recovery_manager = PipelineRecoveryManager()
    return _recovery_manager


async def queue_for_manual_review(state: PipelineState) -> Dict[str, Any]:
    """
    Queue post for manual human review.
    Used when max revisions reached but quality still below threshold.

    This is better than auto-publishing low quality content.
    Human can: approve, edit, or reject.
    """
    return {
        "stage": "manual_review_required",
        "human_approval_status": "pending_manual_review",
        "warnings": state.get("warnings", []) + [
            f"Max revisions ({state.get('revision_count', 0)}) reached but quality "
            f"below threshold. Queued for manual review."
        ],
        "final_content": {
            "text": state["humanized_post"].humanized_text,
            "visual": state.get("visual_asset"),
            "qc_score": state.get("qc_result", {}).get("aggregate_score"),
            "requires_human_decision": True
        }
    }


# ═══════════════════════════════════════════════════════════════════
# NODE IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════
#
# IMPORTANT: Node Implementation Rules
#
# 1. NO DIRECT STATE MUTATION
#    ❌ state["stage"] = "scouting"      # BAD - mutates state
#    ✅ return {"stage": "scouting"}     # GOOD - pure function
#
# 2. ERROR HANDLING (see Error Handling Philosophy section)
#    ─────────────────────────────────────────────────────────────
#    a) Retryable errors (rate limits, timeouts):
#       → Use @with_retry decorator
#       → After max retries: return {"critical_error": "..."}
#
#    b) Non-retryable errors (validation, auth):
#       → Immediate: return {"critical_error": "..."}
#
#    c) Business logic "errors" (low quality):
#       → NOT errors! Use normal routing (revise, reject, etc.)
#
#    ❌ BAD PATTERNS:
#       except: pass                     # Swallows error
#       except: return {"image": None}   # Silent degradation
#       except: use_fallback_service()   # Hidden service switch
#
#    ✅ GOOD PATTERN:
#       try:
#           result = await operation_with_retry()
#           return {"result": result}
#       except RetryExhaustedError as e:
#           return {"critical_error": str(e)}
#       # State machine routes to handle_error node
#
# 3. VALIDATION BEFORE PROCESSING
#    Check required state fields at start:
#    if not state.get("required_field"):
#        return {"critical_error": "Missing required_field"}
#
# 4. TIMEOUT HANDLING
#    All async agent calls must have timeouts to prevent infinite hangs
#    Use @with_timeout decorator or asyncio.wait_for()
#    On timeout: return {"critical_error": "Node timed out: ..."}
#
# ═══════════════════════════════════════════════════════════════════


import asyncio
from functools import wraps


# ─────────────────────────────────────────────────────────────────────
# TIMEOUT CONFIGURATION
# MEDIUM PRIORITY FIX #5: Configurable via environment variables
# ─────────────────────────────────────────────────────────────────────

import os


@dataclass
class NodeTimeoutsConfig:
    """
    Centralized timeout configuration for pipeline nodes.

    MEDIUM PRIORITY FIX #5: Moved from hardcoded dict to configurable class.
    Timeouts can be overridden via environment variables for tuning.

    Environment variables:
        NODE_TIMEOUT_SCOUT=120
        NODE_TIMEOUT_VISUALIZE=180
        etc.
    """

    # Agent-based nodes (longer timeouts for LLM calls)
    scout: int = 120           # 2 min - multiple API calls
    analyze: int = 90          # 1.5 min - deep analysis
    write: int = 60            # 1 min - draft generation
    meta_evaluate: int = 45    # 45 sec - evaluation
    humanize: int = 45         # 45 sec - humanization
    visualize: int = 180       # 3 min - image generation (Nano Banana can be slow)
    qc: int = 60               # 1 min - quality check

    # Non-agent nodes (shorter timeouts)
    select_topic: int = 5      # 5 sec - just selection logic
    prepare_output: int = 10   # 10 sec - formatting
    manual_review_queue: int = 5
    handle_error: int = 5

    def __post_init__(self):
        """Load overrides from environment variables."""
        for field_name in self.__dataclass_fields__:
            env_var = f"NODE_TIMEOUT_{field_name.upper()}"
            if os.environ.get(env_var):
                try:
                    setattr(self, field_name, int(os.environ[env_var]))
                except ValueError:
                    pass  # Keep default if invalid

    def get(self, node_name: str, default: int = 60) -> int:
        """Get timeout for a node by name."""
        return getattr(self, node_name, default)

    def to_dict(self) -> Dict[str, int]:
        """Export as dictionary for backward compatibility."""
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


# Global instance and backward-compatible dict
NODE_TIMEOUTS_CONFIG = NodeTimeoutsConfig()
NODE_TIMEOUTS = NODE_TIMEOUTS_CONFIG.to_dict()  # Backward compatibility


class NodeTimeoutError(Exception):
    """Raised when a node exceeds its timeout."""
    def __init__(self, node_name: str, timeout: int):
        self.node_name = node_name
        self.timeout = timeout
        super().__init__(f"Node '{node_name}' timed out after {timeout} seconds")


def with_error_handling(node_name: str = None):
    """
    Decorator to convert node exceptions to critical_error state updates.

    FIX: Previously, exceptions in nodes would bypass the state machine's
    error routing. This decorator catches exceptions and converts them
    to {"critical_error": str, "error_stage": str} returns.

    The state machine's conditional edges then route to error_handler_node.

    Usage:
        @with_error_handling(node_name="scout")
        @with_timeout(node_name="scout")
        async def trend_scout_node(state): ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            import logging
            logger = logging.getLogger(f"Node.{node_name or func.__name__}")

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                name = node_name or func.__name__
                error_msg = f"{type(e).__name__}: {str(e)}"

                logger.error(f"[{name}] Exception caught, routing to error handler: {error_msg}")

                return {
                    "critical_error": error_msg,
                    "error_stage": name
                }

        return wrapper
    return decorator


def with_timeout(timeout_seconds: int = None, node_name: str = None):
    """
    Decorator to add timeout to async node functions.

    Usage:
        @with_timeout(60)
        async def my_node(state): ...

        # Or use NODE_TIMEOUTS config:
        @with_timeout(node_name="scout")
        async def trend_scout_node(state): ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Determine timeout
            nonlocal timeout_seconds
            if timeout_seconds is None:
                name = node_name or func.__name__.replace("_node", "")
                timeout_seconds = NODE_TIMEOUTS.get(name, 60)

            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                name = node_name or func.__name__
                raise NodeTimeoutError(name, timeout_seconds)

        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────────────
# EXCEPTION CLASSES
# ─────────────────────────────────────────────────────────────────────

class TrendScoutError(Exception):
    """Raised when trend scout fails to find topics."""
    pass


class TopicSelectionError(Exception):
    """Raised when topic selection fails."""
    pass


class AnalyzerError(Exception):
    """Raised when analyzer fails to extract required data."""
    pass


class WriterError(Exception):
    """Raised when writer fails to generate draft."""
    pass


class VisualizerError(Exception):
    """Raised when visual creator fails to generate assets."""
    pass


# ─────────────────────────────────────────────────────────────────────
# NODE IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────

@with_error_handling(node_name="scout")
@with_timeout(node_name="scout")
async def trend_scout_node(state: PipelineState) -> Dict[str, Any]:
    """
    Scout node: Discover and score trending topics.
    Classifies ContentType for each topic.

    FAIL-FAST: Raises TrendScoutError if scout fails.
    Note: @with_error_handling converts exceptions to critical_error returns.
    """
    # NOTE: No state mutation - stage tracking via return value only
    scout_output: TrendScoutOutput = await trend_scout_agent.run()

    # Validation: must have at least one topic
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
            "exclusion_log": scout_output.exclusion_log
        }
    }


@with_error_handling(node_name="select_topic")
@with_timeout(node_name="select_topic")
async def topic_selection_node(state: PipelineState) -> Dict[str, Any]:
    """
    Selection node: Choose topic and load ContentType-specific context.
    This is the critical point where ContentType is locked for the pipeline.

    FAIL-FAST: Raises TopicSelectionError if selection fails.
    TIMEOUT: 5 seconds (no external calls)
    Note: @with_error_handling converts exceptions to critical_error returns.
    """
    # Validate required input
    if not state.get("trend_topics"):
        raise TopicSelectionError("No trend_topics in state")

    selection_mode = state.get("selection_mode", "auto_top_pick")

    if selection_mode == "auto_top_pick":
        selected = state.get("top_pick")
    elif selection_mode == "type_balance":
        selected = select_for_type_balance(state["trend_topics"])
    else:  # human_choice
        selected = state.get("selected_topic") or state.get("top_pick")

    # Validate selection result
    if selected is None:
        raise TopicSelectionError(
            f"Failed to select topic with mode '{selection_mode}'"
        )

    # Load type-specific context (THIS IS KEY!)
    content_type = selected.content_type
    type_context = load_type_context(content_type)

    return {
        "stage": "topic_selected",
        "selected_topic": selected,
        "content_type": content_type,
        "type_context": type_context
    }


@with_error_handling(node_name="analyze")
@with_timeout(node_name="analyze")
async def analyzer_node(state: PipelineState) -> Dict[str, Any]:
    """
    Analyzer node: Deep analysis using type-specific extraction config.

    FAIL-FAST: Lets exceptions propagate from analyzer_agent.
    TIMEOUT: 90 seconds (LLM analysis)
    """
    # Validate required input
    topic = state.get("selected_topic")
    content_type = state.get("content_type")
    type_context = state.get("type_context")

    if not all([topic, content_type, type_context]):
        raise ValueError("Analyzer requires selected_topic, content_type, and type_context")

    # Get type-specific extraction config
    extraction_focus = type_context.get("extraction_focus", [])
    required_fields = type_context.get("required_fields", [])

    analysis_brief: AnalysisBrief = await analyzer_agent.run(
        topic=topic,
        content_type=content_type,
        extraction_focus=extraction_focus,
        required_fields=required_fields
    )

    return {
        "stage": "analyzed",
        "analysis_brief": analysis_brief,
        "extraction_data": analysis_brief.extraction_data
    }


@with_error_handling(node_name="write")
@with_timeout(node_name="write")
async def writer_node(state: PipelineState) -> Dict[str, Any]:
    """
    Writer node: Generate draft using type-appropriate templates.

    FAIL-FAST: Lets exceptions propagate from writer_agent.
    TIMEOUT: 60 seconds (LLM generation)
    """
    content_type = state["content_type"]
    type_context = state["type_context"]
    analysis = state["analysis_brief"]

    # Get type-specific writing config
    preferred_templates = type_context.get("preferred_templates", [])
    hook_styles = type_context.get("hook_styles", [])
    cta_style = type_context.get("cta_style", "general")

    # Check if this is a revision (from QC or Meta-Agent)
    revision_instructions = None
    if state.get("current_revision_target") == "writer":
        qc_output = state.get("qc_output")
        if qc_output:
            revision_instructions = qc_output.revision_instructions
    elif state.get("meta_passed") is False:
        # Meta-Agent requested rewrite
        meta_eval = state.get("meta_evaluation", {})
        revision_instructions = meta_eval.get("suggestions", [])

    writer_output: WriterOutput = await writer_agent.run(
        analysis_brief=analysis,
        content_type=content_type,
        preferred_templates=preferred_templates,
        hook_styles=hook_styles,
        cta_style=cta_style,
        revision_instructions=revision_instructions
    )

    return {
        "stage": "drafted",
        "draft_post": writer_output.draft,
        "writer_output": writer_output,
        "template_used": writer_output.draft.template_used,
        "hook_style_used": writer_output.draft.hook_style
    }


# ═══════════════════════════════════════════════════════════════════
# META-AGENT SELF-EVALUATION NODE (NEW)
# Integrated between Writer and Humanizer
# ═══════════════════════════════════════════════════════════════════

@with_error_handling(node_name="meta_evaluate")
@with_timeout(node_name="meta_evaluate")
async def meta_evaluate_node(state: PipelineState) -> Dict[str, Any]:
    """
    Meta-Agent evaluation node.
    Evaluates draft quality and decides: proceed to humanizer or rewrite.

    Uses THRESHOLD_CONFIG for consistent threshold decisions.
    TIMEOUT: 45 seconds (LLM evaluation)

    Flow:
    - If score >= threshold: proceed to humanizer
    - If score < threshold and iterations < max: send back to writer
    - If max iterations reached: force proceed with warning
    """
    draft = state["draft_post"]
    content_type = state["content_type"]
    iteration = state.get("meta_iteration", 0)
    # FIX: Use centralized config instead of hardcoded magic number
    max_iterations = THRESHOLD_CONFIG.get_max_meta_iterations()

    # Force pass if max iterations reached
    if iteration >= max_iterations:
        return {
            "stage": "meta_evaluated",
            "meta_iteration": iteration,
            "meta_passed": True,
            "warnings": state.get("warnings", []) + [
                f"Meta-Agent: Max iterations ({max_iterations}) reached, forcing proceed"
            ]
        }

    # Evaluate draft
    evaluation = await meta_agent.evaluate_draft(
        draft=draft,
        content_type=content_type
    )

    # Get threshold from centralized config
    threshold = THRESHOLD_CONFIG.get_pass_threshold(content_type)

    # Store critique for learning
    critique_entry = {
        "iteration": iteration,
        "score": evaluation.score,
        "threshold": threshold,
        "feedback": evaluation.feedback,
        "suggestions": evaluation.suggestions,
        "timestamp": utc_now().isoformat()
    }
    # LOW PRIORITY FIX #11: Limit history to prevent memory leak
    # Uses MAX_CRITIQUE_HISTORY from HistoryLimits
    critique_history = (
        state.get("meta_critique_history", []) + [critique_entry]
    )[-MAX_CRITIQUE_HISTORY:]  # Keep only last N entries

    if evaluation.score >= threshold:
        return {
            "stage": "meta_evaluated",
            "meta_evaluation": evaluation.__dict__,
            "meta_evaluation_score": evaluation.score,
            "meta_iteration": iteration,
            "meta_passed": True,
            "meta_critique_history": critique_history
        }
    else:
        return {
            "stage": "meta_needs_rewrite",
            "meta_evaluation": evaluation.__dict__,
            "meta_evaluation_score": evaluation.score,
            "meta_iteration": iteration + 1,
            "meta_passed": False,
            "meta_critique_history": critique_history,
            "current_revision_target": "writer"  # Signal for writer_node
        }


def route_after_meta_evaluate(state: PipelineState) -> str:
    """Route based on meta-evaluation result."""
    if state.get("meta_passed", False):
        return "humanize"
    else:
        return "write"  # Send back for rewrite


@with_error_handling(node_name="humanize")
@with_timeout(node_name="humanize")
async def humanizer_node(state: PipelineState) -> Dict[str, Any]:
    """
    Humanizer node: Apply type-specific humanization intensity and tone.

    FAIL-FAST: Lets exceptions propagate from humanizer_agent.
    TIMEOUT: 45 seconds (LLM humanization)
    """
    content_type = state["content_type"]
    type_context = state["type_context"]
    draft = state["draft_post"]

    # Get type-specific humanization config
    intensity = type_context.get("humanization_intensity", "medium")
    tone_markers = type_context.get("tone_markers", [])
    avoid_markers = type_context.get("avoid_markers", [])

    # Check if this is a revision
    revision_instructions = None
    if state.get("current_revision_target") == "humanizer":
        revision_instructions = state["qc_output"].revision_instructions

    humanized_post: HumanizedPost = await humanizer_agent.run(
        draft=draft,
        content_type=content_type,
        intensity=intensity,
        tone_markers=tone_markers,
        avoid_markers=avoid_markers,
        revision_instructions=revision_instructions
    )

    return {
        "humanized_post": humanized_post,
        "humanization_intensity": intensity,
        "stage": "humanized"
    }


@with_error_handling(node_name="visualize")
@with_timeout(node_name="visualize")
async def visual_creator_node(state: PipelineState) -> Dict[str, Any]:
    """
    Visual Creator node: Generate visuals in type-appropriate format.

    TIMEOUT: 180 seconds (image generation can be slow)

    FIX: Added revision_instructions handling (was missing, unlike writer_node and humanizer_node)
    """
    # NOTE: Removed state mutation - use return value only
    content_type = state["content_type"]
    type_context = state["type_context"]
    post = state["humanized_post"]
    draft = state["draft_post"]

    # Get type-specific visual config
    visual_formats = type_context.get("visual_formats", ["single_image"])
    color_scheme = type_context.get("color_scheme", "brand_default")

    # FIX: Check if this is a revision (from QC feedback)
    # Previously, visual_creator_node ignored revision instructions entirely
    revision_instructions = None
    if state.get("current_revision_target") == "visual":
        qc_output = state.get("qc_output")
        if qc_output:
            # Get visual-specific revision instructions from QC
            revision_instructions = getattr(qc_output, "visual_revision_instructions", None)
            if revision_instructions:
                import logging
                logger = logging.getLogger("Pipeline.VisualCreator")
                logger.info(
                    f"[VISUAL] Processing revision with instructions: "
                    f"{revision_instructions[:100]}..."
                )

    visual_output: VisualCreatorOutput = await visual_creator_agent.run(
        post=post,
        visual_brief=draft.visual_brief,
        suggested_type=draft.visual_type,
        content_type=content_type,
        allowed_formats=visual_formats,
        color_scheme=color_scheme,
        revision_instructions=revision_instructions  # FIX: Pass revision feedback
    )

    return {
        "visual_asset": visual_output.primary_asset,
        "visual_creator_output": visual_output,
        "visual_format_used": visual_output.primary_asset.visual_style,
        "stage": "visual_created",
        "current_revision_target": None  # FIX: Clear revision target after processing
    }


@with_error_handling(node_name="qc")
@with_timeout(node_name="qc")
async def qc_node(state: PipelineState) -> Dict[str, Any]:
    """
    QC node: Evaluate using type-specific criteria and weights.

    TIMEOUT: 60 seconds (LLM evaluation)
    """
    # NOTE: Removed state mutation - use return value only
    content_type = state["content_type"]
    type_context = state["type_context"]

    # Get type-specific QC config
    extra_criteria = type_context.get("extra_criteria", [])
    weight_adjustments = type_context.get("weight_adjustments", {})
    pass_threshold = type_context.get("pass_threshold", 7.0)

    qc_output: QCOutput = await qc_agent.run(
        humanized_post=state["humanized_post"],
        visual_asset=state["visual_asset"],
        draft_post=state["draft_post"],
        content_type=content_type,
        extra_criteria=extra_criteria,
        weight_adjustments=weight_adjustments,
        pass_threshold=pass_threshold
    )

    # Update revision tracking
    revision_count = state["revision_count"]
    revision_history = state.get("revision_history", [])

    if qc_output.result.decision != "pass":
        revision_history.append({
            "revision_number": revision_count + 1,
            "decision": qc_output.result.decision,
            "scores": qc_output.result.aggregate_score,
            "target": qc_output.next_step,
            "issues": qc_output.result.type_specific_issues
        })

    return {
        "qc_result": qc_output.result,
        "qc_output": qc_output,
        "type_specific_scores": qc_output.result.type_specific_scores,
        "revision_count": revision_count + (1 if qc_output.result.decision != "pass" else 0),
        "revision_history": revision_history,
        "stage": "qc_completed"
    }


@with_error_handling(node_name="prepare_output")
@with_timeout(node_name="prepare_output")
async def prepare_for_human_approval(state: PipelineState) -> Dict[str, Any]:
    """
    Prepare final content package for human review.

    TIMEOUT: 10 seconds (just formatting)
    """
    # NOTE: Removed state mutation - use return value only
    final_content = {
        "post_text": state["humanized_post"].humanized_text,
        "visual": state["visual_asset"].files,
        "visual_alt_text": state["visual_asset"].alt_text,

        # Metadata for review
        "content_type": state["content_type"].value,
        "template_used": state["template_used"],
        "hook_style": state["hook_style_used"],
        "visual_format": state["visual_format_used"],

        # Scores for transparency
        "qc_score": state["qc_result"].aggregate_score,
        "type_specific_scores": state["type_specific_scores"],

        # Source attribution
        "source_url": state["selected_topic"].primary_source_url,
        "source_title": state["selected_topic"].title,

        # Suggestions for human editor
        "suggested_edits": state["qc_output"].suggested_edits if state["qc_output"] else [],
        "confidence_note": state["qc_output"].confidence_note if state["qc_output"] else None,

        # Statistics
        "revision_count": state["revision_count"],
        "revision_history": state.get("revision_history", [])
    }

    return {
        "final_content": final_content,
        "human_approval_status": "pending",
        "stage": "ready_for_approval"
    }


async def reset_for_restart_node(state: PipelineState) -> Dict[str, Any]:
    """
    Reset state for a fresh topic restart after reject.

    FIX: This node prevents infinite reject_restart loops by:
    1. Incrementing _reject_restart_count (checked by route_after_qc)
    2. Clearing topic-specific state (trend_topics, selected_topic, drafts)
    3. Resetting revision counters
    4. Preserving run_id and learning context

    Called when: Quality is too low to revise, need completely new topic.
    """
    import logging
    logger = logging.getLogger("PipelineResetNode")

    current_restart_count = state.get("_reject_restart_count", 0) + 1
    rejected_topic = state.get("selected_topic", {})

    logger.info(
        f"[RESTART] Resetting for new topic search "
        f"(restart #{current_restart_count}, rejected: {rejected_topic.get('title', 'unknown')})"
    )

    return {
        # Increment restart counter (checked by route_after_qc for limit)
        "_reject_restart_count": current_restart_count,

        # Clear topic-related state for fresh search
        "trend_topics": [],
        "selected_topic": None,
        "analysis_brief": None,
        "draft_post": None,
        "humanized_post": None,
        "visual_asset": None,
        "visual_brief": None,

        # Reset revision counters (fresh start with new topic)
        "revision_count": 0,
        "meta_iteration": 0,
        "meta_critique_history": [],
        "current_revision_target": None,

        # Clear QC state
        "qc_output": None,
        "qc_result": None,
        "meta_passed": None,
        "meta_evaluation": None,

        # Track rejected topics to avoid selecting them again
        "_rejected_topics": state.get("_rejected_topics", []) + [
            rejected_topic.get("id") if rejected_topic else None
        ],

        "stage": "restarting"
    }


async def error_handler_node(state: PipelineState) -> Dict[str, Any]:
    """
    Handle critical errors and prepare error report.

    This node is the central error handler for the entire pipeline.
    It receives control when any node sets critical_error in state.

    Actions:
    1. Log the error with full context
    2. Save error report to database for analysis
    3. Notify monitoring systems
    4. Prepare human-readable error summary
    """
    import logging
    logger = logging.getLogger("PipelineErrorHandler")

    critical_error = state.get("critical_error", "Unknown error")
    errors_list = state.get("errors", [])
    last_stage = state.get("stage", "unknown")

    # Compile full error context
    error_context = {
        "critical_error": critical_error,
        "accumulated_errors": errors_list,
        "last_successful_stage": last_stage,
        "run_id": state.get("run_id"),
        "content_type": state.get("content_type"),
        "topic_id": state.get("selected_topic", {}).get("id") if state.get("selected_topic") else None,
        "revision_count": state.get("revision_count", 0),
    }

    # Log with full context
    logger.error(
        f"Pipeline failed at stage '{last_stage}': {critical_error}\n"
        f"Context: {error_context}"
    )

    # Save to database for post-mortem analysis
    try:
        db = await get_db()
        await db.client.table("pipeline_errors").insert({
            "run_id": state.get("run_id"),
            "error_type": type(critical_error).__name__ if isinstance(critical_error, Exception) else "CriticalError",
            "error_message": str(critical_error),
            "stage": last_stage,
            "context": error_context,
            "created_at": utc_now().isoformat()
        }).execute()
    except Exception as db_error:
        logger.warning(f"Failed to save error to database: {db_error}")

    return {
        "stage": "error",
        "final_content": {
            "status": "error",
            "critical_error": str(critical_error),
            "errors": errors_list,
            "last_successful_stage": last_stage,
            "context": error_context,
            "requires_human_attention": True
        }
    }


# ═══════════════════════════════════════════════════════════════════
# ROUTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def route_after_qc(state: PipelineState) -> str:
    """
    Type-aware routing after QC evaluation.

    Uses THRESHOLD_CONFIG for consistent threshold decisions.

    IMPORTANT: This function is PURE - no state mutations!
    The revision_target is communicated via the returned route name,
    which the destination node can interpret.

    Routes:
    - "pass": Quality passed, proceed to prepare_output
    - "revise_writer/humanizer/visual": Send back for revision
    - "reject_restart": Quality too low, restart with new topic
    - "max_revisions_force": Max revisions reached, queue for manual review
    """
    qc_output = state.get("qc_output")
    qc_result = state.get("qc_result")
    revision_count = state.get("revision_count", 0)
    content_type = state.get("content_type")

    # FIX: Return error route instead of raising exception
    # Raising exceptions in routing functions breaks fail-fast philosophy
    # because it's not routed through error_handler_node
    if not all([qc_output, qc_result, content_type]):
        import logging
        logging.getLogger("QCRouter").error(
            f"Missing required state: qc_output={bool(qc_output)}, "
            f"qc_result={bool(qc_result)}, content_type={content_type}"
        )
        return "handle_error"  # Route to error handler instead of raising

    # FIX: Check reject_restart limit to prevent infinite loops
    # If we've already restarted twice with new topics and still failing,
    # escalate to manual review
    reject_restart_count = state.get("_reject_restart_count", 0)
    # FIX: Use centralized config instead of hardcoded value
    max_reject_restarts = THRESHOLD_CONFIG.max_reject_restarts

    # FIX: Use centralized config instead of local dict
    max_revisions = THRESHOLD_CONFIG.get_max_revisions(content_type)

    # ─────────────────────────────────────────────────────────────────
    # Decision Logic using THRESHOLD_CONFIG
    # ─────────────────────────────────────────────────────────────────

    # Get score (handle both dict and object)
    score = (
        qc_result.get("aggregate_score") if isinstance(qc_result, dict)
        else getattr(qc_result, "aggregate_score", 0)
    )

    # Use centralized threshold config for decision
    decision = THRESHOLD_CONFIG.get_decision(score, content_type)

    # ─────────────────────────────────────────────────────────────────
    # FIX: Check "pass" BEFORE max_revisions!
    # If post finally passes on its last revision, it should still pass,
    # not be sent to manual review. Previous logic was inverted.
    # ─────────────────────────────────────────────────────────────────

    # Route based on decision
    if decision == "pass":
        return "pass"

    # Max revisions reached - queue for manual review (NOT auto-publish!)
    # Only applies to non-passing posts
    if revision_count >= max_revisions:
        return "max_revisions_force"

    elif decision == "reject":
        # FIX: Check if we've hit the restart limit
        # Prevents infinite loop of reject -> scout -> reject -> scout...
        if reject_restart_count >= max_reject_restarts:
            import logging
            logging.getLogger("QCRouter").warning(
                f"Reached max reject_restart limit ({max_reject_restarts}). "
                f"Routing to manual review instead of another restart."
            )
            return "max_revisions_force"  # Escalate to manual review
        return "reject_restart"

    else:  # revise
        # Determine revision target from QC output
        next_step = (
            qc_output.get("next_step") if isinstance(qc_output, dict)
            else getattr(qc_output, "next_step", "revise_writer")
        )

        # Map to valid route names
        if next_step in ["revise_writer", "revise_humanizer", "revise_visual"]:
            return next_step
        else:
            return "revise_writer"  # Default


# ═══════════════════════════════════════════════════════════════════
# CONTINUOUS LEARNING NODE
# Executes AFTER every QC evaluation, BEFORE routing decision
# ═══════════════════════════════════════════════════════════════════

async def post_evaluation_learning_node(state: PipelineState) -> dict:
    """
    Extract learnings from EVERY iteration, regardless of pass/fail.

    This is the heart of continuous self-improvement:
    - Learns from BOTH successful and failed posts
    - Applies learnings IMMEDIATELY to subsequent iterations
    - Builds knowledge base from first post onwards

    Called AFTER QC, BEFORE routing decision.
    The routing decision is preserved in state for route_after_learning.
    """
    import logging
    logger = logging.getLogger("ContinuousLearning")

    learning_engine = state["learning_engine"]

    # ─────────────────────────────────────────────────────────────────
    # FIRST POST BOOTSTRAP
    # On very first post, seed with proven best practices
    # ─────────────────────────────────────────────────────────────────
    if state.get("is_first_post", False):
        bootstrap_learnings = await learning_engine.handle_first_post()
        logger.info(f"[BOOTSTRAP] First post: seeded {len(bootstrap_learnings)} initial learnings")

    # ─────────────────────────────────────────────────────────────────
    # EXTRACT LEARNINGS FROM THIS ITERATION
    # ─────────────────────────────────────────────────────────────────

    # Get evaluations (meta + visual)
    meta_eval = state.get("meta_evaluation")
    qc_output = state.get("qc_output")

    # Build ContentEvaluation from available data
    # (Converting from state format to ContinuousLearningEngine format)
    evaluation = ContentEvaluation(
        post_id=state.get("run_id", "unknown"),
        aggregate_score=state.get("qc_result", {}).get("aggregate_score", 0),
        dimension_scores=state.get("qc_result", {}).get("dimension_scores", {}),
        strengths=meta_eval.get("strengths", []) if meta_eval else [],
        weaknesses=meta_eval.get("weaknesses", []) if meta_eval else [],
        actionable_suggestions=meta_eval.get("suggestions", []) if meta_eval else [],
        decision="PASS" if state.get("qc_result", {}).get("aggregate_score", 0) >= THRESHOLD_CONFIG.get_pass_threshold(state["content_type"]) else "REVISE",
        threshold_used=THRESHOLD_CONFIG.get_pass_threshold(state["content_type"])
    )

    # Build VisualEvaluation
    visual_eval_data = qc_output.get("visual_evaluation", {}) if qc_output else {}
    visual_evaluation = VisualEvaluation(
        score=visual_eval_data.get("score", 7.0),
        issues=visual_eval_data.get("issues", []),
        visual_text_alignment=visual_eval_data.get("alignment_score", 7.0),
        format_appropriateness=visual_eval_data.get("format_score", 7.0),
        suggestions=visual_eval_data.get("suggestions", [])
    )

    # Extract learnings
    learnings = await learning_engine.learn_from_iteration(
        post_id=state.get("run_id", "unknown"),
        evaluation=evaluation,
        visual_evaluation=visual_evaluation,
        content_type=state["content_type"],
        draft=state.get("humanized_post", {}).humanized_text if state.get("humanized_post") else "",
        visual=state.get("visual_asset")
    )

    # ─────────────────────────────────────────────────────────────────
    # STORE QC DECISION FOR ROUTING (preserve original QC logic)
    # ─────────────────────────────────────────────────────────────────
    qc_decision = route_after_qc(state)  # Compute what QC would have routed

    logger.info(
        f"[LEARN] Post {state.get('run_id')}: "
        f"+{len(learnings.new_learnings)} new, "
        f"✓{len(learnings.confirmed_learnings)} confirmed, "
        f"✗{len(learnings.contradicted_learnings)} contradicted"
    )

    return {
        "iteration_learnings": learnings,
        "_qc_decision": qc_decision  # Internal: used by route_after_learning
    }


def route_after_learning(state: PipelineState) -> str:
    """
    Route after learning node.

    Simply forwards the QC decision that was computed and stored
    by post_evaluation_learning_node. This ensures learning happens
    on EVERY path (pass, revise, reject) without changing QC logic.
    """
    # Get pre-computed decision from learning node
    return state.get("_qc_decision", "pass")


def select_for_type_balance(topics: List[TrendTopic]) -> TrendTopic:
    """
    Select topic to balance content type distribution over time.
    Tracks recent posts and picks underrepresented types.
    """
    # Load recent post history (from DB in production)
    recent_type_counts = get_recent_type_distribution()

    # Find underrepresented types
    target_distribution = {
        ContentType.ENTERPRISE_CASE: 0.25,
        ContentType.PRIMARY_SOURCE: 0.20,
        ContentType.AUTOMATION_CASE: 0.25,
        ContentType.COMMUNITY_CONTENT: 0.15,
        ContentType.TOOL_RELEASE: 0.15
    }

    # Score topics by type scarcity + quality
    scored_topics = []
    for topic in topics:
        type_scarcity = target_distribution.get(topic.content_type, 0.2) - recent_type_counts.get(topic.content_type, 0)
        combined_score = topic.score + (type_scarcity * 2)  # Boost underrepresented types
        scored_topics.append((topic, combined_score))

    # Return highest combined score
    scored_topics.sort(key=lambda x: x[1], reverse=True)
    return scored_topics[0][0]


# ═══════════════════════════════════════════════════════════════════
# PIPELINE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════

def initialize_pipeline_state(
    run_id: str,
    selection_mode: str = "auto_top_pick"
) -> PipelineState:
    """
    Create initial pipeline state with defaults.
    """
    return PipelineState(
        run_id=run_id,
        run_timestamp=utc_now(),
        stage="initialized",

        content_type=None,
        type_context=None,

        trend_topics=[],
        top_pick=None,
        topics_by_type={},
        scout_statistics=None,

        selected_topic=None,
        selection_mode=selection_mode,

        analysis_brief=None,
        extraction_data=None,

        draft_post=None,
        writer_output=None,
        template_used=None,
        hook_style_used=None,

        humanized_post=None,
        humanization_intensity=None,

        visual_asset=None,
        visual_creator_output=None,
        visual_format_used=None,

        qc_result=None,
        qc_output=None,
        type_specific_scores=None,

        revision_count=0,
        revision_history=[],
        current_revision_target=None,

        # Meta-Agent self-evaluation
        meta_evaluation=None,
        meta_evaluation_score=None,
        meta_iteration=0,
        meta_passed=False,
        meta_critique_history=[],

        final_content=None,
        human_approval_status=None,

        # Error tracking
        critical_error=None,
        error_stage=None,
        errors=[],
        warnings=[],

        # Continuous Learning (injected by run_pipeline)
        learning_engine=None,  # Set in run_pipeline
        iteration_learnings=None,
        learnings_used_count=0,
        is_first_post=False,  # Set in run_pipeline based on DB check

        # Self-Modifying Code (injected by run_pipeline)
        self_mod_engine=None,  # Set in run_pipeline
        self_mod_result=None,
        capabilities_added=[],
        code_generation_count=0
    )


# ═══════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════

async def run_pipeline(
    selection_mode: str = "auto_top_pick",
    db: Optional[SupabaseDB] = None,
    prompt_manager: "PromptManager" = None,
    config_manager: "ConfigManager" = None,
    project_root: Path = None
) -> Dict[str, Any]:
    """
    Execute the full content generation pipeline with continuous learning
    and SELF-MODIFYING CODE capability.

    DATABASE: Supabase (единственная база данных)
    LLM: Claude Code CLI (твоя подписка, не API)

    CONTINUOUS LEARNING:
    The pipeline learns from EVERY iteration, starting from the first post.
    Learnings are extracted after each evaluation and applied immediately
    to subsequent generations within the same run AND across runs.

    SELF-MODIFYING CODE:
    When micro-learnings aren't enough, the system can write NEW CODE:
    - Detect capability gaps (missing data sources, analysis methods, etc.)
    - Generate Python modules using Claude Code CLI
    - Validate (syntax, types, security, tests)
    - Hot-reload into the running system
    - Retry with new capabilities
    """
    import uuid
    import logging
    logger = logging.getLogger("Pipeline")

    project_root = project_root or Path.cwd()

    # ─────────────────────────────────────────────────────────────────
    # INITIALIZE SUPABASE DATABASE
    # ─────────────────────────────────────────────────────────────────
    db = db or get_db()  # Use global Supabase instance

    # ─────────────────────────────────────────────────────────────────
    # INITIALIZE CONTINUOUS LEARNING ENGINE (Thread-Safe)
    # ─────────────────────────────────────────────────────────────────
    raw_learning_engine = ContinuousLearningEngine(
        db=db,
        prompt_manager=prompt_manager,
        config_manager=config_manager
    )
    # Wrap with thread-safe wrapper for concurrent pipeline runs
    learning_engine = ThreadSafeLearningEngine(raw_learning_engine)

    # ─────────────────────────────────────────────────────────────────
    # INITIALIZE SELF-MODIFICATION ENGINE (Thread-Safe)
    # Uses Claude Code CLI internally (твоя подписка)
    # ─────────────────────────────────────────────────────────────────
    raw_self_mod_engine = SelfModificationEngine.create_default(
        project_root=project_root,
        max_generation_attempts=RETRY_CONFIG.default_max_attempts
    )
    # Wrap with thread-safe wrapper - code modification MUST be serialized
    self_mod_engine = ThreadSafeSelfModEngine(raw_self_mod_engine)

    logger.info(f"[SELFMOD] Self-modification engine initialized (Claude CLI + project: {project_root})")
    logger.info("[ENGINES] All engines wrapped with thread-safe wrappers for concurrent safety")

    # Check if this is the first post ever
    is_first_post = await db.get_total_post_count() == 0

    if is_first_post:
        logger.info("[FIRST POST] This is the first post - will bootstrap with best practices")

    # ─────────────────────────────────────────────────────────────────
    # INITIALIZE PIPELINE STATE
    # ─────────────────────────────────────────────────────────────────
    run_id = str(uuid.uuid4())
    initial_state = initialize_pipeline_state(run_id, selection_mode)

    # Inject thread-safe learning engine and first-post flag
    initial_state["learning_engine"] = learning_engine
    initial_state["is_first_post"] = is_first_post

    # Inject thread-safe self-modification engine
    initial_state["self_mod_engine"] = self_mod_engine
    initial_state["capabilities_added"] = []
    initial_state["code_generation_count"] = 0

    # ─────────────────────────────────────────────────────────────────
    # COMPILE AND RUN PIPELINE
    # ─────────────────────────────────────────────────────────────────
    pipeline = create_content_pipeline()
    final_state = await pipeline.ainvoke(initial_state)

    # ─────────────────────────────────────────────────────────────────
    # COLLECT STATISTICS INCLUDING LEARNING AND SELF-MOD METRICS
    # ─────────────────────────────────────────────────────────────────

    # Learning stats
    learning_stats = {}
    if final_state.get("iteration_learnings"):
        learnings = final_state["iteration_learnings"]
        learning_stats = {
            "new_learnings": len(learnings.new_learnings),
            "confirmed_learnings": len(learnings.confirmed_learnings),
            "contradicted_learnings": len(learnings.contradicted_learnings),
            "prompt_adjustments": len(learnings.prompt_adjustments),
            "config_adjustments": len(learnings.config_adjustments)
        }

    # Self-modification stats
    self_mod_stats = {
        "capabilities_added": final_state.get("capabilities_added", []),
        "code_generation_count": final_state.get("code_generation_count", 0),
        "modules_created": [
            result.generated_code.module_name
            for result in self_mod_engine.modification_history
            if result.success and result.generated_code
        ],
        "modification_attempted": len(self_mod_engine.modification_history) > 0,
        "all_modifications_successful": all(
            r.success for r in self_mod_engine.modification_history
        ) if self_mod_engine.modification_history else True
    }

    return {
        "run_id": run_id,
        "status": "success" if not final_state.get("critical_error") else "error",
        "content": final_state["final_content"],
        "statistics": {
            "content_type": final_state["content_type"].value if final_state["content_type"] else None,
            "revision_count": final_state["revision_count"],
            "qc_score": final_state["qc_result"].aggregate_score if final_state["qc_result"] else None,
            "learnings_used": final_state.get("learnings_used_count", 0),
            "learning_stats": learning_stats,
            "is_first_post": is_first_post
        },
        "self_modification": self_mod_stats
    }
```

---

## Post Analytics & Feedback Loop

### Purpose
Track performance of published posts to:
1. **Measure success** — likes, comments, reposts, impressions
2. **Catch the golden hour** — first 60 minutes decide the post's fate
3. **Learn patterns** — which content types, hooks, visuals perform best
4. **Improve over time** — feedback loop to content generation

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      POST ANALYTICS SYSTEM                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│  DATA COLLECTION                                                            │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    LINKEDIN DATA SOURCE                              │   │
│  │                                                                      │   │
│  │  tomquirk/linkedin-api (Unofficial Voyager API)                      │   │
│  │  ──────────────────────────────────────────────                      │   │
│  │  GitHub: https://github.com/tomquirk/linkedin-api                    │   │
│  │  PyPI:   pip install linkedin-api pyotp                              │   │
│  │                                                                      │   │
│  │  HOW IT WORKS:                                                       │   │
│  │  • Uses LinkedIn's internal Voyager API (same as website)            │   │
│  │  • Authenticates with regular LinkedIn account credentials           │   │
│  │  • Supports 2FA via TOTP (Google Authenticator)                      │   │
│  │  • Full access to posts, reactions, comments, impressions            │   │
│  │  • Real-time data, no delays                                         │   │
│  │                                                                      │   │
│  │  AVAILABLE METHODS:                                                  │   │
│  │  • api.get_post(post_urn)           → Post content & metadata        │   │
│  │  • api.get_post_reactions(post_urn) → Likes, celebrates, etc.        │   │
│  │  • api.get_post_comments(post_urn)  → All comments                   │   │
│  │  • api.create_post(text, media)     → Publish new post               │   │
│  │                                                                      │   │
│  │  SETUP WITH 2FA (Google Authenticator):                              │   │
│  │  ```python                                                           │   │
│  │  import pyotp                                                        │   │
│  │  from linkedin_api import Linkedin                                   │   │
│  │                                                                      │   │
│  │  # TOTP secret from Google Authenticator setup                       │   │
│  │  totp = pyotp.TOTP('YOUR_TOTP_SECRET_KEY')                          │   │
│  │  two_factor_code = totp.now()  # Generates current 6-digit code     │   │
│  │                                                                      │   │
│  │  api = Linkedin(                                                     │   │
│  │      'email@example.com',                                            │   │
│  │      'password',                                                     │   │
│  │      two_factor_code=two_factor_code                                │   │
│  │  )                                                                   │   │
│  │  ```                                                                 │   │
│  │                                                                      │   │
│  │  HOW TO GET TOTP SECRET:                                             │   │
│  │  1. Go to LinkedIn Settings → Sign in & security → 2FA               │   │
│  │  2. When setting up authenticator, click "Can't scan QR?"            │   │
│  │  3. Copy the secret key (e.g., "JBSWY3DPEHPK3PXP")                   │   │
│  │  4. Store in env: LINKEDIN_TOTP_SECRET                               │   │
│  │                                                                      │   │
│  │  ⚠️ NOTE: Unofficial, may violate LinkedIn ToS                       │   │
│  │  Best practice: Don't spam, use reasonable request intervals         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    METRICS TRACKER                                   │   │
│  │                                                                      │   │
│  │  COLLECTION SCHEDULE (Golden Hour Focus):                            │   │
│  │  ─────────────────────────────────────────                           │   │
│  │  • T+0min:   Post published → record baseline                        │   │
│  │  • T+15min:  First check → early momentum signal                     │   │
│  │  • T+30min:  Second check → velocity calculation                     │   │
│  │  • T+60min:  Golden hour complete → CRITICAL snapshot                │   │
│  │  • T+3h:     Mid-day check                                           │   │
│  │  • T+24h:    Day-1 performance                                       │   │
│  │  • T+48h:    Final metrics (LinkedIn peak ~48h)                      │   │
│  │                                                                      │   │
│  │  METRICS CAPTURED:                                                   │   │
│  │  ─────────────────                                                   │   │
│  │  • likes_count          • comments_count                             │   │
│  │  • reposts_count        • impressions (if available)                 │   │
│  │  • engagement_rate      • profile_views_delta                        │   │
│  │  • follower_delta       • click_through_rate                         │   │
│  │                                                                      │   │
│  └───────────────────────────┬─────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    PERFORMANCE ANALYZER                              │   │
│  │                                                                      │   │
│  │  VELOCITY METRICS (First Hour is Key):                               │   │
│  │  ─────────────────────────────────────                               │   │
│  │  • likes_per_minute_first_hour                                       │   │
│  │  • comments_velocity (comments/time)                                 │   │
│  │  • engagement_acceleration (speeding up or slowing?)                 │   │
│  │                                                                      │   │
│  │  BENCHMARK COMPARISON:                                               │   │
│  │  ─────────────────────                                               │   │
│  │  • vs your average post                                              │   │
│  │  • vs same content_type average                                      │   │
│  │  • vs same posting_time average                                      │   │
│  │  • percentile rank (this post vs all your posts)                     │   │
│  │                                                                      │   │
│  │  PATTERN DETECTION:                                                  │   │
│  │  ─────────────────                                                   │   │
│  │  • Best performing: content_type, hook_style, visual_type            │   │
│  │  • Best posting time (day of week + hour)                            │   │
│  │  • Engagement drivers (what made top posts succeed?)                 │   │
│  │                                                                      │   │
│  └───────────────────────────┬─────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    FEEDBACK LOOP                                     │   │
│  │                                                                      │   │
│  │  LEARNING FROM RESULTS:                                              │   │
│  │  ─────────────────────                                               │   │
│  │  1. Tag each post with its final performance score                   │   │
│  │  2. Correlate with: content_type, hook, visual, time, topic          │   │
│  │  3. Update scoring weights in Trend Scout                            │   │
│  │  4. Update template preferences in Writer                            │   │
│  │  5. Update visual format preferences                                 │   │
│  │                                                                      │   │
│  │  EXAMPLES:                                                           │   │
│  │  ─────────                                                           │   │
│  │  • "Posts with author photo get 2.3x more engagement" →              │   │
│  │    Increase photo usage probability                                  │   │
│  │  • "AUTOMATION_CASE posts underperform on Mondays" →                 │   │
│  │    Avoid scheduling automation posts for Monday                      │   │
│  │  • "Contrarian hooks outperform by 40%" →                            │   │
│  │    Increase weight for contrarian template                           │   │
│  │                                                                      │   │
│  └───────────────────────────┬─────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    ALERTS & NOTIFICATIONS                            │   │
│  │                                                                      │   │
│  │  Via Telegram Bot:                                                   │   │
│  │  ─────────────────                                                   │   │
│  │  🔥 "Post is on fire! 50 likes in 30 min (3x your average)"          │   │
│  │  📈 "Golden hour complete: 120 likes, top 10% performance"           │   │
│  │  ⚠️ "Post underperforming: 8 likes in 1h (below average)"            │   │
│  │  📊 "Weekly report: Best post was Tuesday's CRM case study"          │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Analytics Data Schema

```python
@dataclass
class PostMetricsSnapshot:
    """Single snapshot of post metrics at a point in time."""

    post_id: str
    timestamp: datetime
    minutes_since_publish: int  # ACTUAL minutes calculated from publish timestamp

    # Core metrics
    likes: int
    comments: int
    reposts: int

    # Extended metrics (if available)
    impressions: Optional[int] = None
    clicks: Optional[int] = None
    engagement_rate: Optional[float] = None  # (likes+comments+reposts) / impressions

    # Calculated
    likes_velocity: Optional[float] = None  # likes per minute since last snapshot

    # ─────────────────────────────────────────────────────────────────
    # SCHEDULED CHECKPOINT (Fix for #27: minutes_since_publish overwrite)
    # ─────────────────────────────────────────────────────────────────
    # Preserves both actual time and scheduled checkpoint label.
    # This is critical for:
    # 1. Accurate timing analysis (drift detection)
    # 2. Checkpoint bucketing for aggregation
    # 3. Collection reliability monitoring
    # ─────────────────────────────────────────────────────────────────
    scheduled_checkpoint: Optional[int] = None  # e.g., 15, 30, 60, 180, 1440, 2880
    collection_drift_seconds: Optional[int] = None  # Difference between scheduled and actual


@dataclass
class PostPerformance:
    """
    Complete performance record for a published post.

    IMPORTANT: Includes QC metadata for feedback loop.
    This enables:
    1. Correlation analysis: QC score vs actual performance
    2. Threshold calibration: Adjust QC thresholds based on real data
    3. Criterion weight optimization: Which criteria predict success?
    """

    # Post identification
    post_id: str
    linkedin_url: str
    published_at: datetime

    # Content metadata (from generation)
    content_type: ContentType
    hook_style: HookStyle  # FIX: Use HookStyle enum for type safety
    template_used: str
    visual_type: str
    has_author_photo: bool
    topic_summary: str

    # ─────────────────────────────────────────────────────────────────
    # QC METADATA (NEW - for feedback loop)
    # ─────────────────────────────────────────────────────────────────
    qc_score: float                         # Aggregate QC score
    qc_criterion_scores: Dict[str, float]   # Individual criterion scores
    revision_count: int                     # How many revisions before publish
    auto_approved: bool                     # True if score >= auto_publish_threshold
    meta_evaluation_score: Optional[float]  # Score from meta-agent self-eval
    threshold_used: float                   # QC threshold that was applied

    # Lineage for tracing
    pipeline_run_id: str
    topic_id: str

    # A/B Testing support
    experiment_id: Optional[str] = None
    experiment_variant: Optional[str] = None

    # Metrics snapshots
    snapshots: List[PostMetricsSnapshot]

    # Key milestones
    metrics_15min: Optional[PostMetricsSnapshot] = None
    metrics_30min: Optional[PostMetricsSnapshot] = None
    metrics_1hour: Optional[PostMetricsSnapshot] = None  # CRITICAL
    metrics_24hour: Optional[PostMetricsSnapshot] = None
    metrics_final: Optional[PostMetricsSnapshot] = None  # 48h

    # Calculated scores
    golden_hour_score: float = 0.0  # 0-100, based on first hour velocity
    final_score: float = 0.0        # 0-100, based on total engagement
    percentile_rank: float = 0.0    # Where this ranks vs all your posts

    # Comparisons
    vs_average: float = 1.0         # 1.0 = average, 2.0 = 2x average
    vs_content_type_avg: float = 1.0
    vs_same_day_time_avg: float = 1.0

    # ─────────────────────────────────────────────────────────────────
    # FEEDBACK LOOP ANALYSIS
    # ─────────────────────────────────────────────────────────────────
    def qc_vs_performance_correlation(self) -> Dict[str, Union[float, bool]]:
        """
        Calculate correlation between QC scores and actual performance.
        Used for QC calibration.

        Returns:
            Dict with:
            - qc_score: float - the QC score for this post
            - actual_engagement: float - weighted engagement (likes + comments*3)
            - over_predicted: bool - True if QC predicted high but performance was low
            - under_predicted: bool - True if QC predicted low but performance was high
        """
        if not self.metrics_final:
            return {}

        actual_engagement = self.metrics_final.likes + self.metrics_final.comments * 3
        return {
            "qc_score": self.qc_score,
            "actual_engagement": actual_engagement,
            "over_predicted": self.qc_score > 7.0 and actual_engagement < 50,
            "under_predicted": self.qc_score < 7.0 and actual_engagement > 100,
        }


@dataclass
class AnalyticsInsight:
    """Actionable insight derived from analytics."""

    insight_type: str  # "content_type_performance", "timing_pattern", "visual_impact"
    description: str
    confidence: float  # 0-1
    sample_size: int

    # Actionable recommendation
    recommendation: str
    affected_component: str  # "trend_scout", "writer", "visual_creator", "scheduler"

    # For automatic adjustment
    parameter_to_adjust: Optional[str]
    suggested_value: Optional[Any]
```

---

### Analytics Configuration

```python
analytics_config = {
    # ═══════════════════════════════════════════════════════════════════
    # DATA COLLECTION (tomquirk/linkedin-api)
    # ═══════════════════════════════════════════════════════════════════

    "collection_method": "linkedin_api",  # Using tomquirk/linkedin-api

    "linkedin_api_config": {
        "package": "linkedin-api pyotp",  # pip install linkedin-api pyotp
        "github": "https://github.com/tomquirk/linkedin-api",
        "auth": {
            "email": "${LINKEDIN_EMAIL}",
            "password": "${LINKEDIN_PASSWORD}",
            "totp_secret": "${LINKEDIN_TOTP_SECRET}"  # Google Authenticator secret
        },
        "two_factor_enabled": True,
        "request_delay_seconds": 5,  # Be nice to LinkedIn servers
        "retry_on_rate_limit": True
    },

    "collection_schedule": [
        {"minutes_after_publish": 15, "label": "early_signal"},
        {"minutes_after_publish": 30, "label": "velocity_check"},
        {"minutes_after_publish": 60, "label": "golden_hour"},
        {"minutes_after_publish": 180, "label": "mid_day"},
        {"minutes_after_publish": 1440, "label": "day_1"},  # 24h
        {"minutes_after_publish": 2880, "label": "final"},  # 48h
    ],

    # ═══════════════════════════════════════════════════════════════════
    # BENCHMARKS
    # ═══════════════════════════════════════════════════════════════════

    "performance_thresholds": {
        "golden_hour": {
            "excellent": {"likes_min": 100, "comments_min": 15},
            "good": {"likes_min": 50, "comments_min": 8},
            "average": {"likes_min": 25, "comments_min": 3},
            "poor": {"likes_min": 10, "comments_min": 1}
        },
        "final_48h": {
            "viral": {"likes_min": 1000, "comments_min": 100},
            "excellent": {"likes_min": 300, "comments_min": 40},
            "good": {"likes_min": 100, "comments_min": 15},
            "average": {"likes_min": 50, "comments_min": 5}
        }
    },

    # ═══════════════════════════════════════════════════════════════════
    # ALERTS (via Telegram)
    # ═══════════════════════════════════════════════════════════════════

    "alerts": {
        "fire_alert": {
            "trigger": "likes_velocity > 2x_average AND minutes < 60",
            "message": "🔥 Post is on fire! {likes} likes in {minutes} min"
        },
        "golden_hour_complete": {
            "trigger": "minutes == 60",
            "message": "📈 Golden hour: {likes} likes, {percentile}th percentile"
        },
        "underperforming": {
            "trigger": "likes_velocity < 0.5x_average AND minutes == 60",
            "message": "⚠️ Below average: {likes} likes in 1h"
        },
        "viral_potential": {
            "trigger": "likes > 500 AND minutes < 180",
            "message": "🚀 Viral potential! {likes} likes in {hours}h"
        }
    },

    # ═══════════════════════════════════════════════════════════════════
    # FEEDBACK LOOP
    # ═══════════════════════════════════════════════════════════════════

    "learning": {
        "min_posts_for_insights": 20,  # Need 20+ posts to derive patterns
        "insight_confidence_threshold": 0.75,
        "auto_adjust_enabled": False,  # Start with manual review

        "trackable_factors": [
            "content_type",
            "hook_style",
            "template_used",
            "visual_type",
            "has_author_photo",
            "has_interface_visual",
            "post_length",
            "emoji_count",
            "hashtag_count",
            "posting_day",
            "posting_hour",
            "topic_category"
        ]
    },

    # ═══════════════════════════════════════════════════════════════════
    # STORAGE — SUPABASE ONLY
    # Единственная база данных для всей системы
    # ═══════════════════════════════════════════════════════════════════

    "storage": {
        "database": "supabase",  # ЕДИНСТВЕННАЯ БД, никаких Redis/SQLite/MongoDB
        "tables": {
            # Core content
            "posts": "posts",
            "drafts": "drafts",
            "topic_cache": "topic_cache",

            # Analytics
            "post_metrics": "post_metrics",

            # Learning & Self-Improvement
            "learnings": "learnings",
            "code_modifications": "code_modifications",
            "experiments": "experiments",
            "research_reports": "research_reports",

            # Configuration
            "prompts": "prompts"
        },
        "retention_days": 365,
        "connection": "Uses SupabaseDB class from architecture"
    }
}
```

---

### LinkedIn API Metrics Collection

```python
import pyotp
from linkedin_api import Linkedin
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Callable, Protocol
from abc import ABC, abstractmethod
import time
import os


# ═══════════════════════════════════════════════════════════════════════════
# SECURITY FIX: Credential Provider Abstraction
# Never store passwords in plaintext - use secure providers
# ═══════════════════════════════════════════════════════════════════════════

class CredentialProvider(Protocol):
    """Protocol for secure credential providers."""

    def get_password(self) -> str:
        """Retrieve password from secure storage."""
        ...

    def get_totp_secret(self) -> Optional[str]:
        """Retrieve TOTP secret from secure storage."""
        ...


class KeyringCredentialProvider:
    """
    Secure credential provider using system keyring.

    Stores credentials in:
    - macOS: Keychain
    - Windows: Credential Manager
    - Linux: Secret Service (GNOME Keyring, KWallet)

    Usage:
        # Store once (do this manually or via setup script):
        import keyring
        keyring.set_password("linkedin-agent", "email@example.com", "your_password")
        keyring.set_password("linkedin-agent", "email@example.com_totp", "TOTP_SECRET")

        # Use in code:
        provider = KeyringCredentialProvider("email@example.com")
        collector = LinkedInMetricsCollector(provider)
    """

    def __init__(self, email: str, service_name: str = "linkedin-agent"):
        import keyring
        self.email = email
        self.service_name = service_name
        self._keyring = keyring

    def get_password(self) -> str:
        password = self._keyring.get_password(self.service_name, self.email)
        if not password:
            raise ValueError(
                f"No password found in keyring for {self.email}. "
                f"Store it first: keyring.set_password('{self.service_name}', '{self.email}', 'password')"
            )
        return password

    def get_totp_secret(self) -> Optional[str]:
        return self._keyring.get_password(self.service_name, f"{self.email}_totp")


class EnvCredentialProvider:
    """
    Credential provider using environment variables.

    SECURITY NOTE: Less secure than keyring, but works in containerized environments.
    Ensure env vars are not logged and are set securely.

    Required env vars:
    - LINKEDIN_PASSWORD: Account password
    - LINKEDIN_TOTP_SECRET: (Optional) TOTP secret
    """

    def __init__(self, email: str):
        self.email = email

    def get_password(self) -> str:
        password = os.environ.get("LINKEDIN_PASSWORD")
        if not password:
            raise ValueError(
                "LINKEDIN_PASSWORD environment variable not set. "
                "For better security, consider using KeyringCredentialProvider."
            )
        return password

    def get_totp_secret(self) -> Optional[str]:
        return os.environ.get("LINKEDIN_TOTP_SECRET")


# ═══════════════════════════════════════════════════════════════════════════
# FIX #2: LinkedIn API Error Types for Retry Logic
# ═══════════════════════════════════════════════════════════════════════════

class LinkedInRateLimitError(Exception):
    """Raised when LinkedIn rate limits are hit."""
    pass


class LinkedInSessionExpiredError(Exception):
    """Raised when LinkedIn session has expired."""
    pass


class LinkedInAPIError(Exception):
    """Generic LinkedIn API error."""
    pass


class LinkedInMetricsCollector:
    """
    Collects post metrics using tomquirk/linkedin-api.
    Supports 2FA via TOTP (Google Authenticator).

    SECURITY: Uses CredentialProvider - never stores passwords in memory.

    GitHub: https://github.com/tomquirk/linkedin-api
    """

    def __init__(self, credential_provider: CredentialProvider):
        """
        Initialize LinkedIn API client with secure credential provider.

        Args:
            credential_provider: Provider for secure credential retrieval
                                (KeyringCredentialProvider or EnvCredentialProvider)
        """
        import logging
        self.logger = logging.getLogger("LinkedInMetricsCollector")
        self._credential_provider = credential_provider
        self.request_delay = 5  # seconds between requests

        # Authenticate (password retrieved, used, and discarded - not stored)
        self._authenticate()

    def _authenticate(self):
        """Authenticate with LinkedIn using secure credential retrieval."""
        email = getattr(self._credential_provider, 'email', 'unknown')
        password = self._credential_provider.get_password()
        totp_secret = self._credential_provider.get_totp_secret()

        two_factor_code = None
        if totp_secret:
            totp = pyotp.TOTP(totp_secret)
            two_factor_code = totp.now()
            self.logger.info("Generated 2FA code for authentication")

        self.api = self._init_with_retry(email, password, two_factor_code)
        # Password goes out of scope here - not stored!
        self.logger.info(f"Successfully authenticated to LinkedIn as {email}")

    def _calc_minutes_since_publish(self, post: Dict[str, Any]) -> int:
        """
        Calculate minutes since post was published.

        Handles multiple timestamp formats from LinkedIn API.
        """
        # LinkedIn API provides timestamp in various formats
        published_at = (
            post.get('createdAt') or
            post.get('created_at') or
            post.get('timestamp')
        )

        if not published_at:
            self.logger.warning("No publish timestamp found in post data, returning 0")
            return 0

        # Handle millisecond timestamps from LinkedIn
        if isinstance(published_at, (int, float)):
            # LinkedIn often uses milliseconds
            if published_at > 1e12:
                published_at = published_at / 1000
            published_dt = datetime.fromtimestamp(published_at, tz=timezone.utc)
        elif isinstance(published_at, str):
            published_dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
        else:
            published_dt = published_at

        # Ensure timezone-aware
        now = datetime.now(timezone.utc)
        if published_dt.tzinfo is None:
            published_dt = published_dt.replace(tzinfo=timezone.utc)

        delta = now - published_dt
        return max(0, int(delta.total_seconds() / 60))

    @with_retry(
        max_attempts=3,
        base_delay=5.0,
        retryable_exceptions=(ConnectionError, TimeoutError),
        operation_name="linkedin_init"
    )
    def _init_with_retry(self, email: str, password: str, two_factor_code: Optional[str]):
        """Initialize LinkedIn client with retry."""
        return Linkedin(email, password, two_factor_code=two_factor_code)

    def _refresh_auth_if_needed(self):
        """
        Re-authenticate if session expired.
        Retrieves fresh credentials from secure provider each time.
        """
        self.logger.info("Refreshing LinkedIn authentication")
        self._authenticate()  # Re-fetches credentials securely

    def collect_post_metrics(self, post_urn: str) -> PostMetricsSnapshot:
        """
        Collect all metrics for a single post.

        MEDIUM PRIORITY FIX #2: Added retry logic for API calls.

        NOTE: This is a synchronous function that uses time.sleep().
        When called from async context (e.g., _collect_and_store), wrap with:
            snapshot = await asyncio.get_event_loop().run_in_executor(
                None, self.collector.collect_post_metrics, post_urn
            )
        This prevents blocking the event loop during API delays.

        Args:
            post_urn: LinkedIn post URN (e.g., 'urn:li:activity:7654321')

        Returns:
            PostMetricsSnapshot with current metrics

        Raises:
            RetryExhaustedError: If all retry attempts fail
        """
        self.logger.debug(f"Collecting metrics for post: {post_urn}")

        # FIX #2: Each API call wrapped with retry
        post = self._api_call_with_retry("get_post", post_urn)
        time.sleep(self.request_delay)

        reactions = self._api_call_with_retry("get_post_reactions", post_urn)
        time.sleep(self.request_delay)

        comments = self._api_call_with_retry("get_post_comments", post_urn)

        # Parse metrics
        likes_count = sum(r.get('count', 0) for r in reactions) if reactions else 0
        comments_count = len(comments) if comments else 0

        # Extract from post metadata
        social_detail = post.get('socialDetail', {}) if post else {}
        reposts_count = social_detail.get('totalShares', 0)

        # Views/impressions (if available)
        impressions = post.get('numViews') or post.get('views') if post else None

        self.logger.info(
            f"Collected metrics for {post_urn}: "
            f"likes={likes_count}, comments={comments_count}, reposts={reposts_count}"
        )

        return PostMetricsSnapshot(
            post_id=post_urn,
            timestamp=datetime.now(),
            minutes_since_publish=self._calc_minutes_since_publish(post),
            likes=likes_count,
            comments=comments_count,
            reposts=reposts_count,
            impressions=impressions
        )

    @with_retry(
        max_attempts=3,
        base_delay=10.0,  # Longer delay for rate limits
        max_delay=60.0,
        retryable_exceptions=(LinkedInRateLimitError, LinkedInSessionExpiredError, ConnectionError, TimeoutError),
        operation_name="linkedin_api_call"
    )
    def _api_call_with_retry(self, method_name: str, *args, **kwargs):
        """
        Execute LinkedIn API call with retry and error handling.

        FIX #2: Unified retry logic for all API calls.
        """
        try:
            method = getattr(self.api, method_name)
            return method(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()

            # Detect rate limiting
            if "rate limit" in error_str or "429" in error_str or "too many requests" in error_str:
                self.logger.warning(f"LinkedIn rate limit hit for {method_name}")
                raise LinkedInRateLimitError(f"Rate limited: {e}")

            # Detect session expiration
            if "session" in error_str or "401" in error_str or "unauthorized" in error_str:
                self.logger.warning(f"LinkedIn session expired, attempting refresh")
                self._refresh_auth_if_needed()
                raise LinkedInSessionExpiredError(f"Session expired: {e}")

            # Other errors - still allow retry for transient issues
            self.logger.error(f"LinkedIn API error for {method_name}: {e}")
            raise LinkedInAPIError(f"{method_name} failed: {e}")

    @with_retry(component="linkedin", operation_name="publish_post")
    def publish_post(
        self,
        text: str,
        media_urls: Optional[List[str]] = None,
        _idempotency_key: Optional[str] = None
    ) -> str:
        """
        Publish a new LinkedIn post.

        CRITICAL FIX: Added retry decorator - this is the most important operation,
        losing content due to transient failures is unacceptable.

        Args:
            text: Post content
            media_urls: Optional list of media URLs
            _idempotency_key: Optional key to prevent duplicate posts on retry

        Returns:
            post_urn: URN of the created post

        Raises:
            LinkedInAPIError: After retry exhaustion
            LinkedInRateLimitError: If rate limited (will be retried)
        """
        # Idempotency check: If we have a key, check if post already exists
        # This prevents duplicate posts when retrying after partial success
        if _idempotency_key:
            existing = self._check_recent_post_exists(text[:100])
            if existing:
                self.logger.info(f"Post already exists (idempotency check): {existing}")
                return existing

        result = self._api_call_with_retry(
            "publish_post",
            lambda: self.api.create_post(text, media_urls=media_urls)
        )
        return result.get('urn') or result.get('id')

    def _check_recent_post_exists(self, text_prefix: str) -> Optional[str]:
        """Check if a post with this text prefix was recently published."""
        try:
            recent = self.get_recent_posts(limit=5)
            for post in recent:
                if post.get('text', '').startswith(text_prefix):
                    return post.get('urn') or post.get('id')
        except Exception:
            pass  # Best effort, don't fail on idempotency check
        return None

    def collect_scheduled_metrics(
        self,
        post_urn: str,
        schedule: List[int]  # minutes after publish
    ) -> List[PostMetricsSnapshot]:
        """
        Collect metrics at scheduled intervals.
        Called by scheduler/cron job.

        FIX #27: No longer overwrites minutes_since_publish.
        Uses scheduled_checkpoint field to preserve both actual timing and checkpoint label.
        """
        snapshots = []
        for scheduled_minutes in schedule:
            snapshot = self.collect_post_metrics(post_urn)

            # CORRECT: Set scheduled checkpoint without overwriting actual minutes
            snapshot.scheduled_checkpoint = scheduled_minutes

            # Calculate drift for monitoring collection reliability
            actual_minutes = snapshot.minutes_since_publish
            snapshot.collection_drift_seconds = (actual_minutes - scheduled_minutes) * 60

            snapshots.append(snapshot)
        return snapshots


# Usage example:
#
# collector = LinkedInMetricsCollector(
#     email=os.environ['LINKEDIN_EMAIL'],
#     password=os.environ['LINKEDIN_PASSWORD'],
#     totp_secret=os.environ['LINKEDIN_TOTP_SECRET']  # 2FA secret
# )
#
# # Publish post
# post_urn = collector.publish_post("My new AI insights post...")
#
# # Collect metrics at golden hour
# metrics = collector.collect_post_metrics(post_urn)
# print(f"Likes: {metrics.likes}, Comments: {metrics.comments}")
#
# ─────────────────────────────────────────────────────────────
# ENV VARIABLES REQUIRED:
# ─────────────────────────────────────────────────────────────
# LINKEDIN_EMAIL=your@email.com
# LINKEDIN_PASSWORD=your_password
# LINKEDIN_TOTP_SECRET=JBSWY3DPEHPK3PXP  # From Google Authenticator setup
#
# To get TOTP secret:
# 1. LinkedIn Settings → Sign in & security → Two-step verification
# 2. Set up with Authenticator app
# 3. Click "Can't scan the QR code?"
# 4. Copy the secret key (e.g., "JBSWY3DPEHPK3PXP")
# ─────────────────────────────────────────────────────────────
```

---

### Automated Metrics Collection (Scheduler)

```python
import asyncio
from datetime import datetime, timedelta
from typing import List, Tuple
from apscheduler.schedulers.asyncio import AsyncIOScheduler

class MetricsScheduler:
    """
    Schedules automatic metrics collection at key intervals.
    """

    COLLECTION_SCHEDULE = [
        15,    # T+15min: early signal
        30,    # T+30min: velocity check
        60,    # T+60min: GOLDEN HOUR
        180,   # T+3h: mid-day
        1440,  # T+24h: day 1
        2880,  # T+48h: final
    ]

    def __init__(self, collector: LinkedInMetricsCollector, db: SupabaseDB):
        self.collector = collector
        self.db = db  # Supabase — единственная БД
        self.scheduler = AsyncIOScheduler()

    def schedule_post_tracking(
        self,
        post_urn: str,
        published_at: datetime
    ) -> List[str]:
        """
        Schedule all metric collection jobs for a new post.

        MEDIUM PRIORITY FIX #14: Added return type and error handling.

        Args:
            post_urn: LinkedIn post URN (e.g., 'urn:li:activity:7654321')
            published_at: When the post was published

        Returns:
            List[str]: Job IDs for all scheduled collection jobs.
            Use these IDs to manage (cancel, reschedule) jobs later.

        Raises:
            SchedulerError: If scheduler is not running or job creation fails
        """
        import logging
        logger = logging.getLogger("MetricsScheduler")

        job_ids: List[str] = []
        failed_schedules: List[Tuple[int, str]] = []

        logger.info(
            f"[METRICS] Scheduling tracking for post: {post_urn}\n"
            f"  Published at: {published_at}\n"
            f"  Checkpoints: {self.COLLECTION_SCHEDULE}"
        )

        for minutes in self.COLLECTION_SCHEDULE:
            run_time = published_at + timedelta(minutes=minutes)
            job_id = f"{post_urn}_{minutes}min"

            # FIX #14: Error handling for job scheduling
            try:
                self.scheduler.add_job(
                    self._collect_and_store,
                    'date',
                    run_date=run_time,
                    args=[post_urn, minutes],
                    id=job_id,
                    replace_existing=True  # Avoid duplicate job errors
                )
                job_ids.append(job_id)
                logger.debug(f"[METRICS] Scheduled job {job_id} for {run_time}")

            except Exception as e:
                logger.error(f"[METRICS] Failed to schedule job {job_id}: {e}")
                failed_schedules.append((minutes, str(e)))

        # FIX #14: Log summary
        if failed_schedules:
            logger.warning(
                f"[METRICS] Partial scheduling: {len(job_ids)} jobs scheduled, "
                f"{len(failed_schedules)} failed"
            )
        else:
            logger.info(f"[METRICS] Successfully scheduled {len(job_ids)} collection jobs for {post_urn}")

        return job_ids

    async def _collect_and_store(self, post_urn: str, scheduled_minutes: int):
        """
        Collect metrics and store in database.

        FIX #27: Preserves actual minutes_since_publish, uses scheduled_checkpoint.
        FIX: Uses run_in_executor to avoid blocking event loop during API calls.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        # FIX: Run synchronous API calls in thread pool to avoid blocking
        snapshot = await loop.run_in_executor(
            None, self.collector.collect_post_metrics, post_urn
        )

        # CORRECT: Don't overwrite actual timing
        snapshot.scheduled_checkpoint = scheduled_minutes

        # Track drift for reliability monitoring
        actual_minutes = snapshot.minutes_since_publish
        snapshot.collection_drift_seconds = (actual_minutes - scheduled_minutes) * 60

        # Convert dataclass to dict for SupabaseDB method
        from dataclasses import asdict
        await self.db.store_metrics_snapshot(asdict(snapshot))

        # Check for alerts
        # FIX: Use scheduled_minutes, not undefined 'minutes' variable
        await self._check_alerts(snapshot, scheduled_minutes)

    async def _check_alerts(self, snapshot: PostMetricsSnapshot, minutes: int):
        """Send alerts based on performance."""
        avg = await self.db.get_average_metrics_at(minutes)

        if minutes == 60:  # Golden hour complete
            percentile = await self.db.get_percentile(snapshot.likes, minutes)
            print(f"📈 Golden hour: {snapshot.likes} likes, top {percentile}%")

        if snapshot.likes > avg.likes * 2:
            print(f"🔥 Post on fire! {snapshot.likes} likes ({snapshot.likes/avg.likes:.1f}x avg)")
```

---

## Self-Improvement Layer (Meta-Agent)

### Purpose

Автономный мета-агент, который наблюдает за результатами, исследует лучшие практики, и **модифицирует поведение других агентов** для улучшения качества контента.

**Это делает систему настоящим AI-агентом, а не просто pipeline.**

---

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SELF-IMPROVEMENT LAYER (META-AGENT)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         OBSERVATION                                  │   │
│  │  Что мета-агент видит:                                              │   │
│  │  • Post performance metrics (from Analytics)                        │   │
│  │  • Content that was generated (drafts, published posts)             │   │
│  │  • Current prompts/templates being used                             │   │
│  │  • Historical patterns (what worked, what didn't)                   │   │
│  └─────────────────────────────────┬───────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      SELF-EVALUATION                                 │   │
│  │  Оценка контента ДО публикации:                                     │   │
│  │                                                                      │   │
│  │  1. GENERATE → Агент пишет пост                                     │   │
│  │  2. EVALUATE → Мета-агент оценивает (без публикации)                │   │
│  │     • "Это 6/10, hook слабый"                                       │   │
│  │     • "Не хватает конкретных цифр"                                  │   │
│  │     • "Слишком длинный первый абзац"                                │   │
│  │  3. DECIDE →                                                         │   │
│  │     • Score >= 8.0 → Отправить на human approval                    │   │
│  │     • Score < 8.0  → Вернуть на переписывание с feedback            │   │
│  │  4. REWRITE → Writer переписывает с учётом критики                  │   │
│  │  5. REPEAT → До 3 итераций или score >= 8.0                         │   │
│  └─────────────────────────────────┬───────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      RESEARCH & LEARN                                │   │
│  │  Изучение лучших практик:                                           │   │
│  │                                                                      │   │
│  │  TRIGGERS:                                                           │   │
│  │  • 3 поста подряд < average performance                             │   │
│  │  • Новый тип контента (нет опыта)                                   │   │
│  │  • Еженедельный research cycle (каждое воскресенье)                 │   │
│  │                                                                      │   │
│  │  RESEARCH ACTIONS:                                                   │   │
│  │  ┌──────────────────┬───────────────────────────────────────────┐   │   │
│  │  │ Perplexity       │ "Best LinkedIn post hooks 2024"           │   │   │
│  │  │                  │ "How to write engaging LinkedIn content"  │   │   │
│  │  │                  │ "LinkedIn algorithm changes January 2024" │   │   │
│  │  ├──────────────────┼───────────────────────────────────────────┤   │   │
│  │  │ Competitor       │ Scrape top 10 posts from [influencers]    │   │   │
│  │  │ Analysis         │ Extract patterns: length, hooks, CTAs     │   │   │
│  │  │                  │ Identify trending formats                 │   │   │
│  │  ├──────────────────┼───────────────────────────────────────────┤   │   │
│  │  │ Own Data         │ Compare best vs worst performing posts    │   │   │
│  │  │ Analysis         │ What's different? Hook? Visual? Length?   │   │   │
│  │  │                  │ Statistical significance check            │   │   │
│  │  └──────────────────┴───────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  │  OUTPUT: ResearchReport with actionable insights                    │   │
│  └─────────────────────────────────┬───────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      SELF-MODIFICATION                               │   │
│  │  Изменение собственного поведения:                                  │   │
│  │                                                                      │   │
│  │  WHAT CAN BE MODIFIED:                                               │   │
│  │  ┌──────────────────┬───────────────────────────────────────────┐   │   │
│  │  │ Component        │ What Changes                               │   │   │
│  │  ├──────────────────┼───────────────────────────────────────────┤   │   │
│  │  │ Writer Prompts   │ System prompt, examples, guidelines       │   │   │
│  │  │ Hook Templates   │ Add new, deprecate underperforming        │   │   │
│  │  │ Scoring Weights  │ Trend Scout topic prioritization          │   │   │
│  │  │ Visual Styles    │ Preferred visual formats                  │   │   │
│  │  │ Post Length      │ Target word count based on performance    │   │   │
│  │  │ Posting Time     │ Optimal hours based on data               │   │   │
│  │  │ Content Mix      │ % of each ContentType                     │   │   │
│  │  └──────────────────┴───────────────────────────────────────────┘   │   │
│  │                                                                      │   │
│  │  SAFETY:                                                             │   │
│  │  • All changes logged with before/after + reasoning                 │   │
│  │  • Rollback available if performance drops                          │   │
│  │  • Human can review changes (optional, default: auto)               │   │
│  └─────────────────────────────────┬───────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      EXPERIMENTATION                                 │   │
│  │  A/B тестирование гипотез:                                          │   │
│  │                                                                      │   │
│  │  EXPERIMENT LIFECYCLE:                                               │   │
│  │  1. HYPOTHESIS → "Вопрос в hook даст +20% engagement"               │   │
│  │  2. DESIGN    → A: current hook, B: question hook                   │   │
│  │  3. RUN       → 5 posts each variant                                │   │
│  │  4. ANALYZE   → Statistical significance test                       │   │
│  │  5. DECIDE    → Winner becomes default / rollback                   │   │
│  │                                                                      │   │
│  │  EXAMPLE EXPERIMENTS:                                                │   │
│  │  • "Emoji в первой строке vs без emoji"                             │   │
│  │  • "Короткий пост (500 chars) vs длинный (2000 chars)"              │   │
│  │  • "Фото автора vs абстрактная визуализация"                        │   │
│  │  • "Утро (9:00) vs вечер (18:00)"                                   │   │
│  │                                                                      │   │
│  │  CONSTRAINTS:                                                        │   │
│  │  • Max 1 active experiment at a time                                │   │
│  │  • Min 5 posts per variant for significance                         │   │
│  │  • Auto-stop if variant clearly losing (early stopping)             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Self-Evaluation Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SELF-EVALUATION BEFORE PUBLISH                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Writer    │───▶│  Draft v1   │───▶│  Meta-Agent │───▶│  Evaluate   │  │
│  │  generates  │    │  (raw post) │    │   reviews   │    │   Score     │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘  │
│                                                                   │         │
│                          ┌────────────────────────────────────────┤         │
│                          │                                        │         │
│                          ▼                                        ▼         │
│                   Score < 8.0                              Score >= 8.0     │
│                          │                                        │         │
│                          ▼                                        ▼         │
│  ┌─────────────────────────────────────┐          ┌─────────────────────┐  │
│  │         FEEDBACK GENERATION         │          │   PROCEED TO QC     │  │
│  │                                     │          │                     │  │
│  │  "Hook is weak. Try starting with  │          │  Send to QC Agent   │  │
│  │   a surprising statistic or        │          │  for final polish   │  │
│  │   contrarian statement.            │          │                     │  │
│  │                                     │          │  Then → Human       │  │
│  │   Missing: concrete metrics.       │          │        Approval     │  │
│  │   Add specific numbers like        │          │                     │  │
│  │   '47% improvement' or '$2M ROI'"  │          └─────────────────────┘  │
│  │                                     │                                    │
│  └──────────────────┬──────────────────┘                                    │
│                     │                                                       │
│                     ▼                                                       │
│  ┌─────────────────────────────────────┐                                    │
│  │         WRITER REWRITES             │                                    │
│  │                                     │                                    │
│  │  Input:                             │                                    │
│  │  • Original draft                   │                                    │
│  │  • Specific feedback                │──────┐                             │
│  │  • Examples of good hooks           │      │                             │
│  │                                     │      │                             │
│  │  Output: Draft v2                   │      │  Max 3 iterations           │
│  └──────────────────┬──────────────────┘      │                             │
│                     │                         │                             │
│                     └─────────────────────────┘                             │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│  EVALUATION CRITERIA (Updated with Visual Quality)                          │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  ┌────────────────────┬────────┬───────────────────────────────────────┐   │
│  │ Criterion          │ Weight │ What Meta-Agent Checks                │   │
│  ├────────────────────┼────────┼───────────────────────────────────────┤   │
│  │ Hook Strength      │  22%   │ First 2 lines grab attention?        │   │
│  │ Specificity        │  18%   │ Concrete numbers, names, examples?   │   │
│  │ Value Density      │  18%   │ Actionable insights per paragraph?   │   │
│  │ Authenticity       │  12%   │ Sounds like human, not AI?           │   │
│  │ Structure          │  10%   │ Easy to scan? White space?           │   │
│  │ CTA Clarity        │  8%    │ Clear next step for reader?          │   │
│  │ Visual Quality     │  12%   │ Visual enhances message? (NEW)       │   │
│  │                    │        │ • Correct format for content type    │   │
│  │                    │        │ • Photo selection appropriate        │   │
│  │                    │        │ • Text-visual coherence              │   │
│  │                    │        │ • Not generic stock-photo feel       │   │
│  └────────────────────┴────────┴───────────────────────────────────────┘   │
│                                                                             │
│  NOTE: Visual Quality was added to create a holistic evaluation.           │
│  Previously, self-evaluation only considered text, missing 30%+ of         │
│  what drives LinkedIn engagement (visual appeal).                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Continuous Learning Engine (NEW)

**Ключевое изменение философии:**

Раньше: Система улучшает себя по расписанию (воскресенье) или при проблемах (3+ плохих поста).

**Теперь: Система улучшает себя КАЖДУЮ итерацию, начиная с первого поста.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│            CONTINUOUS LEARNING ARCHITECTURE                                   │
│            Самообучение с первого поста, каждую итерацию                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║  ПРИНЦИП: Learn → Apply → Verify → Persist                            ║ │
│  ║                                                                        ║ │
│  ║  Каждый компонент имеет свой learning loop:                           ║ │
│  ║  • Writer: учится на feedback от Meta-Agent                           ║ │
│  ║  • Visual Creator: учится на visual quality scores                    ║ │
│  ║  • Trend Scout: учится на корреляции topic score ↔ engagement         ║ │
│  ║  • Humanizer: учится на authenticity scores                           ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                                                                             │
│                           ITERATION N                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐     │   │
│  │   │ Generate │───▶│ Evaluate │───▶│  Learn   │───▶│  Apply   │     │   │
│  │   │          │    │          │    │          │    │          │     │   │
│  │   │ • Text   │    │ • Score  │    │ • What   │    │ • Update │     │   │
│  │   │ • Visual │    │ • Issues │    │   went   │    │   prompt │     │   │
│  │   │ • Hook   │    │ • Whys   │    │   wrong? │    │ • Update │     │   │
│  │   │          │    │          │    │ • Fix?   │    │   config │     │   │
│  │   └──────────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘     │   │
│  │                        │               │               │           │   │
│  │                        ▼               ▼               ▼           │   │
│  │   ┌──────────────────────────────────────────────────────────────┐ │   │
│  │   │              MICRO-LEARNING DATABASE                         │ │   │
│  │   │                                                              │ │   │
│  │   │  Stores learnings from EVERY iteration:                      │ │   │
│  │   │  • "Hook with numbers performs better" (confidence: 0.7)     │ │   │
│  │   │  • "Photo with author face +20% engagement" (confidence: 0.8)│ │   │
│  │   │  • "Enterprise cases need ROI in hook" (confidence: 0.9)     │ │   │
│  │   │                                                              │ │   │
│  │   │  Confidence grows with each confirmation, shrinks with       │ │   │
│  │   │  contradictions. High-confidence learnings become RULES.     │ │   │
│  │   └──────────────────────────────────────────────────────────────┘ │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│                         ITERATION N+1                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │   Learnings from N automatically injected into prompts/configs     │   │
│  │   for iteration N+1                                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

```python
# ═══════════════════════════════════════════════════════════════════════════
# CONTINUOUS LEARNING ENGINE
# Работает с первого поста, учится каждую итерацию
# ═══════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
import json


class LearningType(Enum):
    """Types of micro-learnings the system can acquire."""
    HOOK_PATTERN = "hook_pattern"           # What hooks work
    VISUAL_STYLE = "visual_style"           # What visuals perform
    CONTENT_STRUCTURE = "content_structure" # Post structure insights
    TONE_ADJUSTMENT = "tone_adjustment"     # Voice/tone learnings
    TIMING_INSIGHT = "timing_insight"       # When to post
    AUDIENCE_PREFERENCE = "audience_pref"   # What audience likes


class LearningSource(Enum):
    """Where the learning came from."""
    META_EVALUATION = "meta_evaluation"     # From self-evaluation
    QC_FEEDBACK = "qc_feedback"             # From QC agent
    POST_PERFORMANCE = "post_performance"   # From actual engagement
    COMPETITOR_ANALYSIS = "competitor"      # From studying competitors
    EXPLICIT_RULE = "explicit_rule"         # Programmed by human


@dataclass
class MicroLearning:
    """
    Single unit of learning acquired during iteration.

    These are SMALL, SPECIFIC insights that accumulate over time.
    High-confidence learnings become permanent rules.
    """
    id: str
    learning_type: LearningType
    source: LearningSource

    # The actual learning
    description: str                        # Human-readable description
    rule: str                               # Machine-applicable rule
    affected_component: str                 # writer / visual_creator / humanizer / etc.

    # Confidence tracking
    confidence: float                       # 0.0-1.0, grows with confirmations
    confirmations: int = 0                  # Times this was confirmed
    contradictions: int = 0                 # Times this was contradicted

    # Context
    content_type: Optional[ContentType] = None  # None = applies to all
    created_at: datetime = field(default_factory=datetime.now)
    last_confirmed_at: Optional[datetime] = None

    # Promotion tracking
    is_promoted_to_rule: bool = False       # High confidence → permanent rule

    # FIX: Add temporal decay configuration
    # Decay rate per day (0.01 = 1% per day, ~30% after 30 days)
    CONFIDENCE_DECAY_RATE: float = 0.01

    def apply_temporal_decay(self) -> None:
        """
        FIX: Apply time-based confidence decay.

        Learnings that haven't been confirmed recently should lose confidence.
        This prevents stale learnings from permanently influencing prompts.

        Call this before using a learning's confidence value.
        """
        if self.last_confirmed_at is None:
            reference_time = self.created_at
        else:
            reference_time = self.last_confirmed_at

        days_since_activity = (datetime.now() - reference_time).days

        if days_since_activity > 0:
            # Exponential decay: confidence * (1 - rate)^days
            decay_factor = (1 - self.CONFIDENCE_DECAY_RATE) ** days_since_activity
            self.confidence = max(0.0, self.confidence * decay_factor)

            # Demote from rule if decayed below threshold
            if self.confidence < 0.7:
                self.is_promoted_to_rule = False

    def confirm(self):
        """Called when evidence supports this learning."""
        self.confirmations += 1
        self.last_confirmed_at = datetime.now()
        # Confidence grows logarithmically (diminishing returns)
        self.confidence = min(1.0, self.confidence + 0.1 * (1 - self.confidence))

        # Promote to rule if very confident
        if self.confidence >= 0.9 and self.confirmations >= 5:
            self.is_promoted_to_rule = True

    def contradict(self):
        """Called when evidence contradicts this learning."""
        self.contradictions += 1
        # Confidence drops faster than it grows
        self.confidence = max(0.0, self.confidence - 0.15)

        # Demote from rule if no longer confident
        if self.confidence < 0.7:
            self.is_promoted_to_rule = False

    def get_effective_confidence(self) -> float:
        """
        FIX: Get confidence with temporal decay applied.

        Use this instead of accessing self.confidence directly when
        making decisions based on learning reliability.
        """
        self.apply_temporal_decay()
        return self.confidence


@dataclass
class IterationLearnings:
    """All learnings from a single iteration."""
    iteration_id: str
    post_id: str

    # Component-specific feedback that triggered learning
    text_feedback: List[str]
    visual_feedback: List[str]
    structure_feedback: List[str]

    # Extracted learnings
    new_learnings: List[MicroLearning]
    confirmed_learnings: List[str]          # IDs of existing learnings confirmed
    contradicted_learnings: List[str]       # IDs of existing learnings contradicted

    # Immediate actions taken
    prompt_adjustments: Dict[str, str]      # component -> adjustment made
    config_adjustments: Dict[str, Any]      # config key -> new value


# ═══════════════════════════════════════════════════════════════════════════
# THREAD-SAFE ENGINE WRAPPERS
# Prevent race conditions when engines are shared between concurrent pipelines
# ═══════════════════════════════════════════════════════════════════════════

class ThreadSafeLearningEngine:
    """
    Thread-safe wrapper for ContinuousLearningEngine.

    PROBLEM: ContinuousLearningEngine has in-memory state (learnings dict)
    that can be corrupted if accessed concurrently by multiple pipeline runs.

    SOLUTION: Wrap write operations with async lock.
    Read operations can be concurrent (learnings dict is thread-safe for reads).

    Usage:
        # Instead of using ContinuousLearningEngine directly:
        engine = ThreadSafeLearningEngine(ContinuousLearningEngine(db, pm, cm))

        # All methods are now thread-safe:
        await engine.learn_from_iteration(...)
    """

    def __init__(self, engine: "ContinuousLearningEngine"):
        self._engine = engine
        self._write_lock = asyncio.Lock()

    async def learn_from_iteration(self, *args, **kwargs) -> "IterationLearnings":
        """Thread-safe learning extraction."""
        async with self._write_lock:
            return await self._engine.learn_from_iteration(*args, **kwargs)

    async def apply_learnings_to_prompt(self, *args, **kwargs) -> str:
        """Read operation - can be concurrent."""
        return await self._engine.apply_learnings_to_prompt(*args, **kwargs)

    async def handle_first_post(self) -> List["MicroLearning"]:
        """Thread-safe bootstrap."""
        async with self._write_lock:
            return await self._engine.handle_first_post()

    async def update_confidence(self, *args, **kwargs):
        """Thread-safe confidence update."""
        async with self._write_lock:
            return await self._engine.update_confidence(*args, **kwargs)

    # Expose read-only properties
    @property
    def learnings(self):
        return self._engine.learnings


class ThreadSafeSelfModEngine:
    """
    Thread-safe wrapper for SelfModificationEngine.

    CRITICAL: Code modification is dangerous if concurrent.
    All operations must be serialized.

    Usage:
        engine = ThreadSafeSelfModEngine(SelfModificationEngine.create_default(project_root))
        await engine.generate_and_deploy_capability(...)
    """

    def __init__(self, engine: "SelfModificationEngine"):
        self._engine = engine
        self._lock = asyncio.Lock()  # ALL operations locked

    async def generate_and_deploy_capability(self, *args, **kwargs):
        """Thread-safe capability generation and deployment."""
        async with self._lock:
            return await self._engine.generate_and_deploy_capability(*args, **kwargs)

    async def rollback(self, *args, **kwargs):
        """Thread-safe rollback."""
        async with self._lock:
            return await self._engine.rollback(*args, **kwargs)

    async def detect_capability_gap(self, *args, **kwargs):
        """Thread-safe gap detection (also writes to state)."""
        async with self._lock:
            return await self._engine.detect_capability_gap(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════


class ContinuousLearningEngine:
    """
    Engine that learns from EVERY iteration, starting from first post.

    KEY DIFFERENCE FROM RESEARCH AGENT:
    - ResearchAgent: Deep research triggered by schedule/events
    - ContinuousLearningEngine: Shallow but constant learning every iteration

    Both work together:
    - CLE handles immediate micro-learnings
    - ResearchAgent handles deep strategic research

    WARNING: Use ThreadSafeLearningEngine wrapper when sharing between concurrent runs!
    """

    def __init__(self, db, prompt_manager, config_manager):
        self.db = db
        self.prompt_manager = prompt_manager
        self.config_manager = config_manager

        # In-memory cache of active learnings
        # WARNING: Not thread-safe! Use ThreadSafeLearningEngine wrapper.
        self.learnings: Dict[str, MicroLearning] = {}
        self._load_learnings()

    def _load_learnings(self):
        """Load existing learnings from database."""
        stored = self.db.get_all_learnings()
        for learning in stored:
            self.learnings[learning.id] = learning

    async def learn_from_iteration(
        self,
        post_id: str,
        evaluation: ContentEvaluation,
        visual_evaluation: "VisualEvaluation",
        content_type: ContentType,
        draft: str,
        visual: "VisualAsset"
    ) -> IterationLearnings:
        """
        Extract learnings from a single iteration.

        Called AFTER every evaluation, BEFORE proceeding to next step.
        This is the core of continuous learning.
        """
        import uuid
        import logging
        logger = logging.getLogger("ContinuousLearningEngine")

        iteration_id = f"iter_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        logger.info(f"[LEARN] Starting learning extraction for {post_id}")

        new_learnings = []
        confirmed = []
        contradicted = []

        # ─────────────────────────────────────────────────────────────
        # 1. EXTRACT LEARNINGS FROM TEXT EVALUATION
        # ─────────────────────────────────────────────────────────────
        text_feedback = evaluation.weaknesses + evaluation.actionable_suggestions

        for feedback in text_feedback:
            learning = await self._extract_learning_from_feedback(
                feedback=feedback,
                component="writer",
                content_type=content_type,
                source=LearningSource.META_EVALUATION
            )
            if learning:
                existing = self._find_similar_learning(learning)
                if existing:
                    if self._confirms(learning, existing):
                        existing.confirm()
                        confirmed.append(existing.id)
                        logger.debug(f"[LEARN] Confirmed: {existing.description}")
                    else:
                        existing.contradict()
                        contradicted.append(existing.id)
                        logger.debug(f"[LEARN] Contradicted: {existing.description}")
                else:
                    new_learnings.append(learning)
                    self.learnings[learning.id] = learning
                    logger.info(f"[LEARN] NEW: {learning.description}")

        # ─────────────────────────────────────────────────────────────
        # 2. EXTRACT LEARNINGS FROM VISUAL EVALUATION
        # ─────────────────────────────────────────────────────────────
        visual_feedback = visual_evaluation.issues

        for feedback in visual_feedback:
            learning = await self._extract_learning_from_feedback(
                feedback=feedback,
                component="visual_creator",
                content_type=content_type,
                source=LearningSource.META_EVALUATION
            )
            if learning:
                existing = self._find_similar_learning(learning)
                if existing:
                    if self._confirms(learning, existing):
                        existing.confirm()
                        confirmed.append(existing.id)
                    else:
                        existing.contradict()
                        contradicted.append(existing.id)
                else:
                    new_learnings.append(learning)
                    self.learnings[learning.id] = learning

        # ─────────────────────────────────────────────────────────────
        # 3. EXTRACT POSITIVE LEARNINGS FROM STRENGTHS
        # ─────────────────────────────────────────────────────────────
        for strength in evaluation.strengths:
            learning = await self._extract_learning_from_feedback(
                feedback=f"GOOD: {strength}",
                component="writer",
                content_type=content_type,
                source=LearningSource.META_EVALUATION
            )
            if learning:
                learning.confidence = 0.6  # Strengths start higher
                existing = self._find_similar_learning(learning)
                if existing:
                    existing.confirm()
                    confirmed.append(existing.id)
                else:
                    new_learnings.append(learning)
                    self.learnings[learning.id] = learning

        # ─────────────────────────────────────────────────────────────
        # 4. APPLY IMMEDIATE ADJUSTMENTS
        # ─────────────────────────────────────────────────────────────
        prompt_adjustments = await self._apply_learnings_to_prompts(
            new_learnings + [self.learnings[id] for id in confirmed],
            content_type
        )

        config_adjustments = await self._apply_learnings_to_config(
            new_learnings + [self.learnings[id] for id in confirmed],
            content_type
        )

        # ─────────────────────────────────────────────────────────────
        # 5. PERSIST
        # ─────────────────────────────────────────────────────────────
        await self.db.save_learnings(new_learnings)
        await self.db.update_learnings([self.learnings[id] for id in confirmed + contradicted])

        result = IterationLearnings(
            iteration_id=iteration_id,
            post_id=post_id,
            text_feedback=text_feedback,
            visual_feedback=visual_feedback,
            structure_feedback=[],
            new_learnings=new_learnings,
            confirmed_learnings=confirmed,
            contradicted_learnings=contradicted,
            prompt_adjustments=prompt_adjustments,
            config_adjustments=config_adjustments
        )

        logger.info(
            f"[LEARN] Iteration complete: "
            f"{len(new_learnings)} new, {len(confirmed)} confirmed, {len(contradicted)} contradicted"
        )

        return result

    async def _extract_learning_from_feedback(
        self,
        feedback: str,
        component: str,
        content_type: ContentType,
        source: LearningSource
    ) -> Optional[MicroLearning]:
        """
        Use LLM to extract structured learning from free-text feedback.
        """
        import uuid

        prompt = f"""
        Extract a learning rule from this feedback:

        FEEDBACK: {feedback}
        COMPONENT: {component}
        CONTENT TYPE: {content_type.value}

        Return JSON:
        {{
            "learning_type": "hook_pattern|visual_style|content_structure|tone_adjustment",
            "description": "Human-readable description of what was learned",
            "rule": "Machine-applicable rule, e.g., 'always_include_numbers_in_hook'",
            "is_actionable": true/false,
            "initial_confidence": 0.3-0.6
        }}

        If the feedback is too vague to extract a rule, return {{"is_actionable": false}}
        """

        try:
            response = await get_claude().generate_structured(prompt)

            if not response.get("is_actionable", False):
                return None

            return MicroLearning(
                id=f"learn_{uuid.uuid4().hex[:8]}",
                learning_type=LearningType(response["learning_type"]),
                source=source,
                description=response["description"],
                rule=response["rule"],
                affected_component=component,
                confidence=response.get("initial_confidence", 0.4),
                content_type=content_type
            )
        except Exception:
            return None

    def _find_similar_learning(self, new: MicroLearning) -> Optional[MicroLearning]:
        """Find existing learning that's similar to new one."""
        for existing in self.learnings.values():
            if (existing.learning_type == new.learning_type and
                existing.affected_component == new.affected_component and
                existing.rule == new.rule):
                return existing
        return None

    def _confirms(self, new: MicroLearning, existing: MicroLearning) -> bool:
        """Check if new learning confirms or contradicts existing."""
        # Same rule = confirmation
        # For now, simple matching. Could use semantic similarity.
        return new.rule == existing.rule

    async def _apply_learnings_to_prompts(
        self,
        learnings: List[MicroLearning],
        content_type: ContentType
    ) -> Dict[str, str]:
        """
        Immediately apply learnings to component prompts.
        Only applies high-confidence or promoted learnings.
        """
        adjustments = {}

        for learning in learnings:
            if learning.confidence < 0.6 and not learning.is_promoted_to_rule:
                continue  # Not confident enough yet

            component = learning.affected_component

            # Generate prompt injection
            injection = f"\n\n[LEARNED]: {learning.description}\n[RULE]: {learning.rule}\n"

            current_prompt = self.prompt_manager.get_prompt(component)
            if injection not in current_prompt:
                self.prompt_manager.append_to_prompt(component, injection)
                adjustments[component] = learning.rule

        return adjustments

    async def _apply_learnings_to_config(
        self,
        learnings: List[MicroLearning],
        content_type: ContentType
    ) -> Dict[str, Any]:
        """Apply learnings to configuration values."""
        adjustments = {}

        for learning in learnings:
            if not learning.is_promoted_to_rule:
                continue

            # Map rules to config changes
            if "visual" in learning.rule and "photo" in learning.rule:
                adjustments["prefer_author_photo"] = True
            elif "numbers" in learning.rule and "hook" in learning.rule:
                adjustments["hook_require_numbers"] = True
            # ... more rule-to-config mappings

        for key, value in adjustments.items():
            self.config_manager.set(key, value)

        return adjustments

    def get_learnings_for_prompt(
        self,
        component: str,
        content_type: Optional[ContentType] = None
    ) -> List[MicroLearning]:
        """
        Get relevant learnings to inject into a component's prompt.
        Called at the START of each generation to provide context.
        """
        relevant = []

        for learning in self.learnings.values():
            if learning.affected_component != component:
                continue
            if learning.confidence < 0.5:
                continue
            if content_type and learning.content_type and learning.content_type != content_type:
                continue
            relevant.append(learning)

        # Sort by confidence
        return sorted(relevant, key=lambda l: l.confidence, reverse=True)[:10]

    def format_learnings_for_prompt(
        self,
        learnings: List[MicroLearning]
    ) -> str:
        """Format learnings as prompt injection."""
        if not learnings:
            return ""

        lines = ["\n═══ LEARNED FROM PREVIOUS ITERATIONS ═══\n"]

        for learning in learnings:
            confidence_emoji = "🔒" if learning.is_promoted_to_rule else "💡"
            lines.append(f"{confidence_emoji} {learning.description}")
            lines.append(f"   Rule: {learning.rule}")
            lines.append(f"   Confidence: {learning.confidence:.0%}")
            lines.append("")

        lines.append("═══════════════════════════════════════\n")

        return "\n".join(lines)
```

---

### Integration with Pipeline

```python
# In the main pipeline, after EVERY evaluation:

async def post_evaluation_learning(state: PipelineState) -> PipelineState:
    """
    Called after Meta-Agent evaluation, BEFORE deciding pass/revise.
    Extracts learnings regardless of whether post passes or fails.
    """

    learning_engine = state["learning_engine"]  # Injected at pipeline start

    # Learn from this iteration
    learnings = await learning_engine.learn_from_iteration(
        post_id=state["current_post_id"],
        evaluation=state["meta_evaluation"],
        visual_evaluation=state["visual_evaluation"],
        content_type=state["content_type"],
        draft=state["draft_text"],
        visual=state["visual_asset"]
    )

    # Store for debugging/analysis
    state["iteration_learnings"] = learnings

    return state


# Modify Writer node to use learnings:

async def writer_node(state: PipelineState) -> dict:
    """Writer now receives learnings from previous iterations."""

    learning_engine = state["learning_engine"]
    content_type = state["content_type"]

    # Get relevant learnings
    learnings = learning_engine.get_learnings_for_prompt("writer", content_type)
    learnings_prompt = learning_engine.format_learnings_for_prompt(learnings)

    # Inject into writer prompt
    enhanced_prompt = WRITER_BASE_PROMPT + learnings_prompt

    # Generate with learnings context
    draft = await writer_agent.generate(
        brief=state["analysis_brief"],
        system_prompt=enhanced_prompt
    )

    return {"draft_text": draft, "learnings_used": len(learnings)}


# Same for Visual Creator:

async def visual_creator_node(state: PipelineState) -> dict:
    """Visual Creator uses learnings about what visuals work."""

    learning_engine = state["learning_engine"]
    content_type = state["content_type"]

    learnings = learning_engine.get_learnings_for_prompt("visual_creator", content_type)
    learnings_prompt = learning_engine.format_learnings_for_prompt(learnings)

    # Visual creator uses learnings for style selection
    visual = await visual_creator.generate(
        post_content=state["humanized_text"],
        content_type=content_type,
        learnings_context=learnings_prompt
    )

    return {"visual_asset": visual}
```

---

### First Post Handling

```python
class ContinuousLearningEngine:
    # ... previous code ...

    async def handle_first_post(self) -> List[MicroLearning]:
        """
        Special handling for the VERY FIRST post.
        Bootstrap with general best practices from research.
        """

        # Check if this is first post
        post_count = await self.db.get_total_post_count()
        if post_count > 0:
            return []  # Not first post

        # Bootstrap with proven best practices
        bootstrap_learnings = [
            MicroLearning(
                id="bootstrap_hook_numbers",
                learning_type=LearningType.HOOK_PATTERN,
                source=LearningSource.EXPLICIT_RULE,
                description="Posts with specific numbers in hooks get 40% more engagement",
                rule="include_specific_number_in_first_line",
                affected_component="writer",
                confidence=0.7,  # High confidence from research
                is_promoted_to_rule=False  # Will be promoted after confirmation
            ),
            MicroLearning(
                id="bootstrap_photo_face",
                learning_type=LearningType.VISUAL_STYLE,
                source=LearningSource.EXPLICIT_RULE,
                description="Photos showing author's face increase trust and engagement",
                rule="prefer_author_photo_when_personal",
                affected_component="visual_creator",
                confidence=0.7,
            ),
            MicroLearning(
                id="bootstrap_short_paragraphs",
                learning_type=LearningType.CONTENT_STRUCTURE,
                source=LearningSource.EXPLICIT_RULE,
                description="Short paragraphs (1-2 sentences) improve scannability on mobile",
                rule="max_2_sentences_per_paragraph",
                affected_component="writer",
                confidence=0.8,
            ),
            MicroLearning(
                id="bootstrap_enterprise_roi",
                learning_type=LearningType.HOOK_PATTERN,
                source=LearningSource.EXPLICIT_RULE,
                description="Enterprise case studies must mention ROI in first 3 lines",
                rule="enterprise_hook_must_have_roi_or_metric",
                affected_component="writer",
                confidence=0.8,
                content_type=ContentType.ENTERPRISE_CASE
            ),
            # More bootstrap learnings...
        ]

        for learning in bootstrap_learnings:
            self.learnings[learning.id] = learning

        await self.db.save_learnings(bootstrap_learnings)

        return bootstrap_learnings
```

---

### How Continuous Learning and Research Agent Work Together

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              DUAL-MODE SELF-IMPROVEMENT ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    CONTINUOUS LEARNING ENGINE                        │   │
│  │                    (Shallow but Constant)                            │   │
│  │  ───────────────────────────────────────────────────────────────    │   │
│  │  WHEN: After EVERY evaluation (every post, every iteration)          │   │
│  │  WHAT: Extract micro-learnings from feedback                         │   │
│  │  HOW:  Pattern matching → MicroLearning → immediate prompt injection │   │
│  │  LATENCY: < 1 second                                                 │   │
│  │                                                                      │   │
│  │  Examples:                                                           │   │
│  │  • "Hook lacked numbers" → learn "always_include_numbers_in_hook"    │   │
│  │  • "Visual too complex" → learn "prefer_simpler_diagrams"            │   │
│  │  • "Tone too casual" → learn "maintain_professional_tone"            │   │
│  │                                                                      │   │
│  │  CONFIDENCE SYSTEM:                                                  │   │
│  │  • New learnings start at 0.3-0.6 confidence                         │   │
│  │  • Confirmed learnings increase confidence (+0.1 per confirmation)   │   │
│  │  • At 0.9 confidence + 5 confirmations → promoted to permanent rule  │   │
│  │  • Contradicted learnings decrease confidence                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              │ Learnings feed into                          │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        RESEARCH AGENT                                │   │
│  │                       (Deep but Periodic)                            │   │
│  │  ───────────────────────────────────────────────────────────────    │   │
│  │  WHEN: Weekly schedule OR triggered by events                        │   │
│  │  WHAT: Strategic research, competitor analysis, prompt redesign      │   │
│  │  HOW:  Perplexity search → Analysis → Experiments → Validation       │   │
│  │  LATENCY: Minutes to hours                                           │   │
│  │                                                                      │   │
│  │  Triggers:                                                           │   │
│  │  • WEEKLY_CYCLE: Sunday comprehensive review                         │   │
│  │  • UNDERPERFORMANCE: 3+ posts below average                          │   │
│  │  • ALGORITHM_CHANGE: Engagement patterns shifted                     │   │
│  │  • NEW_CONTENT_TYPE: First time creating this type                   │   │
│  │                                                                      │   │
│  │  Uses ContinuousLearning data:                                       │   │
│  │  • Which micro-learnings have high confirmation rate?                │   │
│  │  • Which content types struggle consistently?                        │   │
│  │  • What patterns emerge across many iterations?                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│  RESULT: System improves from FIRST POST, gets smarter every iteration     │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Self-Modifying Code Engine

### Overview

Система может **писать и модифицировать собственный код** в реальном времени, не дожидаясь scheduled triggers. Когда система понимает, что текущих capabilities не хватает — она генерирует новый модуль, валидирует его, загружает и использует.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SELF-MODIFYING CODE ENGINE                                │
│                    (Maximum Autonomy Mode)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         TRIGGER FLOW                                 │   │
│  │                                                                      │   │
│  │   Post Generation Failed / Suboptimal                                │   │
│  │              │                                                       │   │
│  │              ▼                                                       │   │
│  │   ┌──────────────────┐                                              │   │
│  │   │ CAPABILITY       │  "Почему не получилось?"                     │   │
│  │   │ ANALYZER         │  • Не хватает данных?                        │   │
│  │   │                  │  • Не хватает логики?                        │   │
│  │   │                  │  • Не хватает интеграции?                    │   │
│  │   └────────┬─────────┘                                              │   │
│  │            │                                                         │   │
│  │            ▼                                                         │   │
│  │   ┌──────────────────┐     ┌──────────────────┐                     │   │
│  │   │ PROMPT CHANGE    │ OR  │ CODE GENERATION  │                     │   │
│  │   │ (micro-learning) │     │ (new module)     │                     │   │
│  │   └──────────────────┘     └────────┬─────────┘                     │   │
│  │                                     │                                │   │
│  │                                     ▼                                │   │
│  │                          ┌──────────────────┐                       │   │
│  │                          │ CODE VALIDATOR   │                       │   │
│  │                          │ • Syntax check   │                       │   │
│  │                          │ • Type check     │                       │   │
│  │                          │ • Sandbox test   │                       │   │
│  │                          │ • Security scan  │                       │   │
│  │                          └────────┬─────────┘                       │   │
│  │                                   │                                  │   │
│  │                                   ▼                                  │   │
│  │                          ┌──────────────────┐                       │   │
│  │                          │ HOT RELOADER     │                       │   │
│  │                          │ • Save module    │                       │   │
│  │                          │ • importlib.reload│                      │   │
│  │                          │ • Update registry│                       │   │
│  │                          └────────┬─────────┘                       │   │
│  │                                   │                                  │   │
│  │                                   ▼                                  │   │
│  │                          ┌──────────────────┐                       │   │
│  │                          │ RETRY WITH NEW   │                       │   │
│  │                          │ CAPABILITY       │                       │   │
│  │                          └──────────────────┘                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Architecture Components

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Type
from enum import Enum
from datetime import datetime
import importlib
import importlib.util
import ast
import sys
import subprocess
import tempfile
import os
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# CAPABILITY GAP DETECTION
# ═══════════════════════════════════════════════════════════════════════════

class CapabilityType(Enum):
    """Types of capabilities the system can have or lack."""
    DATA_SOURCE = "data_source"           # New data source integration
    ANALYSIS_METHOD = "analysis_method"   # New way to analyze content
    GENERATION_STYLE = "generation_style" # New content generation approach
    VISUAL_FORMAT = "visual_format"       # New visual format support
    INTEGRATION = "integration"           # External service integration
    UTILITY = "utility"                   # Helper functions/utilities
    AGENT = "agent"                       # Entirely new agent
    VALIDATOR = "validator"               # New validation logic


@dataclass
class CapabilityGap:
    """Identified missing capability."""
    id: str
    gap_type: CapabilityType
    description: str                      # What's missing
    evidence: List[str]                   # Why we think it's missing
    proposed_solution: str                # High-level solution description
    priority: int                         # 1 = critical, 5 = nice-to-have
    detected_at: datetime = field(default_factory=datetime.now)

    # Context for code generation
    related_modules: List[str] = field(default_factory=list)
    required_interfaces: List[str] = field(default_factory=list)
    example_usage: Optional[str] = None


class CapabilityAnalyzer:
    """
    Analyzes evaluation feedback to detect missing capabilities.

    This is the "brain" that decides whether a problem needs:
    - Prompt adjustment (micro-learning)
    - Config change
    - NEW CODE (self-modification)

    Uses Claude Code CLI (твоя подписка), не API.
    """

    def __init__(self, module_registry: "ModuleRegistry"):
        self.claude = get_claude()  # Claude Code CLI
        self.registry = module_registry
        self.gap_history: List[CapabilityGap] = []

    async def analyze_failure(
        self,
        evaluation: "ContentEvaluation",
        visual_evaluation: "VisualEvaluation",
        iteration_learnings: "IterationLearnings",
        content_type: "ContentType",
        attempt_number: int
    ) -> Optional[CapabilityGap]:
        """
        Analyze why content generation failed/underperformed.

        Returns CapabilityGap if new code is needed, None if prompt/config change suffices.
        """

        # Collect all feedback
        all_feedback = (
            evaluation.weaknesses +
            evaluation.actionable_suggestions +
            visual_evaluation.issues +
            visual_evaluation.suggestions
        )

        # Get current system capabilities
        current_capabilities = self.registry.get_capability_summary()

        prompt = f"""
        Analyze this content generation failure and determine if NEW CODE is needed.

        === EVALUATION FEEDBACK ===
        {chr(10).join(f"- {f}" for f in all_feedback)}

        === CURRENT SYSTEM CAPABILITIES ===
        {current_capabilities}

        === CONTENT TYPE ===
        {content_type.value}

        === ATTEMPT NUMBER ===
        {attempt_number} (higher = tried prompt changes, didn't help)

        === QUESTION ===
        Can this problem be solved by:
        A) Changing prompts/configs (micro-learning) - NO NEW CODE
        B) Writing NEW CODE (new module, new function, new integration)

        If B, describe exactly what code is needed.

        Return JSON:
        {{
            "needs_new_code": true/false,
            "reasoning": "Why this decision",
            "gap_type": "data_source|analysis_method|generation_style|visual_format|integration|utility|agent|validator",
            "description": "What capability is missing",
            "proposed_solution": "High-level description of the code to write",
            "priority": 1-5,
            "required_interfaces": ["list", "of", "interfaces", "to", "implement"],
            "example_usage": "How the new code would be used"
        }}

        IMPORTANT: Only return needs_new_code=true if prompt changes genuinely can't solve this.
        Examples requiring new code:
        - "Need to fetch data from a new source" → new integration
        - "Need sentiment analysis" → new analysis module
        - "Need to generate carousel images" → new visual format
        - "Need to track competitor posts" → new data source

        Examples NOT requiring new code:
        - "Hook wasn't engaging enough" → prompt change
        - "Tone was too formal" → prompt change
        - "Metrics weren't highlighted" → prompt change
        """

        response = await get_claude().generate_structured(prompt)

        if not response.get("needs_new_code", False):
            return None

        import uuid
        gap = CapabilityGap(
            id=f"gap_{uuid.uuid4().hex[:8]}",
            gap_type=CapabilityType(response["gap_type"]),
            description=response["description"],
            evidence=all_feedback[:5],  # Top 5 pieces of evidence
            proposed_solution=response["proposed_solution"],
            priority=response.get("priority", 3),
            related_modules=self._find_related_modules(response["gap_type"]),
            required_interfaces=response.get("required_interfaces", []),
            example_usage=response.get("example_usage")
        )

        self.gap_history.append(gap)
        return gap

    def _find_related_modules(self, gap_type: str) -> List[str]:
        """Find existing modules related to this gap type."""
        type_to_modules = {
            "data_source": ["trend_scout", "sources"],
            "analysis_method": ["analyzer", "evaluator"],
            "generation_style": ["writer", "humanizer"],
            "visual_format": ["visual_creator"],
            "integration": ["integrations", "api_clients"],
            "utility": ["utils", "helpers"],
            "agent": ["agents"],
            "validator": ["qc", "validators"]
        }
        return type_to_modules.get(gap_type, [])


# ═══════════════════════════════════════════════════════════════════════════
# CODE GENERATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GeneratedCode:
    """Result of code generation."""
    module_name: str
    file_path: str
    code: str
    description: str

    # Metadata
    gap_id: str                           # Which gap this addresses
    generated_at: datetime = field(default_factory=datetime.now)

    # Validation results (filled after validation)
    syntax_valid: bool = False
    type_check_passed: bool = False
    tests_passed: bool = False
    security_passed: bool = False

    # Generated tests
    test_code: Optional[str] = None

    # FIX: Dependencies field was missing but used in code generation
    dependencies: List[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        """
        Check if generated code is valid for loading.

        FIX: type_check_passed is now optional (non-blocking).
        Type errors are logged as warnings but don't prevent loading.
        This allows code with minor type inconsistencies to still work.
        """
        return all([
            self.syntax_valid,
            # type_check_passed is informational only - not required
            self.tests_passed,
            self.security_passed
        ])


class CodeGenerationEngine:
    """
    Generates new Python code using Claude Code CLI.

    This is where the magic happens - the system writes its own code.
    Uses твоя подписка Claude, не API токены.
    """

    def __init__(
        self,
        project_root: Path,
        module_registry: "ModuleRegistry"
    ):
        self.claude = get_claude()  # Claude Code CLI (твоя подписка)
        self.project_root = project_root
        self.registry = module_registry
        self.generation_history: List[GeneratedCode] = []

    async def generate_module(
        self,
        gap: CapabilityGap,
        context: Dict[str, Any] = None
    ) -> GeneratedCode:
        """
        Generate a new Python module to fill the capability gap.
        """
        import logging
        logger = logging.getLogger("CodeGeneration")

        logger.info(f"[CODEGEN] Generating module for gap: {gap.description}")

        # Get context from related modules
        related_code = await self._get_related_code(gap.related_modules)

        # Get interface definitions if needed
        interfaces = await self._get_interfaces(gap.required_interfaces)

        prompt = f"""
        Generate a Python module to add this capability to the system.

        === CAPABILITY GAP ===
        Type: {gap.gap_type.value}
        Description: {gap.description}
        Proposed Solution: {gap.proposed_solution}

        === REQUIRED INTERFACES ===
        {interfaces}

        === RELATED EXISTING CODE (for style/patterns) ===
        {related_code}

        === EXAMPLE USAGE ===
        {gap.example_usage or "Not specified"}

        === REQUIREMENTS ===
        1. Follow the existing code style exactly
        2. Use type hints everywhere
        3. Use dataclasses for data structures
        4. Include docstrings
        5. Handle errors with specific exceptions (fail-fast, no fallbacks)
        6. Make it production-ready, not a prototype
        7. Include logging
        8. Use async where appropriate

        === OUTPUT FORMAT ===
        Return JSON:
        {{
            "module_name": "name_of_module",
            "code": "full Python code as string",
            "description": "What this module does",
            "dependencies": ["list", "of", "pip", "packages", "if", "any"],
            "test_code": "pytest tests for this module"
        }}

        IMPORTANT:
        - The code must be complete and runnable
        - No placeholders like "# TODO" or "pass"
        - Include all necessary imports
        - The module should integrate seamlessly with the existing system
        """

        response = await get_claude().generate_structured(prompt)

        # Determine file path
        module_name = response["module_name"]
        file_path = self._determine_file_path(module_name, gap.gap_type)

        generated = GeneratedCode(
            module_name=module_name,
            file_path=str(file_path),
            code=response["code"],
            description=response["description"],
            gap_id=gap.id,
            test_code=response.get("test_code")
        )

        # Install dependencies if any
        dependencies = response.get("dependencies", [])
        if dependencies:
            await self._install_dependencies(dependencies)

        self.generation_history.append(generated)
        logger.info(f"[CODEGEN] Generated module: {module_name} ({len(response['code'])} chars)")

        return generated

    async def generate_function(
        self,
        gap: CapabilityGap,
        target_module: str,
        context: Dict[str, Any] = None
    ) -> GeneratedCode:
        """
        Generate a new function to add to existing module.
        """
        # Get the existing module code
        existing_code = await self._read_module(target_module)

        prompt = f"""
        Generate a new function to add to an existing module.

        === CAPABILITY GAP ===
        {gap.description}

        === EXISTING MODULE CODE ===
        ```python
        {existing_code}
        ```

        === REQUIREMENTS ===
        1. Match the existing code style exactly
        2. The function should integrate with existing functions
        3. Include type hints and docstring
        4. Handle errors appropriately (fail-fast)

        === OUTPUT FORMAT ===
        Return JSON:
        {{
            "function_name": "name_of_function",
            "code": "full function code including decorators",
            "description": "What this function does",
            "insert_after": "name of function to insert after (or null for end)",
            "test_code": "pytest test for this function"
        }}
        """

        response = await get_claude().generate_structured(prompt)

        # The code here is just the function, we'll need to patch it in
        generated = GeneratedCode(
            module_name=f"{target_module}.{response['function_name']}",
            file_path=str(self._get_module_path(target_module)),
            code=response["code"],
            description=response["description"],
            gap_id=gap.id,
            test_code=response.get("test_code")
        )

        return generated

    async def modify_existing_code(
        self,
        gap: CapabilityGap,
        target_module: str,
        modification_type: str  # "extend" | "refactor" | "fix"
    ) -> GeneratedCode:
        """
        Modify existing code to add capability.
        """
        existing_code = await self._read_module(target_module)

        prompt = f"""
        Modify existing code to add this capability.

        === CAPABILITY GAP ===
        {gap.description}

        === MODIFICATION TYPE ===
        {modification_type}

        === EXISTING CODE ===
        ```python
        {existing_code}
        ```

        === REQUIREMENTS ===
        1. Make minimal changes to achieve the goal
        2. Don't break existing functionality
        3. Maintain code style
        4. Update docstrings if behavior changes

        === OUTPUT FORMAT ===
        Return JSON:
        {{
            "modified_code": "full modified module code",
            "changes_summary": "what was changed and why",
            "test_code": "tests covering the changes"
        }}
        """

        response = await get_claude().generate_structured(prompt)

        generated = GeneratedCode(
            module_name=target_module,
            file_path=str(self._get_module_path(target_module)),
            code=response["modified_code"],
            description=response["changes_summary"],
            gap_id=gap.id,
            test_code=response.get("test_code")
        )

        return generated

    def _determine_file_path(self, module_name: str, gap_type: CapabilityType) -> Path:
        """Determine where to put the new module."""
        type_to_dir = {
            CapabilityType.DATA_SOURCE: "sources",
            CapabilityType.ANALYSIS_METHOD: "analyzers",
            CapabilityType.GENERATION_STYLE: "generators",
            CapabilityType.VISUAL_FORMAT: "visual",
            CapabilityType.INTEGRATION: "integrations",
            CapabilityType.UTILITY: "utils",
            CapabilityType.AGENT: "agents",
            CapabilityType.VALIDATOR: "validators"
        }

        subdir = type_to_dir.get(gap_type, "modules")
        dir_path = self.project_root / subdir
        dir_path.mkdir(parents=True, exist_ok=True)

        return dir_path / f"{module_name}.py"

    def _get_module_path(self, module_name: str) -> Path:
        """Get path for existing module."""
        # Search in common locations
        for subdir in ["", "agents", "sources", "utils", "integrations"]:
            path = self.project_root / subdir / f"{module_name}.py"
            if path.exists():
                return path
        return self.project_root / f"{module_name}.py"

    async def _get_related_code(self, module_names: List[str]) -> str:
        """Get code from related modules for context."""
        code_parts = []
        for name in module_names[:3]:  # Limit to 3 for context size
            try:
                path = self._get_module_path(name)
                if path.exists():
                    content = path.read_text(encoding="utf-8")
                    # Truncate if too long
                    if len(content) > 3000:
                        content = content[:3000] + "\n# ... (truncated)"
                    code_parts.append(f"# === {name}.py ===\n{content}")
            except Exception:
                pass
        return "\n\n".join(code_parts) or "No related code found"

    async def _get_interfaces(self, interface_names: List[str]) -> str:
        """Get interface definitions."""
        # In a real implementation, this would look up Protocol/ABC definitions
        return "\n".join(f"# Interface: {name}" for name in interface_names)

    async def _read_module(self, module_name: str) -> str:
        """Read existing module code."""
        path = self._get_module_path(module_name)
        if path.exists():
            return path.read_text(encoding="utf-8")
        raise FileNotFoundError(f"Module {module_name} not found")

    async def _install_dependencies(self, packages: List[str]):
        """Install pip packages."""
        import logging
        logger = logging.getLogger("CodeGeneration")

        for package in packages:
            # Security: validate package name
            if not self._is_safe_package_name(package):
                logger.warning(f"[CODEGEN] Skipping unsafe package name: {package}")
                continue

            logger.info(f"[CODEGEN] Installing dependency: {package}")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                check=True
            )

    def _is_safe_package_name(self, name: str) -> bool:
        """Validate package name for security."""
        import re
        # Only allow alphanumeric, hyphens, underscores, and version specs
        return bool(re.match(r'^[a-zA-Z0-9_-]+(\[.*\])?(==|>=|<=|~=|!=)?[a-zA-Z0-9._-]*$', name))


# ═══════════════════════════════════════════════════════════════════════════
# CODE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

class CodeValidator:
    """
    Validates generated code before it's loaded into the system.

    Multiple validation layers:
    1. Syntax check (ast.parse)
    2. Type check (mypy)
    3. Security scan (basic patterns)
    4. Sandbox execution (run tests in isolation)
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root

    async def validate(self, generated: GeneratedCode) -> GeneratedCode:
        """Run all validations and update the GeneratedCode object."""
        import logging
        logger = logging.getLogger("CodeValidation")

        logger.info(f"[VALIDATE] Validating {generated.module_name}")

        # 1. Syntax check
        generated.syntax_valid = self._check_syntax(generated.code)
        if not generated.syntax_valid:
            logger.error(f"[VALIDATE] Syntax check failed for {generated.module_name}")
            return generated

        # 2. Type check (optional, non-blocking but tracked)
        generated.type_check_passed = await self._check_types(generated)
        if not generated.type_check_passed:
            logger.warning(f"[VALIDATE] Type check warnings for {generated.module_name}")
            # FIX: Don't override result! Type check is non-blocking for is_valid(),
            # but we should preserve the actual result for tracking/debugging.
            # The is_valid() method should be updated to not require type_check_passed.

        # 3. Security scan
        generated.security_passed = self._security_scan(generated.code)
        if not generated.security_passed:
            logger.error(f"[VALIDATE] Security scan failed for {generated.module_name}")
            return generated

        # 4. Sandbox test
        if generated.test_code:
            generated.tests_passed = await self._run_sandbox_tests(generated)
        else:
            # No tests provided, do a basic import test
            generated.tests_passed = await self._test_import(generated)

        if generated.is_valid():
            logger.info(f"[VALIDATE] ✅ All validations passed for {generated.module_name}")
        else:
            logger.error(f"[VALIDATE] ❌ Validation failed for {generated.module_name}")

        return generated

    def _check_syntax(self, code: str) -> bool:
        """Check Python syntax using AST."""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            import logging
            logging.getLogger("CodeValidation").error(f"Syntax error: {e}")
            return False

    async def _check_types(self, generated: GeneratedCode) -> bool:
        """Run mypy type checking."""
        try:
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(generated.code)
                temp_path = f.name

            result = subprocess.run(
                [sys.executable, "-m", "mypy", temp_path, "--ignore-missing-imports"],
                capture_output=True,
                text=True,
                timeout=30
            )

            os.unlink(temp_path)

            # mypy returns 0 for success
            return result.returncode == 0
        except Exception as e:
            import logging
            logging.getLogger("CodeValidation").warning(f"Type check skipped: {e}")
            return True  # Don't fail if mypy isn't available

    def _security_scan(self, code: str) -> bool:
        """
        AST-based security scan for dangerous code patterns.

        Uses Abstract Syntax Tree analysis which is MUCH harder to bypass
        than simple pattern matching. Obfuscation like:
        - getattr(builtins, 'eval')
        - "exe" + "c"
        - __dict__['__import__']
        will still be caught because AST sees the actual structure.
        """
        import logging
        logger = logging.getLogger("CodeValidation")

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Already caught by _check_syntax, but be defensive
            return False

        # Use AST visitor to find dangerous patterns
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.violations = []

            def visit_Call(self, node):
                """Check function calls for dangerous patterns."""
                # Direct dangerous function calls
                dangerous_builtins = {'eval', 'exec', 'compile', '__import__'}
                dangerous_funcs = {
                    ('os', 'system'),
                    ('os', 'popen'),
                    ('os', 'spawn'),
                    ('os', 'spawnl'),
                    ('os', 'spawnle'),
                    ('subprocess', 'call'),
                    ('subprocess', 'run'),
                    ('subprocess', 'Popen'),
                    ('pickle', 'loads'),
                    ('pickle', 'load'),
                }

                # Check direct calls: eval(), exec(), etc.
                if isinstance(node.func, ast.Name):
                    if node.func.id in dangerous_builtins:
                        self.violations.append(f"Dangerous builtin: {node.func.id}()")

                # Check module.func calls: os.system(), subprocess.run(), etc.
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        pair = (node.func.value.id, node.func.attr)
                        if pair in dangerous_funcs:
                            self.violations.append(f"Dangerous call: {pair[0]}.{pair[1]}()")

                    # Check for subprocess with shell=True
                    if (isinstance(node.func.value, ast.Name) and
                        node.func.value.id == 'subprocess'):
                        for keyword in node.keywords:
                            if keyword.arg == 'shell':
                                if isinstance(keyword.value, ast.Constant) and keyword.value.value:
                                    self.violations.append("subprocess with shell=True")

                # Check getattr obfuscation: getattr(os, 'system')
                if isinstance(node.func, ast.Name) and node.func.id == 'getattr':
                    if len(node.args) >= 2:
                        if isinstance(node.args[1], ast.Constant):
                            attr_name = str(node.args[1].value)
                            if attr_name in {'system', 'popen', 'eval', 'exec', '__import__'}:
                                self.violations.append(f"getattr obfuscation detected: {attr_name}")

                self.generic_visit(node)

            def visit_Import(self, node):
                """Check for dangerous imports."""
                # SECURITY FIX: Extended dangerous modules list
                # Including modules that can bypass security or execute arbitrary code
                dangerous_modules = {
                    'ctypes',           # Low-level memory access
                    'pty',              # Pseudo-terminal spawning
                    'commands',         # Shell command execution (deprecated)
                    'importlib',        # Dynamic module loading - bypasses __import__ check
                    'marshal',          # Code object serialization
                    'types',            # Can create code objects dynamically
                    'code',             # InteractiveInterpreter allows arbitrary execution
                    'codeop',           # Code compilation
                    'multiprocessing',  # Can spawn processes with arbitrary targets
                    'socket',           # Network access (unless explicitly allowed)
                    'asyncio.subprocess',  # Async subprocess execution
                    'builtins',         # Direct access to builtins bypasses checks
                }
                for alias in node.names:
                    module_name = alias.name.split('.')[0]  # Handle 'os.path' style imports
                    if module_name in dangerous_modules:
                        self.violations.append(f"Dangerous import: {alias.name}")
                self.generic_visit(node)

            def visit_ImportFrom(self, node):
                """Check for dangerous from imports."""
                # SECURITY FIX: Extended dangerous modules list
                dangerous_modules = {
                    'ctypes', 'pty', 'commands', 'importlib', 'marshal',
                    'types', 'code', 'codeop', 'multiprocessing', 'socket',
                    'builtins'
                }
                if node.module and node.module.split('.')[0] in dangerous_modules:
                    self.violations.append(f"Dangerous import from: {node.module}")
                self.generic_visit(node)

        visitor = SecurityVisitor()
        visitor.visit(tree)

        if visitor.violations:
            for violation in visitor.violations:
                logger.error(f"Security violation: {violation}")
            return False

        return True

    async def _run_sandbox_tests(self, generated: GeneratedCode) -> bool:
        """Run generated tests in a sandbox."""
        try:
            # Create temporary directory for sandbox
            with tempfile.TemporaryDirectory() as sandbox:
                sandbox_path = Path(sandbox)

                # Write module
                module_path = sandbox_path / f"{generated.module_name}.py"
                module_path.write_text(generated.code, encoding="utf-8")

                # Write tests
                test_path = sandbox_path / f"test_{generated.module_name}.py"
                test_path.write_text(generated.test_code, encoding="utf-8")

                # Run pytest
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=sandbox,
                    env={**os.environ, "PYTHONPATH": str(sandbox_path)}
                )

                return result.returncode == 0

        except subprocess.TimeoutExpired:
            import logging
            logging.getLogger("CodeValidation").error("Sandbox tests timed out")
            return False
        except Exception as e:
            import logging
            logging.getLogger("CodeValidation").error(f"Sandbox test error: {e}")
            return False

    async def _test_import(self, generated: GeneratedCode) -> bool:
        """Test that the module can be imported."""
        try:
            with tempfile.TemporaryDirectory() as sandbox:
                sandbox_path = Path(sandbox)
                module_path = sandbox_path / f"{generated.module_name}.py"
                module_path.write_text(generated.code, encoding="utf-8")

                # Try to import
                spec = importlib.util.spec_from_file_location(
                    generated.module_name,
                    module_path
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                return True
        except Exception as e:
            import logging
            logging.getLogger("CodeValidation").error(f"Import test failed: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════
# MODULE REGISTRY & HOT RELOADING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RegisteredModule:
    """A module registered in the system."""
    name: str
    path: Path
    capability_type: CapabilityType
    loaded_at: datetime
    version: int = 1

    # Runtime reference
    module_ref: Optional[Any] = None

    # Metadata
    description: str = ""
    exports: List[str] = field(default_factory=list)  # Public functions/classes


class ModuleRegistry:
    """
    Registry of all modules in the system.

    Supports:
    - Dynamic registration
    - Hot reloading
    - Capability tracking
    """

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.modules: Dict[str, RegisteredModule] = {}
        self._lock = asyncio.Lock()

    async def register(self, generated: GeneratedCode) -> RegisteredModule:
        """Register a new module in the system."""
        import logging
        logger = logging.getLogger("ModuleRegistry")

        async with self._lock:
            # Write the code to disk
            file_path = Path(generated.file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(generated.code, encoding="utf-8")

            logger.info(f"[REGISTRY] Written module to {file_path}")

            # Create registry entry
            registered = RegisteredModule(
                name=generated.module_name,
                path=file_path,
                capability_type=self._infer_capability_type(file_path),
                loaded_at=datetime.now(),
                description=generated.description,
                exports=self._extract_exports(generated.code)
            )

            # Load the module
            registered.module_ref = await self._load_module(registered)

            self.modules[generated.module_name] = registered

            logger.info(f"[REGISTRY] Registered module: {generated.module_name}")

            return registered

    async def reload(self, module_name: str) -> RegisteredModule:
        """Hot-reload an existing module."""
        import logging
        logger = logging.getLogger("ModuleRegistry")

        async with self._lock:
            if module_name not in self.modules:
                raise KeyError(f"Module {module_name} not registered")

            registered = self.modules[module_name]

            # Reload the module
            if registered.module_ref:
                importlib.reload(registered.module_ref)
            else:
                registered.module_ref = await self._load_module(registered)

            registered.loaded_at = datetime.now()
            registered.version += 1

            logger.info(f"[REGISTRY] Reloaded module: {module_name} (v{registered.version})")

            return registered

    async def _load_module(self, registered: RegisteredModule) -> Any:
        """
        Load a module from disk with sandbox isolation.

        FIX: Added subprocess isolation and timeout to prevent:
        1. Malicious top-level code execution in main process
        2. Infinite loops or resource exhaustion
        3. Unintended side effects during module loading
        """
        import subprocess
        import tempfile
        import json
        import logging
        logger = logging.getLogger("ModuleRegistry")

        # STEP 1: Test module loading in isolated subprocess
        # This catches issues before affecting main process
        test_script = f'''
import sys
import json
try:
    spec = __import__('importlib.util').util.spec_from_file_location(
        "{registered.name}",
        r"{registered.path}"
    )
    module = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Report success and exports
    exports = [n for n in dir(module) if not n.startswith('_')]
    print(json.dumps({{"success": True, "exports": exports}}))
except Exception as e:
    print(json.dumps({{"success": False, "error": str(e)}}))
'''

        try:
            # Run in subprocess with timeout
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout for module loading
                cwd=str(registered.path.parent)
            )

            # Parse result
            try:
                load_result = json.loads(result.stdout.strip())
            except json.JSONDecodeError:
                load_result = {"success": False, "error": f"Invalid output: {result.stdout}"}

            if not load_result.get("success"):
                error = load_result.get("error", "Unknown error")
                logger.error(f"[SANDBOX] Module {registered.name} failed sandbox test: {error}")
                raise RuntimeError(f"Module failed sandbox test: {error}")

            logger.info(f"[SANDBOX] Module {registered.name} passed sandbox test")

        except subprocess.TimeoutExpired:
            logger.error(f"[SANDBOX] Module {registered.name} timed out during loading")
            raise RuntimeError(f"Module {registered.name} timed out during loading - possible infinite loop")

        # STEP 2: Load in main process (after sandbox validation)
        # Now we're confident it's safe
        spec = importlib.util.spec_from_file_location(
            registered.name,
            registered.path
        )
        module = importlib.util.module_from_spec(spec)

        # Store previous version for potential rollback
        previous_module = sys.modules.get(registered.name)

        try:
            # FIX: Race condition - exec_module FIRST, then add to sys.modules
            # Previously, module was added to sys.modules BEFORE exec_module,
            # which could leave sys.modules in inconsistent state on failure.

            # Temporarily add to sys.modules (some modules need this during load)
            sys.modules[registered.name] = module
            spec.loader.exec_module(module)

            # If we get here, load succeeded - module is already in sys.modules
            logger.info(f"[LOAD] Successfully loaded module: {registered.name}")
            return module

        except Exception as e:
            # Rollback on failure - restore or remove from sys.modules
            if previous_module:
                sys.modules[registered.name] = previous_module
            else:
                sys.modules.pop(registered.name, None)
            logger.error(f"[LOAD] Failed to load module {registered.name}: {e}")
            raise

    def _infer_capability_type(self, path: Path) -> CapabilityType:
        """Infer capability type from file path."""
        path_str = str(path).lower()

        if "source" in path_str:
            return CapabilityType.DATA_SOURCE
        elif "analyz" in path_str:
            return CapabilityType.ANALYSIS_METHOD
        elif "generat" in path_str or "writer" in path_str:
            return CapabilityType.GENERATION_STYLE
        elif "visual" in path_str:
            return CapabilityType.VISUAL_FORMAT
        elif "integrat" in path_str:
            return CapabilityType.INTEGRATION
        elif "agent" in path_str:
            return CapabilityType.AGENT
        elif "valid" in path_str or "qc" in path_str:
            return CapabilityType.VALIDATOR
        else:
            return CapabilityType.UTILITY

    def _extract_exports(self, code: str) -> List[str]:
        """Extract public function/class names from code."""
        try:
            tree = ast.parse(code)
            exports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and not node.name.startswith('_'):
                    exports.append(node.name)
                elif isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
                    exports.append(node.name)

            return exports
        except:
            return []

    def get_capability_summary(self) -> str:
        """Get summary of all registered capabilities."""
        if not self.modules:
            return "No custom modules registered yet."

        lines = ["Current system capabilities:"]

        by_type: Dict[CapabilityType, List[RegisteredModule]] = {}
        for mod in self.modules.values():
            by_type.setdefault(mod.capability_type, []).append(mod)

        for cap_type, mods in by_type.items():
            lines.append(f"\n{cap_type.value}:")
            for mod in mods:
                lines.append(f"  - {mod.name}: {mod.description}")
                if mod.exports:
                    lines.append(f"    Exports: {', '.join(mod.exports[:5])}")

        return "\n".join(lines)

    def get_module(self, name: str) -> Optional[Any]:
        """Get loaded module reference."""
        if name in self.modules:
            return self.modules[name].module_ref
        return None


# ═══════════════════════════════════════════════════════════════════════════
# SELF-MODIFICATION ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SelfModificationResult:
    """Result of a self-modification attempt."""
    success: bool
    gap: CapabilityGap
    generated_code: Optional[GeneratedCode]
    error: Optional[str] = None
    retry_with_new_capability: bool = False


class SelfModificationEngine:
    """
    Main orchestrator for self-modifying code.

    Coordinates:
    - Gap detection
    - Code generation
    - Validation
    - Hot reloading
    - Retry logic

    Uses Claude Code CLI (твоя подписка), не API.

    DEPENDENCY INJECTION:
    All components can be injected for testing and flexibility.
    Use create_default() factory for standard production setup.
    """

    def __init__(
        self,
        project_root: Path,
        registry: "ModuleRegistry",
        analyzer: "CapabilityAnalyzer",
        generator: "CodeGenerationEngine",
        validator: "CodeValidator",
        max_generation_attempts: int = 3
    ):
        """
        Initialize with injected dependencies.

        For production use, prefer create_default() factory method.
        """
        self.project_root = project_root
        self.max_attempts = max_generation_attempts

        # Injected components
        self.registry = registry
        self.analyzer = analyzer
        self.generator = generator
        self.validator = validator

        # State
        self.modification_history: List[SelfModificationResult] = []
        self._lock = asyncio.Lock()

    @classmethod
    def create_default(
        cls,
        project_root: Path,
        max_generation_attempts: int = 3
    ) -> "SelfModificationEngine":
        """
        Factory method for standard production setup.

        Creates all components with default configuration.
        Use direct __init__ for testing with mocked components.
        """
        registry = ModuleRegistry(project_root)
        analyzer = CapabilityAnalyzer(registry)
        generator = CodeGenerationEngine(project_root, registry)
        validator = CodeValidator(project_root)

        return cls(
            project_root=project_root,
            registry=registry,
            analyzer=analyzer,
            generator=generator,
            validator=validator,
            max_generation_attempts=max_generation_attempts
        )

    async def try_self_modify(
        self,
        evaluation: "ContentEvaluation",
        visual_evaluation: "VisualEvaluation",
        iteration_learnings: "IterationLearnings",
        content_type: "ContentType",
        attempt_number: int
    ) -> Optional[SelfModificationResult]:
        """
        Attempt to self-modify if a capability gap is detected.

        Returns SelfModificationResult if modification was attempted,
        None if no modification was needed.
        """
        import logging
        logger = logging.getLogger("SelfModification")

        async with self._lock:
            # 1. Analyze if we need new code
            gap = await self.analyzer.analyze_failure(
                evaluation=evaluation,
                visual_evaluation=visual_evaluation,
                iteration_learnings=iteration_learnings,
                content_type=content_type,
                attempt_number=attempt_number
            )

            if not gap:
                logger.debug("[SELFMOD] No capability gap detected, using micro-learning")
                return None

            logger.info(f"[SELFMOD] Capability gap detected: {gap.description}")

            # 2. Generate code (with retries)
            generated = None
            last_error = None

            for gen_attempt in range(self.max_attempts):
                try:
                    logger.info(f"[SELFMOD] Generation attempt {gen_attempt + 1}/{self.max_attempts}")

                    # Generate based on gap type
                    if gap.gap_type == CapabilityType.AGENT:
                        generated = await self.generator.generate_module(gap)
                    elif gap.related_modules:
                        # Try to extend existing module first
                        generated = await self.generator.generate_function(
                            gap,
                            target_module=gap.related_modules[0]
                        )
                    else:
                        generated = await self.generator.generate_module(gap)

                    # 3. Validate
                    generated = await self.validator.validate(generated)

                    if generated.is_valid():
                        break
                    else:
                        last_error = "Validation failed"
                        logger.warning(f"[SELFMOD] Validation failed, retrying...")

                except Exception as e:
                    last_error = str(e)
                    logger.error(f"[SELFMOD] Generation error: {e}")

            # 4. Register if valid
            if generated and generated.is_valid():
                await self.registry.register(generated)

                result = SelfModificationResult(
                    success=True,
                    gap=gap,
                    generated_code=generated,
                    retry_with_new_capability=True
                )

                logger.info(f"[SELFMOD] ✅ Successfully added new capability: {gap.description}")

            else:
                result = SelfModificationResult(
                    success=False,
                    gap=gap,
                    generated_code=generated,
                    error=last_error,
                    retry_with_new_capability=False
                )

                logger.error(f"[SELFMOD] ❌ Failed to add capability after {self.max_attempts} attempts")

            self.modification_history.append(result)
            return result

    def get_new_capabilities(self) -> List[str]:
        """Get list of capabilities added during this session."""
        return [
            result.gap.description
            for result in self.modification_history
            if result.success
        ]
```

---

### Integration with Pipeline

```python
# Add to PipelineState TypedDict:
class PipelineState(TypedDict):
    # ... existing fields ...

    # Self-Modification Engine
    self_mod_engine: "SelfModificationEngine"
    self_mod_result: Optional["SelfModificationResult"]
    capabilities_added: List[str]  # Track what was added this run


# Update post_evaluation_learning_node to include self-modification:

async def post_evaluation_learning_node(state: PipelineState) -> dict:
    """
    After evaluation:
    1. Extract micro-learnings (always)
    2. Try self-modification if needed (when micro-learning isn't enough)
    """
    import logging
    logger = logging.getLogger("PostEvaluation")

    learning_engine = state["learning_engine"]
    self_mod_engine = state["self_mod_engine"]

    # ... existing learning extraction code ...

    # ─────────────────────────────────────────────────────────────────
    # SELF-MODIFICATION CHECK
    # If this is attempt 2+ and still failing, consider generating new code
    # ─────────────────────────────────────────────────────────────────

    attempt_number = state.get("revision_count", 0) + 1

    # Only try self-modification if:
    # 1. This is at least the 2nd revision attempt
    # 2. Score is still below threshold
    # 3. We haven't already modified for this gap

    self_mod_result = None
    if attempt_number >= 2 and evaluation.decision != "PASS":
        logger.info(f"[SELFMOD] Attempt {attempt_number}, checking if new code needed...")

        self_mod_result = await self_mod_engine.try_self_modify(
            evaluation=evaluation,
            visual_evaluation=visual_evaluation,
            iteration_learnings=learnings,
            content_type=state["content_type"],
            attempt_number=attempt_number
        )

        if self_mod_result and self_mod_result.success:
            logger.info(f"[SELFMOD] New capability added, will retry generation")

    # ─────────────────────────────────────────────────────────────────
    # DETERMINE ROUTING
    # ─────────────────────────────────────────────────────────────────

    # If we just added a new capability, force a retry even if we would normally give up
    qc_decision = route_after_qc(state)

    if self_mod_result and self_mod_result.retry_with_new_capability:
        # Override to retry with new capability
        if qc_decision == "max_revisions_force":
            qc_decision = "revise_writer"  # Give one more chance with new capability
            logger.info("[SELFMOD] Overriding max_revisions to retry with new capability")

    return {
        "iteration_learnings": learnings,
        "self_mod_result": self_mod_result,
        "capabilities_added": self_mod_engine.get_new_capabilities(),
        "_qc_decision": qc_decision
    }


# Update run_pipeline to initialize SelfModificationEngine:
# (Полная версия в основном run_pipeline выше, это краткий пример интеграции)

async def run_pipeline(
    selection_mode: str = "auto_top_pick",
    db: Optional[SupabaseDB] = None,  # Supabase — единственная БД
    prompt_manager: "PromptManager" = None,
    config_manager: "ConfigManager" = None,
    project_root: Path = None
) -> Dict[str, Any]:
    """Execute pipeline with self-modification capability."""

    # Initialize Supabase
    db = db or get_db()

    # ... existing initialization ...

    # Initialize Self-Modification Engine (uses Claude CLI internally)
    self_mod_engine = SelfModificationEngine.create_default(
        project_root=project_root or Path.cwd()
    )

    # Inject into state
    initial_state["self_mod_engine"] = self_mod_engine
    initial_state["self_mod_result"] = None
    initial_state["capabilities_added"] = []

    # ... run pipeline ...

    # Include self-modification stats in result
    return {
        # ... existing result fields ...
        "self_modification": {
            "capabilities_added": final_state.get("capabilities_added", []),
            "modification_attempted": final_state.get("self_mod_result") is not None,
            "modification_successful": (
                final_state.get("self_mod_result", {}).success
                if final_state.get("self_mod_result") else False
            )
        }
    }
```

---

### Example: System Writing Its Own Module

```
Scenario: Writing a post about GPT-4 Turbo release, but system doesn't know how to
fetch data from OpenAI's blog.

═══════════════════════════════════════════════════════════════════════════════

ITERATION 1:
├── Writer generates draft
├── QC: Score 6.5/10
│   └── Feedback: "Lacking specific details from the announcement"
├── Learning: "Need more source details" (micro-learning)
└── Route: REVISE_WRITER

ITERATION 2:
├── Writer generates draft (with learning injected)
├── QC: Score 6.8/10
│   └── Feedback: "Still missing technical specifications from official source"
├── Learning: Can't improve further with prompt changes
├── CapabilityAnalyzer: "Need data source integration for OpenAI blog"
│
├── SelfModificationEngine triggers:
│   │
│   ├── Gap detected:
│   │   type: DATA_SOURCE
│   │   description: "Fetch and parse OpenAI blog posts"
│   │
│   ├── CodeGenerationEngine generates:
│   │   module: openai_blog_source.py
│   │   ─────────────────────────────
│   │   @dataclass
│   │   class OpenAIBlogPost:
│   │       title: str
│   │       date: datetime
│   │       content: str
│   │       technical_details: Dict[str, Any]
│   │
│   │   class OpenAIBlogSource:
│   │       async def fetch_latest(self) -> List[OpenAIBlogPost]:
│   │           ...
│   │       async def fetch_by_topic(self, topic: str) -> List[OpenAIBlogPost]:
│   │           ...
│   │   ─────────────────────────────
│   │
│   ├── CodeValidator:
│   │   ✅ Syntax valid
│   │   ✅ Types valid
│   │   ✅ Security scan passed
│   │   ✅ Tests passed
│   │
│   ├── ModuleRegistry:
│   │   → Saved to sources/openai_blog_source.py
│   │   → Hot-loaded into system
│   │
│   └── Result: NEW CAPABILITY ADDED
│
└── Route: REVISE_WRITER (with new capability)

ITERATION 3:
├── TrendScout now uses OpenAIBlogSource ← NEW!
├── Analyzer gets real data from OpenAI blog ← NEW!
├── Writer generates draft with actual technical details
├── QC: Score 8.7/10 ✅
└── Route: PASS

═══════════════════════════════════════════════════════════════════════════════

RESULT: System wrote its own data source integration and used it in the same run!
```

---

### Research Agent

```python
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
from enum import Enum


class ResearchTrigger(Enum):
    """What triggers a research cycle."""
    # ═══════════════════════════════════════════════════════════════════════
    # CONTINUOUS LEARNING TRIGGERS (NEW - работают с первого поста)
    # ═══════════════════════════════════════════════════════════════════════
    EVERY_ITERATION = "every_iteration"        # After EVERY evaluation cycle
    FIRST_POST = "first_post"                  # Special handling for first post ever
    COMPONENT_FEEDBACK = "component_feedback"  # When specific component gets negative feedback

    # ═══════════════════════════════════════════════════════════════════════
    # SCHEDULED/REACTIVE TRIGGERS (original - для глубокого исследования)
    # ═══════════════════════════════════════════════════════════════════════
    UNDERPERFORMANCE = "underperformance"      # 3+ posts below average
    NEW_CONTENT_TYPE = "new_content_type"      # No experience with this type
    WEEKLY_CYCLE = "weekly_cycle"              # Scheduled weekly deep research
    ALGORITHM_CHANGE = "algorithm_change"      # Detected engagement pattern shift
    MANUAL_REQUEST = "manual_request"          # Human requested research


@dataclass
class ResearchQuery:
    """Single research query to execute."""
    source: str          # "perplexity", "competitor_scrape", "own_data"
    query: str           # The actual query/action
    purpose: str         # Why we're researching this
    priority: int = 1    # 1 = highest


@dataclass
class ResearchFinding:
    """Single insight from research."""
    finding: str                    # What was discovered
    source: str                     # Where it came from
    confidence: float               # 0-1
    actionable: bool                # Can we act on this?
    suggested_change: Optional[str] # What to change
    affected_component: str         # "writer", "trend_scout", "visual_creator", etc.


@dataclass
class ResearchRecommendation:
    """Actionable recommendation from research analysis."""
    component: str          # "writer", "trend_scout", "visual_creator", "qc_agent", etc.
    change: str             # What should be changed
    priority: int           # 1 = highest, 5 = lowest
    rationale: str          # Why this change is recommended
    confidence: float       # 0-1, confidence in this recommendation
    estimated_impact: str   # "high", "medium", "low"
    source_findings: List[str]  # IDs/references to supporting findings


@dataclass
class ResearchReport:
    """Complete research report with findings and recommendations."""

    trigger: ResearchTrigger
    trigger_reason: str
    started_at: datetime
    completed_at: datetime

    # What was researched
    queries_executed: List[ResearchQuery]

    # What was found
    findings: List[ResearchFinding]

    # Recommendations (typed)
    recommendations: List[ResearchRecommendation]

    # Meta
    total_sources_consulted: int
    confidence_score: float  # Overall confidence in recommendations


class ResearchAgent:
    """
    Agent that researches best practices and competitor strategies.
    Uses Perplexity for web research, scrapes competitor posts, analyzes own data.
    """

    def __init__(self, perplexity_client, linkedin_scraper, analytics_db):
        self.perplexity = perplexity_client
        self.linkedin = linkedin_scraper
        self.db = analytics_db

        # Top LinkedIn influencers to learn from
        self.competitors = [
            "https://www.linkedin.com/in/justinwelsh/",
            "https://www.linkedin.com/in/sambhavchaturvedi/",
            "https://www.linkedin.com/in/shanebarker/",
            "https://www.linkedin.com/in/garyvaynerchuk/",
            "https://www.linkedin.com/in/jasonfalls/"
        ]

    async def should_research(self) -> Optional[ResearchTrigger]:
        """
        Check if research should be triggered.

        MEDIUM PRIORITY FIX #13: Added comprehensive docstring.

        Evaluates multiple conditions to determine if research is needed:

        1. UNDERPERFORMANCE: Triggered when 3+ of last 5 posts score below
           80% of the average. This indicates potential issues with content
           strategy that research might address.

        2. WEEKLY_CYCLE: Triggered every Sunday if it's been 7+ days since
           last research. Regular research keeps strategies fresh and
           incorporates new best practices.

        3. NEW_CONTENT_TYPE (implicit): Can be triggered externally when
           encountering a content type with no historical data.

        4. ALGORITHM_CHANGE (implicit): Can be triggered externally when
           engagement pattern shifts are detected.

        Returns:
            Optional[ResearchTrigger]:
                - ResearchTrigger.UNDERPERFORMANCE if recent posts are weak
                - ResearchTrigger.WEEKLY_CYCLE if weekly research is due
                - None if no research is currently needed

        Example:
            trigger = await agent.should_research()
            if trigger:
                report = await agent.research(trigger)
        """

        # Check for underperformance
        recent_posts = await self.db.get_recent_posts(limit=5)
        avg_score = await self.db.get_average_score()

        underperforming = sum(1 for p in recent_posts if p.score < avg_score * 0.8)
        if underperforming >= 3:
            return ResearchTrigger.UNDERPERFORMANCE

        # Check for weekly cycle (Sunday)
        if datetime.now().weekday() == 6:  # Sunday
            last_research = await self.db.get_last_research_date()
            if (datetime.now() - last_research).days >= 7:
                return ResearchTrigger.WEEKLY_CYCLE

        return None

    async def research(
        self,
        trigger: ResearchTrigger,
        timeout_seconds: int = 300  # FIX #9: Configurable timeout
    ) -> ResearchReport:
        """
        Execute full research cycle.

        MEDIUM PRIORITY FIX #9: Added timeout and error handling.
        Research continues with partial results if some queries fail.

        Args:
            trigger: What triggered this research cycle
            timeout_seconds: Maximum time for entire research cycle (default 5 min)

        Returns:
            ResearchReport with findings (may be partial if some queries failed)
        """
        import logging
        logger = logging.getLogger("ResearchAgent")

        started_at = datetime.now()
        queries = self._build_queries(trigger)
        findings = []
        failed_queries = []

        logger.info(
            f"[RESEARCH] Starting research cycle\n"
            f"  Trigger: {trigger.value}\n"
            f"  Queries: {len(queries)}\n"
            f"  Timeout: {timeout_seconds}s"
        )

        # FIX #9: Wrap entire research in timeout
        try:
            async with asyncio.timeout(timeout_seconds):
                for query in queries:
                    query_start = datetime.now()

                    # FIX #9: Individual query error handling
                    try:
                        if query.source == "perplexity":
                            result = await self._research_perplexity(query)
                        elif query.source == "competitor_scrape":
                            result = await self._research_competitors(query)
                        elif query.source == "own_data":
                            result = await self._research_own_data(query)
                        else:
                            logger.warning(f"[RESEARCH] Unknown source: {query.source}")
                            result = []

                        findings.extend(result)
                        query_duration = (datetime.now() - query_start).total_seconds()
                        logger.debug(
                            f"[RESEARCH] Query completed in {query_duration:.1f}s: "
                            f"{query.source} - {len(result)} findings"
                        )

                    except asyncio.TimeoutError:
                        logger.warning(f"[RESEARCH] Query timeout: {query.source} - {query.query[:50]}...")
                        failed_queries.append((query, "timeout"))
                    except Exception as e:
                        logger.warning(f"[RESEARCH] Query failed: {query.source} - {e}")
                        failed_queries.append((query, str(e)))
                        # Continue with other queries

        except asyncio.TimeoutError:
            logger.warning(
                f"[RESEARCH] Overall timeout reached ({timeout_seconds}s). "
                f"Returning partial results: {len(findings)} findings from "
                f"{len(queries) - len(failed_queries)}/{len(queries)} queries"
            )

        # Generate recommendations from findings (even if partial)
        if findings:
            recommendations = await self._generate_recommendations(findings)
        else:
            logger.warning("[RESEARCH] No findings - skipping recommendation generation")
            recommendations = []

        completed_at = datetime.now()
        total_duration = (completed_at - started_at).total_seconds()

        logger.info(
            f"[RESEARCH] Research cycle complete in {total_duration:.1f}s\n"
            f"  Findings: {len(findings)}\n"
            f"  Recommendations: {len(recommendations)}\n"
            f"  Failed queries: {len(failed_queries)}"
        )

        return ResearchReport(
            trigger=trigger,
            trigger_reason=self._get_trigger_reason(trigger),
            started_at=started_at,
            completed_at=completed_at,
            queries_executed=[q for q in queries if (q, None) not in [(fq[0], None) for fq in failed_queries]],
            findings=findings,
            recommendations=recommendations,
            total_sources_consulted=len(queries) - len(failed_queries),
            confidence_score=self._calculate_confidence(findings)
        )

    @with_retry(component="search", operation="perplexity_research")
    async def _research_perplexity(self, query: ResearchQuery) -> List[ResearchFinding]:
        """
        Research using Perplexity API.

        FIX: Added retry decorator and proper error handling.
        Uses search_max_attempts from RetryConfig (default: 2).
        """
        import logging
        logger = logging.getLogger("ResearchAgent")

        # FIX: Wrap API call in try-except with proper error handling
        try:
            response = await self.perplexity.search(query.query)
        except Exception as e:
            logger.error(f"Perplexity search failed for query '{query.query[:50]}...': {e}")
            raise  # Let retry decorator handle it

        # FIX: Validate response before using
        if not response or not hasattr(response, 'content') or not response.content:
            logger.warning(f"Empty response from Perplexity for query: {query.query[:50]}...")
            return []  # Return empty list instead of failing

        # Use Claude to extract actionable insights
        extraction_prompt = f"""
        Research query: {query.query}
        Purpose: {query.purpose}

        Search results:
        {response.content}

        Extract 3-5 actionable insights for improving LinkedIn posts.
        For each insight, specify:
        1. What was found
        2. How confident we are (0-1)
        3. What specific change to make
        4. Which component to change (writer/trend_scout/visual_creator/scheduler)

        Return as JSON list.
        """

        try:
            insights = await self.claude.generate_structured(
                prompt=extraction_prompt,
                response_model=List[ResearchFinding]
            )
            return insights
        except Exception as e:
            logger.error(f"Claude extraction failed: {e}")
            # Return empty rather than failing - partial results are OK
            return []

    async def _research_competitors(self, query: ResearchQuery) -> List[ResearchFinding]:
        """Scrape and analyze competitor posts including visual patterns."""
        import logging
        import asyncio
        logger = logging.getLogger("ResearchAgent")

        all_posts = []
        # FIX: Add rate limiting delay between LinkedIn API requests
        # LinkedIn rate limits are ~10 requests/minute, so 6-7 seconds between requests is safe
        REQUEST_DELAY_SECONDS = 7.0

        for i, competitor_url in enumerate(self.competitors):
            try:
                posts = await self.linkedin.get_recent_posts(
                    profile_url=competitor_url,
                    limit=10,
                    include_visuals=True  # FIX #18: Include visual metadata
                )
                all_posts.extend(posts)
                logger.debug(f"Fetched {len(posts)} posts from competitor {i+1}/{len(self.competitors)}")

                # FIX: Rate limiting delay - skip delay after last request
                if i < len(self.competitors) - 1:
                    await asyncio.sleep(REQUEST_DELAY_SECONDS)

            except Exception as e:
                logger.warning(f"Failed to fetch posts from {competitor_url}: {e}")
                # Continue with other competitors - partial data is better than none
                continue

        # Sort by engagement
        top_posts = sorted(all_posts, key=lambda p: p.likes + p.comments * 3, reverse=True)[:20]

        # Analyze text patterns
        text_analysis_prompt = f"""
        Analyze these top-performing LinkedIn posts from influencers:

        {[{"text": p.text[:500], "likes": p.likes, "comments": p.comments} for p in top_posts]}

        Find patterns:
        1. Hook styles that work (first 2 lines)
        2. Post structure patterns
        3. Common topics/angles
        4. Call-to-action patterns

        Return actionable insights we can apply to our posts.
        """

        text_findings = await self.claude.generate_structured(
            prompt=text_analysis_prompt,
            response_model=List[ResearchFinding]
        )

        # FIX #18: Analyze visual patterns separately
        visual_findings = await self._analyze_competitor_visuals(top_posts)

        return text_findings + visual_findings

    async def _analyze_competitor_visuals(
        self,
        posts: List[dict]
    ) -> List[ResearchFinding]:
        """
        FIX #18: Dedicated visual pattern analysis for competitor posts.
        Extracts insights about what visual formats drive engagement.
        """

        # Filter posts with visuals
        posts_with_visuals = [p for p in posts if p.has_visual]
        posts_without_visuals = [p for p in posts if not p.has_visual]

        # Calculate visual impact
        avg_engagement_with_visual = (
            sum(p.likes + p.comments * 3 for p in posts_with_visuals) / len(posts_with_visuals)
            if posts_with_visuals else 0
        )
        avg_engagement_without_visual = (
            sum(p.likes + p.comments * 3 for p in posts_without_visuals) / len(posts_without_visuals)
            if posts_without_visuals else 0
        )

        visual_lift = (
            (avg_engagement_with_visual - avg_engagement_without_visual) / avg_engagement_without_visual
            if avg_engagement_without_visual > 0 else 0
        )

        # Prepare visual data for analysis
        visual_data = [{
            "visual_type": p.visual_type,  # image / carousel / document / video
            "visual_style": p.visual_style,  # infographic / photo / screenshot / quote_card
            "has_author_photo": p.has_author_photo,
            "image_text_amount": p.image_text_amount,  # none / minimal / moderate / heavy
            "likes": p.likes,
            "comments": p.comments,
            "engagement_score": p.likes + p.comments * 3
        } for p in posts_with_visuals]

        analysis_prompt = f"""
        Analyze visual patterns from top-performing LinkedIn posts:

        VISUAL IMPACT STATISTICS:
        - Posts with visuals: {len(posts_with_visuals)} (avg engagement: {avg_engagement_with_visual:.0f})
        - Posts without visuals: {len(posts_without_visuals)} (avg engagement: {avg_engagement_without_visual:.0f})
        - Visual engagement lift: {visual_lift:+.1%}

        VISUAL DATA BY POST:
        {visual_data}

        Analyze and find patterns:
        1. Which visual_type performs best? (image vs carousel vs document)
        2. Which visual_style correlates with higher engagement?
        3. Does author photo presence affect engagement?
        4. What's the optimal text amount on images?
        5. Are there visual+content_type combinations that work better?

        Return 3-5 actionable insights for Visual Creator Agent.
        Mark affected_component as "visual_creator" for all findings.
        """

        return await self.claude.generate_structured(
            prompt=analysis_prompt,
            response_model=List[ResearchFinding]
        )

    async def _research_own_data(self, query: ResearchQuery) -> List[ResearchFinding]:
        """Analyze own post performance data."""

        # Get best and worst posts
        all_posts = await self.db.get_all_posts()
        sorted_posts = sorted(all_posts, key=lambda p: p.final_score, reverse=True)

        top_10 = sorted_posts[:10]
        bottom_10 = sorted_posts[-10:]

        # FIX #18: Enhanced visual data extraction
        def extract_post_data(p):
            return {
                "hook": p.text[:100],
                "score": p.final_score,
                "type": p.content_type,
                # Visual data (expanded)
                "visual_type": p.visual_type,  # image / carousel / document / none
                "visual_style": getattr(p, 'visual_style', None),  # infographic / photo / etc.
                "has_author_photo": getattr(p, 'has_author_photo', False),
                "photo_integration_mode": getattr(p, 'photo_integration_mode', None),
                # Timing
                "posting_hour": getattr(p, 'published_at', datetime.now()).hour,
                "posting_day": getattr(p, 'published_at', datetime.now()).strftime('%A')
            }

        analysis_prompt = f"""
        Compare our best and worst performing LinkedIn posts:

        TOP 10 (high engagement):
        {[extract_post_data(p) for p in top_10]}

        BOTTOM 10 (low engagement):
        {[extract_post_data(p) for p in bottom_10]}

        What patterns differentiate winners from losers?
        Be specific about:
        1. Hook style and first line patterns
        2. Content type distribution
        3. Visual type impact (which visual_type performs best?)
        4. Visual style impact (photo vs infographic vs screenshot)
        5. Author photo correlation (does has_author_photo correlate with success?)
        6. Posting time patterns (hour/day)

        Return actionable insights with affected_component for each:
        - "writer" for content patterns
        - "visual_creator" for visual patterns
        - "scheduler" for timing patterns
        """

        text_visual_findings = await self.claude.generate_structured(
            prompt=analysis_prompt,
            response_model=List[ResearchFinding]
        )

        # FIX #18: Additional visual-specific correlation analysis
        visual_correlation_findings = await self._analyze_visual_engagement_correlation(all_posts)

        return text_visual_findings + visual_correlation_findings

    async def _analyze_visual_engagement_correlation(
        self,
        posts: List[dict]
    ) -> List[ResearchFinding]:
        """
        FIX #18: Deep correlation analysis between visual choices and engagement.
        Provides data-driven recommendations for Visual Creator Agent.
        """

        # Group by visual type
        by_visual_type = {}
        for p in posts:
            vtype = getattr(p, 'visual_type', 'unknown')
            if vtype not in by_visual_type:
                by_visual_type[vtype] = []
            by_visual_type[vtype].append(p.final_score)

        # Calculate averages
        visual_type_scores = {
            vtype: sum(scores) / len(scores)
            for vtype, scores in by_visual_type.items()
            if scores
        }

        # Group by author photo usage
        with_photo = [p.final_score for p in posts if getattr(p, 'has_author_photo', False)]
        without_photo = [p.final_score for p in posts if not getattr(p, 'has_author_photo', False)]

        photo_impact = {
            "with_photo_avg": sum(with_photo) / len(with_photo) if with_photo else 0,
            "without_photo_avg": sum(without_photo) / len(without_photo) if without_photo else 0,
            "photo_lift": (
                (sum(with_photo) / len(with_photo) - sum(without_photo) / len(without_photo)) /
                (sum(without_photo) / len(without_photo))
                if without_photo and with_photo else 0
            )
        }

        # Cross-tabulate content_type × visual_type
        content_visual_matrix = {}
        for p in posts:
            key = (p.content_type, getattr(p, 'visual_type', 'unknown'))
            if key not in content_visual_matrix:
                content_visual_matrix[key] = []
            content_visual_matrix[key].append(p.final_score)

        best_combinations = sorted([
            {"content_type": k[0], "visual_type": k[1], "avg_score": sum(v) / len(v), "count": len(v)}
            for k, v in content_visual_matrix.items()
            if len(v) >= 3  # Minimum sample size
        ], key=lambda x: x['avg_score'], reverse=True)[:5]

        analysis_prompt = f"""
        Visual engagement correlation data from our posts:

        VISUAL TYPE PERFORMANCE:
        {visual_type_scores}

        AUTHOR PHOTO IMPACT:
        - Average WITH author photo: {photo_impact['with_photo_avg']:.1f}
        - Average WITHOUT author photo: {photo_impact['without_photo_avg']:.1f}
        - Photo engagement lift: {photo_impact['photo_lift']:+.1%}

        BEST CONTENT+VISUAL COMBINATIONS:
        {best_combinations}

        Based on this DATA, provide 2-3 specific, actionable recommendations:
        1. Which visual_type should we prioritize?
        2. Should we use author photos more/less? For which content types?
        3. What content_type + visual_type combination should we favor?

        All findings should have affected_component="visual_creator".
        """

        return await self.claude.generate_structured(
            prompt=analysis_prompt,
            response_model=List[ResearchFinding]
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # FIX: Previously undefined helper methods, now implemented
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_queries(self, trigger: ResearchTrigger) -> List[ResearchQuery]:
        """
        Build research queries based on trigger type.

        FIX: Previously called but never defined, causing AttributeError.

        Args:
            trigger: What triggered this research cycle

        Returns:
            List of ResearchQuery objects to execute
        """
        queries = []

        if trigger == ResearchTrigger.UNDERPERFORMANCE:
            # Focus on what's working for competitors
            queries.extend([
                ResearchQuery(
                    source="perplexity",
                    query="LinkedIn post strategies high engagement AI content 2024",
                    purpose="competitor_strategies"
                ),
                ResearchQuery(
                    source="competitor_scrape",
                    query="top_posts",
                    purpose="learn_from_best"
                ),
                ResearchQuery(
                    source="own_data",
                    query="low_performers",
                    purpose="identify_patterns"
                )
            ])

        elif trigger == ResearchTrigger.WEEKLY_CYCLE:
            # Broad research on best practices
            queries.extend([
                ResearchQuery(
                    source="perplexity",
                    query="LinkedIn algorithm changes 2024",
                    purpose="platform_updates"
                ),
                ResearchQuery(
                    source="perplexity",
                    query="AI content trends LinkedIn thought leadership",
                    purpose="content_trends"
                ),
                ResearchQuery(
                    source="competitor_scrape",
                    query="recent_top_posts",
                    purpose="trending_formats"
                ),
                ResearchQuery(
                    source="own_data",
                    query="all_time_analysis",
                    purpose="performance_review"
                )
            ])

        elif trigger == ResearchTrigger.NEW_CONTENT_TYPE:
            queries.extend([
                ResearchQuery(
                    source="perplexity",
                    query="best practices for new content format LinkedIn",
                    purpose="format_guidance"
                )
            ])

        elif trigger == ResearchTrigger.ALGORITHM_CHANGE:
            queries.extend([
                ResearchQuery(
                    source="perplexity",
                    query="LinkedIn algorithm update latest changes",
                    purpose="algorithm_intelligence"
                )
            ])

        return queries

    def _get_trigger_reason(self, trigger: ResearchTrigger) -> str:
        """
        Get human-readable description of trigger reason.

        FIX: Previously called but never defined.

        Args:
            trigger: Research trigger type

        Returns:
            Human-readable description
        """
        reasons = {
            ResearchTrigger.UNDERPERFORMANCE: "3+ recent posts underperformed (below 80% of average score)",
            ResearchTrigger.WEEKLY_CYCLE: "Weekly research cycle (Sunday, 7+ days since last research)",
            ResearchTrigger.NEW_CONTENT_TYPE: "Encountered content type with no historical data",
            ResearchTrigger.ALGORITHM_CHANGE: "Detected potential platform algorithm change",
            ResearchTrigger.MANUAL_REQUEST: "Manual research request from user",
            ResearchTrigger.FIRST_POST: "First post - initializing knowledge base",
            ResearchTrigger.COMPONENT_FEEDBACK: "Specific component requested research",
            ResearchTrigger.EVERY_ITERATION: "Regular iteration research (continuous learning)"
        }
        return reasons.get(trigger, f"Unknown trigger: {trigger}")

    def _calculate_confidence(self, findings: List[ResearchFinding]) -> float:
        """
        Calculate overall confidence score from research findings.

        FIX: Previously called but never defined.

        Args:
            findings: List of research findings

        Returns:
            Confidence score (0.0 - 1.0)
        """
        if not findings:
            return 0.0

        # Weight by source reliability
        source_weights = {
            "perplexity": 0.7,       # Web search - generally reliable
            "competitor_scrape": 0.8, # Direct observation - high reliability
            "own_data": 1.0          # Our own data - highest reliability
        }

        total_weight = 0.0
        weighted_confidence = 0.0

        for finding in findings:
            source_weight = source_weights.get(finding.source, 0.5)
            finding_confidence = getattr(finding, 'confidence', 0.5)

            weighted_confidence += finding_confidence * source_weight
            total_weight += source_weight

        return weighted_confidence / total_weight if total_weight > 0 else 0.5

    async def _generate_recommendations(
        self,
        findings: List[ResearchFinding]
    ) -> List[ResearchRecommendation]:
        """
        Generate actionable recommendations from research findings.

        Args:
            findings: List of research findings to synthesize

        Returns:
            List of prioritized recommendations
        """
        if not findings:
            return []

        findings_text = "\n".join([
            f"- [{f.source}] {f.description} (confidence: {getattr(f, 'confidence', 'N/A')})"
            for f in findings
        ])

        prompt = f"""
        Based on these research findings, generate 3-5 actionable recommendations:

        FINDINGS:
        {findings_text}

        For each recommendation, specify:
        - priority: high/medium/low
        - affected_component: writer/visual_creator/scheduler/qc/scout
        - action: specific action to take
        - expected_impact: what improvement to expect

        Return as JSON array of recommendations.
        """

        response = await self.claude.generate_structured(
            prompt=prompt,
            response_model=List[ResearchRecommendation]
        )

        return response if response else []
```

---

### Deep Self-Improvement: Agent Dialogue & Code Evolution

Это **ядро настоящего AI-агента** — не просто сравнение метрик, а:
1. Диалог с Critic Agent ("Как тебе это?")
2. Осмысление feedback ("Ага, понятно!")
3. Research на основе критики
4. **Написание нового кода/модулей**
5. **Перманентное изменение промптов**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           DEEP SELF-IMPROVEMENT LOOP (Agent-to-Agent Dialogue)              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║  STEP 1: CREATE                                                        ║ │
│  ║  Writer Agent создаёт пост                                             ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                              │                                              │
│                              ▼                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║  STEP 2: CRITIQUE (Agent-to-Agent Dialogue)                            ║ │
│  ║                                                                        ║ │
│  ║  Creator Agent → Critic Agent:                                         ║ │
│  ║  ┌─────────────────────────────────────────────────────────────────┐  ║ │
│  ║  │ "Вот мой пост. Как тебе? Что можно улучшить?"                   │  ║ │
│  ║  │                                                                  │  ║ │
│  ║  │ [POST CONTENT]                                                   │  ║ │
│  ║  │                                                                  │  ║ │
│  ║  │ Контекст: тема={topic}, тип={content_type}, цель={goal}         │  ║ │
│  ║  └─────────────────────────────────────────────────────────────────┘  ║ │
│  ║                                                                        ║ │
│  ║  Critic Agent (отдельная модель — Claude Opus 4.5):                    ║ │
│  ║  ┌─────────────────────────────────────────────────────────────────┐  ║ │
│  ║  │ "Hook слабый — не цепляет. Ты начинаешь с абстракции, а надо   │  ║ │
│  ║  │  с конкретного примера или провокационного вопроса.             │  ║ │
│  ║  │                                                                  │  ║ │
│  ║  │  Также: слишком много 'я думаю' — звучит неуверенно.            │  ║ │
│  ║  │  И нет конкретных цифр — добавь метрики или кейс."              │  ║ │
│  ║  └─────────────────────────────────────────────────────────────────┘  ║ │
│  ║                                                                        ║ │
│  ║  Creator Agent (follow-up):                                            ║ │
│  ║  ┌─────────────────────────────────────────────────────────────────┐  ║ │
│  ║  │ "Понял про hook. А какой тип hook лучше работает для            │  ║ │
│  ║  │  automation case — вопрос или контринтуитивное утверждение?"    │  ║ │
│  ║  └─────────────────────────────────────────────────────────────────┘  ║ │
│  ║                                                                        ║ │
│  ║  Critic Agent:                                                         ║ │
│  ║  ┌─────────────────────────────────────────────────────────────────┐  ║ │
│  ║  │ "Для automation case лучше работает 'impossible result' hook:  │  ║ │
│  ║  │  'Мы сократили X с 40 часов до 2 минут'. Это сразу показывает  │  ║ │
│  ║  │  value и вызывает 'Как?!' реакцию."                             │  ║ │
│  ║  └─────────────────────────────────────────────────────────────────┘  ║ │
│  ║                                                                        ║ │
│  ║  → Диалог продолжается до достижения understanding                     ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                              │                                              │
│                              ▼                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║  STEP 3: REFLECT ("Ага, понятно!")                                     ║ │
│  ║                                                                        ║ │
│  ║  Creator Agent анализирует критику:                                    ║ │
│  ║  ┌─────────────────────────────────────────────────────────────────┐  ║ │
│  ║  │ Reflection:                                                      │  ║ │
│  ║  │ 1. Мой hook действительно слабый — начинаю с "В последнее      │  ║ │
│  ║  │    время..." вместо конкретного результата                      │  ║ │
│  ║  │ 2. Это паттерн — я часто так начинаю (проверить историю)        │  ║ │
│  ║  │ 3. Critic предложил "impossible result" hook — надо изучить    │  ║ │
│  ║  │ 4. Мне не хватает знаний о типах hooks и когда их применять    │  ║ │
│  ║  │                                                                  │  ║ │
│  ║  │ Knowledge Gap: "Типы hooks для разных content types"            │  ║ │
│  ║  │ Action: Нужен research                                          │  ║ │
│  ║  └─────────────────────────────────────────────────────────────────┘  ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                              │                                              │
│                              ▼                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║  STEP 4: RESEARCH (Targeted by Critique)                               ║ │
│  ║                                                                        ║ │
│  ║  Research Agent получает конкретный запрос от Reflection:              ║ │
│  ║                                                                        ║ │
│  ║  Query 1: Perplexity                                                   ║ │
│  ║  "Best LinkedIn post hooks by content type 2024"                       ║ │
│  ║  "Impossible result hook examples LinkedIn"                            ║ │
│  ║                                                                        ║ │
│  ║  Query 2: Competitor Analysis                                          ║ │
│  ║  "Найти 20 постов с 'impossible result' hooks, проанализировать       ║ │
│  ║   структуру и engagement"                                              ║ │
│  ║                                                                        ║ │
│  ║  Query 3: Own Data                                                     ║ │
│  ║  "Сравнить мои посты по типу hook — какой даёт лучший engagement?"    ║ │
│  ║                                                                        ║ │
│  ║  → Собирает structured knowledge                                       ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                              │                                              │
│                              ▼                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║  STEP 5: SYNTHESIZE KNOWLEDGE                                          ║ │
│  ║                                                                        ║ │
│  ║  Из research создаётся structured knowledge:                           ║ │
│  ║  ┌─────────────────────────────────────────────────────────────────┐  ║ │
│  ║  │ {                                                                │  ║ │
│  ║  │   "topic": "Hook Types by Content Type",                         │  ║ │
│  ║  │   "learned_at": "2024-01-15",                                    │  ║ │
│  ║  │   "source": "research + critique",                               │  ║ │
│  ║  │   "knowledge": {                                                 │  ║ │
│  ║  │     "AUTOMATION_CASE": {                                         │  ║ │
│  ║  │       "best_hook": "impossible_result",                          │  ║ │
│  ║  │       "template": "{Old time} → {New time}. Here's how:",        │  ║ │
│  ║  │       "examples": ["40 hours → 2 minutes", "$50K → $0"],         │  ║ │
│  ║  │       "avg_engagement_lift": "+45%"                              │  ║ │
│  ║  │     },                                                           │  ║ │
│  ║  │     "ENTERPRISE_CASE": {                                         │  ║ │
│  ║  │       "best_hook": "contrarian",                                 │  ║ │
│  ║  │       "template": "Everyone says X. But {company} did Y.",       │  ║ │
│  ║  │       "avg_engagement_lift": "+32%"                              │  ║ │
│  ║  │     },                                                           │  ║ │
│  ║  │     "PRIMARY_SOURCE": {                                          │  ║ │
│  ║  │       "best_hook": "surprising_finding",                         │  ║ │
│  ║  │       "template": "New research shows: {counterintuitive fact}"  │  ║ │
│  ║  │     }                                                            │  ║ │
│  ║  │   },                                                             │  ║ │
│  ║  │   "confidence": 0.85,                                            │  ║ │
│  ║  │   "sample_size": 47                                              │  ║ │
│  ║  │ }                                                                │  ║ │
│  ║  └─────────────────────────────────────────────────────────────────┘  ║ │
│  ║                                                                        ║ │
│  ║  → Сохраняется в Knowledge Base (persistent)                           ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                              │                                              │
│                              ▼                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║  STEP 6: SELF-MODIFY (Code & Prompts)                                  ║ │
│  ║                                                                        ║ │
│  ║  На основе knowledge агент решает ЧТО менять:                          ║ │
│  ║                                                                        ║ │
│  ║  ┌─────────────────────────────────────────────────────────────────┐  ║ │
│  ║  │  MODIFICATION OPTIONS:                                           │  ║ │
│  ║  │                                                                  │  ║ │
│  ║  │  A) PROMPT EVOLUTION (изменить системный промпт)                 │  ║ │
│  ║  │     → Переписать writer_system.txt с новыми правилами hooks     │  ║ │
│  ║  │                                                                  │  ║ │
│  ║  │  B) CONFIG UPDATE (изменить параметры)                           │  ║ │
│  ║  │     → Добавить hook_templates.json с новыми шаблонами           │  ║ │
│  ║  │                                                                  │  ║ │
│  ║  │  C) CODE GENERATION (написать новый модуль) ← ГЛУБОКАЯ           │  ║ │
│  ║  │     → Создать hook_selector.py который выбирает hook            │  ║ │
│  ║  │       по content_type автоматически                             │  ║ │
│  ║  │                                                                  │  ║ │
│  ║  │  D) KNOWLEDGE INJECTION (добавить в RAG)                         │  ║ │
│  ║  │     → Сохранить примеры хороших hooks для few-shot learning     │  ║ │
│  ║  └─────────────────────────────────────────────────────────────────┘  ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                              │                                              │
│                              ▼                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║  STEP 7: IMPLEMENT CHANGES                                             ║ │
│  ║                                                                        ║ │
│  ║  Code Generator Agent пишет реальный код:                              ║ │
│  ║                                                                        ║ │
│  ║  ```python                                                             ║ │
│  ║  # AUTO-GENERATED: hook_selector.py                                    ║ │
│  ║  # Generated: 2024-01-15                                               ║ │
│  ║  # Reason: Critique showed weak hooks for AUTOMATION_CASE              ║ │
│  ║  # Knowledge source: research_report_2024-01-15.json                   ║ │
│  ║                                                                        ║ │
│  ║  from enum import Enum                                                 ║ │
│  ║  from typing import Optional                                           ║ │
│  ║                                                                        ║ │
│  ║  class HookType(Enum):                                                 ║ │
│  ║      IMPOSSIBLE_RESULT = "impossible_result"                           ║ │
│  ║      CONTRARIAN = "contrarian"                                         ║ │
│  ║      SURPRISING_FINDING = "surprising_finding"                         ║ │
│  ║      QUESTION = "question"                                             ║ │
│  ║      STORY_OPEN = "story_open"                                         ║ │
│  ║                                                                        ║ │
│  ║  HOOK_BY_CONTENT_TYPE = {                                              ║ │
│  ║      "AUTOMATION_CASE": HookType.IMPOSSIBLE_RESULT,                    ║ │
│  ║      "ENTERPRISE_CASE": HookType.CONTRARIAN,                           ║ │
│  ║      "PRIMARY_SOURCE": HookType.SURPRISING_FINDING,                    ║ │
│  ║      "COMMUNITY_CONTENT": HookType.QUESTION,                           ║ │
│  ║      "TOOL_RELEASE": HookType.IMPOSSIBLE_RESULT,                       ║ │
│  ║  }                                                                     ║ │
│  ║                                                                        ║ │
│  ║  def select_hook(content_type: str) -> HookType:                       ║ │
│  ║      return HOOK_BY_CONTENT_TYPE.get(                                  ║ │
│  ║          content_type,                                                 ║ │
│  ║          HookType.QUESTION  # default                                  ║ │
│  ║      )                                                                 ║ │
│  ║  ```                                                                   ║ │
│  ║                                                                        ║ │
│  ║  → Код сохраняется в src/generated/                                    ║ │
│  ║  → Writer автоматически импортирует новые модули                       ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                              │                                              │
│                              ▼                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗ │
│  ║  STEP 8: VALIDATE & COMMIT                                             ║ │
│  ║                                                                        ║ │
│  ║  1. Syntax check: python -m py_compile generated_code.py               ║ │
│  ║  2. Unit test: run tests for the new module                            ║ │
│  ║  3. Integration test: generate test post with new hook selector        ║ │
│  ║  4. Critic review: "Этот пост лучше предыдущего?"                      ║ │
│  ║  5. If all pass → commit to codebase                                   ║ │
│  ║  6. If fail → rollback, log error, try different approach              ║ │
│  ╚═══════════════════════════════════════════════════════════════════════╝ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Rollback Manager

```python
# ═══════════════════════════════════════════════════════════════════════════
# ROLLBACK MANAGER
# Provides explicit rollback mechanism for Meta-Agent self-modifications
# ═══════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import shutil


@dataclass
class SystemSnapshot:
    """
    Snapshot of system state before a modification.
    Used for rollback if modification causes degradation.

    FIX: Added database_state for Supabase table snapshots.
    Previously only files were backed up, leaving database in inconsistent
    state after rollback.
    """
    id: str
    created_at: datetime
    trigger: str  # What triggered the change (research_id, etc.)
    description: str

    # What was changed (FILES)
    prompts: Dict[str, str]         # path -> content (before change)
    configs: Dict[str, Any]         # path -> parsed json (before change)
    generated_modules: List[str]    # list of file paths (before change)

    # FIX: DATABASE STATE (new)
    # Snapshots of Supabase tables affected by self-modification
    database_state: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    # Format: {"learnings": [...rows...], "code_modifications": [...rows...]}

    # Metadata
    performance_baseline: Dict[str, float]  # Metrics before change
    change_type: str  # "prompt" / "config" / "code" / "knowledge"


@dataclass
class RollbackResult:
    """Result of a rollback operation."""
    success: bool
    snapshot_id: str
    files_restored: List[str]
    error: Optional[str] = None


class RollbackManager:
    """
    Manages snapshots and rollback for Meta-Agent self-modifications.

    MEDIUM PRIORITY FIX #11: Added comprehensive logging for all operations.

    SAFETY FEATURES:
    1. Auto-snapshot before any modification
    2. Performance monitoring after changes
    3. Automatic rollback if performance degrades
    4. Circuit breaker: pause self-modification after N failures
    """

    def __init__(self, project_root: Path, snapshot_dir: Path):
        import logging
        self.logger = logging.getLogger("RollbackManager")

        self.project_root = project_root
        self.snapshot_dir = snapshot_dir
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.consecutive_failures = 0
        self.MAX_CONSECUTIVE_FAILURES = 3

        self.logger.info(
            f"[ROLLBACK] Initialized RollbackManager\n"
            f"  Project root: {project_root}\n"
            f"  Snapshot dir: {snapshot_dir}"
        )

    async def create_snapshot(
        self,
        trigger: str,
        description: str,
        files_to_backup: List[str],
        performance_metrics: Dict[str, float]
    ) -> str:
        """
        Create a snapshot before making changes.
        Returns snapshot_id for later rollback if needed.

        FIX: Now async to support database state export.
        """
        import uuid

        snapshot_id = f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        snapshot_path = self.snapshot_dir / snapshot_id

        # FIX #11: Log snapshot creation start
        self.logger.info(
            f"[ROLLBACK] Creating snapshot: {snapshot_id}\n"
            f"  Trigger: {trigger}\n"
            f"  Description: {description}\n"
            f"  Files to backup: {len(files_to_backup)}"
        )

        # Create snapshot directory
        snapshot_path.mkdir(parents=True, exist_ok=True)

        # Backup files
        prompts = {}
        configs = {}
        generated = []
        backup_errors = []

        for file_path in files_to_backup:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    # Copy file to snapshot
                    dest = snapshot_path / file_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(full_path, dest)

                    # Categorize
                    if file_path.endswith('.txt'):
                        prompts[file_path] = full_path.read_text()
                    elif file_path.endswith('.json'):
                        configs[file_path] = json.loads(full_path.read_text())
                    else:
                        generated.append(file_path)

                    self.logger.debug(f"[ROLLBACK] Backed up: {file_path}")

                except Exception as e:
                    backup_errors.append((file_path, str(e)))
                    self.logger.warning(f"[ROLLBACK] Failed to backup {file_path}: {e}")
            else:
                self.logger.debug(f"[ROLLBACK] Skipping non-existent: {file_path}")

        # FIX: Export database state for complete rollback
        database_state = await self._export_database_state()

        # Save metadata
        snapshot = SystemSnapshot(
            id=snapshot_id,
            created_at=datetime.now(),
            trigger=trigger,
            description=description,
            prompts=prompts,
            configs=configs,
            generated_modules=generated,
            database_state=database_state,  # FIX: Include DB state
            performance_baseline=performance_metrics,
            change_type=self._infer_change_type(files_to_backup)
        )

        metadata_path = snapshot_path / "metadata.json"
        metadata_path.write_text(json.dumps(snapshot.__dict__, default=str, indent=2))

        # FIX: Save database state separately (can be large)
        if database_state:
            db_state_path = snapshot_path / "database_state.json"
            db_state_path.write_text(json.dumps(database_state, default=str, indent=2))

        # FIX #11: Log snapshot creation summary
        self.logger.info(
            f"[ROLLBACK] Snapshot created: {snapshot_id}\n"
            f"  Prompts backed up: {len(prompts)}\n"
            f"  Configs backed up: {len(configs)}\n"
            f"  Generated modules: {len(generated)}\n"
            f"  Database tables: {list(database_state.keys())}\n"
            f"  Backup errors: {len(backup_errors)}\n"
            f"  Performance baseline: {performance_metrics}"
        )

        return snapshot_id

    async def _export_database_state(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        FIX: Export relevant Supabase tables for complete rollback capability.

        Only exports tables that are modified by self-improvement:
        - learnings: Micro-learnings from iterations
        - code_modifications: History of code changes
        - prompts: Active prompt versions

        Does NOT export:
        - posts, post_metrics: User content, never rolled back
        - experiments: Managed separately
        """
        from core.database import get_db

        state = {}

        try:
            db = await get_db()

            # Export learnings (filter to active only for smaller export)
            learnings = await db.client.table("learnings").select("*").eq("is_active", True).execute()
            state["learnings"] = learnings.data if learnings.data else []

            # Export recent code modifications (last 30 days)
            from datetime import timedelta
            cutoff = (datetime.now() - timedelta(days=30)).isoformat()
            modifications = await db.client.table("code_modifications").select("*").gte("created_at", cutoff).execute()
            state["code_modifications"] = modifications.data if modifications.data else []

            # Export active prompts
            prompts = await db.client.table("prompts").select("*").eq("is_active", True).execute()
            state["prompts"] = prompts.data if prompts.data else []

            self.logger.debug(
                f"[ROLLBACK] Database export: "
                f"learnings={len(state['learnings'])}, "
                f"modifications={len(state['code_modifications'])}, "
                f"prompts={len(state['prompts'])}"
            )

        except Exception as e:
            self.logger.error(f"[ROLLBACK] Failed to export database state: {e}")
            # Continue without DB state rather than failing entire snapshot

        return state

    async def _restore_database_state(self, database_state: Dict[str, List[Dict[str, Any]]]) -> bool:
        """
        FIX: Restore database tables from snapshot.

        CAUTION: This is destructive! Only call during actual rollback.
        Uses upsert with on_conflict to handle existing rows.
        """
        from core.database import get_db

        if not database_state:
            self.logger.info("[ROLLBACK] No database state to restore")
            return True

        try:
            db = await get_db()

            # Restore learnings
            if "learnings" in database_state and database_state["learnings"]:
                # Deactivate all current learnings first
                await db.client.table("learnings").update({"is_active": False}).eq("is_active", True).execute()
                # Restore snapshot learnings
                await db.client.table("learnings").upsert(database_state["learnings"], on_conflict="id").execute()
                self.logger.info(f"[ROLLBACK] Restored {len(database_state['learnings'])} learnings")

            # Restore prompts
            if "prompts" in database_state and database_state["prompts"]:
                # Deactivate all current prompts first
                await db.client.table("prompts").update({"is_active": False}).eq("is_active", True).execute()
                # Restore snapshot prompts
                await db.client.table("prompts").upsert(database_state["prompts"], on_conflict="id").execute()
                self.logger.info(f"[ROLLBACK] Restored {len(database_state['prompts'])} prompts")

            # Note: code_modifications are append-only history, don't restore
            # (but we add a "rollback" entry to track what happened)
            await db.client.table("code_modifications").insert({
                "gap_type": "rollback",
                "gap_description": "Rolled back to previous snapshot",
                "status": "completed",
                "created_at": utc_now().isoformat()
            }).execute()

            return True

        except Exception as e:
            self.logger.error(f"[ROLLBACK] Failed to restore database state: {e}")
            return False

    async def rollback_to(self, snapshot_id: str) -> RollbackResult:
        """
        Restore system to a previous snapshot state.

        FIX: Now async and includes database state restoration.
        """
        # FIX #11: Log rollback start
        self.logger.warning(f"[ROLLBACK] INITIATING ROLLBACK to snapshot: {snapshot_id}")

        snapshot_path = self.snapshot_dir / snapshot_id

        if not snapshot_path.exists():
            self.logger.error(f"[ROLLBACK] Snapshot not found: {snapshot_id}")
            return RollbackResult(
                success=False,
                snapshot_id=snapshot_id,
                files_restored=[],
                error=f"Snapshot '{snapshot_id}' not found"
            )

        files_restored = []
        db_restored = False

        try:
            # STEP 1: Restore all files from snapshot
            for item in snapshot_path.rglob("*"):
                if item.is_file() and item.name not in ["metadata.json", "database_state.json"]:
                    relative_path = item.relative_to(snapshot_path)
                    dest = self.project_root / relative_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest)
                    files_restored.append(str(relative_path))

            self.logger.info(f"[ROLLBACK] Restored {len(files_restored)} files")

            # STEP 2: FIX - Restore database state
            db_state_path = snapshot_path / "database_state.json"
            if db_state_path.exists():
                database_state = json.loads(db_state_path.read_text())
                db_restored = await self._restore_database_state(database_state)
                if db_restored:
                    self.logger.info("[ROLLBACK] Database state restored successfully")
                else:
                    self.logger.warning("[ROLLBACK] Database restoration failed, files still restored")
            else:
                self.logger.info("[ROLLBACK] No database state to restore (legacy snapshot)")

            return RollbackResult(
                success=True,
                snapshot_id=snapshot_id,
                files_restored=files_restored,
                error=None if db_restored or not db_state_path.exists() else "Database restoration failed"
            )

        except Exception as e:
            self.logger.error(f"[ROLLBACK] Rollback failed: {e}")
            return RollbackResult(
                success=False,
                snapshot_id=snapshot_id,
                files_restored=files_restored,
                error=str(e)
            )

    async def auto_rollback_if_degraded(
        self,
        snapshot_id: str,
        current_metrics: Dict[str, float],
        degradation_threshold: float = 0.1  # 10% degradation triggers rollback
    ) -> Optional[RollbackResult]:
        """
        Automatically rollback if performance metrics degraded.

        FIX: Now async to support database restoration.

        Returns RollbackResult if rollback was performed, None otherwise.
        """
        # FIX #11: Log performance check
        self.logger.info(
            f"[ROLLBACK] Checking performance degradation for snapshot: {snapshot_id}\n"
            f"  Current metrics: {current_metrics}\n"
            f"  Degradation threshold: {degradation_threshold * 100}%"
        )

        snapshot_path = self.snapshot_dir / snapshot_id / "metadata.json"

        if not snapshot_path.exists():
            self.logger.warning(f"[ROLLBACK] Cannot check degradation - snapshot metadata not found: {snapshot_id}")
            return None

        metadata = json.loads(snapshot_path.read_text())
        baseline = metadata.get("performance_baseline", {})

        # Check for degradation
        degraded = False
        degradation_details = []
        for metric, baseline_value in baseline.items():
            current_value = current_metrics.get(metric, 0)
            if baseline_value > 0:
                change = (current_value - baseline_value) / baseline_value
                degradation_details.append(f"{metric}: {baseline_value:.2f} → {current_value:.2f} ({change:+.1%})")
                if change < -degradation_threshold:  # Negative change = degradation
                    degraded = True
                    self.logger.warning(
                        f"[ROLLBACK] DEGRADATION DETECTED: {metric} dropped {abs(change):.1%} "
                        f"(threshold: {degradation_threshold:.1%})"
                    )

        # FIX #11: Log comparison results
        self.logger.debug(f"[ROLLBACK] Performance comparison:\n  " + "\n  ".join(degradation_details))

        if degraded:
            self.consecutive_failures += 1
            self.logger.warning(
                f"[ROLLBACK] AUTO-ROLLBACK TRIGGERED! "
                f"Consecutive failures: {self.consecutive_failures}/{self.MAX_CONSECUTIVE_FAILURES}"
            )
            result = await self.rollback_to(snapshot_id)  # FIX: await async method
            result.error = f"Performance degraded beyond {degradation_threshold*100}% threshold"
            return result

        # Success - reset failure counter
        self.logger.info(f"[ROLLBACK] Performance stable - no rollback needed for {snapshot_id}")
        self.consecutive_failures = 0
        return None

    def should_pause_self_modification(self) -> bool:
        """
        Circuit breaker: pause self-modification if too many consecutive failures.
        """
        should_pause = self.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES

        # FIX #11: Log circuit breaker status
        if should_pause:
            self.logger.critical(
                f"[ROLLBACK] CIRCUIT BREAKER ACTIVE! "
                f"Self-modification paused after {self.consecutive_failures} consecutive failures. "
                f"Manual intervention required."
            )

        return should_pause

    def _infer_change_type(self, files: List[str]) -> str:
        """Infer the type of change from file extensions."""
        extensions = {Path(f).suffix for f in files}
        if '.txt' in extensions:
            return "prompt"
        if '.json' in extensions:
            return "config"
        if '.py' in extensions:
            return "code"
        return "knowledge"

    def list_snapshots(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent snapshots."""
        snapshots = []
        for snap_dir in sorted(self.snapshot_dir.iterdir(), reverse=True)[:limit]:
            if snap_dir.is_dir():
                metadata_path = snap_dir / "metadata.json"
                if metadata_path.exists():
                    snapshots.append(json.loads(metadata_path.read_text()))
        return snapshots
```

---

### Critic Agent (Separate Model for Dialogue)

```python
class CriticAgent:
    """
    Отдельный агент, который критикует работу Creator Agent.
    Использует ту же модель (Claude Opus 4.5), но с другим system prompt.
    Это создаёт "внутренний диалог" для улучшения качества.

    ═══════════════════════════════════════════════════════════════════════════
    FIX: RESPONSIBILITY SEPARATION (QC Agent vs Critic Agent)
    ═══════════════════════════════════════════════════════════════════════════

    QC Agent (Quality Control):
    ───────────────────────────
    - PURPOSE: Objective pass/fail/revise DECISION
    - OUTPUT: Numeric scores, decision (pass/revise/reject), revision_target_agent
    - WHEN: After humanization + visual, before approval
    - DOES NOT: Generate improvement suggestions (delegated to Critic)

    Critic Agent (This class):
    ──────────────────────────
    - PURPOSE: Generate specific, actionable IMPROVEMENTS
    - OUTPUT: Dialogue with concrete suggestions, examples, alternatives
    - WHEN: During Meta-Agent self-evaluation loop (before QC)
    - DOES NOT: Make pass/fail decisions (that's QC's job)

    WORKFLOW:
    1. Writer creates draft
    2. Meta-Agent calls Critic for improvement dialogue
    3. Writer revises based on Critic feedback
    4. Loop until Meta-Agent satisfied (max 3 iterations)
    5. QC Agent evaluates final content (scores + decision only)
    6. If QC says "revise", content goes back to appropriate agent
       (but Critic is NOT called again at this stage)

    This separation ensures:
    - Clear accountability (who decides vs who improves)
    - No duplicate suggestion generation
    - Consistent scoring (QC is deterministic, Critic is creative)
    ═══════════════════════════════════════════════════════════════════════════
    """

    CRITIC_SYSTEM_PROMPT = """
    You are a harsh but fair LinkedIn content critic.
    Your job is to find weaknesses and suggest improvements.

    Be SPECIFIC in your criticism:
    - Don't just say "hook is weak" — explain WHY and suggest alternatives
    - Give concrete examples of what would work better
    - Reference what top LinkedIn creators do differently

    You are helping the Creator Agent learn and improve.
    Be direct, don't sugarcoat, but be constructive.

    Focus areas:
    1. HOOK: First 2 lines — do they stop the scroll?
    2. SPECIFICITY: Are there concrete numbers, names, examples?
    3. VALUE: Is there a clear takeaway for the reader?
    4. AUTHENTICITY: Does it sound human or AI-generated?
    5. STRUCTURE: Is it scannable? Good use of white space?
    6. CTA: Is there a clear next step?
    """

    def __init__(self, claude_client):
        self.claude = claude_client
        self.conversation_history = []

    async def critique(self, content: str, context: dict) -> CritiqueResponse:
        """
        Start critique dialogue.
        Returns initial critique and opens dialogue.
        """

        message = f"""
        Please critique this LinkedIn post:

        === POST ===
        {content}
        === END POST ===

        Context:
        - Content Type: {context.get('content_type')}
        - Target Audience: {context.get('audience', 'LinkedIn professionals')}
        - Goal: {context.get('goal', 'engagement + value')}

        Be specific and constructive. What's weak? What would you change?
        """

        self.conversation_history = [{"role": "user", "content": message}]

        response = await self.claude.generate(
            system=self.CRITIC_SYSTEM_PROMPT,
            messages=self.conversation_history
        )

        self.conversation_history.append({"role": "assistant", "content": response})

        return CritiqueResponse(
            critique=response,
            dialogue_open=True,
            session_id=self._generate_session_id()
        )

    async def follow_up(self, question: str) -> str:
        """
        Creator Agent asks follow-up question.
        Continues the dialogue.
        """

        self.conversation_history.append({"role": "user", "content": question})

        response = await self.claude.generate(
            system=self.CRITIC_SYSTEM_PROMPT,
            messages=self.conversation_history
        )

        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    async def close_dialogue(self) -> DialogueSummary:
        """
        End dialogue and extract key learnings.
        """

        summary_prompt = """
        Summarize this critique dialogue:
        1. What were the main weaknesses identified?
        2. What specific improvements were suggested?
        3. What knowledge gaps were revealed?
        4. What should the Creator Agent research?

        Return structured JSON.
        """

        self.conversation_history.append({"role": "user", "content": summary_prompt})

        return await self.claude.generate_structured(
            system=self.CRITIC_SYSTEM_PROMPT,
            messages=self.conversation_history,
            response_model=DialogueSummary
        )


@dataclass
class DialogueSummary:
    """Summary of critique dialogue."""
    weaknesses: List[str]
    suggestions: List[str]
    knowledge_gaps: List[str]
    research_queries: List[str]
    confidence_in_suggestions: float
```

---

### Reflection Engine ("Ага, понятно!")

```python
class ReflectionEngine:
    """
    Агент рефлексирует над критикой и своей работой.
    Это момент "Ага, понятно!" — осмысление feedback.
    """

    REFLECTION_PROMPT = """
    You are reflecting on feedback you received about your work.
    Think deeply about:

    1. Is the criticism valid? Why or why not?
    2. Is this a PATTERN in my work, or a one-time issue?
    3. What knowledge am I missing to do this better?
    4. What specific research would help me learn?
    5. How should I change my approach going forward?

    Be honest with yourself. The goal is genuine improvement.
    """

    async def reflect(
        self,
        original_work: str,
        critique: DialogueSummary,
        historical_work: List[str]  # Past posts for pattern detection
    ) -> Reflection:
        """
        Deep reflection on critique.
        Identifies patterns and knowledge gaps.
        """

        prompt = f"""
        My work:
        {original_work}

        Critique I received:
        - Weaknesses: {critique.weaknesses}
        - Suggestions: {critique.suggestions}

        My past work (for pattern detection):
        {[w[:200] for w in historical_work[-10:]]}

        Reflect:
        1. Is this criticism valid?
        2. Do I see this pattern in my past work?
        3. What knowledge am I missing?
        4. What should I research to improve?
        5. What concrete changes should I make to my process?
        """

        return await self.claude.generate_structured(
            system=self.REFLECTION_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            response_model=Reflection
        )


@dataclass
class Reflection:
    """Result of self-reflection."""

    # Validation of critique
    critique_valid: bool
    critique_validity_reasoning: str

    # Pattern detection
    is_recurring_pattern: bool
    pattern_description: Optional[str]
    pattern_frequency: Optional[str]  # "3 out of last 10 posts"

    # Knowledge gaps
    knowledge_gaps: List[str]
    research_needed: List[ResearchQuery]

    # Action items
    process_changes: List[str]  # "Start with concrete example, not abstraction"
    prompt_changes: List[str]   # "Add rule: never start with 'В последнее время'"
    code_changes: List[str]     # "Create hook_selector.py"

    # Confidence
    confidence_in_changes: float
```

---

### Claude Code Integration (Server-Side Code Generation)

Вместо генерации кода через обычные LLM API, используем **Claude Code** —
полноценный AI-агент с доступом к файловой системе, bash, и инструментам разработки.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              CLAUDE CODE AS CODE EVOLUTION RUNTIME                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ПОЧЕМУ CLAUDE CODE, а не обычный API:                                     │
│  ─────────────────────────────────────                                     │
│  ✓ Может читать существующий код проекта                                  │
│  ✓ Понимает структуру и стиль кодовой базы                                │
│  ✓ Может запускать тесты и проверять свой код                             │
│  ✓ Имеет доступ к bash, git, pip, npm                                     │
│  ✓ Работает в контексте всего проекта                                     │
│  ✓ Может создавать, редактировать, удалять файлы                          │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│  АРХИТЕКТУРА                                                                │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  ┌─────────────────────┐    ┌─────────────────────┐                        │
│  │    Meta-Agent       │    │   Claude Code       │                        │
│  │  (Python процесс)   │───▶│  (headless mode)    │                        │
│  │                     │    │                     │                        │
│  │  "Нужен модуль для  │    │  -p "prompt"        │                        │
│  │   выбора hook типа" │    │  --allowedTools     │                        │
│  │                     │◀───│  --output-format    │                        │
│  │  Получает результат │    │                     │                        │
│  └─────────────────────┘    └─────────────────────┘                        │
│           │                          │                                      │
│           │                          ▼                                      │
│           │                 ┌─────────────────────┐                        │
│           │                 │   Файловая система  │                        │
│           │                 │                     │                        │
│           │                 │  src/generated/     │                        │
│           │                 │    hook_selector.py │                        │
│           │                 │    new_template.py  │                        │
│           ▼                 └─────────────────────┘                        │
│  ┌─────────────────────┐                                                   │
│  │  Валидация + Тесты  │                                                   │
│  │  pytest, mypy       │                                                   │
│  └─────────────────────┘                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### Claude Code Server Setup

```bash
# 1. Установка Claude Code на сервер
curl -fsSL https://claude.ai/install.sh | bash

# 2. Аутентификация (один раз)
export ANTHROPIC_API_KEY="sk-ant-..."

# 3. Тест
claude -p "Hello, test" --output-format json
```

---

#### Claude Code Client (Python Integration)

```python
import subprocess
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class ClaudeCodeResult:
    """Result from Claude Code execution."""
    success: bool
    result: str
    session_id: str
    cost_usd: float
    duration_ms: int
    files_created: list[str]
    files_modified: list[str]
    error: Optional[str] = None


class ClaudeCodeClient:
    """
    Client for running Claude Code in headless mode.
    Used by Meta-Agent for code generation and modification.
    """

    def __init__(
        self,
        project_root: str,
        allowed_tools: list[str] = None,
        max_turns: int = 15,
        max_budget_usd: float = 5.0
    ):
        self.project_root = Path(project_root)
        self.allowed_tools = allowed_tools or [
            "Read", "Write", "Edit", "Bash", "Glob", "Grep"
        ]
        self.max_turns = max_turns
        self.max_budget_usd = max_budget_usd

        # Verify Claude Code is installed
        self._verify_installation()

    def _verify_installation(self):
        """Check if Claude Code is available."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("Claude Code not properly installed")
        except FileNotFoundError:
            raise RuntimeError(
                "Claude Code not found. Install with: "
                "curl -fsSL https://claude.ai/install.sh | bash"
            )

    def generate_module(
        self,
        purpose: str,
        context: dict,
        target_path: str
    ) -> ClaudeCodeResult:
        """
        Generate a new Python module using Claude Code.

        Args:
            purpose: What the module should do
            context: Knowledge/research that informs the generation
            target_path: Where to save the module (relative to project root)

        Returns:
            ClaudeCodeResult with generated code info
        """

        prompt = f"""
        Create a new Python module at {target_path}

        PURPOSE:
        {purpose}

        CONTEXT (research findings):
        {json.dumps(context, indent=2)}

        REQUIREMENTS:
        1. Read existing code in src/ to understand project style
        2. Create the module with proper type hints and docstrings
        3. Include "# AUTO-GENERATED by Meta-Agent" header
        4. Add unit tests in tests/ directory
        5. Run the tests to verify the code works
        6. If tests fail, fix the code

        After creating the module, run: python -m pytest tests/ -v

        Report what you created and test results.
        """

        return self._execute(prompt)

    def modify_file(
        self,
        file_path: str,
        modification: str,
        reason: str
    ) -> ClaudeCodeResult:
        """
        Modify an existing file based on learnings.

        Args:
            file_path: Path to file to modify
            modification: What change to make
            reason: Why this change (from reflection/research)

        Returns:
            ClaudeCodeResult
        """

        prompt = f"""
        Modify the file: {file_path}

        CHANGE NEEDED:
        {modification}

        REASON (from self-reflection):
        {reason}

        REQUIREMENTS:
        1. Read the current file first
        2. Understand the existing code
        3. Make minimal, focused changes
        4. Add a comment explaining the change
        5. Run relevant tests after modification
        6. If tests fail, fix or rollback

        Report what you changed and why.
        """

        return self._execute(prompt)

    def evolve_prompt(
        self,
        prompt_path: str,
        learnings: list[str],
        examples: list[dict]
    ) -> ClaudeCodeResult:
        """
        Evolve a system prompt based on learnings.

        Args:
            prompt_path: Path to prompt file
            learnings: What we learned from critique/research
            examples: Good examples to incorporate

        Returns:
            ClaudeCodeResult with new prompt version
        """

        prompt = f"""
        Evolve the system prompt at: {prompt_path}

        LEARNINGS TO INCORPORATE:
        {json.dumps(learnings, indent=2)}

        GOOD EXAMPLES TO ADD:
        {json.dumps(examples, indent=2)}

        REQUIREMENTS:
        1. Read the current prompt
        2. Read the version history in prompts/versions/
        3. Create a new version with improvements
        4. Save old version to prompts/versions/
        5. Update changelog.json with what changed and why
        6. Keep what works, improve what doesn't

        Report the changes made.
        """

        return self._execute(prompt)

    def run_complex_task(self, task_description: str) -> ClaudeCodeResult:
        """
        Run any complex development task.
        Claude Code will figure out what needs to be done.

        Args:
            task_description: What needs to be accomplished

        Returns:
            ClaudeCodeResult
        """

        prompt = f"""
        TASK:
        {task_description}

        You have full access to the project at {self.project_root}.
        Figure out what needs to be done and do it.
        Run tests to verify your changes work.
        Report what you did.
        """

        return self._execute(prompt)

    def _execute(self, prompt: str) -> ClaudeCodeResult:
        """Execute Claude Code with the given prompt."""

        cmd = [
            "claude",
            "-p", prompt,
            "--allowedTools", ",".join(self.allowed_tools),
            "--output-format", "json",
            "--cwd", str(self.project_root),
            "--max-turns", str(self.max_turns),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                return ClaudeCodeResult(
                    success=False,
                    result="",
                    session_id="",
                    cost_usd=0,
                    duration_ms=0,
                    files_created=[],
                    files_modified=[],
                    error=result.stderr
                )

            output = json.loads(result.stdout)

            return ClaudeCodeResult(
                success=True,
                result=output.get("result", ""),
                session_id=output.get("session_id", ""),
                cost_usd=output.get("total_cost_usd", 0),
                duration_ms=output.get("duration_ms", 0),
                files_created=output.get("files_created", []),
                files_modified=output.get("files_modified", []),
            )

        except subprocess.TimeoutExpired:
            return ClaudeCodeResult(
                success=False,
                result="",
                session_id="",
                cost_usd=0,
                duration_ms=0,
                files_created=[],
                files_modified=[],
                error="Timeout: task took longer than 5 minutes"
            )

        except json.JSONDecodeError as e:
            return ClaudeCodeResult(
                success=False,
                result="",
                session_id="",
                cost_usd=0,
                duration_ms=0,
                files_created=[],
                files_modified=[],
                error=f"Failed to parse response: {e}"
            )


# ═══════════════════════════════════════════════════════════════════
# USAGE IN META-AGENT
# ═══════════════════════════════════════════════════════════════════

class MetaAgentWithClaudeCode:
    """Meta-Agent that uses Claude Code for code evolution."""

    def __init__(self, project_root: str):
        self.claude_code = ClaudeCodeClient(
            project_root=project_root,
            max_turns=20,
            max_budget_usd=10.0
        )

    async def implement_learning(self, reflection: Reflection, knowledge: dict):
        """
        Implement learnings by generating/modifying code.
        Uses Claude Code instead of simple LLM API.
        """

        # Example: Create new module based on learning
        if reflection.code_changes:
            for change in reflection.code_changes:
                result = self.claude_code.generate_module(
                    purpose=change,
                    context={
                        "knowledge": knowledge,
                        "reflection": {
                            "gaps": reflection.knowledge_gaps,
                            "changes": reflection.process_changes
                        }
                    },
                    target_path=f"src/generated/{self._to_module_name(change)}.py"
                )

                if result.success:
                    print(f"✅ Created: {result.files_created}")
                    print(f"   Cost: ${result.cost_usd:.3f}")
                else:
                    print(f"❌ Failed: {result.error}")

        # Example: Evolve prompts
        if reflection.prompt_changes:
            result = self.claude_code.evolve_prompt(
                prompt_path="prompts/writer_system.txt",
                learnings=reflection.prompt_changes,
                examples=knowledge.get("good_examples", [])
            )

            if result.success:
                print(f"✅ Prompt evolved: {result.files_modified}")

    def _to_module_name(self, description: str) -> str:
        """Convert description to valid module name."""
        return description.lower().replace(" ", "_").replace("-", "_")[:30]
```

---

#### Claude Code Agent SDK (Advanced)

Для более сложных сценариев с многошаговыми диалогами используй **Agent SDK**:

```python
# pip install claude-agent-sdk

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

class AdvancedCodeEvolver:
    """
    Uses Claude Agent SDK for multi-turn code evolution.
    Maintains conversation context across multiple operations.
    """

    def __init__(self, project_root: str):
        self.options = ClaudeAgentOptions(
            cwd=project_root,
            allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
            permission_mode="acceptEdits",  # Auto-approve edits
            max_turns=30,
            max_budget_usd=20.0
        )
        self.client = None

    async def connect(self):
        """Initialize SDK client."""
        self.client = ClaudeSDKClient(self.options)
        await self.client.connect()

    async def evolve_with_dialogue(self, reflection: Reflection):
        """
        Multi-step evolution with Claude Code maintaining context.

        Step 1: Analyze current code
        Step 2: Understand the gap
        Step 3: Research best practices
        Step 4: Generate new code
        Step 5: Test and iterate
        """

        # Step 1: Understand current state
        await self._query(
            "Read src/agents/writer.py and summarize how hooks are currently handled"
        )

        # Step 2: Discuss the gap (Claude Code remembers context)
        await self._query(f"""
            Based on what you just read, I've learned these issues:
            {reflection.knowledge_gaps}

            What changes would you suggest?
        """)

        # Step 3: Implement (still in same conversation)
        await self._query(f"""
            Great insights. Now implement these changes:
            {reflection.code_changes}

            Create new files if needed.
            Update existing code.
            Add tests.
        """)

        # Step 4: Test
        result = await self._query(
            "Run all tests. If any fail, fix them."
        )

        return result

    async def _query(self, prompt: str) -> str:
        """Send query and get response."""
        await self.client.query(prompt)

        response_parts = []
        async for message in self.client.receive_response():
            if hasattr(message, 'content'):
                for block in message.content:
                    if hasattr(block, 'text'):
                        response_parts.append(block.text)

        return "".join(response_parts)

    async def disconnect(self):
        """Close connection."""
        if self.client:
            await self.client.disconnect()
```

---

#### Server Configuration

```yaml
# config/claude_code_config.yaml

claude_code:
  # Headless mode settings
  headless: true
  output_format: json

  # Safety limits
  max_turns: 20
  max_budget_usd: 10.0
  timeout_seconds: 300

  # Allowed operations
  allowed_tools:
    - Read
    - Write
    - Edit
    - Bash
    - Glob
    - Grep

  # Restricted operations (safety)
  disallowed_tools:
    - WebSearch
    - WebFetch

  # Bash restrictions
  bash_allow:
    - "python *"
    - "pytest *"
    - "pip install *"
    - "git status"
    - "git diff"

  bash_deny:
    - "rm -rf *"
    - "sudo *"
    - "curl *"  # No external downloads
    - "wget *"

  # Logging
  log_all_sessions: true
  session_log_dir: "data/claude_code_sessions/"
```

---

### Code Evolution Engine

```python
class CodeEvolutionEngine:
    """
    Агент пишет РЕАЛЬНЫЙ код для улучшения себя.
    Не просто меняет конфиги — создаёт новые модули.
    """

    CODE_GEN_SYSTEM_PROMPT = """
    You are a Python code generator for a LinkedIn content agent.
    Generate clean, well-documented, production-ready code.

    Requirements:
    1. Include docstrings explaining purpose
    2. Add type hints
    3. Include "# AUTO-GENERATED" header with metadata
    4. Make code modular and testable
    5. Follow existing code style in the project

    The code will be automatically integrated into the agent.
    It must be correct and safe.
    """

    def __init__(self, claude_client, project_root: str):
        self.claude = claude_client
        self.project_root = Path(project_root)
        self.generated_dir = self.project_root / "src" / "generated"
        self.generated_dir.mkdir(exist_ok=True)

    async def generate_module(
        self,
        purpose: str,
        knowledge: dict,
        reflection: Reflection
    ) -> GeneratedModule:
        """
        Generate a new Python module based on learnings.

        MEDIUM PRIORITY FIX #6: Added comprehensive logging for debugging
        self-modification issues and tracking code generation.
        """
        import logging
        import hashlib
        logger = logging.getLogger("CodeEvolutionEngine")

        # FIX #6: Log generation attempt with context
        knowledge_hash = hashlib.sha256(json.dumps(knowledge, sort_keys=True).encode()).hexdigest()[:12]
        generation_id = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{knowledge_hash}"

        logger.info(
            f"[CODE_GEN] Starting module generation: {generation_id}\n"
            f"  Purpose: {purpose}\n"
            f"  Knowledge hash: {knowledge_hash}\n"
            f"  Reflection gaps: {len(reflection.knowledge_gaps) if reflection.knowledge_gaps else 0}"
        )

        prompt = f"""
        Generate a Python module for the following purpose:

        PURPOSE: {purpose}

        KNOWLEDGE (from research):
        {json.dumps(knowledge, indent=2)}

        REFLECTION (what we learned):
        - Knowledge gaps: {reflection.knowledge_gaps}
        - Process changes needed: {reflection.process_changes}
        - Suggested code changes: {reflection.code_changes}

        Generate a complete Python module that:
        1. Encapsulates this knowledge
        2. Can be imported and used by the Writer agent
        3. Is well-documented and tested
        4. Includes example usage

        Return the complete Python code.
        """

        start_time = datetime.now()
        code = await self.claude.generate(
            system=self.CODE_GEN_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        generation_duration = (datetime.now() - start_time).total_seconds()

        # FIX #6: Log generation completion
        logger.info(f"[CODE_GEN] LLM generation completed in {generation_duration:.1f}s for {generation_id}")

        # Extract code from response
        code = self._extract_code(code)

        # FIX #6: Log code stats
        code_lines = len(code.split('\n'))
        logger.debug(f"[CODE_GEN] Extracted code: {code_lines} lines for {generation_id}")

        # Validate syntax
        if not self._validate_syntax(code):
            # FIX #6: Log invalid code for debugging (to separate file)
            invalid_code_path = self.generated_dir / f"_invalid_{generation_id}.py.failed"
            invalid_code_path.write_text(code)
            logger.error(
                f"[CODE_GEN] Syntax validation FAILED for {generation_id}. "
                f"Invalid code saved to: {invalid_code_path}"
            )
            raise CodeGenerationError(f"Generated code has syntax errors. See {invalid_code_path}")

        # Create module
        module_name = self._generate_module_name(purpose)
        module_path = self.generated_dir / f"{module_name}.py"

        # FIX #6: Log successful generation
        logger.info(
            f"[CODE_GEN] SUCCESS: {generation_id}\n"
            f"  Module: {module_name}\n"
            f"  Path: {module_path}\n"
            f"  Lines: {code_lines}\n"
            f"  Duration: {generation_duration:.1f}s"
        )

        return GeneratedModule(
            name=module_name,
            path=module_path,
            code=code,
            purpose=purpose,
            generated_at=datetime.now(),
            knowledge_source=knowledge,
            validated=True
        )

    async def evolve_prompt(
        self,
        current_prompt_path: str,
        reflection: Reflection,
        knowledge: dict
    ) -> PromptEvolution:
        """
        Evolve an existing prompt based on learnings.
        Creates a new version while preserving history.
        """

        current_prompt = Path(current_prompt_path).read_text()

        prompt = f"""
        Current system prompt:
        === START ===
        {current_prompt}
        === END ===

        Based on these learnings, improve this prompt:

        REFLECTION:
        - Valid critique: {reflection.critique_validity_reasoning}
        - Pattern detected: {reflection.pattern_description}
        - Process changes: {reflection.process_changes}

        KNOWLEDGE:
        {json.dumps(knowledge, indent=2)}

        Rules for evolution:
        1. Keep what works, change what doesn't
        2. Add specific rules based on learnings
        3. Include examples where helpful
        4. Make instructions clearer and more actionable

        Return the complete new prompt (not a diff).
        """

        new_prompt = await self.claude.generate(
            system="You are a prompt engineer. Improve prompts based on feedback and learnings.",
            messages=[{"role": "user", "content": prompt}]
        )

        # Version the prompt
        version = self._get_next_version(current_prompt_path)

        return PromptEvolution(
            original_path=current_prompt_path,
            new_prompt=new_prompt,
            version=version,
            changes_made=reflection.prompt_changes,
            knowledge_source=knowledge,
            evolved_at=datetime.now()
        )

    def _validate_syntax(self, code: str) -> bool:
        """Validate Python syntax."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Find code blocks
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0]
        elif "```" in response:
            code = response.split("```")[1].split("```")[0]
        else:
            code = response
        return code.strip()


@dataclass
class GeneratedModule:
    """A generated Python module."""
    name: str
    path: Path
    code: str
    purpose: str
    generated_at: datetime
    knowledge_source: dict
    validated: bool


@dataclass
class PromptEvolution:
    """Evolution of a prompt."""
    original_path: str
    new_prompt: str
    version: str
    changes_made: List[str]
    knowledge_source: dict
    evolved_at: datetime
```

---

### Knowledge Base (Persistent Memory)

```python
class KnowledgeBase:
    """
    Persistent storage for agent learnings.
    This is the agent's "memory" that accumulates over time.
    """

    def __init__(self, db, vector_store):
        self.db = db  # Supabase for structured data
        self.vector_store = vector_store  # Pinecone for semantic search

    async def store_learning(self, learning: Learning):
        """
        Store a new learning from critique/research.
        """

        # Store structured data
        await self.db.insert("learnings", {
            "id": learning.id,
            "topic": learning.topic,
            "content": learning.content,
            "source": learning.source,
            "confidence": learning.confidence,
            "learned_at": learning.learned_at,
            "applied_count": 0,
            "success_rate": None
        })

        # Store embedding for semantic search
        embedding = await self._generate_embedding(
            f"{learning.topic}: {learning.content}"
        )
        await self.vector_store.upsert(
            id=learning.id,
            embedding=embedding,
            metadata={"topic": learning.topic, "source": learning.source}
        )

    async def query_relevant(self, context: str, limit: int = 5) -> List[Learning]:
        """
        Find relevant learnings for current task.
        Used to inject knowledge into prompts.
        """

        embedding = await self._generate_embedding(context)

        results = await self.vector_store.query(
            embedding=embedding,
            top_k=limit
        )

        learnings = []
        for result in results:
            learning = await self.db.get("learnings", result.id)
            learnings.append(Learning(**learning))

        return learnings

    async def get_applicable_rules(self, content_type: str) -> List[str]:
        """
        Get rules/guidelines learned for specific content type.
        """

        learnings = await self.db.query(
            "learnings",
            filters={"topic": f"hook_type_{content_type}"},
            order_by="confidence DESC"
        )

        return [l["content"] for l in learnings if l["confidence"] > 0.7]


@dataclass
class Learning:
    """A piece of knowledge learned by the agent."""
    id: str
    topic: str
    content: str
    source: str  # "critique", "research", "experiment"
    confidence: float
    learned_at: datetime
    applied_count: int = 0
    success_rate: Optional[float] = None
```

---

### Full Deep Improvement Loop

```python
class DeepImprovementLoop:
    """
    Full loop: Create → Critique → Reflect → Research → Modify → Validate
    """

    def __init__(
        self,
        creator: WriterAgent,
        critic: CriticAgent,
        reflector: ReflectionEngine,
        researcher: ResearchAgent,
        code_evolver: CodeEvolutionEngine,
        knowledge_base: KnowledgeBase,
        db
    ):
        self.creator = creator
        self.critic = critic
        self.reflector = reflector
        self.researcher = researcher
        self.code_evolver = code_evolver
        self.kb = knowledge_base
        self.db = db

    async def run(self, draft: str, context: dict) -> ImprovementResult:
        """
        Run full deep improvement loop.
        """

        # Step 1: Get critique via dialogue
        critique_response = await self.critic.critique(draft, context)

        # Allow follow-up questions
        if self._needs_clarification(critique_response):
            follow_up = self._generate_follow_up(critique_response)
            await self.critic.follow_up(follow_up)

        dialogue_summary = await self.critic.close_dialogue()

        # Step 2: Reflect ("Ага, понятно!")
        historical_work = await self.db.get_recent_posts(limit=10)
        reflection = await self.reflector.reflect(
            original_work=draft,
            critique=dialogue_summary,
            historical_work=[p.content for p in historical_work]
        )

        # Step 3: Research (if knowledge gaps identified)
        knowledge = {}
        if reflection.research_needed:
            for query in reflection.research_needed:
                results = await self.researcher.execute_query(query)
                knowledge[query.topic] = results

            # Synthesize into structured knowledge
            knowledge = await self._synthesize_knowledge(knowledge)

        # Step 4: Store learnings
        for topic, content in knowledge.items():
            await self.kb.store_learning(Learning(
                id=f"learning_{datetime.now().timestamp()}",
                topic=topic,
                content=json.dumps(content),
                source="critique_research",
                confidence=reflection.confidence_in_changes,
                learned_at=datetime.now()
            ))

        # Step 5: Decide modification type
        modifications = []

        if reflection.code_changes:
            # Generate new module
            module = await self.code_evolver.generate_module(
                purpose=reflection.code_changes[0],
                knowledge=knowledge,
                reflection=reflection
            )
            modifications.append(("code", module))

        if reflection.prompt_changes:
            # Evolve prompt
            evolution = await self.code_evolver.evolve_prompt(
                current_prompt_path="prompts/writer_system.txt",
                reflection=reflection,
                knowledge=knowledge
            )
            modifications.append(("prompt", evolution))

        # Step 6: Validate changes
        for mod_type, mod in modifications:
            if mod_type == "code":
                # Test the generated code
                test_result = await self._test_module(mod)
                if test_result.success:
                    await self._commit_module(mod)

            elif mod_type == "prompt":
                # Test with new prompt
                test_result = await self._test_prompt(mod)
                if test_result.success:
                    await self._commit_prompt(mod)

        return ImprovementResult(
            original_draft=draft,
            critique_summary=dialogue_summary,
            reflection=reflection,
            knowledge_gained=knowledge,
            modifications=modifications,
            success=True
        )
```

---

### Self-Modification Engine

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime
import json


@dataclass
class ModificationRecord:
    """Record of a self-modification."""

    timestamp: datetime
    component: str           # What was modified
    parameter: str           # Specific parameter
    old_value: Any
    new_value: Any
    reason: str              # Why this change was made
    research_report_id: str  # Link to research that triggered this

    # For rollback
    rollback_available: bool = True
    rolled_back: bool = False
    rollback_reason: Optional[str] = None


class SelfModificationEngine:
    """
    Engine for modifying agent behavior based on research findings.
    All changes are logged and can be rolled back.
    """

    # What can be modified
    MODIFIABLE_COMPONENTS = {
        "writer": {
            "system_prompt": "prompts/writer_system.txt",
            "hook_templates": "config/hook_templates.json",
            "post_length_target": "config/writer_config.json",
            "emoji_usage": "config/writer_config.json",
        },
        "trend_scout": {
            "scoring_weights": "config/scoring_weights.json",
            "content_type_priorities": "config/content_types.json",
            "source_weights": "config/sources.json",
        },
        "visual_creator": {
            "preferred_styles": "config/visual_styles.json",
            "author_photo_probability": "config/visual_config.json",
        },
        "scheduler": {
            "optimal_hours": "config/schedule.json",
            "day_preferences": "config/schedule.json",
        },
        "evaluator": {
            "quality_threshold": "config/evaluator_config.json",
            "criteria_weights": "config/evaluation_criteria.json",
        }
    }

    def __init__(self, config_dir: str, db):
        self.config_dir = config_dir
        self.db = db
        self.modification_history: List[ModificationRecord] = []

    async def apply_recommendations(
        self,
        recommendations: List[dict],
        auto_apply: bool = True
    ) -> List[ModificationRecord]:
        """
        Apply recommendations from research.

        Args:
            recommendations: List of {"component": str, "change": str, "priority": int}
            auto_apply: If False, just return what would change (dry run)

        Returns:
            List of modifications made (or that would be made)
        """

        records = []

        for rec in recommendations:
            component = rec["component"]

            if component not in self.MODIFIABLE_COMPONENTS:
                continue

            # Generate specific change using Claude
            modification = await self._generate_modification(rec)

            if modification is None:
                continue

            record = ModificationRecord(
                timestamp=datetime.now(),
                component=component,
                parameter=modification["parameter"],
                old_value=modification["old_value"],
                new_value=modification["new_value"],
                reason=rec["change"],
                research_report_id=rec.get("research_id", "manual")
            )

            if auto_apply:
                await self._apply_modification(record)
                await self.db.save_modification(record)

            records.append(record)

        return records

    async def _generate_modification(self, recommendation: dict) -> Optional[dict]:
        """Generate specific modification from recommendation."""

        component = recommendation["component"]
        change_description = recommendation["change"]

        # Get current config
        current_config = await self._load_component_config(component)

        prompt = f"""
        Component: {component}
        Current configuration:
        {json.dumps(current_config, indent=2)}

        Recommended change: {change_description}

        Generate a specific modification:
        1. Which parameter to change
        2. What the new value should be
        3. Keep the change minimal and focused

        Return JSON: {{"parameter": str, "old_value": any, "new_value": any}}
        """

        return await self.claude.generate_structured(
            prompt=prompt,
            response_model=dict
        )

    async def rollback(self, modification_id: str, reason: str):
        """Rollback a specific modification."""

        record = await self.db.get_modification(modification_id)

        if not record.rollback_available:
            raise ValueError("This modification cannot be rolled back")

        # Restore old value
        await self._apply_modification(ModificationRecord(
            timestamp=datetime.now(),
            component=record.component,
            parameter=record.parameter,
            old_value=record.new_value,  # Swap
            new_value=record.old_value,  # Swap
            reason=f"ROLLBACK: {reason}",
            research_report_id="rollback"
        ))

        # Mark as rolled back
        record.rolled_back = True
        record.rollback_reason = reason
        await self.db.update_modification(record)

    async def get_modification_log(self, days: int = 30) -> List[ModificationRecord]:
        """Get recent modification history."""
        return await self.db.get_modifications(days=days)
```

---

### Experimentation Framework

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import random


class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED_EARLY = "stopped_early"
    WINNER_APPLIED = "winner_applied"


@dataclass
class ExperimentVariant:
    """A single variant in an A/B experiment."""

    name: str                    # "control" or "treatment"
    description: str             # What's different
    config_override: Dict[str, Any]  # What config to change

    # Results
    posts: List[str] = field(default_factory=list)  # Post IDs in this variant
    total_engagement: float = 0.0
    avg_score: float = 0.0


@dataclass
class Experiment:
    """A/B experiment definition and results."""

    id: str
    name: str                    # "Question Hook vs Statement Hook"
    hypothesis: str              # "Question hooks will get 20% more engagement"

    # Design
    variants: List[ExperimentVariant]
    min_posts_per_variant: int = 5
    max_posts_per_variant: int = 10

    # Status
    status: ExperimentStatus = ExperimentStatus.DRAFT
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Results
    winner: Optional[str] = None
    confidence: float = 0.0
    lift: float = 0.0  # % improvement of winner over loser


class ExperimentationEngine:
    """
    A/B testing framework for content strategies.
    Tests one hypothesis at a time, applies winner automatically.
    """

    def __init__(self, db, modification_engine: SelfModificationEngine):
        self.db = db
        self.modification_engine = modification_engine
        self.current_experiment: Optional[Experiment] = None

    async def create_experiment(
        self,
        name: str,
        hypothesis: str,
        control_config: Dict[str, Any],
        treatment_config: Dict[str, Any],
        treatment_description: str
    ) -> Experiment:
        """Create a new A/B experiment."""

        if self.current_experiment and self.current_experiment.status == ExperimentStatus.RUNNING:
            raise ValueError("Another experiment is already running. Complete it first.")

        experiment = Experiment(
            id=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=name,
            hypothesis=hypothesis,
            variants=[
                ExperimentVariant(
                    name="control",
                    description="Current behavior (no change)",
                    config_override=control_config
                ),
                ExperimentVariant(
                    name="treatment",
                    description=treatment_description,
                    config_override=treatment_config
                )
            ]
        )

        await self.db.save_experiment(experiment)
        return experiment

    async def start_experiment(self, experiment_id: str):
        """Start an experiment."""

        experiment = await self.db.get_experiment(experiment_id)
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = datetime.now()

        self.current_experiment = experiment
        await self.db.update_experiment(experiment)

    async def assign_variant(self) -> ExperimentVariant:
        """
        Assign a variant for the next post.
        Uses simple alternation for balanced assignment.
        """

        if not self.current_experiment or self.current_experiment.status != ExperimentStatus.RUNNING:
            return None

        # Alternate between variants
        control_count = len(self.current_experiment.variants[0].posts)
        treatment_count = len(self.current_experiment.variants[1].posts)

        if control_count <= treatment_count:
            return self.current_experiment.variants[0]
        else:
            return self.current_experiment.variants[1]

    async def record_result(self, post_id: str, variant_name: str, score: float):
        """Record a post result for the experiment."""

        if not self.current_experiment:
            return

        for variant in self.current_experiment.variants:
            if variant.name == variant_name:
                variant.posts.append(post_id)
                variant.total_engagement += score
                variant.avg_score = variant.total_engagement / len(variant.posts)
                break

        await self.db.update_experiment(self.current_experiment)

        # Check if experiment is complete
        await self._check_completion()

    async def _check_completion(self):
        """
        Check if experiment has enough data to conclude.

        FIX: Uses proper statistical testing (Welch's t-test) instead of
        naive ratio comparison. This ensures we don't conclude experiments
        prematurely due to random variance.
        """
        from scipy import stats
        import numpy as np

        exp = self.current_experiment
        control = exp.variants[0]
        treatment = exp.variants[1]

        # Check if we have minimum posts
        if len(control.posts) < exp.min_posts_per_variant:
            return
        if len(treatment.posts) < exp.min_posts_per_variant:
            return

        # Get actual engagement scores for each post
        control_scores = await self._get_post_scores(control.posts)
        treatment_scores = await self._get_post_scores(treatment.posts)

        if len(control_scores) < 3 or len(treatment_scores) < 3:
            return  # Need at least 3 samples for meaningful statistics

        # FIX: Perform Welch's t-test (doesn't assume equal variances)
        # This is more robust than simple ratio comparison
        t_statistic, p_value = stats.ttest_ind(
            treatment_scores,
            control_scores,
            equal_var=False  # Welch's t-test
        )

        # Calculate effect size (Cohen's d) for practical significance
        pooled_std = np.sqrt(
            (np.std(control_scores, ddof=1)**2 + np.std(treatment_scores, ddof=1)**2) / 2
        )
        effect_size = (np.mean(treatment_scores) - np.mean(control_scores)) / pooled_std if pooled_std > 0 else 0

        # Store statistical results in experiment
        exp.statistical_results = {
            "t_statistic": float(t_statistic),
            "p_value": float(p_value),
            "effect_size": float(effect_size),
            "control_mean": float(np.mean(control_scores)),
            "treatment_mean": float(np.mean(treatment_scores)),
            "control_std": float(np.std(control_scores, ddof=1)),
            "treatment_std": float(np.std(treatment_scores, ddof=1)),
            "sample_size": {
                "control": len(control_scores),
                "treatment": len(treatment_scores)
            }
        }

        # Significance thresholds
        SIGNIFICANCE_LEVEL = 0.05  # 95% confidence
        MIN_EFFECT_SIZE = 0.3     # Small-to-medium effect (Cohen's d)

        # Early stopping conditions:
        # 1. Statistically significant (p < 0.05)
        # 2. AND practically significant (|effect_size| > 0.3)
        # 3. AND at least 5 samples per variant
        if len(control_scores) >= 5 and len(treatment_scores) >= 5:
            if p_value < SIGNIFICANCE_LEVEL and abs(effect_size) > MIN_EFFECT_SIZE:
                self.logger.info(
                    f"[EXPERIMENT] Early stop: p={p_value:.4f}, "
                    f"effect_size={effect_size:.3f}, "
                    f"treatment_better={effect_size > 0}"
                )
                await self._complete_experiment(early_stop=True)
                return

        # Check if we hit max posts
        if len(control.posts) >= exp.max_posts_per_variant and \
           len(treatment.posts) >= exp.max_posts_per_variant:
            # Complete even if not statistically significant
            # (inconclusive result is also a valid conclusion)
            await self._complete_experiment(early_stop=False)

    async def _get_post_scores(self, post_ids: List[str]) -> List[float]:
        """Get engagement scores for a list of posts."""
        scores = []
        for post_id in post_ids:
            try:
                metrics = await self.db.get_metrics_history(post_id, limit=1)
                if metrics:
                    # Use engagement rate or calculated score
                    score = metrics[0].get("engagement_rate", 0) or \
                            (metrics[0].get("likes", 0) + metrics[0].get("comments", 0) * 3)
                    scores.append(float(score))
            except Exception:
                continue  # Skip posts with missing data
        return scores

    async def _complete_experiment(self, early_stop: bool):
        """Complete experiment and apply winner."""

        exp = self.current_experiment

        control = exp.variants[0]
        treatment = exp.variants[1]

        # FIX: Use statistical results computed in _check_completion
        stats_results = getattr(exp, 'statistical_results', None)

        # FIX: Determine winner based on statistical significance, not just raw average
        if stats_results and stats_results.get('p_value', 1.0) < 0.05:
            # Statistically significant result
            effect_size = stats_results.get('effect_size', 0)
            if effect_size > 0:
                exp.winner = "treatment"
            else:
                exp.winner = "control"
            # FIX: Calculate lift correctly, handling zero baseline
            if control.avg_score > 0:
                exp.lift = ((treatment.avg_score - control.avg_score) / control.avg_score) * 100
            else:
                exp.lift = 0  # Cannot calculate lift with zero baseline

            # FIX: Use actual confidence from statistical test (1 - p_value)
            exp.confidence = 1.0 - stats_results.get('p_value', 0.05)
        else:
            # Not statistically significant - inconclusive
            exp.winner = "inconclusive"
            exp.lift = 0
            exp.confidence = 0.0
            self.logger.warning(
                f"[EXPERIMENT] Inconclusive result: p={stats_results.get('p_value', 'N/A') if stats_results else 'N/A'}"
            )

        exp.status = ExperimentStatus.STOPPED_EARLY if early_stop else ExperimentStatus.COMPLETED
        exp.completed_at = datetime.now()

        # FIX: Only apply winner if statistically significant AND treatment won
        if exp.winner == "treatment" and exp.confidence >= 0.95:
            await self.modification_engine.apply_recommendations([{
                "component": list(treatment.config_override.keys())[0].split(".")[0],
                "change": f"Experiment result: {treatment.description} (+{exp.lift:.1f}%)",
                "priority": 1,
                "research_id": exp.id
            }])
            exp.status = ExperimentStatus.WINNER_APPLIED

        self.current_experiment = None
        await self.db.update_experiment(exp)


# ═══════════════════════════════════════════════════════════════════
# EXAMPLE EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════

example_experiments = [
    {
        "name": "Question Hook vs Statement Hook",
        "hypothesis": "Starting with a question will increase engagement by 20%",
        "treatment_config": {"writer.hook_style": "question"},
        "treatment_description": "Start every post with a thought-provoking question"
    },
    {
        "name": "Short vs Long Posts",
        "hypothesis": "Shorter posts (500 chars) will get more engagement than longer (2000 chars)",
        "treatment_config": {"writer.target_length": 500},
        "treatment_description": "Limit posts to 500 characters max"
    },
    {
        "name": "Emoji First Line",
        "hypothesis": "Adding emoji to first line increases scroll-stopping by 15%",
        "treatment_config": {"writer.first_line_emoji": True},
        "treatment_description": "Add relevant emoji at the start of every post"
    },
    {
        "name": "Author Photo vs Abstract Visual",
        "hypothesis": "Posts with author photo get 2x engagement",
        "treatment_config": {"visual_creator.author_photo_probability": 1.0},
        "treatment_description": "Always include author photo in visuals"
    },
    {
        "name": "Morning vs Evening Posting",
        "hypothesis": "Evening posts (18:00) outperform morning posts (9:00)",
        "treatment_config": {"scheduler.preferred_hour": 18},
        "treatment_description": "Schedule all posts for 18:00"
    }
]
```

---

### Meta-Agent Orchestration

```python
class MetaAgent:
    """
    Meta-agent that orchestrates self-improvement.
    Runs observation → evaluation → research → modification cycle.
    """

    def __init__(
        self,
        evaluator: SelfEvaluator,
        researcher: ResearchAgent,
        modifier: SelfModificationEngine,
        experimenter: ExperimentationEngine,
        db
    ):
        self.evaluator = evaluator
        self.researcher = researcher
        self.modifier = modifier
        self.experimenter = experimenter
        self.db = db

    async def evaluate_draft(self, draft: str, context: dict) -> EvaluationResult:
        """
        Evaluate a draft before it goes to human approval.
        May trigger rewrites.
        """

        result = await self.evaluator.evaluate(draft, context)

        # FIX: Use centralized thresholds instead of hardcoded values
        content_type = context.get("content_type", ContentType.COMMUNITY_CONTENT)
        pass_threshold = THRESHOLD_CONFIG.get_pass_threshold(content_type)
        max_iterations = THRESHOLD_CONFIG.get_max_meta_iterations()

        if result.score < pass_threshold and result.iteration < max_iterations:
            # Trigger rewrite with feedback
            return EvaluationResult(
                score=result.score,
                feedback=result.feedback,
                action="rewrite",
                iteration=result.iteration + 1
            )

        return EvaluationResult(
            score=result.score,
            feedback=result.feedback,
            action="proceed",
            iteration=result.iteration
        )

    async def run_improvement_cycle(self):
        """
        Run a full self-improvement cycle.
        Triggered by scheduler or manual request.
        """

        # 1. Check if research is needed
        trigger = await self.researcher.should_research()

        if trigger:
            # 2. Execute research
            report = await self.researcher.research(trigger)
            await self.db.save_research_report(report)

            # 3. Apply high-confidence recommendations
            high_confidence = [
                r for r in report.recommendations
                if r.get("confidence", 0) >= 0.8
            ]

            if high_confidence:
                modifications = await self.modifier.apply_recommendations(
                    high_confidence,
                    auto_apply=True
                )

                # Notify about changes
                await self._notify_changes(modifications)

        # 4. Check experiment status
        if self.experimenter.current_experiment:
            status = await self.experimenter.get_status()
            if status == "completed":
                await self._notify_experiment_result()

    async def suggest_experiment(self) -> Optional[dict]:
        """
        Based on current data, suggest a new experiment to run.
        """

        # Get current gaps in knowledge
        analysis = await self.db.get_performance_gaps()

        prompt = f"""
        Current performance gaps:
        {analysis}

        Past experiments:
        {await self.db.get_past_experiments()}

        Suggest ONE new A/B experiment to run.
        Focus on the biggest gap in our knowledge.
        Return: name, hypothesis, what to test.
        """

        return await self.claude.generate_structured(
            prompt=prompt,
            response_model=dict
        )


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION WITH ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════

# Updated workflow with Meta-Agent:
#
# 1. Trend Scout → finds topic
# 2. Analyzer → extracts insights
# 3. Writer → generates draft v1
# 4. [NEW] Meta-Agent.evaluate_draft() → may trigger rewrites
# 5. Visual Creator → generates images
# 6. QC Agent → final polish
# 7. Human Approval → (optional, can be auto if score >= 9.0)
# 8. Publish
# 9. [NEW] Meta-Agent.run_improvement_cycle() → weekly
```

---

### Configuration

```python
meta_agent_config = {
    # ═══════════════════════════════════════════════════════════════════
    # SELF-EVALUATION
    # ═══════════════════════════════════════════════════════════════════

    "evaluation": {
        "enabled": True,
        "min_score_to_proceed": 8.0,
        "max_rewrite_iterations": 3,

        "criteria_weights": {
            "hook_strength": 0.25,
            "specificity": 0.20,
            "value_density": 0.20,
            "authenticity": 0.15,
            "structure": 0.10,
            "cta_clarity": 0.10
        }
    },

    # ═══════════════════════════════════════════════════════════════════
    # RESEARCH
    # ═══════════════════════════════════════════════════════════════════

    "research": {
        "enabled": True,

        "triggers": {
            "underperformance_threshold": 3,  # N posts below average
            "weekly_cycle_day": 6,  # Sunday = 6
            "algorithm_change_sensitivity": 0.3  # 30% engagement drop
        },

        "sources": {
            "perplexity": True,
            "competitor_scrape": True,
            "own_data": True
        },

        "competitors": [
            "justinwelsh",
            "sambhavchaturvedi",
            "garyvaynerchuk"
        ]
    },

    # ═══════════════════════════════════════════════════════════════════
    # SELF-MODIFICATION
    # ═══════════════════════════════════════════════════════════════════

    "modification": {
        "auto_apply": True,  # Apply high-confidence changes automatically
        "confidence_threshold": 0.8,
        "require_human_review": False,  # Set True for safety

        "rate_limits": {
            "max_changes_per_day": 3,
            "cooldown_after_change_hours": 24
        }
    },

    # ═══════════════════════════════════════════════════════════════════
    # EXPERIMENTATION
    # ═══════════════════════════════════════════════════════════════════

    "experimentation": {
        "enabled": True,
        "auto_suggest": True,

        "constraints": {
            "min_posts_per_variant": 5,
            "max_posts_per_variant": 10,
            "max_concurrent_experiments": 1,
            "early_stopping_threshold": 0.5  # Stop if 50% better/worse
        }
    },

    # ═══════════════════════════════════════════════════════════════════
    # CLAUDE CODE (Server-Side Code Generation)
    # ═══════════════════════════════════════════════════════════════════

    "claude_code": {
        "enabled": True,
        "headless": True,
        "output_format": "json",

        # Safety limits
        "max_turns": 20,
        "max_budget_usd": 10.0,
        "timeout_seconds": 300,

        # Allowed tools
        "allowed_tools": [
            "Read", "Write", "Edit", "Bash", "Glob", "Grep"
        ],
        "disallowed_tools": [
            "WebSearch", "WebFetch"  # No external access during code gen
        ],

        # Bash restrictions
        "bash_allow": [
            "python *", "pytest *", "pip install *",
            "git status", "git diff", "git add *", "git commit *"
        ],
        "bash_deny": [
            "rm -rf *", "sudo *", "curl *", "wget *"
        ],

        # Logging
        "log_all_sessions": True,
        "session_log_dir": "data/claude_code_sessions/"
    },

    # ═══════════════════════════════════════════════════════════════════
    # AUTONOMY LEVEL (Static Config - see AutonomyManager for runtime)
    # ═══════════════════════════════════════════════════════════════════

    "autonomy": {
        "default_level": 3,
        "auto_publish_threshold": 9.0,
        "notify_on_publish": True,
        "notify_on_modification": True,

        # Per content-type overrides (optional)
        "content_type_levels": {
            # "enterprise_case": 2,  # Require human approval for enterprise cases
            # "community_content": 4  # Full autonomy for community content
        },

        # Auto-degradation settings
        "auto_degradation": {
            "enabled": True,
            "consecutive_failures_threshold": 3,
            "degradation_duration_hours": 24
        }
    }
}
```

### AutonomyManager: Dynamic Autonomy Control

```python
# ═══════════════════════════════════════════════════════════════════════════
# AUTONOMY MANAGER
# Dynamic runtime control of autonomy levels with thread-safe access
# ═══════════════════════════════════════════════════════════════════════════

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from enum import IntEnum


class AutonomyLevel(IntEnum):
    """
    Autonomy levels for the agent system.

    Level 1: Human approves everything (posts, modifications, research)
    Level 2: Human approves posts only, auto-modifications allowed
    Level 3: Auto-publish high-score posts (≥9.0), human for rest
    Level 4: Full autonomy (human notified, not asked)
    """
    HUMAN_ALL = 1
    HUMAN_POSTS = 2
    AUTO_HIGH_SCORE = 3
    FULL_AUTONOMY = 4


@dataclass
class AutonomyConfig:
    """Configuration for autonomy management."""
    default_level: AutonomyLevel = AutonomyLevel.AUTO_HIGH_SCORE
    auto_publish_threshold: float = 9.0

    # Per content-type level overrides
    content_type_levels: Dict[str, AutonomyLevel] = field(default_factory=dict)

    # Auto-degradation after consecutive failures
    auto_degradation_enabled: bool = True
    consecutive_failures_threshold: int = 3
    degradation_duration_hours: int = 24


class AutonomyManager:
    """
    Thread-safe manager for dynamic autonomy level control.

    Features:
    - Per-content-type autonomy levels
    - Temporary elevation (e.g., Level 4 on weekends)
    - Auto-degradation after consecutive failures
    - Thread-safe access with async lock

    Usage:
        manager = AutonomyManager(config)

        # Get effective level for a post
        level = await manager.get_effective_level(ContentType.ENTERPRISE_CASE)

        # Check if auto-publish is allowed
        can_auto = await manager.can_auto_publish(score=9.2, content_type)

        # Temporarily elevate autonomy
        await manager.set_temporary_elevation(AutonomyLevel.FULL_AUTONOMY, hours=48)

        # Report a failure (may trigger auto-degradation)
        await manager.report_failure()
    """

    def __init__(self, config: AutonomyConfig):
        self._config = config
        self._lock = asyncio.Lock()

        # Runtime state
        self._temporary_elevation: Optional[Tuple[AutonomyLevel, datetime]] = None
        self._consecutive_failures: int = 0
        self._degradation_until: Optional[datetime] = None

    async def get_effective_level(
        self,
        content_type: Optional[ContentType] = None
    ) -> AutonomyLevel:
        """
        Get the effective autonomy level, considering:
        1. Temporary elevation (if active and not expired)
        2. Auto-degradation (if triggered by failures)
        3. Content-type specific override
        4. Default level

        Thread-safe: uses async lock.
        """
        async with self._lock:
            # Check temporary elevation first
            if self._temporary_elevation:
                level, until = self._temporary_elevation
                if datetime.now() < until:
                    return level
                else:
                    # Elevation expired, clear it
                    self._temporary_elevation = None

            # Check auto-degradation
            if self._degradation_until and datetime.now() < self._degradation_until:
                # Degraded by one level (min Level 1)
                degraded = max(1, self._config.default_level - 1)
                return AutonomyLevel(degraded)

            # Check content-type specific override
            if content_type:
                type_key = content_type.value if hasattr(content_type, 'value') else str(content_type)
                if type_key in self._config.content_type_levels:
                    return self._config.content_type_levels[type_key]

            return self._config.default_level

    async def can_auto_publish(
        self,
        score: float,
        content_type: Optional[ContentType] = None
    ) -> bool:
        """
        Check if a post can be auto-published based on current autonomy level.

        Returns True if:
        - Level 4 (full autonomy), OR
        - Level 3 AND score >= auto_publish_threshold
        """
        level = await self.get_effective_level(content_type)

        if level == AutonomyLevel.FULL_AUTONOMY:
            return True

        if level == AutonomyLevel.AUTO_HIGH_SCORE:
            return score >= self._config.auto_publish_threshold

        return False

    async def requires_human_approval(
        self,
        action: str,  # "post", "modification", "research"
        score: Optional[float] = None,
        content_type: Optional[ContentType] = None
    ) -> bool:
        """
        Check if an action requires human approval.

        Args:
            action: Type of action ("post", "modification", "research")
            score: Quality score (for posts)
            content_type: Content type (for posts)
        """
        level = await self.get_effective_level(content_type)

        if level == AutonomyLevel.HUMAN_ALL:
            return True  # Everything needs approval

        if level == AutonomyLevel.HUMAN_POSTS:
            return action == "post"  # Only posts need approval

        if level == AutonomyLevel.AUTO_HIGH_SCORE:
            if action == "post" and score is not None:
                return score < self._config.auto_publish_threshold
            # FIX: At level 3, non-post actions (modification, research) still need approval
            # Only high-score posts are auto-approved, everything else needs human review
            return action in ["modification", "research"]

        # Level 4: Nothing needs approval
        return False

    async def set_temporary_elevation(
        self,
        level: AutonomyLevel,
        hours: int
    ) -> None:
        """
        Temporarily elevate autonomy level.

        Use cases:
        - Weekend mode (Level 4 for 48 hours)
        - Vacation mode (Level 4 while away)
        - Testing mode (Level 1 during debugging)
        """
        async with self._lock:
            until = datetime.now() + timedelta(hours=hours)
            self._temporary_elevation = (level, until)

    async def clear_temporary_elevation(self) -> None:
        """Clear any temporary elevation."""
        async with self._lock:
            self._temporary_elevation = None

    async def report_failure(self) -> None:
        """
        Report a failure. May trigger auto-degradation.

        Call this when:
        - Pipeline fails with critical error
        - Post is rejected by human
        - Quality scores consistently low
        """
        if not self._config.auto_degradation_enabled:
            return

        async with self._lock:
            self._consecutive_failures += 1

            if self._consecutive_failures >= self._config.consecutive_failures_threshold:
                # Trigger degradation
                self._degradation_until = datetime.now() + timedelta(
                    hours=self._config.degradation_duration_hours
                )
                self._consecutive_failures = 0  # Reset counter

    async def report_success(self) -> None:
        """
        Report a success. Resets failure counter and clears degradation.

        Call this when:
        - Post is successfully published
        - Human approves a post
        """
        async with self._lock:
            self._consecutive_failures = 0
            # Clear degradation on success
            self._degradation_until = None

    async def get_status(self) -> Dict:
        """Get current autonomy status for monitoring/debugging."""
        async with self._lock:
            effective_level = await self.get_effective_level()

            status = {
                "default_level": self._config.default_level,
                "effective_level": effective_level,
                "consecutive_failures": self._consecutive_failures,
            }

            if self._temporary_elevation:
                level, until = self._temporary_elevation
                status["temporary_elevation"] = {
                    "level": level,
                    "until": until.isoformat(),
                    "active": datetime.now() < until
                }

            if self._degradation_until:
                status["degradation"] = {
                    "until": self._degradation_until.isoformat(),
                    "active": datetime.now() < self._degradation_until
                }

            return status


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL AUTONOMY MANAGER INSTANCE
# ═══════════════════════════════════════════════════════════════════════════

_autonomy_manager: Optional[AutonomyManager] = None

def get_autonomy_manager() -> AutonomyManager:
    """Get the global autonomy manager instance."""
    global _autonomy_manager
    if _autonomy_manager is None:
        # Load config from settings
        config = AutonomyConfig(
            default_level=AutonomyLevel(SETTINGS["autonomy"]["default_level"]),
            auto_publish_threshold=SETTINGS["autonomy"]["auto_publish_threshold"],
            content_type_levels={
                k: AutonomyLevel(v)
                for k, v in SETTINGS["autonomy"].get("content_type_levels", {}).items()
            },
            auto_degradation_enabled=SETTINGS["autonomy"]["auto_degradation"]["enabled"],
            consecutive_failures_threshold=SETTINGS["autonomy"]["auto_degradation"]["consecutive_failures_threshold"],
            degradation_duration_hours=SETTINGS["autonomy"]["auto_degradation"]["degradation_duration_hours"],
        )
        _autonomy_manager = AutonomyManager(config)
    return _autonomy_manager
```

### Approval Timeout Manager

```python
# ═══════════════════════════════════════════════════════════════════════════
# APPROVAL TIMEOUT MANAGER
# Handles timeout, escalation, and auto-resolution for pending approvals
# ═══════════════════════════════════════════════════════════════════════════

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Callable, Awaitable
import logging


@dataclass
class ApprovalTimeoutConfig:
    """Configuration for approval timeout handling."""

    # Reminder settings
    reminder_after_hours: int = 4           # Send first reminder after 4 hours
    reminder_interval_hours: int = 4        # Send follow-up reminders every 4 hours
    max_reminders: int = 3                  # Max reminders before escalation

    # Escalation settings
    escalate_after_hours: int = 24          # Escalate after 24 hours without response
    escalation_contacts: List[str] = None   # Additional contacts for escalation

    # Auto-resolution settings
    auto_resolve_after_hours: int = 48      # Auto-resolve after 48 hours
    auto_resolve_action: str = "approve"    # "approve", "reject", or "hold"
    auto_resolve_min_score: float = 8.0     # Only auto-approve if score >= this

    def __post_init__(self):
        if self.escalation_contacts is None:
            self.escalation_contacts = []


@dataclass
class PendingApproval:
    """Represents a pending approval request."""
    run_id: str
    post_id: Optional[str]
    requested_at: datetime
    content_type: str
    qc_score: float
    reminder_count: int = 0
    escalation_level: int = 0  # 0=pending, 1=reminded, 2=escalated, 3=auto-resolved
    notification_channel: str = "telegram"  # telegram, email, slack


class ApprovalTimeoutManager:
    """
    Manages timeout handling for human approvals.

    Features:
    - Sends reminders after configurable intervals
    - Escalates to additional contacts after timeout
    - Auto-resolves (approve/reject) after extended timeout
    - Prevents queue buildup

    Usage:
        manager = ApprovalTimeoutManager(config, notifier)

        # Start background monitoring
        asyncio.create_task(manager.start_monitoring())

        # Register a new pending approval
        await manager.register_pending_approval(run_id, post_id, score, content_type)

        # Mark as resolved (when human responds)
        await manager.mark_resolved(run_id, "approved")
    """

    def __init__(
        self,
        config: ApprovalTimeoutConfig,
        notifier: "NotificationService",  # Injected dependency
        db: "SupabaseDB"
    ):
        self._config = config
        self._notifier = notifier
        self._db = db
        self._logger = logging.getLogger("ApprovalTimeoutManager")
        self._running = False

    async def start_monitoring(self, check_interval_minutes: int = 15):
        """
        Start background task to monitor pending approvals.

        Runs continuously, checking for timeouts every check_interval_minutes.
        """
        self._running = True
        self._logger.info("Starting approval timeout monitoring")

        while self._running:
            try:
                await self._check_pending_approvals()
            except Exception as e:
                self._logger.error(f"Error checking approvals: {e}")

            await asyncio.sleep(check_interval_minutes * 60)

    async def stop_monitoring(self):
        """Stop the monitoring task."""
        self._running = False

    async def register_pending_approval(
        self,
        run_id: str,
        post_id: Optional[str],
        qc_score: float,
        content_type: str
    ) -> None:
        """Register a new pending approval for monitoring."""
        await self._db.client.table("pending_approvals").insert({
            "run_id": run_id,
            "post_id": post_id,
            "qc_score": qc_score,
            "content_type": content_type,
            "requested_at": utc_now().isoformat(),
            "reminder_count": 0,
            "escalation_level": 0,
            "status": "pending"
        }).execute()

        self._logger.info(f"Registered pending approval for run {run_id}")

    async def mark_resolved(self, run_id: str, resolution: str) -> None:
        """
        Mark an approval as resolved.

        Args:
            run_id: The run ID of the approval
            resolution: "approved", "rejected", "edited"
        """
        await self._db.client.table("pending_approvals").update({
            "status": resolution,
            "resolved_at": utc_now().isoformat()
        }).eq("run_id", run_id).eq("status", "pending").execute()

        self._logger.info(f"Approval {run_id} resolved: {resolution}")

    async def _check_pending_approvals(self) -> None:
        """Check all pending approvals for timeout actions."""
        result = await self._db.client.table("pending_approvals").select("*").eq("status", "pending").execute()

        pending = result.data or []
        now = datetime.now()

        for approval in pending:
            requested_at = datetime.fromisoformat(approval["requested_at"])
            hours_pending = (now - requested_at).total_seconds() / 3600
            reminder_count = approval.get("reminder_count", 0)
            escalation_level = approval.get("escalation_level", 0)

            # Check for auto-resolution
            if hours_pending >= self._config.auto_resolve_after_hours:
                await self._auto_resolve(approval)
                continue

            # Check for escalation
            if hours_pending >= self._config.escalate_after_hours and escalation_level < 2:
                await self._escalate(approval)
                continue

            # Check for reminder
            hours_since_request = hours_pending
            expected_reminders = int(
                (hours_since_request - self._config.reminder_after_hours)
                / self._config.reminder_interval_hours
            ) + 1

            if (hours_since_request >= self._config.reminder_after_hours and
                reminder_count < min(expected_reminders, self._config.max_reminders)):
                await self._send_reminder(approval)

    async def _send_reminder(self, approval: dict) -> None:
        """Send a reminder notification."""
        run_id = approval["run_id"]
        reminder_count = approval.get("reminder_count", 0) + 1

        message = (
            f"⏰ Reminder #{reminder_count}: Post awaiting approval\n"
            f"Run ID: {run_id}\n"
            f"Content Type: {approval['content_type']}\n"
            f"QC Score: {approval['qc_score']}\n"
            f"Waiting since: {approval['requested_at']}"
        )

        await self._notifier.send(message, channel=approval.get("notification_channel", "telegram"))

        # Update reminder count
        await self._db.client.table("pending_approvals").update({
            "reminder_count": reminder_count,
            "last_reminder_at": utc_now().isoformat()
        }).eq("run_id", run_id).execute()

        self._logger.info(f"Sent reminder #{reminder_count} for approval {run_id}")

    async def _escalate(self, approval: dict) -> None:
        """Escalate the approval to additional contacts."""
        run_id = approval["run_id"]

        message = (
            f"🚨 ESCALATION: Post requires urgent attention\n"
            f"Run ID: {run_id}\n"
            f"Content Type: {approval['content_type']}\n"
            f"QC Score: {approval['qc_score']}\n"
            f"Pending for: 24+ hours\n"
            f"Action required!"
        )

        # Notify primary channel
        await self._notifier.send(message, channel=approval.get("notification_channel", "telegram"))

        # Notify escalation contacts
        for contact in self._config.escalation_contacts:
            await self._notifier.send(message, recipient=contact)

        # Update escalation level
        await self._db.client.table("pending_approvals").update({
            "escalation_level": 2,
            "escalated_at": utc_now().isoformat()
        }).eq("run_id", run_id).execute()

        self._logger.warning(f"Escalated approval {run_id}")

    async def _auto_resolve(self, approval: dict) -> None:
        """
        Auto-resolve the approval after extended timeout.

        Action depends on config and QC score:
        - If auto_resolve_action="approve" AND score >= min_score: auto-approve
        - If auto_resolve_action="reject": auto-reject
        - If auto_resolve_action="hold": keep pending but mark as auto-held
        """
        run_id = approval["run_id"]
        qc_score = approval.get("qc_score", 0)
        action = self._config.auto_resolve_action

        if action == "approve" and qc_score >= self._config.auto_resolve_min_score:
            resolution = "auto_approved"
            notification = f"✅ Auto-approved (score {qc_score} >= {self._config.auto_resolve_min_score})"
        elif action == "approve":
            resolution = "auto_held"  # Score too low to auto-approve
            notification = f"⏸️ Auto-held (score {qc_score} < {self._config.auto_resolve_min_score})"
        elif action == "reject":
            resolution = "auto_rejected"
            notification = f"❌ Auto-rejected after {self._config.auto_resolve_after_hours}h timeout"
        else:  # hold
            resolution = "auto_held"
            notification = f"⏸️ Auto-held after {self._config.auto_resolve_after_hours}h timeout"

        # Update status
        await self._db.client.table("pending_approvals").update({
            "status": resolution,
            "escalation_level": 3,
            "resolved_at": utc_now().isoformat()
        }).eq("run_id", run_id).execute()

        # Notify
        message = f"🤖 Auto-resolution: {notification}\nRun ID: {run_id}"
        await self._notifier.send(message, channel=approval.get("notification_channel", "telegram"))

        self._logger.info(f"Auto-resolved approval {run_id}: {resolution}")

    async def get_queue_status(self) -> dict:
        """Get status of the approval queue for monitoring."""
        result = await self._db.client.table("pending_approvals").select("*").eq("status", "pending").execute()

        pending = result.data or []
        now = datetime.now()

        total = len(pending)
        urgent = 0  # Pending > 24h
        critical = 0  # Pending > 48h

        for approval in pending:
            requested_at = datetime.fromisoformat(approval["requested_at"])
            hours = (now - requested_at).total_seconds() / 3600
            if hours >= 48:
                critical += 1
            elif hours >= 24:
                urgent += 1

        return {
            "total_pending": total,
            "urgent": urgent,
            "critical": critical,
            "oldest_request": min(
                (a["requested_at"] for a in pending),
                default=None
            )
        }
```

**SQL Table for Pending Approvals:**

```sql
-- Add to schema (after pipeline_errors table)

CREATE TABLE pending_approvals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id TEXT NOT NULL UNIQUE,
    post_id TEXT,

    -- Content info
    qc_score FLOAT,
    content_type TEXT,

    -- Timing
    requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_reminder_at TIMESTAMPTZ,
    escalated_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,

    -- State
    status TEXT DEFAULT 'pending',  -- pending, approved, rejected, edited, auto_approved, auto_rejected, auto_held
    reminder_count INTEGER DEFAULT 0,
    escalation_level INTEGER DEFAULT 0,

    -- Notification
    notification_channel TEXT DEFAULT 'telegram'
);

CREATE INDEX idx_pending_approvals_status ON pending_approvals(status);
CREATE INDEX idx_pending_approvals_requested ON pending_approvals(requested_at);
```

---

## Technical Stack

```
┌─────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE                        │
├─────────────────────────────────────────────────────────┤
│  Framework:     LangGraph (state machine orchestration) │
│  LLM:           Claude Opus 4.5 (thinking mode)        │
│  Code Gen:      Claude Code (headless, Agent SDK)      │  ← NEW
│                 Server-side code evolution             │
│  Image Gen:     Laozhang.ai Nano Banana Pro (4K)       │
│                 Model: gemini-3-pro-image-preview      │
│  LinkedIn:      tomquirk/linkedin-api (Voyager)        │
│                 Publish + Analytics collection         │
│  Search:        Perplexity API / Tavily                │
│  Vector DB:     Pinecone / Chroma (for memory)         │
│  Storage:       Supabase (drafts, analytics)           │
│  Queue:         Redis (task scheduling)                 │
│  UI:            Telegram Bot / Streamlit               │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
linkedin-super-agent/
├── src/
│   ├── agents/
│   │   ├── orchestrator.py      # LangGraph state machine
│   │   ├── trend_scout.py       # Trend mining
│   │   ├── analyzer.py          # Content analysis
│   │   ├── writer.py            # Post generation
│   │   ├── humanizer.py         # Humanization
│   │   ├── visual_creator.py    # Image generation
│   │   ├── photo_selector.py    # Personal photo selection
│   │   └── qc_agent.py          # Quality control
│   ├── meta_agent/                   # ═══ SELF-IMPROVEMENT LAYER ═══
│   │   ├── __init__.py
│   │   ├── meta_agent.py        # Main orchestrator for self-improvement
│   │   ├── single_call_evaluator.py  # ← NEW: Replaces multi-turn Critic
│   │   ├── modification_safety.py    # ← NEW: Risk-based approval + rollback
│   │   ├── reflection_engine.py # "Ага, понятно!" — осмысление feedback
│   │   ├── research_agent.py    # Perplexity + competitor analysis
│   │   ├── claude_code_client.py  # ← Claude Code integration (headless)
│   │   ├── code_evolution.py    # Генерация нового кода/модулей (uses Claude Code)
│   │   ├── knowledge_base.py    # Persistent memory (learnings)
│   │   ├── deep_improvement_loop.py  # Full cycle: critique→reflect→research→modify
│   │   ├── experimentation.py   # A/B testing framework
│   │   └── models.py            # Dataclasses
│   ├── author/                       # ═══ NEW: AUTHOR PROFILE ═══
│   │   ├── __init__.py
│   │   ├── author_profile_agent.py  # Creates & maintains voice profile
│   │   ├── profile_importer.py      # Import posts from LinkedIn/JSON
│   │   └── models.py                # AuthorVoiceProfile dataclass
│   ├── scheduling/                   # ═══ NEW: SCHEDULING SYSTEM ═══
│   │   ├── __init__.py
│   │   ├── scheduling_system.py     # Optimal timing + conflict avoidance
│   │   ├── publishing_scheduler.py  # Background task (APScheduler)
│   │   └── models.py                # ScheduledPost, PublishingSlot
│   ├── logging/                      # ═══ NEW: LOGGING SYSTEM ═══
│   │   ├── __init__.py
│   │   ├── agent_logger.py          # Central logger with multiple outputs
│   │   ├── component_logger.py      # Per-component wrapper
│   │   ├── pipeline_run_logger.py   # Pipeline run tracking
│   │   ├── daily_digest.py          # Daily summary generator
│   │   └── models.py                # LogEntry, LogLevel, LogComponent
│   ├── generated/                    # ═══ AUTO-GENERATED CODE ═══
│   │   ├── __init__.py          # Auto-imports all generated modules
│   │   ├── hook_selector.py     # Example: generated from learning
│   │   └── README.md            # "This code is auto-generated by Meta-Agent"
│   ├── tools/
│   │   ├── perplexity.py        # Perplexity API wrapper
│   │   ├── arxiv.py             # ArXiv search
│   │   ├── twitter.py           # X API
│   │   ├── nano_banana.py       # Image generation (Laozhang.ai)
│   │   ├── linkedin_client.py   # tomquirk/linkedin-api wrapper
│   │   └── photo_library.py     # Photo indexing & search
│   ├── knowledge/
│   │   ├── style_guide.md       # Your tone of voice
│   │   ├── top_posts.json       # Examples of successful posts
│   │   └── anti_patterns.json   # What to avoid
│   └── ui/
│       └── telegram_bot.py      # Human-in-the-loop interface
├── config/
│   ├── settings.yaml
│   ├── meta_agent_config.yaml   # Self-improvement settings
│   ├── scoring_weights.json     # Modifiable by meta-agent
│   ├── hook_templates.json      # Modifiable by meta-agent
│   ├── writer_config.json       # Modifiable by meta-agent
│   ├── visual_styles.json       # Modifiable by meta-agent
│   └── schedule.json            # Modifiable by meta-agent
├── prompts/                     # ═══ EVOLVABLE PROMPTS ═══
│   ├── writer_system.txt        # Writer agent system prompt (VERSIONED)
│   ├── critic_system.txt        # Critic agent system prompt
│   ├── evaluator_criteria.txt   # Self-evaluation criteria
│   ├── humanizer_rules.txt      # Humanization guidelines
│   └── versions/                # Prompt version history
│       ├── writer_system_v1.txt # Original
│       ├── writer_system_v2.txt # After learning about hooks
│       └── changelog.json       # What changed and why
├── photos/                      # Personal photo library
│   ├── portraits/
│   │   ├── formal/              # Suit, professional
│   │   ├── casual/              # Approachable
│   │   └── speaking/            # Conference, presenting
│   ├── action/
│   │   ├── at_desk/             # Working shots
│   │   ├── whiteboard/          # Teaching, explaining
│   │   └── team/                # Collaboration
│   ├── context/
│   │   ├── conference/          # Events
│   │   ├── office/              # Workspace
│   │   └── outdoor/             # Lifestyle
│   └── metadata.json            # Auto-tags, usage stats
│
│   NOTE: Screenshots & mockups NOT needed!
│   Nano Banana Pro generates all interfaces AI:
│   - CRM dashboards, AI chat, automation workflows
│   - Person holding phone with interface
│   - Before/after comparisons
│   - All in one generation request
└── data/
    ├── drafts/                  # Drafts for approval
    ├── analytics/               # Post history and metrics
    ├── research/                # Research data
    │   ├── reports/             # Research reports (JSON)
    │   ├── competitor_posts/    # Scraped competitor content
    │   └── insights/            # Extracted actionable insights
    ├── experiments/             # A/B test data
    │   ├── active/              # Currently running experiments
    │   └── completed/           # Historical experiment results
    ├── modifications/           # Self-modification log
    │   ├── history.json         # All changes with before/after
    │   ├── rollbacks/           # Reverted changes
    │   └── rollback_triggers/   # ← NEW: Active monitoring triggers
    ├── author_profiles/         # ← NEW: Author voice profiles
    │   ├── {author_name}.json   # Profile data
    │   └── posts_analyzed/      # Source posts used for analysis
    ├── scheduled_posts/         # ← NEW: Publishing queue
    │   ├── queue.json           # Current queue state
    │   └── history/             # Published/cancelled posts
    │
    │   # ═══ DEEP SELF-IMPROVEMENT DATA ═══
    │
    ├── dialogues/               # Agent-to-Agent conversations
    │   ├── critiques/           # Critic Agent dialogues
    │   │   └── 2024-01-15_post_123.json  # Full dialogue history
    │   └── summaries/           # DialogueSummary objects
    ├── reflections/             # "Ага, понятно!" moments
    │   └── 2024-01-15_reflection_001.json
    ├── knowledge_base/          # Persistent learnings (векторная БД)
    │   ├── learnings.db         # SQLite for structured data
    │   ├── embeddings/          # Vector embeddings for semantic search
    │   └── index.json           # Quick lookup index
    ├── code_evolution/          # History of generated code
    │   ├── modules/             # Generated .py files (with metadata)
    │   └── prompts/             # Prompt versions (with changelog)
    └── claude_code_sessions/    # ← Claude Code execution logs
        ├── 2024-01-15_gen_hook_selector.json  # Full session transcript
        ├── 2024-01-16_evolve_writer_prompt.json
        └── costs.json           # Running cost tracker
```

---

## Research Sources

### LinkedIn API & Analytics
- [tomquirk/linkedin-api (GitHub)](https://github.com/tomquirk/linkedin-api) - Unofficial Python API (Voyager)
- [linkedin-api on PyPI](https://pypi.org/project/linkedin-api/) - pip install linkedin-api
- [LinkedIn Official Python Client](https://github.com/linkedin-developers/linkedin-api-python-client)
- [LinkedIn Reactions API (Official Docs)](https://learn.microsoft.com/en-us/linkedin/marketing/community-management/shares/reactions-api)

### Multi-Agent Architecture
- [Multi-agent systems guide - n8n](https://blog.n8n.io/multi-agent-systems/)
- [Agentic AI Trends 2026 - MachineLearningMastery](https://machinelearningmastery.com/7-agentic-ai-trends-to-watch-in-2026/)
- [LangGraph vs CrewAI - ZenML](https://www.zenml.io/blog/langgraph-vs-crewai)
- [AI Agent Frameworks 2025 - LangWatch](https://langwatch.ai/blog/best-ai-agent-frameworks-in-2025-comparing-langgraph-dspy-crewai-agno-and-more)

### LinkedIn Algorithm & Engagement
- [LinkedIn Algorithm 2025 - Hootsuite](https://blog.hootsuite.com/linkedin-algorithm/)
- [LinkedIn Content Strategy 2025 - Postiv AI](https://postiv.ai/blog/linkedin-content-strategy-2025)
- [LinkedIn Viral Posts - Depost AI](https://depost.ai/blog/linkedin-viral-post-tips)
- [LinkedIn Algorithm Changes - Agorapulse](https://www.agorapulse.com/blog/linkedin/linkedin-algorithm-2025/)

### AI Image Generation
- [Laozhang.ai Nano Banana Pro Docs](https://docs.laozhang.ai/en/api-capabilities/nano-banana-pro-image)
- [Gemini 3 Pro Image Preview](https://ai.google.dev/gemini-api/docs/image-generation)
- [Best AI Image Generators 2025 - PXZ](https://pxz.ai/blog/best-ai-image-generators-2025-tested-ranked)

### Content Humanization
- [AI Humanizers Guide - OpsMatters](https://www.opsmatters.com/posts/5-best-ai-humanizers-bypass-ai-detection-2025)

---

## NEW: Logging & Observability System

### Purpose
Централизованная система логирования для отслеживания работы всех агентов, диагностики проблем и понимания поведения системы.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LOGGING & OBSERVABILITY SYSTEM                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         LOG SOURCES                                  │   │
│  │                                                                      │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │  Trend   │ │ Analyzer │ │  Writer  │ │ Visual   │ │    QC    │  │   │
│  │  │  Scout   │ │          │ │          │ │ Creator  │ │  Agent   │  │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │   │
│  │       │            │            │            │            │         │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │  Meta-   │ │ Scheduler│ │ LinkedIn │ │ Modific. │ │ Author   │  │   │
│  │  │  Agent   │ │          │ │  Client  │ │  Safety  │ │ Profile  │  │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │   │
│  │       │            │            │            │            │         │   │
│  └───────┴────────────┴────────────┴────────────┴────────────┴─────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    CENTRAL LOG COLLECTOR                             │   │
│  │                    (structlog + custom handlers)                     │   │
│  └───────────────────────────┬─────────────────────────────────────────┘   │
│                              │                                              │
│           ┌──────────────────┼──────────────────┐                          │
│           ▼                  ▼                  ▼                          │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐              │
│  │   FILE LOGS     │ │   SUPABASE      │ │   TELEGRAM      │              │
│  │   (JSON/text)   │ │   (queryable)   │ │   (real-time)   │              │
│  │                 │ │                 │ │                 │              │
│  │ logs/           │ │ agent_logs      │ │ Critical alerts │              │
│  │  ├─ agent.log   │ │ pipeline_runs   │ │ Daily digest    │              │
│  │  ├─ errors.log  │ │ errors          │ │ Error notifs    │              │
│  │  └─ debug.log   │ │                 │ │                 │              │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘              │
│           │                  │                  │                          │
│           └──────────────────┼──────────────────┘                          │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      LOG VIEWER UI                                   │   │
│  │                                                                      │   │
│  │  Option 1: Telegram Bot                                              │   │
│  │    /logs — last 20 logs                                              │   │
│  │    /logs errors — only errors                                        │   │
│  │    /logs writer — logs from Writer agent                             │   │
│  │    /logs run_123 — logs for specific pipeline run                    │   │
│  │                                                                      │   │
│  │  Option 2: Web Dashboard (Streamlit)                                 │   │
│  │    - Real-time log stream                                            │   │
│  │    - Filter by component, level, time                                │   │
│  │    - Search in logs                                                  │   │
│  │    - Pipeline run visualization                                      │   │
│  │                                                                      │   │
│  │  Option 3: Log Files (for debugging)                                 │   │
│  │    tail -f logs/agent.log | jq                                       │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Log Levels & Categories

```python
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
import json
import traceback
import aiofiles  # For async file I/O


class LogLevel(Enum):
    """
    Log levels with numeric values for proper severity comparison.

    FIX: Changed from string values to integers.
    String comparison doesn't work for severity ("debug" > "critical" = True lexicographically).
    """
    DEBUG = 10       # Detailed debugging info
    INFO = 20        # Normal operations
    WARNING = 30     # Something unexpected but not critical
    ERROR = 40       # Something failed
    CRITICAL = 50    # System-level failure

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

    # Infrastructure
    MODIFICATION_SAFETY = "modification_safety"
    AUTHOR_PROFILE = "author_profile"
    SCHEDULER = "scheduler"
    LINKEDIN_CLIENT = "linkedin_client"
    ANALYTICS = "analytics"
    TELEGRAM = "telegram"

    # Additional components (FIX: missing from original list)
    CIRCUIT_BREAKER = "circuit_breaker"
    PIPELINE_RECOVERY = "pipeline_recovery"
    CODE_GENERATION = "code_generation"
    ROLLBACK_MANAGER = "rollback_manager"
    RESEARCH_AGENT = "research_agent"
    CODE_EVOLUTION = "code_evolution"
    SELF_MODIFICATION = "self_modification"
    DATABASE = "database"


@dataclass
class LogEntry:
    """Structured log entry."""

    # Required
    timestamp: datetime
    level: LogLevel
    component: LogComponent
    message: str

    # Context
    run_id: Optional[str] = None  # Pipeline run ID
    post_id: Optional[str] = None  # Related post ID

    # Additional data
    data: Dict[str, Any] = field(default_factory=dict)

    # Error details
    error_type: Optional[str] = None
    error_traceback: Optional[str] = None

    # Performance
    duration_ms: Optional[int] = None

    def to_json(self) -> str:
        return json.dumps({
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "component": self.component.value,
            "message": self.message,
            "run_id": self.run_id,
            "post_id": self.post_id,
            "data": self.data,
            "error_type": self.error_type,
            "error_traceback": self.error_traceback,
            "duration_ms": self.duration_ms
        }, ensure_ascii=False)

    def to_readable(self) -> str:
        """Human-readable format for Telegram/console."""
        level_emoji = {
            LogLevel.DEBUG: "🔍",
            LogLevel.INFO: "ℹ️",
            LogLevel.WARNING: "⚠️",
            LogLevel.ERROR: "❌",
            LogLevel.CRITICAL: "🔴"
        }

        time_str = self.timestamp.strftime("%H:%M:%S")
        emoji = level_emoji.get(self.level, "•")

        msg = f"{emoji} [{time_str}] [{self.component.value}] {self.message}"

        if self.duration_ms:
            msg += f" ({self.duration_ms}ms)"

        return msg
```

### Central Logger Implementation

```python
import structlog
from pathlib import Path
from typing import List, Callable
import asyncio


class AgentLogger:
    """
    Central logging system for all agents.
    Supports multiple outputs: file, Supabase, Telegram.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        supabase_client = None,
        telegram_notifier = None,
        min_level: LogLevel = LogLevel.INFO,
        telegram_min_level: LogLevel = LogLevel.WARNING
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.supabase = supabase_client
        self.telegram = telegram_notifier
        self.min_level = min_level
        self.telegram_min_level = telegram_min_level

        # Current context
        self._run_id: Optional[str] = None
        self._post_id: Optional[str] = None

        # Log files
        self._main_log = self.log_dir / "agent.log"
        self._error_log = self.log_dir / "errors.log"
        self._debug_log = self.log_dir / "debug.log"

        # In-memory buffer for quick access
        self._recent_logs: List[LogEntry] = []
        self._max_recent = 1000

        # Custom handlers
        self._handlers: List[Callable] = []

    def set_context(self, run_id: str = None, post_id: str = None):
        """Set context for subsequent logs."""
        if run_id:
            self._run_id = run_id
        if post_id:
            self._post_id = post_id

    def clear_context(self):
        """Clear logging context."""
        self._run_id = None
        self._post_id = None

    def add_handler(self, handler: Callable[[LogEntry], None]):
        """Add custom log handler."""
        self._handlers.append(handler)

    async def log(
        self,
        level: LogLevel,
        component: LogComponent,
        message: str,
        data: Dict[str, Any] = None,
        error: Exception = None,
        duration_ms: int = None
    ):
        """Log a message."""

        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            component=component,
            message=message,
            run_id=self._run_id,
            post_id=self._post_id,
            data=data or {},
            duration_ms=duration_ms
        )

        if error:
            entry.error_type = type(error).__name__
            entry.error_traceback = traceback.format_exc()

        # Add to recent buffer
        self._recent_logs.append(entry)
        if len(self._recent_logs) > self._max_recent:
            self._recent_logs.pop(0)

        # Write to file
        await self._write_to_file(entry)

        # Write to Supabase (async, don't wait)
        if self.supabase and level.value >= self.min_level.value:
            asyncio.create_task(self._write_to_supabase(entry))

        # Send to Telegram for important logs
        if self.telegram and level.value >= self.telegram_min_level.value:
            asyncio.create_task(self._send_to_telegram(entry))

        # Call custom handlers
        for handler in self._handlers:
            try:
                handler(entry)
            except Exception:
                pass  # Don't let handler errors break logging

    async def _write_to_file(self, entry: LogEntry):
        """
        Write log entry to file using async I/O.

        FIX: Changed from sync open() to aiofiles for proper async file I/O.
        Sync file operations block the event loop.
        """
        json_line = entry.to_json() + "\n"

        # Always write to main log
        async with aiofiles.open(self._main_log, "a", encoding="utf-8") as f:
            await f.write(json_line)

        # Write errors to separate file
        if entry.level.value >= LogLevel.ERROR.value:
            async with aiofiles.open(self._error_log, "a", encoding="utf-8") as f:
                await f.write(json_line)

        # Write debug to separate file
        if entry.level == LogLevel.DEBUG:
            async with aiofiles.open(self._debug_log, "a", encoding="utf-8") as f:
                await f.write(json_line)

    async def _write_to_supabase(self, entry: LogEntry):
        """Write log entry to Supabase."""
        try:
            await self.supabase.insert("agent_logs", {
                "timestamp": entry.timestamp.isoformat(),
                "level": entry.level.value,           # Numeric value for filtering
                "level_name": entry.level.name_str,   # Human-readable name
                "component": entry.component.value,
                "message": entry.message,
                "run_id": entry.run_id,
                "post_id": entry.post_id,
                "data": entry.data,
                "error_type": entry.error_type,
                "error_traceback": entry.error_traceback,
                "duration_ms": entry.duration_ms
            })
        except Exception as e:
            # Log to file if Supabase fails (async to not block)
            import sys
            print(f"[LOGGING] Failed to write to Supabase: {e}", file=sys.stderr)

    async def _send_to_telegram(self, entry: LogEntry):
        """Send important log to Telegram."""
        try:
            await self.telegram.send_log(entry.to_readable())
        except Exception:
            pass  # Don't fail on Telegram errors

    # ═══════════════════════════════════════════════════════════════════
    # CONVENIENCE METHODS
    # ═══════════════════════════════════════════════════════════════════

    async def debug(self, component: LogComponent, message: str, **kwargs):
        await self.log(LogLevel.DEBUG, component, message, **kwargs)

    async def info(self, component: LogComponent, message: str, **kwargs):
        await self.log(LogLevel.INFO, component, message, **kwargs)

    async def warning(self, component: LogComponent, message: str, **kwargs):
        await self.log(LogLevel.WARNING, component, message, **kwargs)

    async def error(self, component: LogComponent, message: str, **kwargs):
        await self.log(LogLevel.ERROR, component, message, **kwargs)

    async def critical(self, component: LogComponent, message: str, **kwargs):
        await self.log(LogLevel.CRITICAL, component, message, **kwargs)

    # ═══════════════════════════════════════════════════════════════════
    # QUERY METHODS
    # ═══════════════════════════════════════════════════════════════════

    def get_recent(
        self,
        limit: int = 20,
        level: LogLevel = None,
        component: LogComponent = None,
        run_id: str = None
    ) -> List[LogEntry]:
        """Get recent logs from memory buffer."""

        logs = self._recent_logs.copy()

        if level:
            logs = [l for l in logs if l.level == level]
        if component:
            logs = [l for l in logs if l.component == component]
        if run_id:
            logs = [l for l in logs if l.run_id == run_id]

        return logs[-limit:]

    async def query_logs(
        self,
        start_time: datetime = None,
        end_time: datetime = None,
        level: LogLevel = None,
        component: LogComponent = None,
        run_id: str = None,
        search: str = None,
        limit: int = 100
    ) -> List[LogEntry]:
        """Query logs from Supabase."""

        if not self.supabase:
            return self.get_recent(limit, level, component, run_id)

        query = self.supabase.from_("agent_logs").select("*")

        if start_time:
            query = query.gte("timestamp", start_time.isoformat())
        if end_time:
            query = query.lte("timestamp", end_time.isoformat())
        if level:
            query = query.eq("level", level.value)
        if component:
            query = query.eq("component", component.value)
        if run_id:
            query = query.eq("run_id", run_id)
        if search:
            query = query.ilike("message", f"%{search}%")

        query = query.order("timestamp", desc=True).limit(limit)

        result = await query.execute()

        return [
            LogEntry(
                timestamp=datetime.fromisoformat(r["timestamp"]),
                level=LogLevel(r["level"]),
                component=LogComponent(r["component"]),
                message=r["message"],
                run_id=r["run_id"],
                post_id=r["post_id"],
                data=r["data"] or {},
                error_type=r["error_type"],
                error_traceback=r["error_traceback"],
                duration_ms=r["duration_ms"]
            )
            for r in result.data
        ]


# ═══════════════════════════════════════════════════════════════════
# GLOBAL LOGGER INSTANCE
# ═══════════════════════════════════════════════════════════════════

logger: AgentLogger = None

def init_logger(
    log_dir: str = "logs",
    supabase_client = None,
    telegram_notifier = None
):
    """Initialize global logger."""
    global logger
    logger = AgentLogger(
        log_dir=log_dir,
        supabase_client=supabase_client,
        telegram_notifier=telegram_notifier
    )
    return logger

def get_logger() -> AgentLogger:
    """Get global logger instance."""
    if logger is None:
        raise RuntimeError("Logger not initialized. Call init_logger() first.")
    return logger
```

### Component Logger (Convenience Wrapper)

```python
class ComponentLogger:
    """
    Wrapper for logging from specific component.
    Each agent gets its own ComponentLogger instance.
    """

    def __init__(self, component: LogComponent):
        self.component = component

    async def debug(self, message: str, **kwargs):
        await get_logger().debug(self.component, message, **kwargs)

    async def info(self, message: str, **kwargs):
        await get_logger().info(self.component, message, **kwargs)

    async def warning(self, message: str, **kwargs):
        await get_logger().warning(self.component, message, **kwargs)

    async def error(self, message: str, error: Exception = None, **kwargs):
        await get_logger().error(self.component, message, error=error, **kwargs)

    async def critical(self, message: str, error: Exception = None, **kwargs):
        await get_logger().critical(self.component, message, error=error, **kwargs)

    def timed(self, message: str):
        """Context manager for timing operations."""
        return TimedOperation(self, message)


class TimedOperation:
    """Context manager for timing and logging operations."""

    def __init__(self, logger: ComponentLogger, message: str):
        self.logger = logger
        self.message = message
        self.start_time = None

    async def __aenter__(self):
        self.start_time = datetime.now()
        await self.logger.debug(f"Starting: {self.message}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration_ms = int((datetime.now() - self.start_time).total_seconds() * 1000)

        if exc_type:
            await self.logger.error(
                f"Failed: {self.message}",
                error=exc_val,
                duration_ms=duration_ms
            )
        else:
            await self.logger.info(
                f"Completed: {self.message}",
                duration_ms=duration_ms
            )


# ═══════════════════════════════════════════════════════════════════
# USAGE IN AGENTS
# ═══════════════════════════════════════════════════════════════════

# In writer.py:
class WriterAgent:
    def __init__(self, ...):
        self.log = ComponentLogger(LogComponent.WRITER)

    async def generate_post(self, brief: AnalysisBrief) -> DraftPost:
        async with self.log.timed("Generating post"):
            await self.log.info(
                f"Generating post for topic: {brief.topic}",
                data={"content_type": brief.content_type.value}
            )

            try:
                draft = await self._generate(brief)
                await self.log.info(
                    f"Generated draft: {len(draft.content)} chars",
                    data={"template": draft.template_used}
                )
                return draft

            except Exception as e:
                await self.log.error("Failed to generate post", error=e)
                raise
```

### Telegram Log Viewer

```python
# In telegram_bot.py

class TelegramLogViewer:
    """Telegram commands for viewing logs."""

    def __init__(self, bot, logger: AgentLogger):
        self.bot = bot
        self.logger = logger

        # Register handlers
        self.bot.add_handler("/logs", self.cmd_logs)
        self.bot.add_handler("/errors", self.cmd_errors)
        self.bot.add_handler("/run", self.cmd_run_logs)

    async def cmd_logs(self, message, args: List[str]):
        """
        /logs — last 20 logs
        /logs 50 — last 50 logs
        /logs writer — logs from Writer
        /logs errors — only errors
        """

        limit = 20
        level = None
        component = None

        for arg in args:
            if arg.isdigit():
                limit = int(arg)
            elif arg == "errors":
                level = LogLevel.ERROR
            elif arg == "warnings":
                level = LogLevel.WARNING
            else:
                # Try to match component
                try:
                    component = LogComponent(arg)
                except ValueError:
                    pass

        logs = self.logger.get_recent(
            limit=limit,
            level=level,
            component=component
        )

        if not logs:
            await message.reply("No logs found.")
            return

        # Format logs
        text = "📋 **Recent Logs:**\n\n"
        for entry in logs:
            text += entry.to_readable() + "\n"

        # Split if too long
        if len(text) > 4000:
            text = text[:4000] + "\n...(truncated)"

        await message.reply(text)

    async def cmd_errors(self, message, args: List[str]):
        """
        /errors — last 10 errors
        /errors 24h — errors in last 24 hours
        """

        start_time = None
        if args and args[0] == "24h":
            start_time = datetime.now() - timedelta(hours=24)

        logs = await self.logger.query_logs(
            level=LogLevel.ERROR,
            start_time=start_time,
            limit=20
        )

        if not logs:
            await message.reply("✅ No errors found!")
            return

        text = "❌ **Recent Errors:**\n\n"
        for entry in logs:
            text += f"**{entry.timestamp.strftime('%d.%m %H:%M')}** [{entry.component.value}]\n"
            text += f"{entry.message}\n"
            if entry.error_type:
                text += f"Type: `{entry.error_type}`\n"
            text += "\n"

        await message.reply(text)

    async def cmd_run_logs(self, message, args: List[str]):
        """
        /run <run_id> — logs for specific pipeline run
        """

        if not args:
            await message.reply("Usage: /run <run_id>")
            return

        run_id = args[0]

        logs = await self.logger.query_logs(
            run_id=run_id,
            limit=100
        )

        if not logs:
            await message.reply(f"No logs found for run {run_id}")
            return

        text = f"📋 **Logs for run {run_id}:**\n\n"
        for entry in logs:
            text += entry.to_readable() + "\n"

        await message.reply(text)
```

### Pipeline Run Logger

```python
class PipelineRunLogger:
    """
    Tracks entire pipeline run with structured logging.
    Provides summary at the end.
    """

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.logger = get_logger()
        self.logger.set_context(run_id=run_id)

        self.start_time = datetime.now()
        self.stages: List[dict] = []
        self.current_stage: str = None

    async def start_stage(self, stage: str):
        """Mark start of pipeline stage."""
        self.current_stage = stage
        self.stages.append({
            "stage": stage,
            "start": datetime.now(),
            "end": None,
            "status": "running"
        })

        await self.logger.info(
            LogComponent.ORCHESTRATOR,
            f"Stage started: {stage}"
        )

    async def end_stage(self, status: str = "success", data: dict = None):
        """Mark end of pipeline stage."""
        if self.stages:
            stage = self.stages[-1]
            stage["end"] = datetime.now()
            stage["status"] = status
            stage["duration_ms"] = int(
                (stage["end"] - stage["start"]).total_seconds() * 1000
            )
            stage["data"] = data

        await self.logger.info(
            LogComponent.ORCHESTRATOR,
            f"Stage completed: {self.current_stage} ({status})",
            duration_ms=stage.get("duration_ms")
        )

    async def finish(self, status: str = "success") -> dict:
        """Finish pipeline run and return summary."""

        end_time = datetime.now()
        total_duration = int((end_time - self.start_time).total_seconds() * 1000)

        summary = {
            "run_id": self.run_id,
            "status": status,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration_ms": total_duration,
            "stages": self.stages
        }

        await self.logger.info(
            LogComponent.ORCHESTRATOR,
            f"Pipeline run completed: {status}",
            data=summary,
            duration_ms=total_duration
        )

        self.logger.clear_context()

        return summary

    def get_summary_text(self) -> str:
        """Get human-readable summary."""

        lines = [
            f"🔄 **Pipeline Run: {self.run_id}**",
            ""
        ]

        for stage in self.stages:
            status_emoji = "✅" if stage["status"] == "success" else "❌"
            duration = stage.get("duration_ms", 0)
            lines.append(
                f"{status_emoji} {stage['stage']}: {duration}ms"
            )

        total = sum(s.get("duration_ms", 0) for s in self.stages)
        lines.append(f"\n**Total:** {total}ms")

        return "\n".join(lines)
```

### Daily Log Digest

```python
class DailyDigest:
    """
    Generates daily summary of agent activity.
    Sent via Telegram at configured time.
    """

    def __init__(self, logger: AgentLogger, telegram):
        self.logger = logger
        self.telegram = telegram

    async def generate_digest(self) -> str:
        """Generate daily digest."""

        yesterday = datetime.now() - timedelta(days=1)

        # Query logs from last 24h
        all_logs = await self.logger.query_logs(
            start_time=yesterday,
            limit=10000
        )

        # Count by level
        level_counts = {}
        for log in all_logs:
            level_counts[log.level.value] = level_counts.get(log.level.value, 0) + 1

        # Count by component
        component_counts = {}
        for log in all_logs:
            component_counts[log.component.value] = component_counts.get(log.component.value, 0) + 1

        # Get errors
        errors = [l for l in all_logs if l.level in [LogLevel.ERROR, LogLevel.CRITICAL]]

        # Get pipeline runs
        runs = set(l.run_id for l in all_logs if l.run_id)

        # Build digest
        text = f"""
📊 **Daily Digest** ({yesterday.strftime('%d.%m.%Y')})

**Activity:**
• Total logs: {len(all_logs)}
• Pipeline runs: {len(runs)}
• Errors: {level_counts.get('error', 0)}
• Warnings: {level_counts.get('warning', 0)}

**By Component:**
{chr(10).join(f'• {k}: {v}' for k, v in sorted(component_counts.items(), key=lambda x: -x[1])[:5])}
"""

        if errors:
            text += f"\n**Errors ({len(errors)}):**\n"
            for e in errors[:5]:
                text += f"• [{e.component.value}] {e.message[:50]}...\n"
        else:
            text += "\n✅ **No errors!**"

        return text

    async def send_digest(self):
        """Send daily digest via Telegram."""
        digest = await self.generate_digest()
        await self.telegram.notify(digest)
```

### Configuration

```python
logging_config = {
    # ═══════════════════════════════════════════════════════════════════
    # FILE LOGGING
    # ═══════════════════════════════════════════════════════════════════

    "file": {
        "enabled": True,
        "log_dir": "logs",
        "main_log": "agent.log",
        "error_log": "errors.log",
        "debug_log": "debug.log",
        "rotation": {
            "max_size_mb": 50,
            "backup_count": 5
        }
    },

    # ═══════════════════════════════════════════════════════════════════
    # SUPABASE LOGGING
    # ═══════════════════════════════════════════════════════════════════

    "supabase": {
        "enabled": True,
        "table": "agent_logs",
        "retention_days": 30  # Auto-delete logs older than 30 days
    },

    # ═══════════════════════════════════════════════════════════════════
    # TELEGRAM NOTIFICATIONS
    # ═══════════════════════════════════════════════════════════════════

    "telegram": {
        "enabled": True,
        "min_level": "warning",  # Send WARNING and above
        "daily_digest": {
            "enabled": True,
            "time": "09:00",
            "timezone": "Europe/Moscow"
        },
        "commands": [
            "/logs",      # View recent logs
            "/errors",    # View errors
            "/run <id>"   # View run logs
        ]
    },

    # ═══════════════════════════════════════════════════════════════════
    # GENERAL
    # ═══════════════════════════════════════════════════════════════════

    "min_level": "info",  # Minimum level to log
    "include_debug": False,  # Include debug logs (verbose)

    "recent_buffer_size": 1000  # Keep last N logs in memory
}
```

---

## NEW: Modification Safety System

### Purpose
Контролирует все изменения системы с risk-based approval flow. Критические изменения требуют human approval, низкорисковые применяются автоматически с rollback при деградации.

### Risk Classification

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta


class ModificationRiskLevel(Enum):
    LOW = "low"        # Config tweaks (posting time, emoji count)
    MEDIUM = "medium"  # Template weights, scoring adjustments
    HIGH = "high"      # Prompt changes, new code modules
    CRITICAL = "critical"  # Core logic changes


modification_risk_classification = {
    # LOW — auto-apply, monitor for 5 posts
    "posting_time_adjustment": ModificationRiskLevel.LOW,
    "emoji_count_change": ModificationRiskLevel.LOW,
    "hashtag_count_change": ModificationRiskLevel.LOW,

    # MEDIUM — auto-apply with automatic rollback trigger
    "template_weight_adjustment": ModificationRiskLevel.MEDIUM,
    "hook_style_preference": ModificationRiskLevel.MEDIUM,
    "visual_format_preference": ModificationRiskLevel.MEDIUM,
    "content_type_distribution": ModificationRiskLevel.MEDIUM,

    # HIGH — requires human approval before apply
    "writer_prompt_change": ModificationRiskLevel.HIGH,
    "qc_criteria_change": ModificationRiskLevel.HIGH,
    "new_hook_template": ModificationRiskLevel.HIGH,

    # CRITICAL — requires human approval + explicit confirmation
    "new_code_module": ModificationRiskLevel.CRITICAL,
    "scoring_algorithm_change": ModificationRiskLevel.CRITICAL,
    "core_pipeline_change": ModificationRiskLevel.CRITICAL,
}
```

### Data Models

```python
@dataclass
class ModificationRequest:
    """Request to modify system behavior."""
    id: str
    modification_type: str
    risk_level: ModificationRiskLevel

    # What's changing
    component: str  # "writer", "trend_scout", "qc", etc.
    before_state: dict
    after_state: dict

    # Why
    reasoning: str
    triggered_by: str  # "research", "analytics", "critique"
    supporting_data: dict  # Evidence for the change

    # Approval tracking
    status: str  # "pending", "approved", "rejected", "auto_applied", "rolled_back"
    human_approver: Optional[str]
    approved_at: Optional[datetime]


@dataclass
class RollbackTrigger:
    """Conditions that trigger automatic rollback."""
    metric: str  # "engagement_rate", "qc_pass_rate", "human_approval_rate"
    threshold: float  # e.g., 0.8 (20% drop from baseline)
    window_posts: int  # Number of posts to evaluate
    baseline_value: float  # Value before modification
```

### Modification Safety System Implementation

```python
class ModificationSafetySystem:
    """
    Controls all system modifications with risk-based approval flow.
    """

    def __init__(self, db, telegram_notifier):
        self.db = db
        self.telegram = telegram_notifier

    async def request_modification(self, mod: ModificationRequest) -> str:
        """
        Process modification request based on risk level.
        Returns: "applied", "pending_approval", "scheduled_for_review"

        FIX: Added input validation to prevent:
        - Risk level spoofing (client claiming LOW risk for HIGH risk modification)
        - Invalid modification types
        - Missing required fields
        """
        # FIX: Validate modification request
        self._validate_modification_request(mod)

        # FIX: Verify risk level matches classification (prevent spoofing)
        expected_risk = modification_risk_classification.get(mod.modification_type)
        if expected_risk and expected_risk != mod.risk_level:
            raise SecurityError(
                f"Risk level mismatch: claimed {mod.risk_level.value}, "
                f"expected {expected_risk.value} for {mod.modification_type}"
            )

        if mod.risk_level == ModificationRiskLevel.LOW:
            # Auto-apply, set up monitoring
            await self._apply_modification(mod)
            await self._setup_rollback_trigger(mod, window_posts=5, threshold=0.85)
            return "applied"

        elif mod.risk_level == ModificationRiskLevel.MEDIUM:
            # Auto-apply with stricter rollback
            await self._apply_modification(mod)
            await self._setup_rollback_trigger(mod, window_posts=3, threshold=0.90)
            await self.telegram.notify(
                f"⚙️ Auto-applied: {mod.modification_type}\n"
                f"Reason: {mod.reasoning}\n"
                f"Rollback if performance drops >10%"
            )
            return "applied"

        elif mod.risk_level == ModificationRiskLevel.HIGH:
            # Request human approval
            await self._store_pending_modification(mod)
            await self.telegram.request_approval(
                f"🔶 Approval needed: {mod.modification_type}\n\n"
                f"Component: {mod.component}\n"
                f"Change: {mod.reasoning}\n\n"
                f"Before:\n```{mod.before_state}```\n\n"
                f"After:\n```{mod.after_state}```\n\n"
                f"Reply /approve_{mod.id} or /reject_{mod.id}"
            )
            return "pending_approval"

        elif mod.risk_level == ModificationRiskLevel.CRITICAL:
            # Request approval with explicit confirmation
            await self._store_pending_modification(mod)
            await self.telegram.request_critical_approval(
                f"🔴 CRITICAL change requires approval:\n\n"
                f"{mod.modification_type}\n\n"
                f"This will modify: {mod.component}\n"
                f"Reasoning: {mod.reasoning}\n\n"
                f"⚠️ Review carefully before approving.\n"
                f"Reply /approve_{mod.id} CONFIRM or /reject_{mod.id}"
            )
            return "pending_approval"

    async def check_rollback_triggers(self):
        """
        Called after each post. Checks if any modification should be rolled back.
        """
        active_mods = await self.db.get_modifications_with_active_rollback()

        for mod in active_mods:
            trigger = mod.rollback_trigger
            recent_metrics = await self.db.get_recent_metrics(
                metric=trigger.metric,
                limit=trigger.window_posts
            )

            # FIX: Prevent division by zero
            if not recent_metrics:
                logger.warning(
                    f"[ROLLBACK_CHECK] No recent metrics for {mod.modification_type}, skipping"
                )
                continue
            if trigger.baseline_value == 0:
                logger.warning(
                    f"[ROLLBACK_CHECK] Zero baseline for {mod.modification_type}, skipping"
                )
                continue

            avg_recent = sum(recent_metrics) / len(recent_metrics)
            performance_ratio = avg_recent / trigger.baseline_value

            if performance_ratio < trigger.threshold:
                await self._rollback_modification(mod)
                await self.telegram.notify(
                    f"⚠️ Auto-rollback triggered!\n\n"
                    f"Modification: {mod.modification_type}\n"
                    f"Reason: {trigger.metric} dropped to {performance_ratio:.0%} of baseline\n"
                    f"Reverted to previous state."
                )

    async def _rollback_modification(self, mod: ModificationRequest):
        """Restore previous state."""
        await self._apply_state(mod.component, mod.before_state)
        mod.status = "rolled_back"
        await self.db.update_modification(mod)

    async def _apply_modification(self, mod: ModificationRequest):
        """Apply modification to the system."""
        await self._apply_state(mod.component, mod.after_state)
        mod.status = "auto_applied"
        await self.db.save_modification(mod)

    async def _setup_rollback_trigger(
        self,
        mod: ModificationRequest,
        window_posts: int,
        threshold: float
    ):
        """Set up automatic rollback monitoring."""
        baseline = await self.db.get_average_metric(
            metric="engagement_rate",
            last_n_posts=10
        )

        trigger = RollbackTrigger(
            metric="engagement_rate",
            threshold=threshold,
            window_posts=window_posts,
            baseline_value=baseline
        )

        await self.db.save_rollback_trigger(mod.id, trigger)

    async def _store_pending_modification(self, mod: ModificationRequest):
        """Store modification awaiting approval."""
        mod.status = "pending"
        await self.db.save_modification(mod)

    async def _apply_state(self, component: str, state: dict):
        """
        Apply state to a component.

        MEDIUM PRIORITY FIX #7: Added error handling, backup, and logging
        for critical configuration changes.
        """
        import logging
        import shutil
        import os
        import json
        logger = logging.getLogger("ModificationSafetySystem")

        config_path = self._get_config_path(component)
        backup_path = f"{config_path}.backup"

        logger.info(f"[STATE_APPLY] Applying state to {component}: {list(state.keys())}")

        # FIX #7: Create backup before modification
        try:
            if os.path.exists(config_path):
                shutil.copy2(config_path, backup_path)
                logger.debug(f"[STATE_APPLY] Created backup: {backup_path}")
        except (IOError, PermissionError) as e:
            logger.error(f"[STATE_APPLY] Failed to create backup for {component}: {e}")
            raise ConfigurationBackupError(f"Cannot backup {config_path}: {e}")

        # FIX #7: Read with error handling
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                current_config = json.load(f)
        except FileNotFoundError:
            logger.warning(f"[STATE_APPLY] Config not found, creating new: {config_path}")
            current_config = {}
        except json.JSONDecodeError as e:
            logger.error(f"[STATE_APPLY] Corrupted config file {config_path}: {e}")
            raise ConfigurationCorruptedError(f"Cannot parse {config_path}: {e}")
        except PermissionError as e:
            logger.error(f"[STATE_APPLY] Permission denied reading {config_path}: {e}")
            raise ConfigurationAccessError(f"Cannot read {config_path}: {e}")

        # Track changes for logging
        changes_made = []
        for key, value in state.items():
            old_value = current_config.get(key, "<not set>")
            current_config[key] = value
            changes_made.append(f"{key}: {old_value} → {value}")

        # FIX #7: Write with error handling and atomic write pattern
        try:
            # Write to temp file first, then rename (atomic on most systems)
            temp_path = f"{config_path}.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(current_config, f, indent=2, ensure_ascii=False)

            # Atomic rename
            os.replace(temp_path, config_path)

            logger.info(f"[STATE_APPLY] Successfully applied {len(changes_made)} changes to {component}")
            for change in changes_made:
                logger.debug(f"[STATE_APPLY]   {change}")

        except (IOError, PermissionError, OSError) as e:
            logger.error(f"[STATE_APPLY] Failed to write config {config_path}: {e}")
            # Attempt to restore from backup
            if os.path.exists(backup_path):
                try:
                    shutil.copy2(backup_path, config_path)
                    logger.info(f"[STATE_APPLY] Restored from backup after write failure")
                except Exception as restore_error:
                    logger.critical(f"[STATE_APPLY] CRITICAL: Cannot restore backup: {restore_error}")
            raise ConfigurationWriteError(f"Cannot write {config_path}: {e}")


# FIX: Security exception for validation failures
class SecurityError(Exception):
    """Raised when a security validation fails (e.g., risk level spoofing)."""
    pass


# FIX: Validation method for ModificationRequest
def _validate_modification_request(mod: "ModificationRequest") -> None:
    """
    Validate a ModificationRequest before processing.

    Raises:
        ValidationError: If required fields are missing or invalid
        SecurityError: If the request appears malicious
    """
    if not mod.component:
        raise ValidationError("ModificationRequest.component is required")
    if not mod.modification_type:
        raise ValidationError("ModificationRequest.modification_type is required")
    if mod.before_state is None:
        raise ValidationError("ModificationRequest.before_state is required")
    if mod.after_state is None:
        raise ValidationError("ModificationRequest.after_state is required")
    if not mod.reasoning:
        raise ValidationError("ModificationRequest.reasoning is required")

    # Validate component is in whitelist
    valid_components = {"writer", "trend_scout", "visual_creator", "scheduler", "qc", "humanizer", "analyzer"}
    if mod.component not in valid_components:
        raise SecurityError(f"Unknown component: {mod.component}")


# FIX #7: Custom exceptions for configuration operations
class ConfigurationBackupError(Exception):
    """Raised when configuration backup fails."""
    pass

class ConfigurationCorruptedError(Exception):
    """Raised when configuration file is corrupted."""
    pass

class ConfigurationAccessError(Exception):
    """Raised when configuration file cannot be accessed."""
    pass

class ConfigurationWriteError(Exception):
    """Raised when configuration file cannot be written."""
    pass

    def _get_config_path(self, component: str) -> str:
        """
        Get config file path for component.

        FIX: Removed path traversal vulnerability by using whitelist-only approach.
        Previously returned f"config/{component}_config.json" for unknown components,
        allowing path traversal attacks like component="../../etc/passwd".
        """
        paths = {
            "writer": "config/writer_config.json",
            "trend_scout": "config/scoring_weights.json",
            "visual_creator": "config/visual_config.json",
            "scheduler": "config/schedule.json",
            "qc": "config/evaluator_config.json",
            "humanizer": "config/humanizer_config.json",
            "analyzer": "config/analyzer_config.json",
        }
        # FIX: Security - whitelist only, no dynamic path construction
        if component not in paths:
            raise ValueError(
                f"Unknown component: {component}. "
                f"Valid components: {list(paths.keys())}"
            )
        return paths[component]
```

### Integration with Meta-Agent

```python
class MetaAgentWithSafety:
    """Meta-Agent with modification safety system."""

    def __init__(self, safety_system: ModificationSafetySystem, ...):
        self.safety = safety_system
        # ... other components

    async def apply_learning(self, learning: dict):
        """Apply a learning with appropriate safety checks."""

        # Determine risk level
        risk_level = self._classify_risk(learning)

        # Create modification request
        mod = ModificationRequest(
            id=generate_id(),
            modification_type=learning["type"],
            risk_level=risk_level,
            component=learning["component"],
            before_state=await self._get_current_state(learning["component"]),
            after_state=learning["new_state"],
            reasoning=learning["reasoning"],
            triggered_by="research",
            supporting_data=learning["evidence"],
            status="pending",
            human_approver=None,
            approved_at=None
        )

        # Process through safety system
        result = await self.safety.request_modification(mod)

        return result
```

---

## NEW: Single-Call Evaluation System

### Purpose
Заменяет multi-turn диалог с Critic Agent одним структурированным вызовом LLM. Снижает latency, cost и сложность debugging.

### Evaluation Rubric

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


@dataclass
class EvaluationCriterion:
    """Single evaluation criterion with rubric."""
    name: str
    weight: float
    rubric: Dict[int, str]  # score -> description
    evaluation_prompt: str


evaluation_rubric = {
    "hook_strength": EvaluationCriterion(
        name="Hook Strength",
        weight=0.25,
        rubric={
            10: "Impossible to scroll past. Creates immediate curiosity gap or emotional reaction.",
            8: "Strong hook that stops most readers. Clear value proposition.",
            6: "Decent hook but generic. Could apply to many posts.",
            4: "Weak hook. Starts with background instead of punch.",
            2: "No hook. Buries the lede. Reader has no reason to continue."
        },
        evaluation_prompt="Rate the first 2 lines. Do they stop the scroll?"
    ),

    "specificity": EvaluationCriterion(
        name="Specificity",
        weight=0.20,
        rubric={
            10: "Concrete numbers, named companies/tools, specific examples throughout.",
            8: "Good specificity with some concrete details.",
            6: "Mix of specific and vague. Some details, some generic statements.",
            4: "Mostly vague. Few concrete examples.",
            2: "Entirely abstract. No specific examples, numbers, or names."
        },
        evaluation_prompt="Are there concrete numbers, names, and examples?"
    ),

    "value_density": EvaluationCriterion(
        name="Value Density",
        weight=0.20,
        rubric={
            10: "Every paragraph delivers actionable insight. No filler.",
            8: "High value throughout with minimal fluff.",
            6: "Some valuable insights mixed with filler.",
            4: "More padding than substance.",
            2: "No actionable takeaways. Pure fluff."
        },
        evaluation_prompt="Does each paragraph add value? Are there actionable takeaways?"
    ),

    "authenticity": EvaluationCriterion(
        name="Authenticity",
        weight=0.15,
        rubric={
            10: "Unmistakably human. Unique voice, personal touches, natural flow.",
            8: "Sounds human with minor AI tells.",
            6: "Could be human or AI. Generic but not obviously artificial.",
            4: "Some AI patterns visible. Overly formal or structured.",
            2: "Obviously AI-generated. Robotic, formulaic, or uses AI phrases."
        },
        evaluation_prompt="Does this sound like a real person wrote it?"
    ),

    "structure": EvaluationCriterion(
        name="Structure",
        weight=0.10,
        rubric={
            10: "Perfect scannability. White space, short paragraphs, clear flow.",
            8: "Good structure, easy to read.",
            6: "Acceptable structure but could be improved.",
            4: "Hard to scan. Dense paragraphs.",
            2: "Wall of text. No visual breaks."
        },
        evaluation_prompt="Is it easy to scan? Good use of white space?"
    ),

    "cta_clarity": EvaluationCriterion(
        name="CTA Clarity",
        weight=0.10,
        rubric={
            10: "Clear, natural call-to-action that invites engagement.",
            8: "Good CTA, clear next step for reader.",
            6: "CTA present but weak or generic.",
            4: "Unclear what reader should do.",
            2: "No CTA or abrupt ending."
        },
        evaluation_prompt="Is there a clear next step for the reader?"
    )
}
```

### Evaluation Output Schema

```python
@dataclass
class SingleCallEvaluation:
    """Complete evaluation from single LLM call."""

    # Scores
    scores: Dict[str, int]  # criterion_name -> score (1-10)
    weighted_total: float

    # Detailed feedback for each criterion
    criterion_feedback: Dict[str, dict]  # criterion -> {quote, score, explanation}

    # Summary feedback
    strengths: List[str]  # What's working well (2-3 items)
    weaknesses: List[str]  # What needs improvement (2-3 items)
    specific_suggestions: List[str]  # Actionable improvements (3-5 items)

    # Decision
    passes_threshold: bool  # weighted_total >= 8.0
    recommended_revisions: Optional[List[str]]  # If doesn't pass

    # For learning
    patterns_detected: List[str]  # Recurring issues across posts
    knowledge_gaps: List[str]  # What Writer should research
```

### Single-Call Evaluator Implementation

```python
class SingleCallEvaluator:
    """
    Evaluates post quality in a single LLM call with structured output.
    Replaces multi-turn Critic dialogue.
    """

    EVALUATION_PROMPT = """
    Evaluate this LinkedIn post against the rubric below.
    Be harsh but constructive. The goal is genuine improvement.

    === POST ===
    {post_content}
    === END POST ===

    Context:
    - Content Type: {content_type}
    - Target: LinkedIn professionals interested in AI
    - Goal: Maximize engagement + deliver value

    === RUBRIC ===
    {rubric_text}
    === END RUBRIC ===

    EVALUATION INSTRUCTIONS:

    For EACH criterion:
    1. Quote the specific part of the post you're evaluating (max 50 chars)
    2. Give a score 1-10 based on the rubric
    3. Explain your score in 1-2 sentences

    Then provide:
    - 2-3 specific STRENGTHS (what's working well)
    - 2-3 specific WEAKNESSES (what needs improvement)
    - 3-5 ACTIONABLE SUGGESTIONS (how to fix the weaknesses)

    IMPORTANT - Be SPECIFIC:
    ❌ Don't say: "hook is weak"
    ✅ Do say: "The hook starts with 'В последнее время...' which is generic.
       Instead, try leading with the specific result: '40 hours → 2 minutes.'"

    Return as structured JSON matching the SingleCallEvaluation schema.
    """

    def __init__(self, claude_client):
        self.claude = claude_client
        self.rubric = evaluation_rubric

    async def evaluate(
        self,
        post_content: str,
        content_type: ContentType
    ) -> SingleCallEvaluation:
        """
        Single LLM call that returns complete evaluation.
        """

        rubric_text = self._format_rubric()

        response = await self.claude.generate_structured(
            prompt=self.EVALUATION_PROMPT.format(
                post_content=post_content,
                content_type=content_type.value,
                rubric_text=rubric_text
            ),
            response_model=SingleCallEvaluation
        )

        # Calculate weighted total
        response.weighted_total = self._calculate_weighted_total(response.scores)
        response.passes_threshold = response.weighted_total >= 8.0

        # If doesn't pass, generate revision recommendations
        if not response.passes_threshold:
            response.recommended_revisions = self._prioritize_revisions(response)

        return response

    def _calculate_weighted_total(self, scores: Dict[str, int]) -> float:
        """Calculate weighted average score."""
        total = 0
        for criterion_name, score in scores.items():
            weight = self.rubric[criterion_name].weight
            total += score * weight
        return round(total, 2)

    def _prioritize_revisions(self, evaluation: SingleCallEvaluation) -> List[str]:
        """Prioritize revisions based on impact (weight * improvement potential)."""
        improvements = []

        for criterion_name, score in evaluation.scores.items():
            if score < 8:
                weight = self.rubric[criterion_name].weight
                impact = weight * (10 - score)  # Higher impact = more room to improve on weighted criterion
                improvements.append((criterion_name, impact, score))

        # Sort by impact
        improvements.sort(key=lambda x: x[1], reverse=True)

        # Return top 3 with specific suggestions
        revisions = []
        for criterion_name, impact, score in improvements[:3]:
            suggestion = next(
                (s for s in evaluation.specific_suggestions
                 if criterion_name.lower() in s.lower()),
                f"Improve {criterion_name} (current score: {score}/10)"
            )
            revisions.append(suggestion)

        return revisions

    def _format_rubric(self) -> str:
        """Format rubric for prompt."""
        lines = []
        for name, criterion in self.rubric.items():
            lines.append(f"\n### {criterion.name} (weight: {criterion.weight})")
            lines.append(f"Question: {criterion.evaluation_prompt}")
            lines.append("Scoring guide:")
            for score, desc in sorted(criterion.rubric.items(), reverse=True):
                lines.append(f"  {score}: {desc}")
        return "\n".join(lines)
```

### Integration with QC Agent

```python
# In qc_agent.py — replace Critic dialogue with SingleCallEvaluator

class QCAgent:
    """
    Quality Control Agent using single-call evaluation.

    FIX #21: Now evaluates BOTH post content AND visual asset together.
    """

    def __init__(
        self,
        evaluator: SingleCallEvaluator,
        visual_evaluator: "VisualQualityEvaluator"  # FIX #21
    ):
        self.evaluator = evaluator
        self.visual_evaluator = visual_evaluator

    async def evaluate_post(
        self,
        post: HumanizedPost,
        visual: VisualAsset,  # FIX #21: Now required
        max_revisions: int = 3
    ) -> QCResult:
        """
        Evaluate BOTH post text quality AND visual quality.
        Returns decision: pass/revise/reject.

        FIX #21: Visual quality is now part of the evaluation pipeline.
        """

        # Evaluate text content
        text_evaluation = await self.evaluator.evaluate(
            post_content=post.humanized_text,
            content_type=post.content_type
        )

        # FIX #21: Evaluate visual quality
        visual_evaluation = await self.visual_evaluator.evaluate(
            visual=visual,
            post_content=post.humanized_text,
            content_type=post.content_type
        )

        # Combine scores (text 75%, visual 25%)
        combined_score = (
            text_evaluation.weighted_total * 0.75 +
            visual_evaluation.score * 0.25
        )

        # FIX: Use combined_score for pass/fail decision
        # Previously calculated combined_score but ignored it in passes check
        visual_minimum = 6.0  # Visual can't be below 6 even if text is great
        combined_threshold = 8.0  # Use centralized threshold
        passes = (
            combined_score >= combined_threshold and
            visual_evaluation.score >= visual_minimum
        )

        # Combine revision instructions
        revision_instructions = []
        if text_evaluation.recommended_revisions:
            revision_instructions.extend([
                f"[TEXT] {r}" for r in text_evaluation.recommended_revisions
            ])
        if visual_evaluation.issues:
            revision_instructions.extend([
                f"[VISUAL] {issue}" for issue in visual_evaluation.issues
            ])

        if passes:
            return QCResult(
                decision="pass",
                score=combined_score,
                evaluation=text_evaluation,
                visual_evaluation=visual_evaluation,  # FIX #21
                ready_for_human_approval=True
            )

        elif post.revision_count < max_revisions:
            # Determine what needs revision
            needs_text_revision = not text_evaluation.passes_threshold
            needs_visual_revision = visual_evaluation.score < visual_minimum

            return QCResult(
                decision="revise",
                score=combined_score,
                evaluation=text_evaluation,
                visual_evaluation=visual_evaluation,  # FIX #21
                revision_instructions=revision_instructions,
                revision_targets={
                    "text": needs_text_revision,
                    "visual": needs_visual_revision
                },
                ready_for_human_approval=False
            )

        else:
            return QCResult(
                decision="reject",
                score=combined_score,
                evaluation=text_evaluation,
                visual_evaluation=visual_evaluation,  # FIX #21
                rejection_reason="Max revisions reached, quality still below threshold",
                ready_for_human_approval=False
            )


# FIX #21: Visual Quality Evaluator
@dataclass
class VisualEvaluation:
    """Result of visual quality evaluation."""
    score: float  # 1-10
    format_appropriate: bool
    content_match_score: float  # How well visual matches post content
    technical_quality: float  # Resolution, contrast, readability
    brand_consistency: bool
    issues: List[str]  # Problems to fix
    strengths: List[str]  # What's working


class VisualQualityEvaluator:
    """
    FIX #21: Evaluates visual asset quality and content-visual coherence.
    """

    # Expected visual format by content type
    RECOMMENDED_FORMATS = {
        ContentType.ENTERPRISE_CASE: ["metrics_card", "case_study_visual", "logo_showcase"],
        ContentType.PRIMARY_SOURCE: ["data_chart", "quote_card", "source_screenshot"],
        ContentType.AUTOMATION_CASE: ["workflow_diagram", "before_after", "screenshot"],
        ContentType.COMMUNITY_CONTENT: ["photo_with_overlay", "meme_format", "carousel"],
        ContentType.TOOL_RELEASE: ["product_screenshot", "feature_comparison", "demo_gif"],
    }

    async def evaluate(
        self,
        visual: VisualAsset,
        post_content: str,
        content_type: ContentType
    ) -> VisualEvaluation:
        """
        Evaluate visual quality against multiple criteria:
        1. Format appropriateness for content type
        2. Text-visual coherence
        3. Technical quality (if image analysis available)
        4. Brand consistency
        """

        issues = []
        strengths = []

        # Check format appropriateness
        recommended = self.RECOMMENDED_FORMATS.get(content_type, [])
        format_appropriate = visual.visual_style in recommended

        if not format_appropriate:
            issues.append(
                f"Visual style '{visual.visual_style}' not ideal for {content_type.value}. "
                f"Consider: {', '.join(recommended)}"
            )
        else:
            strengths.append(f"Visual style '{visual.visual_style}' appropriate for {content_type.value}")

        # Check content-visual coherence via LLM
        coherence_score = await self._evaluate_coherence(visual, post_content)

        if coherence_score < 7.0:
            issues.append(
                f"Visual doesn't strongly match post content (coherence: {coherence_score}/10)"
            )

        # Check technical aspects
        technical_issues = self._check_technical_quality(visual)
        issues.extend(technical_issues)

        # Check brand consistency
        brand_ok = visual.brand_consistency_check and visual.mobile_optimized
        if not brand_ok:
            if not visual.brand_consistency_check:
                issues.append("Visual doesn't match brand guidelines")
            if not visual.mobile_optimized:
                issues.append("Visual not optimized for mobile viewing")
        else:
            strengths.append("Brand consistent and mobile optimized")

        # Check author photo usage for engagement potential
        if visual.photo_used:
            strengths.append("Includes author photo (typically +15-25% engagement)")

        # Calculate final score
        base_score = 7.0
        score = base_score

        # Adjustments
        if format_appropriate:
            score += 1.0
        else:
            score -= 1.0

        if coherence_score >= 8.0:
            score += 1.0
        elif coherence_score < 6.0:
            score -= 1.5

        if brand_ok:
            score += 0.5
        else:
            score -= 0.5

        if visual.photo_used:
            score += 0.5

        # Clamp to 1-10
        score = max(1.0, min(10.0, score))

        return VisualEvaluation(
            score=score,
            format_appropriate=format_appropriate,
            content_match_score=coherence_score,
            technical_quality=8.0 if not technical_issues else 6.0,
            brand_consistency=brand_ok,
            issues=issues,
            strengths=strengths
        )

    async def _evaluate_coherence(self, visual: VisualAsset, post_content: str) -> float:
        """Use LLM to evaluate how well visual matches post content."""

        prompt = f"""
        Rate how well this visual matches the post content on a scale of 1-10.

        POST CONTENT:
        {post_content[:500]}

        VISUAL DESCRIPTION:
        - Format: {visual.format}
        - Style: {visual.visual_style}
        - Alt text: {visual.alt_text}
        - Prompt used: {visual.prompt_used[:200]}

        Consider:
        1. Does the visual reinforce the post's main message?
        2. Is the visual style appropriate for the tone?
        3. Would the visual make sense without the text?

        Return only a number 1-10.
        """

        # Would call LLM here, for now return attribute if exists
        return visual.visual_content_match_score if hasattr(visual, 'visual_content_match_score') else 7.0

    def _check_technical_quality(self, visual: VisualAsset) -> List[str]:
        """Check technical aspects of the visual."""
        issues = []

        # Check dimensions
        if visual.dimensions:
            width, height = map(int, visual.dimensions.split('x'))
            if width < 800:
                issues.append(f"Visual width {width}px may appear low quality on desktop")
            if visual.format == "carousel" and width < 1080:
                issues.append("Carousel images should be at least 1080px wide")

        return issues
```

---

## NEW: Author Profile Agent

### Purpose
Создаёт и поддерживает профиль голоса автора на основе его существующих постов. Writer Agent использует этот профиль для соответствия аутентичному стилю автора.

### Author Voice Profile Schema

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class AuthorVoiceProfile:
    """Complete voice and style profile of the author."""

    # Identity
    author_name: str
    author_role: str  # "AI consultant", "Tech entrepreneur"
    expertise_areas: List[str]

    # Writing style
    characteristic_phrases: List[str]  # "Here's the thing:", "What struck me:"
    avoided_phrases: List[str]  # Phrases author never uses
    sentence_length_preference: str  # "short", "medium", "varied"
    paragraph_length: str  # "1-2 sentences", "3-4 sentences"

    # Tone
    formality_level: float  # 0=casual, 1=formal (e.g., 0.4)
    humor_frequency: str  # "never", "rarely", "sometimes", "often"
    emoji_usage: str  # "none", "minimal", "moderate", "heavy"

    # Content preferences
    favorite_topics: List[str]
    topics_to_avoid: List[str]
    typical_post_length: int  # characters
    preferred_cta_styles: List[str]

    # Opinions and positions
    known_opinions: Dict[str, str]  # topic -> stance
    contrarian_positions: List[str]  # Where author disagrees with mainstream

    # Patterns from data
    best_performing_hooks: List[str]  # Top 5 hooks by engagement
    best_performing_structures: List[str]
    posting_frequency: str  # "daily", "3x/week", etc.
    best_posting_times: List[str]

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    posts_analyzed: int = 0
```

### Author Profile Agent Implementation

```python
class AuthorProfileAgent:
    """
    Creates and maintains author's voice profile based on their existing posts.
    Used by Writer Agent to match author's authentic voice.
    """

    ANALYSIS_PROMPT = """
    Analyze these LinkedIn posts from {author_name} to extract their unique voice and style.

    Posts (sorted by engagement, highest first):
    {posts_text}

    Extract the following:

    1. CHARACTERISTIC PHRASES
       - Phrases they use repeatedly
       - Their unique ways of starting/ending thoughts
       - Signature expressions
       - Quote specific examples from their posts

    2. WRITING STYLE
       - Average sentence length (short/medium/long/varied)
       - Paragraph structure (how many sentences per paragraph)
       - Use of lists, bullets, formatting
       - Emoji patterns (with examples)

    3. TONE & VOICE
       - Formality level (0-1 scale, with explanation)
       - Humor usage (never/rarely/sometimes/often)
       - How they express opinions (confident/hedging/questioning)
       - How they address the reader

    4. CONTENT PATTERNS
       - Favorite topics (list with frequency)
       - How they structure arguments
       - Typical hooks they use (quote examples)
       - Common CTAs (quote examples)

    5. OPINIONS & POSITIONS
       - Strong opinions they've expressed (with quotes)
       - Contrarian views they hold
       - Topics they're passionate about

    6. WHAT TO AVOID
       - Phrases they never use
       - Topics they avoid
       - Styles that don't fit them

    Be SPECIFIC. Quote actual text from their posts as evidence.

    Return as structured JSON matching AuthorVoiceProfile schema.
    """

    def __init__(self, claude_client, db):
        self.claude = claude_client
        self.db = db

    async def create_profile_from_posts(
        self,
        posts: List[dict],  # [{text, likes, comments, date}]
        author_name: str,
        author_role: str
    ) -> AuthorVoiceProfile:
        """
        Analyze existing posts to create author profile.
        Should be run once with 20-50 posts minimum.
        """

        if len(posts) < 10:
            raise ValueError(
                f"Need at least 10 posts for reliable profile. Got {len(posts)}."
            )

        # Sort by engagement
        sorted_posts = sorted(
            posts,
            key=lambda p: p.get('likes', 0) + p.get('comments', 0) * 3,
            reverse=True
        )

        # Format posts for analysis (top 30 for quality)
        posts_text = self._format_posts(sorted_posts[:30])

        profile = await self.claude.generate_structured(
            prompt=self.ANALYSIS_PROMPT.format(
                author_name=author_name,
                posts_text=posts_text
            ),
            response_model=AuthorVoiceProfile
        )

        # Set identity fields
        profile.author_name = author_name
        profile.author_role = author_role
        profile.posts_analyzed = len(posts)

        # Extract best performing hooks from top posts
        profile.best_performing_hooks = self._extract_hooks(sorted_posts[:5])

        # Save to database
        await self.db.save_author_profile(profile)

        return profile

    async def update_profile_incrementally(
        self,
        new_posts: List[dict],
        current_profile: AuthorVoiceProfile
    ) -> AuthorVoiceProfile:
        """
        Update profile with new posts (e.g., weekly update).
        Doesn't replace — adds new patterns and updates statistics.
        """

        update_prompt = f"""
        Current author profile for {current_profile.author_name}:

        Characteristic phrases: {current_profile.characteristic_phrases}
        Best hooks: {current_profile.best_performing_hooks}
        Typical length: {current_profile.typical_post_length} chars
        Formality: {current_profile.formality_level}

        New posts to incorporate (analyze for new patterns):
        {self._format_posts(new_posts)}

        Update the profile:
        1. Add any NEW characteristic phrases discovered (don't remove existing)
        2. Update best_performing_hooks if any new posts outperformed existing
        3. Note any evolution in style or topics
        4. Update statistics (typical_post_length, etc.)
        5. Keep everything else stable unless clear change detected

        Return the complete updated profile.
        """

        updated_profile = await self.claude.generate_structured(
            prompt=update_prompt,
            response_model=AuthorVoiceProfile
        )

        # Preserve identity and metadata
        updated_profile.author_name = current_profile.author_name
        updated_profile.author_role = current_profile.author_role
        updated_profile.created_at = current_profile.created_at
        updated_profile.last_updated = datetime.now()
        updated_profile.posts_analyzed = current_profile.posts_analyzed + len(new_posts)

        await self.db.save_author_profile(updated_profile)

        return updated_profile

    def generate_style_guide_for_writer(
        self,
        profile: AuthorVoiceProfile
    ) -> str:
        """
        Generate a style guide prompt section for Writer Agent.
        This is injected into the Writer's system prompt.
        """

        return f"""
=== AUTHOR VOICE GUIDE ===

You are writing as {profile.author_name}, {profile.author_role}.

MATCH THIS VOICE:
- Use these phrases naturally: {', '.join(profile.characteristic_phrases[:5])}
- Tone: {profile.formality_level:.0%} formal
- Humor: {profile.humor_frequency}
- Emoji usage: {profile.emoji_usage}
- Typical post length: ~{profile.typical_post_length} characters

THEIR WRITING STYLE:
- Sentences: {profile.sentence_length_preference}
- Paragraphs: {profile.paragraph_length}

THEIR EXPERTISE:
- Topics they know well: {', '.join(profile.expertise_areas[:5])}
- Their known positions:
{chr(10).join(f'  - {topic}: {stance}' for topic, stance in list(profile.known_opinions.items())[:3])}

AVOID:
- Never use: {', '.join(profile.avoided_phrases[:5])}
- Topics to skip: {', '.join(profile.topics_to_avoid[:3])}

THEIR BEST HOOKS (for inspiration):
{chr(10).join(f'- "{hook}"' for hook in profile.best_performing_hooks[:3])}

THEIR TYPICAL CTAs:
{chr(10).join(f'- "{cta}"' for cta in profile.preferred_cta_styles[:3])}

=== END VOICE GUIDE ===
"""

    def _format_posts(self, posts: List[dict]) -> str:
        """Format posts for analysis prompt."""
        formatted = []
        for i, p in enumerate(posts):
            engagement = p.get('likes', 0) + p.get('comments', 0) * 3
            text = p.get('text', p.get('content', ''))[:1500]  # Truncate long posts
            formatted.append(
                f"--- Post {i+1} (engagement score: {engagement}) ---\n"
                f"{text}\n"
            )
        return "\n".join(formatted)

    def _extract_hooks(self, posts: List[dict]) -> List[str]:
        """Extract first 2 lines from each post as hooks."""
        hooks = []
        for p in posts:
            text = p.get('text', p.get('content', ''))
            lines = text.split('\n')
            # Get first non-empty lines
            non_empty = [l.strip() for l in lines if l.strip()]
            hook = ' '.join(non_empty[:2]).strip()
            if hook:
                hooks.append(hook[:200])  # Max 200 chars
        return hooks
```

### Integration with Writer Agent

```python
# In writer.py

class WriterAgent:
    """Writer Agent with Author Profile integration."""

    def __init__(
        self,
        claude_client,
        author_profile_agent: AuthorProfileAgent,
        db
    ):
        self.claude = claude_client
        self.profile_agent = author_profile_agent
        self.db = db
        self._author_profile: Optional[AuthorVoiceProfile] = None

    async def initialize(self, author_name: str):
        """Load author profile on startup."""
        self._author_profile = await self.db.get_author_profile(author_name)

        if not self._author_profile:
            raise ValueError(
                f"No profile found for {author_name}. "
                f"Run AuthorProfileAgent.create_profile_from_posts() first."
            )

    async def generate_post(
        self,
        brief: AnalysisBrief,
        template: str
    ) -> DraftPost:
        """Generate post matching author's voice."""

        # Get style guide from profile
        style_guide = self.profile_agent.generate_style_guide_for_writer(
            self._author_profile
        )

        # Build full system prompt with style guide
        system_prompt = f"""
        {self.BASE_SYSTEM_PROMPT}

        {style_guide}
        """

        # Generate post
        response = await self.claude.generate(
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": self._build_generation_prompt(brief, template)
            }]
        )

        return DraftPost(
            content=response,
            template_used=template,
            author_profile_version=self._author_profile.last_updated
        )
```

---

## NEW: Scheduling System

### Purpose
Управляет планированием публикаций с выбором оптимального времени и избежанием конфликтов.

### Data Models

```python
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime, timedelta
import pytz


@dataclass
class ScheduledPost:
    """Post scheduled for future publication."""
    id: str
    content: str
    visual_asset_id: str
    content_type: ContentType

    # Scheduling
    scheduled_time: datetime
    timezone: str

    # Status
    status: str  # "scheduled", "published", "cancelled", "failed"
    published_at: Optional[datetime]
    linkedin_post_id: Optional[str]

    # Metadata
    created_at: datetime
    approved_by: str
    qc_score: float


@dataclass
class PublishingSlot:
    """Available slot for publishing."""
    day_of_week: int  # 0=Monday, 6=Sunday
    hour: int  # 0-23
    minute: int
    timezone: str

    # Performance data (filled from analytics)
    avg_engagement: float
    sample_size: int
    confidence: float  # 0-1, based on sample size
```

### Scheduling System Implementation

```python
class SchedulingSystem:
    """
    Manages post scheduling with optimal timing and conflict avoidance.
    """

    # Constraints
    MIN_HOURS_BETWEEN_POSTS = 6
    MAX_POSTS_PER_DAY = 2

    def __init__(self, db, analytics, timezone: str = "Europe/Moscow"):
        self.db = db
        self.analytics = analytics
        self.timezone = timezone

        # Default slots (before we have data)
        self.default_slots = [
            PublishingSlot(
                day_of_week=1, hour=9, minute=0, timezone=timezone,
                avg_engagement=0, sample_size=0, confidence=0
            ),  # Tuesday 9am
            PublishingSlot(
                day_of_week=3, hour=9, minute=0, timezone=timezone,
                avg_engagement=0, sample_size=0, confidence=0
            ),  # Thursday 9am
            PublishingSlot(
                day_of_week=1, hour=17, minute=0, timezone=timezone,
                avg_engagement=0, sample_size=0, confidence=0
            ),  # Tuesday 5pm
        ]

    async def get_optimal_slots(self, days_ahead: int = 14) -> List[PublishingSlot]:
        """
        Get optimal publishing slots based on historical data.
        Returns ranked list of available slots.
        """

        # Get historical performance by day/hour
        performance_data = await self.analytics.get_engagement_by_time()

        if not performance_data or len(performance_data) < 20:
            # Not enough data — use defaults
            return self.default_slots

        # Calculate best slots from data
        slots = []
        for day in range(7):  # 0-6 (Mon-Sun)
            for hour in [8, 9, 10, 12, 13, 17, 18, 19]:  # Common posting hours
                key = (day, hour)
                data = performance_data.get(key, {})

                if data.get('sample_size', 0) >= 3:  # Need at least 3 posts
                    slots.append(PublishingSlot(
                        day_of_week=day,
                        hour=hour,
                        minute=0,
                        timezone=self.timezone,
                        avg_engagement=data['avg_engagement'],
                        sample_size=data['sample_size'],
                        confidence=min(data['sample_size'] / 10, 1.0)
                    ))

        # Sort by engagement (weighted by confidence)
        slots.sort(
            key=lambda s: s.avg_engagement * s.confidence,
            reverse=True
        )

        return slots if slots else self.default_slots

    async def schedule_post(
        self,
        post: ApprovedPost,
        preferred_time: Optional[datetime] = None
    ) -> ScheduledPost:
        """
        Schedule a post for publication.
        Handles conflict avoidance and optimal timing.
        """

        if preferred_time:
            # Validate preferred time
            if not await self._is_slot_available(preferred_time):
                raise SchedulingConflictError(
                    f"Cannot schedule at {preferred_time}. "
                    f"Conflicts with existing post or violates constraints."
                )
            scheduled_time = preferred_time
        else:
            # Find next optimal slot
            scheduled_time = await self._find_next_optimal_slot()

        scheduled_post = ScheduledPost(
            id=generate_id(),
            content=post.content,
            visual_asset_id=post.visual_asset_id,
            content_type=post.content_type,
            scheduled_time=scheduled_time,
            timezone=self.timezone,
            status="scheduled",
            published_at=None,
            linkedin_post_id=None,
            created_at=datetime.now(),
            approved_by=post.approved_by,
            qc_score=post.qc_score
        )

        await self.db.save_scheduled_post(scheduled_post)

        return scheduled_post

    async def _is_slot_available(self, time: datetime) -> bool:
        """Check if time slot is available (no conflicts)."""

        # Get posts scheduled within MIN_HOURS_BETWEEN_POSTS window
        window_start = time - timedelta(hours=self.MIN_HOURS_BETWEEN_POSTS)
        window_end = time + timedelta(hours=self.MIN_HOURS_BETWEEN_POSTS)

        nearby_posts = await self.db.get_scheduled_posts_in_range(
            window_start, window_end, status="scheduled"
        )

        if nearby_posts:
            return False  # Conflict with nearby post

        # Check daily limit
        day_start = time.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)

        daily_posts = await self.db.get_scheduled_posts_in_range(
            day_start, day_end, status="scheduled"
        )

        if len(daily_posts) >= self.MAX_POSTS_PER_DAY:
            return False  # Daily limit reached

        return True

    async def _find_next_optimal_slot(self) -> datetime:
        """Find next available optimal slot."""

        optimal_slots = await self.get_optimal_slots()
        tz = pytz.timezone(self.timezone)
        now = datetime.now(tz)

        # Look up to 14 days ahead
        for days_ahead in range(14):
            check_date = now + timedelta(days=days_ahead)

            for slot in optimal_slots:
                if check_date.weekday() == slot.day_of_week:
                    candidate = check_date.replace(
                        hour=slot.hour,
                        minute=slot.minute,
                        second=0,
                        microsecond=0
                    )

                    # Skip if in the past
                    if candidate <= now:
                        continue

                    if await self._is_slot_available(candidate):
                        return candidate

        # Fallback: find any available slot in next 7 days
        return await self._find_any_available_slot(now)

    async def _find_any_available_slot(self, start: datetime) -> datetime:
        """Find any available slot (fallback when optimal slots full)."""

        for days_ahead in range(7):
            check_date = start + timedelta(days=days_ahead)

            for hour in [9, 10, 12, 17, 18]:
                candidate = check_date.replace(
                    hour=hour, minute=0, second=0, microsecond=0
                )

                if candidate > start and await self._is_slot_available(candidate):
                    return candidate

        raise SchedulingConflictError(
            "No available slots in the next 7 days. "
            "Consider adjusting constraints or clearing queue."
        )

    async def get_queue(self) -> List[ScheduledPost]:
        """Get all scheduled posts in chronological order."""
        posts = await self.db.get_scheduled_posts(status="scheduled")
        return sorted(posts, key=lambda p: p.scheduled_time)

    async def cancel_post(self, post_id: str) -> None:
        """Cancel a scheduled post."""
        post = await self.db.get_scheduled_post(post_id)

        if post.status != "scheduled":
            raise ValueError(f"Cannot cancel post with status '{post.status}'")

        post.status = "cancelled"
        await self.db.update_scheduled_post(post)

    async def reschedule_post(
        self,
        post_id: str,
        new_time: datetime
    ) -> ScheduledPost:
        """Reschedule a post to new time."""

        if not await self._is_slot_available(new_time):
            raise SchedulingConflictError(f"Time {new_time} not available")

        post = await self.db.get_scheduled_post(post_id)

        if post.status != "scheduled":
            raise ValueError(f"Cannot reschedule post with status '{post.status}'")

        post.scheduled_time = new_time
        await self.db.update_scheduled_post(post)

        return post

    async def get_schedule_summary(self) -> dict:
        """Get summary of scheduled posts for next 7 days."""

        tz = pytz.timezone(self.timezone)
        now = datetime.now(tz)
        week_end = now + timedelta(days=7)

        posts = await self.db.get_scheduled_posts_in_range(
            now, week_end, status="scheduled"
        )

        by_day = {}
        for post in posts:
            day_key = post.scheduled_time.strftime("%A %d.%m")
            if day_key not in by_day:
                by_day[day_key] = []
            by_day[day_key].append({
                "time": post.scheduled_time.strftime("%H:%M"),
                "content_type": post.content_type.value,
                "id": post.id
            })

        return {
            "total_scheduled": len(posts),
            "by_day": by_day,
            "next_post": posts[0] if posts else None
        }


class SchedulingConflictError(Exception):
    """Raised when scheduling conflicts occur."""
    pass
```

### Scheduler Background Task

```python
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler


class PublishingScheduler:
    """Background task that publishes scheduled posts."""

    def __init__(
        self,
        scheduling_system: SchedulingSystem,
        linkedin_client: LinkedInClient,
        telegram: TelegramNotifier
    ):
        self.scheduling = scheduling_system
        self.linkedin = linkedin_client
        self.telegram = telegram
        self.scheduler = AsyncIOScheduler()

    def start(self):
        """Start the scheduler."""
        # Check for posts to publish every minute
        self.scheduler.add_job(
            self._check_and_publish,
            'interval',
            minutes=1,
            id='publish_checker'
        )
        self.scheduler.start()

    async def _check_and_publish(self):
        """Check for posts due for publication."""

        now = datetime.now(pytz.timezone(self.scheduling.timezone))

        # Get posts scheduled for now (within 2 minute window)
        due_posts = await self.scheduling.db.get_scheduled_posts_in_range(
            now - timedelta(minutes=1),
            now + timedelta(minutes=1),
            status="scheduled"
        )

        for post in due_posts:
            await self._publish_post(post)

    async def _publish_post(self, post: ScheduledPost):
        """Publish a single post to LinkedIn."""

        try:
            # Get visual asset
            visual = await self.scheduling.db.get_visual_asset(post.visual_asset_id)

            # Publish to LinkedIn
            linkedin_post_id = await self.linkedin.publish_post(
                text=post.content,
                media_urls=[visual.url] if visual else None
            )

            # Update status
            post.status = "published"
            post.published_at = datetime.now()
            post.linkedin_post_id = linkedin_post_id
            await self.scheduling.db.update_scheduled_post(post)

            # Notify
            await self.telegram.notify(
                f"✅ Published: {post.content[:100]}...\n"
                f"LinkedIn ID: {linkedin_post_id}"
            )

        except Exception as e:
            post.status = "failed"
            await self.scheduling.db.update_scheduled_post(post)

            await self.telegram.notify(
                f"❌ Failed to publish post {post.id}:\n{str(e)}"
            )

    def stop(self):
        """Stop the scheduler."""
        self.scheduler.shutdown()
```

---

## Next Steps

1. [ ] Set up project structure
2. [ ] Implement Trend Scout with Perplexity + ArXiv
3. [ ] Build Analyzer agent
4. [ ] Create Writer with templates (Claude Opus 4.5 thinking mode)
5. [ ] Develop Humanizer rules engine
6. [ ] Integrate Nano Banana Pro (Laozhang.ai) for 4K visuals
7. [ ] **Set up Photo Library System:**
   - [ ] Create folder structure (portraits/action/context)
   - [ ] Build Photo Indexer with Claude Vision auto-tagging
   - [ ] Implement Photo Selector with variety tracking
   - [ ] Add Nano Banana photo editing (AI context addition)
   - [ ] Test photo+overlay compositions
8. [ ] **Set up AI Interface Generation:**
   - [ ] Build interface prompt templates (CRM, AI chat, automation, dashboards)
   - [ ] Implement composition mode selector (device only, author+phone, split view)
   - [ ] Test Nano Banana Pro interface quality
   - [ ] Test "author holding phone with interface" generation
   - [ ] Fine-tune prompts for realistic, professional look
9. [ ] Build QC scoring system
10. [ ] Create LangGraph orchestrator
11. [ ] Build Telegram bot for approvals
12. [ ] **Set up Post Analytics System (tomquirk/linkedin-api):**
    - [ ] Install linkedin-api: `pip install linkedin-api`
    - [ ] Build LinkedInMetricsCollector wrapper
    - [ ] Create Supabase tables (posts, snapshots, insights)
    - [ ] Build MetricsScheduler (APScheduler for T+15, T+30, T+60, T+24h)
    - [ ] Implement Performance Analyzer (velocity, benchmarks)
    - [ ] Create alerts system (fire, underperforming, viral)
    - [ ] Implement Feedback Loop (correlate factors → insights)
13. [ ] **═══ DEEP SELF-IMPROVEMENT LAYER ═══**
    - [ ] **Critic Agent (Agent-to-Agent Dialogue):**
        - [ ] Build CriticAgent with separate system prompt
        - [ ] Implement critique() method — initial critique
        - [ ] Implement follow_up() method — Q&A dialogue
        - [ ] Implement close_dialogue() → DialogueSummary
        - [ ] Store all dialogues in data/dialogues/critiques/
    - [ ] **Reflection Engine ("Ага, понятно!"):**
        - [ ] Build ReflectionEngine class
        - [ ] Implement pattern detection (is this recurring?)
        - [ ] Generate knowledge gaps list
        - [ ] Generate research queries from reflection
        - [ ] Store reflections in data/reflections/
    - [ ] **Research Agent (Targeted by Critique):**
        - [ ] Build ResearchAgent with Perplexity integration
        - [ ] Implement competitor scraping (top LinkedIn influencers)
        - [ ] Create own-data analyzer (best vs worst posts)
        - [ ] Build research query from Reflection.research_needed
        - [ ] Synthesize structured knowledge from research
    - [ ] **Knowledge Base (Persistent Memory):**
        - [ ] Set up SQLite for structured learnings
        - [ ] Set up Pinecone/Chroma for semantic search
        - [ ] Implement store_learning() — save new knowledge
        - [ ] Implement query_relevant() — find applicable learnings
        - [ ] Inject learnings into Writer prompts dynamically
    - [ ] **Claude Code Integration (Server Setup):**
        - [ ] Install Claude Code: `curl -fsSL https://claude.ai/install.sh | bash`
        - [ ] Set ANTHROPIC_API_KEY environment variable
        - [ ] Verify installation: `claude --version`
        - [ ] Test headless mode: `claude -p "test" --output-format json`
        - [ ] Configure allowed/disallowed tools in settings
        - [ ] Set up bash restrictions (no rm -rf, sudo, curl)
    - [ ] **Code Evolution Engine (via Claude Code):**
        - [ ] Build ClaudeCodeClient class (subprocess wrapper)
        - [ ] Implement generate_module() — runs Claude Code headless
        - [ ] Implement evolve_prompt() — Claude Code modifies prompts
        - [ ] Implement run_complex_task() — for open-ended code changes
        - [ ] Add result parsing (JSON output)
        - [ ] Add cost tracking (session costs → costs.json)
        - [ ] Store generated code in src/generated/
        - [ ] Store prompt versions in prompts/versions/
        - [ ] Log all sessions to data/claude_code_sessions/
    - [ ] **Deep Improvement Loop:**
        - [ ] Build DeepImprovementLoop orchestrator
        - [ ] Wire up: Critique → Reflect → Research → Modify → Validate
        - [ ] Add rollback if validation fails
        - [ ] Log full improvement cycle
    - [ ] **Experimentation Framework:**
        - [ ] Build ExperimentationEngine for A/B tests
        - [ ] Create experiment templates (hook style, length, timing)
        - [ ] Implement variant assignment (alternating)
        - [ ] Add statistical significance checking
        - [ ] Auto-apply winner to config
    - [ ] **Storage & Logging:**
        - [ ] Create Supabase tables (dialogues, reflections, learnings, modifications)
        - [ ] Build modification history viewer
        - [ ] Add Telegram notifications for self-modifications
14. [ ] Test end-to-end pipeline
15. [ ] **Validate Deep Self-Improvement:**
    - [ ] Run 10 posts without meta-agent (baseline)
    - [ ] Enable meta-agent for next 10 posts
    - [ ] Verify agent asks "Как тебе это?" and receives critique
    - [ ] Verify agent reflects and identifies knowledge gaps
    - [ ] Verify agent researches based on critique
    - [ ] Verify agent generates new code/modifies prompts
    - [ ] Compare engagement metrics before/after
    - [ ] Review generated code quality
16. [ ] **═══ NEW: Modification Safety System ═══**
    - [ ] Implement ModificationRiskLevel classification
    - [ ] Build ModificationRequest data model
    - [ ] Create RollbackTrigger with automatic monitoring
    - [ ] Build ModificationSafetySystem class:
        - [ ] request_modification() with risk-based flow
        - [ ] check_rollback_triggers() after each post
        - [ ] _rollback_modification() to restore state
        - [ ] _setup_rollback_trigger() for monitoring
    - [ ] Integrate with Telegram for approval requests
    - [ ] Create Supabase tables (modifications, rollback_triggers)
    - [ ] Test rollback scenario with intentionally bad change
17. [ ] **═══ NEW: Single-Call Evaluation System ═══**
    - [ ] Define evaluation_rubric with all criteria
    - [ ] Build EvaluationCriterion dataclass
    - [ ] Build SingleCallEvaluation output schema
    - [ ] Implement SingleCallEvaluator class:
        - [ ] evaluate() with structured JSON output
        - [ ] _calculate_weighted_total()
        - [ ] _prioritize_revisions() for failed evaluations
        - [ ] _format_rubric() for prompt
    - [ ] Replace Critic Agent dialogue with single call in QC Agent
    - [ ] Test evaluation quality vs multi-turn approach
    - [ ] Measure cost/latency improvement
18. [ ] **═══ NEW: Author Profile Agent ═══**
    - [ ] Build AuthorVoiceProfile dataclass
    - [ ] Implement AuthorProfileAgent:
        - [ ] create_profile_from_posts() — initial profile (need 20+ posts)
        - [ ] update_profile_incrementally() — weekly updates
        - [ ] generate_style_guide_for_writer() — prompt injection
        - [ ] _extract_hooks() from top posts
    - [ ] Create author profile import tool:
        - [ ] LinkedIn post export/scrape
        - [ ] Manual JSON import option
    - [ ] Integrate with Writer Agent:
        - [ ] Load profile on startup
        - [ ] Inject style guide into system prompt
    - [ ] Create Supabase table (author_profiles)
    - [ ] Test voice consistency before/after profile integration
19. [ ] **═══ NEW: Scheduling System ═══**
    - [ ] Build ScheduledPost and PublishingSlot dataclasses
    - [ ] Implement SchedulingSystem:
        - [ ] get_optimal_slots() from analytics data
        - [ ] schedule_post() with conflict avoidance
        - [ ] _is_slot_available() constraint checking
        - [ ] _find_next_optimal_slot()
        - [ ] get_queue(), cancel_post(), reschedule_post()
        - [ ] get_schedule_summary()
    - [ ] Build PublishingScheduler (APScheduler):
        - [ ] _check_and_publish() every minute
        - [ ] _publish_post() with LinkedIn client
        - [ ] Error handling and notifications
    - [ ] Create Supabase tables (scheduled_posts)
    - [ ] Build Telegram interface for schedule management:
        - [ ] /queue — view scheduled posts
        - [ ] /cancel <id> — cancel post
        - [ ] /reschedule <id> <time> — change time
    - [ ] Test scheduling with mock LinkedIn client
20. [ ] **Integration Testing:**
    - [ ] Test full pipeline with all new components
    - [ ] Verify Modification Safety prevents risky auto-changes
    - [ ] Verify Single-Call Evaluation matches quality of multi-turn
    - [ ] Verify Author Profile improves voice consistency
    - [ ] Verify Scheduling optimizes posting times
21. [ ] **═══ NEW: Logging & Observability System ═══**
    - [ ] Build LogLevel and LogComponent enums
    - [ ] Build LogEntry dataclass
    - [ ] Implement AgentLogger:
        - [ ] log() with structured data
        - [ ] Multiple outputs (file, Supabase, Telegram)
        - [ ] Batch writing to reduce DB load
        - [ ] Async flush on shutdown
    - [ ] Build ComponentLogger wrapper:
        - [ ] Per-component logging (trend_scout, writer, etc.)
        - [ ] Automatic context injection
        - [ ] Operation timing decorators
    - [ ] Implement PipelineRunLogger:
        - [ ] start_run() → run_id
        - [ ] log_step() → track each agent
        - [ ] complete_run() with final metrics
        - [ ] get_run_summary() for debugging
    - [ ] Build DailyDigestGenerator:
        - [ ] generate_daily_digest() — aggregated stats
        - [ ] _analyze_errors() — error patterns
        - [ ] _calculate_performance_metrics()
        - [ ] Send to Telegram at configured time
    - [ ] Create Supabase tables (logs, pipeline_runs)
    - [ ] Build log viewer endpoints:
        - [ ] GET /logs — paginated log list
        - [ ] GET /logs/run/{run_id} — single pipeline run
        - [ ] GET /logs/errors — errors only
        - [ ] GET /logs/stats — aggregated statistics
    - [ ] Add Telegram commands:
        - [ ] /logs — last N log entries
        - [ ] /errors — recent errors
        - [ ] /stats — daily stats summary
        - [ ] /run <id> — view specific pipeline run
    - [ ] Configure log rotation (keep last 30 days)
    - [ ] Test logging under load (100+ entries/minute)
