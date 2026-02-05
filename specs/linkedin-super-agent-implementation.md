# Plan: LinkedIn Super Agent — Full Implementation from Architecture Spec

## Task Description
Implement the entire LinkedIn Super Agent system from scratch based on the comprehensive 24,000+ line `architecture.md` specification. This is a greenfield project — no source code exists yet. The system is a multi-agent AI pipeline that autonomously creates, evaluates, publishes, and optimizes LinkedIn posts about AI topics, with self-improvement capabilities including self-modifying code.

## Objective
Build a fully functional, production-ready LinkedIn Super Agent with:
- 7 specialized agents (Trend Scout, Analyzer, Writer, Humanizer, Visual Creator, Photo Selector, QC Agent)
- LangGraph state machine orchestrator
- Self-improvement meta-agent with code generation capabilities
- Post analytics and feedback loop
- Scheduling and publishing system
- Telegram bot for human-in-the-loop approval
- Supabase database for all persistence
- Comprehensive logging and observability

## Problem Statement
The `architecture.md` contains an exhaustive specification for a LinkedIn content automation system, but zero implementation exists. The `src/` directory is empty. All code, configuration, database schemas, prompts, and project infrastructure must be built from the architecture document.

## Solution Approach
Implement in **7 phases**, starting with foundational infrastructure (project setup, models, database) and building outward to agents, orchestrator, meta-agent, and advanced features. Each phase produces a testable, runnable increment.

**Key Architectural Decisions (from architecture.md):**
- Python + LangGraph for orchestration
- Claude Opus 4.5 (thinking mode) as primary LLM
- Supabase as sole database (no Redis, SQLite, MongoDB)
- Fail-fast error handling (no fallbacks between services)
- Retry with exponential backoff for transient errors
- Configurable autonomy levels (1-4)
- Self-modifying code via Claude Code CLI

## Relevant Files

### Existing Files
- `architecture.md` — The 24K+ line specification document (THE source of truth)
  - Lines 1-124: Overview, error handling philosophy
  - Lines 127-2984: Supabase client, database schema, custom exceptions
  - Lines 2985-3101: System architecture diagram
  - Lines 3102-11138: Agent specifications (Trend Scout, Analyzer, Writer, Humanizer, Visual Creator, Photo Selector, QC)
  - Lines 11139-13176: Orchestrator (LangGraph state machine, PipelineState)
  - Lines 13177-14304: Post Analytics & Feedback Loop
  - Lines 14305-15480: Self-Improvement Layer (Meta-Agent)
  - Lines 15481-21289: Self-Modifying Code Engine
  - Lines 21290-21456: Technical Stack & Project Structure
  - Lines 21460-21488: Research Sources
  - Lines 21490-22437: Logging & Observability System
  - Lines 22438-22936: Modification Safety System
  - Lines 22937-23466: Single-Call Evaluation System
  - Lines 23467-23824: Author Profile Agent
  - Lines 23825-24315: Scheduling System
  - Lines 24316-24513: Next Steps checklist

### New Files to Create

#### Project Root
- `pyproject.toml` — Project configuration, dependencies (uv-managed)
- `.env.example` — Environment variable template
- `.gitignore` — Git ignore rules
- `README.md` — Project documentation (brief)

#### Source Code (`src/`)
- `src/__init__.py`
- `src/models.py` — All shared data types (ContentType, TrendTopic, PipelineState, etc.)
- `src/config.py` — Centralized configuration loader (ThresholdConfig, settings)
- `src/exceptions.py` — Custom exception classes
- `src/utils.py` — Shared utilities (utc_now, generate_id, ensure_utc, retry decorator)
- `src/database.py` — SupabaseDB async client (unified database access)

#### Agents (`src/agents/`)
- `src/agents/__init__.py`
- `src/agents/trend_scout.py` — Trend mining from multiple sources
- `src/agents/analyzer.py` — Content analysis with type-specific extraction
- `src/agents/writer.py` — Post generation with templates and content-type awareness
- `src/agents/humanizer.py` — AI detection avoidance, tone humanization
- `src/agents/visual_creator.py` — Image generation via Nano Banana Pro
- `src/agents/photo_selector.py` — Personal photo selection with variety tracking
- `src/agents/qc_agent.py` — Quality control scoring
- `src/agents/orchestrator.py` — LangGraph state machine

#### Meta-Agent (`src/meta_agent/`)
- `src/meta_agent/__init__.py`
- `src/meta_agent/meta_agent.py` — Main self-improvement orchestrator
- `src/meta_agent/single_call_evaluator.py` — Single-call LLM evaluation
- `src/meta_agent/modification_safety.py` — Risk-based approval + rollback
- `src/meta_agent/reflection_engine.py` — Pattern detection and insight extraction
- `src/meta_agent/research_agent.py` — Perplexity + competitor analysis
- `src/meta_agent/claude_code_client.py` — Claude Code CLI wrapper
- `src/meta_agent/code_evolution.py` — Code generation engine
- `src/meta_agent/knowledge_base.py` — Persistent memory
- `src/meta_agent/deep_improvement_loop.py` — Full improvement cycle
- `src/meta_agent/experimentation.py` — A/B testing framework
- `src/meta_agent/models.py` — Meta-agent data models

#### Author Profile (`src/author/`)
- `src/author/__init__.py`
- `src/author/author_profile_agent.py` — Voice profile creation/maintenance
- `src/author/profile_importer.py` — Import posts from LinkedIn/JSON
- `src/author/models.py` — AuthorVoiceProfile dataclass

#### Scheduling (`src/scheduling/`)
- `src/scheduling/__init__.py`
- `src/scheduling/scheduling_system.py` — Optimal timing + conflict avoidance
- `src/scheduling/publishing_scheduler.py` — Background task (APScheduler)
- `src/scheduling/models.py` — ScheduledPost, PublishingSlot

#### Logging (`src/logging/`)
- `src/logging/__init__.py`
- `src/logging/agent_logger.py` — Central logger with multiple outputs
- `src/logging/component_logger.py` — Per-component wrapper
- `src/logging/pipeline_run_logger.py` — Pipeline run tracking
- `src/logging/daily_digest.py` — Daily summary generator
- `src/logging/models.py` — LogEntry, LogLevel, LogComponent

#### Tools (`src/tools/`)
- `src/tools/__init__.py`
- `src/tools/perplexity.py` — Perplexity API wrapper
- `src/tools/arxiv.py` — ArXiv search
- `src/tools/twitter.py` — X API
- `src/tools/nano_banana.py` — Image generation (Laozhang.ai)
- `src/tools/linkedin_client.py` — tomquirk/linkedin-api wrapper
- `src/tools/photo_library.py` — Photo indexing & search
- `src/tools/claude_client.py` — Claude API client (for agent LLM calls)

#### Generated Code (`src/generated/`)
- `src/generated/__init__.py` — Auto-imports all generated modules
- `src/generated/README.md` — "This code is auto-generated by Meta-Agent"

#### UI (`src/ui/`)
- `src/ui/__init__.py`
- `src/ui/telegram_bot.py` — Human-in-the-loop Telegram interface

#### Knowledge (`src/knowledge/`)
- `src/knowledge/style_guide.md` — Tone of voice guidelines
- `src/knowledge/top_posts.json` — Examples of successful posts
- `src/knowledge/anti_patterns.json` — What to avoid

#### Config (`config/`)
- `config/settings.yaml` — Main configuration
- `config/meta_agent_config.yaml` — Self-improvement settings
- `config/scoring_weights.json` — Modifiable by meta-agent
- `config/hook_templates.json` — Modifiable by meta-agent
- `config/writer_config.json` — Modifiable by meta-agent
- `config/visual_styles.json` — Modifiable by meta-agent
- `config/schedule.json` — Modifiable by meta-agent

#### Prompts (`prompts/`)
- `prompts/writer_system.txt` — Writer agent system prompt
- `prompts/evaluator_system.txt` — SingleCallEvaluator system prompt
- `prompts/evaluator_criteria.txt` — Self-evaluation rubric
- `prompts/humanizer_rules.txt` — Humanization guidelines
- `prompts/versions/` — Prompt version history directory

#### Database Migrations (`migrations/`)
- `migrations/001_initial_schema.sql` — All Supabase tables

## Implementation Phases

### Phase 1: Foundation (Infrastructure & Models)
Set up project structure, install dependencies, define all data models, configure database client, implement shared utilities. This phase produces a runnable Python project with all types defined and database connectivity.

### Phase 2: Core Tools & External Integrations
Implement all API wrappers (Claude, Perplexity, ArXiv, LinkedIn, Nano Banana, Telegram). Each tool is independently testable. This phase establishes all external connectivity.

### Phase 3: Agent Pipeline (Scout → Analyze → Write → Humanize → Visual → QC)
Implement all 7 agents following the architecture spec. Each agent reads from and writes to PipelineState. This phase produces a linear content pipeline.

### Phase 4: Orchestrator (LangGraph State Machine)
Wire all agents into the LangGraph state machine with conditional routing, revision loops, error handling, and human approval flow. This phase produces an end-to-end pipeline.

### Phase 5: Analytics, Scheduling & Publishing
Implement post analytics collection, performance analysis, feedback loop, scheduling system, and LinkedIn publishing. This phase makes the system operational.

### Phase 6: Meta-Agent & Self-Improvement
Implement the self-improvement layer: single-call evaluator, modification safety, reflection engine, research agent, knowledge base, experimentation framework, and deep improvement loop.

### Phase 7: Self-Modifying Code Engine & Polish
Implement Claude Code CLI integration, code generation engine, capability analyzer, module registry, hot reloader. Final integration testing and polish.

## Step by Step Tasks

IMPORTANT: Execute every step in order, top to bottom.

### 1. Initialize Project Structure
- Create `pyproject.toml` with uv, specifying all dependencies:
  ```
  langgraph, langchain-anthropic, langchain-core
  supabase (async), aiohttp, aiofiles
  linkedin-api, pyotp
  perplexity-python (or httpx for API calls)
  arxiv
  python-telegram-bot
  apscheduler
  structlog
  pydantic (for structured outputs)
  pyyaml
  python-dotenv
  Pillow
  pytz
  ```
- Create `.env.example` with all required env vars:
  ```
  SUPABASE_URL=
  SUPABASE_SERVICE_KEY=
  ANTHROPIC_API_KEY=
  PERPLEXITY_API_KEY=
  LINKEDIN_EMAIL=
  LINKEDIN_PASSWORD=
  LINKEDIN_TOTP_SECRET=
  TELEGRAM_BOT_TOKEN=
  TELEGRAM_CHAT_ID=
  LAOZHANG_API_KEY=
  TWITTER_BEARER_TOKEN=
  ```
- Create `.gitignore` (Python, .env, __pycache__, logs/, data/, .venv/)
- Create all directory structures (`src/`, `src/agents/`, `src/meta_agent/`, etc.)
- Create all `__init__.py` files
- Run `uv sync` to install dependencies

### 2. Implement Shared Models (`src/models.py`)
- Define `ContentType` enum (5 types: ENTERPRISE_CASE, PRIMARY_SOURCE, AUTOMATION_CASE, COMMUNITY_CONTENT, TOOL_RELEASE)
- Define `TrendTopic` dataclass with all fields from architecture (title, url, source, content_type, quality_score, etc.)
- Define `TrendScoutOutput` dataclass
- Define `AnalysisBrief` and `TypeSpecificExtraction` dataclasses
- Define `DraftPost` and `WriterOutput` dataclasses
- Define `HumanizedPost` dataclass
- Define `VisualAsset` and `VisualCreatorOutput` dataclasses
- Define `QCResult` and `QCOutput` dataclasses
- Define `PostMetricsSnapshot` dataclass
- Define `PipelineState` TypedDict (the full state from architecture lines 11224-11352)
- All models must use timezone-aware datetimes via `utc_now()`

### 3. Implement Exceptions (`src/exceptions.py`)
- Extract all custom exceptions from architecture (lines 248-305):
  - `AgentBaseError`, `ValidationError`, `DatabaseError`, `ConfigurationError`
  - `SecurityError`, `RetryExhaustedError`
  - `LinkedInRateLimitError`, `LinkedInSessionExpiredError`, `LinkedInAPIError`
  - `ImageGenerationError`, `SchedulingConflictError`

### 4. Implement Utilities (`src/utils.py`)
- `utc_now()` — timezone-aware UTC datetime
- `generate_id()` — UUID4 string generator
- `ensure_utc(dt)` — convert datetime to UTC
- `@with_retry` decorator with exponential backoff (architecture lines 84-120)
  - `max_attempts`, `base_delay`, `retryable_exceptions`
  - Raises `RetryExhaustedError` after exhaustion

### 5. Implement Configuration (`src/config.py`)
- `ThresholdConfig` — centralized threshold management
- Load `config/settings.yaml` for global settings
- Load type-specific contexts (`load_type_context()` from architecture lines 11358-11438)
- Autonomy level configuration (levels 1-4)
- Domain-to-candidate-types mapping (architecture lines 3209-3257)
- URL pattern type hints (architecture lines 3260-3272)

### 6. Implement Database Client (`src/database.py`)
- `SupabaseConfig.from_env()` — load config from environment
- `SupabaseDB` class — unified async database client
- All CRUD operations:
  - Posts: `save_draft()`, `update_post_status()`, `get_post()`
  - Analytics: `save_metrics_snapshot()`, `get_recent_metrics()`
  - Modifications: `save_modification()`, `get_pending_modifications()`
  - Scheduled posts: `save_scheduled_post()`, `get_due_posts()`, `claim_post()`
  - Author profiles: `save_author_profile()`, `get_author_profile()`
  - Logs: `batch_insert_logs()`, `get_logs()`
  - Learnings: `save_learning()`, `query_relevant_learnings()`
  - Experiments: `save_experiment()`, `get_active_experiment()`

### 7. Create Database Migrations (`migrations/001_initial_schema.sql`)
- Tables: `posts`, `post_metrics_snapshots`, `pipeline_runs`, `trend_topics`
- Tables: `modifications`, `rollback_triggers`, `author_profiles`
- Tables: `scheduled_posts`, `agent_logs`, `learnings`, `experiments`
- Tables: `dialogues`, `reflections`, `code_evolution_history`
- All timestamps as `TIMESTAMPTZ`
- Proper indexes on frequently queried columns
- RLS policies for service role access

### 8. Implement Logging System (`src/logging/`)
- `src/logging/models.py` — `LogLevel` enum (DEBUG=10...CRITICAL=50), `LogComponent` enum, `LogEntry` dataclass
- `src/logging/agent_logger.py` — `AgentLogger` class:
  - Multiple outputs: file (JSON), Supabase, Telegram
  - Batch writing to reduce DB load
  - Async flush on shutdown
  - Context management (run_id, post_id)
- `src/logging/component_logger.py` — Per-component wrapper with auto context injection
- `src/logging/pipeline_run_logger.py` — `start_run()`, `log_step()`, `complete_run()`, `get_run_summary()`
- `src/logging/daily_digest.py` — `generate_daily_digest()` with stats aggregation

### 9. Implement Tool Wrappers (`src/tools/`)
- `src/tools/claude_client.py`:
  - Async Claude API client using `anthropic` SDK
  - `generate_text()`, `generate_structured()` methods
  - Token tracking, cost estimation
  - Thinking mode support for Opus 4.5
- `src/tools/perplexity.py`:
  - Async Perplexity search wrapper
  - `search()`, `research_topic()` methods
- `src/tools/arxiv.py`:
  - ArXiv paper search and metadata extraction
  - `search_papers()`, `get_paper_details()` methods
- `src/tools/nano_banana.py`:
  - Laozhang.ai image generation (Nano Banana Pro)
  - `generate_image()` method with prompt engineering for LinkedIn visuals
  - Support for different visual formats (metrics_card, workflow_diagram, etc.)
- `src/tools/linkedin_client.py`:
  - Wrapper around `tomquirk/linkedin-api`
  - 2FA support via pyotp
  - `publish_post()`, `get_post_metrics()`, `get_post_reactions()`, `get_post_comments()`
  - Rate limiting awareness
- `src/tools/photo_library.py`:
  - Photo indexing with Claude Vision auto-tagging
  - `search_photos()`, `get_random_unused()` methods
  - Usage tracking to avoid repetition

### 10. Implement Trend Scout Agent (`src/agents/trend_scout.py`)
- Multi-source scanning: Perplexity, ArXiv, HackerNews, Twitter, Reddit, Medium, Substack
- Two-level content classification (domain hint + LLM refinement)
- Pre-filtering with exclusion rules
- Cross-source deduplication
- Quality scoring with type-specific weights
- Top pick selection ("Most important case of the day")
- Returns `TrendScoutOutput` with ranked topics

### 11. Implement Analyzer Agent (`src/agents/analyzer.py`)
- Content-type-aware deep analysis
- Type-specific extraction (enterprise: company/metrics/ROI; research: thesis/methodology/findings; etc.)
- Key findings extraction
- Angle generation for LinkedIn post
- Returns `AnalysisBrief` with `TypeSpecificExtraction`

### 12. Implement Writer Agent (`src/agents/writer.py`)
- Template-based post generation (METRICS_HERO, LESSONS_LEARNED, HOW_TO_GUIDE, etc.)
- Content-type-aware template selection
- Hook style selection based on content type
- Author voice profile injection (when available)
- Learning injection from knowledge base
- Alternative generation (3 variants)
- Returns `WriterOutput` with `DraftPost`

### 13. Implement Humanizer Agent (`src/agents/humanizer.py`)
- Content-type-aware humanization intensity (low/medium/high)
- Replace AI-typical phrases with natural language
- Add personal touches, conversational markers
- Maintain factual accuracy while humanizing
- Emoji and formatting calibration by content type
- Returns `HumanizedPost`

### 14. Implement Visual Creator Agent (`src/agents/visual_creator.py`)
- Content-type-aware visual format selection
- Nano Banana Pro integration for 4K image generation
- Support for: metrics_card, workflow_diagram, quote_card, architecture_diagram, etc.
- Photo selection integration (when personal photo is appropriate)
- Visual brief generation from post content
- Returns `VisualCreatorOutput` with `VisualAsset`

### 15. Implement Photo Selector Agent (`src/agents/photo_selector.py`)
- Photo library indexing and search
- Context-appropriate photo selection (formal/casual/speaking/action)
- Variety tracking to avoid repetition
- Integration with Nano Banana for photo editing/overlay

### 16. Implement QC Agent (`src/agents/qc_agent.py`)
- Content-type-aware scoring criteria
- Weighted scoring system (hook_strength, specificity, value_density, authenticity, structure, cta_clarity, visual_quality)
- Type-specific weight adjustments
- Pass/revise/reject decision logic
- Revision routing (back to writer, humanizer, or visual creator)
- Returns `QCOutput` with `QCResult`

### 17. Implement LangGraph Orchestrator (`src/agents/orchestrator.py`)
- Define `StateGraph` with all nodes:
  - `scout` → `select_topic` → `analyze` → `write` → `meta_evaluate` → `humanize` → `visualize` → `qc`
- Conditional edges:
  - QC → PASS → `prepare_output`
  - QC → REVISE_WRITER → `write` (max 3 revisions)
  - QC → REVISE_HUMANIZER → `humanize`
  - QC → REJECT → `scout` (new topic, max 2 restarts)
- Meta-evaluation loop between Writer and Humanizer
- Error routing to `handle_error` node
- Human approval flow with Telegram integration
- Pipeline state initialization with `run_id`, `run_timestamp`
- Content type propagation through entire pipeline

### 18. Implement Telegram Bot (`src/ui/telegram_bot.py`)
- Post approval flow (approve/reject/edit)
- Schedule management (/queue, /cancel, /reschedule)
- Log viewing (/logs, /errors, /stats)
- Modification approval for meta-agent changes
- Status commands (/status, /pipeline)
- Alert notifications (fire, underperforming, viral posts)

### 19. Implement Analytics System (within `src/database.py` and `src/agents/`)
- `PostMetricsSnapshot` collection at scheduled intervals (T+15, T+30, T+60, T+3h, T+24h, T+48h)
- Performance analysis (velocity metrics, benchmark comparison)
- Pattern detection (best content_type, hook_style, visual_type)
- Feedback loop to adjust scoring weights and template preferences

### 20. Implement Scheduling System (`src/scheduling/`)
- `src/scheduling/models.py` — `ScheduledPost` (with PostStatus enum), `PublishingSlot`
- `src/scheduling/scheduling_system.py`:
  - `get_optimal_slots()` from analytics data
  - `schedule_post()` with conflict avoidance (MIN_HOURS_BETWEEN_POSTS=6, MAX_POSTS_PER_DAY=2)
  - Database-level locking for race condition prevention
- `src/scheduling/publishing_scheduler.py`:
  - APScheduler background task checking every minute
  - Atomic post claiming (status: scheduled → publishing → published/failed)
  - Error handling and notification

### 21. Implement Single-Call Evaluator (`src/meta_agent/single_call_evaluator.py`)
- Evaluation rubric (hook_strength, specificity, value_density, authenticity, structure, cta_clarity)
- Structured JSON output from single LLM call
- Weighted total calculation
- Pass/fail decision with threshold from ThresholdConfig
- Actionable revision suggestions
- Pattern detection for recurring issues

### 22. Implement Modification Safety System (`src/meta_agent/modification_safety.py`)
- `ModificationRiskLevel` enum (LOW, MEDIUM, HIGH, CRITICAL)
- `ModificationRequest` dataclass with full audit trail
- `RollbackTrigger` dataclass
- Risk-based approval flow:
  - LOW → auto-apply, monitor 5 posts
  - MEDIUM → auto-apply, stricter rollback, notify
  - HIGH → human approval required
  - CRITICAL → human approval + explicit confirmation
- Automatic rollback when performance degrades

### 23. Implement Author Profile Agent (`src/author/`)
- `AuthorVoiceProfile` dataclass (writing style, tone, opinions, patterns)
- `create_profile_from_posts()` — analyze 20+ existing posts
- `update_profile_incrementally()` — weekly updates
- `generate_style_guide_for_writer()` — prompt injection
- Profile import from LinkedIn/JSON

### 24. Implement Meta-Agent Orchestrator (`src/meta_agent/meta_agent.py`)
- Observation of post performance and pipeline results
- Self-evaluation flow (generate → evaluate → feedback → rewrite, max 3 iterations)
- Research triggering (3 underperforming posts, new content type, weekly cycle)
- Self-modification orchestration (prompt changes, config changes, code generation)
- Integration with ModificationSafetySystem

### 25. Implement Research Agent (`src/meta_agent/research_agent.py`)
- Perplexity-powered research on best practices
- Competitor analysis (scrape top LinkedIn influencers)
- Own data analysis (best vs worst posts comparison)
- Structured knowledge synthesis from research
- Integration with Reflection Engine

### 26. Implement Reflection Engine (`src/meta_agent/reflection_engine.py`)
- Pattern detection (is this issue recurring?)
- Knowledge gap identification
- Research query generation from reflection
- "Aga, I see!" moment tracking

### 27. Implement Knowledge Base (`src/meta_agent/knowledge_base.py`)
- Supabase-backed persistent learnings
- Semantic search for relevant learnings
- Learning injection into Writer prompts
- Learning versioning and effectiveness tracking

### 28. Implement Experimentation Framework (`src/meta_agent/experimentation.py`)
- A/B test design (hook styles, length, timing, visuals)
- Variant assignment (alternating)
- Statistical significance checking
- Auto-apply winner, auto-stop losing variant
- Max 1 active experiment at a time

### 29. Implement Deep Improvement Loop (`src/meta_agent/deep_improvement_loop.py`)
- Full cycle: Critique → Reflect → Research → Modify → Validate
- Rollback if validation fails
- Logging of full improvement cycles
- Integration with all meta-agent components

### 30. Implement Self-Modifying Code Engine (`src/meta_agent/code_evolution.py`)
- `CapabilityAnalyzer` — detect missing capabilities from evaluation feedback
- `CodeGenerationEngine` — generate new Python modules via Claude Code CLI
- `CodeValidator` — syntax check, type check (non-blocking), sandbox test, security scan
- `ModuleRegistry` — track all generated modules
- `HotReloader` — `importlib.reload` for generated modules
- Store generated code in `src/generated/`

### 31. Implement Claude Code Client (`src/meta_agent/claude_code_client.py`)
- Subprocess wrapper for Claude Code CLI (headless mode)
- `generate_module()`, `evolve_prompt()`, `run_complex_task()`
- JSON output parsing
- Cost tracking (session costs → costs.json)
- Allowed/disallowed tools configuration
- Bash restrictions (no rm -rf, sudo, curl)

### 32. Create Configuration Files
- `config/settings.yaml` — global settings (autonomy_level, llm_model, timezone, etc.)
- `config/meta_agent_config.yaml` — self-improvement settings (thresholds, intervals, limits)
- `config/scoring_weights.json` — initial scoring weights for Trend Scout
- `config/hook_templates.json` — hook template definitions by content type
- `config/writer_config.json` — writer preferences (length, style, format)
- `config/visual_styles.json` — visual format preferences
- `config/schedule.json` — default posting schedule

### 33. Create Prompt Files
- `prompts/writer_system.txt` — Writer agent system prompt with content-type awareness
- `prompts/evaluator_system.txt` — SingleCallEvaluator system prompt
- `prompts/evaluator_criteria.txt` — Full evaluation rubric
- `prompts/humanizer_rules.txt` — Humanization guidelines and rules

### 34. Create Knowledge Files
- `src/knowledge/style_guide.md` — Author's tone of voice guidelines (placeholder)
- `src/knowledge/top_posts.json` — Examples of successful posts (placeholder)
- `src/knowledge/anti_patterns.json` — Patterns to avoid (placeholder)

### 35. Integration Testing & Validation
- Test database connectivity and all CRUD operations
- Test each tool wrapper independently
- Test each agent in isolation with mock inputs
- Test the full pipeline end-to-end with a mock topic
- Test Telegram bot approval flow
- Test scheduling and publishing flow
- Test meta-agent evaluation and modification flow
- Validate error handling (fail-fast behavior)
- Validate retry logic with transient failures

## Testing Strategy

### Unit Tests
- Each model validation (proper types, required fields, timezone awareness)
- Each utility function (utc_now, generate_id, retry decorator)
- Database client CRUD operations (with mock Supabase)
- Each tool wrapper (with mocked external APIs)
- Each agent (with mocked tool calls and LLM responses)

### Integration Tests
- Full pipeline: Scout → Analyze → Write → Humanize → Visual → QC → Approve
- Revision loop: QC fails → Writer rewrites → QC passes
- Rejection loop: QC rejects → New topic → Full pipeline
- Meta-evaluation loop: Score < 8 → Feedback → Rewrite → Score >= 8
- Scheduling: Schedule → Publish → Collect metrics → Analyze

### End-to-End Tests
- Run 10 posts without meta-agent (baseline)
- Enable meta-agent for next 10 posts
- Verify self-modification works correctly
- Compare engagement metrics

## Acceptance Criteria

1. `uv run python -m py_compile src/**/*.py` — All source files compile without errors
2. `uv run python -c "from src.models import PipelineState, ContentType"` — Models importable
3. `uv run python -c "from src.agents.orchestrator import create_pipeline"` — Orchestrator importable
4. `uv run python -c "from src.database import SupabaseDB"` — Database client importable
5. All 7 agents can be instantiated and have their main methods callable
6. LangGraph state machine can be compiled and has correct node/edge structure
7. Telegram bot can start and respond to commands
8. Scheduling system can queue and claim posts
9. Meta-agent can evaluate a post and produce structured feedback
10. Self-modifying code engine can generate, validate, and load a new module
11. Full pipeline can run end-to-end from topic selection to approved post

## Validation Commands

Execute these commands to validate the task is complete:

- `uv sync` — Install all dependencies
- `uv run python -m py_compile src/models.py` — Compile models
- `uv run python -m py_compile src/database.py` — Compile database client
- `uv run python -m py_compile src/agents/orchestrator.py` — Compile orchestrator
- `uv run python -c "from src.models import ContentType, TrendTopic, PipelineState, DraftPost, QCResult; print('Models OK')"` — Validate model imports
- `uv run python -c "from src.agents.orchestrator import create_pipeline; print('Orchestrator OK')"` — Validate orchestrator
- `uv run python -c "from src.logging.agent_logger import AgentLogger; print('Logger OK')"` — Validate logging
- `uv run python -c "from src.meta_agent.single_call_evaluator import SingleCallEvaluator; print('Evaluator OK')"` — Validate meta-agent
- `uv run python -c "from src.scheduling.scheduling_system import SchedulingSystem; print('Scheduler OK')"` — Validate scheduling

## Notes

### Dependencies (install via `uv add`)
```
uv add langgraph langchain-anthropic langchain-core
uv add supabase aiohttp aiofiles httpx
uv add linkedin-api pyotp
uv add arxiv
uv add python-telegram-bot apscheduler
uv add structlog pydantic pyyaml python-dotenv
uv add Pillow pytz
uv add anthropic
```

### Critical Design Decisions
1. **No fallbacks between services** — If Nano Banana fails, don't switch to DALL-E. Set critical_error.
2. **Retry then fail** — 3 retries with exponential backoff, then `RetryExhaustedError`.
3. **ContentType flows through entire pipeline** — Set at topic selection, informs all downstream decisions.
4. **Timezone-aware datetimes everywhere** — Use `utc_now()` instead of `datetime.now()`.
5. **Supabase is the only database** — No Redis, SQLite, MongoDB.
6. **Self-modifying code is validated** — Syntax check + tests + security scan before loading.
7. **Human approval via Telegram** — Configurable autonomy levels (1-4).

### Estimated File Count
- ~55 Python source files
- ~8 configuration files
- ~4 prompt files
- ~3 knowledge files
- ~1 SQL migration file
- Total: ~71 files to create

### Risk Areas
- **LinkedIn API** — Unofficial Voyager API may break or get rate-limited
- **Self-modifying code** — Security considerations for generated code
- **LLM costs** — Claude Opus 4.5 is expensive; monitor usage carefully
- **Supabase limits** — Batch writes and connection pooling needed for scale
