# LinkedIn Super Agent

Multi-agent AI system for autonomous LinkedIn content creation and optimization. Powered by LangGraph orchestration, Claude Opus 4.5 (thinking mode), and a self-improving meta-agent that writes its own code at runtime.

## Features

- **7 Specialized Agents** -- Trend Scout, Analyzer, Writer, Humanizer, Visual Creator, Photo Selector, QC
- **LangGraph State Machine** -- deterministic orchestration with conditional routing and retry loops
- **Self-Improvement Engine** -- meta-agent that modifies prompts, generates new Python modules, and hot-reloads them in the same run
- **Post Analytics** -- automated metrics collection at T+15min, T+30min, T+1h, T+24h with performance benchmarking
- **Configurable Autonomy (Levels 1-4)** -- from full human approval to fully autonomous publishing
- **Fail-Fast Error Philosophy** -- no fallbacks, no silent failures; retry with exponential backoff then route to error handler
- **Scheduling System** -- optimal posting time selection based on historical engagement data
- **Author Voice Profiling** -- learns your writing style from existing posts for consistent tone

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Orchestration | LangGraph |
| Primary LLM | Claude Opus 4.5 (thinking mode) |
| Database | Supabase (PostgreSQL) |
| Image Generation | Nano Banana Pro (Laozhang.ai) |
| Search / Research | Perplexity API |
| Academic Papers | ArXiv API |
| LinkedIn API | tomquirk/linkedin-api |
| Human-in-the-Loop | Telegram Bot |
| Scheduling | APScheduler |
| Package Manager | uv |

## Architecture Overview

```
                         SCHEDULER
                        (cron/manual)
                             |
  +----------------------------------------------------------+
  |                    ORCHESTRATOR                           |
  |                (LangGraph StateMachine)                   |
  |                                                          |
  |  SCOUT --> ANALYZE --> CREATE --> EVAL --> QC             |
  |    |          |          |         |       |              |
  |  [Trend]  [Analyzer] [Writer]  [Meta]   [QC]            |
  |                       [Visual]  Agent    Agent           |
  |                                   |                      |
  |                           Score < 8? --> REWRITE (3x)    |
  +----------------------------------------------------------+
                             |
              +--------------+--------------+
              |                             |
        Score >= 9.0                  Score < 9.0
        (auto-publish)              (human approval)
              |                             |
              +-------------+---------------+
                            |
                    LINKEDIN PUBLISHER
                            |
                    ANALYTICS COLLECTOR
                            |
                  SELF-IMPROVEMENT ENGINE
```

## Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- Supabase project (database)
- API keys for Claude, Perplexity, Nano Banana, and Telegram

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/personal-brand-agent.git
cd personal-brand-agent

# Install uv if you haven't already
pip install uv

# Install dependencies
uv sync

# Copy environment configuration
cp .env.example .env
```

### Environment Variables

Edit `.env` with your credentials:

```env
# LLM
ANTHROPIC_API_KEY=sk-ant-...

# Database
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=eyJ...

# Search & Research
PERPLEXITY_API_KEY=pplx-...

# Image Generation
NANO_BANANA_API_KEY=nb-...
NANO_BANANA_BASE_URL=https://api.laozhang.ai

# LinkedIn
LINKEDIN_USERNAME=your@email.com
LINKEDIN_PASSWORD=your-password

# Telegram (Human-in-the-Loop)
TELEGRAM_BOT_TOKEN=123456:ABC-...
TELEGRAM_CHAT_ID=your-chat-id

# Autonomy Level (1-4)
AUTONOMY_LEVEL=1
```

### Database Setup

Run the Supabase migration to create required tables:

```bash
# Apply database schema
uv run python -m src.tools.supabase_migrate
```

### Running

```bash
# Start the pipeline (single run)
uv run python -m src.agents.orchestrator

# Start the scheduling daemon (continuous)
uv run python -m src.scheduling.publishing_scheduler

# Start the Telegram bot (human-in-the-loop)
uv run python -m src.ui.telegram_bot
```

## Project Structure

```
personal-brand-agent/
├── src/
│   ├── models.py              # Shared data types (single source of truth)
│   ├── agents/                # Core pipeline agents
│   │   ├── orchestrator.py    # LangGraph state machine
│   │   ├── trend_scout.py     # Trend mining (Perplexity + ArXiv)
│   │   ├── analyzer.py        # Content analysis and brief creation
│   │   ├── writer.py          # Post generation (Claude Opus 4.5)
│   │   ├── humanizer.py       # Humanization and voice matching
│   │   ├── visual_creator.py  # Image generation (Nano Banana Pro)
│   │   ├── photo_selector.py  # Personal photo selection
│   │   └── qc_agent.py        # Quality control scoring
│   ├── meta_agent/            # Self-improvement layer
│   ├── author/                # Author voice profiling
│   ├── scheduling/            # Publishing scheduler
│   ├── logging/               # Structured logging system
│   ├── tools/                 # External API wrappers
│   ├── generated/             # Auto-generated modules (by meta-agent)
│   ├── knowledge/             # Style guide, top posts, anti-patterns
│   └── ui/                    # Telegram bot interface
├── config/                    # YAML/JSON configuration (modifiable by meta-agent)
├── prompts/                   # Evolvable prompt templates (versioned)
├── photos/                    # Personal photo library
├── data/                      # Runtime data (drafts, analytics, experiments)
└── architecture.md            # Complete system specification
```

## Autonomy Levels

| Level | Description |
|---|---|
| 1 | Human approves everything |
| 2 | Human approves posts, auto-modifications allowed |
| 3 | Auto-publish high-score posts (>= 9.0), human approval for the rest |
| 4 | Full autonomy (human notified, not asked) |

## Architecture Documentation

See [architecture.md](architecture.md) for the complete system specification (~24,000 lines), including:

- Detailed agent specifications and prompts
- Supabase database schema and client implementation
- LangGraph orchestrator state machine definition
- Post analytics and feedback loop design
- Meta-agent and self-modifying code engine
- Modification safety and rollback system
- Author voice profiling methodology
- Scheduling system with optimal timing

## License

MIT
