# LinkedIn Super Agent - Architecture Fixes Document

**Generated**: 2026-02-04
**Analyzed by**: 20 parallel analysis agents
**Total Issues Found**: ~230+
**Critical Issues**: 45+

---

## Executive Summary

Архитектура LinkedIn Super Agent содержит хорошо продуманную структуру, но имеет множество критических проблем:

1. **Security Vulnerabilities** (CRITICAL): Prompt injection, path traversal в Claude CLI, AST bypass в code generation
2. **Data Inconsistencies** (CRITICAL): Дублирование таблиц, несовместимые типы данных между SQL и Python
3. **Safety Bypasses** (CRITICAL): SelfModificationEngine обходит ModificationSafetySystem и AutonomyManager
4. **Missing Implementations**: Feedback loop не замкнут, rollback не реализован, revision target всегда fallback на Writer
5. **Cross-module Inconsistencies**: 15 категорий несогласованностей между модулями

---

## Table of Contents

1. [CRITICAL: Security Fixes](#1-critical-security-fixes)
2. [CRITICAL: Database Schema Fixes](#2-critical-database-schema-fixes)
3. [CRITICAL: Safety System Integration](#3-critical-safety-system-integration)
4. [CRITICAL: State Machine Fixes](#4-critical-state-machine-fixes)
5. [HIGH: Agent Consistency Fixes](#5-high-agent-consistency-fixes)
6. [HIGH: Logging & Observability Fixes](#6-high-logging--observability-fixes)
7. [HIGH: Analytics & Feedback Loop Fixes](#7-high-analytics--feedback-loop-fixes)
8. [HIGH: Self-Improvement Layer Fixes](#8-high-self-improvement-layer-fixes)
9. [MEDIUM: Author Profile & Writer Fixes](#9-medium-author-profile--writer-fixes)
10. [MEDIUM: Scheduling System Fixes](#10-medium-scheduling-system-fixes)
11. [LOW: Code Quality Fixes](#11-low-code-quality-fixes)

---

## 1. CRITICAL: Security Fixes

### 1.1 Prompt Injection in Claude CLI (Lines ~1580-1604)

**Problem**: `prompt` передается напрямую в subprocess без санитизации.

**Current Code**:
```python
cmd = [
    "claude",
    "-p", full_prompt,  # DIRECT PROMPT INJECTION POINT
    "--output-format", "text",
]
```

**Fix**:
```python
import re

def _sanitize_prompt(self, prompt: str) -> str:
    """Remove potentially dangerous patterns."""
    dangerous_patterns = [
        r'`.*`',           # Backtick command substitution
        r'\$\(.*\)',       # $(command) substitution
        r';\s*\w+',        # Command chaining
        r'\|\s*\w+',       # Pipe to another command
        r'>\s*/',          # Output redirection to root
    ]
    sanitized = prompt
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, '', sanitized)
    return sanitized

# In generate():
cmd = [
    "claude",
    "-p", self._sanitize_prompt(full_prompt),
    "--output-format", "text",
]
```

### 1.2 Path Traversal in Context Files (Lines ~1588-1596)

**Problem**: `context_files` читаются без проверки path traversal.

**Fix**:
```python
def _validate_context_file(self, file: Path) -> bool:
    """Validate file is within allowed directory."""
    try:
        resolved = file.resolve()
        return resolved.is_relative_to(self.working_dir.resolve())
    except ValueError:
        return False

# In generate():
if context_files:
    for file in context_files:
        if not self._validate_context_file(file):
            raise SecurityError(f"Path traversal attempt: {file}")
        if file.exists():
            content = file.read_text(encoding='utf-8')
```

### 1.3 AST Security Scan Bypass (Lines ~15550-15666)

**Problem**: Unicode normalization bypass, encoding tricks, lambda + globals.

**Fix**:
```python
import unicodedata

class SecurityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.issues = []

    def check_code(self, code: str) -> List[str]:
        # CRITICAL: Normalize unicode before parsing
        normalized = unicodedata.normalize('NFKC', code)

        # Additional dangerous patterns to check
        dangerous_patterns = [
            r'__loader__',
            r'__spec__',
            r'__class__\.__bases__',
            r'types\.FunctionType',
            r'types\.CodeType',
            r'b64decode',
            r'codecs\.decode',
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, normalized):
                self.issues.append(f"Dangerous pattern detected: {pattern}")

        tree = ast.parse(normalized)
        self.visit(tree)
        return self.issues
```

### 1.4 Sandbox Environment Leak (Lines ~15668-15702)

**Problem**: `**os.environ` копирует ВСЕ переменные окружения включая secrets.

**Fix**:
```python
def _get_minimal_env(self) -> dict:
    """Return minimal safe environment."""
    return {
        "PYTHONPATH": str(self.sandbox_path),
        "HOME": os.environ.get("HOME", "/tmp"),
        "PATH": "/usr/bin:/bin",
        "LANG": "en_US.UTF-8",
        # Explicitly DO NOT include API keys, tokens, etc.
    }

# In _run_sandbox_tests():
result = subprocess.run(
    cmd,
    env=self._get_minimal_env(),  # NOT **os.environ
    timeout=30,
    capture_output=True,
)
```

### 1.5 Package Allowlist Bypass (Lines ~15411-15451)

**Problem**: Version specifiers и PEP 508 markers не санитизируются.

**Fix**:
```python
from packaging.requirements import Requirement, InvalidRequirement

def _is_safe_package_name(self, name: str) -> bool:
    """Validate package using packaging library."""
    try:
        req = Requirement(name)
        base_name = req.name.lower().replace('_', '-')

        # Reject any extras
        if req.extras:
            return False

        # Reject any markers (security risk)
        if req.marker:
            return False

        return base_name in self.ALLOWED_PACKAGES
    except InvalidRequirement:
        return False
```

---

## 2. CRITICAL: Database Schema Fixes

### 2.1 Remove Duplicate Photo Tables (Lines ~1088-1127 & 1223-1270)

**Problem**: `author_photos` и `photo_metadata` - две практически идентичные таблицы.

**Fix**:
```sql
-- Drop duplicate table
DROP TABLE IF EXISTS photo_metadata;

-- Add missing fields to author_photos
ALTER TABLE author_photos
    ADD COLUMN IF NOT EXISTS notes TEXT,
    ADD CONSTRAINT author_photos_file_path_unique UNIQUE (file_path);

-- Standardize eye_contact type
ALTER TABLE author_photos
    ALTER COLUMN eye_contact TYPE TEXT;

ALTER TABLE author_photos
    ADD CONSTRAINT valid_eye_contact
    CHECK (eye_contact IN ('direct', 'away', 'profile'));
```

### 2.2 Add Missing Foreign Keys

**Fix**:
```sql
-- Fix posts.topic_id (Line 866)
ALTER TABLE posts
    ALTER COLUMN topic_id TYPE UUID USING topic_id::uuid;

ALTER TABLE posts
    ADD CONSTRAINT fk_posts_topic
    FOREIGN KEY (topic_id) REFERENCES topic_cache(id) ON DELETE SET NULL;

-- Fix drafts.topic_id (Line 1164)
ALTER TABLE drafts
    ALTER COLUMN topic_id TYPE UUID USING topic_id::uuid;

ALTER TABLE drafts
    ADD CONSTRAINT fk_drafts_topic
    FOREIGN KEY (topic_id) REFERENCES topic_cache(id) ON DELETE SET NULL;
```

### 2.3 Add Missing RPC Functions

**Fix**:
```sql
-- get_average_metrics_at_minutes (referenced at Line 430)
CREATE OR REPLACE FUNCTION get_average_metrics_at_minutes(minutes INTEGER)
RETURNS TABLE (
    avg_likes FLOAT,
    avg_comments FLOAT,
    avg_reposts FLOAT,
    avg_engagement_rate FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        AVG(likes)::FLOAT as avg_likes,
        AVG(comments)::FLOAT as avg_comments,
        AVG(reposts)::FLOAT as avg_reposts,
        AVG(engagement_rate)::FLOAT as avg_engagement_rate
    FROM post_metrics
    WHERE minutes_after_post = minutes;
END;
$$ LANGUAGE plpgsql;

-- get_average_qc_score (referenced at Line 438)
CREATE OR REPLACE FUNCTION get_average_qc_score()
RETURNS TABLE (avg_score FLOAT) AS $$
BEGIN
    RETURN QUERY
    SELECT AVG(qc_score)::FLOAT as avg_score
    FROM posts
    WHERE qc_score IS NOT NULL;
END;
$$ LANGUAGE plpgsql;

-- get_likes_percentile - FIXED VERSION (Lines 1390-1399)
CREATE OR REPLACE FUNCTION get_likes_percentile(likes_count INTEGER, minutes INTEGER)
RETURNS TABLE (percentile FLOAT) AS $$
BEGIN
    RETURN QUERY
    SELECT
        CASE
            WHEN COUNT(*) = 0 THEN 50.0  -- Default to median if no data
            ELSE (COUNT(*) FILTER (WHERE likes <= likes_count)::FLOAT / COUNT(*) * 100)::FLOAT
        END as percentile
    FROM post_metrics
    WHERE minutes_after_post = minutes;
END;
$$ LANGUAGE plpgsql;
```

### 2.4 Add Missing Composite Indexes

**Fix**:
```sql
-- For get_learnings_for_component (frequent query)
CREATE INDEX idx_learnings_component_active_confidence
ON learnings(affected_component, is_active, confidence DESC)
WHERE is_active = TRUE AND confidence >= 0.5;

-- For experiments queries
CREATE INDEX idx_experiments_completed
ON experiments(completed_at DESC)
WHERE status = 'completed';

-- For post_metrics percentile queries
CREATE INDEX idx_metrics_minutes
ON post_metrics(minutes_after_post);

-- For research reports
CREATE INDEX idx_research_trigger
ON research_reports(trigger_type);
```

### 2.5 Add Missing SupabaseDB Methods

**Fix** (add to SupabaseDB class after Line ~796):
```python
# ─────────────────────────────────────────────────────────────────────
# AGENT LOGS (missing - referenced in code but not in class)
# ─────────────────────────────────────────────────────────────────────

async def save_agent_log(self, log_entry: Dict[str, Any]) -> str:
    """Save an agent log entry."""
    if not log_entry:
        raise ValidationError("log_entry cannot be None or empty")
    if "timestamp" not in log_entry or "level" not in log_entry:
        raise ValidationError("log_entry must have 'timestamp' and 'level'")

    result = await self.client.table("agent_logs").insert(log_entry).execute()
    if not result.data:
        raise DatabaseError("Insert succeeded but returned no data")
    return result.data[0]["id"]

async def get_agent_logs(
    self,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    level: Optional[int] = None,
    component: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """Query agent logs with filters."""
    query = self.client.table("agent_logs").select("*")

    if start_time:
        query = query.gte("timestamp", start_time.isoformat())
    if end_time:
        query = query.lte("timestamp", end_time.isoformat())
    if level:
        query = query.gte("level", level)
    if component:
        query = query.eq("component", component)

    result = await query.order("timestamp", desc=True).limit(limit).execute()
    return result.data

# ─────────────────────────────────────────────────────────────────────
# PIPELINE ERRORS
# ─────────────────────────────────────────────────────────────────────

async def save_pipeline_error(self, error: Dict[str, Any]) -> str:
    """Save a pipeline error for post-mortem analysis."""
    if not error:
        raise ValidationError("error cannot be None or empty")

    result = await self.client.table("pipeline_errors").insert(error).execute()
    if not result.data:
        raise DatabaseError("Insert succeeded but returned no data")
    return result.data[0]["id"]

# ─────────────────────────────────────────────────────────────────────
# PENDING APPROVALS
# ─────────────────────────────────────────────────────────────────────

async def save_pending_approval(self, approval: Dict[str, Any]) -> str:
    """Save a pending approval request."""
    if not approval or "run_id" not in approval:
        raise ValidationError("approval must have 'run_id'")

    result = await self.client.table("pending_approvals").insert(approval).execute()
    if not result.data:
        raise DatabaseError("Insert succeeded but returned no data")
    return result.data[0]["id"]

async def get_pending_approvals(self, status: str = "pending") -> List[Dict[str, Any]]:
    """Get pending approvals by status."""
    result = await (
        self.client.table("pending_approvals")
        .select("*")
        .eq("status", status)
        .order("requested_at", desc=False)
        .execute()
    )
    return result.data
```

---

## 3. CRITICAL: Safety System Integration

### 3.1 SelfModificationEngine Must Check AutonomyManager (Lines ~19178-19224)

**Problem**: SelfModificationEngine применяет модификации без проверки autonomy level.

**Fix**:
```python
async def apply_recommendations(
    self,
    recommendations: List[dict],
    auto_apply: bool = True
) -> List[ModificationRecord]:
    """Apply recommendations with autonomy checks."""

    # NEW: Check autonomy level before any modifications
    autonomy_mgr = await get_autonomy_manager()

    for rec in recommendations:
        record = ModificationRecord(
            id=generate_id(),
            modification_type=rec["type"],
            # ...
        )

        # NEW: Check if auto-apply is allowed at current autonomy level
        if auto_apply:
            needs_approval = await autonomy_mgr.requires_human_approval(
                action="modification",
                score=rec.get("confidence", 0.5)
            )

            if needs_approval:
                # Store for human review instead of auto-applying
                await self._store_pending_modification(record)
                record.status = "pending_approval"
            else:
                await self._apply_modification(record)
                record.status = "applied"
        else:
            await self._store_pending_modification(record)
            record.status = "pending_approval"

        await self.db.save_modification(record)
        applied.append(record)

    return applied
```

### 3.2 MetaAgent Must Use ModificationSafetySystem (Lines ~19686-19695)

**Problem**: MetaAgent обходит safety system напрямую.

**Fix**:
```python
class MetaAgent:
    def __init__(
        self,
        evaluator: SelfEvaluator,
        researcher: ResearchAgent,
        safety_system: ModificationSafetySystem,  # CHANGED from modifier
        experimenter: ExperimentationEngine,
        db
    ):
        self.safety = safety_system
        # ... rest of init

    async def run_improvement_cycle(self, post_performance: PostPerformance):
        # ... evaluation code ...

        if high_confidence:
            # OLD: modifications = await self.modifier.apply_recommendations(...)
            # NEW: Go through safety system
            for rec in high_confidence:
                result = await self.safety.request_modification(
                    modification_type=rec["type"],
                    description=rec["description"],
                    proposed_change=rec,
                    component=rec.get("component", "meta_agent"),
                    risk_level=self._classify_risk(rec)
                )

                if result == "auto_applied":
                    self.modifications_applied.append(rec)
                elif result == "pending_approval":
                    self.pending_modifications.append(rec)
```

### 3.3 Add Safeguards for Level 4 Autonomy (Lines ~20049-20055)

**Problem**: Level 4 не имеет никаких safeguards - даже посты с очень низким score будут опубликованы.

**Fix**:
```python
MIN_SAFE_SCORE = 5.0  # Even at full autonomy, don't publish garbage

async def can_auto_publish(
    self,
    score: float,
    content_type: Optional[ContentType] = None
) -> bool:
    """Check if content can be auto-published."""
    level = await self.get_effective_level()

    if level == AutonomyLevel.HUMAN_ALL:
        return False

    if level == AutonomyLevel.HUMAN_POSTS_ONLY:
        return False

    if level == AutonomyLevel.AUTO_HIGH_SCORE:
        threshold = self._config.get_auto_publish_threshold(content_type)
        return score >= threshold

    if level == AutonomyLevel.FULL_AUTONOMY:
        # NEW: Even at full autonomy, enforce minimum quality
        return score >= MIN_SAFE_SCORE

    return False
```

### 3.4 Add Timeout and Expiration to Approval Flow (Lines ~21834-21858)

**Problem**: Pending modifications могут висеть бесконечно.

**Fix**:
```python
@dataclass
class ModificationRequest:
    """Request to modify system behavior."""
    id: str
    modification_type: str
    risk_level: ModificationRiskLevel
    description: str
    proposed_change: Dict[str, Any]
    component: str
    before_state: Dict[str, Any]
    status: str  # "pending", "approved", "rejected", "auto_applied", "rolled_back"
    human_approver: Optional[str] = None
    approved_at: Optional[datetime] = None

    # NEW: Audit and timeout fields
    created_at: datetime = field(default_factory=utc_now)
    expires_at: Optional[datetime] = None
    reminder_at: Optional[datetime] = None
    rejected_reason: Optional[str] = None
    modification_chain_id: Optional[str] = None

async def _store_pending_modification(self, mod: ModificationRequest):
    """Store with timeout."""
    mod.status = "pending"
    mod.expires_at = utc_now() + timedelta(hours=24)
    mod.reminder_at = utc_now() + timedelta(hours=4)
    await self.db.save_modification(mod)

async def check_expired_modifications(self):
    """Auto-reject expired modifications. Call from scheduler."""
    expired = await self.db.get_expired_pending_modifications()
    for mod in expired:
        mod.status = "auto_rejected"
        mod.rejected_reason = "Expired - no response within 24 hours"
        await self.db.update_modification(mod)
```

---

## 4. CRITICAL: State Machine Fixes

### 4.1 Add Missing @with_error_handling Decorators (Lines ~12260, ~12480)

**Problem**: `reset_for_restart_node` и `post_evaluation_learning_node` не обернуты в error handling.

**Fix**:
```python
@with_error_handling(node_name="reset_for_restart")
@with_timeout(node_name="reset_for_restart")
async def reset_for_restart_node(state: PipelineState) -> Dict[str, Any]:
    """Reset state for restart after rejection."""
    # ... existing code ...

@with_error_handling(node_name="post_evaluation_learning")
@with_timeout(node_name="post_evaluation_learning")
async def post_evaluation_learning_node(state: PipelineState) -> dict:
    """Extract learnings after QC evaluation."""
    # ... existing code ...
```

### 4.2 Fix Revision Target Logic (Lines ~12461-12472)

**Problem**: Revision target всегда fallback на Writer, даже если проблема в humanization.

**Fix**:
```python
def determine_revision_target(qc_output: QCOutput) -> str:
    """Determine which agent should handle revision based on failed criteria."""
    scores = qc_output.result.universal_scores

    # Define which criteria map to which agent
    humanizer_criteria = {"humanness", "tone_match", "authenticity"}
    visual_criteria = {"visual_match", "visual_quality"}
    writer_criteria = {"hook_strength", "value_density", "specificity", "structure", "cta_clarity"}

    # Find lowest scoring criteria
    lowest_score = float('inf')
    lowest_criterion = None

    for criterion, score in scores.items():
        if score < lowest_score:
            lowest_score = score
            lowest_criterion = criterion

    # Route based on lowest criterion
    if lowest_criterion in humanizer_criteria:
        return "revise_humanizer"
    elif lowest_criterion in visual_criteria:
        return "revise_visual"
    else:
        return "revise_writer"

# In route_after_qc:
else:  # revise
    next_step = determine_revision_target(qc_output)
    return next_step
```

### 4.3 Clear current_revision_target After Revision (Lines ~11975-11981)

**Problem**: `current_revision_target` не очищается в writer_node и humanizer_node.

**Fix** (add to return dict of writer_node and humanizer_node):
```python
# In writer_node return:
return {
    "stage": "drafted",
    "draft": draft,
    # ... existing fields ...
    "current_revision_target": None  # ADDED: Clear after processing
}

# In humanizer_node return:
return {
    "stage": "humanized",
    "humanized_post": post,
    # ... existing fields ...
    "current_revision_target": None  # ADDED: Clear after processing
}
```

### 4.4 Add visual_brief to PipelineState (after Line ~10986)

**Problem**: Поле `visual_brief` очищается в `reset_for_restart_node`, но не определено в `PipelineState`.

**Fix**:
```python
class PipelineState(TypedDict):
    # ... existing fields ...

    # ADD: Visual brief for image generation
    visual_brief: Optional[str]
```

### 4.5 Fix Threshold Inconsistency (Lines ~2072-2078)

**Problem**: pass_threshold (6.8 для COMMUNITY_CONTENT) НИЖЕ базового revision_threshold (7.0).

**Fix**:
```python
# Option 1: Use consistent multipliers that maintain threshold ordering
type_multipliers: Dict[ContentType, float] = field(default_factory=lambda: {
    ContentType.ENTERPRISE_CASE: 0.90,      # 8.0 * 0.90 = 7.2 pass, 6.3 revise, 4.95 reject
    ContentType.PRIMARY_SOURCE: 0.9375,     # 8.0 * 0.9375 = 7.5 pass
    ContentType.AUTOMATION_CASE: 0.90,      # Changed from 0.875 to 0.90
    ContentType.COMMUNITY_CONTENT: 0.90,    # Changed from 0.85 to 0.90
    ContentType.TOOL_RELEASE: 0.90,         # Changed from 0.875 to 0.90
})

# Option 2: Use absolute thresholds instead of multipliers
@dataclass
class ContentTypeThresholds:
    pass_threshold: float
    revision_threshold: float
    rejection_threshold: float

content_type_thresholds = {
    ContentType.COMMUNITY_CONTENT: ContentTypeThresholds(
        pass_threshold=7.0,      # Lower bar for community content
        revision_threshold=5.5,  # Always lower than pass
        rejection_threshold=4.0
    ),
    # ... other types
}
```

---

## 5. HIGH: Agent Consistency Fixes

### 5.1 Unify HookStyle Between TrendScout and Analyzer (Lines ~4554-4605, ~5305-5327)

**Problem**: hooks_prompt в Analyzer использует названия, которых нет в HookStyle enum.

**Fix**:
```python
# Option 1: Update hooks_prompt to use HookStyle enum values
def get_hooks_prompt(content_type: ContentType) -> str:
    """Generate hooks prompt using HookStyle enum."""
    available_styles = [style.value for style in HookStyle]
    style_descriptions = {
        HookStyle.METRICS.value: "Lead with the impressive number",
        HookStyle.LESSONS_LEARNED.value: "Lead with the key learning",
        HookStyle.PROBLEM_SOLUTION.value: "Present problem then solution",
        HookStyle.CONTRARIAN.value: "Challenge common assumptions",
        HookStyle.STORY.value: "Lead with narrative",  # ADD to enum if missing
        # ... etc
    }

    prompt_lines = ["Hook styles to use (MUST use these exact names):"]
    for i, style in enumerate(available_styles, 1):
        desc = style_descriptions.get(style, "")
        prompt_lines.append(f"{i}. {style.upper()}: {desc}")

    return "\n".join(prompt_lines)

# Option 2: Add missing styles to HookStyle enum
class HookStyle(str, Enum):
    METRICS = "metrics"
    LESSONS_LEARNED = "lessons_learned"
    PROBLEM_SOLUTION = "problem_solution"
    CONTRARIAN = "contrarian"
    STORY = "story"                      # ADD
    INDUSTRY_IMPACT = "industry_impact"  # ADD
    # ... existing styles ...
```

### 5.2 Synchronize DraftPost and HumanizedPost (Lines ~7351-7381, ~7791-7809)

**Problem**: HumanizedPost теряет поля и меняет типы.

**Fix**:
```python
@dataclass
class HumanizedPost:
    # Keep ALL fields from DraftPost
    hook: str
    body: str
    cta: str
    hashtags: List[str]
    humanized_text: str  # Changed from full_text for clarity

    # FIXED: Match types with DraftPost
    hook_style: Optional[HookStyle] = None  # Was: Optional[str]
    template_used: str = ""  # Was: Optional[str]
    template_category: str = ""

    # ADDED: Missing fields from DraftPost
    estimated_read_time: str = ""
    type_data_injected: Dict[str, Any] = field(default_factory=dict)

    # Humanizer-specific
    voice_consistency_score: float = 0.0
    humanization_changes: List[str] = field(default_factory=list)
    # ... rest of fields
```

### 5.3 Remove Duplicate _select_composition_mode (Lines ~9037-9055, ~9214-9241)

**Problem**: Функция определена дважды с разной логикой.

**Fix**:
```python
# DELETE first version (Lines 9037-9055)
# KEEP second version (Lines 9214-9241) as it has correct composition keys

# Also ensure composition_instructions has all required keys:
composition_instructions = {
    "device_only": "...",
    "author_holding_phone": "...",
    "author_at_desk": "...",
    "split_view": "...",
    "before_after": "...",
    # ADD missing keys that first version referenced:
    "split_screen": "...",  # Map to split_view or add
    "overlay": "...",
    "workflow": "...",
    "quote_card": "...",
    "demo": "...",
}
```

### 5.4 Unify get_config_path Functions (Lines ~21934-21965, ~22065-22085)

**Problem**: Две версии функции с разными маппингами.

**Fix**:
```python
# DELETE _get_config_path method from class (Lines 21934-21965)
# KEEP standalone get_config_path with unified mapping:

CONFIG_PATH_MAPPING = {
    "writer": "config/writer_config.json",
    "trend_scout": "config/trend_scout_config.json",  # UNIFIED
    "analyzer": "config/analyzer_config.json",
    "humanizer": "config/humanizer_config.json",
    "visual_creator": "config/visual_creator_config.json",  # UNIFIED
    "qc": "config/qc_config.json",  # UNIFIED
    "scheduler": "config/scheduler_config.json",
    "meta_agent": "config/meta_agent_config.json",
}

def get_config_path(component: str) -> str:
    """Get config path for a component. Single source of truth."""
    if component not in CONFIG_PATH_MAPPING:
        raise ValueError(f"Unknown component: {component}. Valid: {list(CONFIG_PATH_MAPPING.keys())}")
    return CONFIG_PATH_MAPPING[component]
```

### 5.5 Unify Scoring Scale (Lines ~3866-3962, ~5728-5734)

**Problem**: TrendScout возвращает 0.0-1.0, но Insight.wow_factor использует 1-10.

**Fix**:
```python
# Define standard score types
Score01 = NewType('Score01', float)  # 0.0 to 1.0
Score10 = NewType('Score10', float)  # 0.0 to 10.0

# In calculate_trend_score, explicitly return Score10:
def calculate_trend_score(topic: Dict, content_type: ContentType) -> Score10:
    weighted_sum = sum(
        base_scores[factor]["weight"] * factor_score
        for factor, factor_score in scores.items()
    )
    # weighted_sum is 0-1, convert to 0-10 scale
    return Score10(weighted_sum * 10)

# In Insight, use Score10:
@dataclass
class Insight:
    wow_factor: Score10  # Was: int (1-10)

    def __post_init__(self):
        if not 0 <= self.wow_factor <= 10:
            raise ValueError(f"wow_factor must be 0-10, got {self.wow_factor}")
```

---

## 6. HIGH: Logging & Observability Fixes

### 6.1 Remove Duplicate Logging Systems (Lines ~133-230, ~20762-21676)

**Problem**: LoggerFactory и AgentLogger - две независимые системы.

**Fix**:
```python
# Option 1: Keep AgentLogger, deprecate LoggerFactory
# In Lines 133-230, add:

import warnings

class LoggerFactory:
    """
    DEPRECATED: Use AgentLogger from logging/agent_logger.py instead.

    This class is kept for backwards compatibility only.
    """

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        warnings.warn(
            "LoggerFactory is deprecated. Use AgentLogger instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Delegate to AgentLogger
        from src.logging.agent_logger import AgentLogger
        return AgentLogger.get_logger(name)

# Option 2: Merge into single system
# Delete LoggerFactory entirely, update all references to use AgentLogger
```

### 6.2 Add Missing LogComponent Values (Lines ~20863-20893)

**Problem**: Отсутствуют компоненты для VALIDATOR, LEARNER, NANO_BANANA и др.

**Fix**:
```python
class LogComponent(Enum):
    # Core agents (existing)
    ORCHESTRATOR = "orchestrator"
    TREND_SCOUT = "trend_scout"
    ANALYZER = "analyzer"
    WRITER = "writer"
    HUMANIZER = "humanizer"
    VISUAL_CREATOR = "visual_creator"
    QC_AGENT = "qc_agent"
    META_AGENT = "meta_agent"
    EVALUATOR = "evaluator"

    # ADD: Missing components
    VALIDATOR = "validator"
    LEARNER = "learner"
    NANO_BANANA = "nano_banana"
    PERPLEXITY = "perplexity"
    HUMAN_APPROVAL = "human_approval"
    AB_TEST = "ab_test"
    STARTUP = "startup"
    CONFIG = "config"

    # Infrastructure (existing)
    MODIFICATION_SAFETY = "modification_safety"
    # ... rest
```

### 6.3 Fix Async Logging Task Handling (Lines ~21051-21057)

**Problem**: `asyncio.create_task()` без сохранения ссылки - задачи могут быть garbage collected.

**Fix**:
```python
class AgentLogger:
    def __init__(self, ...):
        # ... existing init ...
        self._pending_tasks: Set[asyncio.Task] = set()

    def _log(self, level: LogLevel, ...):
        # ... existing code ...

        # Write to Supabase (async, track task)
        if self.supabase and level.value >= self.min_level.value:
            task = asyncio.create_task(self._write_to_supabase(entry))
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)

        # Send to Telegram (async, track task)
        if self.telegram and level.value >= self.telegram_min_level.value:
            task = asyncio.create_task(self._send_to_telegram(entry))
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)

    async def flush(self):
        """Wait for all pending log tasks to complete."""
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
            self._pending_tasks.clear()
```

---

## 7. HIGH: Analytics & Feedback Loop Fixes

### 7.1 Add Missing LinkedIn Reaction Types (Lines ~13517)

**Problem**: Только `likes_count` собирается, не типы реакций.

**Fix**:
```python
def _parse_reactions(self, reactions_data: List[dict]) -> Dict[str, int]:
    """Parse reactions by type."""
    reactions_by_type = {
        "LIKE": 0,
        "CELEBRATE": 0,
        "SUPPORT": 0,
        "LOVE": 0,
        "INSIGHTFUL": 0,
        "FUNNY": 0,
    }

    for r in reactions_data:
        rtype = r.get('reactionType', 'LIKE')
        count = r.get('count', 0)
        if rtype in reactions_by_type:
            reactions_by_type[rtype] += count

    return reactions_by_type

# Update PostMetricsSnapshot:
@dataclass
class PostMetricsSnapshot:
    # ... existing fields ...
    reactions_by_type: Dict[str, int] = field(default_factory=dict)  # ADD

# Update SQL schema:
# ALTER TABLE post_metrics ADD COLUMN reactions_by_type JSONB DEFAULT '{}';
```

### 7.2 Implement InsightApplicator (Lines ~13120-13136)

**Problem**: `AnalyticsInsight` определен, но не применяется.

**Fix**:
```python
class InsightApplicator:
    """Apply analytics insights to system components."""

    def __init__(self, config_manager: ConfigManager, prompt_manager: PromptManager):
        self.config = config_manager
        self.prompts = prompt_manager

        self._component_handlers = {
            "trend_scout": self._apply_to_trend_scout,
            "writer": self._apply_to_writer,
            "visual_creator": self._apply_to_visual_creator,
            "scheduler": self._apply_to_scheduler,
        }

    async def apply(self, insight: AnalyticsInsight) -> bool:
        """Apply insight to affected component."""
        if insight.confidence < 0.7:
            logger.warning(f"Insight confidence too low: {insight.confidence}")
            return False

        handler = self._component_handlers.get(insight.affected_component)
        if not handler:
            logger.error(f"Unknown component: {insight.affected_component}")
            return False

        return await handler(insight)

    async def _apply_to_trend_scout(self, insight: AnalyticsInsight) -> bool:
        """Apply insight to Trend Scout scoring weights."""
        if insight.parameter_to_adjust and insight.suggested_value:
            current_config = self.config.get("trend_scout")
            current_config["scoring_weights"][insight.parameter_to_adjust] = insight.suggested_value
            await self.config.save("trend_scout", current_config)
            return True
        return False

    # ... similar methods for other components
```

### 7.3 Add Checkpoint Recovery (Lines ~13745-13773)

**Problem**: Нет recovery для пропущенных checkpoints.

**Fix**:
```python
class MetricsScheduler:
    # ... existing code ...

    async def recover_missed_checkpoints(self):
        """Recover metrics for posts with missed checkpoints."""
        # Find posts with gaps in checkpoints
        incomplete = await self.db.get_posts_with_incomplete_metrics()

        for post in incomplete:
            collected = set(m["minutes_after_post"] for m in post["metrics"])
            expected = set(self.COLLECTION_SCHEDULE)

            # Find reasonable gaps (within 24 hours of post)
            post_age_minutes = (utc_now() - post["published_at"]).total_seconds() / 60

            for checkpoint in expected - collected:
                if checkpoint <= post_age_minutes <= checkpoint + 1440:  # Within 24h window
                    # Collect late but still valuable
                    await self._collect_and_store(
                        post["linkedin_urn"],
                        checkpoint,
                        is_recovery=True
                    )

        logger.info(f"Recovered metrics for {len(incomplete)} posts")
```

---

## 8. HIGH: Self-Improvement Layer Fixes

### 8.1 Add Rate Limiting for Learning Confirmations (Lines ~14166-14185)

**Problem**: 5 быстрых confirmations могут promote learning to rule.

**Fix**:
```python
MIN_CONFIRMATION_INTERVAL_HOURS = 24

def confirm(self):
    """Called when evidence supports this learning."""
    # NEW: Rate limiting
    if self.last_confirmed_at:
        hours_since = (utc_now() - self.last_confirmed_at).total_seconds() / 3600
        if hours_since < MIN_CONFIRMATION_INTERVAL_HOURS:
            logger.info(f"Confirmation rate limited: {hours_since:.1f}h < {MIN_CONFIRMATION_INTERVAL_HOURS}h")
            return

    # NEW: Require real engagement data before promotion
    if self.confirmations >= 5 and not self._has_engagement_validation():
        logger.warning("Cannot promote to rule without engagement validation")
        return

    self.confirmations += 1
    self.last_confirmed_at = utc_now()
    self.confidence = min(1.0, self.confidence + 0.1 * (1 - self.confidence))

    if self.confidence >= 0.9 and self.confirmations >= 5:
        self.is_promoted_to_rule = True

def _has_engagement_validation(self) -> bool:
    """Check if this learning has been validated against real engagement."""
    return self.source == LearningSource.POST_PERFORMANCE
```

### 8.2 Implement Rollback Mechanism (Lines ~13905-13908)

**Problem**: Rollback упоминается, но не реализован.

**Fix**:
```python
@dataclass
class LearningSnapshot:
    """Snapshot of learning state for rollback."""
    timestamp: datetime
    learnings: Dict[str, MicroLearning]
    prompts: Dict[str, str]
    trigger: str  # Why snapshot was taken

class ContinuousLearningEngine:
    def __init__(self, ...):
        # ... existing init ...
        self.snapshots: List[LearningSnapshot] = []
        self.MAX_SNAPSHOTS = 10

    def create_snapshot(self, trigger: str):
        """Create snapshot before significant changes."""
        snapshot = LearningSnapshot(
            timestamp=utc_now(),
            learnings=deepcopy(self.learnings),
            prompts=self.prompt_manager.get_all_prompts(),
            trigger=trigger
        )
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.MAX_SNAPSHOTS:
            self.snapshots.pop(0)
        logger.info(f"Created learning snapshot: {trigger}")

    async def rollback_to_snapshot(self, snapshot_index: int = -1):
        """Rollback to a previous snapshot."""
        if not self.snapshots:
            raise ValueError("No snapshots available for rollback")

        snapshot = self.snapshots[snapshot_index]

        # Restore learnings
        self.learnings = deepcopy(snapshot.learnings)

        # Restore prompts
        for component, prompt in snapshot.prompts.items():
            self.prompt_manager.set_prompt(component, prompt)

        # Persist to database
        await self.db.restore_learnings(list(self.learnings.values()))

        logger.warning(f"Rolled back to snapshot from {snapshot.timestamp}: {snapshot.trigger}")
```

### 8.3 Add Bootstrap Learning Expiration (Lines ~14754-14801)

**Problem**: Bootstrap learnings не имеют механизма expiration.

**Fix**:
```python
async def handle_first_post(self) -> List[MicroLearning]:
    """Bootstrap with proven best practices - with expiration."""
    BOOTSTRAP_EXPIRY_DAYS = 30  # Expire after 30 days
    BOOTSTRAP_CONFIDENCE = 0.5  # Below application threshold initially

    bootstrap_learnings = [
        MicroLearning(
            id="bootstrap_hook_numbers",
            learning_type=LearningType.PATTERN,
            affected_component="writer",
            description="Hooks with specific numbers perform better",
            rule="include_number_in_hook",
            confidence=BOOTSTRAP_CONFIDENCE,  # Below 0.6 threshold
            source=LearningSource.EXPLICIT_RULE,
            is_bootstrap=True,  # ADD: Mark as bootstrap
            expires_at=utc_now() + timedelta(days=BOOTSTRAP_EXPIRY_DAYS),  # ADD
        ),
        # ... other bootstrap learnings
    ]

    # Save to DB
    await self.db.save_learnings([l.__dict__ for l in bootstrap_learnings])
    return bootstrap_learnings

# In get_learnings_for_prompt, filter expired:
def get_learnings_for_prompt(...) -> List[MicroLearning]:
    now = utc_now()
    relevant = [
        l for l in self.learnings.values()
        if l.affected_component == component
        and l.is_active
        and (l.expires_at is None or l.expires_at > now)  # Filter expired
    ]
    # ... rest
```

---

## 9. MEDIUM: Author Profile & Writer Fixes

### 9.1 Add Missing Fields to Style Guide (Lines ~22876-22908)

**Problem**: generate_style_guide_for_writer() игнорирует 8 важных полей.

**Fix**:
```python
def generate_style_guide_for_writer(
    self,
    profile: AuthorVoiceProfile
) -> str:
    return f"""
=== AUTHOR VOICE GUIDE ===

You are writing as {profile.author_name}, {profile.author_role}.

EXPERTISE AREAS:
{', '.join(profile.expertise_areas)}

MATCH THIS VOICE:
- Use these phrases naturally: {', '.join(profile.characteristic_phrases[:5])}
- AVOID these phrases: {', '.join(profile.avoided_phrases[:5])}
- Sentence style: {profile.sentence_length_preference}
- Paragraph style: {profile.paragraph_length}

TONE:
- Formality: {profile.formality_level}/10
- Humor: {profile.humor_frequency}
- Emoji usage: {profile.emoji_usage}

KNOWN OPINIONS (use when relevant):
{chr(10).join(f'- {topic}: {position}' for topic, position in list(profile.known_opinions.items())[:3])}

CONTRARIAN POSITIONS (for engaging content):
{chr(10).join(f'- {pos}' for pos in profile.contrarian_positions[:3])}

FAVORITE TOPICS (prioritize these):
{', '.join(profile.favorite_topics[:5])}

TOPICS TO AVOID:
{', '.join(profile.topics_to_avoid[:5])}

POST LENGTH: ~{profile.typical_post_length} characters

PROVEN HOOKS (use as inspiration):
{chr(10).join(f'- "{hook}"' for hook in profile.best_performing_hooks[:3])}

BEST PERFORMING STRUCTURES:
{chr(10).join(f'- {struct}' for struct in profile.best_performing_structures[:3])}

CTA STYLES THAT WORK:
{', '.join(profile.preferred_cta_styles[:3])}
"""
```

### 9.2 Add Hook Limit and Emoji Constraints to All Prompts (Lines ~7007-7183)

**Problem**: Только ENTERPRISE_CASE имеет полные constraints.

**Fix**:
```python
# Add to BASE prompt template that all types inherit:
BASE_GENERATION_CONSTRAINTS = """
CONSTRAINTS (apply to ALL content types):
- Hook MUST fit in 210 chars (before "see more" cutoff)
- Maximum {max_emojis} emojis (from style guide)
- Line breaks: After every 1-2 sentences for mobile readability
- Hashtags: 3-5 total, mix of broad and specific
- CTA: Must have clear call-to-action
"""

# Then each content type adds its specific constraints:
generation_prompts_by_type = {
    ContentType.ENTERPRISE_CASE: f"""
{BASE_GENERATION_CONSTRAINTS}

ENTERPRISE-SPECIFIC:
- Length: {{length_target}}
- Lead with the most impressive metric
- Include company name for credibility
- End with {{suggested_cta}}
""",
    ContentType.PRIMARY_SOURCE: f"""
{BASE_GENERATION_CONSTRAINTS}

PRIMARY_SOURCE-SPECIFIC:
- Simplification level: {{complexity_level}}
- Balance accessibility with accuracy
- Credit original authors
""",
    # ... etc for all types
}
```

### 9.3 Fix CTA Placeholder Replacement (Lines ~7201-7214)

**Problem**: CTA шаблоны содержат placeholder-ы без логики замены.

**Fix**:
```python
def replace_cta_placeholders(cta: str, brief: AnalysisBrief) -> str:
    """Replace placeholders in CTA templates with actual values."""
    replacements = {
        "{topic}": brief.topic or "this topic",
        "{claim}": brief.key_insights[0] if brief.key_insights else "this insight",
        "{tool}": _extract_tool_name(brief),
        "{company}": _extract_company_name(brief),
        "{metric}": _extract_key_metric(brief),
    }

    result = cta
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)

    # Warn if unreplaced placeholders remain
    if "{" in result:
        logger.warning(f"Unreplaced placeholder in CTA: {result}")

    return result

def _extract_tool_name(brief: AnalysisBrief) -> str:
    if brief.content_type == ContentType.TOOL_RELEASE:
        extraction = brief.type_specific_extraction
        return extraction.get("tool_name", "AI tool")
    return "AI tool"
```

---

## 10. MEDIUM: Scheduling System Fixes

### 10.1 Use PostStatus Enum Consistently (Lines ~23184, ~23281-23284)

**Problem**: Смешанное использование enum и string literals.

**Fix**:
```python
# Replace all string literals with enum values:

# Line 23184:
status=PostStatus.SCHEDULED,  # Was: status="scheduled"

# Line 23281:
if post.status != PostStatus.SCHEDULED:  # Was: != "scheduled"
    raise ValueError(...)

# Line 23284:
post.status = PostStatus.CANCELLED  # Was: = "cancelled"

# Line 23385-23386:
.eq("status", PostStatus.SCHEDULED.value)  # Use .value for DB queries
```

### 10.2 Add error_message Field to ScheduledPost (Lines ~23034-23055)

**Problem**: Код присваивает `post.error_message`, но поле не определено.

**Fix**:
```python
@dataclass
class ScheduledPost:
    """Post scheduled for future publication."""
    id: str
    content: str
    scheduled_time: datetime
    content_type: ContentType
    visual_asset_path: Optional[str] = None
    status: PostStatus = PostStatus.SCHEDULED

    # ADD: Missing fields
    error_message: Optional[str] = None
    retry_count: int = 0
    claimed_at: Optional[datetime] = None

    # Existing optional fields
    created_at: datetime = field(default_factory=utc_now)
    published_at: Optional[datetime] = None
```

### 10.3 Add Recovery for Stuck Posts (Lines ~23380-23388)

**Problem**: Посты могут застрять в статусе "publishing" навсегда.

**Fix**:
```python
STUCK_TIMEOUT_MINUTES = 10
MAX_RETRIES = 3

async def recover_stuck_posts(self):
    """Recover posts stuck in 'publishing' status."""
    cutoff = utc_now() - timedelta(minutes=STUCK_TIMEOUT_MINUTES)

    stuck_posts = await self.db.get_posts_by_status_and_claimed_before(
        status=PostStatus.PUBLISHING,
        claimed_before=cutoff
    )

    for post in stuck_posts:
        if post.retry_count < MAX_RETRIES:
            # Reset for retry
            post.status = PostStatus.SCHEDULED
            post.retry_count += 1
            post.claimed_at = None
            logger.warning(f"Recovering stuck post {post.id}, retry {post.retry_count}")
        else:
            # Max retries exceeded
            post.status = PostStatus.FAILED
            post.error_message = f"Max retries ({MAX_RETRIES}) exceeded"
            logger.error(f"Post {post.id} failed after {MAX_RETRIES} retries")

        await self.db.update_scheduled_post(post)

# Add to scheduler:
# self.scheduler.add_job(
#     self.recover_stuck_posts,
#     'interval',
#     minutes=5,
#     id='recover_stuck_posts'
# )
```

### 10.4 Add Advisory Lock to reschedule_post (Lines ~23287-23305)

**Problem**: Race condition - нет lock при reschedule.

**Fix**:
```python
async def reschedule_post(
    self,
    post_id: str,
    new_time: datetime
) -> ScheduledPost:
    """Reschedule a post to new time with proper locking."""

    # NEW: Validate timezone
    if new_time.tzinfo is None:
        raise ValueError("new_time must be timezone-aware")
    new_time = ensure_utc(new_time)

    # NEW: Use advisory lock like schedule_post
    async with self.db.advisory_lock(f"schedule_slot_{new_time.isoformat()}"):
        if not await self._is_slot_available(new_time):
            raise SchedulingConflictError(f"Time {new_time} not available")

        post = await self.db.get_scheduled_post(post_id)

        if post.status != PostStatus.SCHEDULED:
            raise ValueError(f"Cannot reschedule post with status {post.status}")

        post.scheduled_time = new_time
        await self.db.update_scheduled_post(post)

        return post
```

---

## 11. LOW: Code Quality Fixes

### 11.1 Remove Dead Code (Lines ~23426-23432)

**Problem**: Код после `raise` никогда не выполнится.

**Fix**:
```python
# DELETE lines 23430-23432:
# except Exception as e:
#     raise
#     await self.telegram.notify(...)  # DEAD CODE - DELETE

# The notification should happen in the caller (_check_and_publish)
```

### 11.2 Remove Unused Imports

**Fix**:
```python
# Line 239: Remove unused aiohttp if not used
# import aiohttp  # DELETE if not used

# Line 808: Remove duplicate threading import
# import threading  # DELETE - already imported at line 142
```

### 11.3 Move uuid Import to Module Level (Lines ~274-276)

**Problem**: Import внутри функции создает overhead.

**Fix**:
```python
# At module level:
import uuid

# In function:
def generate_id() -> str:
    """Generate a unique ID for database records."""
    return str(uuid.uuid4())  # No import needed
```

### 11.4 Add Exception Class Definitions

**Problem**: ValidationError, DatabaseError и др. используются но не определены.

**Fix**:
```python
# Add near top of file, after imports:

# ═══════════════════════════════════════════════════════════════════════════
# CUSTOM EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════

class AgentBaseError(Exception):
    """Base exception for all agent-related errors."""
    pass

class ValidationError(ValueError):
    """Raised when input validation fails."""
    pass

class DatabaseError(Exception):
    """Raised when database operation fails."""
    pass

class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass

class SecurityError(Exception):
    """Raised when security check fails."""
    pass

class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""
    pass

class LinkedInRateLimitError(AgentBaseError):
    """Raised when LinkedIn rate limit is hit."""
    pass

class LinkedInSessionExpiredError(AgentBaseError):
    """Raised when LinkedIn session expires."""
    pass

class LinkedInAPIError(AgentBaseError):
    """Raised for general LinkedIn API errors."""
    pass

class ImageGenerationError(AgentBaseError):
    """Raised when image generation fails."""
    pass

class SchedulingConflictError(AgentBaseError):
    """Raised when scheduling slot is unavailable."""
    pass
```

---

## Summary of Changes

### By Priority

| Priority | Category | Issues | Effort |
|----------|----------|--------|--------|
| CRITICAL | Security | 5 | High |
| CRITICAL | Database | 5 | Medium |
| CRITICAL | Safety Integration | 4 | High |
| CRITICAL | State Machine | 5 | Medium |
| HIGH | Agent Consistency | 5 | Medium |
| HIGH | Logging | 3 | Low |
| HIGH | Analytics | 3 | High |
| HIGH | Self-Improvement | 3 | High |
| MEDIUM | Author Profile | 3 | Medium |
| MEDIUM | Scheduling | 4 | Medium |
| LOW | Code Quality | 4 | Low |

### Implementation Order

1. **Week 1**: Security fixes (CRITICAL) - must be fixed before any production use
2. **Week 2**: Database schema fixes + Safety system integration
3. **Week 3**: State machine fixes + Agent consistency
4. **Week 4**: Analytics, Self-improvement, remaining HIGH items
5. **Week 5**: MEDIUM and LOW priority items

### Testing Requirements

- Unit tests for all security validation functions
- Integration tests for safety system with different autonomy levels
- End-to-end tests for full pipeline with revision loops
- Performance tests for database queries with new indexes
- Regression tests after each category of changes

---

## Appendix: Cross-Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────┐
│                     DEPENDENCY RELATIONSHIPS                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ThresholdConfig ──────────┬──────────────────────────────────────► │
│        │                   │                                        │
│        ▼                   ▼                                        │
│  AutonomyManager ◄──── QC Agent                                     │
│        │                   │                                        │
│        ▼                   ▼                                        │
│  ModificationSafety ◄── MetaAgent ◄── SelfModificationEngine       │
│        │                   │                                        │
│        ▼                   ▼                                        │
│  ContinuousLearning ◄── ResearchAgent                              │
│        │                                                            │
│        ▼                                                            │
│  PromptManager ◄──────── Writer Agent                              │
│        │                   ▲                                        │
│        ▼                   │                                        │
│  AuthorProfile ───────────┘                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Principle**: All modification paths must flow through ModificationSafetySystem and AutonomyManager.

---

*Document generated by parallel architecture analysis. Review and apply fixes in priority order.*
