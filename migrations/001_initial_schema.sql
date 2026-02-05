-- =============================================================================
-- MIGRATION 001: Initial Schema for LinkedIn Super Agent
-- =============================================================================
-- Run this in Supabase SQL Editor to create all tables, indexes, and functions.
--
-- Table creation order respects foreign key dependencies:
--   1. topic_cache       (no FK dependencies)
--   2. posts             (FK -> topic_cache)
--   3. post_metrics      (FK -> posts)
--   4. learnings         (no FK dependencies)
--   5. code_modifications(no FK dependencies)
--   6. experiments       (no FK dependencies)
--   7. research_reports  (no FK dependencies)
--   8. prompts           (no FK dependencies)
--   9. author_photos     (FK -> posts)
--  10. drafts            (FK -> topic_cache)
--  11. pipeline_errors   (no FK dependencies)
--  12. agent_logs        (no FK dependencies)
--
-- After all tables: RPC functions, composite indexes, and ALTER TABLE fixes.
-- =============================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- 1. TOPIC_CACHE: Cached trending topics
--    Created first because posts and drafts reference it via foreign key.
-- =============================================================================
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
CREATE INDEX idx_topic_cache_expires ON topic_cache(expires_at) WHERE expires_at IS NOT NULL;

-- =============================================================================
-- 2. POSTS: Published LinkedIn posts
-- =============================================================================
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
    topic_id UUID,                          -- FK to topic_cache (UUID from the start)
    template_used TEXT,
    hook_style TEXT,
    revision_count INTEGER DEFAULT 0,

    -- Learning context
    learnings_used JSONB DEFAULT '[]',      -- Which learnings were applied

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    published_at TIMESTAMPTZ,

    -- Constraints
    CONSTRAINT valid_content_type CHECK (content_type IN (
        'enterprise_case', 'primary_source', 'automation_case',
        'community_content', 'tool_release'
    )),

    -- Foreign keys
    CONSTRAINT fk_posts_topic
        FOREIGN KEY (topic_id) REFERENCES topic_cache(id) ON DELETE SET NULL
);

CREATE INDEX idx_posts_content_type ON posts(content_type);
CREATE INDEX idx_posts_created_at ON posts(created_at DESC);
CREATE INDEX idx_posts_qc_score ON posts(qc_score DESC);
CREATE INDEX idx_posts_topic_id ON posts(topic_id);
CREATE INDEX idx_posts_published_at ON posts(published_at DESC) WHERE published_at IS NOT NULL;

-- =============================================================================
-- 3. POST_METRICS: Analytics snapshots over time
-- =============================================================================
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

-- =============================================================================
-- 4. LEARNINGS: Micro-learnings from continuous learning engine
-- =============================================================================
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
CREATE INDEX idx_learnings_content_type ON learnings(content_type);
CREATE INDEX idx_learnings_component_type ON learnings(affected_component, content_type);

-- =============================================================================
-- 5. CODE_MODIFICATIONS: Self-modifying code history
-- =============================================================================
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

-- =============================================================================
-- 6. EXPERIMENTS: A/B testing
-- =============================================================================
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

-- =============================================================================
-- 7. RESEARCH_REPORTS: Deep research results
-- =============================================================================
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

-- =============================================================================
-- 8. PROMPTS: Versioned prompts for all components
-- =============================================================================
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

-- =============================================================================
-- 9. AUTHOR_PHOTOS: Author photo library for post personalization
-- =============================================================================
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

    -- Additional fields (DATABASE FIX 2.1)
    notes TEXT,
    dominant_colors TEXT[] DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT author_photos_file_path_unique UNIQUE (file_path),
    CONSTRAINT valid_eye_contact CHECK (eye_contact IN ('direct', 'away', 'profile'))
);

CREATE INDEX idx_author_photos_disabled ON author_photos(disabled) WHERE disabled = FALSE;
CREATE INDEX idx_author_photos_usage ON author_photos(times_used ASC);
CREATE INDEX idx_author_photos_setting ON author_photos(setting);
CREATE INDEX idx_author_photos_pose ON author_photos(pose);
CREATE INDEX idx_author_photos_mood ON author_photos(mood);
CREATE INDEX idx_author_photos_suitable_for ON author_photos USING GIN(suitable_for);

-- =============================================================================
-- 10. DRAFTS: Work in progress posts
-- =============================================================================
CREATE TABLE drafts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- References
    topic_id UUID,                          -- FK to topic_cache (UUID from the start)
    content_type TEXT NOT NULL,

    -- Structured content
    hook TEXT NOT NULL,                     -- First line (must fit in 210 chars)
    body TEXT NOT NULL,                     -- Main content
    cta TEXT,                               -- Call to action
    hashtags TEXT[] DEFAULT '{}',           -- Array of hashtags
    full_text TEXT NOT NULL,                -- Combined, formatted

    -- Template metadata
    template_used TEXT,
    template_category TEXT,                 -- universal / enterprise / research / automation / etc.
    hook_style TEXT,                        -- metrics / lessons / contrarian / how_to / etc.

    -- Content metrics
    character_count INTEGER,
    estimated_read_time TEXT,
    hook_in_limit BOOLEAN DEFAULT TRUE,     -- Is hook under 210 chars?
    length_in_range BOOLEAN DEFAULT TRUE,   -- Is total length in target range?

    -- Type-specific data
    type_data_injected JSONB DEFAULT '{}',  -- What extraction data was used

    -- Visual brief for next agents
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
    )),

    -- Foreign keys
    CONSTRAINT fk_drafts_topic
        FOREIGN KEY (topic_id) REFERENCES topic_cache(id) ON DELETE SET NULL
);

CREATE INDEX idx_drafts_stage ON drafts(stage);
CREATE INDEX idx_drafts_topic_id ON drafts(topic_id);
CREATE INDEX idx_drafts_content_type ON drafts(content_type);

-- =============================================================================
-- 11. PIPELINE_ERRORS: Error tracking for post-mortem analysis
-- =============================================================================
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

-- =============================================================================
-- 12. AGENT_LOGS: Structured logging storage for AgentLogger
-- =============================================================================
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

CREATE INDEX idx_agent_logs_timestamp ON agent_logs(timestamp DESC);
CREATE INDEX idx_agent_logs_level ON agent_logs(level);
CREATE INDEX idx_agent_logs_component ON agent_logs(component);
CREATE INDEX idx_agent_logs_run_id ON agent_logs(run_id) WHERE run_id IS NOT NULL;
CREATE INDEX idx_agent_logs_error ON agent_logs(error_type) WHERE error_type IS NOT NULL;

-- =============================================================================
-- RPC FUNCTIONS: Aggregation queries and utilities
-- =============================================================================

-- Retention policy: auto-delete logs older than N days (default 30)
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

-- Average QC score across all posts
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
-- DATABASE FIX 2.3: Handles empty data case properly (returns 50.0 as default)
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

-- Performance gaps by content type (for research agent)
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

-- Atomic prompt save (prevents race conditions with advisory lock)
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

-- Increment photo usage counter atomically
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

-- =============================================================================
-- COMPOSITE INDEXES: Additional performance indexes (DATABASE FIX 2.4)
-- =============================================================================

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

-- =============================================================================
-- ROW LEVEL SECURITY (optional, for multi-tenant)
-- =============================================================================
-- For single-user agent, RLS is not needed.
-- If you want to support multiple users in the future:
-- ALTER TABLE posts ENABLE ROW LEVEL SECURITY;
-- CREATE POLICY "Users can only see their own posts" ON posts
--     FOR ALL USING (auth.uid() = user_id);
