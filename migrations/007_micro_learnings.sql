-- Migration 004: Micro Learnings table for ContinuousLearningEngine
--
-- This table stores small, incremental learnings extracted from every
-- evaluation iteration. High-confidence learnings are promoted to rules
-- that influence content generation.
--
-- Architecture reference: architecture.md lines 14564-15255

CREATE TABLE IF NOT EXISTS micro_learnings (
    id TEXT PRIMARY KEY,

    -- Learning classification
    learning_type TEXT NOT NULL CHECK (learning_type IN (
        'hook_pattern', 'visual_style', 'content_structure',
        'tone_adjustment', 'timing_insight', 'audience_preference'
    )),
    source TEXT NOT NULL CHECK (source IN (
        'meta_evaluation', 'qc_feedback', 'post_performance',
        'competitor_analysis', 'explicit_rule'
    )),

    -- Learning content
    description TEXT NOT NULL,
    rule TEXT NOT NULL,
    affected_component TEXT NOT NULL CHECK (affected_component IN (
        'writer', 'humanizer', 'visual_creator', 'trend_scout',
        'analyzer', 'qc', 'scheduler'
    )),

    -- Confidence tracking
    confidence FLOAT NOT NULL DEFAULT 0.4 CHECK (confidence >= 0 AND confidence <= 1),
    confirmations INT NOT NULL DEFAULT 0,
    contradictions INT NOT NULL DEFAULT 0,

    -- Context
    content_type TEXT,  -- NULL = applies to all types

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_confirmed_at TIMESTAMPTZ,

    -- Status flags
    is_promoted_to_rule BOOLEAN NOT NULL DEFAULT FALSE,
    is_bootstrap BOOLEAN NOT NULL DEFAULT FALSE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

-- Index for efficient querying by component
CREATE INDEX IF NOT EXISTS idx_micro_learnings_component
    ON micro_learnings(affected_component, is_active);

-- Index for confidence-based retrieval
CREATE INDEX IF NOT EXISTS idx_micro_learnings_confidence
    ON micro_learnings(confidence DESC)
    WHERE is_active = TRUE;

-- Index for content type filtering
CREATE INDEX IF NOT EXISTS idx_micro_learnings_content_type
    ON micro_learnings(content_type, is_active)
    WHERE content_type IS NOT NULL;

-- Trigger to update last_confirmed_at when confirmations increase
CREATE OR REPLACE FUNCTION update_last_confirmed()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.confirmations > OLD.confirmations THEN
        NEW.last_confirmed_at = NOW();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_update_last_confirmed ON micro_learnings;
CREATE TRIGGER trg_update_last_confirmed
    BEFORE UPDATE ON micro_learnings
    FOR EACH ROW
    EXECUTE FUNCTION update_last_confirmed();

-- Comments for documentation
COMMENT ON TABLE micro_learnings IS 'Incremental learnings from ContinuousLearningEngine';
COMMENT ON COLUMN micro_learnings.learning_type IS 'Category of learning (hook_pattern, visual_style, etc.)';
COMMENT ON COLUMN micro_learnings.confidence IS 'Confidence score 0.0-1.0, grows with confirmations';
COMMENT ON COLUMN micro_learnings.is_promoted_to_rule IS 'True if confidence >= 0.9 and confirmations >= 5';
COMMENT ON COLUMN micro_learnings.is_bootstrap IS 'True if this is an initial best-practice learning';
