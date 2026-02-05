-- =============================================================================
-- MIGRATION 002: Additional tables missing from initial migration
-- =============================================================================
-- These tables are used by the codebase but were not included in 001.
-- Tables: pending_approvals, scheduled_posts, author_profiles
-- =============================================================================

-- =============================================================================
-- 1. PENDING_APPROVALS: Telegram approval workflow
-- =============================================================================
CREATE TABLE pending_approvals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- What needs approval
    draft_id UUID REFERENCES drafts(id) ON DELETE CASCADE,
    run_id TEXT,

    -- Content snapshot
    content_preview TEXT NOT NULL,
    content_type TEXT,
    qc_score FLOAT,

    -- Visual
    visual_url TEXT,
    visual_type TEXT,

    -- Approval state
    status TEXT DEFAULT 'pending',
    decision_by TEXT,
    decision_reason TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    decided_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ DEFAULT NOW() + INTERVAL '24 hours',

    CONSTRAINT valid_approval_status CHECK (status IN (
        'pending', 'approved', 'rejected', 'expired'
    ))
);

CREATE INDEX idx_pending_approvals_status ON pending_approvals(status);
CREATE INDEX idx_pending_approvals_created ON pending_approvals(created_at DESC);
CREATE INDEX idx_pending_approvals_expires ON pending_approvals(expires_at) WHERE status = 'pending';

-- =============================================================================
-- 2. SCHEDULED_POSTS: Publishing scheduler queue
-- =============================================================================
CREATE TABLE scheduled_posts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- References
    draft_id UUID REFERENCES drafts(id) ON DELETE SET NULL,
    post_id UUID REFERENCES posts(id) ON DELETE SET NULL,
    approval_id UUID REFERENCES pending_approvals(id) ON DELETE SET NULL,

    -- Scheduling
    scheduled_for TIMESTAMPTZ NOT NULL,
    timezone TEXT DEFAULT 'UTC',
    slot_type TEXT,

    -- Content snapshot (in case draft changes)
    content_snapshot TEXT,
    visual_url TEXT,

    -- Status
    status TEXT DEFAULT 'scheduled',

    -- Publishing result
    linkedin_post_id TEXT,
    published_at TIMESTAMPTZ,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT valid_scheduled_status CHECK (status IN (
        'scheduled', 'publishing', 'published', 'failed', 'cancelled'
    ))
);

CREATE INDEX idx_scheduled_posts_status ON scheduled_posts(status);
CREATE INDEX idx_scheduled_posts_scheduled_for ON scheduled_posts(scheduled_for);
CREATE INDEX idx_scheduled_posts_upcoming ON scheduled_posts(scheduled_for)
    WHERE status = 'scheduled';

-- =============================================================================
-- 3. AUTHOR_PROFILES: Author identity and writing preferences
-- =============================================================================
CREATE TABLE author_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Identity
    name TEXT NOT NULL,
    linkedin_url TEXT,

    -- Professional info
    title TEXT,
    company TEXT,
    industry TEXT,
    expertise_areas TEXT[] DEFAULT '{}',

    -- Writing style profile
    tone TEXT DEFAULT 'professional',
    vocabulary_level TEXT DEFAULT 'advanced',
    emoji_usage TEXT DEFAULT 'minimal',
    preferred_post_length TEXT DEFAULT 'medium',
    signature_phrases TEXT[] DEFAULT '{}',

    -- Content preferences
    preferred_content_types TEXT[] DEFAULT '{}',
    topics_of_interest TEXT[] DEFAULT '{}',
    avoided_topics TEXT[] DEFAULT '{}',

    -- Engagement patterns
    posting_frequency TEXT DEFAULT 'daily',
    best_posting_times JSONB DEFAULT '[]',

    -- Active profile
    is_active BOOLEAN DEFAULT TRUE,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_author_profiles_active ON author_profiles(is_active) WHERE is_active = TRUE;
