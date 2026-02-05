-- Migration 004: Add rich content metadata columns to posts table
-- Covers: shares, article info, document/carousel info, video info, reaction breakdown
-- Apply via Supabase SQL Editor or psql.

-- Engagement: shares count and reaction type breakdown
ALTER TABLE posts ADD COLUMN IF NOT EXISTS shares INTEGER DEFAULT 0;
ALTER TABLE posts ADD COLUMN IF NOT EXISTS reactions_by_type JSONB DEFAULT '{}';

-- Article posts: external URL and title
ALTER TABLE posts ADD COLUMN IF NOT EXISTS article_url TEXT DEFAULT '';
ALTER TABLE posts ADD COLUMN IF NOT EXISTS article_title TEXT DEFAULT '';

-- Document posts (carousels/PDFs): title, transcribed doc URL, page count
ALTER TABLE posts ADD COLUMN IF NOT EXISTS document_title TEXT DEFAULT '';
ALTER TABLE posts ADD COLUMN IF NOT EXISTS document_url TEXT DEFAULT '';
ALTER TABLE posts ADD COLUMN IF NOT EXISTS page_count INTEGER DEFAULT 0;

-- Video posts: duration in seconds, thumbnail URL
ALTER TABLE posts ADD COLUMN IF NOT EXISTS video_duration FLOAT DEFAULT 0;
ALTER TABLE posts ADD COLUMN IF NOT EXISTS video_thumbnail TEXT DEFAULT '';
