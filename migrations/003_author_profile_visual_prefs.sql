-- Migration 003: Add visual content preference columns to author_profiles
-- Apply via Supabase SQL Editor or psql.

ALTER TABLE author_profiles ADD COLUMN IF NOT EXISTS visual_content_ratio FLOAT DEFAULT 0.0;
ALTER TABLE author_profiles ADD COLUMN IF NOT EXISTS preferred_visual_types TEXT[] DEFAULT '{}';
ALTER TABLE author_profiles ADD COLUMN IF NOT EXISTS visual_type_performance JSONB DEFAULT '{}';

-- Add visual_type and visual_url columns to posts table for imported media info
ALTER TABLE posts ADD COLUMN IF NOT EXISTS visual_type TEXT DEFAULT 'none';
ALTER TABLE posts ADD COLUMN IF NOT EXISTS visual_url TEXT DEFAULT '';
