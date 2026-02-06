-- Migration 005: Add reshare metadata columns to posts table
-- Apply via Supabase SQL Editor or psql.

ALTER TABLE posts ADD COLUMN IF NOT EXISTS is_reshare BOOLEAN DEFAULT FALSE;
ALTER TABLE posts ADD COLUMN IF NOT EXISTS original_author TEXT DEFAULT '';
