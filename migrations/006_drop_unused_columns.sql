-- Migration 006: Drop unused columns from posts table
-- visual_url was a duplicate of visual_urls[0]
-- title was not used (hook serves the same purpose)

ALTER TABLE posts DROP COLUMN IF EXISTS visual_url;
ALTER TABLE posts DROP COLUMN IF EXISTS title;
