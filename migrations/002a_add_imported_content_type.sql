-- =============================================================================
-- MIGRATION 002: Add 'imported' content type
-- =============================================================================
-- Allows ProfileImporter to save imported LinkedIn posts into the posts table
-- alongside system-generated posts.
-- =============================================================================

ALTER TABLE posts DROP CONSTRAINT valid_content_type;
ALTER TABLE posts ADD CONSTRAINT valid_content_type CHECK (content_type IN (
    'enterprise_case', 'primary_source', 'automation_case',
    'community_content', 'tool_release', 'imported'
));
