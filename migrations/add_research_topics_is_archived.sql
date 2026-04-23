-- Add soft-archive flag for research topics shown in All Research.
-- Archived topics are hidden from the active section but remain available for restore.
ALTER TABLE research_topics
ADD COLUMN IF NOT EXISTS is_archived BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_research_topics_is_archived
ON research_topics (is_archived);
