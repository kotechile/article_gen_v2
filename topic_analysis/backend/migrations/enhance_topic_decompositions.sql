-- Enhance topic_decompositions table
-- This migration adds research_topic_id foreign key and original_topic_included field

-- Add research_topic_id foreign key column
ALTER TABLE topic_decompositions 
ADD COLUMN IF NOT EXISTS research_topic_id UUID REFERENCES research_topics(id) ON DELETE CASCADE;

-- Add original_topic_included boolean column
ALTER TABLE topic_decompositions 
ADD COLUMN IF NOT EXISTS original_topic_included BOOLEAN NOT NULL DEFAULT TRUE;

-- Create index for research_topic_id foreign key
CREATE INDEX IF NOT EXISTS idx_topic_decompositions_research_topic_id 
ON topic_decompositions(research_topic_id);

-- Create unique constraint for research topic and search query combination
CREATE UNIQUE INDEX IF NOT EXISTS idx_topic_decompositions_research_topic_search 
ON topic_decompositions(research_topic_id, search_query);

-- Add comment
COMMENT ON COLUMN topic_decompositions.research_topic_id IS 'Foreign key to research_topics table';
COMMENT ON COLUMN topic_decompositions.original_topic_included IS 'Whether original topic is included as subtopic';
