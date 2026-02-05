-- Enhance content_ideas table
-- This migration adds research_topic_id foreign key and idea_type field

-- Add research_topic_id foreign key column
ALTER TABLE content_ideas 
ADD COLUMN IF NOT EXISTS research_topic_id UUID REFERENCES research_topics(id) ON DELETE CASCADE;

-- Add idea_type column for content classification
ALTER TABLE content_ideas 
ADD COLUMN IF NOT EXISTS idea_type VARCHAR(50) NOT NULL DEFAULT 'trending' 
CHECK (idea_type IN ('trending', 'evergreen', 'seasonal', 'newsjacking', 'how_to', 'listicle', 'case_study', 'opinion', 'comparison', 'review'));

-- Create index for research_topic_id foreign key
CREATE INDEX IF NOT EXISTS idx_content_ideas_research_topic_id 
ON content_ideas(research_topic_id);

-- Create index for idea_type
CREATE INDEX IF NOT EXISTS idx_content_ideas_idea_type 
ON content_ideas(idea_type);

-- Add comments
COMMENT ON COLUMN content_ideas.research_topic_id IS 'Foreign key to research_topics table';
COMMENT ON COLUMN content_ideas.idea_type IS 'Classification of content idea type';
