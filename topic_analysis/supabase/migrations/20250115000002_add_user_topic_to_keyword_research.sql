-- Add user_id and topic_id columns to keyword_research_data table
-- This migration adds the necessary columns for proper data isolation

ALTER TABLE keyword_research_data
ADD COLUMN user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
ADD COLUMN topic_id UUID REFERENCES research_topics(id) ON DELETE CASCADE;

-- Add indexes for performance
CREATE INDEX IF NOT EXISTS idx_keyword_research_data_user_id ON keyword_research_data(user_id);
CREATE INDEX IF NOT EXISTS idx_keyword_research_data_topic_id ON keyword_research_data(topic_id);
CREATE INDEX IF NOT EXISTS idx_keyword_research_data_user_topic ON keyword_research_data(user_id, topic_id);

-- Update unique constraint to include user_id and topic_id
-- First, drop the old unique constraint if it exists
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'keyword_research_data_keyword_key') THEN
        ALTER TABLE keyword_research_data DROP CONSTRAINT keyword_research_data_keyword_key;
    END IF;
END
$$;

-- Add new unique constraint
ALTER TABLE keyword_research_data
ADD CONSTRAINT keyword_research_data_user_topic_keyword_unique UNIQUE (user_id, topic_id, keyword);

-- For existing data, we'll need to handle this carefully
-- For now, we'll make the columns nullable temporarily to avoid breaking existing data
-- In production, you would want to backfill these values before making them NOT NULL

-- Note: This migration assumes you have existing data that needs to be handled
-- You may need to manually backfill user_id and topic_id for existing records
-- before making these columns NOT NULL