-- Add missing columns to keyword_research_data table
-- This migration fixes the issue where user_id and topic_id columns are missing

-- Add user_id column
ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES users(id) ON DELETE CASCADE;

-- Add topic_id column  
ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS topic_id UUID REFERENCES research_topics(id) ON DELETE CASCADE;

-- Add source column to track where the keyword data came from
ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS source VARCHAR(50) DEFAULT 'dataforseo';

-- Add competition column (different from competition_value)
ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS competition DECIMAL(3,2);

-- Add seed_keywords column to store the original seed keywords used
ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS seed_keywords JSONB DEFAULT '[]';

-- Update the unique constraint to include user_id and topic_id
-- First drop the existing unique constraint on keyword
ALTER TABLE keyword_research_data DROP CONSTRAINT IF EXISTS keyword_research_data_keyword_key;

-- Add new unique constraint that includes user_id and topic_id
-- This allows the same keyword to exist for different users/topics
ALTER TABLE keyword_research_data 
ADD CONSTRAINT keyword_research_data_keyword_user_topic_unique 
UNIQUE (keyword, user_id, topic_id);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_keyword_research_data_user_id 
ON keyword_research_data(user_id);

CREATE INDEX IF NOT EXISTS idx_keyword_research_data_topic_id 
ON keyword_research_data(topic_id);

CREATE INDEX IF NOT EXISTS idx_keyword_research_data_user_topic 
ON keyword_research_data(user_id, topic_id);

-- Add RLS policies for the new columns
-- Enable RLS if not already enabled
ALTER TABLE keyword_research_data ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist
DROP POLICY IF EXISTS "Users can view their own keyword research data" ON keyword_research_data;
DROP POLICY IF EXISTS "Users can insert their own keyword research data" ON keyword_research_data;
DROP POLICY IF EXISTS "Users can update their own keyword research data" ON keyword_research_data;
DROP POLICY IF EXISTS "Users can delete their own keyword research data" ON keyword_research_data;

-- Create new RLS policies
CREATE POLICY "Users can view their own keyword research data" ON keyword_research_data
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own keyword research data" ON keyword_research_data
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own keyword research data" ON keyword_research_data
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own keyword research data" ON keyword_research_data
    FOR DELETE USING (auth.uid() = user_id);

-- Add comments for documentation
COMMENT ON COLUMN keyword_research_data.user_id IS 'User who owns this keyword research data';
COMMENT ON COLUMN keyword_research_data.topic_id IS 'Research topic this keyword belongs to';
COMMENT ON COLUMN keyword_research_data.source IS 'Source of the keyword data (dataforseo, ahrefs, manual, etc.)';
COMMENT ON COLUMN keyword_research_data.competition IS 'Competition level as decimal (0.0-1.0)';
COMMENT ON COLUMN keyword_research_data.seed_keywords IS 'Original seed keywords used to generate this keyword';
