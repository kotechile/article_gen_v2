-- Remove unique constraint on keyword_research_data table
-- This allows storing the same keyword multiple times for the same user and topic

-- Drop the existing unique constraint
ALTER TABLE keyword_research_data 
DROP CONSTRAINT IF EXISTS keyword_research_data_keyword_user_topic_unique;

-- Also drop the old unique constraint on just keyword if it exists
ALTER TABLE keyword_research_data 
DROP CONSTRAINT IF EXISTS keyword_research_data_keyword_key;

-- Add a comment explaining the change
COMMENT ON TABLE keyword_research_data IS 'Keyword research data table - allows duplicate keywords for same user/topic combination';
