-- Add comprehensive keyword fields to keyword_research_data table
-- This migration adds all the fields that the backend is trying to use

-- Add additional fields that the backend code expects
ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS difficulty DECIMAL(5,2);

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS competition_level VARCHAR(50);

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS low_top_of_page_bid DECIMAL(10,4);

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS high_top_of_page_bid DECIMAL(10,4);

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS main_intent VARCHAR(50);

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS monthly_trend JSONB DEFAULT '{}';

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS quarterly_trend JSONB DEFAULT '{}';

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS yearly_trend JSONB DEFAULT '{}';

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS avg_backlinks INTEGER;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS avg_referring_domains INTEGER;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS last_updated_time TIMESTAMP WITH TIME ZONE;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS related_keyword VARCHAR(500);

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS seed_keyword VARCHAR(500);

-- Update the unique constraint to handle the new fields properly
-- First drop the existing constraint
ALTER TABLE keyword_research_data DROP CONSTRAINT IF EXISTS keyword_research_data_keyword_user_topic_unique;

-- Add new unique constraint that allows same keyword for different users/topics
ALTER TABLE keyword_research_data 
ADD CONSTRAINT keyword_research_data_keyword_user_topic_unique 
UNIQUE (keyword, user_id, topic_id);

-- Create additional indexes for performance
CREATE INDEX IF NOT EXISTS idx_keyword_research_data_source 
ON keyword_research_data(source);

CREATE INDEX IF NOT EXISTS idx_keyword_research_data_competition_level 
ON keyword_research_data(competition_level);

CREATE INDEX IF NOT EXISTS idx_keyword_research_data_main_intent 
ON keyword_research_data(main_intent);

CREATE INDEX IF NOT EXISTS idx_keyword_research_data_created_at 
ON keyword_research_data(created_at);

-- Add comments for documentation
COMMENT ON COLUMN keyword_research_data.difficulty IS 'Keyword difficulty as decimal (0.0-100.0)';
COMMENT ON COLUMN keyword_research_data.competition_level IS 'Competition level category (low, medium, high)';
COMMENT ON COLUMN keyword_research_data.low_top_of_page_bid IS 'Low top of page bid in USD';
COMMENT ON COLUMN keyword_research_data.high_top_of_page_bid IS 'High top of page bid in USD';
COMMENT ON COLUMN keyword_research_data.main_intent IS 'Main search intent category';
COMMENT ON COLUMN keyword_research_data.monthly_trend IS 'Monthly trend data as JSON';
COMMENT ON COLUMN keyword_research_data.quarterly_trend IS 'Quarterly trend data as JSON';
COMMENT ON COLUMN keyword_research_data.yearly_trend IS 'Yearly trend data as JSON';
COMMENT ON COLUMN keyword_research_data.avg_backlinks IS 'Average number of backlinks';
COMMENT ON COLUMN keyword_research_data.avg_referring_domains IS 'Average number of referring domains';
COMMENT ON COLUMN keyword_research_data.last_updated_time IS 'Last time the data was updated';
COMMENT ON COLUMN keyword_research_data.related_keyword IS 'Related keyword that generated this keyword';
COMMENT ON COLUMN keyword_research_data.seed_keyword IS 'Seed keyword that generated this keyword';
