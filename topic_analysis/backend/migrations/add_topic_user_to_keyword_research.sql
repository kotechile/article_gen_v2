-- Add topic_id and user_id columns to keyword_research_data table
-- This migration adds the necessary foreign key relationships

-- Add columns if they don't exist
ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS topic_id UUID,
ADD COLUMN IF NOT EXISTS user_id UUID;

-- Add indexes for these new columns
CREATE INDEX IF NOT EXISTS idx_keyword_research_data_topic_id ON keyword_research_data(topic_id);
CREATE INDEX IF NOT EXISTS idx_keyword_research_data_user_id ON keyword_research_data(user_id);
CREATE INDEX IF NOT EXISTS idx_keyword_research_data_user_topic ON keyword_research_data(user_id, topic_id);

-- Add foreign key constraints if the referenced tables exist
DO $$ 
BEGIN
  -- Check if research_topics table exists and add foreign key if it does
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'research_topics') THEN
    ALTER TABLE keyword_research_data 
    ADD CONSTRAINT keyword_research_data_topic_id_fkey 
    FOREIGN KEY (topic_id) REFERENCES research_topics(id) ON DELETE CASCADE;
  END IF;
  
  -- Check if users table exists and add foreign key if it does
  IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'users') THEN
    ALTER TABLE keyword_research_data 
    ADD CONSTRAINT keyword_research_data_user_id_fkey 
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
  END IF;
END $$;

-- Add comments
COMMENT ON COLUMN keyword_research_data.topic_id IS 'Reference to research topic';
COMMENT ON COLUMN keyword_research_data.user_id IS 'Reference to user';









