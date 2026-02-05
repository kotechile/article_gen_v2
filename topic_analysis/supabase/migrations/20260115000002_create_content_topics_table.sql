-- Create content_topics table for clustered content topic suggestions
-- This table stores blog title suggestions generated from clustered keywords
-- Note: Foreign key constraints removed to allow standalone creation

CREATE TABLE IF NOT EXISTS content_topics (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  research_topic_id UUID NOT NULL,  -- Will reference research_topics(id) when that table exists
  user_id UUID NOT NULL, -- references auth.users(id)
  
  -- Content topic details
  title TEXT NOT NULL,
  description TEXT,
  
  -- Keyword relationships
  primary_keyword_id UUID,  -- Will reference keywords(id) when needed
  supporting_keyword_ids UUID[] DEFAULT '{}',
  
  -- Metrics
  estimated_profitability_score NUMERIC(10, 4),
  total_search_volume INTEGER,
  average_cpc NUMERIC(10, 2),
  average_difficulty NUMERIC(5, 2),
  
  -- Intent classification
  intent_type TEXT,
  
  -- Content planning
  content_type TEXT CHECK (content_type IN ('blog_post', 'guide', 'comparison', 'review', 'tutorial', 'listicle')),
  target_word_count INTEGER,
  priority_score NUMERIC(5, 2) CHECK (priority_score IS NULL OR (priority_score >= 0 AND priority_score <= 100)),
  
  -- Status tracking
  status TEXT DEFAULT 'suggested' CHECK (status IN ('suggested', 'planned', 'in_progress', 'published', 'archived')),
  
  -- Metadata
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  
  CONSTRAINT content_topics_title_check CHECK (LENGTH(title) > 0)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_content_topics_topic ON content_topics(research_topic_id);
CREATE INDEX IF NOT EXISTS idx_content_topics_user ON content_topics(user_id);
CREATE INDEX IF NOT EXISTS idx_content_topics_profitability ON content_topics(estimated_profitability_score);
CREATE INDEX IF NOT EXISTS idx_content_topics_intent ON content_topics(intent_type);
CREATE INDEX IF NOT EXISTS idx_content_topics_status ON content_topics(status);
CREATE INDEX IF NOT EXISTS idx_content_topics_priority ON content_topics(priority_score);
CREATE INDEX IF NOT EXISTS idx_content_topics_primary_keyword ON content_topics(primary_keyword_id);

-- Composite index for filtering by topic and profitability
CREATE INDEX IF NOT EXISTS idx_content_topics_topic_profit ON content_topics(research_topic_id, estimated_profitability_score)
  WHERE status = 'suggested';

-- GIN index for supporting_keyword_ids array queries
CREATE INDEX IF NOT EXISTS idx_content_topics_supporting_keywords ON content_topics USING GIN(supporting_keyword_ids);

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_content_topics_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_content_topics_updated_at ON content_topics;

CREATE TRIGGER update_content_topics_updated_at
    BEFORE UPDATE ON content_topics
    FOR EACH ROW
    EXECUTE PROCEDURE update_content_topics_updated_at_column();

-- RLS Policies
ALTER TABLE content_topics ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view their own content topics"
    ON content_topics FOR SELECT
    USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can insert their own content topics"
    ON content_topics FOR INSERT
    WITH CHECK (auth.uid()::text = user_id::text);

CREATE POLICY "Users can update their own content topics"
    ON content_topics FOR UPDATE
    USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can delete their own content topics"
    ON content_topics FOR DELETE
    USING (auth.uid()::text = user_id::text);

-- Add helpful comments
COMMENT ON TABLE content_topics IS 'Stores clustered content topic suggestions generated from profitable keywords';
COMMENT ON COLUMN content_topics.title IS 'Blog post title or content topic suggestion';
COMMENT ON COLUMN content_topics.primary_keyword_id IS 'The main keyword to target (highest profitability)';
COMMENT ON COLUMN content_topics.supporting_keyword_ids IS 'Array of related keyword IDs to include in content';
COMMENT ON COLUMN content_topics.estimated_profitability_score IS 'Aggregated profitability score from all keywords';
COMMENT ON COLUMN content_topics.intent_type IS 'Primary search intent: Commercial, Informational, Transactional, or Comparison';
