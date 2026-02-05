-- Combined Migration Script for Supabase SQL Editor
-- Run this script directly in your Supabase SQL Editor
-- This DROPS and recreates both keywords and content_topics tables

-- ============================================================================
-- DROP EXISTING TABLES (if they exist)
-- ============================================================================

DROP TABLE IF EXISTS content_topics CASCADE;
DROP TABLE IF EXISTS keywords CASCADE;

-- ============================================================================
-- PART 1: CREATE KEYWORDS TABLE
-- ============================================================================

-- Create keywords table for expanded keyword data with profitability metrics
-- This table stores keywords expanded from seed keywords via DataForSEO API

CREATE TABLE keywords (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  research_topic_id UUID NOT NULL,
  subtopic_id UUID,
  user_id UUID NOT NULL,
  
  -- Keyword identification
  seed_keyword TEXT NOT NULL,
  keyword TEXT NOT NULL,
  
  -- Core metrics from DataForSEO
  search_volume INTEGER,
  cpc NUMERIC(10, 2),
  competition NUMERIC(5, 4),
  competition_level TEXT CHECK (competition_level IN ('LOW', 'MEDIUM', 'HIGH')),
  difficulty INTEGER CHECK (difficulty >= 0 AND difficulty <= 100),
  keyword_difficulty INTEGER CHECK (keyword_difficulty >= 0 AND keyword_difficulty <= 100),
  
  -- Intent data
  main_intent TEXT,
  intent_type TEXT,
  
  -- Additional DataForSEO fields
  low_top_of_page_bid NUMERIC(10, 2),
  high_top_of_page_bid NUMERIC(10, 2),
  categories INTEGER[],
  monthly_searches JSONB DEFAULT '[]',
  last_updated_time TIMESTAMP WITH TIME ZONE,
  
  -- Keyword properties
  core_keyword TEXT,
  synonym_clustering_algorithm TEXT,
  detected_language TEXT,
  is_another_language BOOLEAN DEFAULT FALSE,
  
  -- Search intent info
  foreign_intent TEXT[],
  search_intent_last_updated_time TIMESTAMP WITH TIME ZONE,
  
  -- Trend data
  monthly_trend INTEGER,
  quarterly_trend INTEGER,
  yearly_trend INTEGER,
  search_volume_trend JSONB DEFAULT '[]',
  
  -- Clickstream data
  clickstream_search_volume INTEGER,
  clickstream_last_updated_time TIMESTAMP WITH TIME ZONE,
  clickstream_gender_distribution JSONB DEFAULT '{}',
  clickstream_age_distribution JSONB DEFAULT '{}',
  clickstream_monthly_searches JSONB DEFAULT '[]',
  
  -- SERP info
  serp_se_type TEXT,
  serp_check_url TEXT,
  serp_item_types TEXT[],
  se_results_count BIGINT,
  serp_last_updated_time TIMESTAMP WITH TIME ZONE,
  serp_previous_updated_time TIMESTAMP WITH TIME ZONE,
  
  -- Backlinks info
  avg_backlinks INTEGER,
  avg_dofollow INTEGER,
  avg_referring_pages INTEGER,
  avg_referring_domains INTEGER,
  avg_referring_main_domains INTEGER,
  avg_rank INTEGER,
  avg_main_domain_rank INTEGER,
  backlinks_last_updated_time TIMESTAMP WITH TIME ZONE,
  
  -- Normalized data
  normalized_bing_search_volume INTEGER,
  normalized_bing_is_normalized BOOLEAN,
  normalized_bing_last_updated_time TIMESTAMP WITH TIME ZONE,
  normalized_bing_monthly_searches JSONB DEFAULT '[]',
  normalized_clickstream_search_volume INTEGER,
  normalized_clickstream_is_normalized BOOLEAN,
  normalized_clickstream_last_updated_time TIMESTAMP WITH TIME ZONE,
  normalized_clickstream_monthly_searches JSONB DEFAULT '[]',
  
  -- Profitability calculation
  profitability_score NUMERIC(10, 4),
  
  -- Metadata
  source TEXT DEFAULT 'dataforseo_keyword_ideas',
  depth INTEGER DEFAULT 0,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  
  -- Unique constraint
  CONSTRAINT keywords_topic_keyword_unique UNIQUE(research_topic_id, keyword)
);

-- Indexes for keywords table
CREATE INDEX idx_keywords_topic ON keywords(research_topic_id);
CREATE INDEX idx_keywords_subtopic ON keywords(subtopic_id);
CREATE INDEX idx_keywords_user ON keywords(user_id);
CREATE INDEX idx_keywords_profitability ON keywords(profitability_score);
CREATE INDEX idx_keywords_intent ON keywords(main_intent);
CREATE INDEX idx_keywords_seed ON keywords(seed_keyword);
CREATE INDEX idx_keywords_difficulty ON keywords(keyword_difficulty);
CREATE INDEX idx_keywords_volume ON keywords(search_volume);
CREATE INDEX idx_keywords_cpc ON keywords(cpc);
CREATE INDEX idx_keywords_profitable ON keywords(research_topic_id, profitability_score)
  WHERE profitability_score IS NOT NULL;

-- Trigger for keywords updated_at
CREATE OR REPLACE FUNCTION update_keywords_modtime()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_keywords_timestamp
    BEFORE UPDATE ON keywords
    FOR EACH ROW
    EXECUTE PROCEDURE update_keywords_modtime();

-- RLS Policies for keywords
ALTER TABLE keywords ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view their own keywords"
    ON keywords FOR SELECT
    USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can insert their own keywords"
    ON keywords FOR INSERT
    WITH CHECK (auth.uid()::text = user_id::text);

CREATE POLICY "Users can update their own keywords"
    ON keywords FOR UPDATE
    USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can delete their own keywords"
    ON keywords FOR DELETE
    USING (auth.uid()::text = user_id::text);

-- Comments for keywords table
COMMENT ON TABLE keywords IS 'Stores expanded keywords from DataForSEO with profitability metrics';
COMMENT ON COLUMN keywords.profitability_score IS 'Calculated as (log(volume) × cpc) / difficulty';
COMMENT ON COLUMN keywords.seed_keyword IS 'Original seed keyword that generated this expanded keyword';
COMMENT ON COLUMN keywords.keyword IS 'The expanded keyword from DataForSEO';

-- ============================================================================
-- PART 2: CREATE CONTENT_TOPICS TABLE
-- ============================================================================

-- Create content_topics table for clustered content topic suggestions
CREATE TABLE content_topics (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  research_topic_id UUID NOT NULL,
  user_id UUID NOT NULL,
  
  -- Content topic details
  title TEXT NOT NULL,
  description TEXT,
  
  -- Keyword relationships
  primary_keyword_id UUID,
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

-- Indexes for content_topics table
CREATE INDEX idx_content_topics_topic ON content_topics(research_topic_id);
CREATE INDEX idx_content_topics_user ON content_topics(user_id);
CREATE INDEX idx_content_topics_profitability ON content_topics(estimated_profitability_score);
CREATE INDEX idx_content_topics_intent ON content_topics(intent_type);
CREATE INDEX idx_content_topics_status ON content_topics(status);
CREATE INDEX idx_content_topics_priority ON content_topics(priority_score);
CREATE INDEX idx_content_topics_primary_keyword ON content_topics(primary_keyword_id);
CREATE INDEX idx_content_topics_topic_profit ON content_topics(research_topic_id, estimated_profitability_score)
  WHERE status = 'suggested';
CREATE INDEX idx_content_topics_supporting_keywords ON content_topics USING GIN(supporting_keyword_ids);

-- Trigger for content_topics updated_at
CREATE OR REPLACE FUNCTION update_content_topics_modtime()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_content_topics_timestamp
    BEFORE UPDATE ON content_topics
    FOR EACH ROW
    EXECUTE PROCEDURE update_content_topics_modtime();

-- RLS Policies for content_topics
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

-- Comments for content_topics table
COMMENT ON TABLE content_topics IS 'Stores clustered content topic suggestions generated from profitable keywords';
COMMENT ON COLUMN content_topics.title IS 'Blog post title or content topic suggestion';
COMMENT ON COLUMN content_topics.primary_keyword_id IS 'The main keyword to target (highest profitability)';
COMMENT ON COLUMN content_topics.supporting_keyword_ids IS 'Array of related keyword IDs to include in content';
COMMENT ON COLUMN content_topics.estimated_profitability_score IS 'Aggregated profitability score from all keywords';
COMMENT ON COLUMN content_topics.intent_type IS 'Primary search intent: Commercial, Informational, Transactional, or Comparison';

-- ============================================================================
-- VERIFICATION
-- ============================================================================

-- Verify tables were created
SELECT 'keywords' as table_name, COUNT(*) as column_count 
FROM information_schema.columns 
WHERE table_name = 'keywords'
UNION ALL
SELECT 'content_topics' as table_name, COUNT(*) as column_count 
FROM information_schema.columns 
WHERE table_name = 'content_topics';
