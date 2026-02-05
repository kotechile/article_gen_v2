-- Create keywords table for expanded keyword data with profitability metrics
-- This table stores keywords expanded from seed keywords via DataForSEO API
-- Note: Foreign key constraints removed to allow standalone creation

CREATE TABLE IF NOT EXISTS keywords (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  research_topic_id UUID NOT NULL,  -- Will reference research_topics(id) when that table exists
  subtopic_id UUID,  -- Will reference subtopics(id) when needed
  user_id UUID NOT NULL, -- references auth.users(id)
  
  -- Keyword identification
  seed_keyword TEXT NOT NULL,
  keyword TEXT NOT NULL,
  
  -- Core metrics from DataForSEO
  search_volume INTEGER,
  cpc NUMERIC(10, 2),
  competition NUMERIC(5, 4), -- 0.0000 to 1.0000
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
  
  -- Unique constraint: one keyword per topic
  CONSTRAINT keywords_topic_keyword_unique UNIQUE(research_topic_id, keyword)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_keywords_topic ON keywords(research_topic_id);
CREATE INDEX IF NOT EXISTS idx_keywords_subtopic ON keywords(subtopic_id);
CREATE INDEX IF NOT EXISTS idx_keywords_user ON keywords(user_id);
CREATE INDEX IF NOT EXISTS idx_keywords_profitability ON keywords(profitability_score);
CREATE INDEX IF NOT EXISTS idx_keywords_intent ON keywords(main_intent);
CREATE INDEX IF NOT EXISTS idx_keywords_seed ON keywords(seed_keyword);
CREATE INDEX IF NOT EXISTS idx_keywords_difficulty ON keywords(keyword_difficulty);
CREATE INDEX IF NOT EXISTS idx_keywords_volume ON keywords(search_volume);
CREATE INDEX IF NOT EXISTS idx_keywords_cpc ON keywords(cpc);

-- Composite index for filtering profitable keywords
CREATE INDEX IF NOT EXISTS idx_keywords_profitable ON keywords(research_topic_id, profitability_score)
  WHERE profitability_score IS NOT NULL;

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_keywords_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_keywords_updated_at ON keywords;

CREATE TRIGGER update_keywords_updated_at
    BEFORE UPDATE ON keywords
    FOR EACH ROW
    EXECUTE PROCEDURE update_keywords_updated_at_column();

-- RLS Policies
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

-- Add helpful comments
COMMENT ON TABLE keywords IS 'Stores expanded keywords from DataForSEO with profitability metrics';
COMMENT ON COLUMN keywords.profitability_score IS 'Calculated as (log(volume) × cpc) / difficulty';
COMMENT ON COLUMN keywords.seed_keyword IS 'Original seed keyword that generated this expanded keyword';
COMMENT ON COLUMN keywords.keyword IS 'The expanded keyword from DataForSEO';
