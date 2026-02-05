-- Create subtopics table with viability scoring
-- This table normalizes subtopic storage and adds fields for trend, SEO, and affiliate data

CREATE TABLE IF NOT EXISTS subtopics (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  research_topic_id UUID REFERENCES research_topics(id) ON DELETE CASCADE,
  user_id UUID NOT NULL, -- references auth.users(id), but we might not have foreign key constraint depending on setup
  name TEXT NOT NULL,
  
  -- Trend Data
  trend_direction TEXT CHECK (trend_direction IN ('up', 'down', 'stable')),
  trend_score NUMERIC(5,2), -- 0-100
  interest_over_time JSONB DEFAULT '[]',
  
  -- SEO Data (from DataForSEO)
  seo_difficulty NUMERIC(5,2), -- 0-100
  search_volume INTEGER,
  cpc NUMERIC(10,2),
  keywords JSONB DEFAULT '[]',
  
  -- Affiliate Data
  affiliate_offer_count INTEGER DEFAULT 0,
  
  -- Calculated Viability
  viability_score NUMERIC(5,2), -- 0-100, calculated field
  
  -- Metadata
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  
  CONSTRAINT subtopics_name_check CHECK (LENGTH(name) > 0),
  CONSTRAINT subtopics_viability_check CHECK (viability_score IS NULL OR (viability_score >= 0 AND viability_score <= 100))
);

-- Indexes for performance and sorting
CREATE INDEX IF NOT EXISTS idx_subtopics_research_topic ON subtopics(research_topic_id);
CREATE INDEX IF NOT EXISTS idx_subtopics_user ON subtopics(user_id);
CREATE INDEX IF NOT EXISTS idx_subtopics_viability ON subtopics(viability_score DESC);
CREATE INDEX IF NOT EXISTS idx_subtopics_trend ON subtopics(trend_direction, trend_score DESC);

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_subtopics_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_subtopics_updated_at ON subtopics;

CREATE TRIGGER update_subtopics_updated_at
    BEFORE UPDATE ON subtopics
    FOR EACH ROW
    EXECUTE PROCEDURE update_subtopics_updated_at_column();

-- RLS Policies
ALTER TABLE subtopics ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view their own subtopics"
    ON subtopics FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own subtopics"
    ON subtopics FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own subtopics"
    ON subtopics FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own subtopics"
    ON subtopics FOR DELETE
    USING (auth.uid() = user_id);
