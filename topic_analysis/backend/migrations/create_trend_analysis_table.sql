-- Create Trend Analysis table
-- This migration creates the trend_analyses table for storing trend analysis results

-- Create trend_analyses table
CREATE TABLE IF NOT EXISTS trend_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workflow_session_id UUID NOT NULL REFERENCES workflow_sessions(id) ON DELETE CASCADE,
    topic_decomposition_id UUID REFERENCES topic_decompositions(id) ON DELETE SET NULL,
    
    -- Analysis metadata
    analysis_name VARCHAR(255) NOT NULL,
    description VARCHAR(1000),
    keywords JSONB NOT NULL DEFAULT '[]',
    timeframe VARCHAR(50) NOT NULL DEFAULT '12m' CHECK (timeframe IN ('1h', '4h', '1d', '7d', '30d', '90d', '12m', '5y', 'all')),
    geo VARCHAR(10) NOT NULL DEFAULT 'US',
    category INTEGER,
    
    -- Analysis results
    trend_data JSONB NOT NULL DEFAULT '{}',
    analysis_results JSONB NOT NULL DEFAULT '{}',
    insights JSONB NOT NULL DEFAULT '{}',
    
    -- Source and status
    source VARCHAR(50) NOT NULL DEFAULT 'google_trends' CHECK (source IN ('google_trends', 'csv_upload', 'semrush', 'ahrefs', 'ubersuggest', 'fallback')),
    status VARCHAR(50) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'failed')),
    error_message VARCHAR(1000),
    
    -- Processing metadata
    processing_time_ms INTEGER,
    api_calls_made INTEGER NOT NULL DEFAULT 0,
    cache_hit BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_trend_analyses_user_id ON trend_analyses(user_id);
CREATE INDEX IF NOT EXISTS idx_trend_analyses_workflow_session_id ON trend_analyses(workflow_session_id);
CREATE INDEX IF NOT EXISTS idx_trend_analyses_topic_decomposition_id ON trend_analyses(topic_decomposition_id);
CREATE INDEX IF NOT EXISTS idx_trend_analyses_status ON trend_analyses(status);
CREATE INDEX IF NOT EXISTS idx_trend_analyses_source ON trend_analyses(source);
CREATE INDEX IF NOT EXISTS idx_trend_analyses_created_at ON trend_analyses(created_at);
CREATE INDEX IF NOT EXISTS idx_trend_analyses_user_status ON trend_analyses(user_id, status);
CREATE INDEX IF NOT EXISTS idx_trend_analyses_workflow_status ON trend_analyses(workflow_session_id, status);

-- Create GIN index for JSONB columns for better query performance
CREATE INDEX IF NOT EXISTS idx_trend_analyses_keywords_gin ON trend_analyses USING GIN (keywords);
CREATE INDEX IF NOT EXISTS idx_trend_analyses_trend_data_gin ON trend_analyses USING GIN (trend_data);
CREATE INDEX IF NOT EXISTS idx_trend_analyses_analysis_results_gin ON trend_analyses USING GIN (analysis_results);
CREATE INDEX IF NOT EXISTS idx_trend_analyses_insights_gin ON trend_analyses USING GIN (insights);

-- Enable Row Level Security
ALTER TABLE trend_analyses ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
CREATE POLICY "Users can view their own trend analyses" ON trend_analyses
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own trend analyses" ON trend_analyses
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own trend analyses" ON trend_analyses
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own trend analyses" ON trend_analyses
    FOR DELETE USING (auth.uid() = user_id);

-- Add comments
COMMENT ON TABLE trend_analyses IS 'Stores trend analysis results and data for users';
COMMENT ON COLUMN trend_analyses.analysis_name IS 'Name of the trend analysis';
COMMENT ON COLUMN trend_analyses.description IS 'Description of the analysis';
COMMENT ON COLUMN trend_analyses.keywords IS 'Array of keywords analyzed (JSONB)';
COMMENT ON COLUMN trend_analyses.timeframe IS 'Time period for analysis (1h, 4h, 1d, 7d, 30d, 90d, 12m, 5y, all)';
COMMENT ON COLUMN trend_analyses.geo IS 'Geographic location for analysis';
COMMENT ON COLUMN trend_analyses.category IS 'Google Trends category ID';
COMMENT ON COLUMN trend_analyses.trend_data IS 'Raw trend data from APIs (JSONB)';
COMMENT ON COLUMN trend_analyses.analysis_results IS 'Processed analysis results (JSONB)';
COMMENT ON COLUMN trend_analyses.insights IS 'Generated insights and recommendations (JSONB)';
COMMENT ON COLUMN trend_analyses.source IS 'Source of trend data (google_trends, csv_upload, semrush, ahrefs, ubersuggest, fallback)';
COMMENT ON COLUMN trend_analyses.status IS 'Analysis status (pending, in_progress, completed, failed)';
COMMENT ON COLUMN trend_analyses.error_message IS 'Error message if analysis failed';
COMMENT ON COLUMN trend_analyses.processing_time_ms IS 'Processing time in milliseconds';
COMMENT ON COLUMN trend_analyses.api_calls_made IS 'Number of API calls made during analysis';
COMMENT ON COLUMN trend_analyses.cache_hit IS 'Whether result was retrieved from cache';
