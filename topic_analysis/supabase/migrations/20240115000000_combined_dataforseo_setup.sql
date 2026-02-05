-- Combined DataForSEO Migration
-- Description: Complete DataForSEO setup with tables, indexes, and constraints
-- Created: 2024-01-15
-- Author: DataForSEO Integration
-- 
-- This migration combines all DataForSEO setup in one file for easy execution
-- Run this in your Supabase SQL Editor

-- ============================================================================
-- PART 1: CREATE TABLES
-- ============================================================================

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create trend_analysis_data table
CREATE TABLE IF NOT EXISTS trend_analysis_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    subtopic VARCHAR(255) NOT NULL,
    location VARCHAR(100) NOT NULL,
    time_range VARCHAR(20) NOT NULL,
    average_interest DECIMAL(10,2) NOT NULL DEFAULT 0,
    peak_interest DECIMAL(10,2) NOT NULL DEFAULT 0,
    timeline_data JSONB NOT NULL DEFAULT '[]',
    related_queries JSONB NOT NULL DEFAULT '[]',
    demographic_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT trend_analysis_data_subtopic_check CHECK (LENGTH(subtopic) > 0),
    CONSTRAINT trend_analysis_data_location_check CHECK (LENGTH(location) > 0),
    CONSTRAINT trend_analysis_data_time_range_check CHECK (time_range IN ('1m', '3m', '6m', '12m', '24m')),
    CONSTRAINT trend_analysis_data_average_interest_check CHECK (average_interest >= 0),
    CONSTRAINT trend_analysis_data_peak_interest_check CHECK (peak_interest >= 0),
    CONSTRAINT trend_analysis_data_timeline_data_check CHECK (jsonb_typeof(timeline_data) = 'array'),
    CONSTRAINT trend_analysis_data_related_queries_check CHECK (jsonb_typeof(related_queries) = 'array')
);

-- Create keyword_research_data table
CREATE TABLE IF NOT EXISTS keyword_research_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    keyword VARCHAR(500) NOT NULL UNIQUE,
    search_volume INTEGER NOT NULL DEFAULT 0,
    keyword_difficulty INTEGER NOT NULL DEFAULT 0,
    cpc DECIMAL(10,2) NOT NULL DEFAULT 0,
    competition_value INTEGER NOT NULL DEFAULT 0,
    trend_percentage DECIMAL(5,2) NOT NULL DEFAULT 0,
    intent_type VARCHAR(20) NOT NULL DEFAULT 'INFORMATIONAL',
    priority_score DECIMAL(5,2),
    related_keywords JSONB NOT NULL DEFAULT '[]',
    search_volume_trend JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT keyword_research_data_keyword_check CHECK (LENGTH(keyword) > 0),
    CONSTRAINT keyword_research_data_search_volume_check CHECK (search_volume >= 0),
    CONSTRAINT keyword_research_data_difficulty_check CHECK (keyword_difficulty >= 0 AND keyword_difficulty <= 100),
    CONSTRAINT keyword_research_data_cpc_check CHECK (cpc >= 0),
    CONSTRAINT keyword_research_data_competition_check CHECK (competition_value >= 0 AND competition_value <= 100),
    CONSTRAINT keyword_research_data_intent_type_check CHECK (intent_type IN ('INFORMATIONAL', 'COMMERCIAL', 'TRANSACTIONAL')),
    CONSTRAINT keyword_research_data_priority_score_check CHECK (priority_score IS NULL OR (priority_score >= 0 AND priority_score <= 100)),
    CONSTRAINT keyword_research_data_related_keywords_check CHECK (jsonb_typeof(related_keywords) = 'array'),
    CONSTRAINT keyword_research_data_search_volume_trend_check CHECK (jsonb_typeof(search_volume_trend) = 'array')
);

-- Create subtopic_suggestions table
CREATE TABLE IF NOT EXISTS subtopic_suggestions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    topic VARCHAR(255) NOT NULL UNIQUE,
    trending_status VARCHAR(20) NOT NULL DEFAULT 'STABLE',
    growth_potential DECIMAL(5,2) NOT NULL DEFAULT 0,
    search_volume INTEGER NOT NULL DEFAULT 0,
    related_queries JSONB NOT NULL DEFAULT '[]',
    competition_level VARCHAR(10) NOT NULL DEFAULT 'MEDIUM',
    commercial_intent DECIMAL(5,2) NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT subtopic_suggestions_topic_check CHECK (LENGTH(topic) > 0),
    CONSTRAINT subtopic_suggestions_trending_status_check CHECK (trending_status IN ('TRENDING', 'STABLE', 'DECLINING')),
    CONSTRAINT subtopic_suggestions_growth_potential_check CHECK (growth_potential >= 0),
    CONSTRAINT subtopic_suggestions_search_volume_check CHECK (search_volume >= 0),
    CONSTRAINT subtopic_suggestions_competition_level_check CHECK (competition_level IN ('LOW', 'MEDIUM', 'HIGH')),
    CONSTRAINT subtopic_suggestions_commercial_intent_check CHECK (commercial_intent >= 0),
    CONSTRAINT subtopic_suggestions_related_queries_check CHECK (jsonb_typeof(related_queries) = 'array')
);

-- Create dataforseo_api_logs table for monitoring
CREATE TABLE IF NOT EXISTS dataforseo_api_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint VARCHAR(255) NOT NULL,
    request_data JSONB NOT NULL DEFAULT '{}',
    response_data JSONB,
    status_code INTEGER,
    response_time_ms INTEGER,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT dataforseo_api_logs_endpoint_check CHECK (LENGTH(endpoint) > 0),
    CONSTRAINT dataforseo_api_logs_request_data_check CHECK (jsonb_typeof(request_data) = 'object'),
    CONSTRAINT dataforseo_api_logs_response_data_check CHECK (response_data IS NULL OR jsonb_typeof(response_data) = 'object'),
    CONSTRAINT dataforseo_api_logs_status_code_check CHECK (status_code IS NULL OR (status_code >= 100 AND status_code < 600)),
    CONSTRAINT dataforseo_api_logs_response_time_check CHECK (response_time_ms IS NULL OR response_time_ms >= 0)
);

-- ============================================================================
-- PART 2: CREATE INDEXES
-- ============================================================================

-- Performance indexes for trend_analysis_data
CREATE INDEX IF NOT EXISTS idx_trend_analysis_data_average_interest ON trend_analysis_data(average_interest);
CREATE INDEX IF NOT EXISTS idx_trend_analysis_data_peak_interest ON trend_analysis_data(peak_interest);
CREATE INDEX IF NOT EXISTS idx_trend_analysis_data_location_time ON trend_analysis_data(location, time_range);

-- Performance indexes for keyword_research_data
CREATE INDEX IF NOT EXISTS idx_keyword_research_data_cpc ON keyword_research_data(cpc);
CREATE INDEX IF NOT EXISTS idx_keyword_research_data_competition_value ON keyword_research_data(competition_value);
CREATE INDEX IF NOT EXISTS idx_keyword_research_data_trend_percentage ON keyword_research_data(trend_percentage);

-- Performance indexes for subtopic_suggestions
CREATE INDEX IF NOT EXISTS idx_subtopic_suggestions_search_volume ON subtopic_suggestions(search_volume);
CREATE INDEX IF NOT EXISTS idx_subtopic_suggestions_commercial_intent ON subtopic_suggestions(commercial_intent);
CREATE INDEX IF NOT EXISTS idx_subtopic_suggestions_competition_level ON subtopic_suggestions(competition_level);

-- Performance indexes for dataforseo_api_logs
CREATE INDEX IF NOT EXISTS idx_dataforseo_api_logs_response_time ON dataforseo_api_logs(response_time_ms);
CREATE INDEX IF NOT EXISTS idx_dataforseo_api_logs_error_message ON dataforseo_api_logs(error_message) WHERE error_message IS NOT NULL;

-- Partial indexes for common filter conditions
CREATE INDEX IF NOT EXISTS idx_keyword_research_data_high_volume ON keyword_research_data(keyword, search_volume, keyword_difficulty) 
WHERE search_volume >= 1000;

CREATE INDEX IF NOT EXISTS idx_keyword_research_data_low_difficulty ON keyword_research_data(keyword, search_volume, keyword_difficulty) 
WHERE keyword_difficulty <= 50;

CREATE INDEX IF NOT EXISTS idx_keyword_research_data_high_cpc ON keyword_research_data(keyword, cpc, intent_type) 
WHERE cpc >= 1.0;

CREATE INDEX IF NOT EXISTS idx_subtopic_suggestions_trending ON subtopic_suggestions(topic, growth_potential, search_volume) 
WHERE trending_status = 'TRENDING';

CREATE INDEX IF NOT EXISTS idx_subtopic_suggestions_high_intent ON subtopic_suggestions(topic, commercial_intent, competition_level) 
WHERE commercial_intent >= 50;

-- GIN indexes for JSONB columns
CREATE INDEX IF NOT EXISTS idx_trend_analysis_data_timeline_data_gin ON trend_analysis_data USING GIN(timeline_data);
CREATE INDEX IF NOT EXISTS idx_trend_analysis_data_related_queries_gin ON trend_analysis_data USING GIN(related_queries);
CREATE INDEX IF NOT EXISTS idx_trend_analysis_data_demographic_data_gin ON trend_analysis_data USING GIN(demographic_data);

CREATE INDEX IF NOT EXISTS idx_keyword_research_data_related_keywords_gin ON keyword_research_data USING GIN(related_keywords);
CREATE INDEX IF NOT EXISTS idx_keyword_research_data_search_volume_trend_gin ON keyword_research_data USING GIN(search_volume_trend);

CREATE INDEX IF NOT EXISTS idx_subtopic_suggestions_related_queries_gin ON subtopic_suggestions USING GIN(related_queries);

CREATE INDEX IF NOT EXISTS idx_dataforseo_api_logs_request_data_gin ON dataforseo_api_logs USING GIN(request_data);
CREATE INDEX IF NOT EXISTS idx_dataforseo_api_logs_response_data_gin ON dataforseo_api_logs USING GIN(response_data);

-- Text search indexes
CREATE INDEX IF NOT EXISTS idx_trend_analysis_data_subtopic_text ON trend_analysis_data USING GIN(to_tsvector('english', subtopic));
CREATE INDEX IF NOT EXISTS idx_keyword_research_data_keyword_text ON keyword_research_data USING GIN(to_tsvector('english', keyword));
CREATE INDEX IF NOT EXISTS idx_subtopic_suggestions_topic_text ON subtopic_suggestions USING GIN(to_tsvector('english', topic));

-- Composite indexes for complex queries
CREATE INDEX IF NOT EXISTS idx_trend_analysis_data_analysis ON trend_analysis_data(subtopic, location, time_range, average_interest, peak_interest);
CREATE INDEX IF NOT EXISTS idx_keyword_research_data_analysis ON keyword_research_data(intent_type, keyword_difficulty, search_volume, cpc, trend_percentage);
CREATE INDEX IF NOT EXISTS idx_subtopic_suggestions_analysis ON subtopic_suggestions(trending_status, growth_potential, search_volume, commercial_intent);

-- Indexes for time-based queries
CREATE INDEX IF NOT EXISTS idx_trend_analysis_data_created_at ON trend_analysis_data(created_at);
CREATE INDEX IF NOT EXISTS idx_keyword_research_data_created_at ON keyword_research_data(created_at);
CREATE INDEX IF NOT EXISTS idx_subtopic_suggestions_created_at ON subtopic_suggestions(created_at);
CREATE INDEX IF NOT EXISTS idx_trend_analysis_data_updated_at ON trend_analysis_data(updated_at);
CREATE INDEX IF NOT EXISTS idx_keyword_research_data_updated_at ON keyword_research_data(updated_at);
CREATE INDEX IF NOT EXISTS idx_subtopic_suggestions_updated_at ON subtopic_suggestions(updated_at);

-- ============================================================================
-- PART 3: CREATE FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at triggers
DROP TRIGGER IF EXISTS update_trend_analysis_data_updated_at ON trend_analysis_data;
CREATE TRIGGER update_trend_analysis_data_updated_at
    BEFORE UPDATE ON trend_analysis_data
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_keyword_research_data_updated_at ON keyword_research_data;
CREATE TRIGGER update_keyword_research_data_updated_at
    BEFORE UPDATE ON keyword_research_data
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_subtopic_suggestions_updated_at ON subtopic_suggestions;
CREATE TRIGGER update_subtopic_suggestions_updated_at
    BEFORE UPDATE ON subtopic_suggestions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create function to calculate priority score
CREATE OR REPLACE FUNCTION calculate_priority_score(
    search_volume INTEGER,
    keyword_difficulty INTEGER,
    cpc DECIMAL,
    trend_percentage DECIMAL,
    cpc_weight DECIMAL DEFAULT 0.3,
    volume_weight DECIMAL DEFAULT 0.4,
    trend_weight DECIMAL DEFAULT 0.3
)
RETURNS DECIMAL AS $$
DECLARE
    normalized_volume DECIMAL;
    normalized_difficulty DECIMAL;
    normalized_cpc DECIMAL;
    normalized_trend DECIMAL;
    priority_score DECIMAL;
BEGIN
    -- Normalize values to 0-100 scale
    normalized_volume := LEAST(100, GREATEST(0, (search_volume::DECIMAL / 10000) * 100));
    normalized_difficulty := 100 - keyword_difficulty; -- Invert difficulty (lower is better)
    normalized_cpc := LEAST(100, GREATEST(0, cpc * 10)); -- Scale CPC
    normalized_trend := LEAST(100, GREATEST(0, 50 + trend_percentage)); -- Center trend around 50
    
    -- Calculate weighted priority score
    priority_score := (
        normalized_volume * volume_weight +
        normalized_difficulty * (1 - cpc_weight - volume_weight - trend_weight) +
        normalized_cpc * cpc_weight +
        normalized_trend * trend_weight
    );
    
    RETURN ROUND(priority_score, 2);
END;
$$ LANGUAGE plpgsql;

-- Create function to get dataforseo statistics
CREATE OR REPLACE FUNCTION get_dataforseo_stats()
RETURNS JSONB AS $$
DECLARE
    stats JSONB;
BEGIN
    SELECT jsonb_build_object(
        'trend_analysis_data', (
            SELECT jsonb_build_object(
                'total_records', COUNT(*),
                'unique_subtopics', COUNT(DISTINCT subtopic),
                'unique_locations', COUNT(DISTINCT location),
                'avg_interest', ROUND(AVG(average_interest), 2),
                'max_interest', MAX(peak_interest),
                'last_updated', MAX(updated_at)
            )
            FROM trend_analysis_data
        ),
        'keyword_research_data', (
            SELECT jsonb_build_object(
                'total_records', COUNT(*),
                'unique_keywords', COUNT(DISTINCT keyword),
                'avg_search_volume', ROUND(AVG(search_volume), 0),
                'avg_difficulty', ROUND(AVG(keyword_difficulty), 2),
                'avg_cpc', ROUND(AVG(cpc), 2),
                'last_updated', MAX(updated_at)
            )
            FROM keyword_research_data
        ),
        'subtopic_suggestions', (
            SELECT jsonb_build_object(
                'total_records', COUNT(*),
                'trending_count', COUNT(*) FILTER (WHERE trending_status = 'TRENDING'),
                'stable_count', COUNT(*) FILTER (WHERE trending_status = 'STABLE'),
                'declining_count', COUNT(*) FILTER (WHERE trending_status = 'DECLINING'),
                'avg_growth_potential', ROUND(AVG(growth_potential), 2),
                'last_updated', MAX(updated_at)
            )
            FROM subtopic_suggestions
        ),
        'api_logs', (
            SELECT jsonb_build_object(
                'total_requests', COUNT(*),
                'successful_requests', COUNT(*) FILTER (WHERE status_code BETWEEN 200 AND 299),
                'failed_requests', COUNT(*) FILTER (WHERE status_code >= 400),
                'avg_response_time', ROUND(AVG(response_time_ms), 2),
                'last_request', MAX(created_at)
            )
            FROM dataforseo_api_logs
        )
    ) INTO stats;
    
    RETURN stats;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PART 4: CREATE VIEWS
-- ============================================================================

-- Create view for trending subtopics
CREATE OR REPLACE VIEW trending_subtopics AS
SELECT 
    subtopic,
    location,
    time_range,
    average_interest,
    peak_interest,
    ROUND(((peak_interest - average_interest) / NULLIF(average_interest, 0)) * 100, 2) as growth_rate,
    updated_at
FROM trend_analysis_data
WHERE updated_at > NOW() - INTERVAL '7 days'
ORDER BY growth_rate DESC;

-- Create view for high-value keywords
CREATE OR REPLACE VIEW high_value_keywords AS
SELECT 
    keyword,
    search_volume,
    keyword_difficulty,
    cpc,
    trend_percentage,
    COALESCE(priority_score, calculate_priority_score(search_volume, keyword_difficulty, cpc, trend_percentage)) as calculated_priority_score,
    intent_type,
    updated_at
FROM keyword_research_data
WHERE search_volume >= 1000 
    AND keyword_difficulty <= 70 
    AND cpc >= 0.5
    AND updated_at > NOW() - INTERVAL '7 days'
ORDER BY calculated_priority_score DESC;

-- Create view for API performance metrics
CREATE OR REPLACE VIEW api_performance_metrics AS
SELECT 
    endpoint,
    DATE_TRUNC('hour', created_at) as hour,
    COUNT(*) as request_count,
    COUNT(*) FILTER (WHERE status_code BETWEEN 200 AND 299) as success_count,
    COUNT(*) FILTER (WHERE status_code >= 400) as error_count,
    ROUND(AVG(response_time_ms), 2) as avg_response_time,
    ROUND(MAX(response_time_ms), 2) as max_response_time,
    ROUND(MIN(response_time_ms), 2) as min_response_time
FROM dataforseo_api_logs
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY endpoint, DATE_TRUNC('hour', created_at)
ORDER BY hour DESC, request_count DESC;

-- ============================================================================
-- PART 5: ADD COMMENTS
-- ============================================================================

-- Add comments for documentation
COMMENT ON TABLE trend_analysis_data IS 'Stores trend analysis data from DataForSEO Trends API';
COMMENT ON TABLE keyword_research_data IS 'Stores keyword research data from DataForSEO Labs API';
COMMENT ON TABLE subtopic_suggestions IS 'Stores trending subtopic suggestions and recommendations';
COMMENT ON TABLE dataforseo_api_logs IS 'Logs API requests and responses for monitoring and debugging';

COMMENT ON FUNCTION update_updated_at_column() IS 'Trigger function to update updated_at timestamp';
COMMENT ON FUNCTION calculate_priority_score(INTEGER, INTEGER, DECIMAL, DECIMAL, DECIMAL, DECIMAL, DECIMAL) IS 'Calculates keyword priority score based on multiple factors';
COMMENT ON FUNCTION get_dataforseo_stats() IS 'Returns comprehensive statistics for DataForSEO data';

COMMENT ON VIEW trending_subtopics IS 'View of trending subtopics with growth rates';
COMMENT ON VIEW high_value_keywords IS 'View of high-value keywords with priority scores';
COMMENT ON VIEW api_performance_metrics IS 'View of API performance metrics by endpoint and hour';

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================

-- This migration is now complete!
-- You should see all tables, indexes, functions, triggers, and views created successfully.
-- The DataForSEO integration is ready to use!
