-- Migration: Create time-based partial indexes
-- Description: Creates time-based partial indexes using immutable functions
-- Created: 2024-01-15
-- Author: DataForSEO Integration
-- Note: This migration must be run after the main indexes are created

-- Create immutable function for time-based comparisons
CREATE OR REPLACE FUNCTION get_recent_timestamp(days_ago INTEGER)
RETURNS TIMESTAMP WITH TIME ZONE
LANGUAGE SQL
IMMUTABLE
AS $$
    SELECT (CURRENT_DATE - INTERVAL '1 day' * days_ago)::TIMESTAMP WITH TIME ZONE;
$$;

-- Create immutable function for cleanup timestamp
CREATE OR REPLACE FUNCTION get_cleanup_timestamp()
RETURNS TIMESTAMP WITH TIME ZONE
LANGUAGE SQL
IMMUTABLE
AS $$
    SELECT (CURRENT_DATE - INTERVAL '30 days')::TIMESTAMP WITH TIME ZONE;
$$;

-- Time-based partial indexes using immutable functions
CREATE INDEX IF NOT EXISTS idx_trend_analysis_data_recent ON trend_analysis_data(subtopic, location, time_range) 
WHERE updated_at > get_recent_timestamp(7);

CREATE INDEX IF NOT EXISTS idx_dataforseo_api_logs_cleanup ON dataforseo_api_logs(created_at) 
WHERE created_at < get_cleanup_timestamp();

-- Additional time-based indexes for common queries
CREATE INDEX IF NOT EXISTS idx_trend_analysis_data_last_24h ON trend_analysis_data(subtopic, location, time_range, updated_at) 
WHERE updated_at > get_recent_timestamp(1);

CREATE INDEX IF NOT EXISTS idx_trend_analysis_data_last_week ON trend_analysis_data(subtopic, location, time_range, updated_at) 
WHERE updated_at > get_recent_timestamp(7);

CREATE INDEX IF NOT EXISTS idx_keyword_research_data_recent ON keyword_research_data(keyword, search_volume, keyword_difficulty, updated_at) 
WHERE updated_at > get_recent_timestamp(7);

CREATE INDEX IF NOT EXISTS idx_subtopic_suggestions_recent ON subtopic_suggestions(topic, trending_status, growth_potential, updated_at) 
WHERE updated_at > get_recent_timestamp(7);

-- Performance indexes for specific time ranges
CREATE INDEX IF NOT EXISTS idx_trend_analysis_data_today ON trend_analysis_data(subtopic, location, time_range, average_interest) 
WHERE updated_at > get_recent_timestamp(1);

CREATE INDEX IF NOT EXISTS idx_keyword_research_data_today ON keyword_research_data(keyword, search_volume, cpc, trend_percentage) 
WHERE updated_at > get_recent_timestamp(1);

-- Indexes for data freshness monitoring
CREATE INDEX IF NOT EXISTS idx_trend_analysis_data_stale ON trend_analysis_data(subtopic, location, time_range, updated_at) 
WHERE updated_at < get_recent_timestamp(30);

CREATE INDEX IF NOT EXISTS idx_keyword_research_data_stale ON keyword_research_data(keyword, search_volume, keyword_difficulty, updated_at) 
WHERE updated_at < get_recent_timestamp(7);

-- Comments for documentation
COMMENT ON FUNCTION get_recent_timestamp(INTEGER) IS 'Returns timestamp for N days ago (immutable function for index predicates)';
COMMENT ON FUNCTION get_cleanup_timestamp() IS 'Returns timestamp for 30 days ago (immutable function for cleanup operations)';

COMMENT ON INDEX idx_trend_analysis_data_recent IS 'Index for recent trend analysis data (last 7 days)';
COMMENT ON INDEX idx_dataforseo_api_logs_cleanup IS 'Index for old API logs that can be cleaned up (older than 30 days)';
COMMENT ON INDEX idx_trend_analysis_data_last_24h IS 'Index for trend data updated in the last 24 hours';
COMMENT ON INDEX idx_trend_analysis_data_last_week IS 'Index for trend data updated in the last week';
COMMENT ON INDEX idx_keyword_research_data_recent IS 'Index for recent keyword research data (last 7 days)';
COMMENT ON INDEX idx_subtopic_suggestions_recent IS 'Index for recent subtopic suggestions (last 7 days)';
COMMENT ON INDEX idx_trend_analysis_data_today IS 'Index for trend data updated today with key metrics';
COMMENT ON INDEX idx_keyword_research_data_today IS 'Index for keyword data updated today with key metrics';
COMMENT ON INDEX idx_trend_analysis_data_stale IS 'Index for stale trend data (older than 30 days)';
COMMENT ON INDEX idx_keyword_research_data_stale IS 'Index for stale keyword data (older than 7 days)';
