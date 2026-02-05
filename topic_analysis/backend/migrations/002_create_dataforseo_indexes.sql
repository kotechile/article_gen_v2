-- Migration: Create DataForSEO indexes
-- Description: Creates additional indexes for performance optimization
-- Created: 2024-01-15
-- Author: DataForSEO Integration

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
-- Note: Removed time-based partial indexes as NOW() is not immutable
-- These will be created as regular indexes instead

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

-- Indexes for cleanup operations
-- Note: Removed time-based partial index as NOW() is not immutable
-- Cleanup operations will use the created_at index instead
