-- Migration: Create DataForSEO tables
-- Description: Creates tables for trend analysis and keyword research data
-- Created: 2024-01-15
-- Author: DataForSEO Integration

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

-- Add comments for documentation
COMMENT ON TABLE trend_analysis_data IS 'Stores trend analysis data from DataForSEO Trends API';
COMMENT ON TABLE keyword_research_data IS 'Stores keyword research data from DataForSEO Labs API';
COMMENT ON TABLE subtopic_suggestions IS 'Stores trending subtopic suggestions and recommendations';
COMMENT ON TABLE dataforseo_api_logs IS 'Logs API requests and responses for monitoring and debugging';
