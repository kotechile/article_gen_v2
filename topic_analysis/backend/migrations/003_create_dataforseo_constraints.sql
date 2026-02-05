-- Migration: Create DataForSEO constraints
-- Description: Creates additional constraints and triggers for data integrity
-- Created: 2024-01-15
-- Author: DataForSEO Integration

-- Add foreign key constraints (if api_keys table exists)
-- Note: These will only be created if the referenced tables exist
DO $$
BEGIN
    -- Check if api_keys table exists
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'api_keys') THEN
        -- Add foreign key constraint for API logs (if api_keys table exists)
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.table_constraints 
            WHERE constraint_name = 'fk_dataforseo_api_logs_api_key'
        ) THEN
            ALTER TABLE dataforseo_api_logs 
            ADD COLUMN IF NOT EXISTS api_key_id UUID,
            ADD CONSTRAINT fk_dataforseo_api_logs_api_key 
            FOREIGN KEY (api_key_id) REFERENCES api_keys(id) ON DELETE SET NULL;
        END IF;
    END IF;
END $$;

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

-- Create function to validate JSONB timeline data
CREATE OR REPLACE FUNCTION validate_timeline_data(data JSONB)
RETURNS BOOLEAN AS $$
BEGIN
    -- Check if it's an array
    IF jsonb_typeof(data) != 'array' THEN
        RETURN FALSE;
    END IF;
    
    -- Check each element has required fields
    FOR i IN 0..jsonb_array_length(data) - 1 LOOP
        DECLARE
            item JSONB := data->i;
        BEGIN
            IF NOT (item ? 'date' AND item ? 'value') THEN
                RETURN FALSE;
            END IF;
            
            -- Check value is numeric
            IF jsonb_typeof(item->'value') != 'number' THEN
                RETURN FALSE;
            END IF;
        END;
    END LOOP;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Create function to validate JSONB related queries
CREATE OR REPLACE FUNCTION validate_related_queries(data JSONB)
RETURNS BOOLEAN AS $$
BEGIN
    -- Check if it's an array
    IF jsonb_typeof(data) != 'array' THEN
        RETURN FALSE;
    END IF;
    
    -- Check each element is a string
    FOR i IN 0..jsonb_array_length(data) - 1 LOOP
        IF jsonb_typeof(data->i) != 'string' THEN
            RETURN FALSE;
        END IF;
    END LOOP;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Create function to validate JSONB search volume trend
CREATE OR REPLACE FUNCTION validate_search_volume_trend(data JSONB)
RETURNS BOOLEAN AS $$
BEGIN
    -- Check if it's an array
    IF jsonb_typeof(data) != 'array' THEN
        RETURN FALSE;
    END IF;
    
    -- Check each element has required fields
    FOR i IN 0..jsonb_array_length(data) - 1 LOOP
        DECLARE
            item JSONB := data->i;
        BEGIN
            IF NOT (item ? 'month' AND item ? 'volume') THEN
                RETURN FALSE;
            END IF;
            
            -- Check volume is numeric
            IF jsonb_typeof(item->'volume') != 'number' THEN
                RETURN FALSE;
            END IF;
        END;
    END LOOP;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Add JSONB validation constraints
ALTER TABLE trend_analysis_data 
ADD CONSTRAINT trend_analysis_data_timeline_data_valid 
CHECK (validate_timeline_data(timeline_data));

ALTER TABLE trend_analysis_data 
ADD CONSTRAINT trend_analysis_data_related_queries_valid 
CHECK (validate_related_queries(related_queries));

ALTER TABLE keyword_research_data 
ADD CONSTRAINT keyword_research_data_related_keywords_valid 
CHECK (validate_related_queries(related_keywords));

ALTER TABLE keyword_research_data 
ADD CONSTRAINT keyword_research_data_search_volume_trend_valid 
CHECK (validate_search_volume_trend(search_volume_trend));

ALTER TABLE subtopic_suggestions 
ADD CONSTRAINT subtopic_suggestions_related_queries_valid 
CHECK (validate_related_queries(related_queries));

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

-- Create function to clean up old data
CREATE OR REPLACE FUNCTION cleanup_old_dataforseo_data()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
BEGIN
    -- Clean up old API logs (older than 30 days)
    DELETE FROM dataforseo_api_logs 
    WHERE created_at < NOW() - INTERVAL '30 days';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Clean up old trend data (older than 90 days)
    DELETE FROM trend_analysis_data 
    WHERE updated_at < NOW() - INTERVAL '90 days';
    
    GET DIAGNOSTICS deleted_count = deleted_count + ROW_COUNT;
    
    -- Clean up old keyword data (older than 30 days)
    DELETE FROM keyword_research_data 
    WHERE updated_at < NOW() - INTERVAL '30 days';
    
    GET DIAGNOSTICS deleted_count = deleted_count + ROW_COUNT;
    
    -- Clean up old suggestions (older than 7 days)
    DELETE FROM subtopic_suggestions 
    WHERE updated_at < NOW() - INTERVAL '7 days';
    
    GET DIAGNOSTICS deleted_count = deleted_count + ROW_COUNT;
    
    RETURN deleted_count;
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

-- Add comments for functions and views
COMMENT ON FUNCTION update_updated_at_column() IS 'Trigger function to update updated_at timestamp';
COMMENT ON FUNCTION validate_timeline_data(JSONB) IS 'Validates timeline data JSONB structure';
COMMENT ON FUNCTION validate_related_queries(JSONB) IS 'Validates related queries JSONB structure';
COMMENT ON FUNCTION validate_search_volume_trend(JSONB) IS 'Validates search volume trend JSONB structure';
COMMENT ON FUNCTION calculate_priority_score(INTEGER, INTEGER, DECIMAL, DECIMAL, DECIMAL, DECIMAL, DECIMAL) IS 'Calculates keyword priority score based on multiple factors';
COMMENT ON FUNCTION cleanup_old_dataforseo_data() IS 'Cleans up old data to maintain database performance';
COMMENT ON FUNCTION get_dataforseo_stats() IS 'Returns comprehensive statistics for DataForSEO data';

COMMENT ON VIEW trending_subtopics IS 'View of trending subtopics with growth rates';
COMMENT ON VIEW high_value_keywords IS 'View of high-value keywords with priority scores';
COMMENT ON VIEW api_performance_metrics IS 'View of API performance metrics by endpoint and hour';
