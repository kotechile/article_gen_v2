-- Migration: Create RLS policies for DataForSEO tables
-- Description: Enables Row Level Security and creates policies for authenticated users
-- Created: 2024-01-15
-- Author: DataForSEO Integration
-- Target: Remote Supabase database (dgcsqiaciyqvprtpopxg)

-- Enable RLS on all DataForSEO tables
ALTER TABLE trend_analysis_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE keyword_research_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE subtopic_suggestions ENABLE ROW LEVEL SECURITY;
ALTER TABLE dataforseo_api_logs ENABLE ROW LEVEL SECURITY;

-- RLS Policies for trend_analysis_data
-- Allow authenticated users to read all trend data
CREATE POLICY "Allow authenticated users to read trend analysis data" ON trend_analysis_data
    FOR SELECT
    TO authenticated
    USING (true);

-- Allow authenticated users to insert trend data
CREATE POLICY "Allow authenticated users to insert trend analysis data" ON trend_analysis_data
    FOR INSERT
    TO authenticated
    WITH CHECK (true);

-- Allow authenticated users to update trend data
CREATE POLICY "Allow authenticated users to update trend analysis data" ON trend_analysis_data
    FOR UPDATE
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Allow authenticated users to delete trend data
CREATE POLICY "Allow authenticated users to delete trend analysis data" ON trend_analysis_data
    FOR DELETE
    TO authenticated
    USING (true);

-- RLS Policies for keyword_research_data
-- Allow authenticated users to read all keyword data
CREATE POLICY "Allow authenticated users to read keyword research data" ON keyword_research_data
    FOR SELECT
    TO authenticated
    USING (true);

-- Allow authenticated users to insert keyword data
CREATE POLICY "Allow authenticated users to insert keyword research data" ON keyword_research_data
    FOR INSERT
    TO authenticated
    WITH CHECK (true);

-- Allow authenticated users to update keyword data
CREATE POLICY "Allow authenticated users to update keyword research data" ON keyword_research_data
    FOR UPDATE
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Allow authenticated users to delete keyword data
CREATE POLICY "Allow authenticated users to delete keyword research data" ON keyword_research_data
    FOR DELETE
    TO authenticated
    USING (true);

-- RLS Policies for subtopic_suggestions
-- Allow authenticated users to read all subtopic suggestions
CREATE POLICY "Allow authenticated users to read subtopic suggestions" ON subtopic_suggestions
    FOR SELECT
    TO authenticated
    USING (true);

-- Allow authenticated users to insert subtopic suggestions
CREATE POLICY "Allow authenticated users to insert subtopic suggestions" ON subtopic_suggestions
    FOR INSERT
    TO authenticated
    WITH CHECK (true);

-- Allow authenticated users to update subtopic suggestions
CREATE POLICY "Allow authenticated users to update subtopic suggestions" ON subtopic_suggestions
    FOR UPDATE
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Allow authenticated users to delete subtopic suggestions
CREATE POLICY "Allow authenticated users to delete subtopic suggestions" ON subtopic_suggestions
    FOR DELETE
    TO authenticated
    USING (true);

-- RLS Policies for dataforseo_api_logs
-- Allow authenticated users to read all API logs
CREATE POLICY "Allow authenticated users to read API logs" ON dataforseo_api_logs
    FOR SELECT
    TO authenticated
    USING (true);

-- Allow authenticated users to insert API logs
CREATE POLICY "Allow authenticated users to insert API logs" ON dataforseo_api_logs
    FOR INSERT
    TO authenticated
    WITH CHECK (true);

-- Allow authenticated users to update API logs
CREATE POLICY "Allow authenticated users to update API logs" ON dataforseo_api_logs
    FOR UPDATE
    TO authenticated
    USING (true)
    WITH CHECK (true);

-- Allow authenticated users to delete API logs
CREATE POLICY "Allow authenticated users to delete API logs" ON dataforseo_api_logs
    FOR DELETE
    TO authenticated
    USING (true);

-- RLS Policies for api_keys table (if it doesn't have RLS already)
-- Only allow service role to access API keys for security
CREATE POLICY "Allow service role to read API keys" ON api_keys
    FOR SELECT
    TO service_role
    USING (true);

CREATE POLICY "Allow service role to insert API keys" ON api_keys
    FOR INSERT
    TO service_role
    WITH CHECK (true);

CREATE POLICY "Allow service role to update API keys" ON api_keys
    FOR UPDATE
    TO service_role
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow service role to delete API keys" ON api_keys
    FOR DELETE
    TO service_role
    USING (true);

-- Add comments for documentation
COMMENT ON POLICY "Allow authenticated users to read trend analysis data" ON trend_analysis_data IS 'Allows authenticated users to read all trend analysis data';
COMMENT ON POLICY "Allow authenticated users to insert trend analysis data" ON trend_analysis_data IS 'Allows authenticated users to insert new trend analysis data';
COMMENT ON POLICY "Allow authenticated users to update trend analysis data" ON trend_analysis_data IS 'Allows authenticated users to update existing trend analysis data';
COMMENT ON POLICY "Allow authenticated users to delete trend analysis data" ON trend_analysis_data IS 'Allows authenticated users to delete trend analysis data';

COMMENT ON POLICY "Allow authenticated users to read keyword research data" ON keyword_research_data IS 'Allows authenticated users to read all keyword research data';
COMMENT ON POLICY "Allow authenticated users to insert keyword research data" ON keyword_research_data IS 'Allows authenticated users to insert new keyword research data';
COMMENT ON POLICY "Allow authenticated users to update keyword research data" ON keyword_research_data IS 'Allows authenticated users to update existing keyword research data';
COMMENT ON POLICY "Allow authenticated users to delete keyword research data" ON keyword_research_data IS 'Allows authenticated users to delete keyword research data';

COMMENT ON POLICY "Allow authenticated users to read subtopic suggestions" ON subtopic_suggestions IS 'Allows authenticated users to read all subtopic suggestions';
COMMENT ON POLICY "Allow authenticated users to insert subtopic suggestions" ON subtopic_suggestions IS 'Allows authenticated users to insert new subtopic suggestions';
COMMENT ON POLICY "Allow authenticated users to update subtopic suggestions" ON subtopic_suggestions IS 'Allows authenticated users to update existing subtopic suggestions';
COMMENT ON POLICY "Allow authenticated users to delete subtopic suggestions" ON subtopic_suggestions IS 'Allows authenticated users to delete subtopic suggestions';

COMMENT ON POLICY "Allow authenticated users to read API logs" ON dataforseo_api_logs IS 'Allows authenticated users to read all API logs';
COMMENT ON POLICY "Allow authenticated users to insert API logs" ON dataforseo_api_logs IS 'Allows authenticated users to insert new API logs';
COMMENT ON POLICY "Allow authenticated users to update API logs" ON dataforseo_api_logs IS 'Allows authenticated users to update existing API logs';
COMMENT ON POLICY "Allow authenticated users to delete API logs" ON dataforseo_api_logs IS 'Allows authenticated users to delete API logs';
