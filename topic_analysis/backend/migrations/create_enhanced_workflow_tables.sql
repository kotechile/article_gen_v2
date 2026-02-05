-- Create enhanced workflow tables for TrendTap application with RLS
-- This script creates tables for the enhanced research workflow integration

-- 1. Workflow Sessions table
CREATE TABLE IF NOT EXISTS workflow_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_name VARCHAR(255) NOT NULL,
    description TEXT,
    current_step VARCHAR(50) DEFAULT 'upload_csv' CHECK (current_step IN ('upload_csv', 'select_trends', 'generate_keywords', 'export_keywords', 'upload_external', 'analyze_results')),
    progress_percentage INTEGER DEFAULT 0 CHECK (progress_percentage >= 0 AND progress_percentage <= 100),
    workflow_data JSONB DEFAULT '{}',
    completed_steps JSONB DEFAULT '[]',
    error_message TEXT,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'completed', 'failed', 'paused')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Enable RLS on workflow_sessions table
ALTER TABLE workflow_sessions ENABLE ROW LEVEL SECURITY;

-- 2. Trend Selections table
CREATE TABLE IF NOT EXISTS trend_selections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    trend_analysis_id UUID REFERENCES trend_analysis(id) ON DELETE CASCADE,
    trend_name VARCHAR(255) NOT NULL,
    trend_description TEXT,
    trend_category VARCHAR(50) CHECK (trend_category IN ('technology', 'business', 'lifestyle', 'health', 'finance', 'entertainment')),
    search_volume INTEGER DEFAULT 0 CHECK (search_volume >= 0),
    competition_level VARCHAR(20) CHECK (competition_level IN ('low', 'medium', 'high')),
    source VARCHAR(20) NOT NULL CHECK (source IN ('llm_analysis', 'csv_upload')),
    selected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on trend_selections table
ALTER TABLE trend_selections ENABLE ROW LEVEL SECURITY;

-- 3. Keyword Clusters table
CREATE TABLE IF NOT EXISTS keyword_clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    keyword_data_id UUID REFERENCES keyword_data(id) ON DELETE CASCADE,
    cluster_name VARCHAR(255) NOT NULL,
    cluster_description TEXT,
    keywords JSONB NOT NULL CHECK (jsonb_array_length(keywords) > 0),
    cluster_size INTEGER NOT NULL CHECK (cluster_size > 0),
    search_intent VARCHAR(20) CHECK (search_intent IN ('informational', 'commercial', 'transactional')),
    content_theme VARCHAR(255),
    priority_score DECIMAL(3,2) DEFAULT 0.0 CHECK (priority_score >= 0.0 AND priority_score <= 1.0),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on keyword_clusters table
ALTER TABLE keyword_clusters ENABLE ROW LEVEL SECURITY;

-- 4. Add external tool source fields to keyword_data table
ALTER TABLE keyword_data 
ADD COLUMN IF NOT EXISTS external_tool_source VARCHAR(20) CHECK (external_tool_source IN ('dataforseo', 'ahrefs', 'semrush', 'ubersuggest', 'manual')),
ADD COLUMN IF NOT EXISTS external_tool_metrics JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS cluster_id UUID REFERENCES keyword_clusters(id) ON DELETE SET NULL;

-- 5. Add CSV processing fields to trend_analysis table
ALTER TABLE trend_analysis 
ADD COLUMN IF NOT EXISTS csv_upload_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS csv_columns JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS csv_row_count INTEGER DEFAULT 0 CHECK (csv_row_count >= 0),
ADD COLUMN IF NOT EXISTS csv_processing_status VARCHAR(20) DEFAULT 'pending' CHECK (csv_processing_status IN ('pending', 'processing', 'completed', 'failed'));

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_workflow_sessions_user_id ON workflow_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_workflow_sessions_status ON workflow_sessions(status);
CREATE INDEX IF NOT EXISTS idx_workflow_sessions_user_status ON workflow_sessions(user_id, status);

CREATE INDEX IF NOT EXISTS idx_trend_selections_user_id ON trend_selections(user_id);
CREATE INDEX IF NOT EXISTS idx_trend_selections_trend_analysis_id ON trend_selections(trend_analysis_id);
CREATE INDEX IF NOT EXISTS idx_trend_selections_user_trend ON trend_selections(user_id, trend_analysis_id);

CREATE INDEX IF NOT EXISTS idx_keyword_clusters_user_id ON keyword_clusters(user_id);
CREATE INDEX IF NOT EXISTS idx_keyword_clusters_keyword_data_id ON keyword_clusters(keyword_data_id);
CREATE INDEX IF NOT EXISTS idx_keyword_clusters_user_keyword ON keyword_clusters(user_id, keyword_data_id);

CREATE INDEX IF NOT EXISTS idx_keyword_data_external_tool_source ON keyword_data(external_tool_source);
CREATE INDEX IF NOT EXISTS idx_keyword_data_cluster_id ON keyword_data(cluster_id);

-- RLS Policies for workflow_sessions
CREATE POLICY "Users can view their own workflow sessions" ON workflow_sessions
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own workflow sessions" ON workflow_sessions
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own workflow sessions" ON workflow_sessions
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own workflow sessions" ON workflow_sessions
    FOR DELETE USING (auth.uid() = user_id);

-- RLS Policies for trend_selections
CREATE POLICY "Users can view their own trend selections" ON trend_selections
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own trend selections" ON trend_selections
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own trend selections" ON trend_selections
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own trend selections" ON trend_selections
    FOR DELETE USING (auth.uid() = user_id);

-- RLS Policies for keyword_clusters
CREATE POLICY "Users can view their own keyword clusters" ON keyword_clusters
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own keyword clusters" ON keyword_clusters
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own keyword clusters" ON keyword_clusters
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own keyword clusters" ON keyword_clusters
    FOR DELETE USING (auth.uid() = user_id);
