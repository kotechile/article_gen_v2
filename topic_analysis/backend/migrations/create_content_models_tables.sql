-- Create Content Models tables
-- This migration creates the content_ideas, keyword_clusters, and external_tool_results tables

-- Create content_ideas table
CREATE TABLE IF NOT EXISTS content_ideas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workflow_session_id UUID NOT NULL REFERENCES workflow_sessions(id) ON DELETE CASCADE,
    trend_analysis_id UUID REFERENCES trend_analyses(id) ON DELETE SET NULL,
    topic_decomposition_id UUID REFERENCES topic_decompositions(id) ON DELETE SET NULL,
    
    -- Content metadata
    title VARCHAR(500) NOT NULL,
    description TEXT,
    content_type VARCHAR(50) NOT NULL DEFAULT 'blog_post' CHECK (content_type IN ('blog_post', 'article', 'guide', 'review', 'tutorial', 'news', 'opinion', 'comparison', 'landing_page', 'product_page')),
    status VARCHAR(50) NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'in_progress', 'completed', 'published', 'archived')),
    priority VARCHAR(50) NOT NULL DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
    
    -- Content structure
    target_audience VARCHAR(255),
    content_angle VARCHAR(255),
    key_points JSONB NOT NULL DEFAULT '[]',
    content_outline JSONB NOT NULL DEFAULT '[]',
    
    -- SEO and Keywords
    primary_keyword VARCHAR(255) NOT NULL,
    secondary_keywords JSONB NOT NULL DEFAULT '[]',
    enhanced_keywords JSONB NOT NULL DEFAULT '[]',
    keyword_difficulty INTEGER CHECK (keyword_difficulty >= 0 AND keyword_difficulty <= 100),
    search_volume INTEGER,
    cpc VARCHAR(20),
    
    -- Affiliate integration
    affiliate_offers JSONB NOT NULL DEFAULT '[]',
    affiliate_links JSONB NOT NULL DEFAULT '[]',
    monetization_strategy TEXT,
    expected_revenue VARCHAR(20),
    
    -- Content generation metadata
    generation_prompt TEXT,
    generation_model VARCHAR(100),
    generation_parameters JSONB NOT NULL DEFAULT '{}',
    generation_time_ms INTEGER,
    
    -- Quality metrics
    readability_score INTEGER CHECK (readability_score >= 0 AND readability_score <= 100),
    seo_score INTEGER CHECK (seo_score >= 0 AND seo_score <= 100),
    engagement_score INTEGER CHECK (engagement_score >= 0 AND engagement_score <= 100),
    quality_notes TEXT,
    
    -- Publishing metadata
    target_publish_date TIMESTAMP WITH TIME ZONE,
    actual_publish_date TIMESTAMP WITH TIME ZONE,
    publish_url VARCHAR(500),
    word_count INTEGER,
    reading_time_minutes INTEGER,
    
    -- Tags and categories
    tags JSONB NOT NULL DEFAULT '[]',
    categories JSONB NOT NULL DEFAULT '[]',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Create keyword_clusters table
CREATE TABLE IF NOT EXISTS keyword_clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workflow_session_id UUID NOT NULL REFERENCES workflow_sessions(id) ON DELETE CASCADE,
    trend_analysis_id UUID REFERENCES trend_analyses(id) ON DELETE SET NULL,
    
    -- Cluster metadata
    cluster_name VARCHAR(255) NOT NULL,
    cluster_description VARCHAR(1000),
    cluster_type VARCHAR(50) NOT NULL DEFAULT 'semantic' CHECK (cluster_type IN ('semantic', 'topic', 'intent', 'difficulty')),
    
    -- Keywords data
    keywords JSONB NOT NULL DEFAULT '[]',
    primary_keyword VARCHAR(255) NOT NULL,
    secondary_keywords JSONB NOT NULL DEFAULT '[]',
    long_tail_keywords JSONB NOT NULL DEFAULT '[]',
    
    -- SEO metrics
    avg_search_volume INTEGER,
    avg_keyword_difficulty DECIMAL(5,2),
    avg_cpc DECIMAL(10,2),
    total_search_volume INTEGER,
    competition_level VARCHAR(50) CHECK (competition_level IN ('low', 'medium', 'high')),
    
    -- Clustering metrics
    cluster_size INTEGER NOT NULL DEFAULT 0,
    cluster_density DECIMAL(5,3),
    semantic_similarity DECIMAL(5,3),
    intent_consistency DECIMAL(5,3),
    
    -- Content generation
    content_ideas JSONB NOT NULL DEFAULT '[]',
    content_angles JSONB NOT NULL DEFAULT '[]',
    target_audiences JSONB NOT NULL DEFAULT '[]',
    
    -- External tool data
    source_tool VARCHAR(50),
    external_data JSONB NOT NULL DEFAULT '{}',
    processing_notes VARCHAR(1000),
    
    -- Quality metrics
    cluster_quality_score DECIMAL(5,2) CHECK (cluster_quality_score >= 0 AND cluster_quality_score <= 100),
    keyword_relevance_score DECIMAL(5,2) CHECK (keyword_relevance_score >= 0 AND keyword_relevance_score <= 100),
    content_potential_score DECIMAL(5,2) CHECK (content_potential_score >= 0 AND content_potential_score <= 100),
    
    -- Status and flags
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_processed BOOLEAN NOT NULL DEFAULT FALSE,
    is_used_for_content BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Create external_tool_results table
CREATE TABLE IF NOT EXISTS external_tool_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workflow_session_id UUID NOT NULL REFERENCES workflow_sessions(id) ON DELETE CASCADE,
    trend_analysis_id UUID REFERENCES trend_analyses(id) ON DELETE SET NULL,
    
    -- Tool metadata
    tool_name VARCHAR(50) NOT NULL CHECK (tool_name IN ('semrush', 'ahrefs', 'ubersuggest', 'manual')),
    tool_version VARCHAR(20),
    api_key_id UUID REFERENCES api_keys(id) ON DELETE SET NULL,
    
    -- Query information
    query_type VARCHAR(50) NOT NULL CHECK (query_type IN ('keyword_research', 'competitor_analysis', 'content_ideas', 'trend_analysis')),
    query_parameters JSONB NOT NULL DEFAULT '{}',
    seed_keywords JSONB NOT NULL DEFAULT '[]',
    
    -- Results data
    raw_results JSONB NOT NULL DEFAULT '{}',
    processed_results JSONB NOT NULL DEFAULT '{}',
    keywords_data JSONB NOT NULL DEFAULT '[]',
    clusters_data JSONB NOT NULL DEFAULT '[]',
    
    -- Statistics
    total_keywords INTEGER NOT NULL DEFAULT 0,
    total_clusters INTEGER NOT NULL DEFAULT 0,
    avg_search_volume DECIMAL(10,2),
    avg_keyword_difficulty DECIMAL(5,2),
    avg_cpc DECIMAL(10,2),
    total_search_volume INTEGER,
    
    -- Quality metrics
    data_quality_score DECIMAL(5,2) CHECK (data_quality_score >= 0 AND data_quality_score <= 100),
    completeness_score DECIMAL(5,2) CHECK (completeness_score >= 0 AND completeness_score <= 100),
    relevance_score DECIMAL(5,2) CHECK (relevance_score >= 0 AND relevance_score <= 100),
    
    -- Processing metadata
    processing_time_ms INTEGER,
    api_calls_made INTEGER NOT NULL DEFAULT 0,
    rate_limit_hit BOOLEAN NOT NULL DEFAULT FALSE,
    error_count INTEGER NOT NULL DEFAULT 0,
    warning_count INTEGER NOT NULL DEFAULT 0,
    
    -- Status and flags
    status VARCHAR(50) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    is_processed BOOLEAN NOT NULL DEFAULT FALSE,
    is_used_for_content BOOLEAN NOT NULL DEFAULT FALSE,
    is_archived BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Error handling
    error_message TEXT,
    error_details JSONB NOT NULL DEFAULT '{}',
    retry_count INTEGER NOT NULL DEFAULT 0,
    last_retry_at TIMESTAMP WITH TIME ZONE,
    
    -- Notes and metadata
    processing_notes TEXT,
    user_notes TEXT,
    tags JSONB NOT NULL DEFAULT '[]',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    processed_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for content_ideas table
CREATE INDEX IF NOT EXISTS idx_content_ideas_user_id ON content_ideas(user_id);
CREATE INDEX IF NOT EXISTS idx_content_ideas_workflow_session_id ON content_ideas(workflow_session_id);
CREATE INDEX IF NOT EXISTS idx_content_ideas_trend_analysis_id ON content_ideas(trend_analysis_id);
CREATE INDEX IF NOT EXISTS idx_content_ideas_topic_decomposition_id ON content_ideas(topic_decomposition_id);
CREATE INDEX IF NOT EXISTS idx_content_ideas_content_type ON content_ideas(content_type);
CREATE INDEX IF NOT EXISTS idx_content_ideas_status ON content_ideas(status);
CREATE INDEX IF NOT EXISTS idx_content_ideas_priority ON content_ideas(priority);
CREATE INDEX IF NOT EXISTS idx_content_ideas_created_at ON content_ideas(created_at);
CREATE INDEX IF NOT EXISTS idx_content_ideas_user_status ON content_ideas(user_id, status);
CREATE INDEX IF NOT EXISTS idx_content_ideas_workflow_status ON content_ideas(workflow_session_id, status);
CREATE INDEX IF NOT EXISTS idx_content_ideas_primary_keyword ON content_ideas(primary_keyword);
CREATE INDEX IF NOT EXISTS idx_content_ideas_target_publish_date ON content_ideas(target_publish_date);

-- Create indexes for keyword_clusters table
CREATE INDEX IF NOT EXISTS idx_keyword_clusters_user_id ON keyword_clusters(user_id);
CREATE INDEX IF NOT EXISTS idx_keyword_clusters_workflow_session_id ON keyword_clusters(workflow_session_id);
CREATE INDEX IF NOT EXISTS idx_keyword_clusters_trend_analysis_id ON keyword_clusters(trend_analysis_id);
CREATE INDEX IF NOT EXISTS idx_keyword_clusters_cluster_type ON keyword_clusters(cluster_type);
CREATE INDEX IF NOT EXISTS idx_keyword_clusters_is_active ON keyword_clusters(is_active);
CREATE INDEX IF NOT EXISTS idx_keyword_clusters_is_processed ON keyword_clusters(is_processed);
CREATE INDEX IF NOT EXISTS idx_keyword_clusters_is_used_for_content ON keyword_clusters(is_used_for_content);
CREATE INDEX IF NOT EXISTS idx_keyword_clusters_created_at ON keyword_clusters(created_at);
CREATE INDEX IF NOT EXISTS idx_keyword_clusters_user_active ON keyword_clusters(user_id, is_active);
CREATE INDEX IF NOT EXISTS idx_keyword_clusters_primary_keyword ON keyword_clusters(primary_keyword);

-- Create indexes for external_tool_results table
CREATE INDEX IF NOT EXISTS idx_external_tool_results_user_id ON external_tool_results(user_id);
CREATE INDEX IF NOT EXISTS idx_external_tool_results_workflow_session_id ON external_tool_results(workflow_session_id);
CREATE INDEX IF NOT EXISTS idx_external_tool_results_trend_analysis_id ON external_tool_results(trend_analysis_id);
CREATE INDEX IF NOT EXISTS idx_external_tool_results_tool_name ON external_tool_results(tool_name);
CREATE INDEX IF NOT EXISTS idx_external_tool_results_query_type ON external_tool_results(query_type);
CREATE INDEX IF NOT EXISTS idx_external_tool_results_status ON external_tool_results(status);
CREATE INDEX IF NOT EXISTS idx_external_tool_results_is_processed ON external_tool_results(is_processed);
CREATE INDEX IF NOT EXISTS idx_external_tool_results_is_used_for_content ON external_tool_results(is_used_for_content);
CREATE INDEX IF NOT EXISTS idx_external_tool_results_is_archived ON external_tool_results(is_archived);
CREATE INDEX IF NOT EXISTS idx_external_tool_results_created_at ON external_tool_results(created_at);
CREATE INDEX IF NOT EXISTS idx_external_tool_results_user_status ON external_tool_results(user_id, status);
CREATE INDEX IF NOT EXISTS idx_external_tool_results_workflow_status ON external_tool_results(workflow_session_id, status);

-- Create GIN indexes for JSONB columns
CREATE INDEX IF NOT EXISTS idx_content_ideas_key_points_gin ON content_ideas USING GIN (key_points);
CREATE INDEX IF NOT EXISTS idx_content_ideas_content_outline_gin ON content_ideas USING GIN (content_outline);
CREATE INDEX IF NOT EXISTS idx_content_ideas_secondary_keywords_gin ON content_ideas USING GIN (secondary_keywords);
CREATE INDEX IF NOT EXISTS idx_content_ideas_enhanced_keywords_gin ON content_ideas USING GIN (enhanced_keywords);
CREATE INDEX IF NOT EXISTS idx_content_ideas_affiliate_offers_gin ON content_ideas USING GIN (affiliate_offers);
CREATE INDEX IF NOT EXISTS idx_content_ideas_affiliate_links_gin ON content_ideas USING GIN (affiliate_links);
CREATE INDEX IF NOT EXISTS idx_content_ideas_tags_gin ON content_ideas USING GIN (tags);
CREATE INDEX IF NOT EXISTS idx_content_ideas_categories_gin ON content_ideas USING GIN (categories);

CREATE INDEX IF NOT EXISTS idx_keyword_clusters_keywords_gin ON keyword_clusters USING GIN (keywords);
CREATE INDEX IF NOT EXISTS idx_keyword_clusters_secondary_keywords_gin ON keyword_clusters USING GIN (secondary_keywords);
CREATE INDEX IF NOT EXISTS idx_keyword_clusters_long_tail_keywords_gin ON keyword_clusters USING GIN (long_tail_keywords);
CREATE INDEX IF NOT EXISTS idx_keyword_clusters_content_ideas_gin ON keyword_clusters USING GIN (content_ideas);
CREATE INDEX IF NOT EXISTS idx_keyword_clusters_content_angles_gin ON keyword_clusters USING GIN (content_angles);
CREATE INDEX IF NOT EXISTS idx_keyword_clusters_target_audiences_gin ON keyword_clusters USING GIN (target_audiences);
CREATE INDEX IF NOT EXISTS idx_keyword_clusters_external_data_gin ON keyword_clusters USING GIN (external_data);

CREATE INDEX IF NOT EXISTS idx_external_tool_results_query_parameters_gin ON external_tool_results USING GIN (query_parameters);
CREATE INDEX IF NOT EXISTS idx_external_tool_results_seed_keywords_gin ON external_tool_results USING GIN (seed_keywords);
CREATE INDEX IF NOT EXISTS idx_external_tool_results_raw_results_gin ON external_tool_results USING GIN (raw_results);
CREATE INDEX IF NOT EXISTS idx_external_tool_results_processed_results_gin ON external_tool_results USING GIN (processed_results);
CREATE INDEX IF NOT EXISTS idx_external_tool_results_keywords_data_gin ON external_tool_results USING GIN (keywords_data);
CREATE INDEX IF NOT EXISTS idx_external_tool_results_clusters_data_gin ON external_tool_results USING GIN (clusters_data);
CREATE INDEX IF NOT EXISTS idx_external_tool_results_error_details_gin ON external_tool_results USING GIN (error_details);
CREATE INDEX IF NOT EXISTS idx_external_tool_results_tags_gin ON external_tool_results USING GIN (tags);

-- Enable Row Level Security
ALTER TABLE content_ideas ENABLE ROW LEVEL SECURITY;
ALTER TABLE keyword_clusters ENABLE ROW LEVEL SECURITY;
ALTER TABLE external_tool_results ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for content_ideas
CREATE POLICY "Users can view their own content ideas" ON content_ideas
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own content ideas" ON content_ideas
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own content ideas" ON content_ideas
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own content ideas" ON content_ideas
    FOR DELETE USING (auth.uid() = user_id);

-- Create RLS policies for keyword_clusters
CREATE POLICY "Users can view their own keyword clusters" ON keyword_clusters
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own keyword clusters" ON keyword_clusters
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own keyword clusters" ON keyword_clusters
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own keyword clusters" ON keyword_clusters
    FOR DELETE USING (auth.uid() = user_id);

-- Create RLS policies for external_tool_results
CREATE POLICY "Users can view their own external tool results" ON external_tool_results
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own external tool results" ON external_tool_results
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own external tool results" ON external_tool_results
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own external tool results" ON external_tool_results
    FOR DELETE USING (auth.uid() = user_id);

-- Add comments
COMMENT ON TABLE content_ideas IS 'Stores generated content ideas with enhanced keywords and affiliate offers';
COMMENT ON TABLE keyword_clusters IS 'Stores keyword clustering results from external tools';
COMMENT ON TABLE external_tool_results IS 'Stores results from external keyword research tools';

-- Add column comments for content_ideas
COMMENT ON COLUMN content_ideas.title IS 'Content title';
COMMENT ON COLUMN content_ideas.description IS 'Content description';
COMMENT ON COLUMN content_ideas.content_type IS 'Type of content (blog_post, article, guide, etc.)';
COMMENT ON COLUMN content_ideas.status IS 'Content status (draft, in_progress, completed, published, archived)';
COMMENT ON COLUMN content_ideas.priority IS 'Content priority (low, medium, high, urgent)';
COMMENT ON COLUMN content_ideas.primary_keyword IS 'Primary keyword for SEO';
COMMENT ON COLUMN content_ideas.secondary_keywords IS 'Secondary keywords array';
COMMENT ON COLUMN content_ideas.enhanced_keywords IS 'Enhanced keywords from external tools';
COMMENT ON COLUMN content_ideas.affiliate_offers IS 'Affiliate offer IDs array';
COMMENT ON COLUMN content_ideas.affiliate_links IS 'Affiliate links array';
COMMENT ON COLUMN content_ideas.monetization_strategy IS 'Strategy for monetizing content';
COMMENT ON COLUMN content_ideas.readability_score IS 'Content readability score (0-100)';
COMMENT ON COLUMN content_ideas.seo_score IS 'SEO optimization score (0-100)';
COMMENT ON COLUMN content_ideas.engagement_score IS 'Engagement potential score (0-100)';

-- Add column comments for keyword_clusters
COMMENT ON COLUMN keyword_clusters.cluster_name IS 'Name of the keyword cluster';
COMMENT ON COLUMN keyword_clusters.cluster_type IS 'Type of clustering (semantic, topic, intent, difficulty)';
COMMENT ON COLUMN keyword_clusters.keywords IS 'Array of keyword objects in cluster';
COMMENT ON COLUMN keyword_clusters.primary_keyword IS 'Primary keyword for the cluster';
COMMENT ON COLUMN keyword_clusters.avg_search_volume IS 'Average search volume for keywords in cluster';
COMMENT ON COLUMN keyword_clusters.avg_keyword_difficulty IS 'Average keyword difficulty for cluster';
COMMENT ON COLUMN keyword_clusters.cluster_quality_score IS 'Overall cluster quality score (0-100)';
COMMENT ON COLUMN keyword_clusters.is_used_for_content IS 'Whether cluster has been used for content generation';

-- Add column comments for external_tool_results
COMMENT ON COLUMN external_tool_results.tool_name IS 'Name of external tool (semrush, ahrefs, ubersuggest)';
COMMENT ON COLUMN external_tool_results.query_type IS 'Type of query performed';
COMMENT ON COLUMN external_tool_results.seed_keywords IS 'Seed keywords used for the query';
COMMENT ON COLUMN external_tool_results.keywords_data IS 'Processed keywords data from tool';
COMMENT ON COLUMN external_tool_results.clusters_data IS 'Generated keyword clusters';
COMMENT ON COLUMN external_tool_results.data_quality_score IS 'Quality score of the data (0-100)';
COMMENT ON COLUMN external_tool_results.completeness_score IS 'Completeness score of the data (0-100)';
COMMENT ON COLUMN external_tool_results.relevance_score IS 'Relevance score of the data (0-100)';
COMMENT ON COLUMN external_tool_results.processing_time_ms IS 'Time taken to process the request in milliseconds';
COMMENT ON COLUMN external_tool_results.api_calls_made IS 'Number of API calls made to external tool';
