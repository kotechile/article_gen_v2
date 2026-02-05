-- Enhanced Idea Burst Tables
-- This migration creates tables for individual keyword optimization and enhanced idea management

-- 1. Individual Keywords table for storing Ahrefs keyword data
CREATE TABLE IF NOT EXISTS individual_keywords (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content_idea_id UUID REFERENCES content_ideas(id) ON DELETE CASCADE,
    
    -- Keyword basic info
    keyword VARCHAR(500) NOT NULL,
    keyword_type VARCHAR(50) NOT NULL DEFAULT 'primary' CHECK (keyword_type IN ('primary', 'secondary', 'long_tail', 'semantic', 'question_based', 'comparison', 'technical', 'beginner', 'advanced')),
    
    -- Ahrefs metrics
    search_volume INTEGER,
    keyword_difficulty INTEGER CHECK (keyword_difficulty >= 0 AND keyword_difficulty <= 100),
    cpc DECIMAL(10,2),
    traffic_potential INTEGER,
    clicks INTEGER,
    impressions INTEGER,
    ctr DECIMAL(5,2),
    position DECIMAL(5,2),
    
    -- SEO metrics
    search_intent VARCHAR(50) CHECK (search_intent IN ('informational', 'navigational', 'transactional', 'commercial')),
    competition_level VARCHAR(20) CHECK (competition_level IN ('low', 'medium', 'high', 'very_high')),
    trend_score DECIMAL(5,2) CHECK (trend_score >= 0 AND trend_score <= 100),
    opportunity_score DECIMAL(5,2) CHECK (opportunity_score >= 0 AND opportunity_score <= 100),
    
    -- Content optimization
    content_suggestions JSONB NOT NULL DEFAULT '[]',
    heading_suggestions JSONB NOT NULL DEFAULT '[]',
    internal_link_suggestions JSONB NOT NULL DEFAULT '[]',
    related_questions JSONB NOT NULL DEFAULT '[]',
    
    -- LLM optimization
    llm_optimized_title VARCHAR(500),
    llm_optimized_description TEXT,
    llm_keyword_variations JSONB NOT NULL DEFAULT '[]',
    llm_content_angle VARCHAR(255),
    llm_target_audience VARCHAR(255),
    
    -- Affiliate integration
    affiliate_potential_score DECIMAL(5,2) CHECK (affiliate_potential_score >= 0 AND affiliate_potential_score <= 100),
    suggested_affiliate_networks JSONB NOT NULL DEFAULT '[]',
    monetization_opportunities JSONB NOT NULL DEFAULT '[]',
    
    -- Source tracking
    source_tool VARCHAR(50) NOT NULL DEFAULT 'ahrefs' CHECK (source_tool IN ('ahrefs', 'semrush', 'ubersuggest', 'dataforseo', 'manual')),
    source_file_name VARCHAR(255),
    source_row_number INTEGER,
    
    -- Quality metrics
    relevance_score DECIMAL(5,2) CHECK (relevance_score >= 0 AND relevance_score <= 100),
    optimization_score DECIMAL(5,2) CHECK (optimization_score >= 0 AND optimization_score <= 100),
    priority_score DECIMAL(5,2) CHECK (priority_score >= 0 AND priority_score <= 100),
    
    -- Status and flags
    is_optimized BOOLEAN NOT NULL DEFAULT FALSE,
    is_used_in_content BOOLEAN NOT NULL DEFAULT FALSE,
    is_archived BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    optimized_at TIMESTAMP WITH TIME ZONE
);

-- 2. Enhanced Content Ideas table (extends existing content_ideas)
ALTER TABLE content_ideas 
ADD COLUMN IF NOT EXISTS enhanced_keywords_data JSONB NOT NULL DEFAULT '[]',
ADD COLUMN IF NOT EXISTS seo_optimized_title VARCHAR(500),
ADD COLUMN IF NOT EXISTS seo_optimized_description TEXT,
ADD COLUMN IF NOT EXISTS primary_keywords_optimized JSONB NOT NULL DEFAULT '[]',
ADD COLUMN IF NOT EXISTS keyword_metrics_summary JSONB NOT NULL DEFAULT '{}',
ADD COLUMN IF NOT EXISTS affiliate_networks_suggested JSONB NOT NULL DEFAULT '[]',
ADD COLUMN IF NOT EXISTS content_generation_prompt TEXT,
ADD COLUMN IF NOT EXISTS content_generation_parameters JSONB NOT NULL DEFAULT '{}',
ADD COLUMN IF NOT EXISTS is_enhanced BOOLEAN NOT NULL DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS enhancement_timestamp TIMESTAMP WITH TIME ZONE;

-- 3. Keyword Optimization Sessions table
CREATE TABLE IF NOT EXISTS keyword_optimization_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content_idea_id UUID NOT NULL REFERENCES content_ideas(id) ON DELETE CASCADE,
    
    -- Session metadata
    session_name VARCHAR(255) NOT NULL,
    session_description TEXT,
    optimization_type VARCHAR(50) NOT NULL DEFAULT 'ahrefs_upload' CHECK (optimization_type IN ('ahrefs_upload', 'semrush_upload', 'manual_entry', 'llm_generation')),
    
    -- Source data
    source_file_name VARCHAR(255),
    source_file_size INTEGER,
    keywords_processed INTEGER NOT NULL DEFAULT 0,
    keywords_optimized INTEGER NOT NULL DEFAULT 0,
    
    -- Optimization results
    optimization_summary JSONB NOT NULL DEFAULT '{}',
    top_performing_keywords JSONB NOT NULL DEFAULT '[]',
    optimization_recommendations JSONB NOT NULL DEFAULT '[]',
    
    -- Quality metrics
    overall_optimization_score DECIMAL(5,2) CHECK (overall_optimization_score >= 0 AND overall_optimization_score <= 100),
    seo_improvement_score DECIMAL(5,2) CHECK (seo_improvement_score >= 0 AND seo_improvement_score <= 100),
    content_potential_score DECIMAL(5,2) CHECK (content_potential_score >= 0 AND content_potential_score <= 100),
    
    -- Status
    status VARCHAR(50) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- 4. Content Generation Templates table
CREATE TABLE IF NOT EXISTS content_generation_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Template metadata
    template_name VARCHAR(255) NOT NULL,
    template_description TEXT,
    template_type VARCHAR(50) NOT NULL CHECK (template_type IN ('seo_title', 'meta_description', 'content_outline', 'affiliate_integration', 'full_article')),
    
    -- Template content
    template_prompt TEXT NOT NULL,
    template_variables JSONB NOT NULL DEFAULT '[]',
    template_examples JSONB NOT NULL DEFAULT '[]',
    
    -- SEO optimization
    seo_focus_keywords JSONB NOT NULL DEFAULT '[]',
    target_audience VARCHAR(255),
    content_angle VARCHAR(255),
    tone VARCHAR(50) CHECK (tone IN ('professional', 'casual', 'technical', 'conversational', 'authoritative')),
    
    -- Performance metrics
    usage_count INTEGER NOT NULL DEFAULT 0,
    success_rate DECIMAL(5,2) CHECK (success_rate >= 0 AND success_rate <= 100),
    avg_seo_score DECIMAL(5,2) CHECK (avg_seo_score >= 0 AND avg_seo_score <= 100),
    
    -- Status
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_public BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Create indexes for individual_keywords table
CREATE INDEX IF NOT EXISTS idx_individual_keywords_user_id ON individual_keywords(user_id);
CREATE INDEX IF NOT EXISTS idx_individual_keywords_content_idea_id ON individual_keywords(content_idea_id);
CREATE INDEX IF NOT EXISTS idx_individual_keywords_keyword_type ON individual_keywords(keyword_type);
CREATE INDEX IF NOT EXISTS idx_individual_keywords_source_tool ON individual_keywords(source_tool);
CREATE INDEX IF NOT EXISTS idx_individual_keywords_is_optimized ON individual_keywords(is_optimized);
CREATE INDEX IF NOT EXISTS idx_individual_keywords_is_used_in_content ON individual_keywords(is_used_in_content);
CREATE INDEX IF NOT EXISTS idx_individual_keywords_priority_score ON individual_keywords(priority_score);
CREATE INDEX IF NOT EXISTS idx_individual_keywords_created_at ON individual_keywords(created_at);
CREATE INDEX IF NOT EXISTS idx_individual_keywords_user_optimized ON individual_keywords(user_id, is_optimized);

-- Create indexes for keyword_optimization_sessions table
CREATE INDEX IF NOT EXISTS idx_keyword_optimization_sessions_user_id ON keyword_optimization_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_keyword_optimization_sessions_content_idea_id ON keyword_optimization_sessions(content_idea_id);
CREATE INDEX IF NOT EXISTS idx_keyword_optimization_sessions_optimization_type ON keyword_optimization_sessions(optimization_type);
CREATE INDEX IF NOT EXISTS idx_keyword_optimization_sessions_status ON keyword_optimization_sessions(status);
CREATE INDEX IF NOT EXISTS idx_keyword_optimization_sessions_is_active ON keyword_optimization_sessions(is_active);
CREATE INDEX IF NOT EXISTS idx_keyword_optimization_sessions_created_at ON keyword_optimization_sessions(created_at);

-- Create indexes for content_generation_templates table
CREATE INDEX IF NOT EXISTS idx_content_generation_templates_user_id ON content_generation_templates(user_id);
CREATE INDEX IF NOT EXISTS idx_content_generation_templates_template_type ON content_generation_templates(template_type);
CREATE INDEX IF NOT EXISTS idx_content_generation_templates_is_active ON content_generation_templates(is_active);
CREATE INDEX IF NOT EXISTS idx_content_generation_templates_is_public ON content_generation_templates(is_public);
CREATE INDEX IF NOT EXISTS idx_content_generation_templates_usage_count ON content_generation_templates(usage_count);

-- Create GIN indexes for JSONB columns
CREATE INDEX IF NOT EXISTS idx_individual_keywords_content_suggestions_gin ON individual_keywords USING GIN (content_suggestions);
CREATE INDEX IF NOT EXISTS idx_individual_keywords_heading_suggestions_gin ON individual_keywords USING GIN (heading_suggestions);
CREATE INDEX IF NOT EXISTS idx_individual_keywords_internal_link_suggestions_gin ON individual_keywords USING GIN (internal_link_suggestions);
CREATE INDEX IF NOT EXISTS idx_individual_keywords_related_questions_gin ON individual_keywords USING GIN (related_questions);
CREATE INDEX IF NOT EXISTS idx_individual_keywords_llm_keyword_variations_gin ON individual_keywords USING GIN (llm_keyword_variations);
CREATE INDEX IF NOT EXISTS idx_individual_keywords_suggested_affiliate_networks_gin ON individual_keywords USING GIN (suggested_affiliate_networks);
CREATE INDEX IF NOT EXISTS idx_individual_keywords_monetization_opportunities_gin ON individual_keywords USING GIN (monetization_opportunities);

CREATE INDEX IF NOT EXISTS idx_keyword_optimization_sessions_optimization_summary_gin ON keyword_optimization_sessions USING GIN (optimization_summary);
CREATE INDEX IF NOT EXISTS idx_keyword_optimization_sessions_top_performing_keywords_gin ON keyword_optimization_sessions USING GIN (top_performing_keywords);
CREATE INDEX IF NOT EXISTS idx_keyword_optimization_sessions_optimization_recommendations_gin ON keyword_optimization_sessions USING GIN (optimization_recommendations);

CREATE INDEX IF NOT EXISTS idx_content_generation_templates_template_variables_gin ON content_generation_templates USING GIN (template_variables);
CREATE INDEX IF NOT EXISTS idx_content_generation_templates_template_examples_gin ON content_generation_templates USING GIN (template_examples);
CREATE INDEX IF NOT EXISTS idx_content_generation_templates_seo_focus_keywords_gin ON content_generation_templates USING GIN (seo_focus_keywords);

-- Enable Row Level Security
ALTER TABLE individual_keywords ENABLE ROW LEVEL SECURITY;
ALTER TABLE keyword_optimization_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE content_generation_templates ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for individual_keywords
CREATE POLICY "Users can view their own individual keywords" ON individual_keywords
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own individual keywords" ON individual_keywords
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own individual keywords" ON individual_keywords
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own individual keywords" ON individual_keywords
    FOR DELETE USING (auth.uid() = user_id);

-- Create RLS policies for keyword_optimization_sessions
CREATE POLICY "Users can view their own optimization sessions" ON keyword_optimization_sessions
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own optimization sessions" ON keyword_optimization_sessions
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own optimization sessions" ON keyword_optimization_sessions
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own optimization sessions" ON keyword_optimization_sessions
    FOR DELETE USING (auth.uid() = user_id);

-- Create RLS policies for content_generation_templates
CREATE POLICY "Users can view their own templates and public templates" ON content_generation_templates
    FOR SELECT USING (auth.uid() = user_id OR is_public = true);

CREATE POLICY "Users can insert their own templates" ON content_generation_templates
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own templates" ON content_generation_templates
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own templates" ON content_generation_templates
    FOR DELETE USING (auth.uid() = user_id);

-- Add comments
COMMENT ON TABLE individual_keywords IS 'Stores individual keywords with Ahrefs metrics and LLM optimization';
COMMENT ON TABLE keyword_optimization_sessions IS 'Tracks keyword optimization sessions and results';
COMMENT ON TABLE content_generation_templates IS 'Stores reusable templates for content generation';

-- Add column comments for individual_keywords
COMMENT ON COLUMN individual_keywords.keyword IS 'The actual keyword phrase';
COMMENT ON COLUMN individual_keywords.keyword_type IS 'Type of keyword (primary, secondary, long_tail, etc.)';
COMMENT ON COLUMN individual_keywords.search_volume IS 'Monthly search volume from Ahrefs';
COMMENT ON COLUMN individual_keywords.keyword_difficulty IS 'Keyword difficulty score (0-100)';
COMMENT ON COLUMN individual_keywords.cpc IS 'Cost per click';
COMMENT ON COLUMN individual_keywords.traffic_potential IS 'Estimated traffic potential';
COMMENT ON COLUMN individual_keywords.search_intent IS 'Search intent classification';
COMMENT ON COLUMN individual_keywords.opportunity_score IS 'Overall opportunity score (0-100)';
COMMENT ON COLUMN individual_keywords.llm_optimized_title IS 'LLM-generated optimized title for this keyword';
COMMENT ON COLUMN individual_keywords.llm_optimized_description IS 'LLM-generated optimized description';
COMMENT ON COLUMN individual_keywords.affiliate_potential_score IS 'Affiliate monetization potential (0-100)';
COMMENT ON COLUMN individual_keywords.relevance_score IS 'Relevance to content idea (0-100)';
COMMENT ON COLUMN individual_keywords.optimization_score IS 'Overall optimization score (0-100)';
COMMENT ON COLUMN individual_keywords.priority_score IS 'Priority for content creation (0-100)';

-- Add column comments for keyword_optimization_sessions
COMMENT ON COLUMN keyword_optimization_sessions.session_name IS 'Name of the optimization session';
COMMENT ON COLUMN keyword_optimization_sessions.optimization_type IS 'Type of optimization performed';
COMMENT ON COLUMN keyword_optimization_sessions.keywords_processed IS 'Total keywords processed in session';
COMMENT ON COLUMN keyword_optimization_sessions.keywords_optimized IS 'Keywords successfully optimized';
COMMENT ON COLUMN keyword_optimization_sessions.overall_optimization_score IS 'Overall session optimization score (0-100)';

-- Add column comments for content_generation_templates
COMMENT ON COLUMN content_generation_templates.template_name IS 'Name of the template';
COMMENT ON COLUMN content_generation_templates.template_type IS 'Type of content template';
COMMENT ON COLUMN content_generation_templates.template_prompt IS 'LLM prompt for content generation';
COMMENT ON COLUMN content_generation_templates.template_variables IS 'Available variables for template';
COMMENT ON COLUMN content_generation_templates.usage_count IS 'Number of times template has been used';
COMMENT ON COLUMN content_generation_templates.success_rate IS 'Success rate of template (0-100)';

