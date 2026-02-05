-- Add status and publishing tracking fields to content_ideas table
-- This migration adds status tracking and publishing workflow fields

-- Add status field with proper constraints
ALTER TABLE content_ideas 
ADD COLUMN IF NOT EXISTS status VARCHAR(50) DEFAULT 'draft' 
CHECK (status IN ('draft', 'in_progress', 'completed', 'published', 'archived'));

-- Add publishing tracking fields
ALTER TABLE content_ideas 
ADD COLUMN IF NOT EXISTS published BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS published_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS published_to_titles BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS titles_record_id UUID;

-- Add priority field for content prioritization
ALTER TABLE content_ideas 
ADD COLUMN IF NOT EXISTS priority VARCHAR(50) DEFAULT 'medium' 
CHECK (priority IN ('low', 'medium', 'high', 'urgent'));

-- Add workflow tracking fields
ALTER TABLE content_ideas 
ADD COLUMN IF NOT EXISTS workflow_status VARCHAR(50) DEFAULT 'idea_generated',
ADD COLUMN IF NOT EXISTS content_generated BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS content_brief_generated BOOLEAN DEFAULT FALSE;

-- Add quality scoring fields
ALTER TABLE content_ideas 
ADD COLUMN IF NOT EXISTS overall_quality_score INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS seo_optimization_score INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS traffic_potential_score INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS viral_potential_score INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS competition_score INTEGER DEFAULT 0;

-- Add content structure fields
ALTER TABLE content_ideas 
ADD COLUMN IF NOT EXISTS content_outline JSONB DEFAULT '[]',
ADD COLUMN IF NOT EXISTS key_points JSONB DEFAULT '[]',
ADD COLUMN IF NOT EXISTS target_audience VARCHAR(255),
ADD COLUMN IF NOT EXISTS content_angle VARCHAR(255);

-- Add keyword enhancement fields
ALTER TABLE content_ideas 
ADD COLUMN IF NOT EXISTS primary_keywords JSONB DEFAULT '[]',
ADD COLUMN IF NOT EXISTS secondary_keywords JSONB DEFAULT '[]',
ADD COLUMN IF NOT EXISTS enhanced_keywords JSONB DEFAULT '[]',
ADD COLUMN IF NOT EXISTS keyword_research_data JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS keyword_research_enhanced BOOLEAN DEFAULT FALSE;

-- Add affiliate and monetization fields
ALTER TABLE content_ideas 
ADD COLUMN IF NOT EXISTS affiliate_opportunities JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS monetization_score INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS estimated_annual_revenue DECIMAL(10,2) DEFAULT 0,
ADD COLUMN IF NOT EXISTS monetization_priority VARCHAR(50) DEFAULT 'medium';

-- Add generation metadata
ALTER TABLE content_ideas 
ADD COLUMN IF NOT EXISTS generation_method VARCHAR(50) DEFAULT 'llm',
ADD COLUMN IF NOT EXISTS generation_prompt TEXT,
ADD COLUMN IF NOT EXISTS generation_parameters JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS enhancement_timestamp TIMESTAMP WITH TIME ZONE;

-- Add content metrics
ALTER TABLE content_ideas 
ADD COLUMN IF NOT EXISTS estimated_word_count INTEGER DEFAULT 2500,
ADD COLUMN IF NOT EXISTS estimated_reading_time INTEGER DEFAULT 12,
ADD COLUMN IF NOT EXISTS difficulty_level VARCHAR(20) DEFAULT 'medium';

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_content_ideas_status ON content_ideas(status);
CREATE INDEX IF NOT EXISTS idx_content_ideas_published ON content_ideas(published);
CREATE INDEX IF NOT EXISTS idx_content_ideas_published_to_titles ON content_ideas(published_to_titles);
CREATE INDEX IF NOT EXISTS idx_content_ideas_priority ON content_ideas(priority);
CREATE INDEX IF NOT EXISTS idx_content_ideas_workflow_status ON content_ideas(workflow_status);
CREATE INDEX IF NOT EXISTS idx_content_ideas_published_at ON content_ideas(published_at);

-- Add comments for documentation
COMMENT ON COLUMN content_ideas.status IS 'Current status of the content idea in the workflow';
COMMENT ON COLUMN content_ideas.published IS 'Whether the idea has been published to content generation';
COMMENT ON COLUMN content_ideas.published_at IS 'Timestamp when the idea was published';
COMMENT ON COLUMN content_ideas.published_to_titles IS 'Whether the idea was published to the Titles table';
COMMENT ON COLUMN content_ideas.titles_record_id IS 'ID of the corresponding record in the Titles table';
COMMENT ON COLUMN content_ideas.priority IS 'Priority level for content creation';
COMMENT ON COLUMN content_ideas.workflow_status IS 'Current status in the content creation workflow';
COMMENT ON COLUMN content_ideas.overall_quality_score IS 'Overall quality score (0-100)';
COMMENT ON COLUMN content_ideas.seo_optimization_score IS 'SEO optimization score (0-100)';
COMMENT ON COLUMN content_ideas.traffic_potential_score IS 'Traffic potential score (0-100)';
COMMENT ON COLUMN content_ideas.viral_potential_score IS 'Viral potential score (0-100)';
COMMENT ON COLUMN content_ideas.competition_score IS 'Competition difficulty score (0-100)';





