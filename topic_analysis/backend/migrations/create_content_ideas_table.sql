-- Create content ideas table for storing generated blog and software ideas
CREATE TABLE IF NOT EXISTS content_ideas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    description TEXT,
    content_type VARCHAR(20) NOT NULL CHECK (content_type IN ('blog', 'software')),
    category VARCHAR(50) NOT NULL, -- e.g., 'seo_optimized', 'software_tool', 'saas_idea'
    subtopic TEXT NOT NULL,
    topic_id UUID NOT NULL,
    user_id UUID NOT NULL,
    keywords TEXT[], -- Array of keywords used for this idea
    seo_score INTEGER DEFAULT 0, -- SEO optimization score (0-100)
    difficulty_level VARCHAR(20) DEFAULT 'medium', -- easy, medium, hard
    estimated_read_time INTEGER, -- in minutes
    target_audience TEXT,
    content_angle TEXT,
    monetization_potential VARCHAR(20) DEFAULT 'medium', -- low, medium, high
    technical_complexity VARCHAR(20) DEFAULT 'medium', -- low, medium, high (for software ideas)
    development_effort VARCHAR(20) DEFAULT 'medium', -- low, medium, high (for software ideas)
    market_demand VARCHAR(20) DEFAULT 'medium', -- low, medium, high
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_content_ideas_topic_id ON content_ideas(topic_id);
CREATE INDEX IF NOT EXISTS idx_content_ideas_user_id ON content_ideas(user_id);
CREATE INDEX IF NOT EXISTS idx_content_ideas_content_type ON content_ideas(content_type);
CREATE INDEX IF NOT EXISTS idx_content_ideas_category ON content_ideas(category);
CREATE INDEX IF NOT EXISTS idx_content_ideas_subtopic ON content_ideas(subtopic);
CREATE INDEX IF NOT EXISTS idx_content_ideas_created_at ON content_ideas(created_at);
CREATE INDEX IF NOT EXISTS idx_content_ideas_user_topic ON content_ideas(user_id, topic_id);
CREATE INDEX IF NOT EXISTS idx_content_ideas_type_category ON content_ideas(content_type, category);

-- Enable RLS
ALTER TABLE content_ideas ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
CREATE POLICY "Users can view their own content ideas" ON content_ideas
    FOR SELECT USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can insert their own content ideas" ON content_ideas
    FOR INSERT WITH CHECK (auth.uid()::text = user_id::text);

CREATE POLICY "Users can update their own content ideas" ON content_ideas
    FOR UPDATE USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can delete their own content ideas" ON content_ideas
    FOR DELETE USING (auth.uid()::text = user_id::text);

-- Add comments
COMMENT ON TABLE content_ideas IS 'Stores generated content ideas (blog posts and software ideas)';
COMMENT ON COLUMN content_ideas.title IS 'The title of the content idea';
COMMENT ON COLUMN content_ideas.description IS 'Detailed description of the content idea';
COMMENT ON COLUMN content_ideas.content_type IS 'Type of content: blog or software';
COMMENT ON COLUMN content_ideas.category IS 'Category of the idea (e.g., seo_optimized, software_tool)';
COMMENT ON COLUMN content_ideas.subtopic IS 'The subtopic this idea relates to';
COMMENT ON COLUMN content_ideas.topic_id IS 'Reference to research topic';
COMMENT ON COLUMN content_ideas.user_id IS 'Reference to user';
COMMENT ON COLUMN content_ideas.keywords IS 'Array of keywords used for this idea';
COMMENT ON COLUMN content_ideas.seo_score IS 'SEO optimization score (0-100)';
COMMENT ON COLUMN content_ideas.difficulty_level IS 'Difficulty level of creating this content';
COMMENT ON COLUMN content_ideas.estimated_read_time IS 'Estimated read time in minutes';
COMMENT ON COLUMN content_ideas.target_audience IS 'Target audience for this content';
COMMENT ON COLUMN content_ideas.content_angle IS 'Unique angle or approach for this content';
COMMENT ON COLUMN content_ideas.monetization_potential IS 'Potential for monetization';
COMMENT ON COLUMN content_ideas.technical_complexity IS 'Technical complexity (for software ideas)';
COMMENT ON COLUMN content_ideas.development_effort IS 'Development effort required (for software ideas)';
COMMENT ON COLUMN content_ideas.market_demand IS 'Market demand for this idea';
