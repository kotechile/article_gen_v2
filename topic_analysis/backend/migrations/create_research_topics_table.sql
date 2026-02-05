-- Create research_topics table
-- This migration creates the research_topics table for storing main research subjects

-- Create research_topics table
CREATE TABLE IF NOT EXISTS research_topics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'completed', 'archived')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    version INTEGER NOT NULL DEFAULT 1 CHECK (version > 0)
);

-- Create unique constraint for research topic titles per user
CREATE UNIQUE INDEX IF NOT EXISTS idx_research_topics_user_title 
ON research_topics(user_id, title);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_research_topics_user_id ON research_topics(user_id);
CREATE INDEX IF NOT EXISTS idx_research_topics_status ON research_topics(status);
CREATE INDEX IF NOT EXISTS idx_research_topics_created_at ON research_topics(created_at);

-- Enable Row Level Security
ALTER TABLE research_topics ENABLE ROW LEVEL SECURITY;

-- Create RLS policies for research_topics
CREATE POLICY "Users can view their own research topics" ON research_topics
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own research topics" ON research_topics
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own research topics" ON research_topics
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own research topics" ON research_topics
    FOR DELETE USING (auth.uid() = user_id);

-- Add comments
COMMENT ON TABLE research_topics IS 'Stores main research subjects with version control';
COMMENT ON COLUMN research_topics.title IS 'Research topic title (unique per user)';
COMMENT ON COLUMN research_topics.description IS 'Detailed description of the research topic';
COMMENT ON COLUMN research_topics.status IS 'Research topic status (active, completed, archived)';
COMMENT ON COLUMN research_topics.version IS 'Version number for optimistic concurrency control';
