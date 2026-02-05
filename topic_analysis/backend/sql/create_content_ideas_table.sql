-- Create content_ideas table for storing content ideas per research topic
CREATE TABLE IF NOT EXISTS content_ideas (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    research_id VARCHAR(255) NOT NULL,
    ideas JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(research_id)
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_content_ideas_research_id ON content_ideas(research_id);

-- Create updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_content_ideas_updated_at 
    BEFORE UPDATE ON content_ideas 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

