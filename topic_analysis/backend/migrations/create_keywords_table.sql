-- Create keywords table for storing generated keywords
CREATE TABLE IF NOT EXISTS keywords (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    keyword TEXT NOT NULL,
    search_volume INTEGER,
    difficulty DECIMAL(5,2),
    cpc DECIMAL(10,4),
    topic_id UUID NOT NULL,
    user_id UUID NOT NULL,
    source VARCHAR(20) NOT NULL CHECK (source IN ('llm', 'ahrefs', 'manual')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_keywords_topic_id ON keywords(topic_id);
CREATE INDEX IF NOT EXISTS idx_keywords_user_id ON keywords(user_id);
CREATE INDEX IF NOT EXISTS idx_keywords_source ON keywords(source);
CREATE INDEX IF NOT EXISTS idx_keywords_created_at ON keywords(created_at);
CREATE INDEX IF NOT EXISTS idx_keywords_user_topic ON keywords(user_id, topic_id);

-- Enable RLS
ALTER TABLE keywords ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
CREATE POLICY "Users can view their own keywords" ON keywords
    FOR SELECT USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can insert their own keywords" ON keywords
    FOR INSERT WITH CHECK (auth.uid()::text = user_id::text);

CREATE POLICY "Users can update their own keywords" ON keywords
    FOR UPDATE USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can delete their own keywords" ON keywords
    FOR DELETE USING (auth.uid()::text = user_id::text);

-- Add comments
COMMENT ON TABLE keywords IS 'Stores generated keywords for research topics';
COMMENT ON COLUMN keywords.keyword IS 'The keyword phrase';
COMMENT ON COLUMN keywords.search_volume IS 'Monthly search volume';
COMMENT ON COLUMN keywords.difficulty IS 'Keyword difficulty score (0-100)';
COMMENT ON COLUMN keywords.cpc IS 'Cost per click';
COMMENT ON COLUMN keywords.topic_id IS 'Reference to research topic';
COMMENT ON COLUMN keywords.user_id IS 'Reference to user';
COMMENT ON COLUMN keywords.source IS 'Source of the keyword (llm, ahrefs, manual)';
