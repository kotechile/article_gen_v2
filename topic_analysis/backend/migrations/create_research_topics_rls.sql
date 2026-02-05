-- Create RLS policies for research_topics table
-- This migration ensures proper row-level security for research topics

-- Enable RLS on research_topics (if not already enabled)
ALTER TABLE research_topics ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist (for idempotency)
DROP POLICY IF EXISTS "Users can view their own research topics" ON research_topics;
DROP POLICY IF EXISTS "Users can insert their own research topics" ON research_topics;
DROP POLICY IF EXISTS "Users can update their own research topics" ON research_topics;
DROP POLICY IF EXISTS "Users can delete their own research topics" ON research_topics;

-- Create RLS policies for research_topics
CREATE POLICY "Users can view their own research topics" ON research_topics
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own research topics" ON research_topics
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own research topics" ON research_topics
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own research topics" ON research_topics
    FOR DELETE USING (auth.uid() = user_id);

-- Add comment
COMMENT ON TABLE research_topics IS 'Research topics with RLS policies for user data isolation';
