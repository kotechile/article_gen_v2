-- Enhance trend_analyses table
-- This migration adds subtopic_name field for linking to specific subtopics

-- Add subtopic_name column
ALTER TABLE trend_analyses 
ADD COLUMN IF NOT EXISTS subtopic_name VARCHAR(255);

-- Create index for subtopic_name
CREATE INDEX IF NOT EXISTS idx_trend_analyses_subtopic_name 
ON trend_analyses(subtopic_name);

-- Add comment
COMMENT ON COLUMN trend_analyses.subtopic_name IS 'Name of the subtopic being analyzed';
