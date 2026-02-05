-- Add JSONB columns for rich trend and monetization data
ALTER TABLE subtopics ADD COLUMN IF NOT EXISTS trend_analysis JSONB;
ALTER TABLE subtopics ADD COLUMN IF NOT EXISTS monetization_data JSONB;
