-- Add columns to Titles table for full Idea Burst publishing support
-- Columns: topic_id, source_idea_id, subtopic, total_search_volume, avg_keyword_difficulty, content_type

DO $$
BEGIN
    -- Add topic_id if it doesn't exist
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'Titles'
        AND column_name = 'topic_id'
    ) THEN
        ALTER TABLE "Titles" ADD COLUMN topic_id UUID;
    END IF;

    -- Add source_idea_id if it doesn't exist
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'Titles'
        AND column_name = 'source_idea_id'
    ) THEN
        ALTER TABLE "Titles" ADD COLUMN source_idea_id UUID;
    END IF;

    -- Add subtopic if it doesn't exist
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'Titles'
        AND column_name = 'subtopic'
    ) THEN
        ALTER TABLE "Titles" ADD COLUMN subtopic TEXT;
    END IF;
    
    -- Add content_type if it doesn't exist
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'Titles'
        AND column_name = 'content_type'
    ) THEN
        ALTER TABLE "Titles" ADD COLUMN content_type TEXT;
    END IF;

    -- Add total_search_volume if it doesn't exist
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'Titles'
        AND column_name = 'total_search_volume'
    ) THEN
        ALTER TABLE "Titles" ADD COLUMN total_search_volume INTEGER DEFAULT 0;
    END IF;

    -- Add avg_keyword_difficulty if it doesn't exist
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'Titles'
        AND column_name = 'avg_keyword_difficulty'
    ) THEN
        ALTER TABLE "Titles" ADD COLUMN avg_keyword_difficulty FLOAT DEFAULT 0.0;
    END IF;

    -- Add original_created_at if it doesn't exist (to track original idea creation)
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'Titles'
        AND column_name = 'original_created_at'
    ) THEN
        ALTER TABLE "Titles" ADD COLUMN original_created_at TIMESTAMP WITH TIME ZONE;
    END IF;

END $$;
