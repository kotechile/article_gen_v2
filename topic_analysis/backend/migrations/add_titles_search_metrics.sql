-- Add search metrics to Titles table
-- This migration adds total_search_volume and avg_keyword_difficulty columns

DO $$
BEGIN
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
        ALTER TABLE "Titles" ADD COLUMN avg_keyword_difficulty INTEGER DEFAULT 0;
    END IF;
END $$;
