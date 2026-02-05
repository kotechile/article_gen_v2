-- Add new metric columns to Titles table
DO $$
BEGIN
    -- Add overall_quality_score if it doesn't exist
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'Titles'
        AND column_name = 'overall_quality_score'
    ) THEN
        ALTER TABLE "Titles" ADD COLUMN overall_quality_score INTEGER DEFAULT 0;
    END IF;

    -- Add traffic_potential_score if it doesn't exist
    -- Note: Using FLOAT for potential scores to allow for granular calculations if needed
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'Titles'
        AND column_name = 'traffic_potential_score'
    ) THEN
        ALTER TABLE "Titles" ADD COLUMN traffic_potential_score INTEGER DEFAULT 0;
    END IF;

    -- Add competition_score if it doesn't exist
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'Titles'
        AND column_name = 'competition_score'
    ) THEN
        ALTER TABLE "Titles" ADD COLUMN competition_score FLOAT DEFAULT 0.0;
    END IF;

    -- Reload schema cache to ensure PostgREST sees the new columns
    NOTIFY pgrst, 'reload schema';

END $$;
