-- Add missing metric columns to content_ideas table
-- This allows the generator to validly save the calculated metrics

DO $$
BEGIN
    -- Add keyword_metrics (JSONB)
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'content_ideas'
        AND column_name = 'keyword_metrics'
    ) THEN
        ALTER TABLE "content_ideas" ADD COLUMN keyword_metrics JSONB DEFAULT '{}'::jsonb;
    END IF;

    -- Add seo_score
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'content_ideas'
        AND column_name = 'seo_score'
    ) THEN
        ALTER TABLE "content_ideas" ADD COLUMN seo_score INTEGER DEFAULT 0;
    END IF;

    -- Add additional useful metadata columns if missing
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'content_ideas'
        AND column_name = 'difficulty_level'
    ) THEN
        ALTER TABLE "content_ideas" ADD COLUMN difficulty_level TEXT;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'content_ideas'
        AND column_name = 'monetization_potential'
    ) THEN
        ALTER TABLE "content_ideas" ADD COLUMN monetization_potential TEXT;
    END IF;

    -- Reload schema cache
    NOTIFY pgrst, 'reload schema';

END $$;
