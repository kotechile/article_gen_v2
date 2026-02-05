-- Add missing score columns to content_ideas table
-- These already exist in Titles, so we need them here to persist the source data

DO $$
BEGIN
    ALTER TABLE "content_ideas" ADD COLUMN IF NOT EXISTS viral_potential_score INTEGER DEFAULT 0;
    ALTER TABLE "content_ideas" ADD COLUMN IF NOT EXISTS audience_alignment_score INTEGER DEFAULT 0;
    ALTER TABLE "content_ideas" ADD COLUMN IF NOT EXISTS content_feasibility_score INTEGER DEFAULT 0;
    ALTER TABLE "content_ideas" ADD COLUMN IF NOT EXISTS business_impact_score INTEGER DEFAULT 0;
    
    NOTIFY pgrst, 'reload schema';
END $$;
