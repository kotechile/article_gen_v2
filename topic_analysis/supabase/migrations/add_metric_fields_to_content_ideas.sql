-- Add missing metric fields to content_ideas table
-- These fields are needed to display metrics in the idea burst page

ALTER TABLE "public"."content_ideas" 
ADD COLUMN IF NOT EXISTS "total_search_volume" integer DEFAULT 0,
ADD COLUMN IF NOT EXISTS "average_difficulty" integer DEFAULT 50,
ADD COLUMN IF NOT EXISTS "average_cpc" numeric(10,2) DEFAULT 0;

COMMENT ON COLUMN "public"."content_ideas"."total_search_volume" IS 'Total search volume for all keywords in this idea';
COMMENT ON COLUMN "public"."content_ideas"."average_difficulty" IS 'Average difficulty score (0-100) for keywords in this idea';
COMMENT ON COLUMN "public"."content_ideas"."average_cpc" IS 'Average cost per click for keywords in this idea';
