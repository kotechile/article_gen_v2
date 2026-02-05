-- Remove unique constraint on research_id to allow multiple content ideas per topic
ALTER TABLE "public"."content_ideas" DROP CONSTRAINT IF EXISTS "content_ideas_research_id_key";
ALTER TABLE "public"."content_ideas" DROP CONSTRAINT IF EXISTS "content_ideas_research_id_unique";
