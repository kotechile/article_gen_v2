-- Migration to add source_strategy and idea_metadata columns to Titles table
ALTER TABLE "Titles" ADD COLUMN IF NOT EXISTS "source_strategy" TEXT;
ALTER TABLE "Titles" ADD COLUMN IF NOT EXISTS "idea_metadata" JSONB DEFAULT '{}'::jsonb;

COMMENT ON COLUMN "Titles"."source_strategy" IS
'The primary source configuration strategy for article generation (e.g., rag_only, live_web_only, rag_plus_live_web, none).';

COMMENT ON COLUMN "Titles"."idea_metadata" IS
'Flexible metadata store (JSONB) for in-flight generation steps, including competitor analysis.';

