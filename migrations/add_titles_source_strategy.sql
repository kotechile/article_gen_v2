-- Migration to add source_strategy column to Titles table
ALTER TABLE "Titles" ADD COLUMN IF NOT EXISTS "source_strategy" TEXT;

COMMENT ON COLUMN "Titles"."source_strategy" IS
'The primary source configuration strategy for article generation (e.g., rag_only, live_web_only, rag_plus_live_web, none).';
