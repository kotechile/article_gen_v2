-- Keyword handoff fields between Research -> Content Generation.
-- Safe to run multiple times.

ALTER TABLE "Titles"
ADD COLUMN IF NOT EXISTS "keyword_candidates_json" JSONB;

ALTER TABLE "Titles"
ADD COLUMN IF NOT EXISTS "keyword_clusters_json" JSONB;

ALTER TABLE "Titles"
ADD COLUMN IF NOT EXISTS "keyword_research_status" TEXT;

ALTER TABLE "Titles"
ADD COLUMN IF NOT EXISTS "keyword_research_source" TEXT;

ALTER TABLE "Titles"
ADD COLUMN IF NOT EXISTS "keyword_research_confidence" DOUBLE PRECISION;

ALTER TABLE "Titles"
ADD COLUMN IF NOT EXISTS "keyword_research_generated_at" TIMESTAMPTZ;

ALTER TABLE "Titles"
ADD COLUMN IF NOT EXISTS "primary_keyword" TEXT;

ALTER TABLE "Titles"
ADD COLUMN IF NOT EXISTS "secondary_keywords_json" JSONB;

ALTER TABLE "Titles"
ADD COLUMN IF NOT EXISTS "supporting_entities_json" JSONB;

ALTER TABLE "Titles"
ADD COLUMN IF NOT EXISTS "priority_questions_json" JSONB;

ALTER TABLE "Titles"
ADD COLUMN IF NOT EXISTS "selected_keyword_search_volume" INTEGER;

ALTER TABLE "Titles"
ADD COLUMN IF NOT EXISTS "selected_keyword_difficulty" DOUBLE PRECISION;

ALTER TABLE "Titles"
ADD COLUMN IF NOT EXISTS "selected_keyword_intent" TEXT;

ALTER TABLE "Titles"
ADD COLUMN IF NOT EXISTS "keyword_selection_reason" TEXT;

ALTER TABLE "Titles"
ADD COLUMN IF NOT EXISTS "keyword_strategy_version" TEXT;

ALTER TABLE "Titles"
ADD COLUMN IF NOT EXISTS "keyword_selection_source" TEXT;

COMMENT ON COLUMN "Titles"."keyword_candidates_json" IS
'Research-stage keyword candidates discovered from DataForSEO and/or fallback logic.';

COMMENT ON COLUMN "Titles"."keyword_clusters_json" IS
'Research-stage grouped keyword clusters and metadata.';

COMMENT ON COLUMN "Titles"."keyword_research_status" IS
'Keyword dossier status, e.g. ready, partial, fallback.';

COMMENT ON COLUMN "Titles"."keyword_research_source" IS
'Primary source used for keyword dossier: dataforseo, hybrid, or llm_fallback.';

COMMENT ON COLUMN "Titles"."primary_keyword" IS
'Final selected primary keyword for this specific article.';

COMMENT ON COLUMN "Titles"."secondary_keywords_json" IS
'Final selected secondary keywords for this specific article.';

COMMENT ON COLUMN "Titles"."keyword_selection_source" IS
'Source used for final keyword selection: research_dossier_reused, re-ranked_with_dataforseo, llm_fallback.';
