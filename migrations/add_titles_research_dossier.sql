-- Migration to persist structured deep research artifacts for article generation

ALTER TABLE "Titles"
ADD COLUMN IF NOT EXISTS "research_dossier" JSONB;

ALTER TABLE "Titles"
ADD COLUMN IF NOT EXISTS "dossier_status" TEXT;

ALTER TABLE "Titles"
ADD COLUMN IF NOT EXISTS "dossier_last_updated_at" TIMESTAMP WITH TIME ZONE;

ALTER TABLE "Titles"
ADD COLUMN IF NOT EXISTS "dossier_quality_score" INTEGER DEFAULT 0;

COMMENT ON COLUMN "Titles"."research_dossier" IS
'Structured deep research dossier used by the article generation pipeline.';

COMMENT ON COLUMN "Titles"."dossier_status" IS
'Lifecycle state for dossier readiness (e.g., pending, ready, failed).';

COMMENT ON COLUMN "Titles"."dossier_last_updated_at" IS
'Last time the research dossier was refreshed.';

COMMENT ON COLUMN "Titles"."dossier_quality_score" IS
'Quick dossier quality signal (0-100) based on source coverage and completeness.';
