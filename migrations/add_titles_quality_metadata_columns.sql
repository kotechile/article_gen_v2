-- Add missing quality metadata columns used by generation finalization.
-- Safe to run multiple times.

ALTER TABLE "Titles"
ADD COLUMN IF NOT EXISTS "quality_report" JSONB;

ALTER TABLE "Titles"
ADD COLUMN IF NOT EXISTS "confidence_map" JSONB;

ALTER TABLE "Titles"
ADD COLUMN IF NOT EXISTS "quality_gate" JSONB;

COMMENT ON COLUMN "Titles"."quality_report" IS
'Structured quality scoring payload (overall/humanization/grounding/geo + diagnostics).';

COMMENT ON COLUMN "Titles"."confidence_map" IS
'Claim/paragraph-level grounding confidence map for auditability.';

COMMENT ON COLUMN "Titles"."quality_gate" IS
'Final gate decision and reasons used to set generation_status.';
