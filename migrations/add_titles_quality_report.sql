-- Migration to persist machine-readable article quality diagnostics
-- Phase 0 instrumentation support

ALTER TABLE "Titles"
ADD COLUMN IF NOT EXISTS "quality_report" JSONB;

COMMENT ON COLUMN "Titles"."quality_report" IS
'Machine-readable quality diagnostics report generated during article finalization.';
