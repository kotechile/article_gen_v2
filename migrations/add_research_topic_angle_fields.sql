-- Add structured intent/angle metadata to research_topics.
-- Safe to run multiple times.

ALTER TABLE IF EXISTS research_topics
    ADD COLUMN IF NOT EXISTS intent_bucket TEXT,
    ADD COLUMN IF NOT EXISTS decision_focus TEXT,
    ADD COLUMN IF NOT EXISTS angle_question TEXT,
    ADD COLUMN IF NOT EXISTS value_layer_tags JSONB DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS target_audience TEXT,
    ADD COLUMN IF NOT EXISTS evidence_sources JSONB DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS related_terms JSONB DEFAULT '[]'::jsonb;

