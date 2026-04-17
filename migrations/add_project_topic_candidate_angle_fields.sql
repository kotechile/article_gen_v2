-- Add structured angle metadata to command center topic candidates
ALTER TABLE IF EXISTS project_topic_candidates
    ADD COLUMN IF NOT EXISTS rationale TEXT,
    ADD COLUMN IF NOT EXISTS intent_bucket TEXT,
    ADD COLUMN IF NOT EXISTS decision_focus TEXT,
    ADD COLUMN IF NOT EXISTS angle_question TEXT,
    ADD COLUMN IF NOT EXISTS value_layer_tags JSONB DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS related_terms JSONB DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS source_signals JSONB DEFAULT '[]'::jsonb;
