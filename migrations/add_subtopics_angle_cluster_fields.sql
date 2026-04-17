-- Add angle/cluster metadata to persisted subtopics for richer downstream idea generation
ALTER TABLE IF EXISTS subtopics
    ADD COLUMN IF NOT EXISTS intent_bucket TEXT,
    ADD COLUMN IF NOT EXISTS decision_focus TEXT,
    ADD COLUMN IF NOT EXISTS angle_question TEXT,
    ADD COLUMN IF NOT EXISTS value_layer_tags JSONB DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS cluster_type TEXT,
    ADD COLUMN IF NOT EXISTS primary_user_outcome TEXT,
    ADD COLUMN IF NOT EXISTS serp_intent_match TEXT,
    ADD COLUMN IF NOT EXISTS tool_potential_score INTEGER DEFAULT 0;
