-- Subtopic evidence tables for subtopics-first pipeline
-- Run this migration on Supabase/Postgres before enabling strict persistence to these tables.

CREATE TABLE IF NOT EXISTS subtopic_keyword_candidates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subtopic_id UUID NOT NULL,
    research_topic_id UUID NOT NULL,
    user_id UUID NOT NULL,
    keyword TEXT NOT NULL,
    variant_type TEXT,
    search_volume INTEGER DEFAULT 0,
    cpc NUMERIC(10,2) DEFAULT 0,
    keyword_difficulty INTEGER DEFAULT 0,
    competition TEXT,
    serp_intent TEXT,
    is_selected_primary BOOLEAN DEFAULT FALSE,
    selection_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_subtopic_keyword_candidates_subtopic_id
    ON subtopic_keyword_candidates (subtopic_id);

CREATE INDEX IF NOT EXISTS idx_subtopic_keyword_candidates_topic_id
    ON subtopic_keyword_candidates (research_topic_id);

CREATE INDEX IF NOT EXISTS idx_subtopic_keyword_candidates_user_id
    ON subtopic_keyword_candidates (user_id);

CREATE INDEX IF NOT EXISTS idx_subtopic_keyword_candidates_keyword
    ON subtopic_keyword_candidates (keyword);

CREATE TABLE IF NOT EXISTS subtopic_affiliate_evidence (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subtopic_id UUID NOT NULL,
    research_topic_id UUID NOT NULL,
    user_id UUID NOT NULL,
    source TEXT DEFAULT 'affiliate_research_service',
    confidence NUMERIC(5,4) DEFAULT 0,
    programs JSONB NOT NULL DEFAULT '[]'::jsonb,
    rationale TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_subtopic_affiliate_evidence_subtopic_id
    ON subtopic_affiliate_evidence (subtopic_id);

CREATE INDEX IF NOT EXISTS idx_subtopic_affiliate_evidence_topic_id
    ON subtopic_affiliate_evidence (research_topic_id);

CREATE INDEX IF NOT EXISTS idx_subtopic_affiliate_evidence_user_id
    ON subtopic_affiliate_evidence (user_id);

