-- Topic-level keyword research tables for the article ideas revamp.
-- These tables are intentionally separate from the legacy subtopics-first flow.
-- Safety rule: research regeneration may replace rows in these tables, but must never delete Titles rows.

CREATE TABLE IF NOT EXISTS topic_keyword_research_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    topic_id UUID NOT NULL,
    user_id UUID NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    seed_keywords_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    filters_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    score_config_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    raw_data_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_topic_keyword_research_runs_topic_id
    ON topic_keyword_research_runs (topic_id);

CREATE INDEX IF NOT EXISTS idx_topic_keyword_research_runs_user_id
    ON topic_keyword_research_runs (user_id);

CREATE INDEX IF NOT EXISTS idx_topic_keyword_research_runs_created_at
    ON topic_keyword_research_runs (created_at DESC);

CREATE TABLE IF NOT EXISTS topic_keyword_candidates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    research_run_id UUID NOT NULL,
    topic_id UUID NOT NULL,
    user_id UUID NOT NULL,
    keyword TEXT NOT NULL,
    canonical_keyword TEXT NOT NULL,
    variant_keywords_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    source_endpoints_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    search_volume INTEGER,
    cpc NUMERIC(10,2),
    competition TEXT,
    competition_index INTEGER,
    keyword_difficulty NUMERIC(6,2),
    trend_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    intent_label TEXT,
    topical_fit_score NUMERIC(6,2),
    opportunity_score NUMERIC(8,2),
    is_filtered_out BOOLEAN NOT NULL DEFAULT FALSE,
    filter_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_topic_keyword_candidates_run_canonical
    ON topic_keyword_candidates (research_run_id, canonical_keyword);

CREATE INDEX IF NOT EXISTS idx_topic_keyword_candidates_topic_id
    ON topic_keyword_candidates (topic_id);

CREATE INDEX IF NOT EXISTS idx_topic_keyword_candidates_user_id
    ON topic_keyword_candidates (user_id);

CREATE INDEX IF NOT EXISTS idx_topic_keyword_candidates_opportunity
    ON topic_keyword_candidates (research_run_id, opportunity_score DESC);

CREATE TABLE IF NOT EXISTS topic_keyword_clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    research_run_id UUID NOT NULL,
    topic_id UUID NOT NULL,
    user_id UUID NOT NULL,
    cluster_name TEXT NOT NULL,
    primary_keyword TEXT,
    secondary_keywords_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    keyword_candidates_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    intent_label TEXT,
    serp_validation_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    opportunity_score NUMERIC(8,2),
    software_opportunity_score NUMERIC(8,2),
    article_angle TEXT,
    rationale TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_topic_keyword_clusters_run_id
    ON topic_keyword_clusters (research_run_id);

CREATE INDEX IF NOT EXISTS idx_topic_keyword_clusters_topic_id
    ON topic_keyword_clusters (topic_id);

CREATE INDEX IF NOT EXISTS idx_topic_keyword_clusters_user_id
    ON topic_keyword_clusters (user_id);

CREATE INDEX IF NOT EXISTS idx_topic_keyword_clusters_opportunity
    ON topic_keyword_clusters (research_run_id, opportunity_score DESC);
