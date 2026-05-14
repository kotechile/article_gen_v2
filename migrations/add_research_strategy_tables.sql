CREATE TABLE IF NOT EXISTS research_strategy_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    project_id UUID NOT NULL,
    topic_id UUID NOT NULL REFERENCES research_user_jobs(id) ON DELETE CASCADE,
    primary_category_id UUID,
    secondary_category_id UUID,
    status TEXT NOT NULL DEFAULT 'draft',
    current_stage TEXT NOT NULL DEFAULT 'topic_saved',
    selected_bet_id UUID,
    selected_cluster_id UUID,
    winning_route TEXT,
    confidence_score NUMERIC(5,4),
    limits_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    run_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    validated_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_strategy_runs_status_check
        CHECK (status IN ('draft', 'running', 'screened', 'clustered', 'completed', 'rejected', 'archived'))
);

CREATE INDEX IF NOT EXISTS idx_research_strategy_runs_topic
    ON research_strategy_runs (topic_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_strategy_runs_project
    ON research_strategy_runs (user_id, project_id, created_at DESC);

CREATE TABLE IF NOT EXISTS research_topic_bets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    project_id UUID NOT NULL,
    run_id UUID NOT NULL REFERENCES research_strategy_runs(id) ON DELETE CASCADE,
    topic_id UUID NOT NULL REFERENCES research_user_jobs(id) ON DELETE CASCADE,
    bet_text TEXT NOT NULL,
    searcher_problem TEXT,
    article_format TEXT,
    commercial_angle TEXT,
    buyer_or_seller_intent TEXT,
    route_hint TEXT,
    trend_score NUMERIC(5,4),
    serp_articleability_score NUMERIC(5,4),
    serp_weakness_score NUMERIC(5,4),
    intent_fit_score NUMERIC(5,4),
    article_fit_score NUMERIC(5,4),
    status TEXT NOT NULL DEFAULT 'draft',
    reason_codes JSONB NOT NULL DEFAULT '[]'::jsonb,
    bet_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_topic_bets_status_check
        CHECK (status IN ('draft', 'screening', 'survived', 'killed', 'selected', 'archived'))
);

CREATE INDEX IF NOT EXISTS idx_research_topic_bets_run
    ON research_topic_bets (run_id, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_research_topic_bets_topic
    ON research_topic_bets (topic_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_topic_bets_status
    ON research_topic_bets (status);

CREATE TABLE IF NOT EXISTS research_probe_queries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    project_id UUID NOT NULL,
    run_id UUID NOT NULL REFERENCES research_strategy_runs(id) ON DELETE CASCADE,
    bet_id UUID NOT NULL REFERENCES research_topic_bets(id) ON DELETE CASCADE,
    query_text TEXT NOT NULL,
    query_role TEXT NOT NULL DEFAULT 'primary_probe',
    trend_search_id UUID REFERENCES research_dataforseo_searches(id) ON DELETE SET NULL,
    serp_search_id UUID REFERENCES research_dataforseo_searches(id) ON DELETE SET NULL,
    articleability_passed BOOLEAN,
    serp_classification TEXT,
    probe_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_probe_queries_role_check
        CHECK (query_role IN ('primary_probe', 'secondary_probe')),
    CONSTRAINT research_probe_queries_classification_check
        CHECK (
            serp_classification IS NULL
            OR serp_classification IN ('article_friendly', 'tool_dominant', 'service_dominant', 'ecommerce_dominant', 'mixed', 'editorial')
        )
);

CREATE INDEX IF NOT EXISTS idx_research_probe_queries_run
    ON research_probe_queries (run_id, bet_id, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_research_probe_queries_bet
    ON research_probe_queries (bet_id, created_at ASC);

CREATE TABLE IF NOT EXISTS research_competitor_pages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    project_id UUID NOT NULL,
    run_id UUID NOT NULL REFERENCES research_strategy_runs(id) ON DELETE CASCADE,
    bet_id UUID NOT NULL REFERENCES research_topic_bets(id) ON DELETE CASCADE,
    probe_query_id UUID NOT NULL REFERENCES research_probe_queries(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    title TEXT,
    domain TEXT,
    page_type TEXT,
    rank_group INTEGER,
    mined_search_id UUID REFERENCES research_dataforseo_searches(id) ON DELETE SET NULL,
    selected_for_mining BOOLEAN NOT NULL DEFAULT FALSE,
    page_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_research_competitor_pages_run
    ON research_competitor_pages (run_id, bet_id, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_research_competitor_pages_probe
    ON research_competitor_pages (probe_query_id, created_at ASC);

CREATE TABLE IF NOT EXISTS research_keyword_clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    project_id UUID NOT NULL,
    run_id UUID NOT NULL REFERENCES research_strategy_runs(id) ON DELETE CASCADE,
    bet_id UUID NOT NULL REFERENCES research_topic_bets(id) ON DELETE CASCADE,
    cluster_name TEXT NOT NULL,
    primary_keyword_candidate TEXT,
    secondary_keywords_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    supporting_competitor_urls_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    cluster_type TEXT,
    competitor_support_score NUMERIC(5,4),
    kd_median_score NUMERIC(5,4),
    commercial_value_score NUMERIC(5,4),
    trend_score NUMERIC(5,4),
    articleability_score NUMERIC(5,4),
    serp_weakness_score NUMERIC(5,4),
    article_fit_score NUMERIC(5,4),
    opportunity_score NUMERIC(5,4),
    status TEXT NOT NULL DEFAULT 'draft',
    cluster_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_keyword_clusters_status_check
        CHECK (status IN ('draft', 'survived', 'selected', 'rejected', 'archived'))
);

CREATE INDEX IF NOT EXISTS idx_research_keyword_clusters_run
    ON research_keyword_clusters (run_id, bet_id, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_research_keyword_clusters_status
    ON research_keyword_clusters (status);

ALTER TABLE research_dataforseo_searches
    DROP CONSTRAINT IF EXISTS research_dataforseo_searches_type_check;

ALTER TABLE research_dataforseo_searches
    ADD CONSTRAINT research_dataforseo_searches_type_check
    CHECK (
        search_type IN (
            'related_keywords',
            'keyword_overview',
            'serp',
            'google_trends',
            'serp_probe',
            'ranked_keywords',
            'relevant_pages'
        )
    );

COMMENT ON TABLE research_strategy_runs IS
    'Top-level orchestration records for strategic competitive SERP mining runs.';

COMMENT ON TABLE research_topic_bets IS
    'Article-angle, software-angle, or editorial-angle bets generated from a saved topic.';

COMMENT ON TABLE research_probe_queries IS
    'Probe queries used for trend screening and SERP articleability checks.';

COMMENT ON TABLE research_competitor_pages IS
    'Top competitor URLs harvested from good SERPs and selected for ranked keyword mining.';

COMMENT ON TABLE research_keyword_clusters IS
    'Intent-aware keyword clusters harvested from competitor URLs and scored at the cluster level.';
