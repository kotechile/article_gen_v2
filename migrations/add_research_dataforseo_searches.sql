CREATE TABLE IF NOT EXISTS research_dataforseo_searches (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    project_id UUID NOT NULL,
    user_job_id UUID REFERENCES research_user_jobs(id) ON DELETE SET NULL,
    primary_category_id UUID,
    secondary_category_id UUID,
    search_type TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    query_text TEXT NOT NULL,
    normalized_query_text TEXT,
    request_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    response_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    result_summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    searched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_dataforseo_searches_type_check
        CHECK (search_type IN ('related_keywords', 'keyword_overview', 'serp'))
);

CREATE INDEX IF NOT EXISTS idx_research_dataforseo_searches_user_project
    ON research_dataforseo_searches (user_id, project_id, searched_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_dataforseo_searches_job
    ON research_dataforseo_searches (user_job_id, searched_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_dataforseo_searches_type
    ON research_dataforseo_searches (search_type, searched_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_dataforseo_searches_normalized_query
    ON research_dataforseo_searches (normalized_query_text);

COMMENT ON TABLE research_dataforseo_searches IS
    'Persistent manual DataForSEO lookups captured for research rebuild workflows and future keyword mining.';
