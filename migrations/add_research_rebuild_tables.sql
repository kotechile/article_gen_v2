-- Research rebuild tables
-- Target-state schema for job-first, validation-first research workflows.
-- These tables are designed to coexist with research_topics, content_ideas, and Titles.

CREATE TABLE IF NOT EXISTS research_user_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    project_id UUID NOT NULL,
    primary_category_id UUID,
    secondary_category_id UUID,
    job_text TEXT NOT NULL,
    job_type_hint TEXT,
    job_source TEXT NOT NULL DEFAULT 'llm_generation',
    status TEXT NOT NULL DEFAULT 'draft',
    website_context_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    generation_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    rejection_reason_tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    rejection_reason_free_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_user_jobs_status_check
        CHECK (status IN ('draft', 'approved', 'rejected', 'archived'))
);

CREATE INDEX IF NOT EXISTS idx_research_user_jobs_user_project
    ON research_user_jobs (user_id, project_id);

CREATE INDEX IF NOT EXISTS idx_research_user_jobs_project_categories
    ON research_user_jobs (project_id, primary_category_id, secondary_category_id);

CREATE INDEX IF NOT EXISTS idx_research_user_jobs_status
    ON research_user_jobs (status);


CREATE TABLE IF NOT EXISTS research_opportunity_candidates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    project_id UUID NOT NULL,
    user_job_id UUID NOT NULL REFERENCES research_user_jobs(id) ON DELETE CASCADE,
    candidate_type TEXT NOT NULL,
    candidate_text TEXT NOT NULL,
    normalized_candidate_text TEXT,
    status TEXT NOT NULL DEFAULT 'draft',
    candidate_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_keywords_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    rejection_reason_tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    rejection_reason_free_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_opportunity_candidates_type_check
        CHECK (candidate_type IN ('seo_article', 'software', 'editorial')),
    CONSTRAINT research_opportunity_candidates_status_check
        CHECK (status IN ('draft', 'validated', 'rejected', 'archived'))
);

CREATE INDEX IF NOT EXISTS idx_research_opportunity_candidates_user_job
    ON research_opportunity_candidates (user_job_id);

CREATE INDEX IF NOT EXISTS idx_research_opportunity_candidates_user_project
    ON research_opportunity_candidates (user_id, project_id);

CREATE INDEX IF NOT EXISTS idx_research_opportunity_candidates_type_status
    ON research_opportunity_candidates (candidate_type, status);

CREATE INDEX IF NOT EXISTS idx_research_opportunity_candidates_normalized_text
    ON research_opportunity_candidates (normalized_candidate_text);


CREATE TABLE IF NOT EXISTS research_validation_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    project_id UUID NOT NULL,
    candidate_id UUID NOT NULL REFERENCES research_opportunity_candidates(id) ON DELETE CASCADE,
    validation_version TEXT NOT NULL,
    validated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    freshness_state TEXT NOT NULL DEFAULT 'fresh',
    eligibility_passed BOOLEAN NOT NULL DEFAULT FALSE,
    intent_match_score NUMERIC(5,4),
    serp_weakness_score NUMERIC(5,4),
    serp_gap_score NUMERIC(5,4),
    software_pattern_score NUMERIC(5,4),
    feasibility_score NUMERIC(5,4),
    monetization_fit_score NUMERIC(5,4),
    volume_score NUMERIC(5,4),
    kd_ease_score NUMERIC(5,4),
    niche_drift_score NUMERIC(5,4),
    achievability_score NUMERIC(5,4),
    validation_reason_codes JSONB NOT NULL DEFAULT '[]'::jsonb,
    validation_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_validation_runs_freshness_state_check
        CHECK (freshness_state IN ('fresh', 'stale', 'expired'))
);

CREATE INDEX IF NOT EXISTS idx_research_validation_runs_candidate_validated_at
    ON research_validation_runs (candidate_id, validated_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_validation_runs_user_project
    ON research_validation_runs (user_id, project_id);

CREATE INDEX IF NOT EXISTS idx_research_validation_runs_freshness
    ON research_validation_runs (freshness_state);

CREATE INDEX IF NOT EXISTS idx_research_validation_runs_eligibility
    ON research_validation_runs (eligibility_passed);


CREATE TABLE IF NOT EXISTS research_serp_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    project_id UUID NOT NULL,
    candidate_id UUID NOT NULL REFERENCES research_opportunity_candidates(id) ON DELETE CASCADE,
    validation_run_id UUID NOT NULL REFERENCES research_validation_runs(id) ON DELETE CASCADE,
    query_text TEXT NOT NULL,
    snapshot_source TEXT NOT NULL,
    validated_at TIMESTAMPTZ NOT NULL,
    top_results_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    serp_summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_research_serp_snapshots_candidate_validated_at
    ON research_serp_snapshots (candidate_id, validated_at DESC);

CREATE INDEX IF NOT EXISTS idx_research_serp_snapshots_validation_run
    ON research_serp_snapshots (validation_run_id);


CREATE TABLE IF NOT EXISTS research_routing_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    project_id UUID NOT NULL,
    candidate_id UUID NOT NULL REFERENCES research_opportunity_candidates(id) ON DELETE CASCADE,
    validation_run_id UUID NOT NULL REFERENCES research_validation_runs(id) ON DELETE CASCADE,
    route TEXT NOT NULL,
    route_reason_codes JSONB NOT NULL DEFAULT '[]'::jsonb,
    route_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_routing_decisions_route_check
        CHECK (route IN (
            'article_ready',
            'software_ready',
            'article_plus_software',
            'editorial_only',
            'software_backlog_low_feasibility',
            'needs_more_keyword_validation',
            'rejected_low_achievability'
        ))
);

CREATE INDEX IF NOT EXISTS idx_research_routing_decisions_candidate
    ON research_routing_decisions (candidate_id);

CREATE INDEX IF NOT EXISTS idx_research_routing_decisions_validation_run
    ON research_routing_decisions (validation_run_id);

CREATE INDEX IF NOT EXISTS idx_research_routing_decisions_route
    ON research_routing_decisions (route);


CREATE TABLE IF NOT EXISTS research_keyword_packs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    project_id UUID NOT NULL,
    candidate_id UUID NOT NULL REFERENCES research_opportunity_candidates(id) ON DELETE CASCADE,
    validation_run_id UUID NOT NULL REFERENCES research_validation_runs(id) ON DELETE CASCADE,
    primary_keyword TEXT,
    secondary_keywords_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    keyword_metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    keyword_pack_status TEXT NOT NULL DEFAULT 'draft',
    keyword_pack_reason_codes JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_keyword_packs_status_check
        CHECK (keyword_pack_status IN ('draft', 'ready', 'cluster_too_thin', 'needs_more_keyword_validation'))
);

CREATE INDEX IF NOT EXISTS idx_research_keyword_packs_candidate
    ON research_keyword_packs (candidate_id);

CREATE INDEX IF NOT EXISTS idx_research_keyword_packs_validation_run
    ON research_keyword_packs (validation_run_id);

CREATE INDEX IF NOT EXISTS idx_research_keyword_packs_status
    ON research_keyword_packs (keyword_pack_status);


CREATE TABLE IF NOT EXISTS research_internal_link_candidates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    project_id UUID NOT NULL,
    candidate_id UUID NOT NULL REFERENCES research_opportunity_candidates(id) ON DELETE CASCADE,
    validation_run_id UUID REFERENCES research_validation_runs(id) ON DELETE SET NULL,
    wordpress_imported_post_id UUID,
    link_role TEXT NOT NULL,
    match_score NUMERIC(5,4),
    match_reason_codes JSONB NOT NULL DEFAULT '[]'::jsonb,
    match_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_internal_link_candidates_role_check
        CHECK (link_role IN ('parent_candidate', 'child_candidate', 'sibling_candidate', 'hub_candidate'))
);

CREATE INDEX IF NOT EXISTS idx_research_internal_link_candidates_candidate
    ON research_internal_link_candidates (candidate_id);

CREATE INDEX IF NOT EXISTS idx_research_internal_link_candidates_role
    ON research_internal_link_candidates (link_role);


CREATE TABLE IF NOT EXISTS research_generated_outcomes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    project_id UUID NOT NULL,
    candidate_id UUID NOT NULL REFERENCES research_opportunity_candidates(id) ON DELETE CASCADE,
    validation_run_id UUID REFERENCES research_validation_runs(id) ON DELETE SET NULL,
    routing_decision_id UUID REFERENCES research_routing_decisions(id) ON DELETE SET NULL,
    content_idea_id UUID,
    outcome_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    outcome_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT research_generated_outcomes_type_check
        CHECK (outcome_type IN ('article', 'software', 'editorial')),
    CONSTRAINT research_generated_outcomes_status_check
        CHECK (status IN ('draft', 'generated', 'persisted', 'published', 'archived'))
);

CREATE INDEX IF NOT EXISTS idx_research_generated_outcomes_candidate
    ON research_generated_outcomes (candidate_id);

CREATE INDEX IF NOT EXISTS idx_research_generated_outcomes_content_idea
    ON research_generated_outcomes (content_idea_id);

CREATE INDEX IF NOT EXISTS idx_research_generated_outcomes_type_status
    ON research_generated_outcomes (outcome_type, status);


COMMENT ON TABLE research_user_jobs IS
    'Source-of-truth user jobs generated from website and category context for the research rebuild.';

COMMENT ON TABLE research_opportunity_candidates IS
    'Validated or rejected opportunity candidates derived from user jobs.';

COMMENT ON TABLE research_validation_runs IS
    'Persisted scoring, freshness, and evidence results for research opportunity validation.';

COMMENT ON TABLE research_serp_snapshots IS
    'SERP evidence snapshots captured during candidate validation.';

COMMENT ON TABLE research_routing_decisions IS
    'Final routing decision for an opportunity candidate after validation.';

COMMENT ON TABLE research_keyword_packs IS
    'Primary and secondary keyword handoff object used before Content Studio readiness.';

COMMENT ON TABLE research_internal_link_candidates IS
    'Parent/child/hub internal-link suggestions discovered during validation or keyword pack assembly.';

COMMENT ON TABLE research_generated_outcomes IS
    'Mapping layer between validated opportunities and generated content idea outcomes.';
