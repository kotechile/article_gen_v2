-- Research rebuild RLS policies
-- Supabase-ready row-level security for the research rebuild tables.

-- Enable RLS
ALTER TABLE research_user_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE research_opportunity_candidates ENABLE ROW LEVEL SECURITY;
ALTER TABLE research_validation_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE research_serp_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE research_routing_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE research_keyword_packs ENABLE ROW LEVEL SECURITY;
ALTER TABLE research_internal_link_candidates ENABLE ROW LEVEL SECURITY;
ALTER TABLE research_generated_outcomes ENABLE ROW LEVEL SECURITY;

-- Idempotent policy cleanup
DROP POLICY IF EXISTS "Users can view their own research user jobs" ON research_user_jobs;
DROP POLICY IF EXISTS "Users can insert their own research user jobs" ON research_user_jobs;
DROP POLICY IF EXISTS "Users can update their own research user jobs" ON research_user_jobs;
DROP POLICY IF EXISTS "Users can delete their own research user jobs" ON research_user_jobs;

DROP POLICY IF EXISTS "Users can view their own research opportunity candidates" ON research_opportunity_candidates;
DROP POLICY IF EXISTS "Users can insert their own research opportunity candidates" ON research_opportunity_candidates;
DROP POLICY IF EXISTS "Users can update their own research opportunity candidates" ON research_opportunity_candidates;
DROP POLICY IF EXISTS "Users can delete their own research opportunity candidates" ON research_opportunity_candidates;

DROP POLICY IF EXISTS "Users can view their own research validation runs" ON research_validation_runs;
DROP POLICY IF EXISTS "Users can insert their own research validation runs" ON research_validation_runs;
DROP POLICY IF EXISTS "Users can update their own research validation runs" ON research_validation_runs;
DROP POLICY IF EXISTS "Users can delete their own research validation runs" ON research_validation_runs;

DROP POLICY IF EXISTS "Users can view their own research serp snapshots" ON research_serp_snapshots;
DROP POLICY IF EXISTS "Users can insert their own research serp snapshots" ON research_serp_snapshots;
DROP POLICY IF EXISTS "Users can update their own research serp snapshots" ON research_serp_snapshots;
DROP POLICY IF EXISTS "Users can delete their own research serp snapshots" ON research_serp_snapshots;

DROP POLICY IF EXISTS "Users can view their own research routing decisions" ON research_routing_decisions;
DROP POLICY IF EXISTS "Users can insert their own research routing decisions" ON research_routing_decisions;
DROP POLICY IF EXISTS "Users can update their own research routing decisions" ON research_routing_decisions;
DROP POLICY IF EXISTS "Users can delete their own research routing decisions" ON research_routing_decisions;

DROP POLICY IF EXISTS "Users can view their own research keyword packs" ON research_keyword_packs;
DROP POLICY IF EXISTS "Users can insert their own research keyword packs" ON research_keyword_packs;
DROP POLICY IF EXISTS "Users can update their own research keyword packs" ON research_keyword_packs;
DROP POLICY IF EXISTS "Users can delete their own research keyword packs" ON research_keyword_packs;

DROP POLICY IF EXISTS "Users can view their own research internal link candidates" ON research_internal_link_candidates;
DROP POLICY IF EXISTS "Users can insert their own research internal link candidates" ON research_internal_link_candidates;
DROP POLICY IF EXISTS "Users can update their own research internal link candidates" ON research_internal_link_candidates;
DROP POLICY IF EXISTS "Users can delete their own research internal link candidates" ON research_internal_link_candidates;

DROP POLICY IF EXISTS "Users can view their own research generated outcomes" ON research_generated_outcomes;
DROP POLICY IF EXISTS "Users can insert their own research generated outcomes" ON research_generated_outcomes;
DROP POLICY IF EXISTS "Users can update their own research generated outcomes" ON research_generated_outcomes;
DROP POLICY IF EXISTS "Users can delete their own research generated outcomes" ON research_generated_outcomes;

-- Standard user-scoped policies
CREATE POLICY "Users can view their own research user jobs" ON research_user_jobs
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own research user jobs" ON research_user_jobs
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own research user jobs" ON research_user_jobs
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own research user jobs" ON research_user_jobs
    FOR DELETE USING (auth.uid() = user_id);


CREATE POLICY "Users can view their own research opportunity candidates" ON research_opportunity_candidates
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own research opportunity candidates" ON research_opportunity_candidates
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own research opportunity candidates" ON research_opportunity_candidates
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own research opportunity candidates" ON research_opportunity_candidates
    FOR DELETE USING (auth.uid() = user_id);


CREATE POLICY "Users can view their own research validation runs" ON research_validation_runs
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own research validation runs" ON research_validation_runs
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own research validation runs" ON research_validation_runs
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own research validation runs" ON research_validation_runs
    FOR DELETE USING (auth.uid() = user_id);


CREATE POLICY "Users can view their own research serp snapshots" ON research_serp_snapshots
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own research serp snapshots" ON research_serp_snapshots
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own research serp snapshots" ON research_serp_snapshots
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own research serp snapshots" ON research_serp_snapshots
    FOR DELETE USING (auth.uid() = user_id);


CREATE POLICY "Users can view their own research routing decisions" ON research_routing_decisions
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own research routing decisions" ON research_routing_decisions
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own research routing decisions" ON research_routing_decisions
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own research routing decisions" ON research_routing_decisions
    FOR DELETE USING (auth.uid() = user_id);


CREATE POLICY "Users can view their own research keyword packs" ON research_keyword_packs
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own research keyword packs" ON research_keyword_packs
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own research keyword packs" ON research_keyword_packs
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own research keyword packs" ON research_keyword_packs
    FOR DELETE USING (auth.uid() = user_id);


CREATE POLICY "Users can view their own research internal link candidates" ON research_internal_link_candidates
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own research internal link candidates" ON research_internal_link_candidates
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own research internal link candidates" ON research_internal_link_candidates
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own research internal link candidates" ON research_internal_link_candidates
    FOR DELETE USING (auth.uid() = user_id);


CREATE POLICY "Users can view their own research generated outcomes" ON research_generated_outcomes
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own research generated outcomes" ON research_generated_outcomes
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own research generated outcomes" ON research_generated_outcomes
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own research generated outcomes" ON research_generated_outcomes
    FOR DELETE USING (auth.uid() = user_id);

COMMENT ON TABLE research_user_jobs IS
    'RLS enabled: users can access only their own research user jobs.';

COMMENT ON TABLE research_opportunity_candidates IS
    'RLS enabled: users can access only their own research opportunity candidates.';

COMMENT ON TABLE research_validation_runs IS
    'RLS enabled: users can access only their own research validation runs.';

COMMENT ON TABLE research_serp_snapshots IS
    'RLS enabled: users can access only their own research SERP snapshots.';

COMMENT ON TABLE research_routing_decisions IS
    'RLS enabled: users can access only their own research routing decisions.';

COMMENT ON TABLE research_keyword_packs IS
    'RLS enabled: users can access only their own research keyword packs.';

COMMENT ON TABLE research_internal_link_candidates IS
    'RLS enabled: users can access only their own research internal link candidate suggestions.';

COMMENT ON TABLE research_generated_outcomes IS
    'RLS enabled: users can access only their own research generated outcomes.';
