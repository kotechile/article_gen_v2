ALTER TABLE research_strategy_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE research_topic_bets ENABLE ROW LEVEL SECURITY;
ALTER TABLE research_probe_queries ENABLE ROW LEVEL SECURITY;
ALTER TABLE research_competitor_pages ENABLE ROW LEVEL SECURITY;
ALTER TABLE research_keyword_clusters ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view their own research strategy runs" ON research_strategy_runs;
DROP POLICY IF EXISTS "Users can insert their own research strategy runs" ON research_strategy_runs;
DROP POLICY IF EXISTS "Users can update their own research strategy runs" ON research_strategy_runs;
DROP POLICY IF EXISTS "Users can delete their own research strategy runs" ON research_strategy_runs;

CREATE POLICY "Users can view their own research strategy runs" ON research_strategy_runs
    FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own research strategy runs" ON research_strategy_runs
    FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update their own research strategy runs" ON research_strategy_runs
    FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete their own research strategy runs" ON research_strategy_runs
    FOR DELETE USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can view their own research topic bets" ON research_topic_bets;
DROP POLICY IF EXISTS "Users can insert their own research topic bets" ON research_topic_bets;
DROP POLICY IF EXISTS "Users can update their own research topic bets" ON research_topic_bets;
DROP POLICY IF EXISTS "Users can delete their own research topic bets" ON research_topic_bets;

CREATE POLICY "Users can view their own research topic bets" ON research_topic_bets
    FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own research topic bets" ON research_topic_bets
    FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update their own research topic bets" ON research_topic_bets
    FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete their own research topic bets" ON research_topic_bets
    FOR DELETE USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can view their own research probe queries" ON research_probe_queries;
DROP POLICY IF EXISTS "Users can insert their own research probe queries" ON research_probe_queries;
DROP POLICY IF EXISTS "Users can update their own research probe queries" ON research_probe_queries;
DROP POLICY IF EXISTS "Users can delete their own research probe queries" ON research_probe_queries;

CREATE POLICY "Users can view their own research probe queries" ON research_probe_queries
    FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own research probe queries" ON research_probe_queries
    FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update their own research probe queries" ON research_probe_queries
    FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete their own research probe queries" ON research_probe_queries
    FOR DELETE USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can view their own research competitor pages" ON research_competitor_pages;
DROP POLICY IF EXISTS "Users can insert their own research competitor pages" ON research_competitor_pages;
DROP POLICY IF EXISTS "Users can update their own research competitor pages" ON research_competitor_pages;
DROP POLICY IF EXISTS "Users can delete their own research competitor pages" ON research_competitor_pages;

CREATE POLICY "Users can view their own research competitor pages" ON research_competitor_pages
    FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own research competitor pages" ON research_competitor_pages
    FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update their own research competitor pages" ON research_competitor_pages
    FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete their own research competitor pages" ON research_competitor_pages
    FOR DELETE USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can view their own research keyword clusters" ON research_keyword_clusters;
DROP POLICY IF EXISTS "Users can insert their own research keyword clusters" ON research_keyword_clusters;
DROP POLICY IF EXISTS "Users can update their own research keyword clusters" ON research_keyword_clusters;
DROP POLICY IF EXISTS "Users can delete their own research keyword clusters" ON research_keyword_clusters;

CREATE POLICY "Users can view their own research keyword clusters" ON research_keyword_clusters
    FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own research keyword clusters" ON research_keyword_clusters
    FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update their own research keyword clusters" ON research_keyword_clusters
    FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete their own research keyword clusters" ON research_keyword_clusters
    FOR DELETE USING (auth.uid() = user_id);
