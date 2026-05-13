ALTER TABLE research_dataforseo_searches ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "Users can view their own research dataforseo searches" ON research_dataforseo_searches;
DROP POLICY IF EXISTS "Users can insert their own research dataforseo searches" ON research_dataforseo_searches;
DROP POLICY IF EXISTS "Users can update their own research dataforseo searches" ON research_dataforseo_searches;
DROP POLICY IF EXISTS "Users can delete their own research dataforseo searches" ON research_dataforseo_searches;

CREATE POLICY "Users can view their own research dataforseo searches" ON research_dataforseo_searches
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own research dataforseo searches" ON research_dataforseo_searches
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own research dataforseo searches" ON research_dataforseo_searches
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own research dataforseo searches" ON research_dataforseo_searches
    FOR DELETE USING (auth.uid() = user_id);

COMMENT ON TABLE research_dataforseo_searches IS
    'RLS enabled: users can access only their own persisted manual DataForSEO searches.';
