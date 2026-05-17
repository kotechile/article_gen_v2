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
            'relevant_pages',
            'categories_for_domain',
            'category_index'
        )
    );
