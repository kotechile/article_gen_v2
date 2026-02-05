-- Add missing fields to keyword_research_data table
-- These fields are used by the backend code but missing from the database schema

-- Add fields that the backend code expects
ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS seed_keyword VARCHAR(500);

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS related_keyword VARCHAR(500);

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS competition_level VARCHAR(50);

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS low_top_of_page_bid DECIMAL(10,4);

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS high_top_of_page_bid DECIMAL(10,4);

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS main_intent VARCHAR(50);

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS foreign_intent JSONB DEFAULT '[]';

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS search_intent_last_updated_time TIMESTAMP WITH TIME ZONE;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS monthly_trend INTEGER DEFAULT 0;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS quarterly_trend INTEGER DEFAULT 0;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS yearly_trend INTEGER DEFAULT 0;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS clickstream_search_volume INTEGER;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS clickstream_last_updated_time TIMESTAMP WITH TIME ZONE;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS clickstream_gender_distribution JSONB DEFAULT '{}';

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS clickstream_age_distribution JSONB DEFAULT '{}';

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS clickstream_monthly_searches JSONB DEFAULT '[]';

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS serp_se_type VARCHAR(50);

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS serp_check_url TEXT;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS serp_item_types JSONB DEFAULT '[]';

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS se_results_count BIGINT;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS serp_last_updated_time TIMESTAMP WITH TIME ZONE;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS serp_previous_updated_time TIMESTAMP WITH TIME ZONE;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS avg_dofollow INTEGER;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS avg_referring_pages INTEGER;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS avg_referring_main_domains INTEGER;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS avg_rank INTEGER;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS avg_main_domain_rank INTEGER;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS backlinks_last_updated_time TIMESTAMP WITH TIME ZONE;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS normalized_bing_search_volume INTEGER;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS normalized_bing_is_normalized BOOLEAN DEFAULT FALSE;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS normalized_bing_last_updated_time TIMESTAMP WITH TIME ZONE;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS normalized_bing_monthly_searches JSONB DEFAULT '[]';

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS normalized_clickstream_search_volume INTEGER;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS normalized_clickstream_is_normalized BOOLEAN DEFAULT FALSE;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS normalized_clickstream_last_updated_time TIMESTAMP WITH TIME ZONE;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS normalized_clickstream_monthly_searches JSONB DEFAULT '[]';

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS depth INTEGER DEFAULT 1;

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS core_keyword VARCHAR(500);

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS synonym_clustering_algorithm VARCHAR(100);

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS detected_language VARCHAR(10);

ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS is_another_language BOOLEAN DEFAULT FALSE;

-- Add comments for documentation
COMMENT ON COLUMN keyword_research_data.seed_keyword IS 'Original seed keyword used to generate this keyword';
COMMENT ON COLUMN keyword_research_data.related_keyword IS 'Related keyword if this is a related keyword';
COMMENT ON COLUMN keyword_research_data.competition_level IS 'Competition level (LOW, MEDIUM, HIGH)';
COMMENT ON COLUMN keyword_research_data.low_top_of_page_bid IS 'Lowest bid for top of page position';
COMMENT ON COLUMN keyword_research_data.high_top_of_page_bid IS 'Highest bid for top of page position';
COMMENT ON COLUMN keyword_research_data.main_intent IS 'Main search intent';
COMMENT ON COLUMN keyword_research_data.foreign_intent IS 'Foreign language search intents';
COMMENT ON COLUMN keyword_research_data.search_intent_last_updated_time IS 'When search intent data was last updated';
COMMENT ON COLUMN keyword_research_data.monthly_trend IS 'Monthly search volume trend';
COMMENT ON COLUMN keyword_research_data.quarterly_trend IS 'Quarterly search volume trend';
COMMENT ON COLUMN keyword_research_data.yearly_trend IS 'Yearly search volume trend';
COMMENT ON COLUMN keyword_research_data.clickstream_search_volume IS 'Clickstream search volume';
COMMENT ON COLUMN keyword_research_data.clickstream_last_updated_time IS 'When clickstream data was last updated';
COMMENT ON COLUMN keyword_research_data.clickstream_gender_distribution IS 'Gender distribution from clickstream data';
COMMENT ON COLUMN keyword_research_data.clickstream_age_distribution IS 'Age distribution from clickstream data';
COMMENT ON COLUMN keyword_research_data.clickstream_monthly_searches IS 'Monthly clickstream search data';
COMMENT ON COLUMN keyword_research_data.serp_se_type IS 'Search engine type for SERP data';
COMMENT ON COLUMN keyword_research_data.serp_check_url IS 'URL used for SERP data collection';
COMMENT ON COLUMN keyword_research_data.serp_item_types IS 'Types of items in SERP results';
COMMENT ON COLUMN keyword_research_data.se_results_count IS 'Number of search results';
COMMENT ON COLUMN keyword_research_data.serp_last_updated_time IS 'When SERP data was last updated';
COMMENT ON COLUMN keyword_research_data.serp_previous_updated_time IS 'Previous SERP data update time';
COMMENT ON COLUMN keyword_research_data.avg_dofollow IS 'Average dofollow links';
COMMENT ON COLUMN keyword_research_data.avg_referring_pages IS 'Average referring pages';
COMMENT ON COLUMN keyword_research_data.avg_referring_main_domains IS 'Average referring main domains';
COMMENT ON COLUMN keyword_research_data.avg_rank IS 'Average ranking position';
COMMENT ON COLUMN keyword_research_data.avg_main_domain_rank IS 'Average main domain rank';
COMMENT ON COLUMN keyword_research_data.backlinks_last_updated_time IS 'When backlink data was last updated';
COMMENT ON COLUMN keyword_research_data.normalized_bing_search_volume IS 'Normalized Bing search volume';
COMMENT ON COLUMN keyword_research_data.normalized_bing_is_normalized IS 'Whether Bing data is normalized';
COMMENT ON COLUMN keyword_research_data.normalized_bing_last_updated_time IS 'When normalized Bing data was last updated';
COMMENT ON COLUMN keyword_research_data.normalized_bing_monthly_searches IS 'Monthly normalized Bing search data';
COMMENT ON COLUMN keyword_research_data.normalized_clickstream_search_volume IS 'Normalized clickstream search volume';
COMMENT ON COLUMN keyword_research_data.normalized_clickstream_is_normalized IS 'Whether clickstream data is normalized';
COMMENT ON COLUMN keyword_research_data.normalized_clickstream_last_updated_time IS 'When normalized clickstream data was last updated';
COMMENT ON COLUMN keyword_research_data.normalized_clickstream_monthly_searches IS 'Monthly normalized clickstream search data';
COMMENT ON COLUMN keyword_research_data.depth IS 'Keyword depth level';
COMMENT ON COLUMN keyword_research_data.core_keyword IS 'Core keyword for clustering';
COMMENT ON COLUMN keyword_research_data.synonym_clustering_algorithm IS 'Algorithm used for synonym clustering';
COMMENT ON COLUMN keyword_research_data.detected_language IS 'Detected language of the keyword';
COMMENT ON COLUMN keyword_research_data.is_another_language IS 'Whether keyword is in another language';
