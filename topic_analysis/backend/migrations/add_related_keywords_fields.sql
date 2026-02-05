-- Add comprehensive fields for DataForSEO related keywords data
-- This migration adds all the fields extracted from the DataForSEO API response

-- Add new columns to keyword_research_data table
ALTER TABLE keyword_research_data 
ADD COLUMN IF NOT EXISTS low_top_of_page_bid DECIMAL(10,4),
ADD COLUMN IF NOT EXISTS high_top_of_page_bid DECIMAL(10,4),
ADD COLUMN IF NOT EXISTS categories JSONB,
ADD COLUMN IF NOT EXISTS monthly_searches JSONB,
ADD COLUMN IF NOT EXISTS core_keyword TEXT,
ADD COLUMN IF NOT EXISTS synonym_clustering_algorithm TEXT,
ADD COLUMN IF NOT EXISTS detected_language TEXT,
ADD COLUMN IF NOT EXISTS is_another_language BOOLEAN,
ADD COLUMN IF NOT EXISTS foreign_intent JSONB,
ADD COLUMN IF NOT EXISTS search_intent_last_updated_time TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS clickstream_search_volume INTEGER,
ADD COLUMN IF NOT EXISTS clickstream_last_updated_time TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS clickstream_gender_distribution JSONB,
ADD COLUMN IF NOT EXISTS clickstream_age_distribution JSONB,
ADD COLUMN IF NOT EXISTS clickstream_monthly_searches JSONB,
ADD COLUMN IF NOT EXISTS serp_se_type TEXT,
ADD COLUMN IF NOT EXISTS serp_check_url TEXT,
ADD COLUMN IF NOT EXISTS serp_item_types JSONB,
ADD COLUMN IF NOT EXISTS se_results_count INTEGER,
ADD COLUMN IF NOT EXISTS serp_last_updated_time TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS serp_previous_updated_time TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS avg_dofollow DECIMAL(10,2),
ADD COLUMN IF NOT EXISTS avg_referring_pages DECIMAL(10,2),
ADD COLUMN IF NOT EXISTS avg_referring_main_domains DECIMAL(10,2),
ADD COLUMN IF NOT EXISTS avg_rank DECIMAL(10,2),
ADD COLUMN IF NOT EXISTS avg_main_domain_rank DECIMAL(10,2),
ADD COLUMN IF NOT EXISTS backlinks_last_updated_time TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS normalized_bing_search_volume INTEGER,
ADD COLUMN IF NOT EXISTS normalized_bing_is_normalized BOOLEAN,
ADD COLUMN IF NOT EXISTS normalized_bing_last_updated_time TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS normalized_bing_monthly_searches JSONB,
ADD COLUMN IF NOT EXISTS normalized_clickstream_search_volume INTEGER,
ADD COLUMN IF NOT EXISTS normalized_clickstream_is_normalized BOOLEAN,
ADD COLUMN IF NOT EXISTS normalized_clickstream_last_updated_time TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS normalized_clickstream_monthly_searches JSONB,
ADD COLUMN IF NOT EXISTS depth INTEGER;

-- Add comments for documentation
COMMENT ON COLUMN keyword_research_data.low_top_of_page_bid IS 'Minimum bid for ad to be displayed at top of first page';
COMMENT ON COLUMN keyword_research_data.high_top_of_page_bid IS 'Maximum bid for ad to be displayed at top of first page';
COMMENT ON COLUMN keyword_research_data.categories IS 'Product and service categories for the keyword';
COMMENT ON COLUMN keyword_research_data.monthly_searches IS 'Monthly search volume data for past 12 months';
COMMENT ON COLUMN keyword_research_data.core_keyword IS 'Main keyword in a group determined by synonym clustering';
COMMENT ON COLUMN keyword_research_data.synonym_clustering_algorithm IS 'Algorithm used to identify synonyms';
COMMENT ON COLUMN keyword_research_data.detected_language IS 'Language detected by DataForSEO system';
COMMENT ON COLUMN keyword_research_data.is_another_language IS 'Whether detected language differs from requested language';
COMMENT ON COLUMN keyword_research_data.foreign_intent IS 'Supplementary search intents beyond main intent';
COMMENT ON COLUMN keyword_research_data.clickstream_search_volume IS 'Clickstream-based search volume';
COMMENT ON COLUMN keyword_research_data.clickstream_gender_distribution IS 'Gender distribution from clickstream data';
COMMENT ON COLUMN keyword_research_data.clickstream_age_distribution IS 'Age distribution from clickstream data';
COMMENT ON COLUMN keyword_research_data.serp_check_url IS 'Direct URL to search engine results';
COMMENT ON COLUMN keyword_research_data.serp_item_types IS 'Types of search results found in SERP';
COMMENT ON COLUMN keyword_research_data.se_results_count IS 'Number of search results for the keyword';
COMMENT ON COLUMN keyword_research_data.avg_dofollow IS 'Average number of dofollow backlinks';
COMMENT ON COLUMN keyword_research_data.avg_referring_pages IS 'Average number of referring pages';
COMMENT ON COLUMN keyword_research_data.avg_referring_main_domains IS 'Average number of referring main domains';
COMMENT ON COLUMN keyword_research_data.avg_rank IS 'Average rank of top-10 pages';
COMMENT ON COLUMN keyword_research_data.avg_main_domain_rank IS 'Average main domain rank';
COMMENT ON COLUMN keyword_research_data.normalized_bing_search_volume IS 'Search volume normalized with Bing data';
COMMENT ON COLUMN keyword_research_data.normalized_clickstream_search_volume IS 'Search volume normalized with clickstream data';
COMMENT ON COLUMN keyword_research_data.depth IS 'Keyword search depth from DataForSEO';

-- Create indexes for better performance on new fields
CREATE INDEX IF NOT EXISTS idx_keyword_research_core_keyword ON keyword_research_data(core_keyword);
CREATE INDEX IF NOT EXISTS idx_keyword_research_detected_language ON keyword_research_data(detected_language);
CREATE INDEX IF NOT EXISTS idx_keyword_research_serp_item_types ON keyword_research_data USING GIN(serp_item_types);
CREATE INDEX IF NOT EXISTS idx_keyword_research_categories ON keyword_research_data USING GIN(categories);
CREATE INDEX IF NOT EXISTS idx_keyword_research_foreign_intent ON keyword_research_data USING GIN(foreign_intent);
