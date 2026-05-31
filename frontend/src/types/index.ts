
export interface Article {
    id: string;
    user_id: string;
    Title: string;
    userDescription: string;
    Keywords: string;
    status: string;
    published: boolean;
    dateCreatedOn: string;
    wordCount?: number;
    articleLength: string;
    LLM: string;
    tone: string;
    hook?: string;
    thesis?: string;
    htmlArticle?: string;
    seo_optimization_score?: number;
    readability_score?: number;
    difficulty_level?: string;
    estimated_reading_time?: string;
    target_audience?: string;
    overall_quality_score?: number;
    quality_report?: {
        overall_score?: number;
        humanization_score?: number;
        grounding_score?: number;
        geo_score?: number;
        [key: string]: any;
    };
    confidence_map?: Record<string, any>;
    quality_gate?: {
        decision?: string;
        [key: string]: any;
    };
    primary_keyword?: string;
    secondary_keywords_json?: string[] | string;
    keyword_research_source?: string;
    keyword_research_confidence?: number;
    selected_keyword_intent?: string;
    selected_keyword_search_volume?: number;
    selected_keyword_difficulty?: number;
    selected_keyword_metrics_json?: any;
    keyword_selection_source?: string;
    viral_potential_score?: number;
    audience_alignment_score?: number;
    content_feasibility_score?: number;
    business_impact_score?: number;
    total_search_volume?: number;
    avg_keyword_difficulty?: number;
    traffic_potential_score?: number;
    competition_score?: number;
    featuredImageUrl?: string;
    featuredImageAuthor?: string;
    MediaAltText?: string;
    mediaTitle?: string;
    mediaCaption?: string;
    content_outline?: string;
    source_idea_id?: string;
    topic_id?: string;
    subtopic?: string;
    content_type?: string;
}

/** Generic project/niche model — maps to the `projects` Supabase table */
export interface Project {
    id: string;
    created_at: string;
    user_id: string;
    /** Generic niche name, e.g. "Home DIY Blog" */
    app_name?: string;
    /** Optional website URL, e.g. "wellroost.com" */
    domain?: string;
    /** WordPress Application Password (optional) */
    wordpress_key?: string;
    wpUserName?: string;
    wpusername?: string;
    /** PRIMARY niche description used by AI for content generation */
    site_description?: string;
    websiteDescription?: string;
    websitedescription?: string;
    targetAudienceDescription?: string;
    targetaudiencedescription?: string;
    brand_primary_color?: string;
    brand_text_color?: string;
    brand_secondary_color?: string;
    brand_neutral_color?: string;
    branding_updated_at?: string;
    seo_plugin?: 'unknown' | 'yoast' | 'rankmath' | 'custom' | 'none';
    cms_url?: string;
    site_url_override?: string;
    social_default_image_url?: string;
    categories?: string;
    last_trend_report?: any;
    target_keywords?: string[];
}

/** @deprecated Use Project instead */
export interface WordPressDetail {
    id: string;
    created_at: string;
    user_id: string;
    domain: string;
    wordpress_key: string;
    app_name?: string;
    wpUserName: string;
    websiteDescription?: string;
    targetAudienceDescription?: string;
    site_description?: string;
    brand_primary_color?: string;
    brand_text_color?: string;
    brand_secondary_color?: string;
    brand_neutral_color?: string;
    branding_updated_at?: string;
    seo_plugin?: 'unknown' | 'yoast' | 'rankmath' | 'custom' | 'none';
    cms_url?: string;
    site_url_override?: string;
    social_default_image_url?: string;
    last_trend_report?: any;
    target_keywords?: string[];
}
