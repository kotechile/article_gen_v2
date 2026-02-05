
export interface Article {
    id: string;
    user_id: string;
    Title: string;
    userDescription: string;
    Keywords: string;
    status: string;
    published: boolean;
    dateCreatedOn: string;
    wordCount?: number; // Inferred or from articleLength?
    articleLength: string;
    LLM: string;
    tone: string;
    hook?: string;
    thesis?: string;
    htmlArticle?: string;
    // Stats for the cards
    seo_optimization_score?: number;
    readability_score?: number;
    // New Metrics
    difficulty_level?: string;
    estimated_reading_time?: string;
    target_audience?: string;
    overall_quality_score?: number;
    viral_potential_score?: number;
    audience_alignment_score?: number;
    content_feasibility_score?: number;
    business_impact_score?: number;
    total_search_volume?: number;
    avg_keyword_difficulty?: number;
    traffic_potential_score?: number;
    competition_score?: number;
    // Featured Image
    featuredImageUrl?: string;
    featuredImageAuthor?: string;
    MediaAltText?: string;
    mediaTitle?: string;
    mediaCaption?: string;

    // Mapped Fields from ProfitPath
    content_outline?: string;
    source_idea_id?: string;
    topic_id?: string;
    subtopic?: string;
    content_type?: string;
}

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
    last_trend_report?: any;
    target_keywords?: string[];
}
