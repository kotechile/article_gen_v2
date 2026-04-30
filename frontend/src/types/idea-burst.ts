/**
 * TypeScript types for Idea Burst and Content Ideas
 */

export type ContentType =
    | 'blog'
    | 'software'
    | 'article'
    | 'comparison'
    | 'guide'
    | 'tutorial'
    | 'review'
    | 'list'
    | 'case_study'
    | 'whitepaper'
    | 'infographic'
    | 'video_script'
    | 'podcast_script';

export type ContentIdeaStatus =
    | 'draft'
    | 'in_progress'
    | 'review'
    | 'approved'
    | 'published'
    | 'archived';

export interface ContentIdea {
    id: string;
    title: string;
    content_type: ContentType;
    primary_keywords: string[];
    secondary_keywords: string[];
    seo_optimization_score: number;
    traffic_potential_score: number;
    total_search_volume: number | null;
    average_difficulty: number | null;
    average_cpc: number | null;
    affiliate_offer_count?: number | null;
    affiliate_search_status?: 'success' | 'failed' | 'not_run' | string | null;
    affiliate_search_error?: string | null;
    affiliate_offers_preview?: Array<{
        name?: string | null;
        network?: string | null;
        commission_rate?: string | null;
    }> | null;
    created_at: string;
    updated_at?: string;
    status?: ContentIdeaStatus;
    user_id: string;
    topic_id: string;
    subtopic?: string;
    description?: string;
    topic_rating?: number;
    published?: boolean;
    published_at?: string;
    published_to_titles?: boolean;
    titles_record_id?: string;
    viability_score?: number;
    trend_score?: number;
    monetization_score?: number;
    seo_ease_score?: number;
    content_outline?: string[];
    keywords?: string[];
    monetization_hook?: string;
    target_intent?: string;
    article_format?: string;
    user_decision_helped?: string;
    internal_link_hook?: string;
    product_type?: string;
    user_job_to_be_done?: string;
    key_inputs?: string[];
    output_result?: string;
    build_complexity?: string;
    distribution_angle?: string;
    search_phrase?: string;
    idea_metadata?: {
        target_intent?: string;
        article_format?: string;
        user_decision_helped?: string;
        internal_link_hook?: string;
        product_type?: string;
        user_job_to_be_done?: string;
        key_inputs?: string[];
        output_result?: string;
        build_complexity?: string;
        distribution_angle?: string;
        seo_offer_enrichment?: {
            keywords_used?: string[];
            keyword_metrics?: Record<string, {
                search_volume?: number;
                keyword_difficulty?: number;
                cpc?: number;
            }>;
            affiliate_offer_count?: number;
            affiliate_offers_preview?: Array<{
                name?: string | null;
                network?: string | null;
                commission_rate?: string | null;
            }>;
            affiliate_search_status?: string | null;
            affiliate_search_error?: string | null;
            raw_dataforseo_output?: Record<string, unknown> | null;
        };
    };
    opportunity_score?: number;
    ranking_breakdown?: {
        viability?: number;
        tool_potential?: number;
        intent_match?: number;
        search_opportunity?: number;
        serp_intent_match?: number;
        seo_ease?: number;
        build_complexity_score?: number;
    };
    keyword_metrics?: Record<string, {
        search_volume?: number;
        keyword_difficulty?: number;
        cpc?: number;
    }>;
    raw_dataforseo_output?: Record<string, unknown> | null;
    raw_supabase_output?: Record<string, unknown> | null;
}

export interface KeywordData {
    id: string;
    keyword: string;
    search_volume: number;
    keyword_difficulty: number;
    cpc: number;
    competition_value: number;
    intent_type: string;
    priority_score: number;
    related_keywords: string[];
    search_volume_trend: any[];
    topic_id: string;
    user_id: string;
    source: string;
    created_at: string;
    updated_at: string;
}

export interface ContentIdeaGenerationRequest {
    topic_id: string;
    topic_title: string;
    subtopics: string[];
    keywords: KeywordData[];
    user_id: string;
    content_types?: ContentType[];
}

export interface IdeaBurstResponse {
    success: boolean;
    blog_ideas: ContentIdea[];
    software_ideas: ContentIdea[];
    generated_count?: number;
    persisted_count?: number;
    persisted_idea_ids?: string[];
    persistence_warning?: string | null;
}

export interface ContentIdeaGenerationResponse {
    success: boolean;
    message: string;
    total_ideas: number;
    blog_ideas: number;
    software_ideas: number;
    ideas: ContentIdea[];
}

// ─── DataForSEO Keyword Intelligence Types ──────────────────────────────────

export interface DFSMonthlySearch {
    year: number;
    month: number;
    search_volume: number;
}

export interface DFSKeywordRow {
    keyword: string;
    type: 'seed' | 'related';
    depth: number;
    search_volume: number | null;
    competition: number | null;
    competition_level: 'LOW' | 'MEDIUM' | 'HIGH' | null;
    cpc: number | null;
    keyword_difficulty: number | null;
    main_intent: string | null;
    foreign_intents: string[] | null;
    monthly_searches: DFSMonthlySearch[];
    search_volume_trend: { monthly: number; quarterly: number; yearly: number } | null;
    low_top_of_page_bid: number | null;
    high_top_of_page_bid: number | null;
    se_results_count: number | null;
    related_keywords: string[] | null;
}

export interface DFSParsedOutput {
    seed_keyword: string;
    total_count: number;
    items_count: number;
    rows: DFSKeywordRow[];
}
