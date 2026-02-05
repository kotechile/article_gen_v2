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
    total_search_volume: number;
    average_difficulty: number;
    average_cpc: number;
    created_at: string;
    updated_at?: string;
    status?: ContentIdeaStatus;
    user_id: string;
    topic_id: string;
    subtopic?: string;
    description?: string;
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

export interface ContentIdeaGenerationResponse {
    success: boolean;
    message: string;
    total_ideas: number;
    blog_ideas: number;
    software_ideas: number;
    ideas: ContentIdea[];
}
