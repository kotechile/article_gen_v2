// Research Topics Types
export interface ResearchTopic {
    id: string
    title: string
    description: string
    status: ResearchTopicStatus
    version: number
    created_at: string
    updated_at: string
    user_id: string
    sub_topics?: string[]
}

export enum ResearchTopicStatus {
    ACTIVE = 'active',
    COMPLETED = 'completed',
    ARCHIVED = 'archived'
}

export interface ResearchTopicCreate {
    title: string
    description: string
    status?: ResearchTopicStatus
}

export interface ResearchTopicUpdate {
    title?: string
    description?: string
    status?: ResearchTopicStatus
}

export interface ResearchTopicListResponse {
    items: ResearchTopic[]
    total: number
    page: number
    size: number
    has_next: boolean
    has_prev: boolean
}

export interface ResearchTopicListParams {
    status?: ResearchTopicStatus
    order_by?: string
    order_direction?: 'asc' | 'desc'
    page?: number
    size?: number
}

export interface ResearchTopicStats {
    total_topics: number
    active_topics: number
    completed_topics: number
    archived_topics: number
    total_subtopics: number
    total_analyses: number
    total_content_ideas: number
}

// API Result Types
export interface ApiResult<T> {
    success: boolean
    data?: T
    error?: ApiError
}

export interface ApiError {
    message: string
    code?: string
    details?: any
}

export interface Keyword {
    id: string
    research_topic_id: string
    subtopic_id?: string
    seed_keyword: string
    keyword: string
    search_volume?: number
    cpc?: number
    competition?: number
    competition_level?: string
    difficulty?: number
    keyword_difficulty?: number
    main_intent?: string
    intent_type?: string
    profitability_score?: number
    source: string
    created_at: string
    updated_at: string
}

export interface ContentTopic {
    id: string
    research_topic_id: string
    title: string
    primary_keyword_id?: string
    supporting_keyword_ids: string[]
    estimated_profitability_score?: number
    intent_type?: string
    created_at: string
    updated_at: string
}

export interface InterestDataPoint {
    date: string
    value: number
}

export interface Subtopic {
    id: string
    research_topic_id: string
    name: string
    trend_direction: 'up' | 'down' | 'stable' | null
    trend_score: number | null
    interest_over_time: InterestDataPoint[]
    seo_difficulty: number | null
    search_volume: number | null
    cpc: number | null
    affiliate_offer_count: number
    keywords: string[]
    seed_keywords?: string[]
    viability_score: number | null
    created_at: string
    updated_at: string
    rationale?: string
    target_audience?: string
    trend_analysis?: any
    monetization_data?: any
}
