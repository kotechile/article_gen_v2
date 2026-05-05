export type TopicMode = 'keyword_first' | 'editorial_first' | 'hybrid'
export type KeywordViabilityLabel = 'high' | 'medium' | 'low'

// Research Topics Types
export interface ResearchTopic {
    id: string
    title: string
    description: string
    status: ResearchTopicStatus
    is_archived?: boolean
    topic_rating?: number
    version: number
    created_at: string
    updated_at: string
    user_id: string
    sub_topics?: string[]
    project_id?: string | null
    primary_category_id?: string | null
    secondary_category_id?: string | null
    topic_source?: string | null
    source_topic_id?: string | null
    intent_bucket?: string | null
    decision_focus?: string | null
    angle_question?: string | null
    value_layer_tags?: string[] | null
    target_audience?: string | null
    evidence_sources?: string[] | null
    related_terms?: string[] | null
    topic_mode?: TopicMode | null
    keyword_viability_score?: number | null
    keyword_viability_label?: KeywordViabilityLabel | null
    topic_generation_reasoning?: string | null
    topic_generation_metadata?: Record<string, any> | null
    project_name?: string | null
    primary_category_name?: string | null
    secondary_category_name?: string | null
    subtopics_count?: number | null
    researched_subtopics_count?: number | null
    topic_keyword_candidate_count?: number | null
    topic_keyword_cluster_count?: number | null
    topic_keyword_research_status?: 'pending' | 'running' | 'completed' | 'failed' | string | null
    content_ideas_count?: number | null
    in_library_count?: number | null
    has_underlying_data?: boolean | null
    all_subtopics_researched?: boolean | null
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
    is_archived?: boolean
    topic_rating?: number
    project_id?: string | null
    primary_category_id?: string | null
    secondary_category_id?: string | null
    topic_source?: string | null
    source_topic_id?: string | null
    intent_bucket?: string | null
    decision_focus?: string | null
    angle_question?: string | null
    value_layer_tags?: string[] | null
    target_audience?: string | null
    evidence_sources?: string[] | null
    related_terms?: string[] | null
    topic_mode?: TopicMode | null
    keyword_viability_score?: number | null
    keyword_viability_label?: KeywordViabilityLabel | null
    topic_generation_reasoning?: string | null
    topic_generation_metadata?: Record<string, any> | null
}

export interface ResearchTopicBulkCreateItem extends ResearchTopicCreate {}

export interface ResearchTopicUpdate {
    title?: string
    description?: string
    status?: ResearchTopicStatus
    is_archived?: boolean
    topic_rating?: number
    project_id?: string | null
    primary_category_id?: string | null
    secondary_category_id?: string | null
    topic_source?: string | null
    source_topic_id?: string | null
    intent_bucket?: string | null
    decision_focus?: string | null
    angle_question?: string | null
    value_layer_tags?: string[] | null
    target_audience?: string | null
    evidence_sources?: string[] | null
    related_terms?: string[] | null
    topic_mode?: TopicMode | null
    keyword_viability_score?: number | null
    keyword_viability_label?: KeywordViabilityLabel | null
    topic_generation_reasoning?: string | null
    topic_generation_metadata?: Record<string, any> | null
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
    project_id?: string
    primary_category_id?: string
    secondary_category_id?: string
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
    is_archived?: boolean
    topic_rating?: number
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
    researched?: boolean
    intent_bucket?: string | null
    decision_focus?: string | null
    angle_question?: string | null
    value_layer_tags?: string[] | null
    cluster_type?: string | null
    primary_user_outcome?: string | null
    serp_intent_match?: string | null
    tool_potential_score?: number | null
}

export interface TopicKeywordResearchRun {
    id: string
    topic_id: string
    user_id: string
    status: 'pending' | 'running' | 'completed' | 'failed'
    seed_keywords_json: string[]
    filters_json: Record<string, any>
    score_config_json: Record<string, any>
    summary_json: Record<string, any>
    raw_data_json: Record<string, any>
    error_message?: string | null
    created_at: string
    updated_at: string
}

export interface TopicKeywordCandidate {
    id: string
    research_run_id: string
    topic_id: string
    user_id: string
    keyword: string
    canonical_keyword: string
    variant_keywords_json: string[]
    source_endpoints_json: string[]
    search_volume?: number | null
    cpc?: number | null
    competition?: string | null
    competition_index?: number | null
    keyword_difficulty?: number | null
    trend_json?: Record<string, any> | null
    intent_label?: string | null
    topical_fit_score?: number | null
    opportunity_score?: number | null
    is_filtered_out?: boolean
    filter_reason?: string | null
    created_at: string
    updated_at: string
}

export interface TopicKeywordClusterKeyword {
    keyword: string
    canonical_keyword?: string
    opportunity_score?: number
    search_volume?: number | null
    keyword_difficulty?: number | null
    cpc?: number | null
}

export interface TopicKeywordCluster {
    id: string
    research_run_id: string
    topic_id: string
    user_id: string
    cluster_name: string
    primary_keyword?: string | null
    secondary_keywords_json: string[]
    keyword_candidates_json: TopicKeywordClusterKeyword[]
    intent_label?: string | null
    serp_validation_json?: Record<string, any> | null
    opportunity_score?: number | null
    software_opportunity_score?: number | null
    article_angle?: string | null
    rationale?: string | null
    created_at: string
    updated_at: string
}

export interface TopicKeywordResearchRunResult {
    success: boolean
    run: TopicKeywordResearchRun
    summary?: Record<string, any>
    keyword_count: number
    cluster_count: number
    top_clusters: TopicKeywordCluster[]
}

export interface TopicKeywordCandidateListResponse {
    items: TopicKeywordCandidate[]
    total: number
    include_filtered: boolean
}

export interface TopicKeywordClusterListResponse {
    items: TopicKeywordCluster[]
    total: number
}
