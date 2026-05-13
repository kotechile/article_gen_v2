export type ResearchRebuildJob = {
    id: string
    project_id: string
    primary_category_id?: string | null
    secondary_category_id?: string | null
    job_text: string
    job_type_hint?: string | null
    job_source?: string | null
    status: string
    rejection_reason_tags?: string[]
    rejection_reason_free_text?: string | null
    website_context_snapshot?: Record<string, unknown>
    generation_metadata?: Record<string, unknown>
    created_at?: string
    updated_at?: string
}

export type ResearchRebuildCandidate = {
    id: string
    project_id: string
    user_job_id: string
    candidate_type: 'seo_article' | 'software' | 'editorial'
    candidate_text: string
    normalized_candidate_text?: string | null
    status: string
    candidate_metadata?: Record<string, unknown>
    source_keywords_json?: string[]
    created_at?: string
    updated_at?: string
}

export type ResearchRebuildValidationRun = {
    id: string
    candidate_id: string
    validation_version: string
    validated_at: string
    expires_at?: string | null
    freshness_state: 'fresh' | 'stale' | 'expired'
    eligibility_passed: boolean
    achievability_score?: number | null
    intent_match_score?: number | null
    serp_weakness_score?: number | null
    serp_gap_score?: number | null
    feasibility_score?: number | null
    software_pattern_score?: number | null
    validation_reason_codes?: string[]
    validation_metadata?: Record<string, unknown>
}

export type ResearchRebuildRoutingDecision = {
    id: string
    candidate_id: string
    validation_run_id: string
    route: string
    route_reason_codes?: string[]
    route_metadata?: Record<string, unknown>
}

export type ResearchRebuildKeywordPack = {
    id: string
    candidate_id: string
    validation_run_id: string
    primary_keyword?: string | null
    secondary_keywords_json?: string[]
    keyword_metrics_json?: Record<string, unknown>
    keyword_pack_status: string
    keyword_pack_reason_codes?: string[]
}

export type ResearchRebuildGeneratedOutcome = {
    id: string
    candidate_id: string
    validation_run_id?: string | null
    routing_decision_id?: string | null
    content_idea_id?: string | null
    outcome_type: 'article' | 'software' | 'editorial'
    status: string
    outcome_metadata?: Record<string, unknown>
}

export type ResearchRebuildInternalLinkCandidate = {
    id: string
    candidate_id: string
    validation_run_id?: string | null
    wordpress_imported_post_id?: string | null
    link_role: string
    match_score?: number | null
    match_reason_codes?: string[]
    match_metadata?: Record<string, unknown>
}

export type ResearchRebuildWorkflowCandidateResult = {
    candidate: ResearchRebuildCandidate
    validation_run: ResearchRebuildValidationRun
    routing_decision: ResearchRebuildRoutingDecision
    keyword_pack: ResearchRebuildKeywordPack
    internal_link_candidates?: ResearchRebuildInternalLinkCandidate[]
    generated_outcome: ResearchRebuildGeneratedOutcome
}

export type ResearchRebuildWorkflowJobResult = {
    job_id: string
    job?: ResearchRebuildJob | null
    candidates: ResearchRebuildWorkflowCandidateResult[]
}

export type ResearchRebuildListResponse<T> = {
    items: T[]
    count: number
}

export type ResearchRebuildWorkflowSnapshotResponse = ResearchRebuildListResponse<ResearchRebuildWorkflowJobResult> & {
    total_jobs: number
    limit: number
    offset: number
}

export type ResearchRebuildWorkflowRunSummary = {
    workflow_run_id: string
    started_at?: string
    candidate_count: number
    job_count: number
    primary_category_ids?: string[]
    secondary_category_ids?: string[]
    route_counts?: Record<string, number>
    outcome_counts?: Record<string, number>
}

export type ResearchRebuildTopicScopeSummary = {
    project_id: string
    primary_category_id: string
    secondary_category_id?: string | null
    run_count: number
    latest_run: ResearchRebuildWorkflowRunSummary | null
    dominant_route?: string | null
    route_counts?: Record<string, number>
}

export type ResearchRebuildRunWorkflowResponse = ResearchRebuildListResponse<ResearchRebuildWorkflowJobResult> & {
    workflow_run_id: string
    started_at?: string
}

export type ResearchRebuildTopicReviewItem = {
    id: string
    source: 'rebuild' | 'legacy'
    source_id?: string | null
    title: string
    description?: string | null
    type: string
    status: string
    score?: number | null
    route?: string | null
    keyword?: string | null
    content_idea_id?: string | null
}

export type ResearchRebuildTopicReviewResponse = ResearchRebuildListResponse<ResearchRebuildTopicReviewItem> & {
    suppressed_legacy_count: number
}

export type ResearchRebuildTopicContextResponse = {
    runs: ResearchRebuildWorkflowRunSummary[]
    latest_run?: ResearchRebuildWorkflowRunSummary | null
    latest_workflow_run_id?: string | null
    latest_route?: string | null
    latest_preview_items: ResearchRebuildWorkflowJobResult[]
    review_items: ResearchRebuildTopicReviewItem[]
    suppressed_legacy_count: number
}

export type ResearchRebuildWorkflowContextResponse = {
    runs: ResearchRebuildWorkflowRunSummary[]
    snapshot: ResearchRebuildWorkflowSnapshotResponse
}

export type ResearchRebuildPageContextResponse = {
    jobs: ResearchRebuildListResponse<ResearchRebuildJob>
    workflow: ResearchRebuildWorkflowContextResponse
}

export type ResearchRebuildGenerateJobsResponse = ResearchRebuildListResponse<ResearchRebuildJob> & {
    batch_id?: string
    cleared_count?: number
    exhausted_focus?: boolean
    message?: string | null
}
