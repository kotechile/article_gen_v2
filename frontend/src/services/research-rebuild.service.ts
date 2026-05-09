import { apiClient } from '@/api-client'
import type {
    ResearchRebuildCandidate,
    ResearchRebuildGeneratedOutcome,
    ResearchRebuildInternalLinkCandidate,
    ResearchRebuildJob,
    ResearchRebuildKeywordPack,
    ResearchRebuildListResponse,
    ResearchRebuildRoutingDecision,
    ResearchRebuildValidationRun,
    ResearchRebuildRunWorkflowResponse,
    ResearchRebuildPageContextResponse,
    ResearchRebuildTopicContextResponse,
    ResearchRebuildTopicReviewResponse,
    ResearchRebuildTopicScopeSummary,
    ResearchRebuildWorkflowContextResponse,
    ResearchRebuildWorkflowSnapshotResponse,
    ResearchRebuildWorkflowRunSummary,
} from '@/types/research-rebuild'

class ResearchRebuildService {
    private baseUrl = '/research-rebuild'

    async listJobs(params: {
        project_id: string
        primary_category_id?: string
        secondary_category_id?: string
        status?: string
    }): Promise<ResearchRebuildListResponse<ResearchRebuildJob>> {
        const query = new URLSearchParams()
        query.append('project_id', params.project_id)
        if (params.primary_category_id) query.append('primary_category_id', params.primary_category_id)
        if (params.secondary_category_id) query.append('secondary_category_id', params.secondary_category_id)
        if (params.status) query.append('status', params.status)
        return await apiClient.get(`${this.baseUrl}/jobs?${query.toString()}`)
    }

    async getWorkflowSnapshot(params: {
        project_id: string
        primary_category_id?: string
        secondary_category_id?: string
        job_status?: string
        workflow_run_id?: string
        route?: string
        candidate_type?: string
        outcome_type?: string
        search?: string
        limit?: number
        offset?: number
    }): Promise<ResearchRebuildWorkflowSnapshotResponse> {
        const query = new URLSearchParams()
        query.append('project_id', params.project_id)
        if (params.primary_category_id) query.append('primary_category_id', params.primary_category_id)
        if (params.secondary_category_id) query.append('secondary_category_id', params.secondary_category_id)
        if (params.job_status) query.append('job_status', params.job_status)
        if (params.workflow_run_id) query.append('workflow_run_id', params.workflow_run_id)
        if (params.route) query.append('route', params.route)
        if (params.candidate_type) query.append('candidate_type', params.candidate_type)
        if (params.outcome_type) query.append('outcome_type', params.outcome_type)
        if (params.search) query.append('search', params.search)
        if (params.limit !== undefined) query.append('limit', String(params.limit))
        if (params.offset !== undefined) query.append('offset', String(params.offset))
        return await apiClient.get(`${this.baseUrl}/workflow/snapshot?${query.toString()}`)
    }

    async listWorkflowRuns(params: {
        project_id: string
        primary_category_id?: string
        secondary_category_id?: string
        job_status?: string
        limit?: number
    }): Promise<ResearchRebuildListResponse<ResearchRebuildWorkflowRunSummary>> {
        const query = new URLSearchParams()
        query.append('project_id', params.project_id)
        if (params.primary_category_id) query.append('primary_category_id', params.primary_category_id)
        if (params.secondary_category_id) query.append('secondary_category_id', params.secondary_category_id)
        if (params.job_status) query.append('job_status', params.job_status)
        if (params.limit !== undefined) query.append('limit', String(params.limit))
        return await apiClient.get(`${this.baseUrl}/workflow/runs?${query.toString()}`)
    }

    async getTopicReview(params: {
        topic_id: string
        project_id: string
        primary_category_id?: string
        secondary_category_id?: string
        workflow_run_id?: string
        source?: 'all' | 'rebuild' | 'legacy'
        include_suppressed_legacy?: boolean
        limit?: number
    }): Promise<ResearchRebuildTopicReviewResponse> {
        const query = new URLSearchParams()
        query.append('topic_id', params.topic_id)
        query.append('project_id', params.project_id)
        if (params.primary_category_id) query.append('primary_category_id', params.primary_category_id)
        if (params.secondary_category_id) query.append('secondary_category_id', params.secondary_category_id)
        if (params.workflow_run_id) query.append('workflow_run_id', params.workflow_run_id)
        if (params.source) query.append('source', params.source)
        if (params.include_suppressed_legacy !== undefined) query.append('include_suppressed_legacy', String(params.include_suppressed_legacy))
        if (params.limit !== undefined) query.append('limit', String(params.limit))
        return await apiClient.get(`${this.baseUrl}/topic-review?${query.toString()}`)
    }

    async getTopicContext(params: {
        topic_id: string
        project_id: string
        primary_category_id?: string
        secondary_category_id?: string
        review_source?: 'all' | 'rebuild' | 'legacy'
        include_suppressed_legacy?: boolean
        run_limit?: number
        preview_limit?: number
        review_limit?: number
    }): Promise<ResearchRebuildTopicContextResponse> {
        const query = new URLSearchParams()
        query.append('topic_id', params.topic_id)
        query.append('project_id', params.project_id)
        if (params.primary_category_id) query.append('primary_category_id', params.primary_category_id)
        if (params.secondary_category_id) query.append('secondary_category_id', params.secondary_category_id)
        if (params.review_source) query.append('review_source', params.review_source)
        if (params.include_suppressed_legacy !== undefined) query.append('include_suppressed_legacy', String(params.include_suppressed_legacy))
        if (params.run_limit !== undefined) query.append('run_limit', String(params.run_limit))
        if (params.preview_limit !== undefined) query.append('preview_limit', String(params.preview_limit))
        if (params.review_limit !== undefined) query.append('review_limit', String(params.review_limit))
        return await apiClient.get(`${this.baseUrl}/topic-context?${query.toString()}`)
    }

    async listTopicSummaries(params: {
        project_id?: string
        project_ids?: string[]
        primary_category_id?: string
        secondary_category_id?: string
        job_status?: string
        limit?: number
    }): Promise<ResearchRebuildListResponse<ResearchRebuildTopicScopeSummary>> {
        const query = new URLSearchParams()
        if (params.project_id) query.append('project_id', params.project_id)
        for (const projectId of params.project_ids || []) {
            if (projectId) query.append('project_id', projectId)
        }
        if (params.primary_category_id) query.append('primary_category_id', params.primary_category_id)
        if (params.secondary_category_id) query.append('secondary_category_id', params.secondary_category_id)
        if (params.job_status) query.append('job_status', params.job_status)
        if (params.limit !== undefined) query.append('limit', String(params.limit))
        return await apiClient.get(`${this.baseUrl}/topic-summaries?${query.toString()}`)
    }

    async getWorkflowContext(params: {
        project_id: string
        primary_category_id?: string
        secondary_category_id?: string
        job_status?: string
        workflow_run_id?: string
        route?: string
        candidate_type?: string
        outcome_type?: string
        search?: string
        limit?: number
        offset?: number
        run_limit?: number
    }): Promise<ResearchRebuildWorkflowContextResponse> {
        const query = new URLSearchParams()
        query.append('project_id', params.project_id)
        if (params.primary_category_id) query.append('primary_category_id', params.primary_category_id)
        if (params.secondary_category_id) query.append('secondary_category_id', params.secondary_category_id)
        if (params.job_status) query.append('job_status', params.job_status)
        if (params.workflow_run_id) query.append('workflow_run_id', params.workflow_run_id)
        if (params.route) query.append('route', params.route)
        if (params.candidate_type) query.append('candidate_type', params.candidate_type)
        if (params.outcome_type) query.append('outcome_type', params.outcome_type)
        if (params.search) query.append('search', params.search)
        if (params.limit !== undefined) query.append('limit', String(params.limit))
        if (params.offset !== undefined) query.append('offset', String(params.offset))
        if (params.run_limit !== undefined) query.append('run_limit', String(params.run_limit))
        return await apiClient.get(`${this.baseUrl}/workflow-context?${query.toString()}`)
    }

    async getPageContext(params: {
        project_id: string
        primary_category_id?: string
        secondary_category_id?: string
        job_status?: string
        workflow_run_id?: string
        route?: string
        candidate_type?: string
        outcome_type?: string
        search?: string
        limit?: number
        offset?: number
        run_limit?: number
    }): Promise<ResearchRebuildPageContextResponse> {
        const query = new URLSearchParams()
        query.append('project_id', params.project_id)
        if (params.primary_category_id) query.append('primary_category_id', params.primary_category_id)
        if (params.secondary_category_id) query.append('secondary_category_id', params.secondary_category_id)
        if (params.job_status) query.append('job_status', params.job_status)
        if (params.workflow_run_id) query.append('workflow_run_id', params.workflow_run_id)
        if (params.route) query.append('route', params.route)
        if (params.candidate_type) query.append('candidate_type', params.candidate_type)
        if (params.outcome_type) query.append('outcome_type', params.outcome_type)
        if (params.search) query.append('search', params.search)
        if (params.limit !== undefined) query.append('limit', String(params.limit))
        if (params.offset !== undefined) query.append('offset', String(params.offset))
        if (params.run_limit !== undefined) query.append('run_limit', String(params.run_limit))
        return await apiClient.get(`${this.baseUrl}/page-context?${query.toString()}`)
    }

    async createJob(payload: {
        project_id: string
        primary_category_id?: string
        secondary_category_id?: string
        job_text: string
        job_type_hint?: string
        website_context_snapshot?: Record<string, unknown>
    }): Promise<ResearchRebuildJob> {
        return await apiClient.post(`${this.baseUrl}/jobs`, payload)
    }

    async generateJobs(payload: {
        project_id: string
        primary_category_id?: string
        secondary_category_id?: string
        count?: number
        project_name?: string
        website_description?: string
        primary_category_name?: string
        primary_category_description?: string
        secondary_category_name?: string
        secondary_category_description?: string
        target_audience?: string
        focus_area?: string
        avoid_guidance?: string
        trend_titles?: string[]
    }): Promise<ResearchRebuildListResponse<ResearchRebuildJob>> {
        return await apiClient.post(`${this.baseUrl}/jobs/generate`, payload)
    }

    async approveJob(jobId: string): Promise<ResearchRebuildJob> {
        return await apiClient.post(`${this.baseUrl}/jobs/${jobId}/approve`, {})
    }

    async rejectJob(jobId: string, payload: {
        rejection_reason_tags: string[]
        rejection_reason_free_text?: string
    }): Promise<ResearchRebuildJob> {
        return await apiClient.post(`${this.baseUrl}/jobs/${jobId}/reject`, payload)
    }

    async listCandidates(params: {
        project_id: string
        user_job_id?: string
        candidate_type?: string
        status?: string
    }): Promise<ResearchRebuildListResponse<ResearchRebuildCandidate>> {
        const query = new URLSearchParams()
        query.append('project_id', params.project_id)
        if (params.user_job_id) query.append('user_job_id', params.user_job_id)
        if (params.candidate_type) query.append('candidate_type', params.candidate_type)
        if (params.status) query.append('status', params.status)
        return await apiClient.get(`${this.baseUrl}/candidates?${query.toString()}`)
    }

    async listValidationRuns(params: {
        project_id: string
        candidate_id?: string
        freshness_state?: string
    }): Promise<ResearchRebuildListResponse<ResearchRebuildValidationRun>> {
        const query = new URLSearchParams()
        query.append('project_id', params.project_id)
        if (params.candidate_id) query.append('candidate_id', params.candidate_id)
        if (params.freshness_state) query.append('freshness_state', params.freshness_state)
        return await apiClient.get(`${this.baseUrl}/validation-runs?${query.toString()}`)
    }

    async listRoutingDecisions(params: {
        project_id: string
        candidate_id?: string
        route?: string
    }): Promise<ResearchRebuildListResponse<ResearchRebuildRoutingDecision>> {
        const query = new URLSearchParams()
        query.append('project_id', params.project_id)
        if (params.candidate_id) query.append('candidate_id', params.candidate_id)
        if (params.route) query.append('route', params.route)
        return await apiClient.get(`${this.baseUrl}/routing-decisions?${query.toString()}`)
    }

    async listKeywordPacks(params: {
        project_id: string
        candidate_id?: string
        keyword_pack_status?: string
    }): Promise<ResearchRebuildListResponse<ResearchRebuildKeywordPack>> {
        const query = new URLSearchParams()
        query.append('project_id', params.project_id)
        if (params.candidate_id) query.append('candidate_id', params.candidate_id)
        if (params.keyword_pack_status) query.append('keyword_pack_status', params.keyword_pack_status)
        return await apiClient.get(`${this.baseUrl}/keyword-packs?${query.toString()}`)
    }

    async listInternalLinkCandidates(params: {
        project_id: string
        candidate_id?: string
        link_role?: string
    }): Promise<ResearchRebuildListResponse<ResearchRebuildInternalLinkCandidate>> {
        const query = new URLSearchParams()
        query.append('project_id', params.project_id)
        if (params.candidate_id) query.append('candidate_id', params.candidate_id)
        if (params.link_role) query.append('link_role', params.link_role)
        return await apiClient.get(`${this.baseUrl}/internal-link-candidates?${query.toString()}`)
    }

    async listGeneratedOutcomes(params: {
        project_id: string
        candidate_id?: string
        outcome_type?: string
        status?: string
    }): Promise<ResearchRebuildListResponse<ResearchRebuildGeneratedOutcome>> {
        const query = new URLSearchParams()
        query.append('project_id', params.project_id)
        if (params.candidate_id) query.append('candidate_id', params.candidate_id)
        if (params.outcome_type) query.append('outcome_type', params.outcome_type)
        if (params.status) query.append('status', params.status)
        return await apiClient.get(`${this.baseUrl}/generated-outcomes?${query.toString()}`)
    }

    async generateCandidates(payload: {
        project_id: string
        user_job_id: string
    }): Promise<ResearchRebuildListResponse<ResearchRebuildCandidate>> {
        return await apiClient.post(`${this.baseUrl}/candidates/generate`, payload)
    }

    async rejectCandidate(candidateId: string, payload: {
        rejection_reason_tags: string[]
        rejection_reason_free_text?: string
    }): Promise<ResearchRebuildCandidate> {
        return await apiClient.post(`${this.baseUrl}/candidates/${candidateId}/reject`, payload)
    }

    async refreshValidation(payload: {
        validation_run_id: string
        ttl_days?: number
        freshness_state?: 'fresh' | 'stale' | 'expired'
    }): Promise<ResearchRebuildValidationRun> {
        return await apiClient.post(`${this.baseUrl}/validation/refresh`, payload)
    }

    async runWorkflow(payload: {
        project_id: string
        user_job_ids: string[]
        ttl_days?: number
    }): Promise<ResearchRebuildRunWorkflowResponse> {
        return await apiClient.post(`${this.baseUrl}/workflow/run`, payload, {
            timeout: 300000,
        })
    }

    async releaseSoftwareOutcome(outcomeId: string): Promise<{
        released_software_idea: Record<string, unknown>
        generated_outcome: ResearchRebuildGeneratedOutcome
    }> {
        return await apiClient.post(`${this.baseUrl}/generated-outcomes/${outcomeId}/release-software`, {})
    }

    async persistOutcomeToContentIdea(outcomeId: string, payload: {
        project_id?: string
        topic_id?: string
        category_context?: Record<string, unknown>
    }): Promise<{
        content_idea: Record<string, unknown>
        generated_outcome: ResearchRebuildGeneratedOutcome
    }> {
        return await apiClient.post(`${this.baseUrl}/generated-outcomes/${outcomeId}/persist-content-idea`, payload)
    }
}

export const researchRebuildService = new ResearchRebuildService()
export default researchRebuildService
