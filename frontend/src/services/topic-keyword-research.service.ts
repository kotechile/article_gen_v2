import { apiClient } from '../api-client'
import axios from 'axios'
import type {
    TopicKeywordCandidateListResponse,
    TopicKeywordClusterListResponse,
    TopicKeywordResearchRun,
    TopicKeywordResearchRunResult,
} from '@/types/research'
import type { IdeaBurstResponse } from '@/types/idea-burst'

export interface TopicKeywordResearchRunRequest {
    replace_existing?: boolean
    filters?: Record<string, any>
    score_config?: Record<string, any>
    manual_seed_keywords?: string[]
}

export interface GenerateIdeasFromClustersRequest {
    user_id: string
    cluster_ids: string[]
}

class TopicKeywordResearchService {
    private baseUrl = '/research-topics'

    private extractErrorMessage(error: unknown, fallback: string): string {
        if (axios.isAxiosError(error)) {
            const serverMessage =
                (error.response?.data as any)?.message ||
                (error.response?.data as any)?.error?.message
            if (serverMessage) return String(serverMessage)
        }
        if (error instanceof Error && error.message) {
            return error.message
        }
        return fallback
    }

    async runTopicKeywordResearch(topicId: string, payload: TopicKeywordResearchRunRequest = {}): Promise<TopicKeywordResearchRunResult> {
        try {
            return await apiClient.post<TopicKeywordResearchRunResult>(
                `${this.baseUrl}/${topicId}/keyword-research/run`,
                payload,
                { timeout: 240000 }
            )
        } catch (error) {
            console.error('Failed to run topic keyword research:', error)
            throw new Error(this.extractErrorMessage(error, 'Failed to run topic keyword research.'))
        }
    }

    async getLatestRun(topicId: string): Promise<TopicKeywordResearchRun> {
        try {
            return await apiClient.get<TopicKeywordResearchRun>(
                `${this.baseUrl}/${topicId}/keyword-research/latest`
            )
        } catch (error) {
            console.error('Failed to fetch latest topic keyword research run:', error)
            throw error
        }
    }

    async getRun(topicId: string, runId: string): Promise<TopicKeywordResearchRun> {
        try {
            return await apiClient.get<TopicKeywordResearchRun>(
                `${this.baseUrl}/${topicId}/keyword-research/runs/${runId}`
            )
        } catch (error) {
            console.error('Failed to fetch topic keyword research run:', error)
            throw error
        }
    }

    async listKeywords(topicId: string, runId: string, includeFiltered = true): Promise<TopicKeywordCandidateListResponse> {
        try {
            const query = includeFiltered ? '' : '?include_filtered=false'
            return await apiClient.get<TopicKeywordCandidateListResponse>(
                `${this.baseUrl}/${topicId}/keyword-research/runs/${runId}/keywords${query}`
            )
        } catch (error) {
            console.error('Failed to list topic keyword candidates:', error)
            throw error
        }
    }

    async listClusters(topicId: string, runId: string): Promise<TopicKeywordClusterListResponse> {
        try {
            return await apiClient.get<TopicKeywordClusterListResponse>(
                `${this.baseUrl}/${topicId}/keyword-research/runs/${runId}/clusters`
            )
        } catch (error) {
            console.error('Failed to list topic keyword clusters:', error)
            throw error
        }
    }

    async generateIdeasFromClusters(topicId: string, runId: string, payload: GenerateIdeasFromClustersRequest): Promise<IdeaBurstResponse> {
        try {
            return await apiClient.post<IdeaBurstResponse>(
                `${this.baseUrl}/${topicId}/keyword-research/runs/${runId}/generate-ideas`,
                payload,
                { timeout: 240000 }
            )
        } catch (error) {
            console.error('Failed to generate ideas from topic keyword clusters:', error)
            throw error
        }
    }
}

export const topicKeywordResearchService = new TopicKeywordResearchService()
export default topicKeywordResearchService
