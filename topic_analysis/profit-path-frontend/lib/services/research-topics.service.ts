import { apiClient } from '../api-client'
import type {
    ResearchTopic,
    ResearchTopicCreate,
    ResearchTopicUpdate,
    ResearchTopicListResponse,
    ResearchTopicListParams,
    ResearchTopicStats,
} from '@/types/research'

class ResearchTopicsService {
    private baseUrl = '/research-topics'

    async createResearchTopic(data: ResearchTopicCreate): Promise<ResearchTopic> {
        try {
            const response = await apiClient.post<ResearchTopic>(`${this.baseUrl}/`, data)
            return response
        } catch (error) {
            console.error('Failed to create research topic:', error)
            throw error
        }
    }

    async getResearchTopic(id: string): Promise<ResearchTopic> {
        try {
            const response = await apiClient.get<ResearchTopic>(`${this.baseUrl}/${id}`)
            return response
        } catch (error) {
            console.error('Failed to get research topic:', error)
            throw error
        }
    }

    async updateResearchTopic(id: string, data: ResearchTopicUpdate): Promise<ResearchTopic> {
        try {
            const response = await apiClient.put<ResearchTopic>(`${this.baseUrl}/${id}`, data)
            return response
        } catch (error) {
            console.error('Failed to update research topic:', error)
            throw error
        }
    }

    async deleteResearchTopic(id: string): Promise<void> {
        try {
            await apiClient.delete(`${this.baseUrl}/${id}`)
        } catch (error) {
            console.error('Failed to delete research topic:', error)
            throw error
        }
    }

    async listResearchTopics(params?: ResearchTopicListParams): Promise<ResearchTopicListResponse> {
        try {
            const queryParams = new URLSearchParams()
            if (params?.status) queryParams.append('status', params.status)
            if (params?.order_by) queryParams.append('order_by', params.order_by)
            if (params?.order_direction) queryParams.append('order_direction', params.order_direction)
            if (params?.page) queryParams.append('page', params.page.toString())
            if (params?.size) queryParams.append('size', params.size.toString())

            const url = queryParams.toString() ? `${this.baseUrl}/?${queryParams}` : `${this.baseUrl}/`
            const response = await apiClient.get<ResearchTopicListResponse>(url)
            return response
        } catch (error) {
            console.error('Failed to list research topics:', error)
            throw error
        }
    }

    async getOverviewStats(): Promise<ResearchTopicStats> {
        try {
            const response = await apiClient.get<ResearchTopicStats>(`${this.baseUrl}/stats/overview`)
            return response
        } catch (error) {
            console.error('Failed to get overview stats:', error)
            throw error
        }
    }
}

export const researchTopicsService = new ResearchTopicsService()
export default researchTopicsService
