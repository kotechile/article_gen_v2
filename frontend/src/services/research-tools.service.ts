import { apiClient } from '@/api-client'

export interface KeywordResult {
    keyword: string
    search_volume?: number
    keyword_difficulty?: number
    cpc?: number
    competition?: string
    intent?: string
    monthly_searches?: Array<{
        year: number
        month: number
        search_volume: number
    }>
}

export const researchToolsService = {
    async getBulkMetrics(keywords: string[]): Promise<KeywordResult[]> {
        const response: any = await apiClient.post('/research-tools/bulk-metrics', {
            keywords
        })
        return response?.keywords || []
    },

    async getWebsiteKeywords(domain: string, limit: number = 100): Promise<KeywordResult[]> {
        const response: any = await apiClient.post('/research-tools/website-keywords', {
            domain,
            limit
        })
        return response?.keywords || []
    },

    async getRelatedKeywords(seedKeyword: string, limit: number = 100): Promise<KeywordResult[]> {
        const response: any = await apiClient.post('/research-tools/related-keywords', {
            seed_keyword: seedKeyword,
            limit
        })
        return response?.keywords || []
    }
}

