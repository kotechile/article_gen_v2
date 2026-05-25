import { api } from '@/lib/api'

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
        const response = await api.post('/api/research-tools/bulk-metrics', {
            keywords
        })
        return response.data?.keywords || []
    },

    async getWebsiteKeywords(domain: string, limit: number = 100): Promise<KeywordResult[]> {
        const response = await api.post('/api/research-tools/website-keywords', {
            domain,
            limit
        })
        return response.data?.keywords || []
    },

    async getRelatedKeywords(seedKeyword: string, limit: number = 100): Promise<KeywordResult[]> {
        const response = await api.post('/api/research-tools/related-keywords', {
            seed_keyword: seedKeyword,
            limit
        })
        return response.data?.keywords || []
    }
}
