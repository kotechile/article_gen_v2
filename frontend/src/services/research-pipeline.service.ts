import { apiClient } from '@/api-client'

export const researchPipelineService = {
    async extractKeywords(queryText: string) {
        const response: any = await apiClient.post('/research-pipeline/extract', { query_text: queryText })
        return {
            runId: response?.run_id || response?.data?.run_id,
            keywords: response?.keywords || response?.data?.keywords || []
        }
    },
    async clusterKeywords(keywords: any[]) {
        const response: any = await apiClient.post('/research-pipeline/cluster', { keywords })
        return response?.clusters || response?.data?.clusters || []
    },
    // Backwards compatibility
    async runPipeline(queryText: string) {
        const response: any = await apiClient.post('/research-pipeline', { query_text: queryText })
        return response?.clusters || response?.data?.clusters || []
    }
}
