import { apiClient } from '@/api-client'

export const researchPipelineService = {
    async runPipeline(queryText: string) {
        const response: any = await apiClient.post('/research-pipeline', { query_text: queryText })
        return response.data.clusters
    }
}
