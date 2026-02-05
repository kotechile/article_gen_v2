import { apiClient } from '../api-client';
import { Keyword, ContentTopic, ApiResult } from '../../types/research';

export interface ClusterKeywordsResponse {
    success: boolean;
    clusters_identified: number;
    content_topics_generated: number;
}

export interface KeywordExpansionResponse {
    success: boolean;
    keywords_found: number;
    keywords_saved: number;
}

class KeywordsService {
    /**
     * Get expanded keywords for a specific subtopic
     */
    async getSubtopicKeywords(topicId: string, subtopicId: string): Promise<Keyword[]> {
        try {
            const response = await apiClient.get<Keyword[]>(
                `/research-topics/${topicId}/subtopics/${subtopicId}/keywords`
            );
            return response;
        } catch (error) {
            console.error(`Failed to fetch keywords for subtopic ${subtopicId}:`, error);
            return [];
        }
    }

    /**
     * Get ALL expanded keywords for a research topic (aggregated from all subtopics)
     */
    async getTopicKeywords(topicId: string): Promise<Keyword[]> {
        try {
            const response = await apiClient.get<Keyword[]>(
                `/research-topics/${topicId}/keywords`
            );
            return response;
        } catch (error) {
            console.error(`Failed to fetch keywords for topic ${topicId}:`, error);
            return [];
        }
    }

    /**
     * Expand keywords for ALL subtopics in a topic (Batch)
     */
    async expandAllKeywords(topicId: string): Promise<KeywordExpansionResponse> {
        try {
            const response = await apiClient.post<KeywordExpansionResponse>(
                `/research-topics/${topicId}/keywords/expand_all`,
                {}
            );
            return response;
        } catch (error) {
            console.error('Failed to expand all keywords:', error);
            throw error;
        }
    }

    /**
     * Trigger keyword clustering and content topic generation
     */
    async clusterKeywords(topicId: string): Promise<ClusterKeywordsResponse> {
        try {
            const response = await apiClient.post<ClusterKeywordsResponse>(
                `/research-topics/${topicId}/keywords/cluster`,
                {}
            );
            return response;
        } catch (error) {
            console.error('Failed to cluster keywords:', error);
            throw error;
        }
    }

    /**
     * Get content topics (clustered results)
     */
    async getContentTopics(topicId: string): Promise<ContentTopic[]> {
        try {
            const response = await apiClient.get<ContentTopic[]>(
                `/api/research-topics/${topicId}/content-topics`
            );
            return response;
        } catch (error) {
            console.error('Failed to fetch content topics:', error);
            return [];
        }
    }
}

export const keywordsService = new KeywordsService();
