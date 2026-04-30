import { apiClient } from '../api-client';
import type { Subtopic } from '@/types/research';

export interface SubtopicCreate {
    name: string;
    trend_direction?: 'up' | 'down' | 'stable';
    trend_score?: number;
    seo_difficulty?: number;
    search_volume?: number;
    cpc?: number;
    affiliate_offer_count?: number;
    keywords?: string[];
}

export interface SubtopicUpdate {
    name?: string;
    is_archived?: boolean;
    topic_rating?: number;
    trend_direction?: 'up' | 'down' | 'stable';
    trend_score?: number;
    seo_difficulty?: number;
    search_volume?: number;
    cpc?: number;
    affiliate_offer_count?: number;
    keywords?: string[];
    trend_analysis?: any;
}

export interface SubtopicsResponse {
    items: Subtopic[];
    total: number;
    meta?: {
        success?: boolean;
        message?: string;
        processing_time?: number;
        enhancement_methods?: string[];
        debug?: any;
    };
}

class SubtopicsService {
    private sleep(ms: number): Promise<void> {
        return new Promise((resolve) => setTimeout(resolve, ms))
    }

    private isRetryableFetchError(error: any): boolean {
        const status = error?.response?.status ?? error?.status
        const message = String(error?.message || '').toLowerCase()
        return status === 504 || status === 502 || message.includes('timeout') || message.includes('network error')
    }

    /**
     * Get subtopics for a research topic
     */
    async getSubtopics(topicId: string): Promise<Subtopic[]> {
        const maxAttempts = 4
        for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
            try {
                const response = await apiClient.get<SubtopicsResponse>(
                    `/research-topics/${topicId}/subtopics`,
                    { timeout: 45000 }
                );

                // Handle both old nested format and new flat format for backward compatibility during migration
                if ((response as any).data && (response as any).data.subtopics) {
                    return (response as any).data.subtopics;
                }

                if (response && Array.isArray(response.items)) {
                    return response.items;
                }

                console.warn('Unexpected response format from subtopics API:', response);
                return [];
            } catch (error) {
                const retryable = this.isRetryableFetchError(error)
                const lastAttempt = attempt === maxAttempts
                console.error(`Failed to fetch subtopics (attempt ${attempt}/${maxAttempts}):`, error);
                if (!retryable || lastAttempt) {
                    return [];
                }
                await this.sleep(800 * attempt)
            }
        }
        return []
    }

    /**
     * Generate subtopics using LLM
     */
    async generateSubtopics(topicId: string): Promise<Subtopic[]> {
        try {
            const response = await apiClient.post<SubtopicsResponse>(
                `/research-topics/${topicId}/subtopics/generate`,
                {},
                { timeout: 240000 } // Increased to 4m to handle rate-limited backend
            );
            if ((response.items || []).length === 0 && response.meta?.debug) {
                console.warn('Subtopic generation returned empty result with debug payload:', response.meta.debug);
            }
            return response.items;
        } catch (error) {
            console.error('Failed to generate subtopics:', error);
            throw error;
        }
    }

    /**
     * Enrich subtopics with trend and affiliate data
     */
    async enrichSubtopics(topicId: string): Promise<Subtopic[]> {
        try {
            const response = await apiClient.post<SubtopicsResponse>(
                `/research-topics/${topicId}/enrich`,
                {},
                { timeout: 180000 } // Extended timeout (3m) for deep research
            );
            return response.items;
        } catch (error) {
            console.error('Failed to enrich subtopics:', error);
            throw error;
        }
    }

    /**
     * Enrich a single subtopic with trend and affiliate data
     */
    async enrichSubtopic(topicId: string, subtopicId: string): Promise<Subtopic> {
        try {
            const response = await apiClient.post<Subtopic>(
                `/research-topics/${topicId}/subtopics/${subtopicId}/enrich`,
                {},
                { timeout: 180000 } // Extended timeout (3m) for deep research
            );
            return response;
        } catch (error) {
            console.error(`Failed to enrich subtopic ${subtopicId}:`, error);
            throw error;
        }
    }

    /**
     * Expand keywords for a subtopic
     */
    async expandSubtopicKeywords(topicId: string, subtopicId: string): Promise<{ success: boolean; keywords_found: number; keywords_saved: number }> {
        try {
            const response = await apiClient.post<{ success: boolean; keywords_found: number; keywords_saved: number }>(
                `/research-topics/${topicId}/subtopics/${subtopicId}/keywords/expand`,
                {}
            );
            return response;
        } catch (error) {
            console.error(`Failed to expand keywords for subtopic ${subtopicId}:`, error);
            throw error;
        }
    }

    /**
     * Create a new subtopic
     */
    async createSubtopic(topicId: string, data: SubtopicCreate): Promise<Subtopic | null> {
        try {
            const response = await apiClient.post<Subtopic>(
                `/research-topics/${topicId}/subtopics`,
                data
            );
            return response;
        } catch (error) {
            console.error('Failed to create subtopic:', error);
            throw error;
        }
    }

    /**
     * Update a subtopic
     */
    async updateSubtopic(topicId: string, subtopicId: string, data: SubtopicUpdate): Promise<Subtopic | null> {
        try {
            const response = await apiClient.put<Subtopic>(
                `/research-topics/${topicId}/subtopics/${subtopicId}`,
                data
            );
            return response;
        } catch (error) {
            console.error('Failed to update subtopic:', error);
            throw error;
        }
    }

    /**
     * Delete a subtopic
     */
    async deleteSubtopic(topicId: string, subtopicId: string): Promise<boolean> {
        try {
            await apiClient.delete(
                `/research-topics/${topicId}/subtopics/${subtopicId}`
            );
            return true;
        } catch (error) {
            console.error('Failed to delete subtopic:', error);
            throw error;
        }
    }
}

export const subtopicsService = new SubtopicsService();
