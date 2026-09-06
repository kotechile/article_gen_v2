/**
 * Keyword Optimization Service for Content Generator V2 Frontend.
 *
 * Interfaces with the Keyword Optimization endpoints:
 * - DataForSEO keyword discovery
 * - Competitiveness metrics lookup
 * - AI non-destructive keyword weaving into article HTML
 * - Persisting keywords to Titles records
 */

import { apiClient } from '../api-client';

export interface KeywordCandidate {
    keyword: string;
    search_volume: number;
    keyword_difficulty: number | null;
    cpc: number;
    intent: string;
    competition: string;
    opportunity_score: number;
    in_text_count: number;
    is_seed?: boolean;
}

export interface WeaveKeywordsResponse {
    success: boolean;
    html: string;
    changes: string[];
    placements?: Array<{ keyword: string; type: 'primary' | 'secondary'; count: number }>;
    error?: string;
}

export interface SaveKeywordsPayload {
    title_id: string;
    primary_keyword?: string | null;
    secondary_keywords?: string[];
    primary_metric?: {
        search_volume?: number;
        keyword_difficulty?: number | null;
        cpc?: number;
        intent?: string;
    };
    html?: string;
}

class KeywordOptimizationClientService {
    /**
     * Discover related keywords and live metrics from DataForSEO based on article text or custom seed.
     */
    public async discoverKeywords(params: {
        title?: string;
        content?: string;
        tags?: string[];
        custom_seed?: string;
    }): Promise<KeywordCandidate[]> {
        try {
            const res = await apiClient.post<{
                success: boolean;
                keywords: KeywordCandidate[];
                count: number;
            }>('/api/v1/keywords/discover-for-article', params);

            return res?.keywords || [];
        } catch (err) {
            console.error('[KeywordOptimizationService] Discover failed:', err);
            throw err;
        }
    }

    /**
     * Direct DataForSEO query for a specific phrase.
     */
    public async searchKeyword(query: string): Promise<KeywordCandidate[]> {
        try {
            const res = await apiClient.post<{
                success: boolean;
                keywords: KeywordCandidate[];
                count: number;
            }>('/api/v1/keywords/search-dataforseo', { query });

            return res?.keywords || [];
        } catch (err) {
            console.error('[KeywordOptimizationService] Search failed:', err);
            throw err;
        }
    }

    /**
     * Weave selected primary and secondary keywords into article HTML.
     */
    public async weaveKeywords(params: {
        html: string;
        primary_keyword: string;
        secondary_keywords?: string[];
        instructions?: string;
    }): Promise<WeaveKeywordsResponse> {
        try {
            const res = await apiClient.post<WeaveKeywordsResponse>('/api/v1/keywords/weave-into-article', params);
            return res;
        } catch (err) {
            console.error('[KeywordOptimizationService] Weaving failed:', err);
            throw err;
        }
    }

    /**
     * Save keywords, metrics, and HTML directly to Titles table.
     */
    public async saveToTitle(payload: SaveKeywordsPayload): Promise<{ success: boolean }> {
        try {
            const res = await apiClient.post<{ success: boolean; updated: any[] }>('/api/v1/keywords/save-to-title', payload);
            return { success: Boolean(res?.success) };
        } catch (err) {
            console.error('[KeywordOptimizationService] Save to title failed:', err);
            throw err;
        }
    }
}

export const keywordOptimizationService = new KeywordOptimizationClientService();
