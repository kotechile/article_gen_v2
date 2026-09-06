/**
 * Editorial Factory Service for Content Generator V2 Frontend.
 *
 * Interfaces with the Editorial Factory API endpoints to list and import
 * articles from the secondary Supabase database into local Titles.
 */

import { apiClient } from '../api-client';

export interface EditorialArticle {
    id: string;
    title: string;
    summary: string;
    content: string;
    hook?: string;
    thesis?: string;
    tags?: string[];
    created_at: string;
    author?: string;
    word_count?: number;
    raw_data?: Record<string, any>;
}

export interface ImportEditorialArticlePayload {
    article_id: string;
    domain?: string;
    wordpress_category_id?: number;
    wordpress_parent_category_id?: number;
}

export interface ImportEditorialArticleResponse {
    success: boolean;
    title_id?: string;
    error?: string;
    data?: any;
}

class EditorialFactoryClientService {
    /**
     * Fetch list of available articles from Editorial Factory.
     */
    public async listArticles(params?: {
        search?: string;
        limit?: number;
        offset?: number;
    }): Promise<EditorialArticle[]> {
        try {
            const queryParams = new URLSearchParams();
            if (params?.search) queryParams.set('search', params.search);
            if (params?.limit) queryParams.set('limit', String(params.limit));
            if (params?.offset) queryParams.set('offset', String(params.offset));

            const queryString = queryParams.toString();
            const url = `/api/v1/editorial-factory/articles${queryString ? `?${queryString}` : ''}`;
            
            const res = await apiClient.get<{
                success: boolean;
                articles: EditorialArticle[];
                count: number;
            }>(url);

            return res?.articles || [];
        } catch (err) {
            console.error('[EditorialFactoryService] Failed to list articles:', err);
            throw err;
        }
    }

    /**
     * Import a specific article into the local Titles database.
     */
    public async importArticle(payload: ImportEditorialArticlePayload): Promise<ImportEditorialArticleResponse> {
        try {
            const res = await apiClient.post<ImportEditorialArticleResponse>(
                '/api/v1/editorial-factory/import',
                payload
            );
            return res;
        } catch (err: any) {
            console.error('[EditorialFactoryService] Failed to import article:', err);
            const message = err.response?.data?.error || err.message || 'Failed to import article';
            return {
                success: false,
                error: message,
            };
        }
    }
}

export const editorialFactoryService = new EditorialFactoryClientService();
