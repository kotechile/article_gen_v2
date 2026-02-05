import { apiClient } from '../api-client';

export interface AffiliateProgram {
    name: string;
    trustScore: number;
    commission: string;
    url: string;
    description?: string;
    id?: string;
}

export interface AffiliateSearchRequest {
    search_term: string;
    niche?: string;
    budget_range?: string;
    user_id: string;
}

export interface AffiliateAnalysis {
    topic: string;
    category: string;
    target_audience: string;
    content_opportunities: string[];
    affiliate_types: string[];
    competition_level: string;
    earnings_potential: string;
}

export interface AffiliateSearchResponse {
    success: boolean;
    message: string;
    data: {
        programs: AffiliateProgram[];
        analysis?: AffiliateAnalysis;
        search_term?: string;
        niche?: string;
        budget_range?: string;
        research_id?: string;
        total_programs?: number;
        timestamp?: string;
    };
}

export interface AffiliateSearchResult {
    programs: AffiliateProgram[];
    analysis?: AffiliateAnalysis;
}

class AffiliateResearchService {
    /**
     * Search for affiliate programs based on criteria
     */
    async searchAffiliatePrograms(request: AffiliateSearchRequest): Promise<AffiliateSearchResult> {
        try {
            const response = await apiClient.post<AffiliateSearchResponse>(
                '/api/affiliate-research/search',
                request,
                { timeout: 120000 } // Increase timeout to 120s for long-running research
            );

            // Extract programs
            let programs: AffiliateProgram[] = [];
            if (response && response.data && Array.isArray(response.data.programs)) {
                programs = response.data.programs;
            } else if (Array.isArray(response.data)) {
                programs = response.data;
            }

            // Extract analysis
            const analysis = response?.data?.analysis;

            return {
                programs,
                analysis
            };
        } catch (error) {
            console.error('Failed to search affiliate programs:', error);
            return { programs: [] };
        }
    }

    /**
     * Get affiliate categories
     */
    async getCategories(): Promise<string[]> {
        const response = await apiClient.get<{ data: { categories: string[] } }>('/api/affiliate-research/categories');
        return response.data.categories;
    }

    /**
     * Get affiliate networks
     */
    async getNetworks(): Promise<any[]> {
        const response = await apiClient.get<{ data: { networks: any[] } }>('/api/affiliate-research/networks');
        return response.data.networks;
    }
}

export const affiliateResearchService = new AffiliateResearchService();
