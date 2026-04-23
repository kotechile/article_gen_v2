import { apiClient } from '../api-client';
import { supabase } from '../lib/supabase';
import {
    type ContentIdea,
    type ContentIdeaGenerationRequest,
    type ContentIdeaGenerationResponse,
    type IdeaBurstResponse,
} from '../types/idea-burst';

class ContentIdeasService {
    /**
     * Generate content ideas based on subtopics and keywords
     */
    async generateContentIdeas(request: ContentIdeaGenerationRequest): Promise<ContentIdeaGenerationResponse> {
        console.info('[ContentIdeas] generateContentIdeas request', {
            topic_id: request.topic_id,
            user_id: request.user_id,
            subtopics: request.subtopics?.length || 0,
        });
        return await apiClient.post<ContentIdeaGenerationResponse>('/content-ideas/generate', request);
    }

    /**
     * Generate Idea Burst for specific subtopic (New Flow)
     */
    async generateBurst(request: {
        topicId: string;
        subtopicName: string;
        keywords: Array<string | {
            keyword?: string;
            term?: string;
            search_volume?: number | null;
            keyword_difficulty?: number | null;
            difficulty?: number | null;
            cpc?: number | null;
        }>;
        affiliateOffers: string[];
        userId: string;
        intentBucket?: string | null;
        decisionFocus?: string | null;
        angleQuestion?: string | null;
        valueLayerTags?: string[] | null;
        clusterType?: string | null;
        primaryUserOutcome?: string | null;
        serpIntentMatch?: string | null;
        toolPotentialScore?: number | null;
    }): Promise<IdeaBurstResponse> {
        return await apiClient.post('/research-topics/idea-burst', {
            user_id: request.userId,
            topic_id: request.topicId,
            subtopic: request.subtopicName,
            keywords: request.keywords,
            affiliate_offers: request.affiliateOffers,
            intent_bucket: request.intentBucket,
            decision_focus: request.decisionFocus,
            angle_question: request.angleQuestion,
            value_layer_tags: request.valueLayerTags,
            cluster_type: request.clusterType,
            primary_user_outcome: request.primaryUserOutcome,
            serp_intent_match: request.serpIntentMatch,
            tool_potential_score: request.toolPotentialScore,
        });
    }

    /**
     * Get content ideas for a topic
     */
    async getContentIdeas(
        topicId: string,
        userId: string,
        contentType?: string
    ): Promise<ContentIdea[]> {
        try {
            console.info('[ContentIdeas] list request', { topicId, userId, contentType: contentType || 'all' });
            const data = await apiClient.post<ContentIdea[]>('/content-ideas/list', {
                topic_id: topicId,
                user_id: userId,
                content_type: contentType,
            });

            if (!Array.isArray(data)) {
                console.error('Content ideas API returned non-array:', data);
                return [];
            }
            console.info('[ContentIdeas] list response', {
                topicId,
                total: data.length,
                blog: data.filter((idea) => idea.content_type === 'blog').length,
                software: data.filter((idea) => idea.content_type === 'software').length,
            });
            return data || [];
        } catch (error) {
            console.error('Failed to get content ideas:', error);
            // Fallback to Supabase direct query
            return this.getContentIdeasFromSupabase(topicId, userId, contentType);
        }
    }

    /**
     * Fallback method to get content ideas directly from Supabase
     */
    private async getContentIdeasFromSupabase(
        topicId: string,
        userId: string,
        contentType?: string
    ): Promise<ContentIdea[]> {
        try {
            let query = supabase
                .from('content_ideas')
                .select('*')
                .eq('topic_id', topicId)
                .eq('user_id', userId);

            if (contentType) {
                query = query.eq('content_type', contentType);
            }

            const { data, error } = await query.order('created_at', { ascending: false });

            if (error) {
                console.error('Supabase query error:', error);
                return [];
            }

            return data as ContentIdea[] || [];
        } catch (error) {
            console.error('Failed to get content ideas from Supabase:', error);
            return [];
        }
    }

    /**
     * Delete a content idea
     */
    async deleteContentIdea(ideaId: string, userId: string): Promise<boolean> {
        try {
            console.info('[ContentIdeas] delete request', { ideaId, userId });
            await apiClient.delete(`/content-ideas/${ideaId}?user_id=${userId}`);
            console.info('[ContentIdeas] delete success', { ideaId });
            return true;
        } catch (error) {
            console.error('Failed to delete content idea:', error);
            return false;
        }
    }

    /**
     * Persist user star rating for a content idea (0-5).
     */
    async updateContentIdeaRating(ideaId: string, userId: string, rating: number): Promise<boolean> {
        try {
            const nextRating = Math.max(0, Math.min(5, Number(rating || 0)));
            const { error } = await supabase
                .from('content_ideas')
                .update({
                    topic_rating: nextRating,
                    updated_at: new Date().toISOString(),
                })
                .eq('id', ideaId)
                .eq('user_id', userId);

            if (error) {
                console.error('[ContentIdeas] updateContentIdeaRating error:', error);
                return false;
            }
            return true;
        } catch (err) {
            console.error('[ContentIdeas] updateContentIdeaRating exception:', err);
            return false;
        }
    }

    /**
     * Get content ideas grouped by type
     */
    async getContentIdeasGrouped(
        topicId: string,
        userId: string
    ): Promise<{ blog: ContentIdea[]; software: ContentIdea[] }> {
        const allIdeas = await this.getContentIdeas(topicId, userId);

        return {
            blog: allIdeas.filter(idea => idea.content_type === 'blog'),
            software: allIdeas.filter(idea => idea.content_type === 'software'),
        };
    }

    /**
     * Publish content ideas to Titles
     */
    async publishContentIdeas(ideaIds: string[], userId: string): Promise<{
        success: boolean;
        publishedCount: number;
        publishedToTitlesCount: number;
        requestedCount: number;
        message?: string;
    }> {
        try {
            console.info('[ContentIdeas] publish request', {
                userId,
                ideaCount: ideaIds.length,
                ideaIds,
            });
            const result = await apiClient.post<any>('/content-ideas/publish', {
                idea_ids: ideaIds,
                user_id: userId
            });
            const normalized = {
                success: Boolean(result?.success),
                publishedCount: Number(result?.published_count || 0),
                publishedToTitlesCount: Number(result?.published_to_titles_count || 0),
                requestedCount: Number(result?.requested_count || ideaIds.length),
                message: result?.message,
            };
            console.info('[ContentIdeas] publish success', { userId, ...normalized });
            return normalized;
        } catch (error) {
            console.error('Failed to publish content ideas:', error);
            return {
                success: false,
                publishedCount: 0,
                publishedToTitlesCount: 0,
                requestedCount: ideaIds.length,
                message: 'Request failed',
            };
        }
    }

    /**
     * Enrich selected ideas with SEO metrics and affiliate offer signals.
     */
    async enrichContentIdeas(ideaIds: string[], userId: string): Promise<{
        success: boolean;
        requestedCount: number;
        enrichedCount: number;
        results: Array<{
            idea_id: string;
            status: 'enriched' | 'failed';
            reason?: string;
            metrics?: {
                total_search_volume: number;
                average_cpc: number;
                average_difficulty: number;
                affiliate_offer_count: number;
            };
            keywords_used?: string[];
            selected_primary_keyword?: string;
            restated_title?: string;
            title_restated?: boolean;
            keyword_metrics_map?: Record<string, {
                search_volume?: number;
                keyword_difficulty?: number;
                cpc?: number;
            }>;
        }>;
    }> {
        try {
            const result = await apiClient.post<any>('/content-ideas/enrich', {
                idea_ids: ideaIds,
                user_id: userId,
            });

            return {
                success: Boolean(result?.success),
                requestedCount: Number(result?.requested_count || ideaIds.length),
                enrichedCount: Number(result?.enriched_count || 0),
                results: Array.isArray(result?.results) ? result.results : [],
            };
        } catch (error) {
            console.error('Failed to enrich content ideas:', error);
            return {
                success: false,
                requestedCount: ideaIds.length,
                enrichedCount: 0,
                results: [],
            };
        }
    }

    /**
     * Persist primary and secondary keyword selections for a content idea.
     * Writes directly to Supabase so the update is instant and doesn't
     * require a Flask API round-trip.
     */
    async updateKeywordSelection(
        ideaId: string,
        userId: string,
        primaryKeyword: string,
        secondaryKeywords: string[],
        metrics?: {
            volume?: number | null;
            difficulty?: number | null;
            cpc?: number | null;
        }
    ): Promise<boolean> {
        try {
            const allKeywords = [primaryKeyword, ...secondaryKeywords.filter((k) => k !== primaryKeyword)];
            const now = new Date().toISOString();

            const { error } = await supabase
                .from('content_ideas')
                .update({
                    primary_keywords: [primaryKeyword],
                    secondary_keywords: secondaryKeywords.filter((k) => k !== primaryKeyword),
                    keywords: allKeywords,
                    search_phrase: primaryKeyword,
                    total_search_volume: metrics?.volume ?? undefined,
                    average_difficulty: metrics?.difficulty ?? undefined,
                    average_cpc: metrics?.cpc ?? undefined,
                    updated_at: now,
                })
                .eq('id', ideaId)
                .eq('user_id', userId);

            if (error) {
                console.error('[ContentIdeas] updateKeywordSelection error:', error);
                return false;
            }
            return true;
        } catch (err) {
            console.error('[ContentIdeas] updateKeywordSelection exception:', err);
            return false;
        }
    }

    /**
     * Fetch related keywords for a custom seed keyword using the
     * /keyword-lab/related backend endpoint (DataForSEO Labs live).
     */
    async fetchRelatedKeywords(
        seedKeyword: string,
        excludeKeywords?: string[],
        limit = 20,
    ): Promise<{
        success: boolean;
        seed_keyword: string;
        keywords: Array<{
            keyword: string;
            search_volume: number;
            keyword_difficulty: number;
            cpc: number;
            opportunity: number;
        }>;
    }> {
        try {
            const result = await apiClient.post<any>('/content-ideas/keyword-lab/related', {
                seed_keyword: seedKeyword,
                exclude_keywords: excludeKeywords ?? [],
                limit,
            });
            return {
                success: Boolean(result?.success),
                seed_keyword: result?.seed_keyword ?? seedKeyword,
                keywords: Array.isArray(result?.keywords) ? result.keywords : [],
            };
        } catch (err) {
            console.error('[ContentIdeas] fetchRelatedKeywords error:', err);
            return { success: false, seed_keyword: seedKeyword, keywords: [] };
        }
    }
    /**
     * Persist keyword selections for a Titles record (Content Library).
     * Writes directly to the Titles table in Supabase.
     * Mirrors updateKeywordSelection but operates on the Titles table.
     */
    async updateTitleKeywordSelection(
        titleId: string,
        userId: string,
        primaryKeyword: string,
        secondaryKeywords: string[],
        metrics?: {
            volume?: number | null;
            difficulty?: number | null;
            cpc?: number | null;
        }
    ): Promise<boolean> {
        try {
            const allKeywords = [primaryKeyword, ...secondaryKeywords.filter((k) => k !== primaryKeyword)];
            const now = new Date().toISOString();

            const { error } = await supabase
                .from('Titles')
                .update({
                    primary_keywords: [primaryKeyword],
                    secondary_keywords: secondaryKeywords.filter((k) => k !== primaryKeyword),
                    search_phrase: primaryKeyword,
                    // Also update the canonical keyword fields used by content generation
                    primary_keyword: primaryKeyword,
                    secondary_keywords_json: secondaryKeywords.filter((k) => k !== primaryKeyword),
                    keyword_candidates_json: allKeywords,
                    keyword_research_status: 'ready',
                    keyword_research_source: 'manual_keyword_intelligence',
                    keyword_selection_source: 'keyword_intelligence_modal',
                    selected_keyword_search_volume: metrics?.volume ?? undefined,
                    selected_keyword_difficulty: metrics?.difficulty ?? undefined,
                    keyword_research_generated_at: now,
                })
                .eq('id', titleId)
                .eq('user_id', userId);

            if (error) {
                console.error('[ContentIdeas] updateTitleKeywordSelection error:', error);
                return false;
            }
            return true;
        } catch (err) {
            console.error('[ContentIdeas] updateTitleKeywordSelection exception:', err);
            return false;
        }
    }
}

export const contentIdeasService = new ContentIdeasService();
