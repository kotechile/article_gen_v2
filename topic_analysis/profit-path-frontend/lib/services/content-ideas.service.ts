import { apiClient } from '../api-client';
import { supabase } from '../supabase';
import {
    ContentIdea,
    ContentType,
    ContentIdeaGenerationRequest,
    ContentIdeaGenerationResponse
} from '../../types/idea-burst';

class ContentIdeasService {
    /**
     * Generate content ideas based on subtopics and keywords
     */
    async generateContentIdeas(request: ContentIdeaGenerationRequest): Promise<ContentIdeaGenerationResponse> {
        return await apiClient.post<ContentIdeaGenerationResponse>('/api/content-ideas/generate', request);
    }

    /**
     * Generate Idea Burst for specific subtopic (New Flow)
     */
    async generateBurst(request: {
        topicId: string;
        subtopicName: string;
        keywords: string[];
        affiliateOffers: string[];
        userId: string;
    }): Promise<{ success: boolean, blog_ideas: ContentIdea[], software_ideas: ContentIdea[] }> {
        return await apiClient.post('/api/enhanced-topics/idea-burst', {
            user_id: request.userId,
            topic_id: request.topicId,
            subtopic: request.subtopicName,
            keywords: request.keywords,
            affiliate_offers: request.affiliateOffers
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
            const data = await apiClient.post<ContentIdea[]>('/api/content-ideas/list', {
                topic_id: topicId,
                user_id: userId,
                content_type: contentType,
            });

            if (!Array.isArray(data)) {
                console.error('Content ideas API returned non-array:', data);
                return [];
            }
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
            await apiClient.delete(`/api/content-ideas/${ideaId}?user_id=${userId}`);
            return true;
        } catch (error) {
            console.error('Failed to delete content idea:', error);
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
    async publishContentIdeas(ideaIds: string[], userId: string): Promise<boolean> {
        try {
            await apiClient.post('/api/content-ideas/publish', {
                idea_ids: ideaIds,
                user_id: userId
            });
            return true;
        } catch (error) {
            console.error('Failed to publish content ideas:', error);
            return false;
        }
    }
}

export const contentIdeasService = new ContentIdeasService();
