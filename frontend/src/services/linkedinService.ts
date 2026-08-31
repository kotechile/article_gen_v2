import { apiClient } from '@/api-client';

export interface LinkedInAccount {
    id: string;
    account_type: 'personal' | 'organization';
    linkedin_urn: string;
    account_name: string;
    profile_picture_url?: string;
    expires_at?: string;
}

export interface LinkedInAccountStatus {
    connected: boolean;
    is_expired?: boolean;
    account: LinkedInAccount | null;
    warning?: string;
}

export interface LinkedInPublishPayload {
    article_id?: string;
    commentary: string;
    image_url?: string;
    image_alt_text?: string;
    article_url?: string;
    article_title?: string;
    article_description?: string;
}

export interface LinkedInPublishResponse {
    success: boolean;
    message: string;
    post_urn?: string;
    post_url?: string;
    published_at?: string;
    error?: string;
}

export interface RepurposedLinkedInContent {
    hook: string;
    body: string;
    cta: string;
    hashtags: string[];
    full_post: string;
}

/**
 * Get LinkedIn OAuth authorization URL
 */
export const getLinkedInAuthUrl = async (): Promise<string> => {
    const res = await apiClient.get<{ auth_url: string; message?: string }>('/linkedin/auth-url');
    if (!res?.auth_url) {
        throw new Error(res?.message || 'Could not retrieve LinkedIn authorization URL.');
    }
    return res.auth_url;
};

/**
 * Fetch connected LinkedIn account status
 */
export const getLinkedInAccount = async (): Promise<LinkedInAccountStatus> => {
    try {
        const res = await apiClient.get<LinkedInAccountStatus>('/linkedin/account');
        return res;
    } catch (err: any) {
        console.warn('[LinkedInService] Failed to load LinkedIn account:', err?.response?.data || err?.message);
        return { connected: false, account: null };
    }
};

/**
 * Disconnect current LinkedIn account
 */
export const disconnectLinkedInAccount = async (): Promise<void> => {
    await apiClient.delete('/linkedin/account');
};

/**
 * Publish post to personal LinkedIn feed
 */
export const publishToLinkedIn = async (
    payload: LinkedInPublishPayload
): Promise<LinkedInPublishResponse> => {
    const res = await apiClient.post<LinkedInPublishResponse>('/linkedin/publish', payload);
    return res;
};

/**
 * Repurpose an existing article into a tailored LinkedIn thought leadership post
 */
export const repurposeForLinkedIn = async (
    title: string,
    content: string,
    tone: string = 'thought_leadership'
): Promise<RepurposedLinkedInContent> => {
    const res = await apiClient.post<{
        success: boolean;
        repurposed?: RepurposedLinkedInContent;
        message?: string;
    }>('/linkedin/repurpose', { title, content, tone });
    if (!res?.success || !res?.repurposed) {
        throw new Error(res?.message || 'Failed to repurpose article for LinkedIn');
    }
    return res.repurposed;
};
