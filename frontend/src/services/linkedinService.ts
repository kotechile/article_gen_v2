import axios from 'axios';
import { supabase } from '../lib/supabase';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:5001';

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

const getAuthHeaders = async () => {
    const { data: { session } } = await supabase.auth.getSession();
    const token = session?.access_token;
    return {
        'Content-Type': 'application/json',
        'X-API-Key': import.meta.env.VITE_API_KEY || 'development',
        ...(token ? { 'Authorization': `Bearer ${token}` } : {})
    };
};

/**
 * Get LinkedIn OAuth authorization URL
 */
export const getLinkedInAuthUrl = async (): Promise<string> => {
    const headers = await getAuthHeaders();
    const res = await axios.get(`${API_BASE}/api/linkedin/auth-url`, { headers });
    if (!res.data?.auth_url) {
        throw new Error(res.data?.message || 'Could not retrieve LinkedIn authorization URL.');
    }
    return res.data.auth_url;
};

/**
 * Fetch connected LinkedIn account status
 */
export const getLinkedInAccount = async (): Promise<LinkedInAccountStatus> => {
    try {
        const headers = await getAuthHeaders();
        const res = await axios.get(`${API_BASE}/api/linkedin/account`, { headers });
        return res.data;
    } catch (err: any) {
        console.warn('[LinkedInService] Failed to load LinkedIn account:', err?.response?.data || err?.message);
        return { connected: false, account: null };
    }
};

/**
 * Disconnect current LinkedIn account
 */
export const disconnectLinkedInAccount = async (): Promise<void> => {
    const headers = await getAuthHeaders();
    await axios.delete(`${API_BASE}/api/linkedin/account`, { headers });
};

/**
 * Publish post to personal LinkedIn feed
 */
export const publishToLinkedIn = async (
    payload: LinkedInPublishPayload
): Promise<LinkedInPublishResponse> => {
    const headers = await getAuthHeaders();
    const res = await axios.post(`${API_BASE}/api/linkedin/publish`, payload, { headers });
    return res.data;
};

/**
 * Repurpose an existing article into a tailored LinkedIn thought leadership post
 */
export const repurposeForLinkedIn = async (
    title: string,
    content: string,
    tone: string = 'thought_leadership'
): Promise<RepurposedLinkedInContent> => {
    const headers = await getAuthHeaders();
    const res = await axios.post(
        `${API_BASE}/api/linkedin/repurpose`,
        { title, content, tone },
        { headers }
    );
    if (!res.data?.success || !res.data?.reposed && !res.data?.repurposed) {
        throw new Error(res.data?.message || 'Failed to repurpose article for LinkedIn');
    }
    return res.data.repurposed || res.data.reposed;
};
