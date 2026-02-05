/**
 * Service layer for image-related API calls
 */

import { supabase } from '../lib/supabase';
import type {
    AIImageRequest,
    AIImageResponse,
    StockImageSearchResponse,
    InfographicTemplate,
    InfographicGenerateRequest,
    InfographicGenerateResponse,
    ImageMetadata,
    ImageProviderModel
} from '../types/image';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001';
const API_KEY = import.meta.env.VITE_API_KEY || 'dev-key';

const getHeaders = (headers: Record<string, string> = {}) => ({
    'X-API-Key': API_KEY,
    ...headers
});

/**
 * Generate AI image using configured providers
 */
export async function generateAIImage(request: AIImageRequest): Promise<AIImageResponse> {
    const response = await fetch(`${API_BASE_URL}/api/v1/images/generate-ai`, {
        method: 'POST',
        headers: getHeaders({
            'Content-Type': 'application/json',
        }),
        body: JSON.stringify(request),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to generate AI image');
    }

    return response.json();
}

/**
 * Search stock images from Pexels or Unsplash
 */
export async function searchStockImages(
    provider: 'pexels' | 'unsplash',
    query: string,
    page: number = 1,
    perPage: number = 10
): Promise<StockImageSearchResponse> {
    const params = new URLSearchParams({
        provider,
        query,
        page: page.toString(),
        perPage: perPage.toString(),
    });

    const response = await fetch(`${API_BASE_URL}/api/v1/images/stock-search?${params}`, {
        headers: getHeaders()
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to search stock images');
    }

    return response.json();
}

/**
 * Upload image file to Supabase storage
 */
export async function uploadImageToSupabase(file: File, userId: string): Promise<{ imageUrl: string }> {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('user_id', userId);

    const response = await fetch(`${API_BASE_URL}/api/v1/images/upload`, {
        method: 'POST',
        headers: getHeaders(),
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to upload image');
    }

    return response.json();
}

/**
 * Get list of available infographic templates
 */
export async function getInfographicTemplates(): Promise<InfographicTemplate[]> {
    const response = await fetch(`${API_BASE_URL}/api/v1/images/infographic/templates`, {
        headers: getHeaders()
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to fetch infographic templates');
    }

    const data = await response.json();

    // Normalize backend response (lowercase keys) to frontend interface (CamelCase)
    // InfographicTemplate expects: Value, Label, sampleImage, numberOfItems, etc.
    // DB columns found: id, name, CSS, HTML, numberOfItems, sampleImage, etc.
    return (data.templates || []).map((t: any) => ({
        id: t.id,
        Value: t.value || t.Value || t.id, // Fallback to id
        Label: t.label || t.Label || t.name, // Fallback to name
        HTML: t.html || t.HTML,
        CSS: t.css || t.CSS,
        prompt: t.prompt,
        jsonStructure: t.json_structure || t.jsonStructure,
        sampleImage: t.sampleimage || t.sampleImage,
        numberOfItems: t.numberofitems || t.numberOfItems
    }));
}

/**
 * Generate infographic from text using LLM and template
 */
export async function generateInfographic(
    request: InfographicGenerateRequest
): Promise<InfographicGenerateResponse> {
    const response = await fetch(`${API_BASE_URL}/api/v1/images/infographic/generate`, {
        method: 'POST',
        headers: getHeaders({
            'Content-Type': 'application/json',
        }),
        body: JSON.stringify(request),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to generate infographic');
    }

    return response.json();
}

/**
 * Save image metadata to database
 */
export async function saveImageMetadata(metadata: ImageMetadata): Promise<ImageMetadata> {
    const response = await fetch(`${API_BASE_URL}/api/v1/images/`, {
        method: 'POST',
        headers: getHeaders({
            'Content-Type': 'application/json',
        }),
        body: JSON.stringify(metadata),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to save image metadata');
    }

    const data = await response.json();

    // Normalize backend response (lowercase keys from Postgres) to frontend interface
    return {
        id: data.id,
        user_id: data.user_id,
        ImageUrl: data.ImageUrl || data.imageurl,
        ImageAuthor: data.ImageAuthor || data.imageauthor,
        MediaAltText: data.MediaAltText || data.mediaalttext,
        mediaTitle: data.mediaTitle || data.mediatitle,
        mediaCaption: data.mediaCaption || data.mediacaption,
        created_at: data.created_at
    };
}

/**
 * Get image metadata by ID
 */
export async function getImageMetadata(imageId: string): Promise<ImageMetadata> {
    const response = await fetch(`${API_BASE_URL}/api/v1/images/${imageId}`, {
        headers: getHeaders()
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to fetch image metadata');
    }

    const data = await response.json();
    return {
        id: data.id,
        user_id: data.user_id,
        ImageUrl: data.ImageUrl || data.imageurl,
        ImageAuthor: data.ImageAuthor || data.imageauthor,
        MediaAltText: data.MediaAltText || data.mediaalttext,
        mediaTitle: data.mediaTitle || data.mediatitle,
        mediaCaption: data.mediaCaption || data.mediacaption,
        created_at: data.created_at
    };
}

/**
 * Update image metadata
 */
export async function updateImageMetadata(imageId: string, metadata: Partial<ImageMetadata>): Promise<ImageMetadata> {
    const response = await fetch(`${API_BASE_URL}/api/v1/images/${imageId}`, {
        method: 'PUT',
        headers: getHeaders({
            'Content-Type': 'application/json',
        }),
        body: JSON.stringify(metadata),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to update image metadata');
    }

    const data = await response.json();
    return {
        id: data.id,
        user_id: data.user_id,
        ImageUrl: data.ImageUrl || data.imageurl,
        ImageAuthor: data.ImageAuthor || data.imageauthor,
        MediaAltText: data.MediaAltText || data.mediaalttext,
        mediaTitle: data.mediaTitle || data.mediatitle,
        mediaCaption: data.mediaCaption || data.mediacaption,
        created_at: data.created_at
    };
}

/**
 * Get available AI image generation models from database
 */
export async function getImageProviderModels(): Promise<ImageProviderModel[]> {
    try {
        const { data, error } = await supabase
            .from('llm_providers_image')
            .select('*')
            .eq('is_active', true)
            .order('display_name', { ascending: true });

        if (error) {
            console.error('Supabase error DETAILED:', {
                message: error.message,
                details: error.details,
                hint: error.hint,
                code: error.code
            });
            throw error;
        }

        // Map to ImageProviderModel interface
        return (data || []).map((m: any) => ({
            id: m.id,
            provider_name: m.provider,
            model_name: m.display_name || m.name,
            model_technical_name: m.model_name,
            supports_reference_image: m.model_name?.includes('sd3') || m.model_name?.includes('stable') || false,
            supported_aspect_ratios: ['1:1', '16:9', '4:3', '3:2', '9:16']
        }));
    } catch (error) {
        console.error('Error fetching image provider models:', error);
        throw error;
    }
}

/**
 * Download stock image and upload to Supabase
 */
export async function downloadAndUploadStockImage(
    imageUrl: string,
    userId: string
): Promise<{ imageUrl: string }> {
    try {
        // Use backend proxy to download and upload image (bypassing CORS)
        const response = await fetch(`${API_BASE_URL}/api/v1/images/download-stock`, {
            method: 'POST',
            headers: getHeaders({
                'Content-Type': 'application/json',
            }),
            body: JSON.stringify({
                url: imageUrl,
                user_id: userId
            })
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.message || `Backend upload failed: ${response.statusText}`);
        }

        const data = await response.json();
        return { imageUrl: data.imageUrl };

    } catch (error) {
        console.error('Error downloading and uploading stock image:', error);
        throw error;
    }
}
