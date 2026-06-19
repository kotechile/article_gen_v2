/**
 * TypeScript types and interfaces for image-related functionality
 */

export interface ImageMetadata {
    id?: string;
    user_id?: string;

    // Legacy / DB fields
    ImageUrl?: string;
    ImageAuthor?: string;
    MediaAltText?: string;
    mediaTitle?: string;
    mediaCaption?: string;

    // Standardized fields for frontend components
    url?: string;
    author?: string;
    alt?: string;
    title?: string; // Note: 'title' can conflict with HTML title attribute if not careful, but okay in interface
    caption?: string;

    created_at?: string;
    width?: string;
    alignment?: 'left' | 'center' | 'right';
}

export interface AIImageRequest {
    prompt: string;
    model: string;
    aspectRatio: string;
    referenceImage?: string; // base64 encoded
    user_id: string;
}

export interface AIImageResponse {
    imageUrl: string;
    metadata: Partial<ImageMetadata>;
}

export interface StockImageResult {
    id: string;
    url: string;
    thumbnail: string;
    author: string;
    description: string;
    downloadUrl?: string;
}

export interface StockImageSearchResponse {
    images: StockImageResult[];
    totalPages: number;
}

export interface InfographicTemplate {
    id: number;
    Value: string;
    Label: string;
    HTML: string;
    CSS: string;
    prompt: string;
    jsonStructure: string;
    sampleImage: string;
    numberOfItems: number;
}

export interface InfographicGenerateRequest {
    templateId: number;
    storyText: string;
    user_id: string;
}

export interface InfographicGenerateResponse {
    imageUrl: string;
    metadata: Partial<ImageMetadata>;
}

export interface ImageProviderModel {
    id: string;
    provider_name: string;
    model_name: string;
    model_technical_name: string;
    supports_reference_image: boolean;
    supported_aspect_ratios: string[];
}

export type ImageSourceTab = 'ai' | 'stock' | 'upload' | 'url' | 'infographic';
