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
    link?: string;
}

export interface AIImageRequest {
    prompt: string;
    model?: string;
    application?: 'article_image' | 'infographics' | string;
    aspectRatio: string;
    resolution?: string; // '1K', '2K', '4K'
    referenceImage?: string; // base64 encoded (optional)
    referenceImageUrls?: string[]; // optional
    user_id: string;
    overlay?: OverlayConfig;
}

export interface AIImageResponse {
    imageUrl: string;
    baseImageUrl?: string;
    overlayDetails?: OverlayDetails;
    metadata: Partial<ImageMetadata>;
    model?: string;
    provider?: string;
    application?: string;
    aspectRatio?: string;
    resolution?: string;
}

export interface ImageApplicationConfig {
    application: string;
    provider: string | null;
    model_name: string | null;
    display_name: string | null;
    llm_image_id: string | null;
    has_api_key: boolean;
    source: string;
}

export interface ImageApplicationsResponse {
    applications: {
        article_image?: ImageApplicationConfig;
        infographics?: ImageApplicationConfig;
        [key: string]: ImageApplicationConfig | undefined;
    };
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
    supported_resolutions?: string[];
}

export interface ContextReferenceImage {
    url: string;
    thumbnail_url?: string;
    title?: string;
    source_domain?: string;
    provider?: string;
    score?: number;
}

export interface ContextAnalyzeRequest {
    text: string;
    user_instructions?: string;
    max_reference_images?: number;
    article_title?: string;
    article_context?: {
        title?: string;
        thesis?: string;
        hook?: string;
        deck?: string;
        excerpt?: string;
        vertical?: string;
        topic?: string;
    };
    style_id?: string;
    style?: string;
}

export interface ContextAnalyzeResult {
    has_physical_entity: boolean;
    main_object: string;
    hero_subject?: string;
    core_thesis?: string;
    core_conflict?: string;
    composition?: string;
    style_id?: string;
    style_label?: string;
    alt_text?: string;
    caption?: string;
    title?: string;
    negative_prompt?: string;
    search_query: string;
    generation_prompt: string;
    object_fidelity_weight: number;
    entity_type?: 'physical' | 'metaphorical';
    is_metaphorical?: boolean;
    candidate_references: ContextReferenceImage[];
}

export interface ContextAnalyzeResponse {
    status: string;
    data: ContextAnalyzeResult;
}

export interface ContextGenerateRequest {
    text?: string;
    prompt?: string;
    reference_image_url?: string;
    reference_image_base64?: string;
    referenceImage?: string;
    model?: string;
    aspectRatio?: string;
    resolution?: string;
    user_id?: string;
    application?: string;
    isolate_background?: boolean;
    overlay?: OverlayConfig;
}

export interface ContextGenerateResponse {
    imageUrl: string;
    baseImageUrl?: string;
    overlayDetails?: OverlayDetails;
    metadata: Partial<ImageMetadata>;
    model?: string;
    provider?: string;
    application?: string;
    aspectRatio?: string;
    resolution?: string;
    referenceUsed?: string;
    extractedAnalysis?: ContextAnalyzeResult;
}

export type ImageSourceTab = 'smart' | 'ai' | 'stock' | 'upload' | 'url' | 'infographic';

export type InfographicArchetype =
    | 'auto'
    | 'technical_scientific'
    | 'step_by_step'
    | 'flowchart_whiteboard'
    | 'modular_explainer'
    | 'timeline_historical'
    | 'data_visualization'
    | 'playful_viral'
    // Process & Sequential Flow
    | 'step_by_step_isometric'
    | 'timeline_modern'
    | 'timeline_historical_vintage'
    | 'user_journey_flat'
    | 'lifecycle_loop_watercolor'
    // Comparison & Contrast
    | 'side_by_side_neon'
    | 'pros_cons_scandinavian'
    | 'venn_diagram_glassmorphism'
    | 'quadrant_matrix_bauhaus'
    // Data & Statistics
    | 'corporate_dashboard_ui'
    | 'typography_stat_sheet_swiss'
    | 'geographic_map_hologram'
    | 'funnel_chart_neumorphism'
    // Structure & Hierarchy
    | 'pyramid_hierarchy_lowpoly'
    | 'hub_and_spoke_material'
    | 'anatomy_exploded_blueprint'
    | 'mind_map_doodle'
    // Lists & Summaries
    | 'checklist_synthwave'
    | 'top_10_listicle_popart'
    | 'cheat_sheet_monochrome'
    | 'problem_solution_duotone'
    | (string & {});

export interface AIInfographicRequest {
    text: string;
    archetype?: InfographicArchetype;
    style?: string;
    user_instructions?: string;
    aspectRatio?: string;
    resolution?: string;
    user_id: string;
}

export interface AIInfographicResponse {
    imageUrl: string;
    metadata: Partial<ImageMetadata>;
    archetype: string;
    style?: string;
    model?: string;
    provider?: string;
    application?: string;
    aspectRatio?: string;
    resolution?: string;
    prompt?: string;
    baseImageUrl?: string;
    overlayDetails?: OverlayDetails;
}

export interface OverlayConfig {
    enabled: boolean;
    kicker?: string;
    title?: string;
    hook?: string;
    corner?: 'auto' | 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
}

export interface OverlayDetails {
    corner: string;
    ink_palette: 'dark' | 'light';
    contrast_ratio: number;
    scrim_alpha: number;
    wcag_target: number;
    clutter: number;
    kicker: string;
    title: string;
    hook: string;
    base_width?: number;
    base_height?: number;
    final_width?: number;
    final_height?: number;
}

export interface GenerateOverlayCopyRequest {
    text: string;
    article_title?: string;
    article_context?: Record<string, any>;
    user_instructions?: string;
}

export interface GenerateOverlayCopyResponse {
    status: string;
    kicker: string;
    title: string;
    hook: string;
}

export interface ApplyOverlayRequest {
    image_url?: string;
    image_base64?: string;
    kicker: string;
    title: string;
    hook: string;
    corner?: string;
    user_id?: string;
}

export interface ApplyOverlayResponse {
    status: string;
    imageUrl: string;
    baseImageUrl?: string;
    imageBase64?: string;
    overlayDetails: OverlayDetails;
    metadata?: Partial<ImageMetadata>;
}
