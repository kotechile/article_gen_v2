// WordPress-related type definitions

export type WordPressSeoPlugin = 'unknown' | 'yoast' | 'rankmath' | 'custom' | 'none';

export interface WordPressSite {
    id: number;
    created_at: string;
    user_id: string;
    domain: string;
    wordpress_key: string;
    app_name: string;
    wpUserName: string;
    websiteDescription?: string;
    targetAudienceDescription?: string;
    brand_primary_color?: string;
    brand_text_color?: string;
    brand_secondary_color?: string;
    brand_neutral_color?: string;
    seo_plugin?: WordPressSeoPlugin;
    cms_url?: string;
    site_url_override?: string;
    social_default_image_url?: string;
}

export interface WordPressCategory {
    id: number;
    name: string;
    slug: string;
    count?: number;
    description?: string;
    parent?: number;
}

export interface WordPressPublishSettings {
    siteId: number;
    postStatus: 'draft' | 'publish' | 'future';
    scheduledDate?: Date;
    categoryIds: number[];
}

export interface SEOMetadata {
    focusKeyword?: string;
    metaTitle?: string;
    metaDescription?: string;
    canonicalUrl?: string;
    robotsMeta?: string;
    schemaType?: string;
    breadcrumbTitle?: string;
    primaryKeywords?: string[];
    secondaryKeywords?: string[];
    readabilityScore?: number;
    keywordDensity?: number;
    internalLinks?: any[];
    externalLinks?: any[];
    optimizationTips?: string[];
    ogTitle?: string;
    ogDescription?: string;
    ogImageUrl?: string;
    ogType?: 'article' | 'website';
    twitterTitle?: string;
    twitterDescription?: string;
    twitterImageUrl?: string;
    twitterCardType?: 'summary' | 'summary_large_image';
}

export interface WordPressPostData {
    title: string;
    slug?: string;
    content: string;
    status: 'draft' | 'publish' | 'future';
    date?: string;
    excerpt?: string;
    categories?: number[];
    tags?: number[];
    featured_media?: number;
    meta?: {
        // Yoast SEO fields
        _yoast_wpseo_title?: string;
        _yoast_wpseo_metadesc?: string;
        _yoast_wpseo_focuskw?: string;
        _yoast_wpseo_canonical?: string;
        _yoast_wpseo_meta_robots_noindex?: number;
        _yoast_wpseo_meta_robots_nofollow?: number;
        _yoast_wpseo_primary_category?: number;
        _yoast_wpseo_content_score?: number;

        // RankMath fields
        rank_math_title?: string;
        rank_math_description?: string;
        rank_math_focus_keyword?: string;
        rank_math_canonical_url?: string;
        rank_math_robots?: string[];
        rank_math_primary_category?: number;
        rank_math_facebook_title?: string;
        rank_math_facebook_description?: string;
        rank_math_facebook_image?: string;
        rank_math_twitter_title?: string;
        rank_math_twitter_description?: string;
        rank_math_twitter_image?: string;
        rank_math_twitter_card_type?: string;

        // Custom SEO fields
        seo_focus_keyword?: string;
        seo_primary_keywords?: string;
        seo_secondary_keywords?: string;
        seo_readability_score?: number;
        seo_keyword_density?: number;
        seo_schema_type?: string;
        seo_optimization_tips?: string;
        seo_og_title?: string;
        seo_og_description?: string;
        seo_og_image?: string;
        seo_og_type?: string;
        seo_twitter_title?: string;
        seo_twitter_description?: string;
        seo_twitter_image?: string;
        seo_twitter_card?: string;
        '_yoast_wpseo_opengraph-title'?: string;
        '_yoast_wpseo_opengraph-description'?: string;
        '_yoast_wpseo_opengraph-image'?: string;
        '_yoast_wpseo_twitter-title'?: string;
        '_yoast_wpseo_twitter-description'?: string;
        '_yoast_wpseo_twitter-image'?: string;
        '_yoast_wpseo_twitter-card'?: string;
    };
}

export interface WordPressApiResponse {
    id: number;
    link: string;
    status: string;
    title: {
        rendered: string;
    };
    publish_warnings?: string[];
    loopback_summary?: {
        success: boolean;
        attemptedFields: string[];
        savedFields: string[];
        removedFields: string[];
        error?: string;
    };
}

export interface WordPressMediaResponse {
    id: number;
    source_url: string;
    alt_text?: string;
    caption?: {
        rendered: string;
    };
}
