import { supabase } from '../lib/supabase';
import type {
    WordPressSite,
    WordPressCategory,
    WordPressPostData,
    WordPressApiResponse,
    WordPressMediaResponse,
    SEOMetadata
} from '../types/wordpress';

/**
 * Fetch all WordPress sites configured for a user
 */
export const fetchWordPressSites = async (userId: string): Promise<WordPressSite[]> => {
    try {
        const { data, error } = await supabase
            .from('wordPress_details')
            .select('*')
            .eq('user_id', userId)
            .order('domain', { ascending: true });

        if (error) throw error;
        return data || [];
    } catch (error) {
        console.error('Error fetching WordPress sites:', error);
        throw new Error('Failed to load WordPress sites');
    }
};

/**
 * Fetch categories from a WordPress site via REST API
 */
export const fetchWordPressCategories = async (
    site: WordPressSite
): Promise<WordPressCategory[]> => {
    try {
        const categoriesUrl = `https://${site.domain}/wp-json/wp/v2/categories`;
        const credentials = btoa(`${site.wpUserName}:${site.wordpress_key}`);

        const response = await fetch(categoriesUrl, {
            method: 'GET',
            headers: {
                'Authorization': `Basic ${credentials}`,
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const categories = await response.json();
        return categories.map((cat: any) => ({
            id: cat.id,
            name: cat.name,
            slug: cat.slug,
            count: cat.count,
            description: cat.description,
            parent: cat.parent
        }));
    } catch (error) {
        console.error('Error fetching WordPress categories:', error);
        throw new Error('Failed to load categories from WordPress');
    }
};

/**
 * Upload featured image to WordPress media library
 */
export const uploadFeaturedImage = async (
    imageUrl: string,
    site: WordPressSite,
    metadata: { alt?: string; title?: string; caption?: string }
): Promise<number | null> => {
    try {
        // Fetch the image as a blob
        const imageResponse = await fetch(imageUrl);
        if (!imageResponse.ok) {
            console.warn('Failed to fetch featured image');
            return null;
        }

        const imageBlob = await imageResponse.blob();
        const credentials = btoa(`${site.wpUserName}:${site.wordpress_key}`);

        // Upload to WordPress media library
        const formData = new FormData();
        formData.append('file', imageBlob, 'featured-image.jpg');
        if (metadata.alt) formData.append('alt_text', metadata.alt);
        if (metadata.title) formData.append('title', metadata.title);
        if (metadata.caption) formData.append('caption', metadata.caption);

        const uploadResponse = await fetch(`https://${site.domain}/wp-json/wp/v2/media`, {
            method: 'POST',
            headers: {
                'Authorization': `Basic ${credentials}`
            },
            body: formData
        });

        if (!uploadResponse.ok) {
            console.warn('Failed to upload featured image to WordPress');
            return null;
        }

        const mediaData: WordPressMediaResponse = await uploadResponse.json();
        return mediaData.id;
    } catch (error) {
        console.error('Error uploading featured image:', error);
        return null;
    }
};

/**
 * Build SEO metadata for WordPress post
 */
const buildSEOMetadata = (articleData: any, seoData: SEOMetadata, categoryId?: number) => {
    const metaTitle = seoData.metaTitle || articleData.metaTitle || articleData.Title || articleData.title;
    const metaDescription = seoData.metaDescription || articleData.seo_meta_desc_optimized || articleData.thesis || articleData.Thesis;
    const focusKeyword = seoData.focusKeyword || articleData.focus_keyword || '';
    const canonicalUrl = seoData.canonicalUrl || articleData.canonical_url || '';
    const robotsMeta = seoData.robotsMeta || articleData.robots_meta || 'index,follow';
    const schemaType = seoData.schemaType || articleData.schema_type || 'Article';

    const meta: any = {
        // Yoast SEO fields
        _yoast_wpseo_title: metaTitle,
        _yoast_wpseo_metadesc: metaDescription,
        _yoast_wpseo_focuskw: focusKeyword,
        _yoast_wpseo_canonical: canonicalUrl,
        _yoast_wpseo_meta_robots_noindex: robotsMeta.includes('noindex') ? 1 : 0,
        _yoast_wpseo_meta_robots_nofollow: robotsMeta.includes('nofollow') ? 1 : 0,

        // RankMath fields
        rank_math_title: metaTitle,
        rank_math_description: metaDescription,
        rank_math_focus_keyword: focusKeyword,
        rank_math_canonical_url: canonicalUrl,
        rank_math_robots: robotsMeta.split(','),

        // Custom SEO fields
        seo_focus_keyword: focusKeyword,
        seo_schema_type: schemaType,
        seo_readability_score: seoData.readabilityScore || articleData.readability_score || 0,
        seo_keyword_density: seoData.keywordDensity || articleData.keyword_density || 0
    };

    // Add primary category if available
    if (categoryId) {
        meta._yoast_wpseo_primary_category = categoryId;
        meta.rank_math_primary_category = categoryId;
    }

    // Add keyword lists
    if (seoData.primaryKeywords || articleData.enhanced_primary_keywords) {
        const keywords = seoData.primaryKeywords || articleData.enhanced_primary_keywords || [];
        meta.seo_primary_keywords = Array.isArray(keywords) ? keywords.join(',') : keywords;
    }

    if (seoData.secondaryKeywords || articleData.enhanced_secondary_keywords) {
        const keywords = seoData.secondaryKeywords || articleData.enhanced_secondary_keywords || [];
        meta.seo_secondary_keywords = Array.isArray(keywords) ? keywords.join(',') : keywords;
    }

    // Add optimization tips
    if (seoData.optimizationTips || articleData.content_optimization_tips) {
        const tips = seoData.optimizationTips || articleData.content_optimization_tips || [];
        meta.seo_optimization_tips = Array.isArray(tips) ? tips.join('; ') : tips;
    }

    return meta;
};

/**
 * Formats the article body to include metadata elements (Deck, Hook, Thesis, Image)
 * logically integrated into the text with professional styling.
 */
const formatArticleBody = (articleData: any): string => {
    let content = '';

    // 1. Featured Image - REMOVED to avoid duplication (WP Theme handles this via featured_media ID)

    // 2. The Deck (Lead/Intro text) - Italicized, larger font
    if (articleData.deck) {
        content += `
            <div class="article-deck" style="font-size: 1.25em; line-height: 1.6; color: #4b5563; font-style: italic; margin-bottom: 2em; border-left: 4px solid #4f46e5; padding-left: 1em;">
                ${articleData.deck}
            </div>`;
    }

    // 3. Hook & Thesis (Context Setting)
    if (articleData.hook || articleData.thesis) {
        content += `<div class="article-context" style="background-color: #f9fafb; border-radius: 0.75rem; padding: 1.5em; margin-bottom: 2em; border: 1px solid #e5e7eb;">`;

        if (articleData.hook) {
            content += `<p style="margin-bottom: 1em; font-size: 1.1em;">${articleData.hook}</p>`;
        }

        if (articleData.thesis) {
            content += `
                <div style="display: flex; gap: 0.75em; align-items: flex-start;">
                    <span style="font-size: 1.25em;">💡</span>
                    <p style="margin: 0; font-weight: 500; color: #1f2937;"><strong>Key Takeaway:</strong> ${articleData.thesis}</p>
                </div>`;
        }

        content += `</div>`;
    }

    // 4. Main Body Content
    content += articleData.htmlArticle || articleData.htmlarticle || '';

    return content;
};



/**
 * Generate SEO-friendly slug respecting length limits
 */
const generateSlug = (title: string, maxLength: number): string => {
    let slug = title
        .toLowerCase()
        .replace(/[^a-z0-9\s-]/g, '') // Remove non-alphanumeric chars
        .trim()
        .replace(/\s+/g, '-'); // Replace spaces with hyphens

    if (slug.length <= maxLength) return slug;

    // Truncate to maxLength, trying not to break words
    const truncated = slug.substring(0, maxLength);
    const lastHyphen = truncated.lastIndexOf('-');

    // If there's a hyphen reasonably close to the end, cut there
    if (lastHyphen > maxLength * 0.5) {
        return truncated.substring(0, lastHyphen);
    }

    return truncated;
};

/**
 * Publish article to WordPress
 */
export const publishToWordPress = async (
    site: WordPressSite,
    articleData: any,
    settings: {
        postStatus: 'draft' | 'publish' | 'future';
        scheduledDate?: Date;
        categoryIds: number[];
        featuredImageUrl?: string;
        featuredImageMetadata?: { alt?: string; title?: string; caption?: string };
    },
    seoData: SEOMetadata = {}
): Promise<WordPressApiResponse> => {
    try {
        const credentials = btoa(`${site.wpUserName}:${site.wordpress_key}`);

        // Upload featured image if provided
        let featuredMediaId: number | null = null;
        if (settings.featuredImageUrl) {
            featuredMediaId = await uploadFeaturedImage(
                settings.featuredImageUrl,
                site,
                settings.featuredImageMetadata || {}
            );
        }

        // Prepare post data
        const styledContent = formatArticleBody(articleData);

        // Calculate max slug length for 90-char URL limit
        // URL = https://domain.com/slug
        const domainLength = site.domain.length;
        const protocolLength = 8; // https://
        const buffer = 2; // Extra safety
        const baseLength = protocolLength + domainLength + 1; // +1 for slash
        const maxSlugLength = Math.max(10, 90 - baseLength - buffer);

        const slug = generateSlug(articleData.Title || articleData.title || '', maxSlugLength);
        console.log(`Generated slug: '${slug}' (max len: ${maxSlugLength}) for domain: ${site.domain}`);

        const postData: WordPressPostData = {
            title: articleData.Title || articleData.title || '',
            slug: slug,
            content: styledContent,
            status: settings.postStatus,
            excerpt: articleData.excerpt || articleData.hook || articleData.Hook || '',
            categories: settings.categoryIds,
            meta: buildSEOMetadata(articleData, seoData, settings.categoryIds[0])
        };

        // Add scheduled date if publishing in the future
        if (settings.postStatus === 'future' && settings.scheduledDate) {
            postData.date = settings.scheduledDate.toISOString();
        }

        // Add featured image if uploaded successfully
        if (featuredMediaId) {
            postData.featured_media = featuredMediaId;
        }

        // Publish to WordPress
        const response = await fetch(`https://${site.domain}/wp-json/wp/v2/posts`, {
            method: 'POST',
            headers: {
                'Authorization': `Basic ${credentials}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(postData)
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`WordPress API error (${response.status}): ${errorText}`);
        }

        const result: WordPressApiResponse = await response.json();

        // Update Supabase status
        if (articleData.id) {
            const newStatus = settings.postStatus === 'future' ? 'Scheduled' : 'Published';
            await supabase
                .from('Titles')
                .update({
                    status: newStatus,
                    published: true
                })
                .eq('id', articleData.id);
        }

        console.log('✅ Article published to WordPress:', {
            id: result.id,
            url: result.link,
            status: result.status
        });

        return result;
    } catch (error) {
        console.error('Error publishing to WordPress:', error);
        throw error;
    }
};

/**
 * Save WordPress export settings for future use
 */
export const saveWordPressSettings = async (
    titleId: string,
    settings: {
        siteId: number;
        postStatus: string;
        categoryId?: number;
    }
): Promise<void> => {
    try {
        const { error } = await supabase
            .from('Titles')
            .update({
                last_wp_site_id: settings.siteId,
                last_wp_post_status: settings.postStatus,
                last_wp_category_id: settings.categoryId || null
            })
            .eq('id', titleId);

        if (error) throw error;
    } catch (error) {
        console.error('Error saving WordPress settings:', error);
        // Non-critical error, don't throw
    }
};

/**
 * Load previously saved WordPress export settings
 */
export const loadWordPressSettings = async (
    titleId: string
): Promise<{ siteId?: number; postStatus?: string; categoryId?: number } | null> => {
    try {
        const { data, error } = await supabase
            .from('Titles')
            .select('last_wp_site_id, last_wp_post_status, last_wp_category_id')
            .eq('id', titleId)
            .single();

        if (error) throw error;

        return {
            siteId: data?.last_wp_site_id,
            postStatus: data?.last_wp_post_status,
            categoryId: data?.last_wp_category_id
        };
    } catch (error) {
        console.error('Error loading WordPress settings:', error);
        return null;
    }
};
