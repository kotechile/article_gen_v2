import React, { useState, useEffect } from 'react';
import { X, Loader2, CheckCircle2, AlertCircle, Globe, Calendar, FolderTree } from 'lucide-react';
import {
    fetchWordPressSites,
    fetchWordPressCategories,
    publishToWordPress,
    saveWordPressSettings,
    loadWordPressSettings,
    resolveLinkedWordPressCategoryIds
} from '../services/wordpressService';
import type { WordPressSite, WordPressCategory } from '../types/wordpress';

interface WordPressExportModalProps {
    articleData: any;
    articleId: string;
    userId: string;
    onClose: () => void;
    onSuccess: (postUrl: string) => void;
}

export const WordPressExportModal: React.FC<WordPressExportModalProps> = ({
    articleData,
    articleId,
    userId,
    onClose,
    onSuccess
}) => {
    // State management
    const [sites, setSites] = useState<WordPressSite[]>([]);
    const [categories, setCategories] = useState<WordPressCategory[]>([]);
    const [selectedSiteId, setSelectedSiteId] = useState<number | null>(null);
    const [postStatus, setPostStatus] = useState<'draft' | 'publish' | 'future'>('draft');
    const [scheduledDate, setScheduledDate] = useState<string>('');
    const [scheduledTime, setScheduledTime] = useState<string>('12:00');
    const [selectedCategoryIds, setSelectedCategoryIds] = useState<number[]>([]);

    // Loading states
    const [loadingSites, setLoadingSites] = useState(true);
    const [loadingCategories, setLoadingCategories] = useState(false);
    const [publishing, setPublishing] = useState(false);

    // Error state
    const [error, setError] = useState<string | null>(null);
    const [autoCategoryHint, setAutoCategoryHint] = useState<string | null>(null);

    // Load WordPress sites on mount
    useEffect(() => {
        const loadSites = async () => {
            try {
                setLoadingSites(true);
                const wpSites = await fetchWordPressSites(userId);
                setSites(wpSites);

                // Load saved settings
                const savedSettings = await loadWordPressSettings(articleId);
                if (savedSettings && savedSettings.siteId && wpSites.find(s => s.id === savedSettings.siteId)) {
                    setSelectedSiteId(savedSettings.siteId);
                    if (savedSettings.postStatus) {
                        setPostStatus(savedSettings.postStatus as 'draft' | 'publish' | 'future');
                    }
                    // Category will be set after categories load
                } else if (wpSites.length > 0) {
                    // Default to first site
                    setSelectedSiteId(wpSites[0].id);
                }
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Failed to load WordPress sites');
            } finally {
                setLoadingSites(false);
            }
        };

        loadSites();
    }, [userId, articleId]);

    // Load categories when site changes
    useEffect(() => {
        const loadCategories = async () => {
            if (!selectedSiteId) {
                setCategories([]);
                setAutoCategoryHint(null);
                return;
            }

            const selectedSite = sites.find(s => s.id === selectedSiteId);
            if (!selectedSite) return;

            try {
                setLoadingCategories(true);
                setError(null);
                const wpCategories = await fetchWordPressCategories(selectedSite);
                setCategories(wpCategories);
                setAutoCategoryHint(null);

                // Restore saved category if available
                const savedSettings = await loadWordPressSettings(articleId);
                if (savedSettings?.categoryId && wpCategories.find(c => c.id === savedSettings.categoryId)) {
                    setSelectedCategoryIds([savedSettings.categoryId]);
                    setAutoCategoryHint('Using your previously saved category selection.');
                    return;
                }

                // Fallback: auto-link from article topic -> research topic -> project categories synced to WordPress
                const linkedCategoryIds = await resolveLinkedWordPressCategoryIds(articleData, selectedSite.domain);
                const validLinkedIds = linkedCategoryIds.filter((id) => wpCategories.some((cat) => cat.id === id));
                if (validLinkedIds.length > 0) {
                    setSelectedCategoryIds(validLinkedIds);
                    setAutoCategoryHint('Auto-selected from linked project category/subcategory.');
                }
            } catch (err) {
                setError('Failed to load categories from WordPress. Please check your credentials.');
                setCategories([]);
                setAutoCategoryHint(null);
            } finally {
                setLoadingCategories(false);
            }
        };

        loadCategories();
    }, [selectedSiteId, sites, articleId, articleData]);

    // Handle category selection
    const handleCategoryToggle = (categoryId: number) => {
        setSelectedCategoryIds(prev => {
            if (prev.includes(categoryId)) {
                return prev.filter(id => id !== categoryId);
            } else {
                return [...prev, categoryId];
            }
        });
    };

    // Check if form is valid
    const isFormValid = () => {
        if (!selectedSiteId) return false;
        if (selectedCategoryIds.length === 0) return false;
        if (postStatus === 'future' && !scheduledDate) return false;
        return true;
    };

    // Handle publish
    const handlePublish = async () => {
        if (!isFormValid()) return;

        const selectedSite = sites.find(s => s.id === selectedSiteId);
        if (!selectedSite) return;

        try {
            setPublishing(true);
            setError(null);

            // Prepare scheduled date if needed
            let scheduledDateTime: Date | undefined;
            if (postStatus === 'future' && scheduledDate) {
                const [hours, minutes] = scheduledTime.split(':').map(Number);
                scheduledDateTime = new Date(scheduledDate);
                scheduledDateTime.setHours(hours, minutes, 0, 0);
            }

            // Publish to WordPress
            const result = await publishToWordPress(
                selectedSite,
                articleData,
                {
                    postStatus,
                    scheduledDate: scheduledDateTime,
                    categoryIds: selectedCategoryIds,
                    featuredImageUrl: articleData.featuredImageUrl || articleData.featuredimageurl,
                    featuredImageMetadata: {
                        alt: articleData.MediaAltText || articleData.mediaalttext,
                        title: articleData.mediaTitle || articleData.mediatitle,
                        caption: articleData.mediaCaption || articleData.mediacaption
                    }
                },
                {
                    focusKeyword: articleData.focus_keyword,
                    metaTitle: articleData.seo_title_optimized || articleData.metaTitle,
                    metaDescription: articleData.seo_meta_desc_optimized || articleData.metaDescription,
                    canonicalUrl: articleData.canonical_url,
                    robotsMeta: articleData.robots_meta,
                    schemaType: articleData.schema_type,
                    primaryKeywords: articleData.enhanced_primary_keywords || articleData.primary_keywords_json,
                    secondaryKeywords: articleData.enhanced_secondary_keywords || articleData.secondary_keywords_json,
                    readabilityScore: articleData.readability_score,
                    keywordDensity: articleData.keyword_density,
                    optimizationTips: articleData.content_optimization_tips
                }
            );

            // Save settings for next time
            await saveWordPressSettings(articleId, {
                siteId: selectedSiteId!, // Non-null assertion safe here since form validation checks this
                postStatus,
                categoryId: selectedCategoryIds[0]
            });

            // Show success and close
            onSuccess(result.link);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to publish to WordPress');
        } finally {
            setPublishing(false);
        }
    };

    // Get minimum date for scheduling (today)
    const getMinDate = () => {
        const today = new Date();
        return today.toISOString().split('T')[0];
    };

    return (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden flex flex-col">
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
                    <div className="flex-1 pr-4">
                        <h2 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                            <Globe className="w-6 h-6 text-indigo-600" />
                            Export to WordPress
                        </h2>
                        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1 truncate">
                            {articleData.Title || articleData.title || 'Untitled Article'}
                        </p>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Body */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    {/* Error message */}
                    {error && (
                        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-4 flex items-start gap-3">
                            <AlertCircle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
                            <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
                        </div>
                    )}

                    {/* Website Selection */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                            WordPress Website *
                        </label>
                        {loadingSites ? (
                            <div className="flex items-center justify-center py-8">
                                <Loader2 className="w-6 h-6 animate-spin text-indigo-600" />
                            </div>
                        ) : sites.length === 0 ? (
                            <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-xl p-4">
                                <p className="text-sm text-yellow-700 dark:text-yellow-300">
                                    No WordPress sites configured. Please add a WordPress site in your settings.
                                </p>
                            </div>
                        ) : (
                            <select
                                value={selectedSiteId || ''}
                                onChange={(e) => setSelectedSiteId(Number(e.target.value))}
                                className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                            >
                                <option value="">Select a website...</option>
                                {sites.map(site => (
                                    <option key={site.id} value={site.id}>
                                        {site.domain}
                                    </option>
                                ))}
                            </select>
                        )}
                    </div>

                    {/* Post Status */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                            Publication Status *
                        </label>
                        <select
                            value={postStatus}
                            onChange={(e) => setPostStatus(e.target.value as 'draft' | 'publish' | 'future')}
                            className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                        >
                            <option value="draft">Save as Draft</option>
                            <option value="publish">Publish Immediately</option>
                            <option value="future">Schedule for Later</option>
                        </select>
                    </div>

                    {/* Scheduling (conditional) */}
                    {postStatus === 'future' && (
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 flex items-center gap-2">
                                    <Calendar className="w-4 h-4" />
                                    Schedule Date *
                                </label>
                                <input
                                    type="date"
                                    value={scheduledDate}
                                    onChange={(e) => setScheduledDate(e.target.value)}
                                    min={getMinDate()}
                                    className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Time
                                </label>
                                <input
                                    type="time"
                                    value={scheduledTime}
                                    onChange={(e) => setScheduledTime(e.target.value)}
                                    className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                />
                            </div>
                        </div>
                    )}

                    {/* Category Selection */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 flex items-center gap-2">
                            <FolderTree className="w-4 h-4" />
                            Categories * (select at least one)
                        </label>
                        {autoCategoryHint && (
                            <p className="text-xs text-indigo-600 dark:text-indigo-300 mb-2">
                                {autoCategoryHint}
                            </p>
                        )}
                        {loadingCategories ? (
                            <div className="flex items-center justify-center py-8">
                                <Loader2 className="w-6 h-6 animate-spin text-indigo-600" />
                                <span className="ml-2 text-sm text-gray-500">Loading categories...</span>
                            </div>
                        ) : categories.length === 0 ? (
                            <div className="bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-xl p-4">
                                <p className="text-sm text-gray-500 dark:text-gray-400">
                                    {selectedSiteId ? 'No categories found or unable to connect to WordPress.' : 'Select a website first to load categories.'}
                                </p>
                            </div>
                        ) : (
                            <div className="border border-gray-200 dark:border-gray-700 rounded-xl overflow-hidden max-h-48 overflow-y-auto">
                                {categories.map(category => (
                                    <label
                                        key={category.id}
                                        className="flex items-center gap-3 px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-700/50 cursor-pointer border-b border-gray-100 dark:border-gray-700 last:border-b-0"
                                    >
                                        <input
                                            type="checkbox"
                                            checked={selectedCategoryIds.includes(category.id)}
                                            onChange={() => handleCategoryToggle(category.id)}
                                            className="w-4 h-4 text-indigo-600 border-gray-300 rounded focus:ring-indigo-500"
                                        />
                                        <span className="flex-1 text-sm text-gray-900 dark:text-white">
                                            {category.name}
                                        </span>
                                        <span className="text-xs text-gray-500 dark:text-gray-400">
                                            ({category.count || 0} posts)
                                        </span>
                                    </label>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* SEO Status Summary */}
                    <div className="bg-indigo-50 dark:bg-indigo-900/20 border border-indigo-200 dark:border-indigo-800 rounded-xl p-4">
                        <div className="flex items-start gap-3">
                            <CheckCircle2 className="w-5 h-5 text-indigo-600 dark:text-indigo-400 flex-shrink-0 mt-0.5" />
                            <div className="flex-1">
                                <h3 className="text-sm font-semibold text-indigo-900 dark:text-indigo-300 mb-2">
                                    SEO Metadata Ready
                                </h3>
                                <ul className="text-xs text-indigo-700 dark:text-indigo-400 space-y-1">
                                    <li>✓ Title, hook, and featured image</li>
                                    <li>✓ Meta title and description</li>
                                    <li>✓ Focus keywords and optimization data</li>
                                    <li>✓ Compatible with Yoast SEO & RankMath</li>
                                    <li className="flex items-center gap-1.5">
                                        {(articleData.Title || articleData.title || '').length <= 60 ? (
                                            <>
                                                <span>✓</span>
                                                <span>Title length optimized ({(articleData.Title || articleData.title || '').length}/60)</span>
                                            </>
                                        ) : (
                                            <>
                                                <AlertCircle className="w-3 h-3 text-amber-500" />
                                                <span className="text-amber-600 dark:text-amber-400">
                                                    Title exceeds 60 chars ({(articleData.Title || articleData.title || '').length}/60)
                                                </span>
                                            </>
                                        )}
                                    </li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Footer */}
                <div className="flex items-center justify-end gap-3 p-6 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50">
                    <button
                        onClick={onClose}
                        disabled={publishing}
                        className="px-6 py-2.5 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-xl font-medium transition disabled:opacity-50"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handlePublish}
                        disabled={!isFormValid() || publishing}
                        className="flex items-center gap-2 px-6 py-2.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl font-medium shadow-lg shadow-indigo-500/25 transition disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {publishing ? (
                            <>
                                <Loader2 className="w-4 h-4 animate-spin" />
                                <span>Publishing...</span>
                            </>
                        ) : (
                            <>
                                <Globe className="w-4 h-4" />
                                <span>Publish to WordPress</span>
                            </>
                        )}
                    </button>
                </div>
            </div>
        </div>
    );
};
