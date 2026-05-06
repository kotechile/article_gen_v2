import React, { useState, useEffect, useMemo } from 'react';
import { X, Loader2, CheckCircle2, AlertCircle, Globe, Calendar, FolderTree, ShieldCheck, AlertTriangle } from 'lucide-react';
import {
    fetchWordPressSites,
    fetchWordPressCategories,
    publishToWordPress,
    saveWordPressSettings,
    loadWordPressSettings,
    resolveLinkedWordPressCategoryIds
} from '../services/wordpressService';
import type { WordPressSite, WordPressCategory } from '../types/wordpress';
import { computeSEOQualityScore } from '../utils/seoUtils';

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
    const [loopbackBanner, setLoopbackBanner] = useState<{
        type: 'success' | 'warning';
        message: string;
    } | null>(null);

    // GEO/SEO quality report (computed once from article data)
    const seoReport = useMemo(() => computeSEOQualityScore(articleData), [articleData]);

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

                const candidateIds: number[] = [];

                const directCategoryIds = [
                    articleData?.wordpress_category_id,
                    articleData?.wordpress_parent_category_id,
                ]
                    .map((value) => Number(value))
                    .filter((value) => Number.isFinite(value))
                    .filter((value) => wpCategories.some((cat) => cat.id === value));
                candidateIds.push(...directCategoryIds);

                const savedSettings = await loadWordPressSettings(articleId);
                if (savedSettings?.categoryId && wpCategories.find(c => c.id === savedSettings.categoryId)) {
                    candidateIds.push(savedSettings.categoryId);
                }

                const linkedCategoryIds = await resolveLinkedWordPressCategoryIds(articleData, selectedSite.domain);
                const validLinkedIds = linkedCategoryIds.filter((id) => wpCategories.some((cat) => cat.id === id));
                candidateIds.push(...validLinkedIds);

                const mergedIds = Array.from(new Set(candidateIds)).filter((id) =>
                    wpCategories.some((cat) => cat.id === id)
                );

                if (mergedIds.length > 0) {
                    setSelectedCategoryIds(mergedIds);
                    if (validLinkedIds.length > 1) {
                        setAutoCategoryHint('Auto-selected from linked project category and subcategory.');
                    } else if (validLinkedIds.length === 1) {
                        setAutoCategoryHint('Auto-selected from linked project category/subcategory.');
                    } else if (directCategoryIds.length > 0) {
                        setAutoCategoryHint('Auto-selected from article\'s WordPress category IDs.');
                    } else if (savedSettings?.categoryId) {
                        setAutoCategoryHint('Using your previously saved category selection.');
                    }
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
                    featuredImageUrl:
                        articleData.featuredImageUrl ||
                        articleData.featuredImageURL ||
                        articleData.featuredimageurl,
                    featuredImageMetadata: {
                        alt: articleData.mediaAltText || articleData.MediaAltText || articleData.mediaalttext,
                        title: articleData.mediaTitle || articleData.MediaTitle || articleData.mediatitle,
                        caption: articleData.mediaCaption || articleData.MediaCaption || articleData.mediacaption
                    }
                },
                {
                    focusKeyword: articleData.focus_keyword ?? articleData.primary_keyword ?? articleData.primary_keywords?.[0] ?? articleData.search_phrase,
                    metaTitle: articleData.seo_title_optimized || articleData.metaTitle,
                    metaDescription: articleData.seo_meta_desc_optimized || articleData.metaDescription,
                    canonicalUrl: articleData.canonical_url,
                    robotsMeta: articleData.robots_meta,
                    schemaType: articleData.schema_type,
                    // Use canonical fields first, fall back to legacy enhanced/json variants
                    primaryKeywords: articleData.primary_keywords ?? articleData.enhanced_primary_keywords ?? articleData.primary_keywords_json,
                    secondaryKeywords: articleData.secondary_keywords ?? articleData.enhanced_secondary_keywords ?? articleData.secondary_keywords_json,
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

            const loopbackSummary = result.loopback_summary;
            const publishWarnings = result.publish_warnings || [];
            if (publishWarnings.length > 0 && loopbackSummary?.success) {
                const savedCount = loopbackSummary.savedFields.length;
                setLoopbackBanner({
                    type: 'warning',
                    message: `Published and synced ${savedCount} Titles fields. ${publishWarnings.join(' ')}`,
                });
            } else if (publishWarnings.length > 0 && loopbackSummary) {
                const removedCount = loopbackSummary.removedFields.length;
                setLoopbackBanner({
                    type: 'warning',
                    message: removedCount > 0
                        ? `Published, but ${removedCount} loopback fields were skipped due to schema mismatch. ${publishWarnings.join(' ')}`
                        : `Published, but Titles loopback update had issues. ${publishWarnings.join(' ')}`,
                });
            } else if (publishWarnings.length > 0) {
                setLoopbackBanner({
                    type: 'warning',
                    message: `Published, but ${publishWarnings.join(' ')}`,
                });
            } else if (loopbackSummary?.success) {
                const savedCount = loopbackSummary.savedFields.length;
                setLoopbackBanner({
                    type: 'success',
                    message: `Published and synced ${savedCount} Titles fields.`,
                });
            } else if (loopbackSummary) {
                const removedCount = loopbackSummary.removedFields.length;
                setLoopbackBanner({
                    type: 'warning',
                    message: removedCount > 0
                        ? `Published, but ${removedCount} loopback fields were skipped due to schema mismatch.`
                        : 'Published, but Titles loopback update had issues.',
                });
            } else {
                setLoopbackBanner({
                    type: 'success',
                    message: 'Published successfully.',
                });
            }

            // Give user a brief moment to read the verification banner.
            setTimeout(() => onSuccess(result.link), 1400);
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

                    {loopbackBanner && (
                        <div
                            className={`rounded-xl p-3 border flex items-start gap-2 ${
                                loopbackBanner.type === 'success'
                                    ? 'bg-emerald-50 dark:bg-emerald-900/20 border-emerald-200 dark:border-emerald-800'
                                    : 'bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800'
                            }`}
                        >
                            {loopbackBanner.type === 'success' ? (
                                <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400 mt-0.5 flex-shrink-0" />
                            ) : (
                                <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400 mt-0.5 flex-shrink-0" />
                            )}
                            <p
                                className={`text-xs ${
                                    loopbackBanner.type === 'success'
                                        ? 'text-emerald-700 dark:text-emerald-300'
                                        : 'text-amber-700 dark:text-amber-300'
                                }`}
                            >
                                {loopbackBanner.message}
                            </p>
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

                    {/* GEO/SEO Quality Gate */}
                    {(() => {
                        const report = seoReport;
                        const gradeColor = {
                            A: 'text-emerald-500', B: 'text-sky-400', C: 'text-amber-400',
                            D: 'text-orange-500', F: 'text-destructive'
                        }[report.grade];
                        const borderColor = report.score < 40
                            ? 'border-destructive/40 bg-destructive/5'
                            : report.score < 60
                                ? 'border-amber-500/40 bg-amber-500/5'
                                : 'border-emerald-500/30 bg-emerald-500/5';
                        return (
                            <div className={`border rounded-xl p-4 ${borderColor}`}>
                                <div className="flex items-center justify-between mb-3">
                                    <div className="flex items-center gap-2">
                                        {report.score < 40
                                            ? <AlertTriangle className="w-4 h-4 text-destructive" />
                                            : <ShieldCheck className="w-4 h-4 text-emerald-500" />
                                        }
                                        <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
                                            SEO + GEO Quality Gate
                                        </h3>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <span className={`text-2xl font-bold ${gradeColor}`}>{report.grade}</span>
                                        <span className="text-xs text-gray-500 dark:text-gray-400">{report.score}/100</span>
                                    </div>
                                </div>
                                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5 mb-3">
                                    <div
                                        className={`h-1.5 rounded-full transition-all ${
                                            report.score >= 75 ? 'bg-emerald-500'
                                            : report.score >= 60 ? 'bg-sky-400'
                                            : report.score >= 40 ? 'bg-amber-400'
                                            : 'bg-destructive'
                                        }`}
                                        style={{ width: `${report.score}%` }}
                                    />
                                </div>
                                <ul className="space-y-1">
                                    {report.checks.map((check, i) => (
                                        <li key={i} className="flex items-center gap-2 text-xs">
                                            {check.passed
                                                ? <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 flex-shrink-0" />
                                                : <AlertCircle className="w-3.5 h-3.5 text-amber-500 flex-shrink-0" />
                                            }
                                            <span className={check.passed ? 'text-gray-600 dark:text-gray-300' : 'text-amber-700 dark:text-amber-300'}>
                                                {check.label}
                                            </span>
                                        </li>
                                    ))}
                                </ul>
                                {report.score < 40 && (
                                    <p className="mt-3 text-xs text-destructive font-medium">
                                        ⚠ Publication blocked — SEO/GEO score too low ({report.score}/100). Set a primary keyword and ensure it appears in the title.
                                    </p>
                                )}
                                {report.score >= 40 && report.score < 60 && (
                                    <p className="mt-3 text-xs text-amber-700 dark:text-amber-300">
                                        ⚠ Score below recommended threshold ({report.score}/100). One secondary keyword is enough for generation, but this GEO export check expects 3+ secondary keywords or research mode for stronger AI citation-readiness.
                                    </p>
                                )}
                            </div>
                        );
                    })()}
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
                        disabled={!isFormValid() || publishing || !seoReport.canPublish}
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
