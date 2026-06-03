import React, { useState, useEffect } from 'react';
import { X, Loader2, CheckCircle2, AlertCircle, Globe, FolderTree } from 'lucide-react';
import { fetchWordPressSites, fetchWordPressCategories } from '../services/wordpressService';
import type { WordPressSite, WordPressCategory } from '../types/wordpress';

interface CategoryMappingModalProps {
    isOpen: boolean;
    onClose: () => void;
    userId: string;
    currentDomain?: string;
    currentCategoryId?: number;
    currentParentCategoryId?: number;
    onSave: (domain: string, categoryId: number, parentCategoryId: number | null) => Promise<void>;
}

export const CategoryMappingModal: React.FC<CategoryMappingModalProps> = ({
    isOpen,
    onClose,
    userId,
    currentDomain,
    currentCategoryId,
    currentParentCategoryId,
    onSave
}) => {
    const [sites, setSites] = useState<WordPressSite[]>([]);
    const [selectedSiteId, setSelectedSiteId] = useState<number | null>(null);
    const [categories, setCategories] = useState<WordPressCategory[]>([]);
    
    // Dropdown selections
    const [selectedCategoryVal, setSelectedCategoryVal] = useState<number | null>(null);
    const [selectedSubCategoryVal, setSelectedSubCategoryVal] = useState<number | null>(null);

    // Loaders and errors
    const [loadingSites, setLoadingSites] = useState(true);
    const [loadingCategories, setLoadingCategories] = useState(false);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState(false);

    // Initial site list fetch
    useEffect(() => {
        if (!isOpen) return;

        const loadSites = async () => {
            try {
                setLoadingSites(true);
                setError(null);
                const wpSites = await fetchWordPressSites(userId);
                setSites(wpSites);

                // Find matching site based on domain
                if (currentDomain && wpSites.length > 0) {
                    const match = wpSites.find(
                        s => s.domain.toLowerCase().replace(/^https?:\/\//, '').replace(/\/$/, '') === 
                             currentDomain.toLowerCase().replace(/^https?:\/\//, '').replace(/\/$/, '')
                    );
                    if (match) {
                        setSelectedSiteId(match.id);
                    } else {
                        setSelectedSiteId(wpSites[0].id);
                    }
                } else if (wpSites.length > 0) {
                    setSelectedSiteId(wpSites[0].id);
                }
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Failed to load WordPress sites');
            } finally {
                setLoadingSites(false);
            }
        };

        loadSites();
    }, [isOpen, userId, currentDomain]);

    // Fetch categories when site changes
    useEffect(() => {
        if (!selectedSiteId) {
            setCategories([]);
            return;
        }

        const loadCategories = async () => {
            const selectedSite = sites.find(s => s.id === selectedSiteId);
            if (!selectedSite) return;

            try {
                setLoadingCategories(true);
                setError(null);
                const wpCategories = await fetchWordPressCategories(selectedSite);
                setCategories(wpCategories);

                // Determine category/subcategory structure from currentCategoryId & currentParentCategoryId
                if (currentCategoryId) {
                    const targetCategory = wpCategories.find(c => c.id === currentCategoryId);
                    if (targetCategory) {
                        if (targetCategory.parent && targetCategory.parent !== 0) {
                            // The current category IS a subcategory
                            setSelectedCategoryVal(targetCategory.parent);
                            setSelectedSubCategoryVal(targetCategory.id);
                        } else {
                            // The current category is a parent category
                            setSelectedCategoryVal(targetCategory.id);
                            setSelectedSubCategoryVal(null);
                        }
                    } else if (currentParentCategoryId) {
                        // Fallback using currentParentCategoryId
                        setSelectedCategoryVal(currentParentCategoryId);
                        setSelectedSubCategoryVal(currentCategoryId);
                    } else {
                        setSelectedCategoryVal(currentCategoryId);
                        setSelectedSubCategoryVal(null);
                    }
                } else {
                    setSelectedCategoryVal(null);
                    setSelectedSubCategoryVal(null);
                }
            } catch (err) {
                setError('Failed to load categories from WordPress. Please check credentials.');
                setCategories([]);
            } finally {
                setLoadingCategories(false);
            }
        };

        loadCategories();
    }, [selectedSiteId, sites, currentCategoryId, currentParentCategoryId]);

    // Filter parent categories (parent is 0 or undefined)
    const parentCategories = categories.filter(c => !c.parent || c.parent === 0);

    // Filter subcategories for the selected parent category
    const subCategories = selectedCategoryVal 
        ? categories.filter(c => c.parent === selectedCategoryVal) 
        : [];

    const handleCategoryChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const val = e.target.value ? Number(e.target.value) : null;
        setSelectedCategoryVal(val);
        setSelectedSubCategoryVal(null); // Reset subcategory when category changes
    };

    const handleSubCategoryChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const val = e.target.value ? Number(e.target.value) : null;
        setSelectedSubCategoryVal(val);
    };

    const handleSaveSubmit = async () => {
        const selectedSite = sites.find(s => s.id === selectedSiteId);
        if (!selectedSite) {
            setError('Please select a valid WordPress website.');
            return;
        }

        if (!selectedCategoryVal) {
            setError('Please select a WordPress category.');
            return;
        }

        try {
            setSaving(true);
            setError(null);
            
            // If subcategory is selected, wordpress_category_id is the subcategory ID,
            // and wordpress_parent_category_id is the parent category ID.
            // If no subcategory is selected, wordpress_category_id is the parent category ID,
            // and wordpress_parent_category_id is null.
            const finalCategoryId = selectedSubCategoryVal ? selectedSubCategoryVal : selectedCategoryVal;
            const finalParentCategoryId = selectedSubCategoryVal ? selectedCategoryVal : null;

            await onSave(selectedSite.domain, finalCategoryId, finalParentCategoryId);
            setSuccess(true);
            setTimeout(() => {
                setSuccess(false);
                onClose();
            }, 1000);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to save mapping');
        } finally {
            setSaving(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-in fade-in duration-200">
            <div className="bg-gray-900 border border-gray-800 rounded-2xl shadow-2xl max-w-md w-full overflow-hidden flex flex-col animate-in zoom-in-95 duration-200">
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-gray-800">
                    <div className="flex items-center gap-2">
                        <FolderTree className="w-5 h-5 text-indigo-400" />
                        <h2 className="text-xl font-bold text-white">Map Category & Domain</h2>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-1.5 hover:bg-gray-800 text-gray-400 hover:text-white rounded-lg transition-colors"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Content */}
                <div className="p-6 space-y-5 flex-1">
                    {error && (
                        <div className="bg-red-500/10 border border-red-500/20 text-red-400 px-4 py-3 rounded-xl flex items-start gap-2.5 text-sm">
                            <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                            <span>{error}</span>
                        </div>
                    )}

                    {success && (
                        <div className="bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 px-4 py-3 rounded-xl flex items-start gap-2.5 text-sm">
                            <CheckCircle2 className="w-4 h-4 mt-0.5 flex-shrink-0" />
                            <span>WordPress category mapping saved successfully!</span>
                        </div>
                    )}

                    {loadingSites ? (
                        <div className="flex flex-col items-center justify-center py-8 space-y-3">
                            <Loader2 className="w-8 h-8 animate-spin text-indigo-400" />
                            <span className="text-sm text-gray-400">Loading website connections...</span>
                        </div>
                    ) : (
                        <div className="space-y-4">
                            {/* Website Selection */}
                            <div className="space-y-2">
                                <label className="flex items-center gap-1.5 text-sm font-semibold text-gray-300">
                                    <Globe className="w-4 h-4 text-gray-400" />
                                    WordPress Website (Project)
                                </label>
                                <select
                                    value={selectedSiteId || ''}
                                    onChange={(e) => setSelectedSiteId(Number(e.target.value))}
                                    className="w-full bg-gray-950 border border-gray-800 text-white rounded-xl px-3 py-2.5 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none transition text-sm"
                                >
                                    {sites.length === 0 ? (
                                        <option value="" disabled>No WordPress sites connected</option>
                                    ) : (
                                        sites.map(site => (
                                            <option key={site.id} value={site.id}>
                                                {site.app_name || site.domain} ({site.domain})
                                            </option>
                                        ))
                                    )}
                                </select>
                            </div>

                            {/* Category Selection */}
                            {selectedSiteId && (
                                <div className="space-y-4">
                                    {loadingCategories ? (
                                        <div className="flex items-center gap-2 py-4 text-gray-400 text-sm">
                                            <Loader2 className="w-4 h-4 animate-spin text-indigo-400" />
                                            <span>Loading categories...</span>
                                        </div>
                                    ) : (
                                        <>
                                            {/* Parent Category */}
                                            <div className="space-y-2">
                                                <label className="text-sm font-semibold text-gray-300 block">
                                                    WordPress Category
                                                </label>
                                                <select
                                                    value={selectedCategoryVal || ''}
                                                    onChange={handleCategoryChange}
                                                    className="w-full bg-gray-950 border border-gray-800 text-white rounded-xl px-3 py-2.5 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none transition text-sm"
                                                >
                                                    <option value="">-- Select Category --</option>
                                                    {parentCategories.map(cat => (
                                                        <option key={cat.id} value={cat.id}>
                                                            {cat.name}
                                                        </option>
                                                    ))}
                                                </select>
                                            </div>

                                            {/* Sub-category (Optional) */}
                                            {selectedCategoryVal && (
                                                <div className="space-y-2 animate-in slide-in-from-top-1 duration-150">
                                                    <label className="text-sm font-semibold text-gray-300 block">
                                                        Sub-Category (Optional)
                                                    </label>
                                                    <select
                                                        value={selectedSubCategoryVal || ''}
                                                        onChange={handleSubCategoryChange}
                                                        className="w-full bg-gray-950 border border-gray-800 text-white rounded-xl px-3 py-2.5 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none transition text-sm"
                                                    >
                                                        <option value="">-- No Sub-Category --</option>
                                                        {subCategories.map(cat => (
                                                            <option key={cat.id} value={cat.id}>
                                                                {cat.name}
                                                            </option>
                                                        ))}
                                                    </select>
                                                </div>
                                            )}
                                        </>
                                    )}
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="p-6 border-t border-gray-800 flex justify-end gap-3 bg-gray-950/40">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 border border-gray-800 hover:bg-gray-800 text-gray-300 rounded-xl transition text-sm"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleSaveSubmit}
                        disabled={saving || loadingSites || loadingCategories || !selectedSiteId || !selectedCategoryVal}
                        className="flex items-center gap-2 px-5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl font-medium transition disabled:opacity-50 disabled:cursor-not-allowed text-sm shadow-lg shadow-indigo-600/10"
                    >
                        {saving && <Loader2 className="w-4 h-4 animate-spin" />}
                        Save Mapping
                    </button>
                </div>
            </div>
        </div>
    );
};
