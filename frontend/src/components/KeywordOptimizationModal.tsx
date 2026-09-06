import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    X,
    Search,
    Loader2,
    Sparkles,
    CheckCircle2,
    AlertCircle,
    TrendingUp,
    Star,
    Target,
    Plus,
    Check,
    ArrowRight,
    SlidersHorizontal,
    FileText,
    Diff,
} from 'lucide-react';
import {
    keywordOptimizationService,
    type KeywordCandidate,
    type WeaveKeywordsResponse,
} from '../services/keyword-optimization.service';

interface KeywordOptimizationModalProps {
    isOpen: boolean;
    onClose: () => void;
    titleId?: string;
    articleTitle: string;
    articleContent: string;
    initialPrimaryKeyword?: string;
    initialSecondaryKeywords?: string[];
    onApplyKeywords: (data: {
        primaryKeyword: string;
        secondaryKeywords: string[];
        primaryMetric?: KeywordCandidate;
        updatedHtml?: string;
    }) => Promise<void> | void;
}

export const KeywordOptimizationModal: React.FC<KeywordOptimizationModalProps> = ({
    isOpen,
    onClose,
    titleId,
    articleTitle,
    articleContent,
    initialPrimaryKeyword = '',
    initialSecondaryKeywords = [],
    onApplyKeywords,
}) => {
    const [searchQuery, setSearchQuery] = useState('');
    const [suggestedSeeds, setSuggestedSeeds] = useState<string[]>([]);
    const [keywords, setKeywords] = useState<KeywordCandidate[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Selected Keywords
    const [selectedPrimary, setSelectedPrimary] = useState<string>(initialPrimaryKeyword);
    const [selectedSecondaries, setSelectedSecondaries] = useState<string[]>(initialSecondaryKeywords);
    const [selectedPrimaryMetric, setSelectedPrimaryMetric] = useState<KeywordCandidate | undefined>(undefined);

    // Weaving State
    const [weaving, setWeaving] = useState(false);
    const [weaveResult, setWeaveResult] = useState<WeaveKeywordsResponse | null>(null);
    const [activeTab, setActiveTab] = useState<'all' | 'quick-wins' | 'high-volume'>('all');
    const [viewMode, setViewMode] = useState<'discover' | 'preview-diff'>('discover');
    const [saving, setSaving] = useState(false);

    useEffect(() => {
        if (isOpen) {
            setSelectedPrimary(initialPrimaryKeyword);
            setSelectedSecondaries(initialSecondaryKeywords);
            setViewMode('discover');
            setWeaveResult(null);
            setError(null);
            // Initial auto-discovery without sending raw long title as custom_seed
            void handleDiscover();
        }
    }, [isOpen, articleTitle, initialPrimaryKeyword]);

    const handleDiscover = async (seed?: string) => {
        setLoading(true);
        setError(null);
        try {
            const result = await keywordOptimizationService.discoverKeywords({
                title: articleTitle,
                content: articleContent,
                custom_seed: seed || undefined,
            });

            const kwList = Array.isArray(result) ? result : (result?.keywords || []);
            const seedList = (!Array.isArray(result) && result?.seeds) ? result.seeds : [];

            setKeywords(kwList);
            if (seedList.length > 0) {
                setSuggestedSeeds(seedList);
                if (!searchQuery || searchQuery === articleTitle || (seed && seed === seedList[0])) {
                    setSearchQuery(seed || seedList[0]);
                }
            } else if (seed) {
                setSearchQuery(seed);
            }

            // Match initial primary metric if exists
            if (initialPrimaryKeyword) {
                const match = kwList.find((k) => k.keyword.toLowerCase() === initialPrimaryKeyword.toLowerCase());
                if (match) setSelectedPrimaryMetric(match);
            }
        } catch (err: any) {
            const msg = err.response?.data?.error || err.message || 'Failed to fetch DataForSEO keyword suggestions.';
            setError(msg);
        } finally {
            setLoading(false);
        }
    };

    const handleSearchSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!searchQuery.trim()) return;
        void handleDiscover(searchQuery.trim());
    };

    const handleSelectPrimary = (kw: KeywordCandidate) => {
        if (selectedPrimary.toLowerCase() === kw.keyword.toLowerCase()) {
            setSelectedPrimary('');
            setSelectedPrimaryMetric(undefined);
        } else {
            setSelectedPrimary(kw.keyword);
            setSelectedPrimaryMetric(kw);
            // Remove from secondaries if present
            setSelectedSecondaries((prev) => prev.filter((s) => s.toLowerCase() !== kw.keyword.toLowerCase()));
        }
    };

    const handleToggleSecondary = (kw: KeywordCandidate) => {
        const kwLower = kw.keyword.toLowerCase();
        if (selectedSecondaries.some((s) => s.toLowerCase() === kwLower)) {
            setSelectedSecondaries((prev) => prev.filter((s) => s.toLowerCase() !== kwLower));
        } else {
            // Cannot be secondary if already primary
            if (selectedPrimary.toLowerCase() === kwLower) {
                setSelectedPrimary('');
                setSelectedPrimaryMetric(undefined);
            }
            if (selectedSecondaries.length >= 6) {
                setError('You can select a maximum of 6 secondary keywords.');
                return;
            }
            setSelectedSecondaries((prev) => [...prev, kw.keyword]);
        }
    };

    const handleTriggerWeaving = async () => {
        if (!selectedPrimary && selectedSecondaries.length === 0) {
            setError('Please select at least a primary or secondary keyword before weaving.');
            return;
        }

        setWeaving(true);
        setError(null);
        try {
            const res = await keywordOptimizationService.weaveKeywords({
                html: articleContent,
                primary_keyword: selectedPrimary,
                secondary_keywords: selectedSecondaries,
            });

            if (res.success) {
                setWeaveResult(res);
                setViewMode('preview-diff');
            } else {
                setError(res.error || 'Failed to weave keywords.');
            }
        } catch (err: any) {
            const msg = err.response?.data?.error || err.message || 'An error occurred during AI keyword weaving.';
            setError(msg);
        } finally {
            setWeaving(false);
        }
    };

    const handleApplyAndSave = async () => {
        setSaving(true);
        try {
            const updatedHtml = weaveResult?.html || undefined;
            if (titleId) {
                await keywordOptimizationService.saveToTitle({
                    title_id: titleId,
                    primary_keyword: selectedPrimary || null,
                    secondary_keywords: selectedSecondaries,
                    primary_metric: selectedPrimaryMetric,
                    html: updatedHtml,
                });
            }

            await onApplyKeywords({
                primaryKeyword: selectedPrimary,
                secondaryKeywords: selectedSecondaries,
                primaryMetric: selectedPrimaryMetric,
                updatedHtml: updatedHtml,
            });

            onClose();
        } catch (err: any) {
            const msg = err.response?.data?.error || err.message || 'Failed to apply keyword changes.';
            setError(msg);
        } finally {
            setSaving(false);
        }
    };

    // Filtered keyword view
    const filteredKeywords = keywords.filter((kw) => {
        if (activeTab === 'quick-wins') return (kw.keyword_difficulty ?? 100) < 30;
        if (activeTab === 'high-volume') return kw.search_volume >= 500;
        return true;
    });

    if (!isOpen) return null;

    return (
        <AnimatePresence>
            <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-sm">
                <motion.div
                    initial={{ opacity: 0, scale: 0.95, y: 10 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.95, y: 10 }}
                    className="relative w-full max-w-5xl max-h-[92vh] flex flex-col rounded-2xl border border-border bg-card shadow-2xl overflow-hidden"
                >
                    {/* Modal Header */}
                    <div className="flex items-center justify-between px-6 py-4 border-b border-border bg-muted/20">
                        <div className="flex items-center gap-2.5">
                            <div className="p-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
                                <Sparkles className="w-5 h-5" />
                            </div>
                            <div>
                                <h2 className="text-lg font-bold text-foreground">DataForSEO Keyword Intelligence & Weaving</h2>
                                <p className="text-xs text-muted-foreground">
                                    Analyze search demand & difficulty, select target keywords, and weave them into the text.
                                </p>
                            </div>
                        </div>
                        <button
                            onClick={onClose}
                            className="p-1.5 text-muted-foreground hover:text-foreground hover:bg-muted/80 rounded-lg transition"
                        >
                            <X className="w-5 h-5" />
                        </button>
                    </div>

                    {/* Mode Toggle Bar (if weaving preview available) */}
                    {weaveResult && (
                        <div className="flex items-center gap-2 px-6 py-2 bg-muted/30 border-b border-border">
                            <button
                                onClick={() => setViewMode('discover')}
                                className={`px-3 py-1 text-xs font-medium rounded-md transition ${
                                    viewMode === 'discover'
                                        ? 'bg-background text-foreground shadow-sm'
                                        : 'text-muted-foreground hover:text-foreground'
                                }`}
                            >
                                <SlidersHorizontal className="w-3.5 h-3.5 inline mr-1" />
                                Keyword Suggestions ({keywords.length})
                            </button>
                            <button
                                onClick={() => setViewMode('preview-diff')}
                                className={`px-3 py-1 text-xs font-medium rounded-md transition ${
                                    viewMode === 'preview-diff'
                                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                                        : 'text-muted-foreground hover:text-foreground'
                                }`}
                            >
                                <Diff className="w-3.5 h-3.5 inline mr-1" />
                                AI Weaving Preview ({weaveResult.changes.length} changes)
                            </button>
                        </div>
                    )}

                    {/* Content Body */}
                    <div className="flex-1 overflow-y-auto p-6 space-y-5">
                        {error && (
                            <div className="p-3.5 rounded-xl bg-destructive/10 border border-destructive/20 text-destructive text-sm flex items-center gap-2">
                                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                                <span>{error}</span>
                            </div>
                        )}

                        {viewMode === 'discover' ? (
                            <>
                                {/* Search Bar & Seed Extraction */}
                                <div className="space-y-2.5">
                                    <form onSubmit={handleSearchSubmit} className="flex gap-2">
                                        <div className="relative flex-1">
                                            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                                            <input
                                                type="text"
                                                value={searchQuery}
                                                onChange={(e) => setSearchQuery(e.target.value)}
                                                placeholder="Enter seed topic (e.g. heat pump rebate, gas furnace)..."
                                                className="w-full h-10 pl-10 pr-4 rounded-xl border border-border bg-muted/40 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-ring/60 transition"
                                            />
                                        </div>
                                        <button
                                            type="submit"
                                            disabled={loading}
                                            className="inline-flex items-center gap-2 px-4 h-10 rounded-xl bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition disabled:opacity-50"
                                        >
                                            {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
                                            <span>Search DataForSEO</span>
                                        </button>
                                    </form>

                                    {/* Suggested Seeds Chips */}
                                    {suggestedSeeds.length > 0 && (
                                        <div className="flex items-center gap-1.5 flex-wrap text-xs pt-0.5">
                                            <span className="text-muted-foreground flex items-center gap-1 font-medium">
                                                <Sparkles className="w-3.5 h-3.5 text-primary" /> Suggested seeds:
                                            </span>
                                            {suggestedSeeds.map((seed) => (
                                                <button
                                                    key={seed}
                                                    type="button"
                                                    disabled={loading}
                                                    onClick={() => {
                                                        setSearchQuery(seed);
                                                        void handleDiscover(seed);
                                                    }}
                                                    className={`px-2.5 py-1 rounded-md border text-xs font-medium transition ${
                                                        searchQuery.toLowerCase() === seed.toLowerCase()
                                                            ? 'bg-primary/20 text-primary border-primary/40 shadow-xs'
                                                            : 'bg-muted/40 hover:bg-muted text-foreground border-border'
                                                    }`}
                                                >
                                                    {seed}
                                                </button>
                                            ))}
                                        </div>
                                    )}
                                </div>

                                {/* Tabs */}
                                <div className="flex items-center gap-2 border-b border-border pb-2">
                                    <button
                                        onClick={() => setActiveTab('all')}
                                        className={`px-3 py-1.5 text-xs font-medium rounded-lg transition ${
                                            activeTab === 'all'
                                                ? 'bg-muted text-foreground'
                                                : 'text-muted-foreground hover:text-foreground'
                                        }`}
                                    >
                                        All Keywords ({keywords.length})
                                    </button>
                                    <button
                                        onClick={() => setActiveTab('quick-wins')}
                                        className={`px-3 py-1.5 text-xs font-medium rounded-lg transition ${
                                            activeTab === 'quick-wins'
                                                ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/20'
                                                : 'text-muted-foreground hover:text-foreground'
                                        }`}
                                    >
                                        ⚡ Quick Wins (KD &lt; 30)
                                    </button>
                                    <button
                                        onClick={() => setActiveTab('high-volume')}
                                        className={`px-3 py-1.5 text-xs font-medium rounded-lg transition ${
                                            activeTab === 'high-volume'
                                                ? 'bg-blue-500/15 text-blue-400 border border-blue-500/20'
                                                : 'text-muted-foreground hover:text-foreground'
                                        }`}
                                    >
                                        🔥 High Volume (&ge; 500)
                                    </button>
                                </div>

                                {/* Results Table */}
                                {loading ? (
                                    <div className="py-16 text-center text-muted-foreground space-y-3">
                                        <Loader2 className="w-7 h-7 animate-spin mx-auto text-primary" />
                                        <p className="text-sm">Querying DataForSEO Labs for difficulty, volume, and competitive terms…</p>
                                    </div>
                                ) : filteredKeywords.length === 0 ? (
                                    <div className="py-12 text-center text-muted-foreground border border-dashed border-border rounded-xl space-y-3 px-4">
                                        <Target className="w-8 h-8 mx-auto opacity-40" />
                                        <div>
                                            <p className="text-sm font-medium text-foreground">
                                                {searchQuery ? `No keywords found for "${searchQuery}"` : 'No keyword suggestions found.'}
                                            </p>
                                            <p className="text-xs text-muted-foreground mt-0.5">
                                                DataForSEO requires short 1–3 word phrases (e.g. "heat pump rebate", "gas furnace replacement").
                                            </p>
                                        </div>
                                        {suggestedSeeds.length > 0 && (
                                            <div className="pt-2">
                                                <p className="text-xs font-medium text-muted-foreground mb-2">Try one of these high-intent seeds:</p>
                                                <div className="flex items-center justify-center gap-2 flex-wrap">
                                                    {suggestedSeeds.map((seed) => (
                                                        <button
                                                            key={seed}
                                                            type="button"
                                                            disabled={loading}
                                                            onClick={() => {
                                                                setSearchQuery(seed);
                                                                void handleDiscover(seed);
                                                            }}
                                                            className="px-3 py-1.5 rounded-lg bg-primary/10 hover:bg-primary/20 text-primary border border-primary/20 text-xs font-medium transition"
                                                        >
                                                            🔍 Search "{seed}"
                                                        </button>
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                ) : (
                                    <div className="rounded-xl border border-border overflow-hidden bg-muted/10">
                                        <div className="overflow-x-auto max-h-[340px]">
                                            <table className="w-full text-left text-sm">
                                                <thead className="sticky top-0 bg-muted/80 backdrop-blur-sm border-b border-border text-[11px] uppercase tracking-wider text-muted-foreground">
                                                    <tr>
                                                        <th className="px-4 py-2.5">Keyword</th>
                                                        <th className="px-3 py-2.5">Search Volume</th>
                                                        <th className="px-3 py-2.5">KD / Difficulty</th>
                                                        <th className="px-3 py-2.5">In Text</th>
                                                        <th className="px-3 py-2.5">CPC / Intent</th>
                                                        <th className="px-4 py-2.5 text-right">Actions</th>
                                                    </tr>
                                                </thead>
                                                <tbody className="divide-y divide-border/60">
                                                    {filteredKeywords.map((kw) => {
                                                        const isPrimary = selectedPrimary.toLowerCase() === kw.keyword.toLowerCase();
                                                        const isSecondary = selectedSecondaries.some(
                                                            (s) => s.toLowerCase() === kw.keyword.toLowerCase()
                                                        );
                                                        const kd = kw.keyword_difficulty;
                                                        const kdColor =
                                                            kd == null
                                                                ? 'text-muted-foreground'
                                                                : kd < 30
                                                                ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20'
                                                                : kd < 60
                                                                ? 'text-amber-400 bg-amber-500/10 border-amber-500/20'
                                                                : 'text-red-400 bg-red-500/10 border-red-500/20';

                                                        return (
                                                            <tr
                                                                key={kw.keyword}
                                                                className={`hover:bg-muted/40 transition ${
                                                                    isPrimary
                                                                        ? 'bg-primary/5'
                                                                        : isSecondary
                                                                        ? 'bg-blue-500/5'
                                                                        : ''
                                                                }`}
                                                            >
                                                                <td className="px-4 py-3 font-medium text-foreground">
                                                                    <div className="flex items-center gap-2">
                                                                        <span>{kw.keyword}</span>
                                                                        {kw.is_seed && (
                                                                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground border border-border">
                                                                                Seed
                                                                            </span>
                                                                        )}
                                                                        {isPrimary && (
                                                                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/20 text-primary font-semibold">
                                                                                Primary
                                                                            </span>
                                                                        )}
                                                                        {isSecondary && (
                                                                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-400 font-semibold">
                                                                                Secondary
                                                                            </span>
                                                                        )}
                                                                    </div>
                                                                </td>
                                                                <td className="px-3 py-3 text-muted-foreground font-mono">
                                                                    {kw.search_volume > 0
                                                                        ? kw.search_volume.toLocaleString()
                                                                        : '0'}
                                                                </td>
                                                                <td className="px-3 py-3">
                                                                    {kd != null ? (
                                                                        <span
                                                                            className={`inline-block px-2 py-0.5 text-xs font-semibold rounded-md border ${kdColor}`}
                                                                        >
                                                                            {kd} ({kd < 30 ? 'Easy' : kd < 60 ? 'Med' : 'Hard'})
                                                                        </span>
                                                                    ) : (
                                                                        <span className="text-muted-foreground text-xs">-</span>
                                                                    )}
                                                                </td>
                                                                <td className="px-3 py-3">
                                                                    <span
                                                                        className={`text-xs px-2 py-0.5 rounded font-mono ${
                                                                            kw.in_text_count > 0
                                                                                ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                                                                                : 'bg-muted text-muted-foreground'
                                                                        }`}
                                                                    >
                                                                        {kw.in_text_count}x
                                                                    </span>
                                                                </td>
                                                                <td className="px-3 py-3 text-xs text-muted-foreground">
                                                                    <div>${kw.cpc.toFixed(2)}</div>
                                                                    <div className="text-[10px] uppercase text-muted-foreground/70">
                                                                        {kw.intent}
                                                                    </div>
                                                                </td>
                                                                <td className="px-4 py-3 text-right space-x-1.5">
                                                                    <button
                                                                        onClick={() => handleSelectPrimary(kw)}
                                                                        className={`px-2.5 py-1 text-xs rounded-lg border font-medium transition ${
                                                                            isPrimary
                                                                                ? 'bg-primary text-primary-foreground border-primary'
                                                                                : 'bg-muted/50 border-border text-muted-foreground hover:text-foreground hover:bg-muted'
                                                                        }`}
                                                                    >
                                                                        <Star className="w-3 h-3 inline mr-1" />
                                                                        {isPrimary ? 'Primary' : 'Set Primary'}
                                                                    </button>
                                                                    <button
                                                                        onClick={() => handleToggleSecondary(kw)}
                                                                        className={`px-2.5 py-1 text-xs rounded-lg border font-medium transition ${
                                                                            isSecondary
                                                                                ? 'bg-blue-500 text-white border-blue-500'
                                                                                : 'bg-muted/50 border-border text-muted-foreground hover:text-foreground hover:bg-muted'
                                                                        }`}
                                                                    >
                                                                        <Plus className="w-3 h-3 inline mr-1" />
                                                                        {isSecondary ? 'Added' : 'Secondary'}
                                                                    </button>
                                                                </td>
                                                            </tr>
                                                        );
                                                    })}
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                )}
                            </>
                        ) : (
                            /* Before / After Weaving Preview View */
                            <div className="space-y-4">
                                <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20 space-y-2">
                                    <div className="flex items-center gap-2 text-emerald-400 font-semibold text-sm">
                                        <CheckCircle2 className="w-4 h-4" />
                                        <span>AI Keyword Weaving Complete</span>
                                    </div>
                                    <ul className="text-xs text-muted-foreground space-y-1 list-disc list-inside">
                                        {weaveResult?.changes.map((ch, idx) => (
                                            <li key={idx}>{ch}</li>
                                        ))}
                                    </ul>
                                </div>

                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div className="rounded-xl border border-border p-4 bg-muted/10 max-h-[360px] overflow-y-auto">
                                        <h4 className="text-xs font-semibold text-muted-foreground uppercase mb-2">Original Content</h4>
                                        <div
                                            className="text-xs text-foreground/80 prose prose-invert max-w-none"
                                            dangerouslySetInnerHTML={{ __html: articleContent }}
                                        />
                                    </div>
                                    <div className="rounded-xl border border-emerald-500/30 p-4 bg-emerald-500/5 max-h-[360px] overflow-y-auto">
                                        <h4 className="text-xs font-semibold text-emerald-400 uppercase mb-2">Weaved & Optimized Content</h4>
                                        <div
                                            className="text-xs text-foreground prose prose-invert max-w-none"
                                            dangerouslySetInnerHTML={{ __html: weaveResult?.html || '' }}
                                        />
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Selected Keyword Summary & Footer */}
                    <div className="px-6 py-4 border-t border-border bg-muted/30 flex flex-col sm:flex-row items-center justify-between gap-4">
                        <div className="flex items-center gap-2 flex-wrap text-xs">
                            <span className="font-semibold text-foreground">Target Pack:</span>
                            {selectedPrimary ? (
                                <span className="px-2.5 py-1 rounded-lg bg-primary/20 text-primary border border-primary/30 font-medium">
                                    ★ {selectedPrimary} {selectedPrimaryMetric ? `(${selectedPrimaryMetric.search_volume} vol, KD ${selectedPrimaryMetric.keyword_difficulty})` : ''}
                                </span>
                            ) : (
                                <span className="text-muted-foreground italic">No primary selected</span>
                            )}
                            {selectedSecondaries.map((sec) => (
                                <span
                                    key={sec}
                                    className="px-2 py-0.5 rounded-lg bg-blue-500/20 text-blue-400 border border-blue-500/30"
                                >
                                    + {sec}
                                </span>
                            ))}
                        </div>

                        <div className="flex items-center gap-2.5 w-full sm:w-auto justify-end">
                            <button
                                onClick={onClose}
                                className="px-4 py-2 rounded-xl border border-border bg-muted/40 text-sm font-medium text-muted-foreground hover:text-foreground transition"
                            >
                                Cancel
                            </button>

                            {viewMode === 'discover' ? (
                                <button
                                    onClick={handleTriggerWeaving}
                                    disabled={weaving || (!selectedPrimary && selectedSecondaries.length === 0)}
                                    className="inline-flex items-center gap-2 px-5 py-2 rounded-xl bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition disabled:opacity-50 shadow-md"
                                >
                                    {weaving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
                                    <span>✨ Weave into Article</span>
                                </button>
                            ) : (
                                <button
                                    onClick={handleApplyAndSave}
                                    disabled={saving}
                                    className="inline-flex items-center gap-2 px-5 py-2 rounded-xl bg-emerald-600 text-white text-sm font-medium hover:bg-emerald-500 transition disabled:opacity-50 shadow-md"
                                >
                                    {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Check className="w-4 h-4" />}
                                    <span>Apply Changes & Save</span>
                                </button>
                            )}
                        </div>
                    </div>
                </motion.div>
            </div>
        </AnimatePresence>
    );
};
