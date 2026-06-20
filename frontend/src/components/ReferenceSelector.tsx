import React, { useMemo, useState } from 'react';
import { X, Check, Filter, Globe } from 'lucide-react';
import { rankCitationDomains } from '../lib/citationAuthority';

interface Citation {
    id?: string;
    title: string;
    url?: string;
    author?: string;
    source_type?: string;
    publication_date?: string;
    publisher?: string;
    domain?: string;
    domain_rank?: number;
    domain_frequency?: number;
    authority_score?: number;
}

interface ReferenceSelectorProps {
    citations: Citation[];
    selectedCitations: Set<number>;
    showInTextCitations: boolean;
    onClose: () => void;
    onApply: (selectedIndices: Set<number>, showInText: boolean) => void | Promise<void>;
}

export const ReferenceSelector: React.FC<ReferenceSelectorProps> = ({
    citations,
    selectedCitations: initialSelected,
    showInTextCitations: initialShowInText,
    onClose,
    onApply
}) => {
    const authorityMeta = useMemo(() => rankCitationDomains(citations), [citations]);
    const uniqueDomainCount = useMemo(() => new Set(authorityMeta.map((item) => item.domain)).size, [authorityMeta]);

    // Compute inferred initial mode helper
    const getInferredInitialMode = (): 'all' | 'curated' | 'manual' => {
        if (initialSelected.size === citations.length) return 'all';
        // Check if matches curated
        const limit = initialSelected.size;
        const ranked = citations
            .map((_, index) => ({
                index,
                domainRank: authorityMeta[index]?.domainRank ?? Number.MAX_SAFE_INTEGER,
                domainFrequency: authorityMeta[index]?.domainFrequency ?? 0,
            }))
            .sort((a, b) => {
                if (a.domainRank !== b.domainRank) return a.domainRank - b.domainRank;
                if (a.domainFrequency !== b.domainFrequency) return b.domainFrequency - a.domainFrequency;
                return a.index - b.index;
            })
            .slice(0, limit)
            .map((item) => item.index);
        const matchesCurated = ranked.length > 0 && ranked.every(idx => initialSelected.has(idx));
        return matchesCurated ? 'curated' : 'manual';
    };

    const inferredInitialCitationLimit = Math.max(
        1,
        Math.min(
            citations.length || 1,
            initialSelected.size || Math.min(10, citations.length || 1)
        )
    );

    const [filterMode, setFilterMode] = useState<'all' | 'curated' | 'manual'>(getInferredInitialMode());
    const [topCitationLimit, setTopCitationLimit] = useState<number>(inferredInitialCitationLimit);
    const [manualSelection, setManualSelection] = useState<Set<number>>(new Set(initialSelected));
    const [showInText, setShowInText] = useState(initialShowInText);
    const [isApplying, setIsApplying] = useState(false);

    const curatedIndices = useMemo(() => {
        if (!citations.length) return new Set<number>();
        const limit = Math.max(1, Math.min(topCitationLimit, citations.length));
        const ranked = citations
            .map((_, index) => ({
                index,
                domainRank: authorityMeta[index]?.domainRank ?? Number.MAX_SAFE_INTEGER,
                domainFrequency: authorityMeta[index]?.domainFrequency ?? 0,
            }))
            .sort((a, b) => {
                if (a.domainRank !== b.domainRank) return a.domainRank - b.domainRank;
                if (a.domainFrequency !== b.domainFrequency) return b.domainFrequency - a.domainFrequency;
                return a.index - b.index;
            })
            .slice(0, limit)
            .map((item) => item.index);

        return new Set(ranked);
    }, [authorityMeta, citations, topCitationLimit]);

    const appliedSelection = useMemo(() => {
        if (filterMode === 'all') {
            return new Set(citations.map((_, i) => i));
        }
        if (filterMode === 'curated') {
            return curatedIndices;
        }
        return manualSelection;
    }, [filterMode, citations, curatedIndices, manualSelection]);

    const handleToggleReference = (index: number) => {
        const next = new Set(appliedSelection);
        if (next.has(index)) {
            next.delete(index);
        } else {
            next.add(index);
        }
        setManualSelection(next);
        setFilterMode('manual');
    };

    const handleApply = async () => {
        try {
            setIsApplying(true);
            await onApply(appliedSelection, showInText);
            onClose();
        } finally {
            setIsApplying(false);
        }
    };

    const currentDomainSummary = useMemo(() => {
        return new Set(Array.from(appliedSelection).map((idx) => authorityMeta[idx]?.domain)).size;
    }, [appliedSelection, authorityMeta]);

    return (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-background rounded-2xl shadow-2xl max-w-4xl w-full max-h-[85vh] flex flex-col border border-border">
                <div className="flex items-center justify-between p-6 border-b border-border">
                    <h2 className="text-xl font-bold text-foreground flex items-center gap-2">
                        <Filter className="w-5 h-5 text-primary" />
                        Reference Filter
                    </h2>
                    <button
                        onClick={onClose}
                        className="p-2 text-muted-foreground hover:text-foreground rounded-lg hover:bg-muted transition"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                <div className="p-6 border-b border-border space-y-5 bg-muted/30">
                    <p className="text-sm text-muted-foreground">
                        Current research contains <strong>{citations.length}</strong> references across <strong>{uniqueDomainCount}</strong> domains.
                    </p>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <button
                            type="button"
                            onClick={() => setFilterMode('all')}
                            className={`rounded-xl border p-3 text-left transition ${filterMode === 'all' ? 'border-primary bg-primary/10 text-foreground' : 'border-border hover:bg-muted text-muted-foreground'}`}
                        >
                            <div className="font-semibold">Display All</div>
                            <div className="text-xs mt-1">Show every citation and reference.</div>
                        </button>
                        <button
                            type="button"
                            onClick={() => setFilterMode('curated')}
                            className={`rounded-xl border p-3 text-left transition ${filterMode === 'curated' ? 'border-primary bg-primary/10 text-foreground' : 'border-border hover:bg-muted text-muted-foreground'}`}
                        >
                            <div className="font-semibold">Curated View</div>
                            <div className="text-xs mt-1">Show references from top-ranked domains only.</div>
                        </button>
                        <button
                            type="button"
                            onClick={() => {
                                setManualSelection(new Set(appliedSelection));
                                setFilterMode('manual');
                            }}
                            className={`rounded-xl border p-3 text-left transition ${filterMode === 'manual' ? 'border-primary bg-primary/10 text-foreground' : 'border-border hover:bg-muted text-muted-foreground'}`}
                        >
                            <div className="font-semibold">Manual Selection</div>
                            <div className="text-xs mt-1">Pick and choose reference links manually.</div>
                        </button>
                    </div>

                    {filterMode === 'curated' && (
                        <div className="space-y-2">
                            <label className="text-sm font-medium text-foreground" htmlFor="top-domain-limit">
                                Show Top [{topCitationLimit}] References (ranked by domain authority)
                            </label>
                            <input
                                id="top-domain-limit"
                                type="range"
                                min={1}
                                max={Math.max(citations.length, 1)}
                                step={1}
                                value={Math.min(topCitationLimit, Math.max(citations.length, 1))}
                                onChange={(e) => setTopCitationLimit(parseInt(e.target.value, 10) || 1)}
                                className="w-full"
                            />
                            <p className="text-xs text-muted-foreground">
                                Showing {appliedSelection.size} citations from {currentDomainSummary} domains.
                            </p>
                        </div>
                    )}

                    <div className="flex items-center justify-between">
                        <label className="flex items-center gap-3 cursor-pointer">
                            <input
                                type="checkbox"
                                checked={showInText}
                                onChange={(e) => setShowInText(e.target.checked)}
                                className="w-5 h-5 rounded border-border text-primary focus:ring-primary cursor-pointer"
                            />
                            <span className="text-sm font-medium text-foreground">Show in-text citation numbers</span>
                        </label>
                        <span className="text-xs text-muted-foreground">
                            {showInText ? 'e.g., [1], [2], [3]' : 'In-text markers hidden'}
                        </span>
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto p-6">
                    {citations.length === 0 ? (
                        <div className="text-center py-12">
                            <p className="text-muted-foreground">No references found for this article.</p>
                        </div>
                    ) : (
                        <div className="space-y-3">
                            {citations.map((citation, index) => {
                                const meta = authorityMeta[index];
                                const included = appliedSelection.has(index);
                                return (
                                    <div
                                        key={index}
                                        onClick={() => handleToggleReference(index)}
                                        className={`flex items-start gap-4 p-4 rounded-xl border transition cursor-pointer select-none ${included ? 'border-primary/50 bg-primary/5' : 'border-border opacity-70 hover:opacity-100 hover:bg-muted/30'}`}
                                    >
                                        <div className="flex items-center gap-3 mt-1">
                                            <input
                                                type="checkbox"
                                                checked={included}
                                                onChange={() => handleToggleReference(index)}
                                                onClick={(e) => e.stopPropagation()}
                                                className="w-4 h-4 rounded border-border text-primary focus:ring-primary cursor-pointer"
                                            />
                                            <div className={`text-xs font-bold px-2 py-1 rounded ${included ? 'bg-primary/15 text-primary' : 'bg-muted text-muted-foreground'}`}>
                                                [{index + 1}]
                                            </div>
                                        </div>

                                        <div className="flex-1 min-w-0">
                                            <div className="flex flex-wrap items-center gap-2 mb-1">
                                                <span className="font-semibold text-sm text-foreground line-clamp-2">
                                                    {citation.title || 'Unknown Source'}
                                                </span>
                                                {citation.source_type && (
                                                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-muted text-muted-foreground uppercase">
                                                        {citation.source_type}
                                                    </span>
                                                )}
                                                <span className="text-[10px] px-2 py-0.5 rounded-full bg-muted text-muted-foreground inline-flex items-center gap-1">
                                                    <Globe className="w-3 h-3" />
                                                    {meta.domain}
                                                </span>
                                                <span className="text-[10px] px-2 py-0.5 rounded-full bg-muted text-muted-foreground">
                                                    Rank #{meta.domainRank} • {meta.domainFrequency} refs
                                                </span>
                                            </div>

                                            {citation.author && citation.author !== 'Unknown Author' && (
                                                <p className="text-xs text-muted-foreground mb-1">
                                                    {citation.author}
                                                    {citation.publication_date && ` (${citation.publication_date})`}
                                                </p>
                                            )}

                                            {citation.url && citation.url !== '#' && (
                                                <a
                                                    href={citation.url}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    onClick={(e) => e.stopPropagation()}
                                                    className="text-xs text-primary hover:underline truncate block"
                                                >
                                                    {citation.url}
                                                </a>
                                            )}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>

                <div className="p-6 border-t border-border flex items-center justify-between gap-3">
                    <p className="text-xs text-muted-foreground">
                        Active filter: {filterMode === 'all' ? 'Display All' : `Top ${topCitationLimit} citations`}.
                    </p>
                    <div className="flex items-center gap-3">
                        <button
                            onClick={onClose}
                            className="px-4 py-2 text-sm font-medium text-muted-foreground hover:bg-muted rounded-xl transition"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={handleApply}
                            disabled={isApplying}
                            className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary/90 text-primary-foreground rounded-xl font-medium text-sm transition"
                        >
                            <Check className="w-4 h-4" />
                            {isApplying ? 'Applying...' : 'Apply Filter'}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};
