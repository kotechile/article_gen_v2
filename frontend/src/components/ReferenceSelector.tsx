import React, { useMemo, useState } from 'react';
import { X, Check, Filter, Globe } from 'lucide-react';
import { rankCitationDomains, topDomains } from '../lib/citationAuthority';

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
    onApply: (selectedIndices: Set<number>, showInText: boolean) => void;
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

    const inferredInitialMode = initialSelected.size < citations.length ? 'curated' : 'all';
    const inferredInitialDomainLimit = Math.max(
        1,
        Math.min(
            uniqueDomainCount || 1,
            new Set(Array.from(initialSelected).map((index) => authorityMeta[index]?.domain).filter(Boolean)).size || 5
        )
    );

    const [filterMode, setFilterMode] = useState<'all' | 'curated'>(inferredInitialMode);
    const [topDomainLimit, setTopDomainLimit] = useState<number>(inferredInitialDomainLimit);
    const [showInText, setShowInText] = useState(initialShowInText);

    const curatedIndices = useMemo(() => {
        const selectedDomains = new Set(topDomains(citations, topDomainLimit));
        const indices = citations
            .map((_, index) => ({ index, domain: authorityMeta[index]?.domain || 'unknown' }))
            .filter((item) => selectedDomains.has(item.domain))
            .map((item) => item.index);
        return new Set(indices);
    }, [authorityMeta, citations, topDomainLimit]);

    const appliedSelection = filterMode === 'all'
        ? new Set(citations.map((_, i) => i))
        : curatedIndices;

    const handleApply = () => {
        onApply(appliedSelection, showInText);
        onClose();
    };

    const currentDomainSummary = filterMode === 'all'
        ? uniqueDomainCount
        : new Set(Array.from(appliedSelection).map((idx) => authorityMeta[idx]?.domain)).size;

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

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
                            <div className="text-xs mt-1">Show references from top-ranked source domains only.</div>
                        </button>
                    </div>

                    {filterMode === 'curated' && (
                        <div className="space-y-2">
                            <label className="text-sm font-medium text-foreground" htmlFor="top-domain-limit">
                                Show Top [{topDomainLimit}] References (by domain authority)
                            </label>
                            <input
                                id="top-domain-limit"
                                type="range"
                                min={1}
                                max={Math.max(uniqueDomainCount, 1)}
                                step={1}
                                value={Math.min(topDomainLimit, Math.max(uniqueDomainCount, 1))}
                                onChange={(e) => setTopDomainLimit(parseInt(e.target.value, 10) || 1)}
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
                                        className={`flex items-start gap-4 p-4 rounded-xl border transition ${included ? 'border-primary/50 bg-primary/5' : 'border-border opacity-60'}`}
                                    >
                                        <div className={`mt-1 text-xs font-bold px-2 py-1 rounded ${included ? 'bg-primary/15 text-primary' : 'bg-muted text-muted-foreground'}`}>
                                            [{index + 1}]
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
                        Active filter: {filterMode === 'all' ? 'Display All' : `Top ${topDomainLimit} domains`}.
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
                            className="flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary/90 text-primary-foreground rounded-xl font-medium text-sm transition"
                        >
                            <Check className="w-4 h-4" />
                            Apply Filter
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};
