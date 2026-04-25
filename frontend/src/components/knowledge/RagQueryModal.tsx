import React, { useEffect, useMemo, useState } from 'react';
import { Bot, Clock3, Loader2, Search, Sparkles, X } from 'lucide-react';
import type { Collection, RagQueryResponse } from '../../types/knowledge';

type QueryType = 'simple' | 'hybrid' | 'agentic_iterative' | 'truly_agentic' | 'agentic_fixed';
type VerboseMode = 'concise' | 'balanced' | 'detailed';

interface RagQueryModalProps {
    isOpen: boolean;
    onClose: () => void;
    currentCollection: Collection | null;
    llmModels: string[];
    onSearch: (params: {
        queryType: QueryType;
        llm: string;
        query: string;
        numResults: number;
        topK: number;
        balanceEmphasis: string;
        maxIterations: number;
        maxDocs: number;
        verboseMode: VerboseMode;
    }) => Promise<RagQueryResponse>;
}

const QUERY_TYPE_OPTIONS: Array<{ label: string; value: QueryType }> = [
    { label: 'Simple', value: 'simple' },
    { label: 'Hybrid Enhanced', value: 'hybrid' },
    { label: 'Agentic Iterative', value: 'agentic_iterative' },
    { label: 'Truly Agentic', value: 'truly_agentic' },
    { label: 'Agentic Fixed', value: 'agentic_fixed' },
];

const DEFAULT_LLM_MODELS = ['gpt-4o', 'deepseek', 'claude-3-5-sonnet'];
const DEFAULT_BALANCE_OPTIONS = ['balanced', 'comprehensive', 'news_focused', 'auto'];
const DEFAULT_RESPONSE: RagQueryResponse | null = null;
const SELECT_CONTROL_CLASS = 'h-12 w-full rounded-xl border border-gray-300 bg-white px-4 text-sm outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500 dark:border-gray-700 dark:bg-gray-900 dark:text-white';

function escapeHtml(value: string): string {
    return value
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function renderInlineMarkdown(value: string): string {
    return value
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/\*([^*]+)\*/g, '<em>$1</em>')
        .replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
}

function markdownToHtml(markdown: string): string {
    const lines = markdown.split(/\r?\n/);
    const html: string[] = [];
    let inList = false;
    let inCodeBlock = false;
    let codeLines: string[] = [];

    const closeList = () => {
        if (inList) {
            html.push('</ul>');
            inList = false;
        }
    };

    const closeCodeBlock = () => {
        if (inCodeBlock) {
            html.push(`<pre><code>${codeLines.join('\n')}</code></pre>`);
            inCodeBlock = false;
            codeLines = [];
        }
    };

    for (const rawLine of lines) {
        const line = escapeHtml(rawLine);
        const trimmed = line.trim();

        if (trimmed.startsWith('```')) {
            closeList();
            if (inCodeBlock) {
                closeCodeBlock();
            } else {
                inCodeBlock = true;
            }
            continue;
        }

        if (inCodeBlock) {
            codeLines.push(line);
            continue;
        }

        if (!trimmed) {
            closeList();
            html.push('<br />');
            continue;
        }

        const headingMatch = trimmed.match(/^(#{1,6})\s+(.+)$/);
        if (headingMatch) {
            closeList();
            const level = headingMatch[1].length;
            html.push(`<h${level}>${renderInlineMarkdown(headingMatch[2])}</h${level}>`);
            continue;
        }

        const bulletMatch = trimmed.match(/^[-*]\s+(.+)$/);
        if (bulletMatch) {
            if (!inList) {
                html.push('<ul>');
                inList = true;
            }
            html.push(`<li>${renderInlineMarkdown(bulletMatch[1])}</li>`);
            continue;
        }

        closeList();
        html.push(`<p>${renderInlineMarkdown(trimmed)}</p>`);
    }

    closeList();
    closeCodeBlock();

    return html.join('');
}

function QueryResponseMarkdown({ content }: { content: string }) {
    const html = useMemo(() => markdownToHtml(content), [content]);

    return (
        <div
            className="prose prose-sm max-w-none dark:prose-invert prose-p:my-3 prose-headings:mb-3 prose-headings:mt-5 prose-ul:my-3 prose-code:rounded prose-code:bg-muted prose-code:px-1 prose-code:py-0.5 prose-pre:bg-slate-950 prose-pre:text-slate-100"
            dangerouslySetInnerHTML={{ __html: html }}
        />
    );
}

export const RagQueryModal: React.FC<RagQueryModalProps> = ({
    isOpen,
    onClose,
    currentCollection,
    llmModels,
    onSearch,
}) => {
    const modelOptions = llmModels.length > 0 ? llmModels : DEFAULT_LLM_MODELS;
    const [queryType, setQueryType] = useState<QueryType>('hybrid');
    const [llm, setLlm] = useState(modelOptions[0] || DEFAULT_LLM_MODELS[0]);
    const [query, setQuery] = useState('');
    const [numResults, setNumResults] = useState(5);
    const [topK, setTopK] = useState(10);
    const [balanceEmphasis, setBalanceEmphasis] = useState('balanced');
    const [maxIterations, setMaxIterations] = useState(3);
    const [maxDocs, setMaxDocs] = useState(3);
    const [verboseMode, setVerboseMode] = useState<VerboseMode>('balanced');
    const [isSearching, setIsSearching] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [result, setResult] = useState<RagQueryResponse | null>(DEFAULT_RESPONSE);

    useEffect(() => {
        if (!isOpen) return;
        setError(null);
        setResult(DEFAULT_RESPONSE);
        setIsSearching(false);
    }, [isOpen, currentCollection?.id]);

    useEffect(() => {
        if (!modelOptions.includes(llm)) {
            setLlm(modelOptions[0] || DEFAULT_LLM_MODELS[0]);
        }
    }, [llm, modelOptions]);

    if (!isOpen || !currentCollection) return null;

    const handleSubmit = async () => {
        if (!query.trim()) {
            setError('Enter a question before searching.');
            return;
        }

        setIsSearching(true);
        setError(null);
        setResult(DEFAULT_RESPONSE);

        try {
            const response = await onSearch({
                queryType,
                llm,
                query: query.trim(),
                numResults,
                topK,
                balanceEmphasis,
                maxIterations,
                maxDocs,
                verboseMode,
            });

            setResult(response);
            if (response.status !== 'success') {
                setError(response.error || 'The RAG service returned a non-success status.');
            }
        } catch (err: any) {
            setError(err?.message || 'The query could not be completed.');
        } finally {
            setIsSearching(false);
        }
    };

    const methodBadge = result?.method || QUERY_TYPE_OPTIONS.find((option) => option.value === queryType)?.label;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm">
            <div className="flex h-[min(92vh,880px)] w-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-xl dark:border-gray-700 dark:bg-gray-800">
                <div className="flex items-center justify-between border-b border-gray-100 p-6 dark:border-gray-700">
                    <div>
                        <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
                            Query Collection: {currentCollection.name}
                        </h3>
                        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                            Run an ad hoc RAG query against this document collection.
                        </p>
                    </div>
                    <button onClick={onClose} className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300">
                        <X className="h-5 w-5" />
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto p-6">
                    <div className="grid gap-6 lg:grid-cols-2">
                        <div className="space-y-5">
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                                    Query Type
                                </label>
                                <select
                                    value={queryType}
                                    onChange={(e) => setQueryType(e.target.value as QueryType)}
                                    className={SELECT_CONTROL_CLASS}
                                >
                                    {QUERY_TYPE_OPTIONS.map((option) => (
                                        <option key={option.value} value={option.value}>
                                            {option.label}
                                        </option>
                                    ))}
                                </select>
                            </div>

                            <div className="space-y-2">
                                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                                    LLM Model
                                </label>
                                <select
                                    value={llm}
                                    onChange={(e) => setLlm(e.target.value)}
                                    className={SELECT_CONTROL_CLASS}
                                >
                                    {modelOptions.map((model) => (
                                        <option key={model} value={model}>
                                            {model}
                                        </option>
                                    ))}
                                </select>
                            </div>

                            {queryType === 'simple' && (
                                <>
                                    <NumberField label="num_results" value={numResults} onChange={setNumResults} min={1} />
                                    <BalanceField value={balanceEmphasis} onChange={setBalanceEmphasis} />
                                </>
                            )}

                            {queryType === 'hybrid' && (
                                <>
                                    <NumberField label="top_k" value={topK} onChange={setTopK} min={1} />
                                    <BalanceField value={balanceEmphasis} onChange={setBalanceEmphasis} />
                                </>
                            )}

                            {queryType === 'agentic_iterative' && (
                                <NumberField label="max_iterations" value={maxIterations} onChange={setMaxIterations} min={1} />
                            )}

                            {queryType === 'agentic_fixed' && (
                                <>
                                    <NumberField label="max_docs" value={maxDocs} onChange={setMaxDocs} min={1} />
                                    <div className="space-y-2">
                                        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                                            verbose_mode
                                        </label>
                                        <select
                                            value={verboseMode}
                                            onChange={(e) => setVerboseMode(e.target.value as VerboseMode)}
                                            className={SELECT_CONTROL_CLASS}
                                        >
                                            <option value="concise">concise</option>
                                            <option value="balanced">balanced</option>
                                            <option value="detailed">detailed</option>
                                        </select>
                                    </div>
                                </>
                            )}
                        </div>

                        <div className="space-y-2">
                            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                                Query Input
                            </label>
                            <textarea
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                placeholder="Ask a question about the selected collection..."
                                className="min-h-[280px] w-full rounded-xl border border-gray-300 bg-white p-4 text-sm outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500 dark:border-gray-700 dark:bg-gray-900 dark:text-white"
                            />
                            <p className="text-xs text-gray-500 dark:text-gray-400">
                                The selected collection name is inherited automatically from the active collection.
                            </p>
                        </div>
                    </div>

                    <div className="mt-6 flex flex-wrap justify-end gap-3 border-t border-gray-100 pt-5 dark:border-gray-700">
                        <button
                            onClick={onClose}
                            className="rounded-lg px-4 py-2 text-sm font-medium text-gray-700 transition hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-700"
                        >
                            Close
                        </button>
                        <button
                            onClick={handleSubmit}
                            disabled={isSearching}
                            className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-5 py-2 text-sm font-medium text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
                        >
                            {isSearching ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                            {isSearching ? 'Searching...' : 'Search'}
                        </button>
                    </div>

                    {(isSearching || error || result) && (
                        <div className="mt-6 rounded-2xl border border-gray-200 bg-gray-50 p-5 dark:border-gray-700 dark:bg-gray-900/40">
                            {isSearching && (
                                <div className="flex min-h-[240px] flex-col items-center justify-center gap-3 text-center text-sm text-gray-500 dark:text-gray-400">
                                    <Loader2 className="h-6 w-6 animate-spin text-blue-600" />
                                    <p>Running query against {currentCollection.name}...</p>
                                </div>
                            )}

                            {!isSearching && error && (
                                <div className="mb-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-300">
                                    {error}
                                </div>
                            )}

                            {!isSearching && result && (
                                <div className="space-y-5">
                                    <div className="flex flex-wrap items-center gap-3">
                                        <SummaryPill icon={<Sparkles className="h-3.5 w-3.5" />} label={`Status: ${result.status || 'unknown'}`} />
                                        <SummaryPill icon={<Clock3 className="h-3.5 w-3.5" />} label={`${Number(result.time_seconds || 0).toFixed(1)}s`} />
                                        {methodBadge && <SummaryPill icon={<Bot className="h-3.5 w-3.5" />} label={methodBadge} />}
                                    </div>

                                    {result.response ? (
                                        <div className="max-h-[320px] overflow-y-auto rounded-xl border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-950/40">
                                            <QueryResponseMarkdown content={result.response} />
                                        </div>
                                    ) : (
                                        <p className="text-sm text-gray-500 dark:text-gray-400">
                                            No response text was returned for this query.
                                        </p>
                                    )}

                                    <div className="space-y-3">
                                        <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                                            Source Attribution
                                        </h4>
                                        {result.source_attribution && result.source_attribution.length > 0 ? (
                                            <ul className="list-disc space-y-2 pl-5 text-sm text-gray-700 dark:text-gray-300">
                                                {result.source_attribution.map((item, index) => (
                                                    <li key={`${item}-${index}`}>{item}</li>
                                                ))}
                                            </ul>
                                        ) : (
                                            <p className="text-sm text-gray-500 dark:text-gray-400">
                                                No source attribution was returned.
                                            </p>
                                        )}
                                    </div>

                                    <div className="space-y-3">
                                        <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
                                            Documents Used
                                        </h4>
                                        {result.documents_used && result.documents_used.length > 0 ? (
                                            <div className="grid gap-3 md:grid-cols-2">
                                                {result.documents_used.map((doc, index) => (
                                                    <div
                                                        key={`${doc.doc_id || doc.title || 'doc'}-${index}`}
                                                        className="rounded-xl border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-950/40"
                                                    >
                                                        <p className="font-semibold text-gray-900 dark:text-gray-100">
                                                            {doc.title || 'Untitled document'}
                                                        </p>
                                                        <div className="mt-3 flex flex-wrap gap-2">
                                                            <Tag label={`${doc.chunks_contributed ?? 0} chunks`} />
                                                            <Tag label={`weight ${typeof doc.importance_weight === 'number' ? doc.importance_weight.toFixed(1) : '0.0'}`} />
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        ) : (
                                            <p className="text-sm text-gray-500 dark:text-gray-400">
                                                No document usage metadata was returned.
                                            </p>
                                        )}
                                    </div>

                                    <p className="text-xs text-gray-500 dark:text-gray-400">
                                        Search completed in {Number(result.time_seconds || 0).toFixed(1)} seconds
                                    </p>
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

function NumberField({
    label,
    value,
    onChange,
    min,
}: {
    label: string;
    value: number;
    onChange: (value: number) => void;
    min: number;
}) {
    return (
        <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                {label}
            </label>
            <input
                type="number"
                min={min}
                value={value}
                onChange={(e) => onChange(Math.max(min, Number(e.target.value) || min))}
                className="w-full rounded-xl border border-gray-300 bg-white px-4 py-2.5 text-sm outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500 dark:border-gray-700 dark:bg-gray-900 dark:text-white"
            />
        </div>
    );
}

function BalanceField({
    value,
    onChange,
}: {
    value: string;
    onChange: (value: string) => void;
}) {
    return (
        <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                balance_emphasis
            </label>
            <div className="flex gap-2">
                <input
                    type="text"
                    list="balance-emphasis-options"
                    value={value}
                    onChange={(e) => onChange(e.target.value)}
                    className="w-full rounded-xl border border-gray-300 bg-white px-4 py-2.5 text-sm outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500 dark:border-gray-700 dark:bg-gray-900 dark:text-white"
                />
                <datalist id="balance-emphasis-options">
                    {DEFAULT_BALANCE_OPTIONS.map((option) => (
                        <option key={option} value={option} />
                    ))}
                </datalist>
            </div>
        </div>
    );
}

function SummaryPill({ icon, label }: { icon: React.ReactNode; label: string }) {
    return (
        <span className="inline-flex items-center gap-1.5 rounded-full border border-gray-200 bg-white px-3 py-1 text-xs font-medium text-gray-700 dark:border-gray-700 dark:bg-gray-950/40 dark:text-gray-200">
            {icon}
            {label}
        </span>
    );
}

function Tag({ label }: { label: string }) {
    return (
        <span className="rounded-full bg-indigo-50 px-2.5 py-1 text-xs font-medium text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300">
            {label}
        </span>
    );
}
