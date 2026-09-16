import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Search, Loader2, BookOpen, ExternalLink, Sparkles, CheckCircle2, AlertCircle, ArrowRight, Globe, ArrowUpDown, Calendar, Filter } from 'lucide-react';
import { editorialFactoryService, type EditorialArticle } from '../services/editorial-factory.service';
import { useProject } from '../context/project-context';

interface EditorialImportModalProps {
    isOpen: boolean;
    onClose: () => void;
    onImportSuccess: (titleId: string) => void;
}

export const EditorialImportModal: React.FC<EditorialImportModalProps> = ({
    isOpen,
    onClose,
    onImportSuccess,
}) => {
    const { activeProject, projects } = useProject();
    const [articles, setArticles] = useState<EditorialArticle[]>([]);
    const [loading, setLoading] = useState(false);
    const [importingId, setImportingId] = useState<string | null>(null);
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedDomain, setSelectedDomain] = useState<string>(activeProject?.domain || '');
    const [filterStatus, setFilterStatus] = useState<'all' | 'new' | 'imported'>('all');
    const [sortBy, setSortBy] = useState<'date_desc' | 'date_asc' | 'title_asc' | 'word_count_desc'>('date_desc');
    const [error, setError] = useState<string | null>(null);
    const [previewArticle, setPreviewArticle] = useState<EditorialArticle | null>(null);

    useEffect(() => {
        if (activeProject?.domain && !selectedDomain) {
            setSelectedDomain(activeProject.domain);
        }
    }, [activeProject?.domain]);

    const fetchArticles = async (query = '', domain = selectedDomain) => {
        setLoading(true);
        setError(null);
        try {
            const data = await editorialFactoryService.listArticles({
                search: query,
                limit: 50,
                domain: domain || undefined,
            });
            setArticles(data);
        } catch (err: any) {
            const msg = err.response?.data?.error || err.message || 'Failed to connect to Editorial Factory database.';
            setError(msg);
            setArticles([]);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (isOpen) {
            void fetchArticles(searchQuery, selectedDomain);
        }
    }, [isOpen, selectedDomain]);

    const handleSearchSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        void fetchArticles(searchQuery, selectedDomain);
    };

    const handleImport = async (article: EditorialArticle) => {
        setImportingId(article.id);
        setError(null);
        try {
            const res = await editorialFactoryService.importArticle({
                article_id: article.id,
                domain: selectedDomain || undefined,
            });

            if (res.success && res.title_id) {
                // Update local state to mark article as imported
                setArticles(prev => prev.map(a => a.id === article.id ? {
                    ...a,
                    is_imported: true,
                    imported_title_id: res.title_id,
                    imported_at: new Date().toISOString(),
                } : a));
                onClose();
                onImportSuccess(res.title_id);
            } else {
                setError(res.error || 'Failed to import article.');
            }
        } catch (err: any) {
            setError(err.message || 'An error occurred during import.');
        } finally {
            setImportingId(null);
        }
    };

    const counts = {
        all: articles.length,
        new: articles.filter(a => !a.is_imported).length,
        imported: articles.filter(a => a.is_imported).length,
    };

    const filteredArticles = useMemo(() => {
        const filtered = articles.filter(article => {
            if (filterStatus === 'new' && article.is_imported) return false;
            if (filterStatus === 'imported' && !article.is_imported) return false;
            return true;
        });

        return [...filtered].sort((a, b) => {
            if (sortBy === 'date_desc') {
                const dateA = new Date(a.created_at).getTime() || 0;
                const dateB = new Date(b.created_at).getTime() || 0;
                return dateB - dateA;
            }
            if (sortBy === 'date_asc') {
                const dateA = new Date(a.created_at).getTime() || 0;
                const dateB = new Date(b.created_at).getTime() || 0;
                return dateA - dateB;
            }
            if (sortBy === 'title_asc') {
                return (a.title || '').localeCompare(b.title || '');
            }
            if (sortBy === 'word_count_desc') {
                return (b.word_count || 0) - (a.word_count || 0);
            }
            return 0;
        });
    }, [articles, filterStatus, sortBy]);

    if (!isOpen) return null;

    return (
        <AnimatePresence>
            <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-sm">
                <motion.div
                    initial={{ opacity: 0, scale: 0.95, y: 10 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.95, y: 10 }}
                    className="relative w-full max-w-4xl max-h-[90vh] flex flex-col rounded-2xl border border-border bg-card shadow-2xl overflow-hidden"
                >
                    {/* Header */}
                    <div className="flex items-center justify-between px-6 py-4 border-b border-border bg-muted/20">
                        <div className="flex items-center gap-2.5">
                            <div className="p-2 rounded-xl bg-primary/10 border border-primary/20 text-primary">
                                <BookOpen className="w-5 h-5" />
                            </div>
                            <div>
                                <h2 className="text-lg font-bold text-foreground">Import from Editorial Factory</h2>
                                <p className="text-xs text-muted-foreground">
                                    Browse articles from your secondary Supabase database, copy them into the editor, and enrich with illustrations & SEO.
                                </p>
                            </div>
                        </div>
                        <button
                            onClick={onClose}
                            className="p-2 text-muted-foreground hover:text-foreground hover:bg-muted rounded-xl transition"
                        >
                            <X className="w-5 h-5" />
                        </button>
                    </div>

                    {/* Toolbar: Search + Status Filter + Sort By + Project Domain selector */}
                    <div className="px-6 py-3 border-b border-border bg-muted/10 flex flex-col gap-2.5">
                        {/* Row 1: Search */}
                        <div className="flex flex-col sm:flex-row gap-2.5 items-center justify-between">
                            <form onSubmit={handleSearchSubmit} className="relative flex-1 w-full">
                                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                                <input
                                    type="text"
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    placeholder="Search articles by title or keyword..."
                                    className="w-full pl-9 pr-4 py-2 text-sm rounded-xl border border-border bg-background text-foreground focus:ring-2 focus:ring-ring outline-none"
                                />
                            </form>

                            <button
                                type="button"
                                onClick={() => void fetchArticles(searchQuery, selectedDomain)}
                                disabled={loading}
                                className="w-full sm:w-auto px-4 py-2 text-xs font-semibold rounded-xl bg-primary text-primary-foreground hover:bg-primary/90 transition disabled:opacity-50 shrink-0 flex items-center justify-center gap-1.5"
                            >
                                {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : 'Refresh'}
                            </button>
                        </div>

                        {/* Row 2: Status Filter Pills + Sort Dropdown + Domain Selector */}
                        <div className="flex flex-wrap items-center justify-between gap-2 pt-0.5 text-xs">
                            {/* Status Filter Tabs */}
                            <div className="flex items-center gap-1 bg-background border border-border rounded-xl p-1 shrink-0">
                                <button
                                    type="button"
                                    onClick={() => setFilterStatus('all')}
                                    className={`px-3 py-1 rounded-lg font-medium transition ${
                                        filterStatus === 'all'
                                            ? 'bg-primary text-primary-foreground shadow-sm'
                                            : 'text-muted-foreground hover:text-foreground'
                                    }`}
                                >
                                    All ({counts.all})
                                </button>
                                <button
                                    type="button"
                                    onClick={() => setFilterStatus('new')}
                                    className={`px-3 py-1 rounded-lg font-medium transition flex items-center gap-1 ${
                                        filterStatus === 'new'
                                            ? 'bg-primary text-primary-foreground shadow-sm'
                                            : 'text-muted-foreground hover:text-foreground'
                                    }`}
                                >
                                    <Sparkles className="w-3 h-3" />
                                    New Only ({counts.new})
                                </button>
                                <button
                                    type="button"
                                    onClick={() => setFilterStatus('imported')}
                                    className={`px-3 py-1 rounded-lg font-medium transition flex items-center gap-1 ${
                                        filterStatus === 'imported'
                                            ? 'bg-emerald-600 text-white shadow-sm'
                                            : 'text-muted-foreground hover:text-foreground'
                                    }`}
                                >
                                    <CheckCircle2 className="w-3 h-3" />
                                    Already Imported ({counts.imported})
                                </button>
                            </div>

                            {/* Sort & Domain Controls */}
                            <div className="flex items-center gap-2 flex-wrap">
                                {/* Sort dropdown */}
                                <div className="flex items-center gap-1.5 border border-border rounded-xl px-2.5 py-1 bg-background">
                                    <ArrowUpDown className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
                                    <select
                                        value={sortBy}
                                        onChange={(e) => setSortBy(e.target.value as any)}
                                        className="py-1 text-xs bg-transparent text-foreground focus:ring-0 outline-none font-medium cursor-pointer"
                                        title="Sort articles"
                                    >
                                        <option value="date_desc">Newest First (Created Date)</option>
                                        <option value="date_asc">Oldest First (Created Date)</option>
                                        <option value="title_asc">Title (A to Z)</option>
                                        <option value="word_count_desc">Word Count (High to Low)</option>
                                    </select>
                                </div>

                                {/* Domain selector */}
                                <div className="flex items-center gap-1.5 border border-border rounded-xl px-2.5 py-1 bg-background">
                                    <Globe className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
                                    <select
                                        value={selectedDomain}
                                        onChange={(e) => setSelectedDomain(e.target.value)}
                                        className="py-1 text-xs bg-transparent text-foreground focus:ring-0 outline-none cursor-pointer"
                                        title="Target Project Domain"
                                    >
                                        <option value="">No Domain (Unassigned)</option>
                                        {projects.map((p) => (
                                            <option key={p.id} value={p.domain || p.app_name}>
                                                {p.domain || p.app_name}
                                            </option>
                                        ))}
                                    </select>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Error Banner */}
                    {error && (
                        <div className="mx-6 mt-4 p-3 rounded-xl bg-destructive/10 border border-destructive/20 text-destructive text-xs flex items-center gap-2">
                            <AlertCircle className="w-4 h-4 shrink-0" />
                            <span>{error}</span>
                        </div>
                    )}

                    {/* Content List */}
                    <div className="flex-1 overflow-y-auto p-6 space-y-4">
                        {loading ? (
                            <div className="flex flex-col items-center justify-center py-16 space-y-3">
                                <Loader2 className="w-8 h-8 animate-spin text-primary" />
                                <p className="text-sm text-muted-foreground">Connecting to Editorial Factory database...</p>
                            </div>
                        ) : filteredArticles.length === 0 ? (
                            <div className="text-center py-16 space-y-3">
                                <BookOpen className="w-10 h-10 mx-auto text-muted-foreground/50" />
                                <h3 className="text-base font-semibold text-foreground">
                                    {filterStatus === 'new'
                                        ? 'No New Articles to Import'
                                        : filterStatus === 'imported'
                                            ? 'No Articles Imported Yet'
                                            : 'No Editorial Articles Found'}
                                </h3>
                                <p className="text-xs text-muted-foreground max-w-md mx-auto">
                                    {filterStatus === 'new'
                                        ? 'All available articles from this database have already been imported into your library.'
                                        : filterStatus === 'imported'
                                            ? 'You haven\'t imported any articles from Editorial Factory yet.'
                                            : 'No articles matched your search query in the secondary Supabase database. Try clearing the search or adding articles in Editorial Factory.'}
                                </p>
                            </div>
                        ) : (
                            filteredArticles.map((article) => {
                                const isImported = Boolean(article.is_imported);

                                return (
                                    <div
                                        key={article.id}
                                        className={`p-4 rounded-2xl border transition flex flex-col sm:flex-row gap-4 items-start justify-between ${
                                            isImported
                                                ? 'border-emerald-500/30 bg-emerald-500/[0.03] hover:bg-emerald-500/[0.06]'
                                                : 'border-border bg-card/60 hover:bg-muted/30'
                                        }`}
                                    >
                                        <div className="flex-1 space-y-1.5 min-w-0">
                                            <div className="flex items-center gap-2 flex-wrap">
                                                <h3 className="font-semibold text-foreground text-sm truncate">
                                                    {article.title}
                                                </h3>
                                                {isImported ? (
                                                    <span className="inline-flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30 shrink-0">
                                                        <CheckCircle2 className="w-3 h-3" />
                                                        Already Imported
                                                    </span>
                                                ) : (
                                                    <span className="inline-flex items-center gap-1 text-[10px] font-medium px-2 py-0.5 rounded-full bg-primary/10 text-primary border border-primary/20 shrink-0">
                                                        <Sparkles className="w-2.5 h-2.5" />
                                                        New
                                                    </span>
                                                )}
                                                {article.word_count ? (
                                                    <span className="shrink-0 text-[10px] font-medium px-2 py-0.5 rounded-full bg-muted text-muted-foreground border border-border">
                                                        ~{article.word_count} words
                                                    </span>
                                                ) : null}
                                            </div>

                                            {article.summary && (
                                                <p className="text-xs text-muted-foreground line-clamp-2">
                                                    {article.summary}
                                                </p>
                                            )}

                                            <div className="flex flex-wrap items-center gap-2 pt-1 text-[11px] text-muted-foreground">
                                                <span>By {article.author || 'Editorial Factory'}</span>
                                                <span>·</span>
                                                <span>{new Date(article.created_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })}</span>
                                                {article.tags && article.tags.length > 0 && (
                                                    <>
                                                        <span>·</span>
                                                        <div className="flex flex-wrap gap-1">
                                                            {article.tags.slice(0, 3).map((tag, i) => (
                                                                <span key={i} className="px-1.5 py-0.5 rounded bg-primary/10 text-primary border border-primary/20 text-[10px]">
                                                                    {tag}
                                                                </span>
                                                            ))}
                                                        </div>
                                                    </>
                                                )}
                                            </div>
                                        </div>

                                        <div className="shrink-0 flex items-center gap-2 self-end sm:self-center flex-wrap justify-end">
                                            <button
                                                type="button"
                                                onClick={() => setPreviewArticle(previewArticle?.id === article.id ? null : article)}
                                                className="px-3 py-1.5 text-xs font-medium rounded-xl border border-border hover:bg-muted transition text-foreground"
                                            >
                                                {previewArticle?.id === article.id ? 'Hide Preview' : 'Preview'}
                                            </button>

                                            {isImported ? (
                                                <>
                                                    {article.imported_title_id && (
                                                        <button
                                                            type="button"
                                                            onClick={() => {
                                                                onClose();
                                                                onImportSuccess(article.imported_title_id!);
                                                            }}
                                                            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-xl border border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 hover:bg-emerald-500/20 transition"
                                                            title="Open the existing article in Article Editor"
                                                        >
                                                            <span>Open in Editor</span>
                                                            <ArrowRight className="w-3.5 h-3.5" />
                                                        </button>
                                                    )}
                                                    <button
                                                        type="button"
                                                        onClick={() => handleImport(article)}
                                                        disabled={importingId === article.id}
                                                        className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-xl border border-border bg-background hover:bg-muted text-foreground transition shadow-sm disabled:opacity-50"
                                                        title="Import again as a fresh copy"
                                                    >
                                                        {importingId === article.id ? (
                                                            <>
                                                                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                                                <span>Importing...</span>
                                                            </>
                                                        ) : (
                                                            <>
                                                                <Sparkles className="w-3.5 h-3.5 text-primary" />
                                                                <span>Import Again</span>
                                                            </>
                                                        )}
                                                    </button>
                                                </>
                                            ) : (
                                                <button
                                                    type="button"
                                                    onClick={() => handleImport(article)}
                                                    disabled={importingId === article.id}
                                                    className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold rounded-xl bg-primary text-primary-foreground hover:bg-primary/90 transition shadow-sm disabled:opacity-50"
                                                >
                                                    {importingId === article.id ? (
                                                        <>
                                                            <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                                            <span>Importing...</span>
                                                        </>
                                                    ) : (
                                                        <>
                                                            <Sparkles className="w-3.5 h-3.5" />
                                                            <span>Import & Edit</span>
                                                            <ArrowRight className="w-3.5 h-3.5" />
                                                        </>
                                                    )}
                                                </button>
                                            )}
                                        </div>

                                        {/* Inline Preview */}
                                        {previewArticle?.id === article.id && (
                                            <div className="w-full mt-3 p-4 rounded-xl border border-border/80 bg-background/90 text-xs space-y-2">
                                                <div className="font-semibold text-foreground">Content Preview:</div>
                                                <div className="max-h-48 overflow-y-auto font-mono text-[11px] text-muted-foreground whitespace-pre-wrap">
                                                    {article.content.slice(0, 1000)}...
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                );
                            })
                        )}
                    </div>

                    {/* Footer */}
                    <div className="px-6 py-3 border-t border-border bg-muted/20 flex items-center justify-between text-xs text-muted-foreground">
                        <span>
                            Showing {filteredArticles.length} of {articles.length} article{articles.length === 1 ? '' : 's'}
                            {counts.imported > 0 && ` (${counts.imported} already imported)`}
                        </span>
                        <button
                            onClick={onClose}
                            className="px-4 py-1.5 rounded-xl border border-border hover:bg-muted transition text-foreground font-medium"
                        >
                            Close
                        </button>
                    </div>
                </motion.div>
            </div>
        </AnimatePresence>
    );
};
