import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Search, Loader2, BookOpen, ExternalLink, Sparkles, CheckCircle2, AlertCircle, ArrowRight, Globe } from 'lucide-react';
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
    const [error, setError] = useState<string | null>(null);
    const [previewArticle, setPreviewArticle] = useState<EditorialArticle | null>(null);

    useEffect(() => {
        if (activeProject?.domain && !selectedDomain) {
            setSelectedDomain(activeProject.domain);
        }
    }, [activeProject?.domain]);

    const fetchArticles = async (query = '') => {
        setLoading(true);
        setError(null);
        try {
            const data = await editorialFactoryService.listArticles({ search: query, limit: 50 });
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
            void fetchArticles(searchQuery);
        }
    }, [isOpen]);

    const handleSearchSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        void fetchArticles(searchQuery);
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

                    {/* Toolbar: Search + Project Domain selector */}
                    <div className="px-6 py-3 border-b border-border bg-muted/10 flex flex-col sm:flex-row gap-3 items-center justify-between">
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

                        <div className="flex items-center gap-2 w-full sm:w-auto">
                            <Globe className="w-4 h-4 text-muted-foreground shrink-0" />
                            <select
                                value={selectedDomain}
                                onChange={(e) => setSelectedDomain(e.target.value)}
                                className="px-3 py-2 text-xs rounded-xl border border-border bg-background text-foreground focus:ring-2 focus:ring-ring outline-none"
                                title="Target Project Domain"
                            >
                                <option value="">No Domain (Unassigned)</option>
                                {projects.map((p) => (
                                    <option key={p.id} value={p.domain || p.app_name}>
                                        {p.domain || p.app_name}
                                    </option>
                                ))}
                            </select>
                            <button
                                type="button"
                                onClick={() => void fetchArticles(searchQuery)}
                                disabled={loading}
                                className="px-3 py-2 text-xs font-semibold rounded-xl bg-primary text-primary-foreground hover:bg-primary/90 transition disabled:opacity-50"
                            >
                                {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : 'Refresh'}
                            </button>
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
                        ) : articles.length === 0 ? (
                            <div className="text-center py-16 space-y-3">
                                <BookOpen className="w-10 h-10 mx-auto text-muted-foreground/50" />
                                <h3 className="text-base font-semibold text-foreground">No Editorial Articles Found</h3>
                                <p className="text-xs text-muted-foreground max-w-md mx-auto">
                                    No articles matched your search query in the secondary Supabase database. Try clearing the search or adding articles in Editorial Factory.
                                </p>
                            </div>
                        ) : (
                            articles.map((article) => (
                                <div
                                    key={article.id}
                                    className="p-4 rounded-2xl border border-border bg-card/60 hover:bg-muted/30 transition flex flex-col sm:flex-row gap-4 items-start justify-between"
                                >
                                    <div className="flex-1 space-y-1.5 min-w-0">
                                        <div className="flex items-center gap-2">
                                            <h3 className="font-semibold text-foreground text-sm truncate">
                                                {article.title}
                                            </h3>
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

                                    <div className="shrink-0 flex items-center gap-2 self-end sm:self-center">
                                        <button
                                            type="button"
                                            onClick={() => setPreviewArticle(previewArticle?.id === article.id ? null : article)}
                                            className="px-3 py-1.5 text-xs font-medium rounded-xl border border-border hover:bg-muted transition text-foreground"
                                        >
                                            {previewArticle?.id === article.id ? 'Hide Preview' : 'Preview'}
                                        </button>
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
                            ))
                        )}
                    </div>

                    {/* Footer */}
                    <div className="px-6 py-3 border-t border-border bg-muted/20 flex items-center justify-between text-xs text-muted-foreground">
                        <span>Database: {projects.find(p => p.domain === selectedDomain)?.domain || 'Editorial Factory'}</span>
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
