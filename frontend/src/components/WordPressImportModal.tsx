import React, { useState, useEffect } from 'react';
import { X, Search, RefreshCw, Loader2, Globe, ExternalLink, CheckCircle2, ArrowRight, Sparkles, Tag, Layers, Edit3, Clock, AlertCircle } from 'lucide-react';
import { supabase } from '../lib/supabase';
import { useAuth } from '../context/auth-context';
import { syncWordPressPosts, importPostToTitles, importAllPostsToTitles, getImportedPosts } from '../services/wordpressService';
import { useNavigate } from 'react-router-dom';
import type { WordPressImportedPost } from '../types/wordpress';

interface WordPressImportModalProps {
    isOpen: boolean;
    onClose: () => void;
    onImportSuccess?: (importedTitleId?: string) => void;
}

export const WordPressImportModal: React.FC<WordPressImportModalProps> = ({
    isOpen,
    onClose,
    onImportSuccess
}) => {
    const { user } = useAuth();
    const navigate = useNavigate();

    const [posts, setPosts] = useState<WordPressImportedPost[]>([]);
    const [loading, setLoading] = useState(false);
    const [syncing, setSyncing] = useState(false);
    const [syncResult, setSyncResult] = useState<string | null>(null);
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
    const [importingId, setImportingId] = useState<string | null>(null);
    const [importingBatch, setImportingBatch] = useState(false);
    const [filterDomain, setFilterDomain] = useState<string>('all');
    const [filterWpStatus, setFilterWpStatus] = useState<string>('all');
    const [filterLibraryStatus, setFilterLibraryStatus] = useState<'all' | 'imported' | 'not_imported'>('all');

    useEffect(() => {
        if (isOpen && user) {
            fetchPosts();
        }
    }, [isOpen, user]);

    const fetchPosts = async () => {
        if (!user) return;
        try {
            setLoading(true);
            const apiPosts = await getImportedPosts(user.id);
            if (apiPosts && apiPosts.length > 0) {
                setPosts(apiPosts as WordPressImportedPost[]);
            } else {
                const { data, error } = await supabase
                    .from('wordpress_imported_posts')
                    .select('*')
                    .eq('user_id', user.id)
                    .order('published_at', { ascending: false, nullsFirst: false });

                if (!error && data) {
                    setPosts((data as WordPressImportedPost[]) || []);
                } else {
                    setPosts((apiPosts as WordPressImportedPost[]) || []);
                }
            }
        } catch (error) {
            console.error('Error fetching imported posts:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleSync = async () => {
        if (!user) return;
        try {
            setSyncing(true);
            setSyncResult(null);
            const res = await syncWordPressPosts(user.id, false);
            const msg = res?.message || `Synced ${res?.total_synced || 0} posts from WordPress`;
            setSyncResult(msg);
            await fetchPosts();
        } catch (error: any) {
            console.error('Error syncing posts:', error);
            setSyncResult(`Sync failed: ${error.message || 'Unknown error'}`);
        } finally {
            setSyncing(false);
        }
    };

    const handleLoadToEditor = async (post: WordPressImportedPost) => {
        if (!user) return;
        try {
            setImportingId(post.id);
            const res = await importPostToTitles({
                user_id: user.id,
                imported_post_id: post.id,
                post_id: post.post_id,
                wordpress_detail_id: post.wordpress_detail_id
            });

            if (res?.title_id) {
                setPosts(prev => prev.map(p => p.id === post.id ? { ...p, titles_record_id: res.title_id } : p));
                onImportSuccess?.(res.title_id);
                onClose();
                navigate(`/article-editor/${res.title_id}`);
            } else {
                alert('Failed to import post into Content Library');
            }
        } catch (error) {
            console.error('Error importing post to Titles:', error);
            alert('Failed to import post into Content Library');
        } finally {
            setImportingId(null);
        }
    };

    const handleLoadToStudio = async (post: WordPressImportedPost) => {
        if (!user) return;
        try {
            setImportingId(post.id);
            const res = await importPostToTitles({
                user_id: user.id,
                imported_post_id: post.id,
                post_id: post.post_id,
                wordpress_detail_id: post.wordpress_detail_id
            });

            if (res?.title_id) {
                setPosts(prev => prev.map(p => p.id === post.id ? { ...p, titles_record_id: res.title_id } : p));
                onImportSuccess?.(res.title_id);
                onClose();
                navigate(`/content-studio?id=${res.title_id}`);
            }
        } catch (error) {
            console.error('Error importing post to Studio:', error);
            alert('Failed to import post into Content Studio');
        } finally {
            setImportingId(null);
        }
    };

    const handleImportBatch = async () => {
        if (!user || selectedIds.size === 0) return;
        try {
            setImportingBatch(true);
            const res = await importAllPostsToTitles({
                user_id: user.id,
                imported_ids: Array.from(selectedIds)
            });

            if (res?.success) {
                setSelectedIds(new Set());
                await fetchPosts();
                onImportSuccess?.();
            }
        } catch (error) {
            console.error('Error batch importing posts:', error);
            alert('Failed to batch import posts');
        } finally {
            setImportingBatch(false);
        }
    };

    const toggleSelect = (id: string) => {
        setSelectedIds(prev => {
            const next = new Set(prev);
            if (next.has(id)) next.delete(id);
            else next.add(id);
            return next;
        });
    };

    const toggleSelectAll = (filteredList: WordPressImportedPost[]) => {
        const selectable = filteredList.filter(p => !p.titles_record_id);
        if (selectedIds.size >= selectable.length && selectable.length > 0) {
            setSelectedIds(new Set());
        } else {
            setSelectedIds(new Set(selectable.map(p => p.id)));
        }
    };

    const getWpStatusBadge = (rawStatus?: string) => {
        const s = String(rawStatus || 'publish').toLowerCase().trim();
        switch (s) {
            case 'publish':
                return { label: 'Published', className: 'bg-emerald-500/10 text-emerald-500 dark:text-emerald-400 border-emerald-500/20' };
            case 'draft':
                return { label: 'Draft', className: 'bg-amber-500/10 text-amber-500 dark:text-amber-400 border-amber-500/20' };
            case 'future':
                return { label: 'Scheduled', className: 'bg-blue-500/10 text-blue-500 dark:text-blue-400 border-blue-500/20' };
            case 'pending':
                return { label: 'Pending Review', className: 'bg-orange-500/10 text-orange-500 dark:text-orange-400 border-orange-500/20' };
            case 'private':
                return { label: 'Private', className: 'bg-zinc-500/10 text-zinc-400 border-zinc-500/20' };
            default:
                return { label: s.charAt(0).toUpperCase() + s.slice(1), className: 'bg-muted text-muted-foreground border-border' };
        }
    };

    // Filter available domains
    const uniqueDomains = Array.from(new Set(posts.map(p => p.source_site).filter(Boolean)));

    const filteredPosts = posts.filter(post => {
        if (filterDomain !== 'all' && post.source_site !== filterDomain) return false;

        // WP Status filter
        const postWpStatus = String(post.status || post.raw_post_json?.status || 'publish').toLowerCase().trim();
        if (filterWpStatus !== 'all' && postWpStatus !== filterWpStatus) return false;

        // Library status filter
        if (filterLibraryStatus === 'imported' && !post.titles_record_id) return false;
        if (filterLibraryStatus === 'not_imported' && post.titles_record_id) return false;

        if (searchQuery.trim()) {
            const q = searchQuery.toLowerCase();
            const titleMatch = post.title?.toLowerCase().includes(q);
            const kwMatch = post.focus_keyword?.toLowerCase().includes(q) || post.primary_keyword?.toLowerCase().includes(q);
            const catMatch = post.category_names?.some(c => c.toLowerCase().includes(q));
            const linkMatch = post.link?.toLowerCase().includes(q);
            return titleMatch || kwMatch || catMatch || linkMatch;
        }
        return true;
    });

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="flex max-h-[90vh] w-full max-w-5xl flex-col rounded-2xl border border-border bg-card shadow-2xl overflow-hidden">
                {/* Modal Header */}
                <div className="flex items-center justify-between border-b border-border px-6 py-4 bg-muted/30">
                    <div className="flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                            <Globe className="h-5 w-5" />
                        </div>
                        <div>
                            <h2 className="text-lg font-semibold text-foreground flex items-center gap-2">
                                Import from WordPress
                                <span className="inline-flex items-center gap-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 text-xs font-medium text-indigo-400">
                                    <Sparkles className="h-3 w-3" /> Ready for Editor
                                </span>
                            </h2>
                            <p className="text-xs text-muted-foreground">
                                Browse available articles from your WordPress sites, view and filter by status, and pick an article to read and edit.
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="rounded-lg p-2 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
                    >
                        <X className="h-5 w-5" />
                    </button>
                </div>

                {/* Toolbar */}
                <div className="border-b border-border bg-card/60 p-4 space-y-3">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                        {/* Search & Filters */}
                        <div className="flex flex-1 items-center gap-2 min-w-[280px] flex-wrap sm:flex-nowrap">
                            <div className="relative flex-1 min-w-[180px]">
                                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                                <input
                                    type="text"
                                    placeholder="Search by title, keyword, category, or URL..."
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    className="h-9 w-full rounded-lg border border-border bg-muted/40 pl-9 pr-3 text-xs text-foreground placeholder:text-muted-foreground focus:border-ring focus:outline-none"
                                />
                            </div>

                            {uniqueDomains.length > 0 && (
                                <select
                                    value={filterDomain}
                                    onChange={(e) => setFilterDomain(e.target.value)}
                                    className="h-9 rounded-lg border border-border bg-muted/40 px-2.5 text-xs text-foreground focus:border-ring focus:outline-none"
                                >
                                    <option value="all">All Sites</option>
                                    {uniqueDomains.map(d => (
                                        <option key={d} value={d}>{d}</option>
                                    ))}
                                </select>
                            )}

                            {/* Filter by WordPress Status */}
                            <select
                                value={filterWpStatus}
                                onChange={(e) => setFilterWpStatus(e.target.value)}
                                className="h-9 rounded-lg border border-border bg-muted/40 px-2.5 text-xs text-foreground focus:border-ring focus:outline-none font-medium"
                            >
                                <option value="all">All Statuses</option>
                                <option value="publish">Published</option>
                                <option value="draft">Drafts</option>
                                <option value="future">Scheduled</option>
                                <option value="pending">Pending Review</option>
                                <option value="private">Private</option>
                            </select>

                            {/* Filter by Library Status */}
                            <select
                                value={filterLibraryStatus}
                                onChange={(e) => setFilterLibraryStatus(e.target.value as any)}
                                className="h-9 rounded-lg border border-border bg-muted/40 px-2.5 text-xs text-foreground focus:border-ring focus:outline-none"
                            >
                                <option value="all">All Library States</option>
                                <option value="not_imported">Not Yet in Library</option>
                                <option value="imported">Already in Library</option>
                            </select>
                        </div>

                        {/* Sync Button */}
                        <div className="flex items-center gap-2">
                            <button
                                onClick={handleSync}
                                disabled={syncing}
                                className="inline-flex h-9 items-center gap-2 rounded-lg bg-indigo-600 px-3.5 text-xs font-medium text-white shadow-sm transition hover:bg-indigo-700 disabled:opacity-50"
                            >
                                <RefreshCw className={`h-3.5 w-3.5 ${syncing ? 'animate-spin' : ''}`} />
                                <span>{syncing ? 'Syncing…' : 'Sync WordPress'}</span>
                            </button>
                        </div>
                    </div>

                    {syncResult && (
                        <div className="text-xs px-3 py-1.5 rounded-lg bg-muted border border-border text-foreground flex items-center justify-between">
                            <span>{syncResult}</span>
                            <button onClick={() => setSyncResult(null)} className="text-muted-foreground hover:text-foreground">
                                <X className="h-3.5 w-3.5" />
                            </button>
                        </div>
                    )}
                </div>

                {/* Posts List */}
                <div className="flex-1 overflow-y-auto p-4 space-y-2.5 min-h-[320px]">
                    {loading ? (
                        <div className="flex flex-col items-center justify-center py-16 text-center text-muted-foreground gap-3">
                            <Loader2 className="h-7 w-7 animate-spin text-indigo-500" />
                            <p className="text-sm">Loading articles from WordPress...</p>
                        </div>
                    ) : filteredPosts.length === 0 ? (
                        <div className="flex flex-col items-center justify-center py-16 text-center text-muted-foreground gap-3">
                            <Globe className="h-10 w-10 text-muted-foreground/40" />
                            <div>
                                <h3 className="text-sm font-semibold text-foreground">No articles found</h3>
                                <p className="text-xs text-muted-foreground mt-1">
                                    {posts.length === 0
                                        ? "Click 'Sync WordPress' above to fetch available articles from your connected WordPress sites."
                                        : "No articles match the selected filters or search query."}
                                </p>
                            </div>
                        </div>
                    ) : (
                        filteredPosts.map((post) => {
                            const isImported = Boolean(post.titles_record_id);
                            const isSelected = selectedIds.has(post.id);
                            const isImporting = importingId === post.id;
                            const focusKw = post.focus_keyword || post.primary_keyword;
                            const postStatus = post.status || post.raw_post_json?.status || 'publish';
                            const statusBadge = getWpStatusBadge(postStatus);

                            return (
                                <div
                                    key={post.id}
                                    className={`group relative flex flex-col md:flex-row items-start md:items-center justify-between gap-3.5 rounded-xl border p-3.5 transition-all ${
                                        isImported
                                            ? 'border-indigo-500/20 bg-muted/20 hover:border-indigo-500/30'
                                            : isSelected
                                                ? 'border-indigo-500/40 bg-indigo-500/5'
                                                : 'border-border bg-card hover:border-border hover:bg-muted/30'
                                    }`}
                                >
                                    {/* Select Checkbox & Main Info */}
                                    <div className="flex items-start gap-3 flex-1 min-w-0">
                                        {!isImported && (
                                            <input
                                                type="checkbox"
                                                checked={isSelected}
                                                onChange={() => toggleSelect(post.id)}
                                                className="mt-1 h-4 w-4 rounded border-border text-indigo-600 focus:ring-indigo-500"
                                            />
                                        )}
                                        {isImported && (
                                            <CheckCircle2 className="mt-1 h-4 w-4 shrink-0 text-emerald-500" />
                                        )}

                                        <div className="flex-1 min-w-0 space-y-1">
                                            {/* Status Badge, Site Badge & External Link */}
                                            <div className="flex items-center gap-2 flex-wrap">
                                                <span className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-[11px] font-semibold ${statusBadge.className}`}>
                                                    {statusBadge.label}
                                                </span>

                                                <span className="text-[10px] font-bold uppercase px-2 py-0.5 bg-muted/50 rounded-md text-muted-foreground border border-border/50">
                                                    {post.source_site || 'WordPress'}
                                                </span>

                                                {isImported && (
                                                    <span className="inline-flex items-center gap-1 rounded-md bg-blue-500/10 border border-blue-500/20 px-2 py-0.5 text-[11px] font-medium text-blue-400">
                                                        In Library
                                                    </span>
                                                )}

                                                {post.link && (
                                                    <a
                                                        href={post.link}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        className="inline-flex items-center gap-1 text-[11px] text-muted-foreground hover:text-indigo-400 hover:underline"
                                                    >
                                                        <ExternalLink className="h-3 w-3" />
                                                        <span className="max-w-[180px] truncate">{post.link}</span>
                                                    </a>
                                                )}
                                            </div>

                                            {/* Title */}
                                            <h4 className="text-sm font-semibold text-foreground line-clamp-1 group-hover:text-indigo-400 transition-colors">
                                                {post.title}
                                            </h4>

                                            {/* SEO Description / Excerpt */}
                                            {(post.seo_description || post.excerpt) && (
                                                <p
                                                    className="text-xs text-muted-foreground line-clamp-2"
                                                    dangerouslySetInnerHTML={{ __html: post.seo_description || post.excerpt || '' }}
                                                />
                                            )}

                                            {/* Metadata Badges: Focus KW, Categories, Tags */}
                                            <div className="flex flex-wrap items-center gap-1.5 pt-1">
                                                {focusKw && (
                                                    <span className="inline-flex items-center gap-1 rounded-md bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 text-[11px] font-medium text-emerald-400">
                                                        <Sparkles className="h-2.5 w-2.5" />
                                                        KW: {focusKw}
                                                    </span>
                                                )}

                                                {post.category_names && post.category_names.length > 0 && (
                                                    <span className="inline-flex items-center gap-1 rounded-md bg-blue-500/10 border border-blue-500/20 px-2 py-0.5 text-[11px] text-blue-400">
                                                        <Layers className="h-2.5 w-2.5" />
                                                        {post.category_names.slice(0, 2).join(', ')}
                                                    </span>
                                                )}

                                                {post.tag_names && post.tag_names.length > 0 && (
                                                    <span className="inline-flex items-center gap-1 rounded-md bg-purple-500/10 border border-purple-500/20 px-2 py-0.5 text-[11px] text-purple-400">
                                                        <Tag className="h-2.5 w-2.5" />
                                                        {post.tag_names.slice(0, 2).join(', ')}
                                                    </span>
                                                )}

                                                {post.published_at && (
                                                    <span className="inline-flex items-center gap-1 text-[11px] text-muted-foreground/80">
                                                        <Clock className="h-2.5 w-2.5" />
                                                        {new Date(post.published_at).toLocaleDateString()}
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                    </div>

                                    {/* Action Buttons: Pick one and load to editor */}
                                    <div className="flex items-center gap-2 shrink-0 self-end md:self-center">
                                        {isImported ? (
                                            <>
                                                <button
                                                    onClick={() => {
                                                        onClose();
                                                        navigate(`/article-editor/${post.titles_record_id}`);
                                                    }}
                                                    className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-indigo-500/30 bg-indigo-500/10 px-3 text-xs font-semibold text-indigo-400 hover:bg-indigo-500/20 transition-colors shadow-sm"
                                                    title="Open in Article Editor to read, update images/links, and edit"
                                                >
                                                    <Edit3 className="h-3 w-3" />
                                                    <span>Open in Editor</span>
                                                </button>
                                                <button
                                                    onClick={() => {
                                                        onClose();
                                                        navigate(`/content-studio?id=${post.titles_record_id}`);
                                                    }}
                                                    className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-border bg-muted/40 px-2.5 text-xs font-medium text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
                                                >
                                                    <span>Studio</span>
                                                </button>
                                            </>
                                        ) : (
                                            <>
                                                <button
                                                    onClick={() => handleLoadToEditor(post)}
                                                    disabled={isImporting}
                                                    className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-indigo-500/40 bg-indigo-600 text-white px-3.5 text-xs font-semibold hover:bg-indigo-700 transition-colors disabled:opacity-50 shadow-sm"
                                                    title="Import into Content Library and open in Article Editor"
                                                >
                                                    {isImporting ? (
                                                        <>
                                                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                                            <span>Loading…</span>
                                                        </>
                                                    ) : (
                                                        <>
                                                            <Edit3 className="h-3 w-3" />
                                                            <span>Load into Editor</span>
                                                        </>
                                                    )}
                                                </button>
                                                <button
                                                    onClick={() => handleLoadToStudio(post)}
                                                    disabled={isImporting}
                                                    className="inline-flex h-8 items-center gap-1 rounded-lg border border-border bg-muted/40 px-2.5 text-xs font-medium text-muted-foreground hover:bg-muted hover:text-foreground transition-colors disabled:opacity-50"
                                                >
                                                    <span>Studio</span>
                                                </button>
                                            </>
                                        )}
                                    </div>
                                </div>
                            );
                        })
                    )}
                </div>

                {/* Footer Batch Actions */}
                <div className="flex items-center justify-between border-t border-border px-6 py-3.5 bg-muted/20">
                    <div className="flex items-center gap-3">
                        <button
                            onClick={() => toggleSelectAll(filteredPosts)}
                            className="text-xs text-muted-foreground hover:text-foreground transition-colors underline"
                        >
                            {selectedIds.size > 0 ? 'Deselect All' : 'Select All Available'}
                        </button>
                        {selectedIds.size > 0 && (
                            <span className="text-xs font-medium text-indigo-400">
                                {selectedIds.size} post{selectedIds.size > 1 ? 's' : ''} selected
                            </span>
                        )}
                    </div>

                    <div className="flex items-center gap-2">
                        <button
                            onClick={onClose}
                            className="rounded-lg border border-border bg-muted/40 px-4 py-2 text-xs font-medium text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
                        >
                            Close
                        </button>
                        {selectedIds.size > 0 && (
                            <button
                                onClick={handleImportBatch}
                                disabled={importingBatch}
                                className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-xs font-medium text-white shadow-sm hover:bg-indigo-700 disabled:opacity-50 transition-colors"
                            >
                                {importingBatch ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
                                <span>Import Selected ({selectedIds.size}) to Library</span>
                            </button>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};
