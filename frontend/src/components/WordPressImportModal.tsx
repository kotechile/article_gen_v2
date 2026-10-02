import React, { useState, useEffect, useMemo } from 'react';
import {
    X,
    Search,
    RefreshCw,
    Loader2,
    Globe,
    ExternalLink,
    CheckCircle2,
    Sparkles,
    Tag,
    Layers,
    Edit3,
    Clock,
    Calendar,
    ArrowUpDown,
    FileText,
    Check
} from 'lucide-react';
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
    const [sortBy, setSortBy] = useState<'newest' | 'oldest' | 'status' | 'title'>('newest');

    useEffect(() => {
        if (isOpen && user) {
            // Immediately load cached posts for instant UI display
            fetchPosts();
            // Automatically sync live with WordPress in background to ensure all articles, statuses, and dates are fresh
            handleSync(true);
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
                    .order('created_at', { ascending: false });

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

    const handleSync = async (silent = false) => {
        if (!user) return;
        try {
            setSyncing(true);
            if (!silent) setSyncResult(null);
            const res = await syncWordPressPosts(user.id, false);
            const count = res?.total_synced ?? 0;
            const msg = res?.details || `Fetched ${count} articles across all statuses from WordPress`;
            setSyncResult(msg);
            await fetchPosts();
        } catch (error: any) {
            console.error('Error syncing posts from WordPress:', error);
            if (!silent) {
                setSyncResult(`Sync failed: ${error.message || 'Check WordPress credentials in Settings'}`);
            }
        } finally {
            setSyncing(false);
        }
    };

    const handleLoadToEditor = async (post: WordPressImportedPost) => {
        if (!user) return;
        try {
            setImportingId(String(post.id));
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
        } catch (error: any) {
            console.error('Error importing post to Titles:', error);
            const msg = error?.response?.data?.error || error?.message || 'Failed to import post into Content Library';
            alert(`Failed to import post: ${msg}`);
        } finally {
            setImportingId(null);
        }
    };

    const handleLoadToStudio = async (post: WordPressImportedPost) => {
        if (!user) return;
        try {
            setImportingId(String(post.id));
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
        } catch (error: any) {
            console.error('Error importing post to Studio:', error);
            const msg = error?.response?.data?.error || error?.message || 'Failed to import post into Content Studio';
            alert(`Failed to import post: ${msg}`);
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
                return { label: 'Published', className: 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20' };
            case 'draft':
                return { label: 'Draft', className: 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20' };
            case 'future':
                return { label: 'Scheduled', className: 'bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20' };
            case 'pending':
                return { label: 'Pending Review', className: 'bg-orange-500/10 text-orange-600 dark:text-orange-400 border-orange-500/20' };
            case 'private':
                return { label: 'Private', className: 'bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/20' };
            default:
                return { label: s.charAt(0).toUpperCase() + s.slice(1), className: 'bg-muted text-muted-foreground border-border' };
        }
    };

    const formatDateDisplay = (dateStr?: string) => {
        if (!dateStr) return null;
        try {
            const d = new Date(dateStr);
            if (isNaN(d.getTime())) return null;
            return d.toLocaleDateString(undefined, {
                year: 'numeric',
                month: 'short',
                day: 'numeric'
            });
        } catch {
            return null;
        }
    };

    // Filter available domains
    const uniqueDomains = useMemo(() => {
        const set = new Set<string>();
        posts.forEach(p => {
            if (p.source_site) set.add(p.source_site);
            else if (p.link) {
                try {
                    set.add(new URL(p.link).hostname.replace('cms.', ''));
                } catch {}
            }
        });
        return Array.from(set).filter(Boolean);
    }, [posts]);

    // Quick status counts
    const counts = useMemo(() => {
        const total = posts.length;
        const published = posts.filter(p => (p.status || 'publish').toLowerCase() === 'publish').length;
        const drafts = posts.filter(p => (p.status || '').toLowerCase() === 'draft').length;
        const future = posts.filter(p => (p.status || '').toLowerCase() === 'future').length;
        const pending = posts.filter(p => ['pending', 'private'].includes((p.status || '').toLowerCase())).length;
        const inLibrary = posts.filter(p => Boolean(p.titles_record_id)).length;
        const notInLibrary = total - inLibrary;
        return { total, published, drafts, future, pending, inLibrary, notInLibrary };
    }, [posts]);

    // Filter posts
    const filteredPosts = useMemo(() => {
        return posts.filter(post => {
            if (filterDomain !== 'all') {
                const postDomain = (post.source_site || (post.link ? new URL(post.link).hostname.replace('cms.', '') : '')).toLowerCase();
                if (postDomain !== filterDomain.toLowerCase()) return false;
            }

            // WP Status filter
            const postWpStatus = String(post.status || post.raw_post_json?.status || 'publish').toLowerCase().trim();
            if (filterWpStatus !== 'all') {
                if (filterWpStatus === 'pending_or_private') {
                    if (!['pending', 'private'].includes(postWpStatus)) return false;
                } else if (postWpStatus !== filterWpStatus) {
                    return false;
                }
            }

            // Library status filter
            if (filterLibraryStatus === 'imported' && !post.titles_record_id) return false;
            if (filterLibraryStatus === 'not_imported' && post.titles_record_id) return false;

            if (searchQuery.trim()) {
                const rawQ = searchQuery.toLowerCase().trim();
                // Strip protocol and cms. prefix if user pasted a URL
                const q = rawQ.replace(/^https?:\/\/(?:cms\.)?/, '').replace(/\/$/, '');
                const titleMatch = post.title?.toLowerCase().includes(q) || post.title?.toLowerCase().includes(rawQ);
                const slugMatch = post.slug?.toLowerCase().includes(q) || post.slug?.toLowerCase().includes(rawQ);
                const kwMatch = post.focus_keyword?.toLowerCase().includes(rawQ) || post.primary_keyword?.toLowerCase().includes(rawQ);
                const catMatch = post.category_names?.some(c => c.toLowerCase().includes(rawQ));
                const tagMatch = post.tag_names?.some(t => t.toLowerCase().includes(rawQ));
                const linkMatch = post.link?.toLowerCase().includes(q) || post.link?.toLowerCase().includes(rawQ);
                const excerptMatch = post.excerpt?.toLowerCase().includes(rawQ);
                const idMatch = String(post.post_id) === rawQ;
                return titleMatch || slugMatch || kwMatch || catMatch || tagMatch || linkMatch || excerptMatch || idMatch;
            }
            return true;
        });
    }, [posts, filterDomain, filterWpStatus, filterLibraryStatus, searchQuery]);

    // Sort posts
    const sortedPosts = useMemo(() => {
        return [...filteredPosts].sort((a, b) => {
            if (sortBy === 'newest') {
                const dateA = new Date(a.published_at || a.created_at || a.modified_at || 0).getTime();
                const dateB = new Date(b.published_at || b.created_at || b.modified_at || 0).getTime();
                return dateB - dateA;
            }
            if (sortBy === 'oldest') {
                const dateA = new Date(a.published_at || a.created_at || 0).getTime();
                const dateB = new Date(b.published_at || b.created_at || 0).getTime();
                return dateA - dateB;
            }
            if (sortBy === 'title') {
                return (a.title || '').localeCompare(b.title || '');
            }
            if (sortBy === 'status') {
                const statusOrder: Record<string, number> = { draft: 1, future: 2, pending: 3, private: 4, publish: 5 };
                const orderA = statusOrder[String(a.status).toLowerCase()] || 99;
                const orderB = statusOrder[String(b.status).toLowerCase()] || 99;
                return orderA - orderB;
            }
            return 0;
        });
    }, [filteredPosts, sortBy]);

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="flex max-h-[92vh] w-full max-w-5xl flex-col rounded-2xl border border-border bg-card shadow-2xl overflow-hidden">
                {/* Modal Header */}
                <div className="flex items-center justify-between border-b border-border px-6 py-4 bg-muted/30">
                    <div className="flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                            <Globe className="h-5 w-5" />
                        </div>
                        <div>
                            <div className="flex items-center gap-2">
                                <h2 className="text-lg font-semibold text-foreground">
                                    Import from WordPress
                                </h2>
                                <span className="inline-flex items-center gap-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 text-xs font-medium text-indigo-400">
                                    <Sparkles className="h-3 w-3" /> Live Reader & Editor
                                </span>
                            </div>
                            <p className="text-xs text-muted-foreground mt-0.5">
                                Select any article from your WordPress site to read, load its media and links, and edit it.
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

                {/* Auto-sync Notification Banner */}
                {syncing && (
                    <div className="flex items-center justify-between px-6 py-2 bg-indigo-500/10 border-b border-indigo-500/20 text-xs text-indigo-400 animate-pulse">
                        <div className="flex items-center gap-2">
                            <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                            <span>Reading all articles, statuses, and dates from your WordPress site...</span>
                        </div>
                    </div>
                )}

                {/* Toolbar */}
                <div className="border-b border-border bg-card/60 p-4 space-y-3">
                    {/* Quick Status Filter Tabs */}
                    <div className="flex items-center gap-1.5 flex-wrap text-xs">
                        <button
                            onClick={() => setFilterWpStatus('all')}
                            className={`px-3 py-1.5 rounded-lg font-medium transition-colors ${
                                filterWpStatus === 'all'
                                    ? 'bg-indigo-600 text-white shadow-sm'
                                    : 'bg-muted/60 text-muted-foreground hover:bg-muted hover:text-foreground'
                            }`}
                        >
                            All ({counts.total})
                        </button>
                        <button
                            onClick={() => setFilterWpStatus('publish')}
                            className={`px-3 py-1.5 rounded-lg font-medium transition-colors ${
                                filterWpStatus === 'publish'
                                    ? 'bg-emerald-600 text-white shadow-sm'
                                    : 'bg-muted/60 text-muted-foreground hover:bg-muted hover:text-foreground'
                            }`}
                        >
                            Published ({counts.published})
                        </button>
                        <button
                            onClick={() => setFilterWpStatus('draft')}
                            className={`px-3 py-1.5 rounded-lg font-medium transition-colors ${
                                filterWpStatus === 'draft'
                                    ? 'bg-amber-600 text-white shadow-sm'
                                    : 'bg-muted/60 text-muted-foreground hover:bg-muted hover:text-foreground'
                            }`}
                        >
                            Drafts ({counts.drafts})
                        </button>
                        <button
                            onClick={() => setFilterWpStatus('future')}
                            className={`px-3 py-1.5 rounded-lg font-medium transition-colors ${
                                filterWpStatus === 'future'
                                    ? 'bg-blue-600 text-white shadow-sm'
                                    : 'bg-muted/60 text-muted-foreground hover:bg-muted hover:text-foreground'
                            }`}
                        >
                            Scheduled ({counts.future})
                        </button>
                        {counts.pending > 0 && (
                            <button
                                onClick={() => setFilterWpStatus('pending_or_private')}
                                className={`px-3 py-1.5 rounded-lg font-medium transition-colors ${
                                    filterWpStatus === 'pending_or_private'
                                        ? 'bg-orange-600 text-white shadow-sm'
                                        : 'bg-muted/60 text-muted-foreground hover:bg-muted hover:text-foreground'
                                }`}
                            >
                                Pending/Private ({counts.pending})
                            </button>
                        )}
                        <span className="w-px h-4 bg-border mx-1" />
                        <button
                            onClick={() => setFilterLibraryStatus(prev => prev === 'not_imported' ? 'all' : 'not_imported')}
                            className={`px-3 py-1.5 rounded-lg font-medium transition-colors ${
                                filterLibraryStatus === 'not_imported'
                                    ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/40'
                                    : 'bg-muted/40 text-muted-foreground hover:bg-muted hover:text-foreground'
                            }`}
                        >
                            Available ({counts.notInLibrary})
                        </button>
                        <button
                            onClick={() => setFilterLibraryStatus(prev => prev === 'imported' ? 'all' : 'imported')}
                            className={`px-3 py-1.5 rounded-lg font-medium transition-colors ${
                                filterLibraryStatus === 'imported'
                                    ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/40'
                                    : 'bg-muted/40 text-muted-foreground hover:bg-muted hover:text-foreground'
                            }`}
                        >
                            In Library ({counts.inLibrary})
                        </button>
                    </div>

                    <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
                        {/* Search & Site Filters */}
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

                            {uniqueDomains.length > 1 && (
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

                            {/* Sort Dropdown */}
                            <div className="flex items-center gap-1.5 bg-muted/40 border border-border rounded-lg px-2 py-0.5 h-9">
                                <ArrowUpDown className="h-3.5 w-3.5 text-muted-foreground" />
                                <select
                                    value={sortBy}
                                    onChange={(e) => setSortBy(e.target.value as any)}
                                    className="bg-transparent text-xs text-foreground focus:outline-none pr-1 cursor-pointer font-medium"
                                >
                                    <option value="newest">Newest First</option>
                                    <option value="oldest">Oldest First</option>
                                    <option value="status">Drafts First</option>
                                    <option value="title">Title (A-Z)</option>
                                </select>
                            </div>
                        </div>

                        {/* Force Refresh Button */}
                        <div className="flex items-center gap-2">
                            <button
                                onClick={() => handleSync(false)}
                                disabled={syncing}
                                className="inline-flex h-9 items-center gap-2 rounded-lg bg-indigo-600 px-3.5 text-xs font-medium text-white shadow-sm transition hover:bg-indigo-700 disabled:opacity-50"
                                title="Fetch fresh articles, statuses, and dates directly from WordPress"
                            >
                                <RefreshCw className={`h-3.5 w-3.5 ${syncing ? 'animate-spin' : ''}`} />
                                <span>{syncing ? 'Syncing...' : 'Sync with WordPress'}</span>
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
                <div className="flex-1 overflow-y-auto p-4 space-y-2.5 min-h-[340px]">
                    {loading && posts.length === 0 ? (
                        <div className="flex flex-col items-center justify-center py-20 text-center text-muted-foreground gap-3">
                            <Loader2 className="h-8 w-8 animate-spin text-indigo-500" />
                            <p className="text-sm font-medium">Connecting to WordPress and loading articles...</p>
                        </div>
                    ) : sortedPosts.length === 0 ? (
                        <div className="flex flex-col items-center justify-center py-20 text-center text-muted-foreground gap-3">
                            <Globe className="h-10 w-10 text-muted-foreground/40" />
                            <div>
                                <h3 className="text-sm font-semibold text-foreground">No articles found</h3>
                                <p className="text-xs text-muted-foreground mt-1 max-w-md">
                                    {posts.length === 0
                                        ? "No articles found from your WordPress sites. Click 'Sync with WordPress' to load all available articles, or verify your application password credentials in Settings."
                                        : "No articles match your current status filter or search query."}
                                </p>
                            </div>
                            {posts.length === 0 && (
                                <button
                                    onClick={() => handleSync(false)}
                                    disabled={syncing}
                                    className="mt-2 inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-xs font-medium text-white shadow-sm hover:bg-indigo-700 disabled:opacity-50"
                                >
                                    <RefreshCw className={`h-3.5 w-3.5 ${syncing ? 'animate-spin' : ''}`} />
                                    <span>Sync All Articles from WordPress</span>
                                </button>
                            )}
                        </div>
                    ) : (
                        sortedPosts.map((post) => {
                            const isImported = Boolean(post.titles_record_id);
                            const isSelected = selectedIds.has(post.id);
                            const isImporting = importingId === String(post.id);
                            const focusKw = post.focus_keyword || post.primary_keyword;
                            const postStatus = String(post.status || post.raw_post_json?.status || 'publish').toLowerCase();
                            const statusBadge = getWpStatusBadge(postStatus);
                            const pubDate = formatDateDisplay(post.published_at);
                            const modDate = formatDateDisplay(post.modified_at);

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
                                                className="mt-1 h-4 w-4 rounded border-border text-indigo-600 focus:ring-indigo-500 cursor-pointer"
                                            />
                                        )}
                                        {isImported && (
                                            <CheckCircle2 className="mt-1 h-4 w-4 shrink-0 text-emerald-500" />
                                        )}

                                        <div className="flex-1 min-w-0 space-y-1.5">
                                            {/* Status Badge, Date Badge, Site Badge & External Link */}
                                            <div className="flex items-center gap-2 flex-wrap">
                                                <span className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-[11px] font-semibold ${statusBadge.className}`}>
                                                    {statusBadge.label}
                                                </span>

                                                {/* Prominent Date Tag */}
                                                {postStatus === 'future' ? (
                                                    <span className="inline-flex items-center gap-1 text-[11px] font-medium text-blue-600 dark:text-blue-400 bg-blue-500/10 border border-blue-500/20 px-2 py-0.5 rounded-md" title={`Scheduled date: ${post.published_at}`}>
                                                        <Calendar className="h-3 w-3" />
                                                        Scheduled: {pubDate || 'Upcoming'}
                                                    </span>
                                                ) : postStatus === 'draft' ? (
                                                    <span className="inline-flex items-center gap-1 text-[11px] font-medium text-amber-600 dark:text-amber-400 bg-amber-500/10 border border-amber-500/20 px-2 py-0.5 rounded-md" title={`Last updated: ${post.modified_at || post.published_at}`}>
                                                        <Clock className="h-3 w-3" />
                                                        Saved: {modDate || pubDate || 'Draft'}
                                                    </span>
                                                ) : (
                                                    <span className="inline-flex items-center gap-1 text-[11px] text-muted-foreground bg-muted/50 border border-border/60 px-2 py-0.5 rounded-md" title={`Published: ${post.published_at}${post.modified_at ? ` · Modified: ${post.modified_at}` : ''}`}>
                                                        <Calendar className="h-3 w-3" />
                                                        {pubDate ? `Published: ${pubDate}` : 'Published'}
                                                        {modDate && pubDate && modDate !== pubDate && (
                                                            <span className="text-muted-foreground/70 ml-1">
                                                                (Updated {modDate})
                                                            </span>
                                                        )}
                                                    </span>
                                                )}

                                                <span className="text-[10px] font-bold uppercase px-2 py-0.5 bg-muted/50 rounded-md text-muted-foreground border border-border/50">
                                                    {post.source_site || 'WordPress'}
                                                </span>

                                                {isImported && (
                                                    <span className="inline-flex items-center gap-1 rounded-md bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 text-[11px] font-medium text-indigo-400">
                                                        <Check className="h-3 w-3" /> In Library
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

                                            {/* Slug / URL identifier */}
                                            {post.link && (
                                                <div className="text-[11px] font-mono text-muted-foreground/75 truncate" title={post.link}>
                                                    /{post.slug || post.link.replace(/^https?:\/\/[^/]+\/?/, '').replace(/\/$/, '')}
                                                </div>
                                            )}

                                            {/* SEO Description / Excerpt */}
                                            {(post.seo_description || post.excerpt) && (
                                                <p
                                                    className="text-xs text-muted-foreground line-clamp-2"
                                                    dangerouslySetInnerHTML={{ __html: post.seo_description || post.excerpt || '' }}
                                                />
                                            )}

                                            {/* Metadata Badges: Focus KW, Categories, Tags */}
                                            <div className="flex flex-wrap items-center gap-1.5 pt-0.5">
                                                {focusKw && (
                                                    <span className="inline-flex items-center gap-1 rounded-md bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 text-[11px] font-medium text-emerald-500 dark:text-emerald-400">
                                                        <Sparkles className="h-2.5 w-2.5" />
                                                        KW: {focusKw}
                                                    </span>
                                                )}

                                                {post.category_names && post.category_names.length > 0 && (
                                                    <span className="inline-flex items-center gap-1 rounded-md bg-blue-500/10 border border-blue-500/20 px-2 py-0.5 text-[11px] text-blue-500 dark:text-blue-400">
                                                        <Layers className="h-2.5 w-2.5" />
                                                        {post.category_names.slice(0, 2).join(', ')}
                                                    </span>
                                                )}

                                                {post.tag_names && post.tag_names.length > 0 && (
                                                    <span className="inline-flex items-center gap-1 rounded-md bg-purple-500/10 border border-purple-500/20 px-2 py-0.5 text-[11px] text-purple-500 dark:text-purple-400">
                                                        <Tag className="h-2.5 w-2.5" />
                                                        {post.tag_names.slice(0, 2).join(', ')}
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
                                                    title="Import from WordPress and open in Article Editor"
                                                >
                                                    {isImporting ? (
                                                        <>
                                                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                                            <span>Loading…</span>
                                                        </>
                                                    ) : (
                                                        <>
                                                            <FileText className="h-3.5 w-3.5" />
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
                            onClick={() => toggleSelectAll(sortedPosts)}
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
