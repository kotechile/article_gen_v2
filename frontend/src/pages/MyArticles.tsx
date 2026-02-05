
import React, { useEffect, useState, useMemo } from 'react';
import { supabase } from '../lib/supabase';
import { useAuth } from '../context/auth-context';
import type { Article } from '../types';
import { Plus, Search, Trash2, Sparkles, Edit } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import clsx from 'clsx';

export const MyArticles: React.FC = () => {
    const { user } = useAuth();
    const navigate = useNavigate();
    const [articles, setArticles] = useState<Article[]>([]);
    const [loading, setLoading] = useState(true);
    const [sortKey, setSortKey] = useState<keyof Article>('dateCreatedOn');
    const [sortAsc, setSortAsc] = useState(false);
    const [search, setSearch] = useState('');
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

    // Fetch Articles
    const fetchArticles = async () => {
        if (!user) return;
        try {
            const { data, error } = await supabase
                .from('Titles')
                .select('*')
                .eq('user_id', user.id)
                .order('dateCreatedOn', { ascending: false });

            if (error) throw error;
            setArticles(data || []);
        } catch (error) {
            console.error('Error fetching articles:', error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchArticles();
    }, [user]);

    // Compute sorted articles based on sortKey and sortAsc
    const sortedArticles = useMemo(() => {
        const copy = [...articles];
        copy.sort((a, b) => {
            const aVal = a[sortKey];
            const bVal = b[sortKey];
            if (aVal == null) return 1;
            if (bVal == null) return -1;
            if (typeof aVal === 'number' && typeof bVal === 'number') {
                return sortAsc ? aVal - bVal : bVal - aVal;
            }
            return sortAsc
                ? String(aVal).localeCompare(String(bVal))
                : String(bVal).localeCompare(String(aVal));
        });
        return copy;
    }, [articles, sortKey, sortAsc]);

    // Create New Article
    const handleCreateNew = async () => {
        if (!user) return;
        try {
            // Insert a new blank record to get an ID
            // Note: Check if table allows nulls for required fields or if we need defaults
            const { data, error } = await supabase
                .from('Titles')
                .insert([{
                    user_id: user.id,
                    dateCreatedOn: new Date().toISOString(),
                    status: 'New',
                    Title: 'Untitled Article'
                }])
                .select()
                .single();

            if (error) throw error;
            if (data) {
                navigate(`/content-studio?id=${data.id}`);
            }
        } catch (error) {
            console.error('Error creating article:', error);
            alert('Failed to create new article');
        }
    };

    const handleDelete = async (id: string) => {
        if (!confirm('Are you sure you want to delete this article?')) return;
        try {
            const { error } = await supabase
                .from('Titles')
                .delete()
                .eq('id', id);

            if (error) throw error;
            // Refresh list
            setArticles(articles.filter(a => a.id !== id));
        } catch (error) {
            console.error('Error deleting article:', error);
        }
    };

    // Mass Action Handlers
    const handleSelectAllNew = () => {
        const newArticles = articles.filter(a => {
            const s = a.status?.toLowerCase() || '';
            return s === 'new' || s === 'draft';
        });

        // Toggle behavior: if all new are selected, deselect them. Otherwise, select them.
        const allNewSelected = newArticles.length > 0 && newArticles.every(a => selectedIds.has(a.id));

        if (allNewSelected) {
            const next = new Set(selectedIds);
            newArticles.forEach(a => next.delete(a.id));
            setSelectedIds(next);
        } else {
            const next = new Set(selectedIds);
            newArticles.forEach(a => next.add(a.id));
            setSelectedIds(next);
        }
    };

    const handleToggleSelect = (id: string) => {
        const next = new Set(selectedIds);
        if (next.has(id)) {
            next.delete(id);
        } else {
            next.add(id);
        }
        setSelectedIds(next);
    };

    const handleDeleteSelected = async () => {
        if (selectedIds.size === 0) return;
        if (!confirm(`Are you sure you want to delete ${selectedIds.size} articles? This cannot be undone.`)) return;

        try {
            const idsToDelete = Array.from(selectedIds);
            const { error } = await supabase
                .from('Titles')
                .delete()
                .in('id', idsToDelete);

            if (error) throw error;

            // Refresh list and clear selection
            setArticles(articles.filter(a => !selectedIds.has(a.id)));
            setSelectedIds(new Set());
        } catch (error) {
            console.error('Error deleting selected articles:', error);
            alert('Failed to delete selected articles');
        }
    };



    const getStatusColor = (status: string, published: boolean) => {
        if (published || status === 'Published') return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400';
        if (status === 'Scheduled') return 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400';
        if (status === 'Generated') return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400';
        if (status === 'Draft' || status === 'New') return 'bg-gray-100 text-gray-800 dark:bg-gray-700/50 dark:text-gray-300';
        if (status === 'Error' || status === 'Failed') return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400';
        if (status === 'Review' || status === 'Editing') return 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400';
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400';
    };

    // Helper to color‑code numeric scores (e.g., SEO)
    const getScoreColor = (score?: number) => {
        if (score == null) return '';
        if (score >= 70) return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400';
        if (score >= 40) return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400';
        return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400';
    };

    const getDifficultyColor = (score?: number) => {
        if (score == null) return '';
        if (score < 30) return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400';
        if (score < 70) return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400';
        return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400';
    };

    const getVolumeColor = (volume?: number) => {
        if (volume == null) return '';
        if (volume >= 1000) return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400';
        if (volume >= 100) return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400';
        return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400';
    };

    // Sortable column header component
    type SortableHeaderProps = {
        label: string;
        field: keyof Article;
    };

    const SortableHeader: React.FC<SortableHeaderProps> = ({ label, field }) => {
        const handleClick = () => {
            setSortKey(field);
            setSortAsc(prev => (sortKey === field ? !prev : true));
        };
        const isActive = sortKey === field;
        return (
            <th
                className="px-6 py-4 font-medium text-gray-500 dark:text-gray-400 cursor-pointer select-none"
                onClick={handleClick}
            >
                {label}
                {isActive && (sortAsc ? ' ▲' : ' ▼')}
            </th>
        );
    };

    // Filter articles


    // Calculate stats
    const totalWords = useMemo(() => {
        const count = articles.reduce((acc, article: any) => {
            return acc + (parseInt(article.articleLength) || 0);
        }, 0);

        if (count >= 1000) {
            return (count / 1000).toFixed(1) + 'K';
        }
        return count.toString();
    }, [articles]);

    return (
        <div className="space-y-8">
            {/* Header Stats */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {[
                    { label: 'Total Articles', value: articles.length, change: '', color: 'from-blue-500 to-indigo-500' },
                    { label: 'Published', value: articles.filter(a => a.published).length, change: '', color: 'from-green-500 to-emerald-500' },
                    { label: 'In Progress', value: articles.filter(a => !a.published && a.status !== 'Generated').length, change: '', color: 'from-orange-500 to-amber-500' },
                    { label: 'Total Words', value: totalWords, change: '', color: 'from-purple-500 to-pink-500' }
                ].map((stat, i) => (
                    <div key={i} className="bg-white dark:bg-gray-800 p-6 rounded-2xl border border-gray-100 dark:border-gray-700 shadow-sm relative overflow-hidden group hover:shadow-md transition-all">
                        {/* Background decoration */}
                        <div className={`absolute top-0 right-0 w-24 h-24 bg-gradient-to-br ${stat.color} opacity-5 rounded-bl-full group-hover:scale-110 transition-transform`} />

                        <h3 className="text-gray-500 dark:text-gray-400 text-sm font-medium">{stat.label}</h3>
                        <div className="mt-2 flex items-baseline gap-2">
                            <span className="text-3xl font-bold text-gray-900 dark:text-white">{stat.value}</span>
                            {stat.change && (
                                <span className="text-xs font-medium text-green-600 bg-green-50 dark:bg-green-900/20 px-1.5 py-0.5 rounded-full">{stat.change}</span>
                            )}
                        </div>
                    </div>
                ))}
            </div>

            {/* Main Content */}
            <div className="bg-white dark:bg-gray-800 rounded-2xl border border-gray-100 dark:border-gray-700 shadow-sm overflow-hidden">
                {/* Toolbar */}
                <div className="p-6 border-b border-gray-100 dark:border-gray-700 flex flex-col sm:flex-row items-center justify-between gap-4">
                    <h2 className="text-xl font-bold text-gray-900 dark:text-white">My Articles</h2>
                    <div className="flex w-full sm:w-auto items-center gap-3">
                        {/* Mass Actions */}
                        {selectedIds.size > 0 && (
                            <button
                                onClick={handleDeleteSelected}
                                className="flex items-center gap-2 bg-red-100 hover:bg-red-200 text-red-700 px-4 py-2 rounded-xl font-medium transition-colors text-sm"
                            >
                                <Trash2 className="w-4 h-4" />
                                <span className="hidden sm:inline">Delete Selected ({selectedIds.size})</span>
                            </button>
                        )}
                        <button
                            onClick={handleSelectAllNew}
                            className="flex items-center gap-2 bg-gray-100 hover:bg-gray-200 text-gray-700 dark:bg-gray-700 dark:hover:bg-gray-600 dark:text-gray-200 px-4 py-2 rounded-xl font-medium transition-colors text-sm"
                        >
                            <span className="hidden sm:inline">Select All New</span>
                            <span className="sm:hidden">All New</span>
                        </button>

                        <div className="relative flex-1 sm:flex-initial">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                            <input
                                type="text"
                                placeholder="Search articles..."
                                value={search}
                                onChange={(e) => setSearch(e.target.value)}
                                className="w-full sm:w-64 pl-9 pr-4 py-2 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm"
                            />
                        </div>
                        <button
                            onClick={handleCreateNew}
                            className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-xl font-medium transition-colors text-sm shadow-lg shadow-indigo-500/20"
                        >
                            <Plus className="w-4 h-4" />
                            <span className="hidden sm:inline">Add New Article</span>
                            <span className="sm:hidden">New</span>
                        </button>
                    </div>
                </div>

                {/* Table */}
                <div className="overflow-auto max-h-[75vh] scrollbar-thin">
                    <table className="w-full text-left text-sm">
                        <thead>
                            <tr className="sticky top-0 z-20 bg-gray-50 dark:bg-gray-900 border-b border-gray-100 dark:border-gray-700 shadow-sm">
                                <th className="px-6 py-4 w-10 text-center">
                                    <input
                                        type="checkbox"
                                        className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                                        checked={articles.length > 0 && selectedIds.size === articles.length}
                                        onChange={(e) => {
                                            if (e.target.checked) {
                                                setSelectedIds(new Set(articles.map(a => a.id)));
                                            } else {
                                                setSelectedIds(new Set());
                                            }
                                        }}
                                    />
                                </th>
                                <SortableHeader label="Title" field="Title" />
                                <SortableHeader label="Status" field="status" />
                                <SortableHeader label="SEO" field="seo_optimization_score" />

                                <SortableHeader label="KD" field="avg_keyword_difficulty" />
                                <SortableHeader label="Search Vol" field="total_search_volume" />
                                <SortableHeader label="Audience" field="target_audience" />
                                <SortableHeader label="Quality" field="overall_quality_score" />
                                <SortableHeader label="Traffic" field="traffic_potential_score" />
                                <SortableHeader label="Category" field="Keywords" />
                                <SortableHeader label="Date" field="dateCreatedOn" />
                                <th className="px-6 py-4 font-medium text-gray-500 dark:text-gray-400 sticky right-0 bg-gray-50 dark:bg-gray-900 shadow-xl z-20 text-right">Actions</th>
                            </tr>

                        </thead>
                        <tbody className="divide-y divide-gray-100 dark:divide-gray-700">
                            {loading ? (
                                <tr><td colSpan={5} className="px-6 py-8 text-center text-gray-500">Loading articles...</td></tr>
                            ) : sortedArticles.length === 0 ? (
                                <tr><td colSpan={5} className="px-6 py-8 text-center text-gray-500">No articles found. Create one to get started!</td></tr>
                            ) : (
                                sortedArticles.filter(article => article.Title?.toLowerCase().includes(search.toLowerCase())).map((article) => (
                                    <tr key={article.id} className={clsx(
                                        "group hover:shadow-[inset_0_-2px_0_0_#6366f1] transition-all duration-200",
                                        selectedIds.has(article.id) && "bg-indigo-50 dark:bg-indigo-900/20"
                                    )}>
                                        <td className="px-6 py-4 text-center">
                                            <input
                                                type="checkbox"
                                                className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                                                checked={selectedIds.has(article.id)}
                                                onChange={() => handleToggleSelect(article.id)}
                                            />
                                        </td>
                                        <td className="px-6 py-4">
                                            <div className="font-medium text-gray-900 dark:text-white max-w-sm sm:max-w-md truncate" title={article.Title}>
                                                {article.Title || 'Untitled'}
                                            </div>
                                            <div className="text-xs text-gray-500 mt-1 max-w-xs truncate">
                                                {article.userDescription}
                                            </div>
                                        </td>
                                        <td className="px-6 py-4">
                                            <span className={clsx(
                                                "px-2.5 py-1 rounded-full text-xs font-medium border border-transparent",
                                                getStatusColor(article.status, article.published)
                                            )}>
                                                {article.published ? 'Published' : article.status || 'Draft'}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 text-gray-500 dark:text-gray-400">
                                            <span className={getScoreColor(article.seo_optimization_score)}>{article.seo_optimization_score != null ? article.seo_optimization_score : '-'}</span>
                                        </td>

                                        <td className="px-6 py-4 text-gray-500 dark:text-gray-400">
                                            <span className={getDifficultyColor(article.avg_keyword_difficulty)}>{article.avg_keyword_difficulty != null ? article.avg_keyword_difficulty : '-'}</span>
                                        </td>
                                        <td className="px-6 py-4 text-gray-500 dark:text-gray-400">
                                            <span className={getVolumeColor(article.total_search_volume)}>{article.total_search_volume != null ? article.total_search_volume : '-'}</span>
                                        </td>
                                        <td className="px-6 py-4 text-gray-500 dark:text-gray-400">
                                            {article.target_audience || '-'}
                                        </td>
                                        <td className="px-6 py-4 text-gray-500 dark:text-gray-400">
                                            <span className={getScoreColor(article.overall_quality_score)}>{article.overall_quality_score != null ? article.overall_quality_score : '-'}</span>
                                        </td>
                                        <td className="px-6 py-4 text-gray-500 dark:text-gray-400">
                                            <span className={getScoreColor(article.traffic_potential_score)}>{article.traffic_potential_score != null ? article.traffic_potential_score : '-'}</span>
                                        </td>
                                        <td className="px-6 py-4 text-gray-500 dark:text-gray-400">
                                            {article.Keywords ? article.Keywords.split(',')[0] : 'General'}
                                        </td>
                                        <td className="px-6 py-4 text-gray-500 dark:text-gray-400">
                                            {new Date(article.dateCreatedOn).toLocaleDateString()}
                                        </td>


                                        <td className="px-6 py-4 text-right sticky right-0 bg-white dark:bg-gray-800 shadow-xl z-10">
                                            <div className="flex items-center justify-end gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                                <button
                                                    onClick={() => navigate(`/content-studio?id=${article.id}`)}
                                                    className="p-2 text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400 hover:bg-indigo-50 dark:hover:bg-indigo-900/30 rounded-lg transition-colors"
                                                    title="Generate"
                                                >
                                                    <Sparkles className="w-4 h-4" />
                                                </button>
                                                <button
                                                    onClick={() => navigate(`/article-editor/${article.id}`)}
                                                    className="p-2 text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/30 rounded-lg transition-colors"
                                                    title="Edit Content"
                                                >
                                                    <Edit className="w-4 h-4" />
                                                </button>
                                                <button
                                                    onClick={() => handleDelete(article.id)}
                                                    className="p-2 text-gray-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/30 rounded-lg transition-colors"
                                                    title="Delete"
                                                >
                                                    <Trash2 className="w-4 h-4" />
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>

                <div className="px-6 py-4 border-t border-gray-100 dark:border-gray-700 flex items-center justify-between text-sm text-gray-500">
                    <span>Showing {sortedArticles.filter(article => article.Title?.toLowerCase().includes(search.toLowerCase())).length} entries</span>
                    <div className="flex gap-2">
                        <button disabled className="px-3 py-1 rounded-lg border border-gray-200 dark:border-gray-700 disabled:opacity-50">Previous</button>
                        <button disabled className="px-3 py-1 rounded-lg border border-gray-200 dark:border-gray-700 disabled:opacity-50">Next</button>
                    </div>
                </div>
            </div>
        </div>
    );
};
