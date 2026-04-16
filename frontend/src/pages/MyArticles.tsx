import React, { useEffect, useState, useMemo } from 'react'
import { supabase } from '../lib/supabase'
import { useAuth } from '../context/auth-context'
import type { Article } from '../types'
import { Plus, Search, Trash2, Sparkles, Edit, X } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'

function getStatusStyle(status: string, published: boolean) {
    if (published || status === 'Published') return { label: 'Published', color: 'text-emerald-400' }
    if (status === 'Scheduled') return { label: 'Scheduled', color: 'text-purple-400' }
    if (status === 'Generated') return { label: 'Generated', color: 'text-blue-400' }
    if (status === 'Draft' || status === 'New') return { label: status || 'Draft', color: 'text-slate-500' }
    if (status === 'Error' || status === 'Failed') return { label: status, color: 'text-red-400' }
    if (status === 'Review' || status === 'Editing') return { label: status, color: 'text-amber-400' }
    return { label: status || 'Draft', color: 'text-slate-500' }
}

function getScoreColor(score?: number) {
    if (score == null) return 'text-slate-600'
    if (score >= 70) return 'text-emerald-400'
    if (score >= 40) return 'text-amber-400'
    return 'text-red-400'
}

export const MyArticles: React.FC = () => {
    const { user } = useAuth()
    const navigate = useNavigate()
    const [articles, setArticles] = useState<Article[]>([])
    const [loading, setLoading] = useState(true)
    const [sortKey, setSortKey] = useState<keyof Article>('dateCreatedOn')
    const [sortAsc, setSortAsc] = useState(false)
    const [search, setSearch] = useState('')
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())

    const fetchArticles = async () => {
        if (!user) return
        try {
            const { data, error } = await supabase
                .from('Titles')
                .select('*')
                .eq('user_id', user.id)
                .order('dateCreatedOn', { ascending: false })

            if (error) throw error
            setArticles(data || [])
        } catch (error) {
            console.error('Error fetching articles:', error)
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        fetchArticles()
    }, [user])

    const sortedArticles = useMemo(() => {
        const copy = [...articles]
        copy.sort((a, b) => {
            const aVal = a[sortKey]
            const bVal = b[sortKey]
            if (aVal == null) return 1
            if (bVal == null) return -1
            if (typeof aVal === 'number' && typeof bVal === 'number') {
                return sortAsc ? aVal - bVal : bVal - aVal
            }
            return sortAsc
                ? String(aVal).localeCompare(String(bVal))
                : String(bVal).localeCompare(String(aVal))
        })
        return copy
    }, [articles, sortKey, sortAsc])

    const filteredArticles = useMemo(
        () => sortedArticles.filter(article => article.Title?.toLowerCase().includes(search.toLowerCase())),
        [sortedArticles, search],
    )

    const handleCreateNew = async () => {
        if (!user) return
        try {
            const { data, error } = await supabase
                .from('Titles')
                .insert([{
                    user_id: user.id,
                    dateCreatedOn: new Date().toISOString(),
                    status: 'New',
                    Title: 'Untitled Article',
                }])
                .select()
                .single()

            if (error) throw error
            if (data) {
                navigate(`/content-studio?id=${data.id}`)
            }
        } catch (error) {
            console.error('Error creating article:', error)
        }
    }

    const handleDelete = async (id: string) => {
        if (!confirm('Are you sure you want to delete this article?')) return
        try {
            const { error } = await supabase
                .from('Titles')
                .delete()
                .eq('id', id)

            if (error) throw error
            setArticles(articles.filter(a => a.id !== id))
            setSelectedIds(prev => {
                const next = new Set(prev)
                next.delete(id)
                return next
            })
        } catch (error) {
            console.error('Error deleting article:', error)
        }
    }

    const handleToggleSelect = (id: string) => {
        setSelectedIds(prev => {
            const next = new Set(prev)
            if (next.has(id)) next.delete(id)
            else next.add(id)
            return next
        })
    }

    const handleDeleteSelected = async () => {
        if (selectedIds.size === 0) return
        if (!confirm(`Are you sure you want to delete ${selectedIds.size} articles? This cannot be undone.`)) return

        try {
            const idsToDelete = Array.from(selectedIds)
            const { error } = await supabase
                .from('Titles')
                .delete()
                .in('id', idsToDelete)

            if (error) throw error
            setArticles(articles.filter(a => !selectedIds.has(a.id)))
            setSelectedIds(new Set())
        } catch (error) {
            console.error('Error deleting selected articles:', error)
        }
    }

    const handleSort = (field: keyof Article) => {
        if (sortKey === field) setSortAsc(prev => !prev)
        else { setSortKey(field); setSortAsc(true) }
    }

    const totalWords = useMemo(() => {
        const count = articles.reduce((acc, article: any) => acc + (parseInt(article.articleLength) || 0), 0)
        return count >= 1000 ? (count / 1000).toFixed(1) + 'K' : count.toString()
    }, [articles])

    const publishedCount = articles.filter(a => a.published).length
    const allSelected = articles.length > 0 && selectedIds.size === filteredArticles.length

    return (
        <div className="min-h-screen bg-[#08101d]">
            <div className="mx-auto max-w-5xl px-8 py-10 lg:py-14">

                {/* Page header */}
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.25 }}
                >
                    <h1 className="text-2xl font-semibold tracking-tight text-white">
                        Content Library
                    </h1>
                    <p className="mt-1 text-sm text-slate-500">
                        {articles.length} articles
                        {publishedCount > 0 && ` · ${publishedCount} published`}
                        {totalWords !== '0' && ` · ${totalWords} words`}
                    </p>
                </motion.div>

                {/* Toolbar */}
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.25, delay: 0.04 }}
                    className="mt-8 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between"
                >
                    <div className="relative flex-1 sm:max-w-xs">
                        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                        <input
                            type="text"
                            placeholder="Search articles..."
                            value={search}
                            onChange={(e) => setSearch(e.target.value)}
                            className="h-10 w-full rounded-lg border border-white/10 bg-white/[0.04] pl-10 pr-4 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-blue-400/30 hover:border-white/15"
                        />
                        {search && (
                            <button
                                type="button"
                                onClick={() => setSearch('')}
                                className="absolute right-2 top-1/2 -translate-y-1/2 rounded-md p-1 text-slate-500 hover:text-slate-300"
                            >
                                <X className="h-3 w-3" />
                            </button>
                        )}
                    </div>

                    <div className="flex items-center gap-2">
                        {selectedIds.size > 0 && (
                            <button
                                onClick={handleDeleteSelected}
                                className="inline-flex h-10 items-center gap-1.5 rounded-lg border border-red-400/20 bg-red-500/10 px-3.5 text-sm text-red-300 transition hover:bg-red-500/15 hover:text-red-200"
                            >
                                <Trash2 className="h-3.5 w-3.5" />
                                <span>Delete ({selectedIds.size})</span>
                            </button>
                        )}
                        <button
                            onClick={handleCreateNew}
                            className="inline-flex h-10 items-center gap-1.5 rounded-lg border border-white/10 bg-white/[0.04] px-3.5 text-sm text-slate-300 transition hover:border-white/15 hover:bg-white/[0.07] hover:text-white"
                        >
                            <Plus className="h-3.5 w-3.5" />
                            <span>New Article</span>
                        </button>
                    </div>
                </motion.div>

                {/* Divider */}
                <div className="mt-6 border-t border-white/8" />

                {/* Article list */}
                <motion.div
                    initial={{ opacity: 0, y: 14 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: 0.08 }}
                    className="mt-4"
                >
                    {loading ? (
                        <div className="space-y-2">
                            {[1, 2, 3, 4, 5].map(i => (
                                <div key={i} className="h-14 animate-pulse rounded-lg bg-white/[0.03]" />
                            ))}
                        </div>
                    ) : filteredArticles.length === 0 ? (
                        <div className="py-20 text-center">
                            <p className="text-sm text-slate-500">
                                {search ? 'No articles match your search.' : 'No articles yet. Create one to get started.'}
                            </p>
                        </div>
                    ) : (
                        <>
                            {/* Table header */}
                            <div className="grid grid-cols-[2.5rem_1fr_5.5rem_4rem_6rem_auto] items-center gap-2 border-b border-white/8 px-1 pb-3 text-[11px] uppercase tracking-wider text-slate-600">
                                <span className="flex justify-center">
                                    <input
                                        type="checkbox"
                                        className="h-4 w-4 rounded border-white/15 bg-transparent text-blue-400 focus:ring-blue-400 focus:ring-offset-0"
                                        checked={allSelected}
                                        onChange={(e) => {
                                            setSelectedIds(e.target.checked
                                                ? new Set(filteredArticles.map(a => a.id))
                                                : new Set())
                                        }}
                                    />
                                </span>
                                <button
                                    type="button"
                                    onClick={() => handleSort('Title')}
                                    className="text-left hover:text-slate-400 transition"
                                >
                                    Title{sortKey === 'Title' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                                </button>
                                <button
                                    type="button"
                                    onClick={() => handleSort('status')}
                                    className="text-left hover:text-slate-400 transition"
                                >
                                    Status{sortKey === 'status' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                                </button>
                                <button
                                    type="button"
                                    onClick={() => handleSort('seo_optimization_score')}
                                    className="text-left hover:text-slate-400 transition"
                                >
                                    SEO{sortKey === 'seo_optimization_score' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                                </button>
                                <button
                                    type="button"
                                    onClick={() => handleSort('dateCreatedOn')}
                                    className="text-left hover:text-slate-400 transition"
                                >
                                    Date{sortKey === 'dateCreatedOn' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                                </button>
                                <span className="text-right">Actions</span>
                            </div>

                            {/* Table rows */}
                            <div className="divide-y divide-white/5">
                                {filteredArticles.map(article => {
                                    const selected = selectedIds.has(article.id)
                                    const status = getStatusStyle(article.status, article.published)
                                    return (
                                        <div
                                            key={article.id}
                                            className={`grid grid-cols-[2.5rem_1fr_5.5rem_4rem_6rem_auto] items-center gap-2 px-1 py-3 transition ${
                                                selected ? 'bg-blue-500/[0.05]' : ''
                                            }`}
                                        >
                                            <span className="flex justify-center">
                                                <input
                                                    type="checkbox"
                                                    className="h-4 w-4 rounded border-white/15 bg-transparent text-blue-400 focus:ring-blue-400 focus:ring-offset-0"
                                                    checked={selected}
                                                    onChange={() => handleToggleSelect(article.id)}
                                                />
                                            </span>

                                            <div className="min-w-0">
                                                <p className="truncate text-sm font-medium text-white">
                                                    {article.Title || 'Untitled'}
                                                </p>
                                                {article.userDescription && (
                                                    <p className="mt-0.5 truncate text-xs text-slate-500">
                                                        {article.userDescription}
                                                    </p>
                                                )}
                                            </div>

                                            <span className={`text-xs font-medium ${status.color}`}>
                                                {status.label}
                                            </span>

                                            <span className={`text-xs font-medium ${getScoreColor(article.seo_optimization_score)}`}>
                                                {article.seo_optimization_score != null ? article.seo_optimization_score : '—'}
                                            </span>

                                            <span className="text-xs text-slate-500">
                                                {new Date(article.dateCreatedOn).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                                            </span>

                                            <div className="flex items-center justify-end gap-1">
                                                <button
                                                    onClick={() => navigate(`/content-studio?id=${article.id}`)}
                                                    className="rounded-md p-1.5 text-slate-600 transition hover:bg-white/[0.06] hover:text-blue-400"
                                                    title="Generate"
                                                >
                                                    <Sparkles className="h-3.5 w-3.5" />
                                                </button>
                                                <button
                                                    onClick={() => navigate(`/article-editor/${article.id}`)}
                                                    className="rounded-md p-1.5 text-slate-600 transition hover:bg-white/[0.06] hover:text-white"
                                                    title="Edit"
                                                >
                                                    <Edit className="h-3.5 w-3.5" />
                                                </button>
                                                <button
                                                    onClick={() => handleDelete(article.id)}
                                                    className="rounded-md p-1.5 text-slate-600 transition hover:bg-white/[0.06] hover:text-red-400"
                                                    title="Delete"
                                                >
                                                    <Trash2 className="h-3.5 w-3.5" />
                                                </button>
                                            </div>
                                        </div>
                                    )
                                })}
                            </div>
                        </>
                    )}

                    {/* Footer */}
                    {!loading && filteredArticles.length > 0 && (
                        <div className="mt-4 border-t border-white/8 pt-4">
                            <p className="text-xs text-slate-600">
                                Showing {filteredArticles.length} of {articles.length} articles
                            </p>
                        </div>
                    )}
                </motion.div>
            </div>
        </div>
    )
}