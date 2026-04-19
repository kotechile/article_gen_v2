import React, { useEffect, useMemo, useState } from 'react'
import { supabase } from '../lib/supabase'
import { useAuth } from '../context/auth-context'
import type { Article } from '../types'
import { Plus, Search, Trash2, Sparkles, Edit, X } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { apiClient } from '../api-client'

type LibraryArticle = Article & {
    _source: 'titles' | 'content_ideas'
    _keywordTelemetry?: {
        tier?: string
        calls?: number
        nonZero?: number
        bestVolume?: number
    }
}

type ContentIdeaRow = {
    id: string
    user_id: string
    title: string
    description: string | null
    content_type: string | null
    status: string | null
    published: boolean | null
    published_to_titles: boolean | null
    titles_record_id: string | null
    seo_optimization_score: number | null
    created_at: string
    idea_metadata?: any
}

function hasWrittenContent(article: any) {
    return Boolean(
        String(article?.htmlArticle || '').trim() ||
        String(article?.articleText || '').trim()
    )
}

function getStatusStyle(article: any, source: 'titles' | 'content_ideas') {
    const status = article?.status || ''
    const normalized = (status || '').trim().toLowerCase()
    const wpStatus = String(article?.last_wp_post_status || '').trim().toLowerCase()
    const written = hasWrittenContent(article)
    const isWpPublished =
        normalized === 'wp published' ||
        normalized === 'scheduled' ||
        wpStatus === 'publish' ||
        wpStatus === 'future'

    if (isWpPublished) {
        return { label: 'WP Published', color: 'text-emerald-500 dark:text-emerald-400' }
    }
    if (written) {
        return { label: 'Written', color: 'text-cyan-500 dark:text-cyan-400' }
    }
    if (source === 'content_ideas' && (status === 'Published' || status === 'published')) {
        return { label: 'New', color: 'text-muted-foreground' }
    }
    if (normalized === 'ready for content' || normalized === 'ready_for_content' || normalized === 'sent to content library' || normalized === 'in library') {
        return { label: 'New', color: 'text-muted-foreground' }
    }
    if (normalized === 'generated' || normalized === 'started') {
        return { label: 'Started', color: 'text-indigo-300' }
    }
    if (normalized === 'not generated' || normalized === 'not started' || normalized === 'not started.') {
        return { label: 'Not Started', color: 'text-muted-foreground' }
    }
    if (status === 'Draft' || status === 'New') return { label: 'New', color: 'text-muted-foreground' }
    if (status === 'Scheduled') return { label: 'Scheduled', color: 'text-purple-500 dark:text-purple-400' }
    if (status === 'Saved') return { label: 'Saved', color: 'text-cyan-500 dark:text-cyan-400' }
    if (status === 'Error' || status === 'Failed') return { label: status, color: 'text-red-500 dark:text-red-400' }
    if (status === 'Review' || status === 'Editing') return { label: status, color: 'text-amber-500 dark:text-amber-400' }
    return { label: 'New', color: 'text-muted-foreground' }
}

function getScoreColor(score?: number) {
    if (score == null) return 'text-muted-foreground'
    if (score >= 70) return 'text-emerald-500 dark:text-emerald-400'
    if (score >= 40) return 'text-amber-500 dark:text-amber-400'
    return 'text-red-500 dark:text-red-400'
}

function safeNumber(value: unknown): number | null {
    if (typeof value === 'number' && Number.isFinite(value)) return value
    if (typeof value === 'string') {
        const parsed = Number(value)
        if (Number.isFinite(parsed)) return parsed
    }
    return null
}

function getKeywordMetricSource(article: any): string {
    const metricSource = String(article?.selected_keyword_metrics_json?.primary?.metric_source || '').trim()
    if (metricSource) return metricSource
    const selectionSource = String(article?.keyword_selection_source || article?.keyword_research_source || '').trim()
    return selectionSource || 'unknown'
}

function isKeywordMetricEstimated(article: any): boolean {
    const explicit = article?.selected_keyword_metrics_json?.primary?.is_estimated
    if (typeof explicit === 'boolean') return explicit
    const source = getKeywordMetricSource(article).toLowerCase()
    return source.includes('aggregate') || source.includes('fallback') || source === 'unknown'
}

function getKeywordOpportunityScore(article: any): number | null {
    const volume = safeNumber(article?.selected_keyword_search_volume ?? article?.total_search_volume)
    const difficulty = safeNumber(article?.selected_keyword_difficulty ?? article?.avg_keyword_difficulty)
    if (volume == null || difficulty == null) return null
    if (volume <= 0) return 0
    const volScore = Math.min(volume / 50, 100)
    const diffScore = Math.max(0, 100 - difficulty)
    const score = Math.round(volScore * 0.55 + diffScore * 0.45)
    return Math.max(0, Math.min(100, score))
}

function getKeywordTelemetryLabel(article: any): string | null {
    const telemetry = article?._keywordTelemetry
    if (!telemetry) return null
    const tier = String(telemetry.tier || '').trim()
    const calls = safeNumber(telemetry.calls)
    const nonZero = safeNumber(telemetry.nonZero)
    const bestVolume = safeNumber(telemetry.bestVolume)
    const parts: string[] = []
    if (tier) parts.push(`Tier ${tier}`)
    if (calls != null) parts.push(`DFS calls ${calls}`)
    if (nonZero != null) parts.push(`Non-zero ${nonZero}`)
    if (bestVolume != null) parts.push(`Best Vol ${bestVolume}`)
    return parts.length ? parts.join(' · ') : null
}

function getKeywordRows(article: any): Array<{ keyword: string; volume: number; difficulty: number; cpc: number; isEstimated: boolean }> {
    const payload = article?.selected_keyword_metrics_json || {}
    const rows: Array<{ keyword: string; volume: number; difficulty: number; cpc: number; isEstimated: boolean }> = []
    const primary = payload?.primary
    if (primary?.keyword) {
        rows.push({
            keyword: String(primary.keyword),
            volume: safeNumber(primary.search_volume) ?? 0,
            difficulty: safeNumber(primary.keyword_difficulty) ?? 0,
            cpc: safeNumber(primary.cpc) ?? 0,
            isEstimated: Boolean(primary.is_estimated),
        })
    }
    const secondary = Array.isArray(payload?.secondary) ? payload.secondary : []
    for (const item of secondary) {
        if (!item?.keyword) continue
        rows.push({
            keyword: String(item.keyword),
            volume: safeNumber(item.search_volume) ?? 0,
            difficulty: safeNumber(item.keyword_difficulty) ?? 0,
            cpc: safeNumber(item.cpc) ?? 0,
            isEstimated: Boolean(item.is_estimated),
        })
    }
    if (rows.length > 0) return rows

    const fallbackPrimary = String(article?.primary_keyword || '').trim()
    const fallbackSecondaryRaw = article?.secondary_keywords_json
    const fallbackSecondary = Array.isArray(fallbackSecondaryRaw)
        ? fallbackSecondaryRaw
        : typeof fallbackSecondaryRaw === 'string'
            ? fallbackSecondaryRaw.split(',').map((s) => s.trim()).filter(Boolean)
            : []
    const fallback = [fallbackPrimary, ...fallbackSecondary].filter(Boolean)
    return fallback.slice(0, 3).map((keyword) => ({
        keyword,
        volume: safeNumber(article?.selected_keyword_search_volume) ?? 0,
        difficulty: safeNumber(article?.selected_keyword_difficulty) ?? 0,
        cpc: 0,
        isEstimated: true,
    }))
}

export const MyArticles: React.FC = () => {
    const { user } = useAuth()
    const navigate = useNavigate()
    const [articles, setArticles] = useState<LibraryArticle[]>([])
    const [loading, setLoading] = useState(true)
    const [sortKey, setSortKey] = useState<keyof Article>('dateCreatedOn')
    const [sortAsc, setSortAsc] = useState(false)
    const [search, setSearch] = useState('')
    const [exactMetricsOnly, setExactMetricsOnly] = useState(false)
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
    const [refreshingKeywords, setRefreshingKeywords] = useState(false)

    const fetchArticles = async () => {
        if (!user) return
        try {
            const [titlesResult, ideasResult] = await Promise.all([
                supabase
                    .from('Titles')
                    .select('*')
                    .eq('user_id', user.id)
                    .order('dateCreatedOn', { ascending: false }),
                supabase
                    .from('content_ideas')
                    .select('*')
                    .eq('user_id', user.id)
                    .neq('content_type', 'software')
                    .order('created_at', { ascending: false })
            ])

            if (titlesResult.error) throw titlesResult.error
            if (ideasResult.error) {
                console.warn('[ContentLibrary] content_ideas query failed; continuing with Titles only', ideasResult.error)
            }

            const titleRows = ((titlesResult.data || []) as Article[]).map((row) => ({
                ...row,
                _source: 'titles' as const,
            }))

            const titleIdSet = new Set(titleRows.map((row) => row.id))
            const titleBySourceIdeaId = new Set(
                titleRows.map((row) => row.source_idea_id).filter(Boolean) as string[]
            )

            const ideaRows = ((ideasResult.data || []) as any[]).map((row) => ({
                id: row.id,
                user_id: row.user_id,
                title: row.title,
                description: row.description ?? null,
                content_type: row.content_type ?? null,
                status: row.status ?? null,
                published: row.published ?? null,
                published_to_titles: row.published_to_titles ?? null,
                titles_record_id: row.titles_record_id ?? null,
                seo_optimization_score: row.seo_optimization_score ?? null,
                created_at: row.created_at,
                idea_metadata: row.idea_metadata ?? null,
            })) as ContentIdeaRow[]
            const telemetryByIdeaId = new Map<string, { tier?: string; calls?: number; nonZero?: number; bestVolume?: number }>()
            const telemetryByTitleId = new Map<string, { tier?: string; calls?: number; nonZero?: number; bestVolume?: number }>()
            for (const idea of ideaRows as any[]) {
                const seoEnrichment = idea?.idea_metadata?.seo_offer_enrichment || {}
                const ladder = Array.isArray(seoEnrichment?.keyword_budget_ladder_used)
                    ? seoEnrichment.keyword_budget_ladder_used
                    : []
                const lastTier = ladder.length ? ladder[ladder.length - 1] : null
                const quality = seoEnrichment?.keyword_quality_summary || {}
                const telemetry = {
                    tier: String(lastTier?.name || '').trim() || undefined,
                    calls: safeNumber(seoEnrichment?.dataforseo_call_count_estimate) ?? undefined,
                    nonZero: safeNumber(quality?.non_zero_count) ?? undefined,
                    bestVolume: safeNumber(quality?.best_volume) ?? undefined,
                }
                telemetryByIdeaId.set(idea.id, telemetry)
                if (idea.titles_record_id) telemetryByTitleId.set(String(idea.titles_record_id), telemetry)
            }

            const enrichedTitleRows = titleRows.map((row: any) => {
                const ideaTelemetry =
                    telemetryByIdeaId.get(String(row.source_idea_id || '')) ||
                    telemetryByTitleId.get(String(row.id || ''))
                return {
                    ...row,
                    _keywordTelemetry: ideaTelemetry,
                }
            })
            const publishedIdeas = ideaRows.filter((idea) => {
                const isPublished = Boolean(idea.published || idea.published_to_titles || idea.status?.toLowerCase() === 'published')
                const hasTitleMirror = Boolean(idea.titles_record_id && titleIdSet.has(idea.titles_record_id))
                const alreadyLinked = titleBySourceIdeaId.has(idea.id)
                return isPublished && !hasTitleMirror && !alreadyLinked
            })

            const mappedIdeas: LibraryArticle[] = publishedIdeas.map((idea) => ({
                id: idea.id,
                user_id: idea.user_id,
                Title: idea.title,
                userDescription: idea.description || '',
                Keywords: '',
                status: idea.status || 'ready_for_content',
                published: Boolean(idea.published || idea.published_to_titles || idea.status?.toLowerCase() === 'published'),
                dateCreatedOn: idea.created_at,
                articleLength: '0',
                LLM: '',
                tone: '',
                seo_optimization_score: idea.seo_optimization_score ?? undefined,
                content_type: idea.content_type || 'blog',
                _source: 'content_ideas',
            }))

            const combined = [...enrichedTitleRows, ...mappedIdeas].sort((a, b) =>
                new Date(b.dateCreatedOn).getTime() - new Date(a.dateCreatedOn).getTime()
            )

            console.info('[ContentLibrary] data loaded', {
                titles: enrichedTitleRows.length,
                publishedIdeasOnly: mappedIdeas.length,
                combined: combined.length,
            })
            setArticles(combined)
        } catch (error) {
            console.error('Error fetching library items:', error)
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
            const aVal = sortKey === ('selected_keyword_search_volume' as keyof Article)
                ? (safeNumber((a as any).selected_keyword_search_volume ?? (a as any).total_search_volume) ?? -1)
                : sortKey === ('selected_keyword_difficulty' as keyof Article)
                    ? (safeNumber((a as any).selected_keyword_difficulty ?? (a as any).avg_keyword_difficulty) ?? 9999)
                    : sortKey === ('traffic_potential_score' as keyof Article)
                        ? (getKeywordOpportunityScore(a) ?? -1)
                    : a[sortKey]
            const bVal = sortKey === ('selected_keyword_search_volume' as keyof Article)
                ? (safeNumber((b as any).selected_keyword_search_volume ?? (b as any).total_search_volume) ?? -1)
                : sortKey === ('selected_keyword_difficulty' as keyof Article)
                    ? (safeNumber((b as any).selected_keyword_difficulty ?? (b as any).avg_keyword_difficulty) ?? 9999)
                    : sortKey === ('traffic_potential_score' as keyof Article)
                        ? (getKeywordOpportunityScore(b) ?? -1)
                    : b[sortKey]
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
        () =>
            sortedArticles.filter(article => {
                const titleMatch = article.Title?.toLowerCase().includes(search.toLowerCase())
                if (!titleMatch) return false
                if (!exactMetricsOnly) return true
                return !isKeywordMetricEstimated(article)
            }),
        [sortedArticles, search, exactMetricsOnly],
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
        const target = articles.find(a => a.id === id)
        if (!target) return
        if (!confirm('Are you sure you want to delete this item?')) return

        try {
            if (target._source === 'content_ideas') {
                const { error } = await supabase
                    .from('content_ideas')
                    .delete()
                    .eq('id', id)
                if (error) throw error
            } else {
                const { error } = await supabase
                    .from('Titles')
                    .delete()
                    .eq('id', id)
                if (error) throw error
            }

            setArticles(articles.filter(a => a.id !== id))
            setSelectedIds(prev => {
                const next = new Set(prev)
                next.delete(id)
                return next
            })
        } catch (error) {
            console.error('Error deleting item:', error)
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
        if (!confirm(`Are you sure you want to delete ${selectedIds.size} items? This cannot be undone.`)) return

        try {
            const titleIds = articles
                .filter(a => selectedIds.has(a.id) && a._source === 'titles')
                .map(a => a.id)
            const ideaIds = articles
                .filter(a => selectedIds.has(a.id) && a._source === 'content_ideas')
                .map(a => a.id)

            if (titleIds.length > 0) {
                const { error } = await supabase.from('Titles').delete().in('id', titleIds)
                if (error) throw error
            }
            if (ideaIds.length > 0) {
                const { error } = await supabase.from('content_ideas').delete().in('id', ideaIds)
                if (error) throw error
            }

            setArticles(articles.filter(a => !selectedIds.has(a.id)))
            setSelectedIds(new Set())
        } catch (error) {
            console.error('Error deleting selected items:', error)
        }
    }

    const handleRefreshSelectedKeywords = async () => {
        if (!user || refreshingKeywords) return
        const selectedRows = articles.filter((a) => selectedIds.has(a.id))
        const titleIds = selectedRows
            .filter((a) => a._source === 'titles')
            .map((a) => a.id)
        const ideaIds = selectedRows
            .filter((a) => a._source === 'content_ideas')
            .map((a) => a.id)

        if (titleIds.length === 0 && ideaIds.length === 0) {
            return
        }

        setRefreshingKeywords(true)
        try {
            await apiClient.post('/content-ideas/refresh-keywords', {
                user_id: user.id,
                title_ids: titleIds,
                idea_ids: ideaIds,
            })
            await fetchArticles()
        } catch (error) {
            console.error('Error refreshing keyword metrics:', error)
        } finally {
            setRefreshingKeywords(false)
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

    const publishedCount = articles.filter((a: any) => {
        const status = String(a?.status || '').trim().toLowerCase()
        const wpStatus = String(a?.last_wp_post_status || '').trim().toLowerCase()
        return status === 'wp published' || status === 'scheduled' || wpStatus === 'publish' || wpStatus === 'future'
    }).length
    const allSelected = articles.length > 0 && selectedIds.size === filteredArticles.length
    const titlesOnly = useMemo(() => articles.filter(a => a._source === 'titles'), [articles])
    const qualityMetrics = useMemo(() => {
        const reports = titlesOnly
            .map((a) => a.quality_report || {})
            .filter((r) => Object.keys(r).length > 0)
        const avg = (key: 'humanization_score' | 'grounding_score' | 'geo_score') => {
            const values = reports
                .map((r: any) => safeNumber(r?.[key]))
                .filter((n): n is number => n != null)
            if (values.length === 0) return null
            return Math.round(values.reduce((acc, n) => acc + n, 0) / values.length)
        }
        const gatePassed = titlesOnly.filter((a: any) => {
            const decision = String(a?.quality_gate?.decision || '').toLowerCase()
            return decision === 'created' || decision === 'pass'
        }).length
        return {
            avgHumanization: avg('humanization_score'),
            avgGrounding: avg('grounding_score'),
            avgGeo: avg('geo_score'),
            gatePassRate: titlesOnly.length ? Math.round((gatePassed / titlesOnly.length) * 100) : null,
        }
    }, [titlesOnly])
    const exactKeywordCoverage = useMemo(() => {
        if (!titlesOnly.length) return null
        const exact = titlesOnly.filter((a: any) => !isKeywordMetricEstimated(a)).length
        return Math.round((exact / titlesOnly.length) * 100)
    }, [titlesOnly])
    const selectedRefreshableCount = useMemo(
        () => articles.filter((a) => selectedIds.has(a.id) && isKeywordMetricEstimated(a)).length,
        [articles, selectedIds],
    )

    return (
        <div className="min-h-screen bg-background">
            <div className="mx-auto max-w-5xl px-8 py-10 lg:py-14">
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.25 }}
                >
                    <h1 className="text-2xl font-semibold tracking-tight text-foreground">
                        Content Library
                    </h1>
                    <p className="mt-1 text-sm text-muted-foreground">
                        {articles.length} items
                        {publishedCount > 0 && ` · ${publishedCount} published`}
                        {totalWords !== '0' && ` · ${totalWords} words`}
                    </p>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.25, delay: 0.04 }}
                    className="mt-8 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between"
                >
                    <div className="relative flex-1 sm:max-w-xs">
                        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                        <input
                            type="text"
                            placeholder="Search articles..."
                            value={search}
                            onChange={(e) => setSearch(e.target.value)}
                            className="h-10 w-full rounded-lg border border-border bg-muted/50 pl-10 pr-4 text-sm text-foreground outline-none transition placeholder:text-muted-foreground focus:border-ring/50 hover:border-border"
                        />
                        {search && (
                            <button
                                type="button"
                                onClick={() => setSearch('')}
                                className="absolute right-2 top-1/2 -translate-y-1/2 rounded-md p-1 text-muted-foreground hover:text-foreground"
                            >
                                <X className="h-3 w-3" />
                            </button>
                        )}
                    </div>

                    <div className="flex items-center gap-2">
                        <button
                            type="button"
                            onClick={() => setExactMetricsOnly((prev) => !prev)}
                            className={`inline-flex h-10 items-center gap-1.5 rounded-lg border px-3.5 text-sm transition ${
                                exactMetricsOnly
                                    ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-400'
                                    : 'border-border bg-muted/50 text-muted-foreground hover:border-border hover:bg-muted hover:text-foreground'
                            }`}
                        >
                            <span>{exactMetricsOnly ? 'Exact Metrics Only' : 'Show All Metrics'}</span>
                        </button>
                        {selectedIds.size > 0 && (
                            <button
                                onClick={handleRefreshSelectedKeywords}
                                disabled={selectedRefreshableCount === 0 || refreshingKeywords}
                                className={`inline-flex h-10 items-center gap-1.5 rounded-lg border px-3.5 text-sm transition ${
                                    selectedRefreshableCount === 0 || refreshingKeywords
                                        ? 'cursor-not-allowed border-border bg-muted/40 text-muted-foreground/60'
                                        : 'border-emerald-500/20 bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/15'
                                }`}
                            >
                                <span>{refreshingKeywords ? 'Refreshing…' : `Refresh Keywords (${selectedRefreshableCount})`}</span>
                            </button>
                        )}
                        {selectedIds.size > 0 && (
                            <button
                                onClick={handleDeleteSelected}
                                className="inline-flex h-10 items-center gap-1.5 rounded-lg border border-destructive/20 bg-destructive/10 px-3.5 text-sm text-destructive transition hover:bg-destructive/15"
                            >
                                <Trash2 className="h-3.5 w-3.5" />
                                <span>Delete ({selectedIds.size})</span>
                            </button>
                        )}
                        <button
                            onClick={handleCreateNew}
                            className="inline-flex h-10 items-center gap-1.5 rounded-lg border border-border bg-muted/50 px-3.5 text-sm text-muted-foreground transition hover:border-border hover:bg-muted hover:text-foreground"
                        >
                            <Plus className="h-3.5 w-3.5" />
                            <span>New Article</span>
                        </button>
                    </div>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.25, delay: 0.06 }}
                    className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-5"
                >
                    <div className="rounded-lg border border-border bg-muted/20 p-3">
                        <p
                            className="text-[11px] uppercase tracking-wide text-muted-foreground cursor-help"
                            title="Humanization: Measures how natural, fluent, and human-like the writing sounds."
                        >
                            Humanization
                        </p>
                        <p className={`mt-1 text-lg font-semibold ${getScoreColor(qualityMetrics.avgHumanization ?? undefined)}`}>
                            {qualityMetrics.avgHumanization ?? '—'}
                        </p>
                    </div>
                    <div className="rounded-lg border border-border bg-muted/20 p-3">
                        <p
                            className="text-[11px] uppercase tracking-wide text-muted-foreground cursor-help"
                            title="Grounding: Measures factual support and citation alignment with research sources."
                        >
                            Grounding
                        </p>
                        <p className={`mt-1 text-lg font-semibold ${getScoreColor(qualityMetrics.avgGrounding ?? undefined)}`}>
                            {qualityMetrics.avgGrounding ?? '—'}
                        </p>
                    </div>
                    <div className="rounded-lg border border-border bg-muted/20 p-3">
                        <p
                            className="text-[11px] uppercase tracking-wide text-muted-foreground cursor-help"
                            title="GEO (Generative Engine Optimization): Measures how well content is structured for AI answer engines."
                        >
                            GEO
                        </p>
                        <p className={`mt-1 text-lg font-semibold ${getScoreColor(qualityMetrics.avgGeo ?? undefined)}`}>
                            {qualityMetrics.avgGeo ?? '—'}
                        </p>
                    </div>
                    <div className="rounded-lg border border-border bg-muted/20 p-3">
                        <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Quality Gate Pass</p>
                        <p className="mt-1 text-lg font-semibold text-foreground">
                            {qualityMetrics.gatePassRate != null ? `${qualityMetrics.gatePassRate}%` : '—'}
                        </p>
                    </div>
                    <div className="rounded-lg border border-border bg-muted/20 p-3">
                        <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Exact Metrics Coverage</p>
                        <p className="mt-1 text-lg font-semibold text-foreground">
                            {exactKeywordCoverage != null ? `${exactKeywordCoverage}%` : '—'}
                        </p>
                    </div>
                </motion.div>

                <div className="mt-6 border-t border-border" />

                <motion.div
                    initial={{ opacity: 0, y: 14 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: 0.08 }}
                    className="mt-4"
                >
                    {loading ? (
                        <div className="space-y-2">
                            {[1, 2, 3, 4, 5].map(i => (
                                <div key={i} className="h-14 animate-pulse rounded-lg bg-muted" />
                            ))}
                        </div>
                    ) : filteredArticles.length === 0 ? (
                        <div className="py-20 text-center">
                            <p className="text-sm text-muted-foreground">
                                {search ? 'No items match your search.' : 'No content items yet. Create one to get started.'}
                            </p>
                        </div>
                    ) : (
                        <>
                            <div className="grid grid-cols-[2.5rem_1fr_5.5rem_4rem_4rem_4rem_4rem_4.5rem_4.5rem_5.5rem_6rem_auto] items-center gap-2 border-b border-border px-1 pb-3 text-[11px] uppercase tracking-wider text-muted-foreground">
                                <span className="flex justify-center">
                                    <input
                                        type="checkbox"
                                        className="h-4 w-4 rounded border-border bg-transparent text-primary focus:ring-ring focus:ring-offset-0"
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
                                    className="text-left hover:text-foreground transition"
                                >
                                    Title{sortKey === 'Title' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                                </button>
                                <button
                                    type="button"
                                    onClick={() => handleSort('status')}
                                    className="text-left hover:text-foreground transition"
                                >
                                    Status{sortKey === 'status' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                                </button>
                                <button
                                    type="button"
                                    onClick={() => handleSort('seo_optimization_score')}
                                    className="text-left hover:text-foreground transition cursor-help"
                                    title="SEO: Search Engine Optimization score based on on-page relevance and keyword targeting."
                                >
                                    SEO{sortKey === 'seo_optimization_score' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                                </button>
                                <span
                                    className="text-left cursor-help"
                                    title="HUM: Humanization score."
                                >
                                    Hum
                                </span>
                                <span
                                    className="text-left cursor-help"
                                    title="GRD: Grounding score."
                                >
                                    Grd
                                </span>
                                <span
                                    className="text-left cursor-help"
                                    title="GEO: Generative Engine Optimization score."
                                >
                                    GEO
                                </span>
                                <button
                                    type="button"
                                    onClick={() => handleSort('selected_keyword_search_volume' as keyof Article)}
                                    className="text-left hover:text-foreground transition"
                                >
                                    Vol{sortKey === 'selected_keyword_search_volume' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                                </button>
                                <button
                                    type="button"
                                    onClick={() => handleSort('selected_keyword_difficulty' as keyof Article)}
                                    className="text-left hover:text-foreground transition cursor-help"
                                    title="KD: Keyword Difficulty. Lower values are generally easier to rank for."
                                >
                                    KD{sortKey === 'selected_keyword_difficulty' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                                </button>
                                <button
                                    type="button"
                                    onClick={() => handleSort('traffic_potential_score' as keyof Article)}
                                    className="text-left hover:text-foreground transition cursor-help"
                                    title="Opp: Opportunity score balancing search volume and keyword difficulty."
                                >
                                    Opp{sortKey === 'traffic_potential_score' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                                </button>
                                <span className="text-left">Gate</span>
                                <button
                                    type="button"
                                    onClick={() => handleSort('dateCreatedOn')}
                                    className="text-left hover:text-foreground transition"
                                >
                                    Date{sortKey === 'dateCreatedOn' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                                </button>
                                <span className="text-right">Actions</span>
                            </div>

                            <div className="divide-y divide-border">
                                {filteredArticles.map(article => {
                                    const selected = selectedIds.has(article.id)
                                    const status = getStatusStyle(article, article._source)
                                    const metricSource = getKeywordMetricSource(article)
                                    const keywordEstimated = isKeywordMetricEstimated(article)
                                    const keywordRows = getKeywordRows(article)
                                    const volume = safeNumber((article as any).selected_keyword_search_volume ?? (article as any).total_search_volume)
                                    const difficulty = safeNumber((article as any).selected_keyword_difficulty ?? (article as any).avg_keyword_difficulty)
                                    const opportunity = getKeywordOpportunityScore(article)
                                    return (
                                        <div
                                            key={article.id}
                                            className={`grid grid-cols-[2.5rem_1fr_5.5rem_4rem_4rem_4rem_4rem_4.5rem_4.5rem_5.5rem_6rem_auto] items-center gap-2 px-1 py-3 transition ${
                                                selected ? 'bg-primary/[0.05]' : ''
                                            }`}
                                        >
                                            <span className="flex justify-center">
                                                <input
                                                    type="checkbox"
                                                    className="h-4 w-4 rounded border-border bg-transparent text-primary focus:ring-ring focus:ring-offset-0"
                                                    checked={selected}
                                                    onChange={() => handleToggleSelect(article.id)}
                                                />
                                            </span>

                                            <div className="min-w-0">
                                                <p className="truncate text-sm font-medium text-foreground">
                                                    {article.Title || 'Untitled'}
                                                </p>
                                                {article.userDescription && (
                                                    <p className="mt-0.5 truncate text-xs text-muted-foreground">
                                                        {article.userDescription}
                                                    </p>
                                                )}
                                                <p className="mt-0.5 truncate text-[11px] text-muted-foreground">
                                                    {keywordEstimated ? 'Needs Keyword Refresh' : 'Exact Keyword Metrics'} · {metricSource}
                                                </p>
                                                {getKeywordTelemetryLabel(article) && (
                                                    <p className="mt-0.5 truncate text-[11px] text-muted-foreground">
                                                        {getKeywordTelemetryLabel(article)}
                                                    </p>
                                                )}
                                                <div className="mt-1 flex flex-wrap gap-1">
                                                    {keywordRows.slice(0, 3).map((row, idx) => (
                                                        <span
                                                            key={`${row.keyword}-${idx}`}
                                                            className={`inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] ${
                                                                row.isEstimated
                                                                    ? 'border-amber-500/30 text-amber-400'
                                                                    : 'border-emerald-500/30 text-emerald-400'
                                                            }`}
                                                            title={`Keyword: ${row.keyword} | Vol: ${row.volume} | KD: ${row.difficulty} | CPC: ${row.cpc}`}
                                                        >
                                                            {row.keyword} · V{row.volume} · KD{row.difficulty}
                                                        </span>
                                                    ))}
                                                    {keywordRows.length === 0 && (
                                                        <span className="text-[10px] text-muted-foreground">No keyword metrics yet</span>
                                                    )}
                                                </div>
                                            </div>

                                            <span className={`text-xs font-medium ${status.color} ${
                                                status.label === 'WP Published'
                                                    ? 'inline-flex items-center rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5'
                                                    : ''
                                            }`}>
                                                {status.label}
                                            </span>

                                            <span className={`text-xs font-medium ${getScoreColor(article.seo_optimization_score)}`}>
                                                {article.seo_optimization_score != null ? article.seo_optimization_score : '—'}
                                            </span>
                                            <span className={`text-xs font-medium ${getScoreColor(safeNumber((article as any)?.quality_report?.humanization_score) ?? undefined)}`}>
                                                {safeNumber((article as any)?.quality_report?.humanization_score) ?? '—'}
                                            </span>
                                            <span className={`text-xs font-medium ${getScoreColor(safeNumber((article as any)?.quality_report?.grounding_score) ?? undefined)}`}>
                                                {safeNumber((article as any)?.quality_report?.grounding_score) ?? '—'}
                                            </span>
                                            <span className={`text-xs font-medium ${getScoreColor(safeNumber((article as any)?.quality_report?.geo_score) ?? undefined)}`}>
                                                {safeNumber((article as any)?.quality_report?.geo_score) ?? '—'}
                                            </span>
                                            <span className={`text-xs font-medium ${getScoreColor(opportunity ?? undefined)}`}>
                                                {volume ?? '—'}
                                            </span>
                                            <span className={`text-xs font-medium ${difficulty != null ? getScoreColor(100 - difficulty) : 'text-muted-foreground'}`}>
                                                {difficulty ?? '—'}
                                            </span>
                                            <span className={`text-xs font-medium ${getScoreColor(opportunity ?? undefined)}`}>
                                                {opportunity ?? '—'}
                                            </span>
                                            <span className="text-xs text-muted-foreground">
                                                {(article as any)?.quality_gate?.decision || '—'}
                                            </span>

                                            <span className="text-xs text-muted-foreground">
                                                {new Date(article.dateCreatedOn).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                                            </span>

                                            <div className="flex items-center justify-end gap-1">
                                                <button
                                                    onClick={() => navigate(`/content-studio?id=${article.id}`)}
                                                    className="rounded-md p-1.5 text-muted-foreground transition hover:bg-muted hover:text-primary"
                                                    title="Generate"
                                                >
                                                    <Sparkles className="h-3.5 w-3.5" />
                                                </button>
                                                <button
                                                    onClick={() => navigate(article._source === 'content_ideas' ? `/content-studio?id=${article.id}` : `/article-editor/${article.id}`)}
                                                    className="rounded-md p-1.5 text-muted-foreground transition hover:bg-muted hover:text-foreground"
                                                    title={article._source === 'content_ideas' ? 'Open in Studio' : 'Edit'}
                                                >
                                                    <Edit className="h-3.5 w-3.5" />
                                                </button>
                                                <button
                                                    onClick={() => handleDelete(article.id)}
                                                    className="rounded-md p-1.5 text-muted-foreground transition hover:bg-muted hover:text-destructive"
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

                    {!loading && filteredArticles.length > 0 && (
                        <div className="mt-4 border-t border-border pt-4">
                            <p className="text-xs text-muted-foreground">
                                Showing {filteredArticles.length} of {articles.length} items
                            </p>
                        </div>
                    )}
                </motion.div>
            </div>
        </div>
    )
}
