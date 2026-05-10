import React, { useEffect, useMemo, useState } from 'react'
import { supabase } from '../lib/supabase'
import { useAuth } from '../context/auth-context'
import { useProject } from '../context/project-context'
import type { Article } from '../types'
import { Plus, Search, Trash2, Sparkles, Edit, X, ChevronDown, ChevronUp } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { apiClient } from '../api-client'
import { KeywordIntelligenceModal } from '../components/KeywordIntelligenceModal'
import type { ContentIdea } from '../types/idea-burst'
import { contentIdeasService } from '../services/content-ideas.service'

type LibraryArticle = Article & {
    _source: 'titles' | 'content_ideas'
    _keywordTelemetry?: {
        tier?: string
        calls?: number
        nonZero?: number
        bestVolume?: number
    }
    _topic_id?: string | null
    _project_id?: string | null
    _primary_category_id?: string | null
    _secondary_category_id?: string | null
    _project_name?: string | null
    _primary_category_name?: string | null
    _secondary_category_name?: string | null
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
    topic_id?: string | null
}

type ProjectCategory = {
    id: string
    name: string
    level: number
    parent_category_id: string | null
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
    const editing = hasWrittenContent(article)
    const isWpPublished =
        normalized === 'wp published' ||
        normalized === 'scheduled' ||
        wpStatus === 'publish' ||
        wpStatus === 'future'

    if (isWpPublished) {
        return { label: 'WP Published', color: 'text-emerald-500 dark:text-emerald-400' }
    }
    if (editing || normalized === 'editing' || normalized === 'review') {
        return { label: 'Editing', color: 'text-cyan-500 dark:text-cyan-400' }
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
    const parseKeywordValues = (value: any, depth = 0): string[] => {
        if (depth > 5 || value === null || value === undefined) return []
        if (Array.isArray(value)) return value.flatMap((item) => parseKeywordValues(item, depth + 1))
        if (typeof value === 'object') {
            if (typeof value.keyword === 'string') return parseKeywordValues(value.keyword, depth + 1)
            return []
        }
        if (typeof value !== 'string') return []
        const raw = value.trim()
        if (!raw) return []
        try {
            const parsed = JSON.parse(raw)
            if (parsed !== raw) return parseKeywordValues(parsed, depth + 1)
        } catch {
            // Fallback parsing.
        }
        if ((raw.startsWith('"') && raw.endsWith('"')) || (raw.startsWith("'") && raw.endsWith("'"))) {
            return parseKeywordValues(raw.slice(1, -1), depth + 1)
        }
        const unescaped = raw.replace(/\\"/g, '"').replace(/\\'/g, "'").replace(/\\\\/g, '\\').trim()
        if (unescaped !== raw) return parseKeywordValues(unescaped, depth + 1)
        const split = raw.split(',').map((s) => s.trim()).filter(Boolean)
        return split.length > 1 ? split : [raw]
    }

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
    const fallbackSecondary = parseKeywordValues(fallbackSecondaryRaw)
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
    const ITEMS_PER_PAGE = 5
    const { user } = useAuth()
    const { activeProject, projects } = useProject()
    const navigate = useNavigate()
    const [articles, setArticles] = useState<LibraryArticle[]>([])
    const [loading, setLoading] = useState(true)
    const [sortKey, setSortKey] = useState<keyof Article>('dateCreatedOn')
    const [sortAsc, setSortAsc] = useState(false)
    const [search, setSearch] = useState('')
    const [exactMetricsOnly, setExactMetricsOnly] = useState(false)
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
    const [refreshingKeywords, setRefreshingKeywords] = useState(false)
    const [expandedKeywordRows, setExpandedKeywordRows] = useState<Set<string>>(new Set())
    const [projectFilter, setProjectFilter] = useState('')
    const [primaryCategoryFilter, setPrimaryCategoryFilter] = useState('')
    const [secondaryCategoryFilter, setSecondaryCategoryFilter] = useState('')
    const [currentPage, setCurrentPage] = useState(1)
    const [projectCategories, setProjectCategories] = useState<ProjectCategory[]>([])
    // Keyword Intelligence Modal (replaces legacy Keyword Lab)
    const [kwIntelOpen, setKwIntelOpen] = useState(false)
    const [kwIntelArticle, setKwIntelArticle] = useState<LibraryArticle | null>(null)

    useEffect(() => {
        if (activeProject?.id && !projectFilter) {
            setProjectFilter(activeProject.id)
        }
    }, [activeProject?.id])

    useEffect(() => {
        const loadProjectCategories = async () => {
            if (!user?.id || !projectFilter) {
                setProjectCategories([])
                return
            }
            const { data, error } = await supabase
                .from('project_categories')
                .select('id, name, level, parent_category_id')
                .eq('user_id', user.id)
                .eq('project_id', projectFilter)
                .order('level', { ascending: true })
                .order('sort_order', { ascending: true })
                .order('name', { ascending: true })

            if (error) {
                console.error('Failed to load project categories:', error)
                setProjectCategories([])
                return
            }
            setProjectCategories((data || []) as ProjectCategory[])
        }
        void loadProjectCategories()
    }, [projectFilter, user?.id])

    useEffect(() => {
        setPrimaryCategoryFilter('')
        setSecondaryCategoryFilter('')
    }, [projectFilter])

    useEffect(() => {
        if (!secondaryCategoryFilter) return
        const exists = projectCategories.some(
            (category) => category.id === secondaryCategoryFilter && category.parent_category_id === primaryCategoryFilter
        )
        if (!exists) {
            setSecondaryCategoryFilter('')
        }
    }, [primaryCategoryFilter, secondaryCategoryFilter, projectCategories])

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
                topic_id: row.topic_id ?? null,
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

            const ideaTopicById = new Map<string, string>()
            for (const idea of ideaRows) {
                if (idea.id && idea.topic_id) {
                    ideaTopicById.set(String(idea.id), String(idea.topic_id))
                }
            }

            const topicIds = new Set<string>()
            for (const row of titleRows as any[]) {
                const topicId = row.topic_id || (row.source_idea_id ? ideaTopicById.get(String(row.source_idea_id)) : null)
                if (topicId) topicIds.add(String(topicId))
            }
            for (const idea of ideaRows) {
                if (idea.topic_id) topicIds.add(String(idea.topic_id))
            }

            const topicById = new Map<string, { project_id?: string | null; primary_category_id?: string | null; secondary_category_id?: string | null }>()
            const projectNameById = new Map<string, string>()
            const categoryNameById = new Map<string, string>()

            if (topicIds.size > 0) {
                const topicIdsArray = Array.from(topicIds)
                const { data: topicRows, error: topicError } = await supabase
                    .from('research_topics')
                    .select('id, project_id, primary_category_id, secondary_category_id')
                    .in('id', topicIdsArray)

                if (topicError) {
                    console.warn('[ContentLibrary] research_topics metadata query failed', topicError)
                } else {
                    for (const row of (topicRows || []) as any[]) {
                        topicById.set(String(row.id), {
                            project_id: row.project_id || null,
                            primary_category_id: row.primary_category_id || null,
                            secondary_category_id: row.secondary_category_id || null,
                        })
                    }
                }
            }

            const projectIds = Array.from(
                new Set(
                    Array.from(topicById.values())
                        .map((row) => row.project_id)
                        .filter(Boolean)
                )
            ) as string[]
            const categoryIds = Array.from(
                new Set(
                    Array.from(topicById.values())
                        .flatMap((row) => [row.primary_category_id, row.secondary_category_id])
                        .filter(Boolean)
                )
            ) as string[]

            if (projectIds.length > 0) {
                const { data: projectRows, error: projectError } = await supabase
                    .from('projects')
                    .select('id, domain, app_name')
                    .in('id', projectIds)

                if (projectError) {
                    console.warn('[ContentLibrary] projects metadata query failed', projectError)
                } else {
                    for (const row of (projectRows || []) as any[]) {
                        projectNameById.set(String(row.id), row.domain || row.app_name || 'Untitled Project')
                    }
                }
            }

            if (categoryIds.length > 0) {
                const { data: categoryRows, error: categoryError } = await supabase
                    .from('project_categories')
                    .select('id, name')
                    .in('id', categoryIds)

                if (categoryError) {
                    console.warn('[ContentLibrary] project_categories metadata query failed', categoryError)
                } else {
                    for (const row of (categoryRows || []) as any[]) {
                        categoryNameById.set(String(row.id), String(row.name || ''))
                    }
                }
            }

            const enrichedTitleRows = titleRows.map((row: any) => {
                const ideaTelemetry =
                    telemetryByIdeaId.get(String(row.source_idea_id || '')) ||
                    telemetryByTitleId.get(String(row.id || ''))
                const topicId = row.topic_id || (row.source_idea_id ? ideaTopicById.get(String(row.source_idea_id)) : null)
                const topicMeta = topicId ? topicById.get(String(topicId)) : null
                return {
                    ...row,
                    _keywordTelemetry: ideaTelemetry,
                    _topic_id: topicId || null,
                    _project_id: topicMeta?.project_id || null,
                    _primary_category_id: topicMeta?.primary_category_id || null,
                    _secondary_category_id: topicMeta?.secondary_category_id || null,
                    _project_name: topicMeta?.project_id ? (projectNameById.get(String(topicMeta.project_id)) || null) : null,
                    _primary_category_name: topicMeta?.primary_category_id ? (categoryNameById.get(String(topicMeta.primary_category_id)) || null) : null,
                    _secondary_category_name: topicMeta?.secondary_category_id ? (categoryNameById.get(String(topicMeta.secondary_category_id)) || null) : null,
                }
            })
            const libraryIdeas = ideaRows.filter((idea) => {
                const status = String(idea.status || '').toLowerCase()
                const hasTitleMirror = Boolean(idea.titles_record_id && titleIdSet.has(idea.titles_record_id))
                const alreadyLinked = titleBySourceIdeaId.has(idea.id)
                const isArchived = status === 'archived'
                return !isArchived && !hasTitleMirror && !alreadyLinked
            })

            const mappedIdeas: LibraryArticle[] = libraryIdeas.map((idea) => ({
                ...(idea.topic_id ? (() => {
                    const topicMeta = topicById.get(String(idea.topic_id))
                    return {
                        _topic_id: idea.topic_id,
                        _project_id: topicMeta?.project_id || null,
                        _primary_category_id: topicMeta?.primary_category_id || null,
                        _secondary_category_id: topicMeta?.secondary_category_id || null,
                        _project_name: topicMeta?.project_id ? (projectNameById.get(String(topicMeta.project_id)) || null) : null,
                        _primary_category_name: topicMeta?.primary_category_id ? (categoryNameById.get(String(topicMeta.primary_category_id)) || null) : null,
                        _secondary_category_name: topicMeta?.secondary_category_id ? (categoryNameById.get(String(topicMeta.secondary_category_id)) || null) : null,
                    }
                })() : {
                    _topic_id: null,
                    _project_id: null,
                    _primary_category_id: null,
                    _secondary_category_id: null,
                    _project_name: null,
                    _primary_category_name: null,
                    _secondary_category_name: null,
                }),
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
                contentIdeasVisible: mappedIdeas.length,
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

    const primaryCategories = useMemo(
        () => projectCategories.filter((category) => category.level === 1),
        [projectCategories]
    )
    const secondaryCategories = useMemo(
        () =>
            projectCategories.filter(
                (category) => category.level === 2 && (!primaryCategoryFilter || category.parent_category_id === primaryCategoryFilter)
            ),
        [projectCategories, primaryCategoryFilter]
    )

    const filteredArticles = useMemo(
        () =>
            sortedArticles.filter(article => {
                if (projectFilter && article._project_id !== projectFilter) return false
                if (primaryCategoryFilter && article._primary_category_id !== primaryCategoryFilter) return false
                if (secondaryCategoryFilter && article._secondary_category_id !== secondaryCategoryFilter) return false

                const haystack = [
                    article.Title,
                    article.userDescription,
                    article._project_name,
                    article._primary_category_name,
                    article._secondary_category_name,
                ]
                    .filter(Boolean)
                    .join(' ')
                    .toLowerCase()
                const titleMatch = haystack.includes(search.toLowerCase())
                if (!titleMatch) return false
                if (!exactMetricsOnly) return true
                return !isKeywordMetricEstimated(article)
            }),
        [sortedArticles, search, exactMetricsOnly, projectFilter, primaryCategoryFilter, secondaryCategoryFilter],
    )

    useEffect(() => {
        setCurrentPage(1)
    }, [search, exactMetricsOnly, projectFilter, primaryCategoryFilter, secondaryCategoryFilter, sortKey, sortAsc])

    const totalPages = useMemo(
        () => Math.max(1, Math.ceil(filteredArticles.length / ITEMS_PER_PAGE)),
        [filteredArticles.length, ITEMS_PER_PAGE]
    )

    useEffect(() => {
        setCurrentPage((prev) => Math.min(prev, totalPages))
    }, [totalPages])

    const paginatedArticles = useMemo(() => {
        const start = (currentPage - 1) * ITEMS_PER_PAGE
        return filteredArticles.slice(start, start + ITEMS_PER_PAGE)
    }, [currentPage, filteredArticles, ITEMS_PER_PAGE])

    const pageStart = filteredArticles.length === 0 ? 0 : (currentPage - 1) * ITEMS_PER_PAGE + 1
    const pageEnd = Math.min(currentPage * ITEMS_PER_PAGE, filteredArticles.length)

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

    const handleOpenKnowledgeGaps = () => {
        navigate('/knowledge-gaps')
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

    const openKwIntel = (article: LibraryArticle) => {
        setKwIntelArticle(article)
        setKwIntelOpen(true)
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
    const allSelected = paginatedArticles.length > 0 && paginatedArticles.every((article) => selectedIds.has(article.id))
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
    const selectedCount = selectedIds.size

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
                            <span>Exact Metrics Only</span>
                        </button>
                        {selectedIds.size > 0 && (
                            <button
                                onClick={handleRefreshSelectedKeywords}
                                disabled={refreshingKeywords}
                                className={`inline-flex h-10 items-center gap-1.5 rounded-lg border px-3.5 text-sm transition ${
                                    refreshingKeywords
                                        ? 'cursor-not-allowed border-border bg-muted/40 text-muted-foreground/60'
                                        : 'border-emerald-500/20 bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/15'
                                }`}
                            >
                                <span>
                                    {refreshingKeywords
                                        ? 'Refreshing…'
                                        : selectedRefreshableCount > 0
                                            ? `Refresh Keywords (${selectedRefreshableCount}/${selectedCount})`
                                            : `Refresh Keywords (${selectedCount})`}
                                </span>
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
                            onClick={handleOpenKnowledgeGaps}
                            className="inline-flex h-10 items-center gap-1.5 rounded-lg border border-blue-500/20 bg-blue-500/10 px-3.5 text-sm text-blue-400 transition hover:bg-blue-500/15"
                        >
                            <Sparkles className="h-3.5 w-3.5" />
                            <span>Knowledge Gaps</span>
                        </button>
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
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.25, delay: 0.05 }}
                    className="mt-3 grid gap-3 md:grid-cols-3"
                >
                    <select
                        className="h-10 rounded-lg border border-border bg-muted/50 px-3 text-sm text-foreground outline-none focus:border-ring/50"
                        value={projectFilter}
                        onChange={(e) => setProjectFilter(e.target.value)}
                    >
                        <option value="">All projects</option>
                        {projects.map((project) => (
                            <option key={project.id} value={project.id}>
                                {project.domain || project.app_name || 'Untitled Project'}
                            </option>
                        ))}
                    </select>

                    <select
                        className="h-10 rounded-lg border border-border bg-muted/50 px-3 text-sm text-foreground outline-none focus:border-ring/50 disabled:opacity-50"
                        value={primaryCategoryFilter}
                        disabled={!projectFilter}
                        onChange={(e) => {
                            setPrimaryCategoryFilter(e.target.value)
                            setSecondaryCategoryFilter('')
                        }}
                    >
                        <option value="">All categories</option>
                        {primaryCategories.map((category) => (
                            <option key={category.id} value={category.id}>
                                {category.name}
                            </option>
                        ))}
                    </select>

                    <select
                        className="h-10 rounded-lg border border-border bg-muted/50 px-3 text-sm text-foreground outline-none focus:border-ring/50 disabled:opacity-50"
                        value={secondaryCategoryFilter}
                        disabled={!projectFilter || !primaryCategoryFilter}
                        onChange={(e) => setSecondaryCategoryFilter(e.target.value)}
                    >
                        <option value="">All subcategories</option>
                        {secondaryCategories.map((category) => (
                            <option key={category.id} value={category.id}>
                                {category.name}
                            </option>
                        ))}
                    </select>
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
                            <div className="flex items-center gap-3 border-b border-border px-1 pb-3 text-[11px] uppercase tracking-wider text-muted-foreground">
                                <span className="flex justify-center">
                                    <input
                                        type="checkbox"
                                        className="h-4 w-4 rounded border-border bg-transparent text-primary focus:ring-ring focus:ring-offset-0"
                                        checked={allSelected}
                                        onChange={(e) => {
                                            setSelectedIds(e.target.checked
                                                ? new Set(paginatedArticles.map(a => a.id))
                                                : new Set())
                                        }}
                                    />
                                </span>
                                <span className="text-left">Select</span>
                                <button
                                    type="button"
                                    onClick={() => handleSort('dateCreatedOn')}
                                    className="text-left hover:text-foreground transition"
                                >
                                    Date{sortKey === 'dateCreatedOn' ? (sortAsc ? ' ▲' : ' ▼') : ''}
                                </button>
                            </div>

                            <div className="divide-y divide-border">
                                {paginatedArticles.map(article => {
                                    const selected = selectedIds.has(article.id)
                                    const status = getStatusStyle(article, article._source)
                                    const metricSource = getKeywordMetricSource(article)
                                    const keywordEstimated = isKeywordMetricEstimated(article)
                                    const keywordRows = getKeywordRows(article)
                                    const volume = safeNumber((article as any).selected_keyword_search_volume ?? (article as any).total_search_volume)
                                    const difficulty = safeNumber((article as any).selected_keyword_difficulty ?? (article as any).avg_keyword_difficulty)
                                    const opportunity = getKeywordOpportunityScore(article)
                                    const keywordsExpanded = expandedKeywordRows.has(article.id)
                                    return (
                                        <div
                                            key={article.id}
                                            className={`px-1 py-4 transition ${
                                                selected ? 'bg-primary/[0.05] rounded-lg' : ''
                                            }`}
                                        >
                                            <div className="flex items-start gap-3">
                                                <span className="mt-1 flex justify-center">
                                                    <input
                                                        type="checkbox"
                                                        className="h-4 w-4 rounded border-border bg-transparent text-primary focus:ring-ring focus:ring-offset-0"
                                                        checked={selected}
                                                        onChange={() => handleToggleSelect(article.id)}
                                                    />
                                                </span>

                                                <div className="min-w-0 flex-1">
                                                    <div className="flex flex-wrap items-start justify-between gap-3">
                                                        <div className="min-w-0 flex-1">
                                                            <p className="text-xl font-semibold text-foreground leading-tight">
                                                                {article.Title || 'Untitled'}
                                                            </p>
                                                            {article.userDescription && (
                                                                <p className="mt-1 text-sm text-muted-foreground">
                                                                    {article.userDescription}
                                                                </p>
                                                            )}
                                                            <p className="mt-1 text-xs text-muted-foreground">
                                                                {keywordEstimated ? 'Needs Keyword Refresh' : 'Exact Keyword Metrics'} · {metricSource}
                                                            </p>
                                                            {getKeywordTelemetryLabel(article) && (
                                                                <p className="mt-0.5 text-xs text-muted-foreground">
                                                                    {getKeywordTelemetryLabel(article)}
                                                                </p>
                                                            )}
                                                            {(article._project_name || article._primary_category_name || article._secondary_category_name) && (
                                                                <div className="mt-2 flex flex-wrap gap-1.5">
                                                                    {article._project_name && (
                                                                        <span className="rounded-full border border-border bg-muted/50 px-2 py-0.5 text-[10px] text-foreground">
                                                                            {article._project_name}
                                                                        </span>
                                                                    )}
                                                                    {(article._primary_category_name || article._secondary_category_name) && (
                                                                        <span className="rounded-full border border-primary/20 bg-primary/10 px-2 py-0.5 text-[10px] text-primary">
                                                                            {[article._primary_category_name, article._secondary_category_name].filter(Boolean).join(' / ')}
                                                                        </span>
                                                                    )}
                                                                </div>
                                                            )}
                                                        </div>

                                                        <div className="flex items-center gap-2">
                                                            <span className={`text-xs font-medium ${status.color} ${
                                                                status.label === 'WP Published'
                                                                    ? 'inline-flex items-center rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5'
                                                                    : ''
                                                            }`}>
                                                                {status.label}
                                                            </span>
                                                            <span className="text-xs text-muted-foreground">
                                                                {new Date(article.dateCreatedOn).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                                                            </span>
                                                            <div className="ml-1 flex items-center gap-1">
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
                                                                {article._source === 'titles' && (
                                                                    <button
                                                                        onClick={() => openKwIntel(article)}
                                                                        className="rounded-md p-1.5 text-muted-foreground transition hover:bg-muted hover:text-indigo-400"
                                                                        title="Keyword Intelligence"
                                                                    >
                                                                        <span className="text-[11px] font-semibold">KW</span>
                                                                    </button>
                                                                )}
                                                            </div>
                                                        </div>
                                                    </div>

                                                    <div className="mt-3 grid grid-cols-2 gap-2 md:grid-cols-4 lg:grid-cols-8">
                                                        <div className="rounded-md border border-border/70 bg-muted/20 p-2">
                                                            <p className="text-[10px] uppercase tracking-wide text-muted-foreground">SEO</p>
                                                            <p className={`text-xs font-semibold ${getScoreColor(article.seo_optimization_score)}`}>{article.seo_optimization_score ?? '—'}</p>
                                                        </div>
                                                        <div className="rounded-md border border-border/70 bg-muted/20 p-2">
                                                            <p className="text-[10px] uppercase tracking-wide text-muted-foreground">HUM</p>
                                                            <p className={`text-xs font-semibold ${getScoreColor(safeNumber((article as any)?.quality_report?.humanization_score) ?? undefined)}`}>{safeNumber((article as any)?.quality_report?.humanization_score) ?? '—'}</p>
                                                        </div>
                                                        <div className="rounded-md border border-border/70 bg-muted/20 p-2">
                                                            <p className="text-[10px] uppercase tracking-wide text-muted-foreground">GRD</p>
                                                            <p className={`text-xs font-semibold ${getScoreColor(safeNumber((article as any)?.quality_report?.grounding_score) ?? undefined)}`}>{safeNumber((article as any)?.quality_report?.grounding_score) ?? '—'}</p>
                                                        </div>
                                                        <div className="rounded-md border border-border/70 bg-muted/20 p-2">
                                                            <p className="text-[10px] uppercase tracking-wide text-muted-foreground">GEO</p>
                                                            <p className={`text-xs font-semibold ${getScoreColor(safeNumber((article as any)?.quality_report?.geo_score) ?? undefined)}`}>{safeNumber((article as any)?.quality_report?.geo_score) ?? '—'}</p>
                                                        </div>
                                                        <div className="rounded-md border border-border/70 bg-muted/20 p-2">
                                                            <p className="text-[10px] uppercase tracking-wide text-muted-foreground">Vol</p>
                                                            <p className={`text-xs font-semibold ${getScoreColor(opportunity ?? undefined)}`}>{volume ?? '—'}</p>
                                                        </div>
                                                        <div className="rounded-md border border-border/70 bg-muted/20 p-2">
                                                            <p className="text-[10px] uppercase tracking-wide text-muted-foreground">KD</p>
                                                            <p className={`text-xs font-semibold ${difficulty != null ? getScoreColor(100 - difficulty) : 'text-muted-foreground'}`}>{difficulty ?? '—'}</p>
                                                        </div>
                                                        <div className="rounded-md border border-border/70 bg-muted/20 p-2">
                                                            <p className="text-[10px] uppercase tracking-wide text-muted-foreground">Opp</p>
                                                            <p className={`text-xs font-semibold ${getScoreColor(opportunity ?? undefined)}`}>{opportunity ?? '—'}</p>
                                                        </div>
                                                        <div className="rounded-md border border-border/70 bg-muted/20 p-2">
                                                            <p className="text-[10px] uppercase tracking-wide text-muted-foreground">Gate</p>
                                                            <p className="text-xs font-semibold text-muted-foreground">{(article as any)?.quality_gate?.decision || '—'}</p>
                                                        </div>
                                                    </div>

                                                    <div className="mt-3">
                                                        <button
                                                            type="button"
                                                            onClick={() => {
                                                                setExpandedKeywordRows((prev) => {
                                                                    const next = new Set(prev)
                                                                    if (next.has(article.id)) next.delete(article.id)
                                                                    else next.add(article.id)
                                                                    return next
                                                                })
                                                            }}
                                                            className="inline-flex items-center gap-1 rounded-md border border-border bg-muted/20 px-2 py-1 text-xs text-foreground hover:bg-muted/30"
                                                        >
                                                            {keywordsExpanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                                                            Keywords ({keywordRows.length})
                                                        </button>
                                                        {keywordsExpanded && (
                                                            <div className="mt-2 overflow-x-auto rounded-lg border border-border">
                                                                {keywordRows.length === 0 ? (
                                                                    <p className="p-3 text-xs text-muted-foreground">No keyword metrics yet</p>
                                                                ) : (
                                                                    <table className="w-full text-left text-xs">
                                                                        <thead className="bg-muted/20 text-muted-foreground">
                                                                            <tr>
                                                                                <th className="px-3 py-2 font-medium">Keyword</th>
                                                                                <th className="px-3 py-2 font-medium">Vol</th>
                                                                                <th className="px-3 py-2 font-medium">KD</th>
                                                                                <th className="px-3 py-2 font-medium">CPC</th>
                                                                            </tr>
                                                                        </thead>
                                                                        <tbody className="divide-y divide-border">
                                                                            {keywordRows.map((row, idx) => (
                                                                                <tr key={`${row.keyword}-${idx}`}>
                                                                                    <td className="px-3 py-2 text-foreground">{row.keyword}</td>
                                                                                    <td className="px-3 py-2 text-muted-foreground">{row.volume}</td>
                                                                                    <td className="px-3 py-2 text-muted-foreground">{row.difficulty}</td>
                                                                                    <td className="px-3 py-2 text-muted-foreground">{row.cpc}</td>
                                                                                </tr>
                                                                            ))}
                                                                        </tbody>
                                                                    </table>
                                                                )}
                                                            </div>
                                                        )}
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    )
                                })}
                            </div>
                        </>
                    )}

                    {!loading && filteredArticles.length > 0 && (
                        <div className="mt-4 flex flex-col gap-3 border-t border-border pt-4 sm:flex-row sm:items-center sm:justify-between">
                            <p className="text-xs text-muted-foreground">
                                Showing {pageStart}-{pageEnd} of {filteredArticles.length} filtered items
                                {filteredArticles.length !== articles.length ? ` (${articles.length} total)` : ''}
                            </p>
                            <div className="flex items-center justify-end gap-2">
                                <button
                                    type="button"
                                    onClick={() => setCurrentPage((prev) => Math.max(1, prev - 1))}
                                    disabled={currentPage === 1}
                                    className="inline-flex h-9 items-center rounded-lg border border-border bg-muted/30 px-3 text-sm text-foreground transition hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
                                >
                                    Previous
                                </button>
                                <span className="text-xs text-muted-foreground">
                                    Page {currentPage} of {totalPages}
                                </span>
                                <button
                                    type="button"
                                    onClick={() => setCurrentPage((prev) => Math.min(totalPages, prev + 1))}
                                    disabled={currentPage === totalPages}
                                    className="inline-flex h-9 items-center rounded-lg border border-border bg-muted/30 px-3 text-sm text-foreground transition hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
                                >
                                    Next
                                </button>
                            </div>
                        </div>
                    )}
                </motion.div>

                {/* ── Keyword Intelligence Modal (Titles context) ── */}
                {kwIntelOpen && kwIntelArticle && (() => {
                    const parseKeywordValues = (value: any, depth = 0): string[] => {
                        if (depth > 5 || value === null || value === undefined) return []
                        if (Array.isArray(value)) return value.flatMap((item) => parseKeywordValues(item, depth + 1))
                        if (typeof value === 'object') {
                            if (typeof value.keyword === 'string') return parseKeywordValues(value.keyword, depth + 1)
                            return []
                        }
                        if (typeof value !== 'string') return []
                        const raw = value.trim()
                        if (!raw) return []
                        try {
                            const parsed = JSON.parse(raw)
                            if (parsed !== raw) return parseKeywordValues(parsed, depth + 1)
                        } catch {
                            // Fallback parsing.
                        }
                        if ((raw.startsWith('"') && raw.endsWith('"')) || (raw.startsWith("'") && raw.endsWith("'"))) {
                            return parseKeywordValues(raw.slice(1, -1), depth + 1)
                        }
                        const unescaped = raw.replace(/\\"/g, '"').replace(/\\'/g, "'").replace(/\\\\/g, '\\').trim()
                        if (unescaped !== raw) return parseKeywordValues(unescaped, depth + 1)
                        const split = raw.split(',').map((s) => s.trim()).filter(Boolean)
                        return split.length > 1 ? split : [raw]
                    }

                    // Build a ContentIdea-shaped object from the Titles record.
                    // The modal reads raw_dataforseo_output for its table;
                    // selections are persisted back to Titles via onSave.
                    const ideaProxy = {
                        id: kwIntelArticle.id,
                        user_id: kwIntelArticle.user_id,
                        title: kwIntelArticle.Title ?? '',
                        content_type: (kwIntelArticle.content_type ?? 'blog') as any,
                        primary_keywords: (() => {
                            const raw = (kwIntelArticle as any).primary_keywords ?? (kwIntelArticle as any).primary_keyword
                            return parseKeywordValues(raw)
                        })(),
                        secondary_keywords: (() => {
                            const raw = (kwIntelArticle as any).secondary_keywords ?? (kwIntelArticle as any).secondary_keywords_json
                            return parseKeywordValues(raw)
                        })(),
                        search_phrase: (kwIntelArticle as any).search_phrase ?? (kwIntelArticle as any).primary_keyword ?? '',
                        raw_dataforseo_output: (kwIntelArticle as any).raw_dataforseo_output ?? null,
                        seo_optimization_score: kwIntelArticle.seo_optimization_score ?? 0,
                        traffic_potential_score: 0,
                        total_search_volume: (kwIntelArticle as any).selected_keyword_search_volume ?? null,
                        average_difficulty: (kwIntelArticle as any).selected_keyword_difficulty ?? null,
                        average_cpc: null,
                        created_at: kwIntelArticle.dateCreatedOn ?? new Date().toISOString(),
                        topic_id: '',
                    } satisfies ContentIdea

                    return (
                        <KeywordIntelligenceModal
                            isOpen={kwIntelOpen}
                            onClose={() => setKwIntelOpen(false)}
                            idea={ideaProxy}
                            onSave={async (primary, secondary, metrics, rawOutput) => {
                                if (!user) return false
                                const primaryClean = String(primary || '').trim()
                                const secondaryClean = Array.from(
                                    new Set(
                                        (Array.isArray(secondary) ? secondary : [])
                                            .map((k) => String(k || '').trim())
                                            .filter(Boolean)
                                    )
                                ).filter((k) => k !== primaryClean)
                                if (!primaryClean) return false
                                const ok = await contentIdeasService.updateTitleKeywordSelection(
                                    kwIntelArticle.id,
                                    user.id,
                                    primaryClean,
                                    secondaryClean,
                                    metrics,
                                    rawOutput
                                )
                                if (ok) {
                                    // Optimistically update local state so the card
                                    // reflects the new primary keyword immediately.
                                    setArticles((prev) => prev.map((a) =>
                                        a.id === kwIntelArticle.id
                                            ? {
                                                ...a,
                                                primary_keywords: [primaryClean],
                                                secondary_keywords: secondaryClean,
                                                search_phrase: primaryClean,
                                                selected_keyword_search_volume: metrics.volume ?? undefined,
                                                selected_keyword_difficulty: metrics.difficulty ?? undefined,
                                                raw_dataforseo_output: rawOutput,
                                            } as any
                                            : a
                                    ))
                                }
                                return ok
                            }}
                            onSaved={() => setKwIntelOpen(false)}
                        />
                    )
                })()}

            </div>
        </div>
    )
}
