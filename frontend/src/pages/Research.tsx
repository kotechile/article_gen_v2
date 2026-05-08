
import * as React from "react"
import { useAuth } from "@/context/auth-context"
import { useProject } from "@/context/project-context"
import { researchRebuildService } from "@/services/research-rebuild.service"
import { researchTopicsService } from "@/services/research-topics.service"
import type { ResearchTopic } from "@/types/research"
import type { ResearchRebuildTopicScopeSummary, ResearchRebuildWorkflowRunSummary } from "@/types/research-rebuild"
import { supabase } from "@/lib/supabase"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { Search, Clock, Trash2, ArrowUpRight, Loader2, Archive, RotateCcw, ChevronDown, ChevronUp, Star, Sparkles } from "lucide-react"
import { useNavigate } from "react-router-dom"
import { motion } from "framer-motion"

type ProjectCategory = {
    id: string
    name: string
    level: number
    parent_category_id: string | null
}

type RebuildSummary = {
    runCount: number
    latestRun: ResearchRebuildWorkflowRunSummary | null
    dominantRoute: string | null
}

export function Research() {
    const { user, isLoading: authLoading } = useAuth()
    const { projects, activeProject } = useProject()
    const navigate = useNavigate()
    const [topics, setTopics] = React.useState<ResearchTopic[]>([])
    const [loading, setLoading] = React.useState(true)
    const [searchTerm, setSearchTerm] = React.useState("")
    const [error, setError] = React.useState<string | null>(null)
    const [projectFilter, setProjectFilter] = React.useState<string>("")
    const [primaryCategoryFilter, setPrimaryCategoryFilter] = React.useState<string>("")
    const [secondaryCategoryFilter, setSecondaryCategoryFilter] = React.useState<string>("")
    const [projectCategories, setProjectCategories] = React.useState<ProjectCategory[]>([])
    const [rebuildTopicSummaries, setRebuildTopicSummaries] = React.useState<ResearchRebuildTopicScopeSummary[]>([])
    const [topicPendingDelete, setTopicPendingDelete] = React.useState<ResearchTopic | null>(null)
    const [deleteLoading, setDeleteLoading] = React.useState(false)
    const [archivingTopicIds, setArchivingTopicIds] = React.useState<Set<string>>(new Set())
    const [expandedArchivedIds, setExpandedArchivedIds] = React.useState<Set<string>>(new Set())
    const [ratingTopicIds, setRatingTopicIds] = React.useState<Set<string>>(new Set())

    React.useEffect(() => {
        // Default to the active project when available.
        if (activeProject?.id && !projectFilter) {
            setProjectFilter(activeProject.id)
        }
    }, [activeProject?.id])

    const [page, setPage] = React.useState(1)
    const [hasMore, setHasMore] = React.useState(true)
    const PAGE_SIZE = 12

    React.useEffect(() => {
        if (!authLoading && user) {
            setPage(1)
            loadTopics(1, false)
        }
    }, [authLoading, user, projectFilter, primaryCategoryFilter, secondaryCategoryFilter])

    React.useEffect(() => {
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

    React.useEffect(() => {
        const loadRebuildTopicSummaries = async () => {
            const projectIds = projectFilter
                ? [projectFilter]
                : Array.from(
                    new Set(
                        (topics || [])
                            .map((topic) => String(topic.project_id || ''))
                            .filter(Boolean),
                    ),
                )

            if (projectIds.length === 0) {
                setRebuildTopicSummaries([])
                return
            }

            try {
                const response = await researchRebuildService.listTopicSummaries({
                    project_id: projectFilter || undefined,
                    project_ids: projectFilter ? undefined : projectIds,
                    primary_category_id: primaryCategoryFilter || undefined,
                    secondary_category_id: secondaryCategoryFilter || undefined,
                    limit: 200,
                })
                setRebuildTopicSummaries(response.items || [])
            } catch (err) {
                console.error('Failed to load rebuild topic summaries for research list:', err)
                setRebuildTopicSummaries([])
            }
        }
        void loadRebuildTopicSummaries()
    }, [primaryCategoryFilter, projectFilter, secondaryCategoryFilter, topics])

    React.useEffect(() => {
        setPrimaryCategoryFilter("")
        setSecondaryCategoryFilter("")
    }, [projectFilter])

    React.useEffect(() => {
        if (!secondaryCategoryFilter) return
        const exists = projectCategories.some(
            (category) => category.id === secondaryCategoryFilter && category.parent_category_id === primaryCategoryFilter
        )
        if (!exists) {
            setSecondaryCategoryFilter("")
        }
    }, [primaryCategoryFilter, secondaryCategoryFilter, projectCategories])

    const loadTopics = async (pageNum: number = 1, append: boolean = false) => {
        try {
            setLoading(true)
            const response = await researchTopicsService.listResearchTopics({
                order_by: 'created_at',
                order_direction: 'desc',
                page: pageNum,
                size: PAGE_SIZE,
                project_id: projectFilter || undefined,
                primary_category_id: primaryCategoryFilter || undefined,
                secondary_category_id: secondaryCategoryFilter || undefined,
            })

            if (append) {
                setTopics(prev => [...prev, ...response.items])
            } else {
                setTopics(response.items)
            }

            setHasMore(response.items.length === PAGE_SIZE && response.has_next)
            setError(null)
        } catch (err) {
            console.error('Failed to load topics:', err)
            setError('Failed to load research topics')
        } finally {
            setLoading(false)
        }
    }

    const handleLoadMore = () => {
        const nextPage = page + 1
        setPage(nextPage)
        loadTopics(nextPage, true)
    }

    const handleDeleteTopic = async (e: React.MouseEvent, topic: ResearchTopic) => {
        e.stopPropagation()
        setTopicPendingDelete(topic)
    }

    const handleConfirmDelete = async () => {
        if (!topicPendingDelete) return
        setDeleteLoading(true)
        try {
            await researchTopicsService.deleteResearchTopic(topicPendingDelete.id)
            setTopics((prev) => prev.filter((topic) => topic.id !== topicPendingDelete.id))
            setExpandedArchivedIds((current) => {
                const next = new Set(current)
                next.delete(topicPendingDelete.id)
                return next
            })
            setTopicPendingDelete(null)
        } catch (err) {
            console.error('Failed to delete topic:', err)
            setError('Failed to delete topic')
        } finally {
            setDeleteLoading(false)
        }
    }

    const handleToggleArchived = async (
        e: React.MouseEvent,
        topic: ResearchTopic,
        isArchived: boolean,
    ) => {
        e.stopPropagation()
        setArchivingTopicIds((current) => new Set(current).add(topic.id))
        try {
            const updated = await researchTopicsService.updateResearchTopic(topic.id, { is_archived: isArchived })
            setTopics((prev) => prev.map((item) => (item.id === topic.id ? updated : item)))

            if (!isArchived) {
                setExpandedArchivedIds((current) => {
                    const next = new Set(current)
                    next.delete(topic.id)
                    return next
                })
            }
        } catch (err) {
            console.error('Failed to update archive status:', err)
            setError(`Failed to ${isArchived ? 'archive' : 'restore'} topic`)
        } finally {
            setArchivingTopicIds((current) => {
                const next = new Set(current)
                next.delete(topic.id)
                return next
            })
        }
    }

    const toggleArchivedExpansion = (e: React.MouseEvent, topicId: string) => {
        e.stopPropagation()
        setExpandedArchivedIds((current) => {
            const next = new Set(current)
            if (next.has(topicId)) {
                next.delete(topicId)
            } else {
                next.add(topicId)
            }
            return next
        })
    }

    const handleSetTopicRating = async (e: React.MouseEvent, topic: ResearchTopic, rating: number) => {
        e.stopPropagation()
        const nextRating = Math.max(0, Math.min(5, rating))
        setRatingTopicIds((current) => new Set(current).add(topic.id))
        try {
            const updated = await researchTopicsService.updateResearchTopic(topic.id, { topic_rating: nextRating })
            setTopics((prev) => prev.map((item) => (item.id === topic.id ? updated : item)))
        } catch (err) {
            console.error('Failed to update topic rating:', err)
            setError('Failed to update topic rating')
        } finally {
            setRatingTopicIds((current) => {
                const next = new Set(current)
                next.delete(topic.id)
                return next
            })
        }
    }

    const formatDate = (dateString: string) => {
        const date = new Date(dateString)
        const now = new Date()
        const diffInDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24))

        if (diffInDays === 0) return 'Today'
        if (diffInDays === 1) return 'Yesterday'
        if (diffInDays < 7) return `${diffInDays} days ago`
        return date.toLocaleDateString()
    }

    if (authLoading) {
        return (
            <div className="flex items-center justify-center min-h-screen bg-background">
                <Skeleton className="h-12 w-12 rounded-full" />
            </div>
        )
    }

    const filteredTopics = topics.filter((topic) => {
        if (!searchTerm.trim()) return true
        const haystack = [
            topic.title,
            topic.project_name,
            topic.primary_category_name,
            topic.secondary_category_name,
        ]
            .filter(Boolean)
            .join(" ")
            .toLowerCase()
        return haystack.includes(searchTerm.trim().toLowerCase())
    })
    const activeTopics = filteredTopics.filter((topic) => !topic.is_archived)
    const archivedTopics = filteredTopics.filter((topic) => Boolean(topic.is_archived))

    const primaryCategories = React.useMemo(
        () => projectCategories.filter((category) => category.level === 1),
        [projectCategories]
    )
    const secondaryCategories = React.useMemo(
        () => projectCategories.filter(
            (category) => category.level === 2 && (!primaryCategoryFilter || category.parent_category_id === primaryCategoryFilter)
        ),
        [projectCategories, primaryCategoryFilter]
    )

    const filterQuery = React.useMemo(() => {
        const params = new URLSearchParams()
        if (projectFilter) params.set('project_id', projectFilter)
        if (primaryCategoryFilter) params.set('primary_category_id', primaryCategoryFilter)
        if (secondaryCategoryFilter) params.set('secondary_category_id', secondaryCategoryFilter)
        const q = params.toString()
        return q ? `?${q}` : ''
    }, [projectFilter, primaryCategoryFilter, secondaryCategoryFilter])

    const buildRebuildUrl = React.useCallback((topic?: ResearchTopic, workflowRunId?: string) => {
        const params = new URLSearchParams()
        const scopedProjectId = topic?.project_id || projectFilter || activeProject?.id || ''
        const scopedPrimaryCategoryId = topic?.primary_category_id || primaryCategoryFilter || ''
        const scopedSecondaryCategoryId = topic?.secondary_category_id || secondaryCategoryFilter || ''

        if (scopedProjectId) params.set('project_id', scopedProjectId)
        if (scopedPrimaryCategoryId) params.set('primary_category_id', scopedPrimaryCategoryId)
        if (scopedSecondaryCategoryId) params.set('secondary_category_id', scopedSecondaryCategoryId)
        if (workflowRunId) params.set('workflow_run_id', workflowRunId)

        const query = params.toString()
        return `/research-rebuild${query ? `?${query}` : ''}`
    }, [activeProject?.id, primaryCategoryFilter, projectFilter, secondaryCategoryFilter])

    const getTopicRebuildSummary = React.useCallback((topic: ResearchTopic): RebuildSummary | null => {
        const topicProjectId = String(topic.project_id || '')
        const topicPrimaryCategoryId = topic.primary_category_id || ''
        const topicSecondaryCategoryId = topic.secondary_category_id || ''

        if (!topicProjectId || !topicPrimaryCategoryId) {
            return null
        }

        const primaryOnlySummary =
            rebuildTopicSummaries.find(
                (summary) =>
                    summary.project_id === topicProjectId &&
                    summary.primary_category_id === topicPrimaryCategoryId &&
                    !summary.secondary_category_id,
            ) || null
        const secondarySummary = topicSecondaryCategoryId
            ? rebuildTopicSummaries.find(
                (summary) =>
                    summary.project_id === topicProjectId &&
                    summary.primary_category_id === topicPrimaryCategoryId &&
                    summary.secondary_category_id === topicSecondaryCategoryId,
            ) || null
            : null

        const matchingSummaries = [primaryOnlySummary, secondarySummary].filter(Boolean) as ResearchRebuildTopicScopeSummary[]
        if (matchingSummaries.length === 0) {
            return null
        }

        const routeCounts = matchingSummaries.reduce<Record<string, number>>((accumulator, summary) => {
            for (const [route, count] of Object.entries(summary.route_counts || {})) {
                accumulator[route] = (accumulator[route] || 0) + Number(count || 0)
            }
            return accumulator
        }, {})

        const dominantRoute = Object.entries(routeCounts)
            .reduce<{ route: string | null; count: number }>(
                (best, [route, count]) => (count > best.count ? { route, count } : best),
                { route: null, count: 0 }
            ).route

        const latestRun = matchingSummaries
            .map((summary) => summary.latest_run)
            .filter(Boolean)
            .sort((left, right) => String(right?.started_at || '').localeCompare(String(left?.started_at || '')))[0] || null

        return {
            runCount: matchingSummaries.reduce((sum, summary) => sum + Number(summary.run_count || 0), 0),
            latestRun,
            dominantRoute,
        }
    }, [rebuildTopicSummaries])

    return (
        <div className="min-h-screen bg-background relative overflow-hidden">
            {/* Radial gradient background */}
            <div className="absolute inset-0 bg-gradient-to-b from-primary/10 to-transparent pointer-events-none" />

            <div className="relative z-10 max-w-7xl mx-auto px-6 py-10">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                    className="mb-10"
                >
                    <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
                        <div>
                            <h1 className="text-3xl md:text-4xl font-semibold tracking-tight text-foreground">
                                All Research Topics
                            </h1>
                            <p className="mt-2 text-muted-foreground">
                                Browse research topics across projects, with the category context attached.
                            </p>
                        </div>
                        <Button
                            variant="outline"
                            className="border-border hover:bg-muted"
                            onClick={() => navigate('/')}
                        >
                            New Research
                            <ArrowUpRight className="ml-2 h-4 w-4" />
                        </Button>
                    </div>

                    <div className="mt-5 rounded-2xl border border-border bg-muted/30 p-4">
                        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                            <div>
                                <div className="flex items-center gap-2 text-foreground">
                                    <Sparkles className="h-4 w-4 text-amber-400" />
                                    <span className="text-sm font-semibold">Research Rebuild</span>
                                </div>
                                <p className="mt-1 text-sm text-muted-foreground">
                                    Try the job-first workflow for article and software opportunity discovery with validation, routing, and run history.
                                </p>
                            </div>
                            <Button
                                variant="secondary"
                                className="shrink-0"
                                onClick={() => navigate(`/research-rebuild${filterQuery}`)}
                            >
                                Open Rebuild
                                <ArrowUpRight className="ml-2 h-4 w-4" />
                            </Button>
                        </div>
                    </div>

                    <div className="mt-6 grid gap-3 md:grid-cols-4">
                        <select
                            className="h-12 rounded-2xl border border-border bg-muted/50 px-4 text-sm text-foreground outline-none focus:border-ring/50"
                            value={projectFilter}
                            onChange={(e) => {
                                setProjectFilter(e.target.value)
                            }}
                        >
                            <option value="">All projects</option>
                            {projects.map((p) => (
                                <option key={p.id} value={p.id}>
                                    {p.domain || p.app_name || 'Untitled Project'}
                                </option>
                            ))}
                        </select>

                        <select
                            className="h-12 rounded-2xl border border-border bg-muted/50 px-4 text-sm text-foreground outline-none focus:border-ring/50 disabled:opacity-50"
                            value={primaryCategoryFilter}
                            disabled={!projectFilter}
                            onChange={(e) => {
                                setPrimaryCategoryFilter(e.target.value)
                                setSecondaryCategoryFilter("")
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
                            className="h-12 rounded-2xl border border-border bg-muted/50 px-4 text-sm text-foreground outline-none focus:border-ring/50 disabled:opacity-50"
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

                        <div className="relative">
                            <Search className="absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                            <Input
                                placeholder="Search titles, projects, categories..."
                                className="h-12 pl-11 bg-muted/50 border-border text-foreground placeholder:text-muted-foreground rounded-2xl focus-visible:ring-0 focus-visible:ring-offset-0 focus-visible:border-ring/50"
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                            />
                        </div>
                    </div>
                </motion.div>

                {/* Error Display */}
                {error && (
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="mb-8 p-4 rounded-xl bg-destructive/10 border border-destructive/20 text-destructive text-center"
                    >
                        {error}
                    </motion.div>
                )}

                <div className="space-y-6">
                    <h2 className="text-xl font-semibold text-foreground tracking-tight">Recent Topics</h2>

                    {loading ? (
                        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                            {[1, 2, 3].map((i) => (
                                <div key={i} className="bg-muted/30 backdrop-blur-md border border-border rounded-2xl p-8 h-48">
                                    <Skeleton className="h-6 w-3/4 mb-4" />
                                    <Skeleton className="h-4 w-1/2 mb-6" />
                                    <Skeleton className="h-4 w-full" />
                                </div>
                            ))}
                        </div>
                    ) : filteredTopics.length === 0 ? (
                        <motion.div
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ duration: 0.4 }}
                            className="bg-muted/30 backdrop-blur-md border border-border border-dashed rounded-2xl p-12 text-center"
                        >
                            <div className="h-16 w-16 rounded-3xl border border-border bg-muted/50 mx-auto mb-4" />
                            <h3 className="text-xl font-semibold text-foreground mb-2">No topics yet</h3>
                            <p className="text-muted-foreground">
                                Create a new research queue from the command center.
                            </p>
                        </motion.div>
                    ) : (
                        <>
                            {activeTopics.length > 0 ? (
                                <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                                    {activeTopics.map((topic, index) => (
                                        (() => {
                                            const rebuildSummary = getTopicRebuildSummary(topic)

                                            return (
                                        <motion.div
                                            key={topic.id}
                                            initial={{ opacity: 0, y: 20 }}
                                            animate={{ opacity: 1, y: 0 }}
                                            transition={{ duration: 0.4, delay: index * 0.05 }}
                                            onClick={() => navigate(`/research/${topic.id}${filterQuery}`)}
                                            className="group relative bg-muted/30 backdrop-blur-md border border-border rounded-2xl p-8 cursor-pointer hover:-translate-y-1 hover:border-ring/50 transition-all duration-300"
                                        >
                                            {/* Header with Progress Chips */}
                                            <div className="mb-4">
                                                <h3
                                                    className="text-lg font-bold text-foreground line-clamp-2 pr-36"
                                                    title={topic.title || ''}
                                                >
                                                    {topic.title}
                                                </h3>
                                                <div className="absolute right-6 top-6 flex flex-col items-end gap-2">
                                                    {(() => {
                                                        const subtopicsCount = Number(topic.subtopics_count || 0)
                                                        const researchedSubtopicsCount = Number(topic.researched_subtopics_count || 0)
                                                        const keywordCount = Number(topic.topic_keyword_candidate_count || 0)
                                                        const clusterCount = Number(topic.topic_keyword_cluster_count || 0)
                                                        const keywordResearchStatus = String(topic.topic_keyword_research_status || '').trim().toLowerCase()
                                                        const ideasCount = Number(topic.content_ideas_count || 0)
                                                        const inLibraryCount = Number(topic.in_library_count || 0)
                                                        const topicMode = String(topic.topic_mode || 'hybrid')
                                                        const viabilityLabel = String(topic.keyword_viability_label || 'medium')
                                                        const shouldPreferEditorialRoute =
                                                            (topicMode === 'editorial_first' || topicMode === 'hybrid') &&
                                                            keywordResearchStatus === 'completed' &&
                                                            keywordCount === 0 &&
                                                            clusterCount === 0
                                                        const isFullyResearched = Boolean(topic.all_subtopics_researched)
                                                        const hasKeywordResearch = keywordCount > 0 || clusterCount > 0 || ['pending', 'running', 'completed', 'failed'].includes(keywordResearchStatus)
                                                        const hasAnyProgress =
                                                            subtopicsCount > 0 ||
                                                            hasKeywordResearch ||
                                                            ideasCount > 0 ||
                                                            inLibraryCount > 0 ||
                                                            Boolean(topic.has_underlying_data)
                                                        const isNotStarted = !hasAnyProgress && !isFullyResearched
                                                        const chipBase =
                                                            "inline-flex flex-shrink-0 items-center whitespace-nowrap rounded-full border px-2 py-1 text-xs font-medium leading-none"

                                                        return (
                                                            <div className="flex flex-col items-end gap-1">
                                                                <span className={`${chipBase} border-blue-500/20 bg-blue-500/10 text-blue-300`}>
                                                                    {topicMode === 'keyword_first' ? 'Keyword First' : topicMode === 'editorial_first' ? 'Editorial First' : 'Hybrid'}
                                                                </span>
                                                                <span className={`${chipBase} border-slate-500/20 bg-slate-500/10 text-slate-300`}>
                                                                    {viabilityLabel.charAt(0).toUpperCase() + viabilityLabel.slice(1)} Keyword Potential
                                                                </span>
                                                                {subtopicsCount > 0 && (
                                                                    <span className={`${chipBase} border-indigo-500/20 bg-indigo-500/10 text-indigo-300`}>
                                                                        {subtopicsCount} Sub-Topics
                                                                    </span>
                                                                )}
                                                                {keywordCount > 0 && (
                                                                    <span className={`${chipBase} border-violet-500/20 bg-violet-500/10 text-violet-300`}>
                                                                        {keywordCount} Keywords
                                                                    </span>
                                                                )}
                                                                {clusterCount > 0 && (
                                                                    <span className={`${chipBase} border-fuchsia-500/20 bg-fuchsia-500/10 text-fuchsia-300`}>
                                                                        {clusterCount} Clusters
                                                                    </span>
                                                                )}
                                                                {isFullyResearched && (
                                                                    <span className={`${chipBase} border-emerald-500/30 bg-emerald-500/10 text-emerald-300`}>
                                                                        Researched
                                                                    </span>
                                                                )}
                                                                {!isFullyResearched && keywordResearchStatus === 'running' && (
                                                                    <span className={`${chipBase} border-amber-500/20 bg-amber-500/10 text-amber-300`}>
                                                                        Keyword Research Running
                                                                    </span>
                                                                )}
                                                                {!isFullyResearched && keywordResearchStatus === 'completed' && hasKeywordResearch && ideasCount === 0 && (
                                                                    <span className={`${chipBase} border-cyan-500/20 bg-cyan-500/10 text-cyan-300`}>
                                                                        Keywords Ready
                                                                    </span>
                                                                )}
                                                                {shouldPreferEditorialRoute && (
                                                                    <span className={`${chipBase} border-amber-500/20 bg-amber-500/10 text-amber-300`}>
                                                                        Editorial Route
                                                                    </span>
                                                                )}
                                                                {isNotStarted && (
                                                                    <span className={`${chipBase} border-border bg-muted/60 text-muted-foreground`}>
                                                                        Not Started
                                                                    </span>
                                                                )}
                                                                {researchedSubtopicsCount > 0 && !isFullyResearched && (
                                                                    <span className={`${chipBase} border-emerald-500/20 bg-emerald-500/10 text-emerald-300`}>
                                                                        {researchedSubtopicsCount}/{subtopicsCount} Researched
                                                                    </span>
                                                                )}
                                                                {ideasCount > 0 && (
                                                                    <span className={`${chipBase} border-sky-500/20 bg-sky-500/10 text-sky-300`}>
                                                                        {ideasCount} Ideas
                                                                    </span>
                                                                )}
                                                                {inLibraryCount > 0 && (
                                                                    <span className={`${chipBase} border-emerald-500/30 bg-emerald-500/10 text-emerald-300`}>
                                                                        {inLibraryCount} In Library
                                                                    </span>
                                                                )}
                                                            </div>
                                                        )
                                                    })()}

                                                    <div className="flex items-center gap-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
                                                        <Button
                                                            variant="ghost"
                                                            size="icon"
                                                            className="h-8 w-8 text-muted-foreground hover:text-foreground"
                                                            onClick={(e) => {
                                                                e.stopPropagation()
                                                                navigate(buildRebuildUrl(topic))
                                                            }}
                                                            aria-label={`Open ${topic.title} in Research Rebuild`}
                                                        >
                                                            <Sparkles className="h-4 w-4" />
                                                        </Button>
                                                        <Button
                                                            variant="ghost"
                                                            size="icon"
                                                            className="h-8 w-8 text-muted-foreground hover:text-foreground"
                                                            onClick={(e) => handleToggleArchived(e, topic, true)}
                                                            disabled={archivingTopicIds.has(topic.id)}
                                                            aria-label={`Archive ${topic.title}`}
                                                        >
                                                            {archivingTopicIds.has(topic.id) ? <Loader2 className="h-4 w-4 animate-spin" /> : <Archive className="h-4 w-4" />}
                                                        </Button>
                                                        <Button
                                                            variant="ghost"
                                                            size="icon"
                                                            className="h-8 w-8 text-muted-foreground hover:text-destructive"
                                                            onClick={(e) => handleDeleteTopic(e, topic)}
                                                            aria-label={`Delete ${topic.title}`}
                                                        >
                                                            <Trash2 className="h-4 w-4" />
                                                        </Button>
                                                    </div>
                                                </div>
                                            </div>

                                            {/* Timestamp */}
                                            <div className="flex items-center text-xs text-muted-foreground mb-4">
                                                <Clock className="h-3 w-3 mr-1.5 opacity-70" />
                                                {formatDate(topic.created_at)}
                                            </div>

                                            {/* Description */}
                                            <p
                                                className="text-sm text-muted-foreground line-clamp-2 leading-relaxed"
                                                title={topic.description || ''}
                                            >
                                                {topic.description}
                                            </p>

                                            <div className="mt-4 flex items-center gap-1">
                                                {[1, 2, 3, 4, 5].map((value) => {
                                                    const isFilled = value <= Number(topic.topic_rating || 0)
                                                    return (
                                                        <button
                                                            key={`${topic.id}-rating-${value}`}
                                                            type="button"
                                                            onClick={(e) => handleSetTopicRating(e, topic, value)}
                                                            disabled={ratingTopicIds.has(topic.id)}
                                                            className="rounded p-0.5 transition hover:scale-105 disabled:cursor-not-allowed disabled:opacity-60"
                                                            aria-label={`Rate ${topic.title} ${value} star${value === 1 ? '' : 's'}`}
                                                        >
                                                            <Star
                                                                className={`h-4 w-4 ${isFilled ? 'fill-amber-400 text-amber-400' : 'text-muted-foreground'}`}
                                                            />
                                                        </button>
                                                    )
                                                })}
                                                {ratingTopicIds.has(topic.id) && <Loader2 className="ml-1 h-3.5 w-3.5 animate-spin text-muted-foreground" />}
                                            </div>

                                                    {(topic.project_name || topic.primary_category_name || topic.secondary_category_name) && (
                                                <div className="mt-4 flex flex-wrap gap-2">
                                                    {topic.project_name && (
                                                        <span className="rounded-full border border-border bg-muted/50 px-3 py-1 text-[11px] text-foreground">
                                                            {topic.project_name}
                                                        </span>
                                                    )}
                                                    {(topic.primary_category_name || topic.secondary_category_name) && (
                                                        <span className="rounded-full border border-primary/20 bg-primary/10 px-3 py-1 text-[11px] text-primary">
                                                            {[topic.primary_category_name, topic.secondary_category_name].filter(Boolean).join(' / ')}
                                                        </span>
                                                    )}
                                                    {rebuildSummary && (
                                                        <RebuildStatusBadge
                                                            summary={rebuildSummary}
                                                            onClick={(e) => {
                                                                e.stopPropagation()
                                                                navigate(buildRebuildUrl(topic, rebuildSummary.latestRun?.workflow_run_id))
                                                            }}
                                                        />
                                                    )}
                                                </div>
                                            )}
                                        </motion.div>
                                            )
                                        })()
                                    ))}
                                </div>
                            ) : (
                                <div className="rounded-2xl border border-dashed border-border bg-muted/20 p-6 text-sm text-muted-foreground">
                                    No active topics match the current filters.
                                </div>
                            )}

                            {archivedTopics.length > 0 && (
                                <div className="pt-2">
                                    <div className="mb-4 flex items-center justify-between">
                                        <h3 className="text-lg font-semibold tracking-tight text-foreground">Archived Topics</h3>
                                        <span className="rounded-full border border-border bg-muted/40 px-3 py-1 text-xs text-muted-foreground">
                                            {archivedTopics.length}
                                        </span>
                                    </div>

                                    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                                        {archivedTopics.map((topic, index) => {
                                            const expanded = expandedArchivedIds.has(topic.id)
                                            const rebuildSummary = getTopicRebuildSummary(topic)
                                            return (
                                                <motion.div
                                                    key={topic.id}
                                                    initial={{ opacity: 0, y: 20 }}
                                                    animate={{ opacity: 1, y: 0 }}
                                                    transition={{ duration: 0.35, delay: index * 0.04 }}
                                                    className="relative rounded-2xl border border-border/80 bg-muted/20 p-6 opacity-70 saturate-0 transition-all"
                                                >
                                                    <div className="flex items-start justify-between gap-3">
                                                        <h4
                                                            className="text-base font-semibold text-foreground line-clamp-2"
                                                            title={topic.title || ''}
                                                        >
                                                            {topic.title}
                                                        </h4>
                                                        <div className="flex items-center gap-1">
                                                            <Button
                                                                variant="ghost"
                                                                size="icon"
                                                                className="h-8 w-8 text-muted-foreground hover:text-foreground"
                                                                onClick={(e) => handleToggleArchived(e, topic, false)}
                                                                disabled={archivingTopicIds.has(topic.id)}
                                                                aria-label={`Restore ${topic.title}`}
                                                            >
                                                                {archivingTopicIds.has(topic.id) ? <Loader2 className="h-4 w-4 animate-spin" /> : <RotateCcw className="h-4 w-4" />}
                                                            </Button>
                                                            <Button
                                                                variant="ghost"
                                                                size="icon"
                                                                className="h-8 w-8 text-muted-foreground hover:text-destructive"
                                                                onClick={(e) => handleDeleteTopic(e, topic)}
                                                                aria-label={`Delete ${topic.title}`}
                                                            >
                                                                <Trash2 className="h-4 w-4" />
                                                            </Button>
                                                        </div>
                                                    </div>

                                                    {expanded && (
                                                        <>
                                                            <div className="mt-4 flex items-center text-xs text-muted-foreground">
                                                                <Clock className="mr-1.5 h-3 w-3 opacity-70" />
                                                                {formatDate(topic.created_at)}
                                                            </div>

                                                            <p
                                                                className="mt-3 text-sm text-muted-foreground line-clamp-3 leading-relaxed"
                                                                title={topic.description || ''}
                                                            >
                                                                {topic.description || "No description provided."}
                                                            </p>

                                                            {rebuildSummary && (
                                                                <div className="mt-3 flex flex-wrap gap-2">
                                                                    <RebuildStatusBadge
                                                                        summary={rebuildSummary}
                                                                        muted
                                                                        onClick={(e) => {
                                                                            e.stopPropagation()
                                                                            navigate(buildRebuildUrl(topic, rebuildSummary.latestRun?.workflow_run_id))
                                                                        }}
                                                                    />
                                                                </div>
                                                            )}

                                                            <div className="mt-3 flex items-center gap-1">
                                                                {[1, 2, 3, 4, 5].map((value) => {
                                                                    const isFilled = value <= Number(topic.topic_rating || 0)
                                                                    return (
                                                                        <button
                                                                            key={`${topic.id}-archived-rating-${value}`}
                                                                            type="button"
                                                                            onClick={(e) => handleSetTopicRating(e, topic, value)}
                                                                            disabled={ratingTopicIds.has(topic.id)}
                                                                            className="rounded p-0.5 transition hover:scale-105 disabled:cursor-not-allowed disabled:opacity-60"
                                                                            aria-label={`Rate ${topic.title} ${value} star${value === 1 ? '' : 's'}`}
                                                                        >
                                                                            <Star
                                                                                className={`h-4 w-4 ${isFilled ? 'fill-amber-400 text-amber-400' : 'text-muted-foreground'}`}
                                                                            />
                                                                        </button>
                                                                    )
                                                                })}
                                                                {ratingTopicIds.has(topic.id) && <Loader2 className="ml-1 h-3.5 w-3.5 animate-spin text-muted-foreground" />}
                                                            </div>
                                                        </>
                                                    )}

                                                    <div className="mt-4 border-t border-border/80 pt-3">
                                                        <div className="flex items-center justify-between gap-3">
                                                            <button
                                                                type="button"
                                                                onClick={(e) => toggleArchivedExpansion(e, topic.id)}
                                                                className="inline-flex items-center gap-1.5 text-xs text-muted-foreground transition hover:text-foreground"
                                                            >
                                                                {expanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                                                                {expanded ? "Collapse" : "Expand"}
                                                            </button>
                                                            <Button
                                                                variant="ghost"
                                                                size="sm"
                                                                className="h-7 px-2 text-xs text-muted-foreground hover:text-foreground"
                                                                onClick={(e) => {
                                                                    e.stopPropagation()
                                                                    navigate(buildRebuildUrl(topic))
                                                                }}
                                                            >
                                                                <Sparkles className="mr-1.5 h-3.5 w-3.5" />
                                                                Rebuild
                                                            </Button>
                                                        </div>
                                                    </div>
                                                </motion.div>
                                            )
                                        })}
                                    </div>
                                </div>
                            )}

                            {/* Load More Button */}
                            {hasMore && (
                                <div className="flex justify-center mt-8">
                                    <Button
                                        variant="outline"
                                        onClick={handleLoadMore}
                                        disabled={loading}
                                        className="px-8 py-3 bg-muted/50 border-border text-foreground hover:bg-muted hover:border-ring/50"
                                    >
                                        {loading ? (
                                            <>
                                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                                Loading...
                                            </>
                                        ) : (
                                            <>Load More Topics</>
                                        )}
                                    </Button>
                                </div>
                            )}
                        </>
                    )}
                </div>
            </div>

            {topicPendingDelete && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 px-4 backdrop-blur-sm">
                    <div className="w-full max-w-md rounded-2xl border border-border bg-background p-6">
                        <h3 className="text-base font-semibold text-foreground">Delete Research Topic</h3>
                        <p className="mt-3 text-sm text-muted-foreground">
                            Warning: This action is permanent. All associated subtopics and content ideas will also be deleted.
                        </p>
                        <p className="mt-3 text-sm text-foreground">
                            <span className="font-medium">Topic:</span> {topicPendingDelete.title}
                        </p>
                        <div className="mt-6 flex justify-end gap-2">
                            <button
                                type="button"
                                onClick={() => setTopicPendingDelete(null)}
                                disabled={deleteLoading}
                                className="h-10 rounded-xl border border-border px-4 text-sm text-muted-foreground transition hover:bg-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
                            >
                                Cancel
                            </button>
                            <button
                                type="button"
                                onClick={handleConfirmDelete}
                                disabled={deleteLoading}
                                className="inline-flex h-10 items-center gap-1.5 rounded-xl bg-destructive px-4 text-sm font-medium text-destructive-foreground transition hover:bg-destructive/90 disabled:cursor-not-allowed disabled:opacity-60"
                            >
                                {deleteLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                                Delete
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

function RebuildStatusBadge({
    summary,
    muted = false,
    onClick,
}: {
    summary: RebuildSummary
    muted?: boolean
    onClick?: React.MouseEventHandler<HTMLButtonElement>
}) {
    const routeLabel = formatRouteLabel(summary.dominantRoute)
    const toneClasses = muted
        ? "border-amber-500/15 bg-amber-500/5 text-amber-200"
        : "border-amber-500/20 bg-amber-500/10 text-amber-300"

    return (
        <button
            type="button"
            onClick={onClick}
            className={`rounded-full border px-3 py-1 text-[11px] transition hover:border-amber-400/40 hover:bg-amber-500/15 ${toneClasses}`}
        >
            Rebuild: {summary.runCount} run{summary.runCount === 1 ? '' : 's'}
            {routeLabel ? ` · ${routeLabel}` : ''}
        </button>
    )
}

function formatRouteLabel(route: string | null | undefined) {
    if (!route) return ''
    return route
        .split('_')
        .filter(Boolean)
        .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1))
        .join(' ')
}
