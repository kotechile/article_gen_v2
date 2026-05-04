
import * as React from "react"
import { useParams, useNavigate, useLocation } from "react-router-dom"
import { useAuth } from "@/context/auth-context"
import { researchTopicsService } from "@/services/research-topics.service"
import { topicKeywordResearchService } from "@/services/topic-keyword-research.service"
import { contentIdeasService } from "@/services/content-ideas.service"
import type { ContentIdea } from "@/types/idea-burst"
import type {
    ResearchTopic,
    TopicKeywordCandidate,
    TopicKeywordCluster,
    TopicKeywordResearchRun,
} from "@/types/research"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { ArrowLeft, ListTree, LibraryBig, Search, Sparkles } from "lucide-react"
import { motion } from "framer-motion"
import { GeneratedIdeasPanel } from "@/components/GeneratedIdeasPanel"
import { TopicKeywordResearchPanel } from "@/components/TopicKeywordResearchPanel"
import { toast } from "sonner"

export function TopicDetail() {
    const { id } = useParams<{ id: string }>()
    const navigate = useNavigate()
    const location = useLocation()
    const { user, isLoading: authLoading } = useAuth()
    const backToResearchUrl = React.useMemo(
        () => `/research${location.search || ''}`,
        [location.search]
    )

    // State
    const [topic, setTopic] = React.useState<ResearchTopic | null>(null)
    const [loading, setLoading] = React.useState(true)
    const [error, setError] = React.useState<string | null>(null)
    const [hasStoredIdeas, setHasStoredIdeas] = React.useState(false)
    const [storedIdeas, setStoredIdeas] = React.useState<ContentIdea[]>([])
    const [keywordResearchRun, setKeywordResearchRun] = React.useState<TopicKeywordResearchRun | null>(null)
    const [keywordCandidates, setKeywordCandidates] = React.useState<TopicKeywordCandidate[]>([])
    const [keywordClusters, setKeywordClusters] = React.useState<TopicKeywordCluster[]>([])
    const [keywordResearchLoading, setKeywordResearchLoading] = React.useState(false)
    const [runningKeywordResearch, setRunningKeywordResearch] = React.useState(false)
    const [keywordResearchError, setKeywordResearchError] = React.useState<string | null>(null)
    const [selectedClusterIds, setSelectedClusterIds] = React.useState<Set<string>>(new Set())
    const [generatingClusterIdeas, setGeneratingClusterIdeas] = React.useState(false)
    const [selectedGeneratedIdeaIds, setSelectedGeneratedIdeaIds] = React.useState<Set<string>>(new Set())
    const [publishingGeneratedIdeas, setPublishingGeneratedIdeas] = React.useState(false)
    const [activeGeneratedIdea, setActiveGeneratedIdea] = React.useState<ContentIdea | null>(null)
    const [generatedIdeaTypeFilter, setGeneratedIdeaTypeFilter] = React.useState<'all' | 'blog' | 'software'>('all')
    const [generatedIdeaStatusFilter, setGeneratedIdeaStatusFilter] = React.useState<'all' | 'draft' | 'published'>('draft')
    const [generatedIdeaSort, setGeneratedIdeaSort] = React.useState<'score' | 'volume' | 'difficulty' | 'recent'>('score')

    const mergeContentIdeas = React.useCallback((baseIdeas: ContentIdea[], incomingIdeas: ContentIdea[]) => {
        const merged = new Map<string, ContentIdea>()

        const buildKey = (idea: ContentIdea) => {
            if (idea.id) {
                return `id:${idea.id}`
            }
            return `fallback:${idea.content_type || 'unknown'}:${(idea.title || '').trim().toLowerCase()}:${(idea.subtopic || '').trim().toLowerCase()}`
        }

        for (const idea of baseIdeas || []) {
            merged.set(buildKey(idea), idea)
        }
        for (const idea of incomingIdeas || []) {
            merged.set(buildKey(idea), idea)
        }

        return Array.from(merged.values()).sort((a, b) => {
            return new Date(b.created_at || 0).getTime() - new Date(a.created_at || 0).getTime()
        })
    }, [])

    const loadKeywordResearch = React.useCallback(async (topicId: string) => {
        setKeywordResearchLoading(true)
        try {
            const latestRun = await topicKeywordResearchService.getLatestRun(topicId)
            setKeywordResearchRun(latestRun)

            const [keywordsResponse, clustersResponse] = await Promise.all([
                topicKeywordResearchService.listKeywords(topicId, latestRun.id, false),
                topicKeywordResearchService.listClusters(topicId, latestRun.id),
            ])

            setKeywordCandidates(keywordsResponse.items || [])
            setKeywordClusters(clustersResponse.items || [])
            setSelectedClusterIds(new Set((clustersResponse.items || []).slice(0, 1).map((cluster) => cluster.id)))
            setKeywordResearchError(null)
        } catch (err) {
            const status = (err as any)?.response?.status ?? (err as any)?.status
            if (status === 404) {
                setKeywordResearchRun(null)
                setKeywordCandidates([])
                setKeywordClusters([])
                setSelectedClusterIds(new Set())
                setKeywordResearchError(null)
            } else {
                console.error('Failed to load topic keyword research:', err)
                setKeywordResearchError('Keyword research is not available yet. Apply the new SQL migration, then run the topic-level research flow.')
            }
        } finally {
            setKeywordResearchLoading(false)
        }
    }, [])

    const refreshStoredIdeasState = React.useCallback(async (topicId: string) => {
        if (!user?.id) {
            setHasStoredIdeas(false)
            return [] as ContentIdea[]
        }

        const storedIdeas = await contentIdeasService.getContentIdeas(topicId, user.id)
        console.info('[TopicDetail] refreshed stored ideas', {
            topicId,
            total: storedIdeas.length,
            blog: storedIdeas.filter((idea) => idea.content_type === 'blog').length,
            software: storedIdeas.filter((idea) => idea.content_type === 'software').length,
        })

        setStoredIdeas(storedIdeas || [])
        setHasStoredIdeas(Array.isArray(storedIdeas) && storedIdeas.length > 0)
        return storedIdeas || []
    }, [user?.id])

    React.useEffect(() => {
        if (!authLoading && user && id) {
            loadData(id)
        }
    }, [authLoading, user, id])

    const loadData = async (topicId: string) => {
        try {
            setLoading(true)
            const topicData = await researchTopicsService.getResearchTopic(topicId)
            setTopic(topicData)
            await refreshStoredIdeasState(topicId)
            await loadKeywordResearch(topicId)
            setError(null)
        } catch (err) {
            console.error('Failed to load topic data:', err)
            setError('Failed to load topic details')
        } finally {
            setLoading(false)
        }
    }

    const handleRunKeywordResearch = async () => {
        if (!id) return
        try {
            setRunningKeywordResearch(true)
            setKeywordResearchError(null)
            toast.loading('Running topic keyword research. This can take a couple of minutes while keyword data is collected.', {
                id: 'topic-keyword-research',
            })

            const result = await topicKeywordResearchService.runTopicKeywordResearch(id, {
                replace_existing: true,
            })

            toast.success(
                `Keyword research completed with ${result.keyword_count} keywords and ${result.cluster_count} clusters.`,
                { id: 'topic-keyword-research' }
            )
            await loadKeywordResearch(id)
        } catch (err) {
            console.error('Failed to run topic keyword research:', err)
            setKeywordResearchError('Failed to run topic keyword research.')
            toast.error('Topic keyword research failed. If the migration has not been applied yet, apply it in Supabase and try again.', {
                id: 'topic-keyword-research',
            })
        } finally {
            setRunningKeywordResearch(false)
        }
    }

    const toggleClusterSelection = React.useCallback((clusterId: string) => {
        setSelectedClusterIds((current) => {
            const next = new Set(current)
            if (next.has(clusterId)) {
                next.delete(clusterId)
            } else {
                next.add(clusterId)
            }
            return next
        })
    }, [])

    const handleGenerateIdeasFromClusters = React.useCallback(async () => {
        if (!id || !user?.id || !keywordResearchRun) return
        const clusterIds = Array.from(selectedClusterIds)
        if (clusterIds.length === 0) {
            toast.error('Select at least one keyword cluster first.')
            return
        }

        try {
            setGeneratingClusterIdeas(true)
            toast.loading('Generating content ideas from the selected keyword clusters.', {
                id: 'topic-cluster-ideas',
            })

            const result = await topicKeywordResearchService.generateIdeasFromClusters(id, keywordResearchRun.id, {
                user_id: user.id,
                cluster_ids: clusterIds,
            })

            const responseIdeas = [
                ...((result.blog_ideas || []) as ContentIdea[]),
                ...((result.software_ideas || []) as ContentIdea[]),
            ]
            const refreshedIdeas = await refreshStoredIdeasState(id)
            const mergedIdeas = mergeContentIdeas(refreshedIdeas || [], responseIdeas)

            setStoredIdeas(mergedIdeas)
            setHasStoredIdeas(mergedIdeas.length > 0)

            toast.success(
                `Generated ${result.generated_count || 0} ideas from ${clusterIds.length} cluster${clusterIds.length === 1 ? '' : 's'}.`,
                { id: 'topic-cluster-ideas' }
            )

            if (result.persistence_warning) {
                setError(result.persistence_warning)
                toast.warning(result.persistence_warning, {
                    id: 'topic-cluster-ideas-warning',
                })
            } else {
                setError(null)
            }
        } catch (err) {
            console.error('Failed to generate ideas from keyword clusters:', err)
            toast.error('Failed to generate ideas from the selected clusters.', {
                id: 'topic-cluster-ideas',
            })
        } finally {
            setGeneratingClusterIdeas(false)
        }
    }, [id, keywordResearchRun, mergeContentIdeas, refreshStoredIdeasState, selectedClusterIds, user?.id])

    const keywordResearchSummary = React.useMemo(() => {
        const summary = (keywordResearchRun?.summary_json || {}) as Record<string, any>
        return {
            seedCount: Number(summary.seed_count || keywordResearchRun?.seed_keywords_json?.length || 0),
            candidateCount: Number(summary.active_candidate_count || keywordCandidates.length || 0),
            clusterCount: Number(summary.cluster_count || keywordClusters.length || 0),
            generatedAt: summary.generated_at || keywordResearchRun?.updated_at || null,
            topKeywords: Array.isArray(summary.top_keywords) ? summary.top_keywords : [],
            topClusters: keywordClusters.slice(0, 4),
        }
    }, [keywordResearchRun, keywordCandidates.length, keywordClusters])

    const generatedClusterIdeas = React.useMemo(() => {
        return (storedIdeas || []).filter((idea) => {
            const metadata = (idea.idea_metadata || {}) as any
            const topicKeywordResearch = metadata?.topic_keyword_research || {}
            return topicKeywordResearch?.generation_origin === 'topic_keyword_pipeline_v1'
        })
    }, [storedIdeas])

    const visibleGeneratedClusterIdeas = React.useMemo(() => {
        const filtered = generatedClusterIdeas.filter((idea) => {
            const isPublished = Boolean(
                idea.published ||
                idea.published_to_titles ||
                idea.titles_record_id ||
                idea.status?.toLowerCase() === 'published'
            )

            if (generatedIdeaStatusFilter === 'draft' && isPublished) {
                return false
            }
            if (generatedIdeaStatusFilter === 'published' && !isPublished) {
                return false
            }
            if (generatedIdeaTypeFilter === 'all') {
                return true
            }
            return idea.content_type === generatedIdeaTypeFilter
        })

        const sorted = [...filtered].sort((a, b) => {
            if (generatedIdeaSort === 'volume') {
                return Number(b.total_search_volume || 0) - Number(a.total_search_volume || 0)
            }
            if (generatedIdeaSort === 'difficulty') {
                return Number(a.average_difficulty || 999) - Number(b.average_difficulty || 999)
            }
            if (generatedIdeaSort === 'recent') {
                return new Date(b.created_at || 0).getTime() - new Date(a.created_at || 0).getTime()
            }
            return Number(b.opportunity_score || 0) - Number(a.opportunity_score || 0)
        })

        return sorted
    }, [generatedClusterIdeas, generatedIdeaSort, generatedIdeaStatusFilter, generatedIdeaTypeFilter])

    const formatDateTime = React.useCallback((value?: string | null) => {
        if (!value) return 'Not available'
        try {
            return new Date(value).toLocaleString()
        } catch {
            return value
        }
    }, [])

    const toggleGeneratedIdeaSelection = React.useCallback((ideaId: string) => {
        setSelectedGeneratedIdeaIds((current) => {
            const next = new Set(current)
            if (next.has(ideaId)) {
                next.delete(ideaId)
            } else {
                next.add(ideaId)
            }
            return next
        })
    }, [])

    const openGeneratedIdeaDetail = React.useCallback((idea: ContentIdea) => {
        setActiveGeneratedIdea(idea)
    }, [])

    const closeGeneratedIdeaDetail = React.useCallback(() => {
        setActiveGeneratedIdea(null)
    }, [])

    const syncGeneratedIdeaReviewState = React.useCallback((latestIdeas: ContentIdea[]) => {
        setSelectedGeneratedIdeaIds((current) => {
            const next = new Set<string>()
            const latestIds = new Set((latestIdeas || []).map((idea) => idea.id))
            current.forEach((ideaId) => {
                if (latestIds.has(ideaId)) {
                    next.add(ideaId)
                }
            })
            return next
        })

        setActiveGeneratedIdea((current) => {
            if (!current?.id) {
                return null
            }
            const refreshed = (latestIdeas || []).find((idea) => idea.id === current.id) || null
            if (!refreshed) {
                return null
            }
            const isPublished = Boolean(
                refreshed.published ||
                refreshed.published_to_titles ||
                refreshed.titles_record_id ||
                refreshed.status?.toLowerCase() === 'published'
            )
            if (generatedIdeaStatusFilter === 'draft' && isPublished) {
                return null
            }
            if (generatedIdeaStatusFilter === 'published' && !isPublished) {
                return null
            }
            return refreshed
        })
    }, [generatedIdeaStatusFilter])

    const activeGeneratedIdeaIndex = React.useMemo(() => {
        if (!activeGeneratedIdea?.id) {
            return -1
        }
        return visibleGeneratedClusterIdeas.findIndex((idea) => idea.id === activeGeneratedIdea.id)
    }, [activeGeneratedIdea?.id, visibleGeneratedClusterIdeas])

    const hasPreviousGeneratedIdea = activeGeneratedIdeaIndex > 0
    const hasNextGeneratedIdea = activeGeneratedIdeaIndex >= 0 && activeGeneratedIdeaIndex < visibleGeneratedClusterIdeas.length - 1

    const openPreviousGeneratedIdea = React.useCallback(() => {
        if (activeGeneratedIdeaIndex <= 0) {
            return
        }
        setActiveGeneratedIdea(visibleGeneratedClusterIdeas[activeGeneratedIdeaIndex - 1] || null)
    }, [activeGeneratedIdeaIndex, visibleGeneratedClusterIdeas])

    const openNextGeneratedIdea = React.useCallback(() => {
        if (activeGeneratedIdeaIndex < 0 || activeGeneratedIdeaIndex >= visibleGeneratedClusterIdeas.length - 1) {
            return
        }
        setActiveGeneratedIdea(visibleGeneratedClusterIdeas[activeGeneratedIdeaIndex + 1] || null)
    }, [activeGeneratedIdeaIndex, visibleGeneratedClusterIdeas])

    const handleSelectVisibleGeneratedIdeas = React.useCallback(() => {
        setSelectedGeneratedIdeaIds(new Set(visibleGeneratedClusterIdeas.map((idea) => idea.id)))
    }, [visibleGeneratedClusterIdeas])

    const handleSelectTopGeneratedIdeas = React.useCallback((count: number) => {
        setSelectedGeneratedIdeaIds(new Set(visibleGeneratedClusterIdeas.slice(0, count).map((idea) => idea.id)))
    }, [visibleGeneratedClusterIdeas])

    const handleClearGeneratedIdeaSelection = React.useCallback(() => {
        setSelectedGeneratedIdeaIds(new Set())
    }, [])

    const handlePublishSingleGeneratedIdea = React.useCallback(async (ideaId: string) => {
        if (!user?.id || !id) return

        try {
            setPublishingGeneratedIdeas(true)
            toast.loading('Publishing idea to Content Studio.', {
                id: 'publish-single-generated-idea',
            })

            const result = await contentIdeasService.publishContentIdeas([ideaId], user.id)
            const refreshedIdeas = await refreshStoredIdeasState(id)
            syncGeneratedIdeaReviewState(refreshedIdeas || [])

            if (result.success) {
                toast.success('Published idea to Titles.', {
                    id: 'publish-single-generated-idea',
                })
            } else {
                toast.error(result.message || 'Failed to publish idea.', {
                    id: 'publish-single-generated-idea',
                })
            }
        } catch (err) {
            console.error('Failed to publish generated idea:', err)
            toast.error('Failed to publish idea.', {
                id: 'publish-single-generated-idea',
            })
        } finally {
            setPublishingGeneratedIdeas(false)
        }
    }, [id, refreshStoredIdeasState, syncGeneratedIdeaReviewState, user?.id])

    const handlePublishGeneratedIdeas = React.useCallback(async () => {
        if (!user?.id || !id) return
        const ideaIds = Array.from(selectedGeneratedIdeaIds)
        if (ideaIds.length === 0) {
            toast.error('Select at least one generated idea first.')
            return
        }

        try {
            setPublishingGeneratedIdeas(true)
            toast.loading('Publishing selected ideas to Content Studio.', {
                id: 'publish-generated-ideas',
            })

            const result = await contentIdeasService.publishContentIdeas(ideaIds, user.id)
            const refreshedIdeas = await refreshStoredIdeasState(id)
            syncGeneratedIdeaReviewState(refreshedIdeas || [])

            if (result.success) {
                toast.success(
                    `Published ${result.publishedToTitlesCount} idea${result.publishedToTitlesCount === 1 ? '' : 's'} to Titles.`,
                    { id: 'publish-generated-ideas' }
                )
                setSelectedGeneratedIdeaIds(new Set())
            } else {
                toast.error(result.message || 'Failed to publish generated ideas.', {
                    id: 'publish-generated-ideas',
                })
            }
        } catch (err) {
            console.error('Failed to publish generated ideas:', err)
            toast.error('Failed to publish generated ideas.', {
                id: 'publish-generated-ideas',
            })
        } finally {
            setPublishingGeneratedIdeas(false)
        }
    }, [id, refreshStoredIdeasState, selectedGeneratedIdeaIds, syncGeneratedIdeaReviewState, user?.id])

    const summary = React.useMemo(() => {
        const generatedIdeas = (storedIdeas || []).filter((idea) => {
            const metadata = (idea.idea_metadata || {}) as Record<string, any>
            return metadata?.topic_keyword_research?.generation_origin === 'topic_keyword_pipeline_v1'
        })
        const publishedIdeas = generatedIdeas.filter((idea) =>
            Boolean(
                idea.published ||
                idea.published_to_titles ||
                idea.titles_record_id ||
                idea.status?.toLowerCase() === 'published'
            )
        )
        return {
            rankedKeywords: keywordCandidates.length,
            clusterCount: keywordClusters.length,
            totalIdeas: generatedIdeas.length,
            publishedIdeas: publishedIdeas.length,
            hasProgress: keywordCandidates.length > 0 || keywordClusters.length > 0 || generatedIdeas.length > 0,
        }
    }, [storedIdeas, keywordCandidates.length, keywordClusters.length])

    React.useEffect(() => {
        if (!id) return
        console.info('[TopicDetail] status snapshot', {
            topicId: id,
            hasStoredIdeas,
            hasProgress: summary.hasProgress,
            rankedKeywords: summary.rankedKeywords,
            clusterCount: summary.clusterCount,
            totalIdeas: summary.totalIdeas,
            publishedIdeas: summary.publishedIdeas,
        })
    }, [id, hasStoredIdeas, summary.hasProgress, summary.rankedKeywords, summary.clusterCount, summary.totalIdeas, summary.publishedIdeas])

    if (authLoading || loading) {
        return (
            <div className="flex items-center justify-center min-h-screen bg-background">
                <Skeleton className="h-12 w-12 rounded-full" />
            </div>
        )
    }

    if (error && !topic) {
        return (
            <div className="flex flex-col items-center justify-center min-h-screen bg-background text-foreground">
                <p className="text-red-500 dark:text-red-400 mb-4">{error || "Topic not found"}</p>
                <Button onClick={() => navigate(backToResearchUrl)} variant="outline">
                    Back to Research
                </Button>
            </div>
        )
    }

    return (
        <div className="min-h-screen bg-background relative overflow-hidden">
            {/* Background gradient */}
            <div className="absolute inset-0 bg-gradient-to-b from-primary/10 to-transparent pointer-events-none" />

            <div className="relative z-10 p-6 md:p-8">
                {/* Header */}
                <div className="max-w-7xl mx-auto mb-8">
                    <div className="flex items-center gap-4 mb-2">
                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => navigate(backToResearchUrl)}
                            className="text-muted-foreground hover:text-foreground hover:bg-muted/50"
                        >
                            <ArrowLeft className="h-5 w-5" />
                        </Button>
                        <span className="text-sm text-muted-foreground">Back to Dashboard</span>
                    </div>

                    <div className="flex items-start justify-between">
                        <div>
                            <h1 className="text-3xl md:text-4xl font-bold text-foreground mb-2">
                                {topic?.title || 'Loading...'}
                            </h1>
                            <div className="flex items-center gap-3">
                                <span className={`px-3 py-1 rounded-full text-xs font-medium uppercase tracking-wide ${topic?.status === 'active'
                                        ? 'bg-emerald-500/10 text-emerald-500 dark:text-emerald-400 border border-emerald-500/20'
                                        : topic?.status === 'completed'
                                            ? 'bg-accent text-accent-foreground border border-border'
                                            : 'bg-muted/50 text-muted-foreground border border-border'
                                    }`}>
                                    {topic?.status || 'Unknown'}
                                </span>
                                {topic?.project_name && (
                                    <span className="px-3 py-1 rounded-full text-xs font-medium border border-border bg-muted/50 text-foreground">
                                        {topic.project_name}
                                    </span>
                                )}
                                {(topic?.primary_category_name || topic?.secondary_category_name) && (
                                    <span className="px-3 py-1 rounded-full text-xs font-medium border border-primary/20 bg-primary/10 text-primary">
                                        {[topic.primary_category_name, topic.secondary_category_name].filter(Boolean).join(' / ')}
                                    </span>
                                )}
                            </div>
                        </div>
                    </div>
                </div>

                {/* Pipeline Summary */}
                <div className="max-w-7xl mx-auto mb-8">
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                        {/* Ranked Keywords */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.1 }}
                            className="bg-muted/30 backdrop-blur-md border border-border rounded-xl p-6"
                        >
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-sm text-muted-foreground">Ranked Keywords</span>
                                <Search className="h-4 w-4 text-muted-foreground" />
                            </div>
                            <div className="text-2xl font-bold text-foreground">
                                {summary.rankedKeywords.toLocaleString()}
                            </div>
                            <div className="text-xs text-muted-foreground mt-1">
                                Scored opportunities ready for clustering
                            </div>
                        </motion.div>

                        {/* Clusters */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.2 }}
                            className="bg-muted/30 backdrop-blur-md border border-border rounded-xl p-6"
                        >
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-sm text-muted-foreground">Clusters</span>
                                <ListTree className="h-4 w-4 text-blue-500 dark:text-blue-400" />
                            </div>
                            <div className="text-2xl font-bold text-foreground">
                                {summary.clusterCount.toLocaleString()}
                            </div>
                            <div className="text-xs text-muted-foreground mt-1">
                                Distinct intents ready for idea generation
                            </div>
                        </motion.div>

                        {/* Ideas */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.3 }}
                            className="bg-muted/30 backdrop-blur-md border border-border rounded-xl p-6"
                        >
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-sm text-muted-foreground">Ideas</span>
                                <Sparkles className="h-4 w-4 text-amber-500 dark:text-amber-400" />
                            </div>
                            <div className="text-2xl font-bold text-foreground">
                                {summary.totalIdeas.toLocaleString()}
                            </div>
                            <div className="text-xs text-muted-foreground mt-1">
                                Saved content ideas
                            </div>
                        </motion.div>

                        {/* Published */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.4 }}
                            className="bg-muted/30 backdrop-blur-md border border-border rounded-xl p-6"
                        >
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-sm text-muted-foreground">Published</span>
                                <LibraryBig className="h-4 w-4 text-emerald-500 dark:text-emerald-400" />
                            </div>
                            <div className="text-2xl font-bold text-foreground">
                                {summary.publishedIdeas.toLocaleString()}
                            </div>
                            <div className="text-xs text-muted-foreground mt-1">
                                Ideas already sent into the library
                            </div>
                        </motion.div>
                    </div>
                </div>

                <TopicKeywordResearchPanel
                    topicId={id || ''}
                    keywordResearchRun={keywordResearchRun}
                    keywordCandidates={keywordCandidates}
                    keywordClusters={keywordClusters}
                    keywordResearchLoading={keywordResearchLoading}
                    runningKeywordResearch={runningKeywordResearch}
                    keywordResearchError={keywordResearchError}
                    generatingClusterIdeas={generatingClusterIdeas}
                    selectedClusterIds={selectedClusterIds}
                    canGenerateIdeas={Boolean(user?.id)}
                    keywordResearchSummary={keywordResearchSummary}
                    onRefreshKeywordResearch={() => loadKeywordResearch(id || '')}
                    onRunKeywordResearch={handleRunKeywordResearch}
                    onGenerateIdeasFromClusters={handleGenerateIdeasFromClusters}
                    onToggleClusterSelection={toggleClusterSelection}
                    formatDateTime={formatDateTime}
                />

                {/* Topic Context */}
                <div className="max-w-7xl mx-auto mb-8">
                    <div className="bg-muted/30 backdrop-blur-md border border-border rounded-2xl p-6">
                        <h2 className="text-lg font-semibold text-foreground mb-4">Topic Context</h2>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                            <div className="rounded-xl border border-border bg-muted/20 p-4">
                                <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">Intent Bucket</div>
                                <div className="text-sm font-medium text-foreground">{topic?.intent_bucket || 'Not set'}</div>
                            </div>
                            <div className="rounded-xl border border-border bg-muted/20 p-4">
                                <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">Decision Focus</div>
                                <div className="text-sm font-medium text-foreground">{topic?.decision_focus || 'Not set'}</div>
                            </div>
                            <div className="rounded-xl border border-border bg-muted/20 p-4">
                                <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">Value Tags</div>
                                <div className="flex flex-wrap gap-1">
                                    {(topic?.value_layer_tags && topic.value_layer_tags.length > 0) ? (
                                        topic.value_layer_tags.map((tag, idx) => (
                                            <span key={`${tag}-${idx}`} className="text-[11px] px-2 py-0.5 rounded-full border border-primary/20 bg-primary/10 text-primary">
                                                {tag}
                                            </span>
                                        ))
                                    ) : (
                                        <span className="text-sm text-muted-foreground">Not set</span>
                                    )}
                                </div>
                            </div>
                        </div>
                        {topic?.angle_question && (
                            <div className="rounded-xl border border-border bg-muted/20 p-4">
                                <div className="text-xs uppercase tracking-wide text-muted-foreground mb-1">Angle Question</div>
                                <div className="text-sm text-foreground">{topic.angle_question}</div>
                            </div>
                        )}
                    </div>
                </div>

                <GeneratedIdeasPanel
                    generatedClusterIdeas={generatedClusterIdeas}
                    visibleGeneratedClusterIdeas={visibleGeneratedClusterIdeas}
                    selectedGeneratedIdeaIds={selectedGeneratedIdeaIds}
                    publishingGeneratedIdeas={publishingGeneratedIdeas}
                    generatedIdeaTypeFilter={generatedIdeaTypeFilter}
                    generatedIdeaStatusFilter={generatedIdeaStatusFilter}
                    generatedIdeaSort={generatedIdeaSort}
                    setGeneratedIdeaTypeFilter={setGeneratedIdeaTypeFilter}
                    setGeneratedIdeaStatusFilter={setGeneratedIdeaStatusFilter}
                    setGeneratedIdeaSort={setGeneratedIdeaSort}
                    activeGeneratedIdea={activeGeneratedIdea}
                    activeGeneratedIdeaIndex={activeGeneratedIdeaIndex}
                    hasPreviousGeneratedIdea={hasPreviousGeneratedIdea}
                    hasNextGeneratedIdea={hasNextGeneratedIdea}
                    canPublish={Boolean(user?.id)}
                    openGeneratedIdeaDetail={openGeneratedIdeaDetail}
                    closeGeneratedIdeaDetail={closeGeneratedIdeaDetail}
                    toggleGeneratedIdeaSelection={toggleGeneratedIdeaSelection}
                    handlePublishGeneratedIdeas={handlePublishGeneratedIdeas}
                    handlePublishSingleGeneratedIdea={handlePublishSingleGeneratedIdea}
                    handleSelectVisibleGeneratedIdeas={handleSelectVisibleGeneratedIdeas}
                    handleSelectTopGeneratedIdeas={handleSelectTopGeneratedIdeas}
                    handleClearGeneratedIdeaSelection={handleClearGeneratedIdeaSelection}
                    openPreviousGeneratedIdea={openPreviousGeneratedIdea}
                    openNextGeneratedIdea={openNextGeneratedIdea}
                />

                {/* Error Display */}
                {error && topic && (
                    <div className="max-w-7xl mx-auto mt-4">
                        <motion.div
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-500 dark:text-red-400 text-center"
                        >
                            {error}
                        </motion.div>
                    </div>
                )}

            </div>
        </div>
    )
}
