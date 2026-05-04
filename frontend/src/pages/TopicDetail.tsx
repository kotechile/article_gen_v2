
import * as React from "react"
import { useParams, useNavigate, useLocation } from "react-router-dom"
import { useAuth } from "@/context/auth-context"
import { researchTopicsService } from "@/services/research-topics.service"
import { subtopicsService } from "@/services/subtopics.service"
import { topicKeywordResearchService } from "@/services/topic-keyword-research.service"
import { contentIdeasService } from "@/services/content-ideas.service"
import type { ContentIdea } from "@/types/idea-burst"
import type {
    ResearchTopic,
    Subtopic,
    TopicKeywordCandidate,
    TopicKeywordCluster,
    TopicKeywordResearchRun,
} from "@/types/research"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { ArrowLeft, ListTree, CheckCircle2, LibraryBig, Sparkles, RefreshCw, Lightbulb, Trash2, Archive, RotateCcw, Star, ChevronDown, ChevronRight } from "lucide-react"
import { motion } from "framer-motion"
import { IdeaBurstModal } from "@/components/IdeaBurstModal"
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
    const [subtopics, setSubtopics] = React.useState<Subtopic[]>([])
    const [loading, setLoading] = React.useState(true)
    const [decomposing, setDecomposing] = React.useState(false)
    const [decomposeStartedAt, setDecomposeStartedAt] = React.useState<number | null>(null)
    const [decomposeElapsedSec, setDecomposeElapsedSec] = React.useState(0)
    const [error, setError] = React.useState<string | null>(null)
    const [selectedSubtopic, setSelectedSubtopic] = React.useState<Subtopic | null>(null)
    const [showIdeaModal, setShowIdeaModal] = React.useState(false)
    const [deletingSubtopicId, setDeletingSubtopicId] = React.useState<string | null>(null)
    const [archivingSubtopicIds, setArchivingSubtopicIds] = React.useState<Set<string>>(new Set())
    const [ratingSubtopicIds, setRatingSubtopicIds] = React.useState<Set<string>>(new Set())
    const [hasStoredIdeas, setHasStoredIdeas] = React.useState(false)
    const [storedIdeas, setStoredIdeas] = React.useState<ContentIdea[]>([])
    const [savedIdeasCountBySubtopicName, setSavedIdeasCountBySubtopicName] = React.useState<Map<string, number>>(new Map())
    const [subtopicsReadyForContent, setSubtopicsReadyForContent] = React.useState<Set<string>>(new Set())
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
    const [showLegacySubtopics, setShowLegacySubtopics] = React.useState(false)
    const [activeGeneratedIdea, setActiveGeneratedIdea] = React.useState<ContentIdea | null>(null)
    const [generatedIdeaTypeFilter, setGeneratedIdeaTypeFilter] = React.useState<'all' | 'blog' | 'software'>('all')
    const [generatedIdeaStatusFilter, setGeneratedIdeaStatusFilter] = React.useState<'all' | 'draft' | 'published'>('draft')
    const [generatedIdeaSort, setGeneratedIdeaSort] = React.useState<'score' | 'volume' | 'difficulty' | 'recent'>('score')
    const decomposeToastIdRef = React.useRef<string | number | null>(null)
    const sleep = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms))

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
            setSavedIdeasCountBySubtopicName(new Map())
            setSubtopicsReadyForContent(new Set())
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
        const savedSubtopicCounts = new Map<string, number>()
        for (const idea of storedIdeas || []) {
            const key = (idea.subtopic || '').trim().toLowerCase()
            if (!key) continue
            savedSubtopicCounts.set(key, (savedSubtopicCounts.get(key) || 0) + 1)
        }
        const readySubtopicNames = new Set(
            (storedIdeas || [])
                .filter((idea) =>
                    Boolean(
                        idea.published ||
                        idea.published_to_titles ||
                        idea.titles_record_id ||
                        idea.status?.toLowerCase() === 'published'
                    )
                )
                .map((idea) => (idea.subtopic || '').trim().toLowerCase())
                .filter(Boolean)
        )
        setSavedIdeasCountBySubtopicName(savedSubtopicCounts)
        setSubtopicsReadyForContent(readySubtopicNames)
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
            // Parallel fetch
            const [topicData, subtopicsData] = await Promise.all([
                researchTopicsService.getResearchTopic(topicId),
                subtopicsService.getSubtopics(topicId)
            ])

            setTopic(topicData)
            setSubtopics(subtopicsData || [])
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

            await refreshStoredIdeasState(id)

            toast.success(
                `Generated ${result.generated_count || 0} ideas from ${clusterIds.length} cluster${clusterIds.length === 1 ? '' : 's'}.`,
                { id: 'topic-cluster-ideas' }
            )

            if (result.persistence_warning) {
                setError(result.persistence_warning)
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
    }, [id, keywordResearchRun, refreshStoredIdeasState, selectedClusterIds, user?.id])

    const handleIdeaModalClose = React.useCallback(async () => {
        setShowIdeaModal(false)
        if (!id) return
        try {
            await refreshStoredIdeasState(id)
        } catch (err) {
            console.warn('Failed to refresh content ideas after closing idea modal:', err)
        }
    }, [id, refreshStoredIdeasState])

    const handleDecompose = async () => {
        if (!id) return
        const preCount = subtopics.length
        try {
            setDecomposing(true)
            setDecomposeStartedAt(Date.now())
            setDecomposeElapsedSec(0)
            decomposeToastIdRef.current = toast.loading('Generating sub-topics. This can take 30-90 seconds while SEO evidence is collected.')
            // Call the generation endpoint
            const newSubtopics = await subtopicsService.generateSubtopics(id)
            let finalSubtopics = newSubtopics
            setSubtopics(newSubtopics)

            // Some deployments persist rows shortly after the generate response returns.
            // Keep the "in progress" state while we reconcile the final list.
            if (newSubtopics.length === 0) {
                toast.loading('Finalizing sub-topics... waiting for persistence.', {
                    id: decomposeToastIdRef.current ?? undefined,
                })
                const maxAttempts = 15
                for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
                    await sleep(3000)
                    const latest = await subtopicsService.getSubtopics(id)
                    if ((latest || []).length > 0) {
                        finalSubtopics = latest
                        setSubtopics(latest)
                        break
                    }
                    if (attempt % 5 === 0) {
                        toast.loading('Still processing sub-topics... this run is taking longer than usual.', {
                            id: decomposeToastIdRef.current ?? undefined,
                        })
                    }
                }
            }

            if ((finalSubtopics || []).length > 0) {
                const generatedDelta = Math.max(0, finalSubtopics.length - preCount)
                const messageCount = generatedDelta > 0 ? generatedDelta : finalSubtopics.length
                toast.success(`Generated ${messageCount} sub-topic${messageCount === 1 ? '' : 's'}.`, {
                    id: decomposeToastIdRef.current ?? undefined,
                })
            } else {
                toast.warning('Generation finished, but no sub-topics were returned yet. Try Refresh in a few seconds.', {
                    id: decomposeToastIdRef.current ?? undefined,
                })
            }
        } catch (err) {
            console.error('Failed to decompose topic:', err)
            const status =
                (err as any)?.response?.status ??
                (err as any)?.status ??
                null
            const errorCode = String((err as any)?.code || '')
            const errorMessage = String((err as any)?.message || '').toLowerCase()
            const isRecoverableTimeout =
                status === 504 ||
                errorCode === 'ECONNABORTED' ||
                errorMessage.includes('timeout') ||
                errorMessage.includes('504')

            // Recovery path: backend may still complete after proxy or client timeout.
            if (isRecoverableTimeout) {
                toast.loading('Request timed out, but processing may still be running. Checking for new sub-topics...', {
                    id: decomposeToastIdRef.current ?? undefined,
                })

                let recoveredSubtopics: Subtopic[] = []
                const maxAttempts = 24 // ~72s
                for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
                    await sleep(3000)
                    const latest = await subtopicsService.getSubtopics(id)
                    const grew = (latest || []).length > preCount
                    if (grew || (latest || []).length > 0) {
                        recoveredSubtopics = latest
                        setSubtopics(latest)
                        break
                    }
                    if (attempt % 6 === 0) {
                        toast.loading('Still checking background generation... please keep this page open.', {
                            id: decomposeToastIdRef.current ?? undefined,
                        })
                    }
                }

                if (recoveredSubtopics.length > 0) {
                    const generatedDelta = Math.max(0, recoveredSubtopics.length - preCount)
                    const messageCount = generatedDelta > 0 ? generatedDelta : recoveredSubtopics.length
                    toast.success(`Background generation completed: ${messageCount} sub-topic${messageCount === 1 ? '' : 's'} ready.`, {
                        id: decomposeToastIdRef.current ?? undefined,
                    })
                    setError(null)
                } else {
                    setError('Generation is still processing in the background. Click Refresh in ~30 seconds.')
                    toast.warning('Still no sub-topics visible yet. Processing may still be in progress.', {
                        id: decomposeToastIdRef.current ?? undefined,
                    })
                }
            } else {
                setError('Failed to decompose topic. Please try again.')
                toast.error('Sub-topic generation failed. Please try again.', {
                    id: decomposeToastIdRef.current ?? undefined,
                })
            }
        } finally {
            setDecomposing(false)
            setDecomposeStartedAt(null)
            setDecomposeElapsedSec(0)
            decomposeToastIdRef.current = null
        }
    }

    React.useEffect(() => {
        if (!decomposing || !decomposeStartedAt) {
            return
        }

        const interval = window.setInterval(() => {
            const sec = Math.max(0, Math.floor((Date.now() - decomposeStartedAt) / 1000))
            setDecomposeElapsedSec(sec)
        }, 1000)

        return () => {
            window.clearInterval(interval)
        }
    }, [decomposing, decomposeStartedAt])

    const decomposeStatus = React.useMemo(() => {
        if (!decomposing) {
            return ''
        }
        if (decomposeElapsedSec < 15) {
            return 'Preparing editorial sub-topics...'
        }
        if (decomposeElapsedSec < 35) {
            return 'Mining keyword evidence for each sub-topic...'
        }
        if (decomposeElapsedSec < 70) {
            return 'Validating SEO metrics. Processing is healthy, this run is just heavier than usual...'
        }
        return 'Still working and waiting on upstream data providers. You can keep this page open.'
    }, [decomposing, decomposeElapsedSec])

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

    const decomposeElapsedLabel = React.useMemo(() => {
        if (!decomposing) {
            return ''
        }
        if (decomposeElapsedSec < 60) {
            return `${decomposeElapsedSec}s elapsed`
        }
        const min = Math.floor(decomposeElapsedSec / 60)
        const sec = decomposeElapsedSec % 60
        return `${min}m ${sec}s elapsed`
    }, [decomposing, decomposeElapsedSec])

    const handleSubtopicClick = (sub: Subtopic) => {
        setSelectedSubtopic(sub)
        setShowIdeaModal(true)
    }

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

    const handleDeleteSubtopic = async (subtopic: Subtopic) => {
        if (!id || !subtopic?.id) return
        const name = (subtopic.name || 'this sub-topic').trim()
        const confirmed = window.confirm(
            `Delete "${name}" and all associated content ideas? This cannot be undone.`
        )
        if (!confirmed) return

        try {
            setDeletingSubtopicId(subtopic.id)
            await subtopicsService.deleteSubtopic(id, subtopic.id)
            setSubtopics((prev) => prev.filter((item) => item.id !== subtopic.id))
            setSavedIdeasCountBySubtopicName((prev) => {
                const next = new Map(prev)
                next.delete(name.toLowerCase())
                return next
            })
            setSubtopicsReadyForContent((prev) => {
                const next = new Set(prev)
                next.delete(name.toLowerCase())
                return next
            })
            toast.success(`Deleted "${name}" and its content ideas`)
        } catch (err) {
            console.error('Failed to delete subtopic:', err)
            toast.error('Failed to delete sub-topic')
        } finally {
            setDeletingSubtopicId(null)
        }
    }

    const intentChipClass = (intent?: string | null) => {
        const value = (intent || '').toLowerCase()
        if (value.includes('transactional')) return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
        if (value.includes('commercial')) return 'bg-amber-500/10 text-amber-400 border-amber-500/20'
        return 'bg-blue-500/10 text-blue-400 border-blue-500/20'
    }

    const clusterChipClass = (clusterType?: string | null) => {
        const value = (clusterType || '').toLowerCase()
        if (value.includes('calculator') || value.includes('tool')) return 'bg-violet-500/10 text-violet-400 border-violet-500/20'
        if (value.includes('comparison')) return 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20'
        if (value.includes('checklist') || value.includes('audit')) return 'bg-orange-500/10 text-orange-400 border-orange-500/20'
        return 'bg-slate-500/10 text-slate-300 border-slate-500/20'
    }

    const isSubtopicResearched = (sub: Subtopic) => {
        return Boolean((sub as any).researched || sub.trend_analysis?.manual_researched)
    }

    const handleToggleSubtopicResearched = async (sub: Subtopic, next: boolean) => {
        if (!id || !sub?.id) return

        const previousSubtopics = subtopics

        const nextTrendAnalysis = {
            ...(sub.trend_analysis || {}),
            manual_researched: next,
            manual_researched_at: next ? new Date().toISOString() : null,
        }

        setSubtopics((prev) =>
            prev.map((item) =>
                item.id === sub.id
                    ? {
                          ...item,
                          researched: next,
                          trend_analysis: nextTrendAnalysis,
                      }
                    : item
            )
        )

        try {
            await subtopicsService.updateSubtopic(id, sub.id, {
                trend_analysis: nextTrendAnalysis,
            })
            toast.success(`Marked as ${next ? "researched" : "not researched"}.`)
        } catch (err) {
            console.error("Failed to update researched state:", err)
            setSubtopics(previousSubtopics)
            setError("Failed to update researched state")
            toast.error("Could not save researched state.")
        }
    }

    const handleToggleSubtopicArchived = async (e: React.MouseEvent, sub: Subtopic, nextArchived: boolean) => {
        e.stopPropagation()
        if (!id || !sub?.id) return

        setArchivingSubtopicIds((current) => new Set(current).add(sub.id))
        try {
            const updated = await subtopicsService.updateSubtopic(id, sub.id, { is_archived: nextArchived })
            if (updated) {
                setSubtopics((prev) => prev.map((item) => (item.id === sub.id ? { ...item, ...updated } : item)))
            }
        } catch (err) {
            console.error("Failed to update archived state:", err)
            toast.error(`Could not ${nextArchived ? 'archive' : 'restore'} sub-topic.`)
        } finally {
            setArchivingSubtopicIds((current) => {
                const next = new Set(current)
                next.delete(sub.id)
                return next
            })
        }
    }

    const handleSetSubtopicRating = async (e: React.MouseEvent, sub: Subtopic, rating: number) => {
        e.stopPropagation()
        if (!id || !sub?.id) return

        const nextRating = Math.max(0, Math.min(5, rating))
        setRatingSubtopicIds((current) => new Set(current).add(sub.id))
        try {
            const updated = await subtopicsService.updateSubtopic(id, sub.id, { topic_rating: nextRating })
            if (updated) {
                setSubtopics((prev) => prev.map((item) => (item.id === sub.id ? { ...item, ...updated } : item)))
            }
        } catch (err) {
            console.error("Failed to update sub-topic rating:", err)
            toast.error("Could not save sub-topic rating.")
        } finally {
            setRatingSubtopicIds((current) => {
                const next = new Set(current)
                next.delete(sub.id)
                return next
            })
        }
    }

    const activeSubtopics = React.useMemo(
        () => subtopics.filter((sub) => !sub.is_archived),
        [subtopics]
    )
    const archivedSubtopics = React.useMemo(
        () => subtopics.filter((sub) => Boolean(sub.is_archived)),
        [subtopics]
    )

    const summary = React.useMemo(() => {
        const totalSubtopics = subtopics.length
        const completedSubtopics = subtopics.filter(isSubtopicResearched).length
        const inLibrarySubtopics = subtopicsReadyForContent.size
        let totalIdeas = 0
        for (const sub of subtopics) {
            const normalizedName = (sub.name || '').trim().toLowerCase()
            const persistedIdeasCount = savedIdeasCountBySubtopicName.get(normalizedName) || 0
            totalIdeas += persistedIdeasCount
        }
        return {
            totalSubtopics,
            completedSubtopics,
            inLibrarySubtopics,
            totalIdeas,
            hasProgress: totalIdeas > 0 || hasStoredIdeas || inLibrarySubtopics > 0 || completedSubtopics > 0,
        }
    }, [subtopics, subtopicsReadyForContent, savedIdeasCountBySubtopicName, hasStoredIdeas])

    React.useEffect(() => {
        if (!id) return
        console.info('[TopicDetail] status snapshot', {
            topicId: id,
            subtopics: subtopics.length,
            hasStoredIdeas,
            hasProgress: summary.hasProgress,
            totalIdeas: summary.totalIdeas,
            completedSubtopics: summary.completedSubtopics,
            inLibrarySubtopics: summary.inLibrarySubtopics,
        })
    }, [id, subtopics.length, hasStoredIdeas, summary.hasProgress, summary.totalIdeas, summary.completedSubtopics, summary.inLibrarySubtopics])

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

                {/* Sub-topic Summary */}
                <div className="max-w-7xl mx-auto mb-8">
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                        {/* Total Sub-Topics */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.1 }}
                            className="bg-muted/30 backdrop-blur-md border border-border rounded-xl p-6"
                        >
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-sm text-muted-foreground">Legacy Sub-Topics</span>
                                <ListTree className="h-4 w-4 text-muted-foreground" />
                            </div>
                            <div className="text-2xl font-bold text-foreground">
                                {summary.totalSubtopics.toLocaleString()}
                            </div>
                            <div className="text-xs text-muted-foreground mt-1">
                                Fallback workflow items
                            </div>
                        </motion.div>

                        {/* Completed */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.2 }}
                            className="bg-muted/30 backdrop-blur-md border border-border rounded-xl p-6"
                        >
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-sm text-muted-foreground">Completed</span>
                                <CheckCircle2 className="h-4 w-4 text-blue-500 dark:text-blue-400" />
                            </div>
                            <div className="text-2xl font-bold text-foreground">
                                {summary.completedSubtopics.toLocaleString()}
                            </div>
                            <div className="text-xs text-muted-foreground mt-1">
                                Marked as reviewed
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

                        {/* In Library */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.4 }}
                            className="bg-muted/30 backdrop-blur-md border border-border rounded-xl p-6"
                        >
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-sm text-muted-foreground">In Library</span>
                                <LibraryBig className="h-4 w-4 text-emerald-500 dark:text-emerald-400" />
                            </div>
                            <div className="text-2xl font-bold text-foreground">
                                {summary.inLibrarySubtopics.toLocaleString()}
                            </div>
                            <div className="text-xs text-muted-foreground mt-1">
                                Sub-topics with published ideas
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

                {/* Sub-Topics Section */}
                <div className="max-w-7xl mx-auto">
                    <div className="bg-muted/30 backdrop-blur-md border border-border rounded-2xl p-6">
                        <div className="mb-6 flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                            <div>
                                <div className="mb-2 flex items-center gap-2">
                                    <h2 className="text-xl font-semibold text-foreground">Legacy Sub-Topic Workflow</h2>
                                    <span className="rounded-full border border-amber-500/20 bg-amber-500/10 px-2 py-0.5 text-[11px] text-amber-300">
                                        Fallback
                                    </span>
                                </div>
                                <p className="text-sm text-muted-foreground">
                                    Use this only when you need the older subtopic-first process. The keyword research and cluster-based generation flow above is now the primary path.
                                </p>
                            </div>
                            <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                onClick={() => setShowLegacySubtopics((current) => !current)}
                                className="border-border hover:bg-muted"
                            >
                                {showLegacySubtopics ? (
                                    <>
                                        <ChevronDown className="mr-2 h-4 w-4" />
                                        Hide Legacy Workflow
                                    </>
                                ) : (
                                    <>
                                        <ChevronRight className="mr-2 h-4 w-4" />
                                        Show Legacy Workflow
                                    </>
                                )}
                            </Button>
                        </div>

                        {!showLegacySubtopics ? (
                            <div className="rounded-xl border border-dashed border-border bg-muted/20 p-6 text-center">
                                <ListTree className="mx-auto mb-3 h-10 w-10 text-muted-foreground" />
                                <h3 className="text-base font-semibold text-foreground mb-2">Legacy workflow hidden</h3>
                                <p className="text-sm text-muted-foreground max-w-2xl mx-auto">
                                    Open this section only if you need the older subtopic generation and idea-burst workflow.
                                </p>
                            </div>
                        ) : (
                            <>
                        <div className="flex items-center justify-between mb-6">
                            <h3 className="text-lg font-semibold text-foreground">
                                Sub-Topics for {topic?.title || 'this topic'}
                            </h3>
                            {subtopics.length > 0 && (
                                <div className="flex items-center gap-2">
                                    <Button
                                        onClick={handleDecompose}
                                        disabled={decomposing}
                                        variant="outline"
                                        size="sm"
                                        className="border-border hover:bg-muted"
                                    >
                                        {decomposing ? (
                                            <>
                                                <RefreshCw className="mr-2 h-3 w-3 animate-spin" />
                                                Refreshing...
                                            </>
                                        ) : (
                                            <>
                                                <RefreshCw className="mr-2 h-3 w-3" />
                                                Refresh
                                            </>
                                        )}
                                    </Button>
                                </div>
                            )}
                        </div>

                        {decomposing && (
                            <div className="mb-5 rounded-xl border border-blue-500/20 bg-blue-500/10 p-4">
                                <div className="flex items-center justify-between gap-3">
                                    <div className="flex items-center gap-2">
                                        <RefreshCw className="h-4 w-4 animate-spin text-blue-400" />
                                        <span className="text-sm font-medium text-blue-100">{decomposeStatus}</span>
                                    </div>
                                    <span className="text-xs text-blue-200/80">{decomposeElapsedLabel}</span>
                                </div>
                                <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-blue-500/20">
                                    <div className="h-full w-1/3 animate-pulse rounded-full bg-blue-400/70" />
                                </div>
                            </div>
                        )}

                        {subtopics.length === 0 ? (
                            <motion.div
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                className="py-16 text-center"
                            >
                                <div className="max-w-md mx-auto">
                                    <Sparkles className="h-16 w-16 text-muted-foreground mx-auto mb-6" />
                                    <h3 className="text-xl font-semibold text-foreground mb-3">
                                        No subtopics found
                                    </h3>
                                    <p className="text-muted-foreground mb-8 leading-relaxed">
                                        Click "Generate Sub-Topics" to create sub-topics for this research topic.
                                    </p>
                                    <Button
                                        onClick={handleDecompose}
                                        disabled={decomposing}
                                        size="lg"
                                        className="bg-primary hover:bg-primary/90 text-primary-foreground px-8"
                                    >
                                        {decomposing ? (
                                            <>
                                                <RefreshCw className="mr-2 h-5 w-5 animate-spin" />
                                                Generating Sub-Topics...
                                            </>
                                        ) : (
                                            <>
                                                <Sparkles className="mr-2 h-5 w-5" />
                                                Generate Sub-Topics
                                            </>
                                        )}
                                    </Button>
                                    {decomposing && (
                                        <p className="mt-4 text-sm text-muted-foreground">
                                            This can take longer when keyword sources are busy. Please keep this tab open.
                                        </p>
                                    )}
                                </div>
                            </motion.div>
                        ) : (
                            <div className="space-y-6">
                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                    {activeSubtopics.map((sub, i) => {
                                        const normalizedSubtopicName = (sub.name || '').trim().toLowerCase()
                                        const persistedIdeasCount = savedIdeasCountBySubtopicName.get(normalizedSubtopicName) || 0
                                        const readyForContent = subtopicsReadyForContent.has(normalizedSubtopicName)
                                        const totalIdeasCount = persistedIdeasCount
                                        return (
                                            <motion.div
                                                key={sub.id || i}
                                                initial={{ opacity: 0, y: 20 }}
                                                animate={{ opacity: 1, y: 0 }}
                                                transition={{ delay: i * 0.05 }}
                                                onClick={() => handleSubtopicClick(sub)}
                                                className="bg-muted/30 backdrop-blur-sm border border-border rounded-xl p-5 cursor-pointer hover:bg-muted/50 hover:border-ring/50 transition-all duration-200 group"
                                            >
                                                <div className="flex flex-col gap-3 mb-3">
                                                    <div className="flex items-start justify-between">
                                                        <h3
                                                            className="font-semibold text-foreground pr-2 group-hover:text-foreground transition-colors flex items-start gap-2"
                                                            title={sub.name || ''}
                                                        >
                                                            <Lightbulb className="w-4 h-4 text-primary flex-shrink-0 mt-1" />
                                                            <span className="leading-tight">{sub.name}</span>
                                                        </h3>
                                                        <div className="flex items-center gap-1 flex-shrink-0">
                                                            <Button
                                                                variant="ghost"
                                                                size="icon"
                                                                className="h-7 w-7 text-muted-foreground hover:text-foreground"
                                                                title="Archive sub-topic"
                                                                onClick={(e) => handleToggleSubtopicArchived(e, sub, true)}
                                                                disabled={archivingSubtopicIds.has(sub.id)}
                                                            >
                                                                {archivingSubtopicIds.has(sub.id) ? (
                                                                    <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                                                                ) : (
                                                                    <Archive className="h-3.5 w-3.5" />
                                                                )}
                                                            </Button>
                                                            <Button
                                                                variant="ghost"
                                                                size="icon"
                                                                className="h-7 w-7 text-muted-foreground hover:text-red-400 hover:bg-red-500/10"
                                                                title="Delete sub-topic and related content ideas"
                                                                onClick={(e) => {
                                                                    e.stopPropagation()
                                                                    handleDeleteSubtopic(sub)
                                                                }}
                                                                disabled={deletingSubtopicId === sub.id}
                                                            >
                                                                {deletingSubtopicId === sub.id ? (
                                                                    <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                                                                ) : (
                                                                    <Trash2 className="h-3.5 w-3.5" />
                                                                )}
                                                            </Button>
                                                        </div>
                                                    </div>
                                                    <div className="flex items-center gap-2 flex-wrap">
                                                        <label
                                                            className={`text-xs px-2 py-1 rounded-full border transition-colors inline-flex items-center gap-1.5 cursor-pointer ${
                                                                isSubtopicResearched(sub)
                                                                    ? 'text-emerald-500 dark:text-emerald-400 border-emerald-500/30 bg-emerald-500/10'
                                                                    : 'text-muted-foreground border-border bg-muted/50 hover:bg-muted'
                                                            }`}
                                                            title="Mark subtopic as completed"
                                                            onClick={(e) => e.stopPropagation()}
                                                        >
                                                            <input
                                                                type="checkbox"
                                                                checked={isSubtopicResearched(sub)}
                                                                onClick={(e) => e.stopPropagation()}
                                                                onChange={(e) => {
                                                                    e.stopPropagation()
                                                                    handleToggleSubtopicResearched(sub, e.target.checked)
                                                                }}
                                                                className="h-3 w-3 accent-emerald-500 cursor-pointer"
                                                            />
                                                            Completed
                                                        </label>
                                                        {totalIdeasCount > 0 && (
                                                            <span className="text-xs px-2 py-1 rounded-full border flex-shrink-0 text-indigo-300 border-indigo-500/30 bg-indigo-500/10">
                                                                {totalIdeasCount} {totalIdeasCount === 1 ? 'Idea' : 'Ideas'}
                                                            </span>
                                                        )}
                                                        <span className={`text-xs px-2 py-1 rounded-full border flex-shrink-0 ${
                                                            readyForContent
                                                                ? 'text-emerald-500 dark:text-emerald-400 border-emerald-500/30 bg-emerald-500/10'
                                                                : 'text-muted-foreground border-border bg-muted/50'
                                                        }`}>
                                                            {readyForContent ? 'In Library' : 'Empty'}
                                                        </span>
                                                    </div>
                                                </div>

                                                <div className="mb-3 flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                                                    {[1, 2, 3, 4, 5].map((value) => {
                                                        const isFilled = value <= Number(sub.topic_rating || 0)
                                                        return (
                                                            <button
                                                                key={`${sub.id}-rating-${value}`}
                                                                type="button"
                                                                onClick={(e) => handleSetSubtopicRating(e, sub, value)}
                                                                disabled={ratingSubtopicIds.has(sub.id)}
                                                                className="rounded p-0.5 transition hover:scale-105 disabled:cursor-not-allowed disabled:opacity-60"
                                                                aria-label={`Rate ${sub.name} ${value} star${value === 1 ? '' : 's'}`}
                                                            >
                                                                <Star className={`h-4 w-4 ${isFilled ? 'fill-amber-400 text-amber-400' : 'text-muted-foreground'}`} />
                                                            </button>
                                                        )
                                                    })}
                                                    {ratingSubtopicIds.has(sub.id) && <RefreshCw className="h-3 w-3 animate-spin text-muted-foreground" />}
                                                </div>

                                                {(sub.intent_bucket || sub.cluster_type || sub.primary_user_outcome || sub.decision_focus) && (
                                                    <div className="mt-3 pt-3 border-t border-white/5 space-y-2">
                                                        <div className="flex flex-wrap gap-1.5">
                                                            {sub.intent_bucket && (
                                                                <span className={`text-[10px] px-2 py-0.5 rounded-full border ${intentChipClass(sub.intent_bucket)}`}>
                                                                    Intent: {sub.intent_bucket}
                                                                </span>
                                                            )}
                                                            {sub.cluster_type && (
                                                                <span className={`text-[10px] px-2 py-0.5 rounded-full border ${clusterChipClass(sub.cluster_type)}`}>
                                                                    Cluster: {sub.cluster_type}
                                                                </span>
                                                            )}
                                                        </div>
                                                        {sub.decision_focus && (
                                                            <p className="text-[11px] text-slate-300 line-clamp-2" title={sub.decision_focus}>
                                                                <span className="text-indigo-400 font-medium">Decision:</span> {sub.decision_focus}
                                                            </p>
                                                        )}
                                                        {sub.primary_user_outcome && (
                                                            <p className="text-[11px] text-slate-300 line-clamp-2" title={sub.primary_user_outcome}>
                                                                <span className="text-indigo-400 font-medium">Outcome:</span> {sub.primary_user_outcome}
                                                            </p>
                                                        )}
                                                    </div>
                                                )}
                                            </motion.div>
                                        )
                                    })}
                                </div>

                                {archivedSubtopics.length > 0 && (
                                    <div className="pt-1">
                                        <div className="mb-3 flex items-center justify-between">
                                            <h3 className="text-base font-semibold text-foreground">Archived Sub-Topics</h3>
                                            <span className="rounded-full border border-border bg-muted/40 px-2.5 py-1 text-xs text-muted-foreground">
                                                {archivedSubtopics.length}
                                            </span>
                                        </div>
                                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                            {archivedSubtopics.map((sub, i) => {
                                                const normalizedSubtopicName = (sub.name || '').trim().toLowerCase()
                                                const persistedIdeasCount = savedIdeasCountBySubtopicName.get(normalizedSubtopicName) || 0
                                                return (
                                                    <motion.div
                                                        key={sub.id || i}
                                                        initial={{ opacity: 0, y: 20 }}
                                                        animate={{ opacity: 1, y: 0 }}
                                                        transition={{ delay: i * 0.04 }}
                                                        onClick={() => handleSubtopicClick(sub)}
                                                        className="bg-muted/20 backdrop-blur-sm border border-border/80 rounded-xl p-5 cursor-pointer transition-all duration-200 opacity-70 saturate-0"
                                                    >
                                                        <div className="flex flex-col gap-3 mb-3">
                                                            <div className="flex items-start justify-between">
                                                                <h3 className="font-semibold text-foreground pr-2 flex items-start gap-2" title={sub.name || ''}>
                                                                    <Lightbulb className="w-4 h-4 text-primary flex-shrink-0 mt-1" />
                                                                    <span className="leading-tight">{sub.name}</span>
                                                                </h3>
                                                                <div className="flex items-center gap-1 flex-shrink-0">
                                                                    <Button
                                                                        variant="ghost"
                                                                        size="icon"
                                                                        className="h-7 w-7 text-muted-foreground hover:text-foreground"
                                                                        title="Restore sub-topic"
                                                                        onClick={(e) => handleToggleSubtopicArchived(e, sub, false)}
                                                                        disabled={archivingSubtopicIds.has(sub.id)}
                                                                    >
                                                                        {archivingSubtopicIds.has(sub.id) ? (
                                                                            <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                                                                        ) : (
                                                                            <RotateCcw className="h-3.5 w-3.5" />
                                                                        )}
                                                                    </Button>
                                                                    <Button
                                                                        variant="ghost"
                                                                        size="icon"
                                                                        className="h-7 w-7 text-muted-foreground hover:text-red-400 hover:bg-red-500/10"
                                                                        title="Delete sub-topic and related content ideas"
                                                                        onClick={(e) => {
                                                                            e.stopPropagation()
                                                                            handleDeleteSubtopic(sub)
                                                                        }}
                                                                        disabled={deletingSubtopicId === sub.id}
                                                                    >
                                                                        {deletingSubtopicId === sub.id ? (
                                                                            <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                                                                        ) : (
                                                                            <Trash2 className="h-3.5 w-3.5" />
                                                                        )}
                                                                    </Button>
                                                                </div>
                                                            </div>
                                                        </div>

                                                        <div className="mb-3 flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                                                            {[1, 2, 3, 4, 5].map((value) => {
                                                                const isFilled = value <= Number(sub.topic_rating || 0)
                                                                return (
                                                                    <button
                                                                        key={`${sub.id}-archived-rating-${value}`}
                                                                        type="button"
                                                                        onClick={(e) => handleSetSubtopicRating(e, sub, value)}
                                                                        disabled={ratingSubtopicIds.has(sub.id)}
                                                                        className="rounded p-0.5 transition hover:scale-105 disabled:cursor-not-allowed disabled:opacity-60"
                                                                        aria-label={`Rate ${sub.name} ${value} star${value === 1 ? '' : 's'}`}
                                                                    >
                                                                        <Star className={`h-4 w-4 ${isFilled ? 'fill-amber-400 text-amber-400' : 'text-muted-foreground'}`} />
                                                                    </button>
                                                                )
                                                            })}
                                                            {ratingSubtopicIds.has(sub.id) && <RefreshCw className="h-3 w-3 animate-spin text-muted-foreground" />}
                                                        </div>

                                                        {(sub.decision_focus || sub.primary_user_outcome || sub.rationale) && (
                                                            <p className="text-[11px] text-slate-300 line-clamp-2" title={sub.decision_focus || sub.primary_user_outcome || sub.rationale || ''}>
                                                                {sub.decision_focus || sub.primary_user_outcome || sub.rationale}
                                                            </p>
                                                        )}

                                                        {persistedIdeasCount > 0 && (
                                                            <span className="mt-3 inline-flex text-xs px-2 py-1 rounded-full border text-indigo-300 border-indigo-500/30 bg-indigo-500/10">
                                                                {persistedIdeasCount} {persistedIdeasCount === 1 ? 'Idea' : 'Ideas'}
                                                            </span>
                                                        )}
                                                    </motion.div>
                                                )
                                            })}
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}
                            </>
                        )}
                    </div>
                </div>

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

                {/* Idea Burst Modal */}
                <IdeaBurstModal
                    isOpen={showIdeaModal}
                    onClose={handleIdeaModalClose}
                    subtopic={selectedSubtopic}
                    topicId={id || ''}
                    topicTitle={topic?.title || ''}
                    projectName={topic?.project_name || null}
                    categoryPath={[topic?.primary_category_name, topic?.secondary_category_name].filter(Boolean).join(' / ') || null}
                />
            </div>
        </div>
    )
}
