
import * as React from "react"
import { useParams, useNavigate, useLocation } from "react-router-dom"
import { useAuth } from "@/context/auth-context"
import { researchTopicsService } from "@/services/research-topics.service"
import { subtopicsService } from "@/services/subtopics.service"
import { contentIdeasService } from "@/services/content-ideas.service"
import type { ResearchTopic, Subtopic } from "@/types/research"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { ArrowLeft, ListTree, CheckCircle2, LibraryBig, Sparkles, RefreshCw, Lightbulb, Trash2, Archive, RotateCcw, Star } from "lucide-react"
import { motion } from "framer-motion"
import { IdeaBurstModal } from "@/components/IdeaBurstModal"
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
    const [savedIdeasCountBySubtopicName, setSavedIdeasCountBySubtopicName] = React.useState<Map<string, number>>(new Map())
    const [subtopicsReadyForContent, setSubtopicsReadyForContent] = React.useState<Set<string>>(new Set())
    const decomposeToastIdRef = React.useRef<string | number | null>(null)
    const sleep = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms))

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
            if (user?.id) {
                const storedIdeas = await contentIdeasService.getContentIdeas(topicId, user.id)
                console.info('[TopicDetail] loaded stored ideas', {
                    topicId,
                    total: storedIdeas.length,
                    blog: storedIdeas.filter((idea) => idea.content_type === 'blog').length,
                    software: storedIdeas.filter((idea) => idea.content_type === 'software').length,
                })
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
            } else {
                setHasStoredIdeas(false)
                setSavedIdeasCountBySubtopicName(new Map())
                setSubtopicsReadyForContent(new Set())
            }
            setError(null)
        } catch (err) {
            console.error('Failed to load topic data:', err)
            setError('Failed to load topic details')
        } finally {
            setLoading(false)
        }
    }

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
            const isGatewayTimeout =
                status === 504 ||
                String((err as any)?.message || '').includes('504')

            // Recovery path: backend may still complete after proxy timeout.
            if (isGatewayTimeout) {
                toast.loading('Request timed out at gateway, but processing may still be running. Checking for new sub-topics...', {
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
                                <span className="text-sm text-muted-foreground">Sub-Topics</span>
                                <ListTree className="h-4 w-4 text-muted-foreground" />
                            </div>
                            <div className="text-2xl font-bold text-foreground">
                                {summary.totalSubtopics.toLocaleString()}
                            </div>
                            <div className="text-xs text-muted-foreground mt-1">
                                Total generated clusters
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

                {/* Sub-Topics Section */}
                <div className="max-w-7xl mx-auto">
                    <div className="bg-muted/30 backdrop-blur-md border border-border rounded-2xl p-6">
                        <div className="flex items-center justify-between mb-6">
                            <h2 className="text-xl font-semibold text-foreground">
                                Sub-Topics for {topic?.title || 'this topic'}
                            </h2>
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
                                                <div className="flex items-start justify-between mb-3">
                                                    <h3
                                                        className="font-semibold text-foreground line-clamp-2 flex-1 pr-2 group-hover:text-foreground transition-colors flex items-center gap-2"
                                                        title={sub.name || ''}
                                                    >
                                                        <Lightbulb className="w-4 h-4 text-primary flex-shrink-0" />
                                                        {sub.name}
                                                    </h3>
                                                    <div className="flex items-center gap-2">
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
                                                        <label
                                                            className={`text-xs px-2 py-1 rounded-full border transition-colors inline-flex items-center gap-1.5 ${
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
                                                                className="h-3 w-3 accent-emerald-500"
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
                                                        <div className="flex items-start justify-between mb-3">
                                                            <h3 className="font-semibold text-foreground line-clamp-2 flex-1 pr-2 flex items-center gap-2" title={sub.name || ''}>
                                                                <Lightbulb className="w-4 h-4 text-primary flex-shrink-0" />
                                                                {sub.name}
                                                            </h3>
                                                            <div className="flex items-center gap-2">
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
                    onClose={() => setShowIdeaModal(false)}
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
