
import * as React from "react"
import { useAuth } from "@/context/auth-context"
import { useProject } from "@/context/project-context"
import { researchTopicsService } from "@/services/research-topics.service"
import type { ResearchTopic } from "@/types/research"
import { ResearchTopicStatus } from "@/types/research"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { Search, Clock, Trash2, ArrowUpRight, Loader2 } from "lucide-react"
import { useNavigate } from "react-router-dom"
import { motion } from "framer-motion"

export function Research() {
    const { user, isLoading: authLoading } = useAuth()
    const { projects, activeProject } = useProject()
    const navigate = useNavigate()
    const [topics, setTopics] = React.useState<ResearchTopic[]>([])
    const [loading, setLoading] = React.useState(true)
    const [searchTerm, setSearchTerm] = React.useState("")
    const [error, setError] = React.useState<string | null>(null)
    const [projectFilter, setProjectFilter] = React.useState<string>("")

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
    }, [authLoading, user, projectFilter])

    const loadTopics = async (pageNum: number = 1, append: boolean = false) => {
        try {
            setLoading(true)
            const response = await researchTopicsService.listResearchTopics({
                order_by: 'created_at',
                order_direction: 'desc',
                status: ResearchTopicStatus.ACTIVE,
                page: pageNum,
                size: PAGE_SIZE,
                project_id: projectFilter || undefined,
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

    const handleDeleteTopic = async (e: React.MouseEvent, topicId: string) => {
        e.stopPropagation()
        if (!confirm('Delete this research topic? This cannot be undone.')) return

        try {
            await researchTopicsService.deleteResearchTopic(topicId)
            setTopics(topics.filter(t => t.id !== topicId))
        } catch (err) {
            console.error('Failed to delete topic:', err)
            setError('Failed to delete topic')
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
            <div className="flex items-center justify-center min-h-screen bg-slate-950">
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

    return (
        <div className="min-h-screen bg-slate-950 relative overflow-hidden">
            {/* Radial gradient background */}
            <div className="absolute inset-0 bg-gradient-to-b from-indigo-900/20 to-transparent pointer-events-none" />

            <div className="relative z-10 max-w-7xl mx-auto px-6 py-10">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                    className="mb-10"
                >
                    <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
                        <div>
                            <h1 className="text-3xl md:text-4xl font-semibold tracking-tight text-white">
                                All Research
                            </h1>
                            <p className="mt-2 text-slate-400">
                                Browse research topics across projects, with the category context attached.
                            </p>
                        </div>
                        <Button
                            variant="outline"
                            className="border-white/10 text-slate-300 hover:bg-white/5"
                            onClick={() => navigate('/')}
                        >
                            New Research
                            <ArrowUpRight className="ml-2 h-4 w-4" />
                        </Button>
                    </div>

                    <div className="mt-6 grid gap-3 md:grid-cols-[220px_1fr]">
                        <select
                            className="h-12 rounded-2xl border border-white/10 bg-slate-900/70 px-4 text-sm text-white outline-none focus:border-indigo-500/50"
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

                        <div className="relative">
                            <Search className="absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                            <Input
                                placeholder="Search titles, projects, categories..."
                                className="h-12 pl-11 bg-slate-900/70 border-white/10 text-white placeholder:text-slate-500 rounded-2xl focus-visible:ring-0 focus-visible:ring-offset-0 focus-visible:border-indigo-500/50"
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
                        className="mb-8 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-center"
                    >
                        {error}
                    </motion.div>
                )}

                <div className="space-y-6">
                    <h2 className="text-xl font-semibold text-white tracking-tight">Recent Topics</h2>

                    {loading ? (
                        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                            {[1, 2, 3].map((i) => (
                                <div key={i} className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-8 h-48">
                                    <Skeleton className="h-6 w-3/4 bg-white/10 mb-4" />
                                    <Skeleton className="h-4 w-1/2 bg-white/10 mb-6" />
                                    <Skeleton className="h-4 w-full bg-white/10" />
                                </div>
                            ))}
                        </div>
                    ) : filteredTopics.length === 0 ? (
                        <motion.div
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ duration: 0.4 }}
                            className="bg-white/5 backdrop-blur-md border border-white/10 border-dashed rounded-2xl p-12 text-center"
                        >
                            <div className="h-16 w-16 rounded-3xl border border-white/10 bg-white/5 mx-auto mb-4" />
                            <h3 className="text-xl font-semibold text-white mb-2">No topics yet</h3>
                            <p className="text-slate-400">
                                Create a new research queue from the command center.
                            </p>
                        </motion.div>
                    ) : (
                        <>
                            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                                {filteredTopics.map((topic, index) => (
                                    <motion.div
                                        key={topic.id}
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ duration: 0.4, delay: index * 0.05 }}
                                        onClick={() => navigate(`/research/${topic.id}`)}
                                        className="group relative bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-8 cursor-pointer hover:-translate-y-1 hover:border-indigo-500/50 transition-all duration-300"
                                    >
                                        {/* Header with Status Badge */}
                                        <div className="flex items-start justify-between mb-4">
                                            <h3 className="text-lg font-bold text-white line-clamp-2 pr-4 flex-1">
                                                {topic.title}
                                            </h3>
                                            <div className="flex items-center gap-2 flex-shrink-0">
                                                <span className="px-3 py-1 rounded-full text-xs font-medium bg-green-500/10 text-green-400 border border-green-500/20 uppercase tracking-wide">
                                                    {topic.status}
                                                </span>
                                                <Button
                                                    variant="ghost"
                                                    size="icon"
                                                    className="h-8 w-8 text-slate-500 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
                                                    onClick={(e) => handleDeleteTopic(e, topic.id)}
                                                >
                                                    <Trash2 className="h-4 w-4" />
                                                </Button>
                                            </div>
                                        </div>

                                        {/* Timestamp */}
                                        <div className="flex items-center text-xs text-slate-500 mb-4">
                                            <Clock className="h-3 w-3 mr-1.5 opacity-70" />
                                            {formatDate(topic.created_at)}
                                        </div>

                                        {/* Description */}
                                        <p className="text-sm text-slate-400 line-clamp-2 leading-relaxed">
                                            {topic.description}
                                        </p>

                                        {(topic.project_name || topic.primary_category_name || topic.secondary_category_name) && (
                                            <div className="mt-4 flex flex-wrap gap-2">
                                                {topic.project_name && (
                                                    <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-[11px] text-slate-300">
                                                        {topic.project_name}
                                                    </span>
                                                )}
                                                {(topic.primary_category_name || topic.secondary_category_name) && (
                                                    <span className="rounded-full border border-indigo-500/20 bg-indigo-500/10 px-3 py-1 text-[11px] text-indigo-200">
                                                        {[topic.primary_category_name, topic.secondary_category_name].filter(Boolean).join(' / ')}
                                                    </span>
                                                )}
                                            </div>
                                        )}
                                    </motion.div>
                                ))}
                            </div>

                            {/* Load More Button */}
                            {hasMore && (
                                <div className="flex justify-center mt-8">
                                    <Button
                                        variant="outline"
                                        onClick={handleLoadMore}
                                        disabled={loading}
                                        className="px-8 py-3 bg-white/5 border-white/10 text-white hover:bg-white/10 hover:border-indigo-500/30"
                                    >
                                        {loading ? (
                                            <>
                                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                                Loading...
                                            </>
                                        ) : (
                                            <>Load More Projects</>
                                        )}
                                    </Button>
                                </div>
                            )}
                        </>
                    )}
                </div>
            </div>
        </div>
    )
}
