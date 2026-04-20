
import * as React from "react"
import { useAuth } from "@/context/auth-context"
import { useProject } from "@/context/project-context"
import { researchTopicsService } from "@/services/research-topics.service"
import type { ResearchTopic } from "@/types/research"
import { ResearchTopicStatus } from "@/types/research"
import { supabase } from "@/lib/supabase"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { Search, Clock, Trash2, ArrowUpRight, Loader2 } from "lucide-react"
import { useNavigate } from "react-router-dom"
import { motion } from "framer-motion"

type ProjectCategory = {
    id: string
    name: string
    level: number
    parent_category_id: string | null
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
                status: ResearchTopicStatus.ACTIVE,
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
                                All Research
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
                            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                                {filteredTopics.map((topic, index) => (
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
                                            <h3 className="text-lg font-bold text-foreground line-clamp-2 pr-32">
                                                {topic.title}
                                            </h3>
                                            <div className="absolute right-6 top-6 flex flex-col items-end gap-2">
                                                {(() => {
                                                    const subtopicsCount = Number(topic.subtopics_count || 0)
                                                    const researchedSubtopicsCount = Number(topic.researched_subtopics_count || 0)
                                                    const ideasCount = Number(topic.content_ideas_count || 0)
                                                    const inLibraryCount = Number(topic.in_library_count || 0)
                                                    const isFullyResearched = Boolean(topic.all_subtopics_researched)
                                                    const hasAnyProgress = subtopicsCount > 0 || ideasCount > 0 || inLibraryCount > 0 || Boolean(topic.has_underlying_data)
                                                    const isNotStarted = !hasAnyProgress && !isFullyResearched
                                                    const chipBase =
                                                        "inline-flex flex-shrink-0 items-center whitespace-nowrap rounded-full border px-2 py-1 text-xs font-medium leading-none"

                                                    return (
                                                        <div className="flex flex-col items-end gap-1">
                                                            {subtopicsCount > 0 && (
                                                                <span className={`${chipBase} border-indigo-500/20 bg-indigo-500/10 text-indigo-300`}>
                                                                    {subtopicsCount} Sub-Topics
                                                                </span>
                                                            )}
                                                            {isFullyResearched && (
                                                                <span className={`${chipBase} border-emerald-500/30 bg-emerald-500/10 text-emerald-300`}>
                                                                    Researched
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
                                                <Button
                                                    variant="ghost"
                                                    size="icon"
                                                    className="h-8 w-8 text-muted-foreground hover:text-destructive opacity-0 group-hover:opacity-100 transition-opacity"
                                                    onClick={(e) => handleDeleteTopic(e, topic.id)}
                                                >
                                                    <Trash2 className="h-4 w-4" />
                                                </Button>
                                            </div>
                                        </div>

                                        {/* Timestamp */}
                                        <div className="flex items-center text-xs text-muted-foreground mb-4">
                                            <Clock className="h-3 w-3 mr-1.5 opacity-70" />
                                            {formatDate(topic.created_at)}
                                        </div>

                                        {/* Description */}
                                        <p className="text-sm text-muted-foreground line-clamp-2 leading-relaxed">
                                            {topic.description}
                                        </p>

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
                                        className="px-8 py-3 bg-muted/50 border-border text-foreground hover:bg-muted hover:border-ring/50"
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
