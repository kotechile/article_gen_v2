
import * as React from "react"
import { useAuth } from "@/context/auth-context"
import { researchTopicsService } from "@/services/research-topics.service"
import type { ResearchTopic } from "@/types/research"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { Search, Sparkles, Clock, Trash2 } from "lucide-react"
import { useNavigate } from "react-router-dom"
import { motion } from "framer-motion"

export function Research() {
    const { user, isLoading: authLoading } = useAuth()
    const navigate = useNavigate()
    const [topics, setTopics] = React.useState<ResearchTopic[]>([])
    const [loading, setLoading] = React.useState(true)
    const [searchTerm, setSearchTerm] = React.useState("")
    const [error, setError] = React.useState<string | null>(null)

    React.useEffect(() => {
        if (!authLoading && user) {
            loadTopics()
        }
    }, [authLoading, user])

    const loadTopics = async () => {
        try {
            setLoading(true)
            const response = await researchTopicsService.listResearchTopics({
                order_by: 'created_at',
                order_direction: 'desc',
                size: 10
            })
            setTopics(response.items)
            setError(null)
        } catch (err) {
            console.error('Failed to load topics:', err)
            setError('Failed to load research topics')
        } finally {
            setLoading(false)
        }
    }

    const handleCreateTopic = async () => {
        if (!searchTerm.trim()) return

        try {
            const newTopic = await researchTopicsService.createResearchTopic({
                title: searchTerm,
                description: `Research topic: ${searchTerm}`
            })
            setTopics([newTopic, ...topics])
            setSearchTerm("")
        } catch (err) {
            console.error('Failed to create topic:', err)
            setError('Failed to create research topic')
        }
    }

    const handleDeleteTopic = async (e: React.MouseEvent, topicId: string) => {
        e.stopPropagation()
        if (!confirm('Are you sure you want to delete this project? This cannot be undone.')) return

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

    return (
        <div className="min-h-screen bg-slate-950 relative overflow-hidden">
            {/* Radial gradient background */}
            <div className="absolute inset-0 bg-gradient-to-b from-indigo-900/20 to-transparent pointer-events-none" />

            <div className="relative z-10 max-w-7xl mx-auto px-6 py-12">
                {/* Hero Section with Search */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                    className="text-center mb-16"
                >
                    {/* Header */}
                    <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight mb-4 text-white">
                        Discover Your Next{" "}
                        <span className="bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
                            Profitable Niche
                        </span>
                    </h1>
                    <p className="text-lg text-slate-400 mb-12 max-w-2xl mx-auto">
                        Combine SEO data, Google Trends, and affiliate offers in one powerful workflow
                    </p>

                    {/* Premium Search Bar */}
                    <div className="max-w-3xl mx-auto">
                        <div className="relative group">
                            {/* Glow effect on focus */}
                            <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500/20 to-purple-500/20 rounded-2xl blur-xl opacity-0 group-focus-within:opacity-100 transition-opacity duration-300" />

                            <div className="relative flex items-center bg-slate-900/80 backdrop-blur-md border border-white/10 rounded-2xl overflow-hidden focus-within:border-indigo-500/50 transition-all">
                                <Search className="absolute left-6 h-5 w-5 text-slate-500" />
                                <Input
                                    placeholder='e.g., "Sustainable Fashion", "SaaS Marketing", "Web3 Gaming"...'
                                    className="flex-1 pl-14 pr-4 h-16 text-base bg-transparent border-0 text-white placeholder:text-slate-500 focus-visible:ring-0 focus-visible:ring-offset-0"
                                    value={searchTerm}
                                    onChange={(e) => setSearchTerm(e.target.value)}
                                    onKeyDown={(e) => e.key === 'Enter' && handleCreateTopic()}
                                />
                                <Button
                                    className="m-2 h-12 px-6 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl transition-colors"
                                    onClick={handleCreateTopic}
                                    disabled={!searchTerm.trim()}
                                >
                                    <Sparkles className="mr-2 h-4 w-4" />
                                    Start Research
                                </Button>
                            </div>
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

                {/* Recent Projects Section */}
                <div className="space-y-6">
                    <h2 className="text-2xl font-bold text-white tracking-tight">Recent Projects</h2>

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
                    ) : topics.length === 0 ? (
                        <motion.div
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ duration: 0.4 }}
                            className="bg-white/5 backdrop-blur-md border border-white/10 border-dashed rounded-2xl p-12 text-center"
                        >
                            <Sparkles className="h-16 w-16 text-slate-600 mx-auto mb-4" />
                            <h3 className="text-xl font-semibold text-white mb-2">No projects yet</h3>
                            <p className="text-slate-400">
                                Start your first research project by entering a topic above
                            </p>
                        </motion.div>
                    ) : (
                        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                            {topics.map((topic, index) => (
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
                                </motion.div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}
