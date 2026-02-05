
import * as React from "react"
import { useParams, useNavigate } from "react-router-dom"
import { useAuth } from "@/context/auth-context"
import { researchTopicsService } from "@/services/research-topics.service"
import { subtopicsService } from "@/services/subtopics.service"
import type { ResearchTopic, Subtopic } from "@/types/research"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { ArrowLeft, FileText, TrendingUp, DollarSign, Target, Sparkles, RefreshCw } from "lucide-react"
import { motion } from "framer-motion"

export function TopicDetail() {
    const { id } = useParams<{ id: string }>()
    const navigate = useNavigate()
    const { user, isLoading: authLoading } = useAuth()

    // State
    const [topic, setTopic] = React.useState<ResearchTopic | null>(null)
    const [subtopics, setSubtopics] = React.useState<Subtopic[]>([])

    const [loading, setLoading] = React.useState(true)
    const [decomposing, setDecomposing] = React.useState(false)
    const [error, setError] = React.useState<string | null>(null)

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
        try {
            setDecomposing(true)
            // Call the generation endpoint
            const newSubtopics = await subtopicsService.generateSubtopics(id)
            setSubtopics(newSubtopics)
        } catch (err) {
            console.error('Failed to decompose topic:', err)
            setError('Failed to decompose topic. Please try again.')
        } finally {
            setDecomposing(false)
        }
    }

    // Calculate metrics from subtopics
    const metrics = React.useMemo(() => {
        if (subtopics.length === 0) {
            return {
                totalVolume: 0,
                totalOpportunities: 0,
                avgDifficulty: 0,
                potential: 'Low'
            }
        }

        const totalVolume = subtopics.reduce((sum, sub) => sum + (sub.search_volume || 0), 0)
        const totalOpportunities = subtopics.reduce((sum, sub) => sum + (sub.affiliate_offer_count || 0), 0)
        const avgDifficulty = Math.round(
            subtopics.reduce((sum, sub) => sum + (sub.seo_difficulty || 0), 0) / subtopics.length
        )

        // Calculate potential based on viability scores
        const avgViability = subtopics.reduce((sum, sub) => sum + (sub.viability_score || 0), 0) / subtopics.length
        let potential = 'Low'
        if (avgViability >= 70) potential = 'High'
        else if (avgViability >= 40) potential = 'Medium'

        return { totalVolume, totalOpportunities, avgDifficulty, potential }
    }, [subtopics])

    if (authLoading || loading) {
        return (
            <div className="flex items-center justify-center min-h-screen bg-slate-950">
                <Skeleton className="h-12 w-12 rounded-full bg-white/10" />
            </div>
        )
    }

    if (error && !topic) {
        return (
            <div className="flex flex-col items-center justify-center min-h-screen bg-slate-950 text-white">
                <p className="text-red-400 mb-4">{error || "Topic not found"}</p>
                <Button onClick={() => navigate('/research')} variant="outline">
                    Back to Research
                </Button>
            </div>
        )
    }

    return (
        <div className="min-h-screen bg-slate-950 text-white relative overflow-hidden">
            {/* Background gradient */}
            <div className="absolute inset-0 bg-gradient-to-b from-indigo-900/10 to-transparent pointer-events-none" />

            <div className="relative z-10 p-6 md:p-8">
                {/* Header */}
                <div className="max-w-7xl mx-auto mb-8">
                    <div className="flex items-center gap-4 mb-2">
                        <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => navigate('/research')}
                            className="text-slate-400 hover:text-white hover:bg-white/5"
                        >
                            <ArrowLeft className="h-5 w-5" />
                        </Button>
                        <span className="text-sm text-slate-500">Back to Dashboard</span>
                    </div>

                    <div className="flex items-start justify-between">
                        <div>
                            <h1 className="text-3xl md:text-4xl font-bold text-white mb-2">
                                {topic?.title || 'Loading...'}
                            </h1>
                            <div className="flex items-center gap-3">
                                <span className={`px-3 py-1 rounded-full text-xs font-medium uppercase tracking-wide ${topic?.status === 'active'
                                        ? 'bg-green-500/10 text-green-400 border border-green-500/20'
                                        : topic?.status === 'completed'
                                            ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                                            : 'bg-slate-500/10 text-slate-400 border border-slate-500/20'
                                    }`}>
                                    {topic?.status || 'Unknown'}
                                </span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Metrics Dashboard */}
                <div className="max-w-7xl mx-auto mb-8">
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                        {/* Total Volume */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.1 }}
                            className="bg-white/5 backdrop-blur-md border border-white/10 rounded-xl p-6"
                        >
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-sm text-slate-400">Total Volume</span>
                                <FileText className="h-4 w-4 text-slate-500" />
                            </div>
                            <div className="text-2xl font-bold text-white">
                                {metrics.totalVolume.toLocaleString()}
                            </div>
                            <div className="text-xs text-slate-500 mt-1">Monthly Searches</div>
                        </motion.div>

                        {/* Total Opportunities */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.2 }}
                            className="bg-white/5 backdrop-blur-md border border-white/10 rounded-xl p-6"
                        >
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-sm text-slate-400">Total Opportunities</span>
                                <TrendingUp className="h-4 w-4 text-blue-400" />
                            </div>
                            <div className="text-2xl font-bold text-white">
                                {metrics.totalOpportunities}
                            </div>
                            <div className="text-xs text-slate-500 mt-1">Affiliate Offers</div>
                        </motion.div>

                        {/* Avg. Difficulty */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.3 }}
                            className="bg-white/5 backdrop-blur-md border border-white/10 rounded-xl p-6"
                        >
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-sm text-slate-400">Avg. Difficulty</span>
                                <DollarSign className="h-4 w-4 text-yellow-400" />
                            </div>
                            <div className="text-2xl font-bold text-white">
                                {metrics.avgDifficulty}
                            </div>
                            <div className="text-xs text-slate-500 mt-1">Keyword Difficulty</div>
                        </motion.div>

                        {/* Potential */}
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.4 }}
                            className="bg-white/5 backdrop-blur-md border border-white/10 rounded-xl p-6"
                        >
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-sm text-slate-400">Potential</span>
                                <Target className="h-4 w-4 text-green-400" />
                            </div>
                            <div className={`text-2xl font-bold ${metrics.potential === 'High' ? 'text-green-400' :
                                    metrics.potential === 'Medium' ? 'text-yellow-400' :
                                        'text-slate-400'
                                }`}>
                                {metrics.potential}
                            </div>
                            <div className="text-xs text-slate-500 mt-1">Overall Viability</div>
                        </motion.div>
                    </div>
                </div>

                {/* Content Opportunities Section */}
                <div className="max-w-7xl mx-auto">
                    <div className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-6">
                        <div className="flex items-center justify-between mb-6">
                            <h2 className="text-xl font-semibold text-white">Content Opportunities</h2>
                            {subtopics.length > 0 && (
                                <Button
                                    onClick={handleDecompose}
                                    disabled={decomposing}
                                    variant="outline"
                                    size="sm"
                                    className="border-white/10 text-slate-300 hover:bg-white/5"
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
                            )}
                        </div>

                        {subtopics.length === 0 ? (
                            /* Empty State */
                            <motion.div
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                className="py-16 text-center"
                            >
                                <div className="max-w-md mx-auto">
                                    <Sparkles className="h-16 w-16 text-slate-600 mx-auto mb-6" />
                                    <h3 className="text-xl font-semibold text-white mb-3">
                                        No subtopics found
                                    </h3>
                                    <p className="text-slate-400 mb-8 leading-relaxed">
                                        Click "Decompose Topic" to generate ideas and discover content opportunities for this research topic.
                                    </p>
                                    <Button
                                        onClick={handleDecompose}
                                        disabled={decomposing}
                                        size="lg"
                                        className="bg-indigo-600 hover:bg-indigo-700 text-white px-8"
                                    >
                                        {decomposing ? (
                                            <>
                                                <RefreshCw className="mr-2 h-5 w-5 animate-spin" />
                                                Decomposing Topic...
                                            </>
                                        ) : (
                                            <>
                                                <Sparkles className="mr-2 h-5 w-5" />
                                                Decompose Topic
                                            </>
                                        )}
                                    </Button>
                                </div>
                            </motion.div>
                        ) : (
                            /* Subtopics Grid */
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                {subtopics.map((sub, i) => (
                                    <motion.div
                                        key={sub.id || i}
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ delay: i * 0.05 }}
                                        className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-5 cursor-pointer hover:bg-white/10 hover:border-indigo-500/30 transition-all duration-200 group"
                                    >
                                        <div className="flex items-start justify-between mb-3">
                                            <h3 className="font-semibold text-slate-200 line-clamp-2 flex-1 pr-2 group-hover:text-white transition-colors">
                                                {sub.name}
                                            </h3>
                                            <span className={`text-xs px-2 py-1 rounded-full border flex-shrink-0 ${sub.trend_direction === 'up'
                                                    ? 'text-green-400 border-green-400/20 bg-green-400/10'
                                                    : sub.trend_direction === 'down'
                                                        ? 'text-red-400 border-red-400/20 bg-red-400/10'
                                                        : 'text-slate-400 border-slate-600 bg-slate-600/10'
                                                }`}>
                                                {sub.trend_direction?.toUpperCase() || 'N/A'}
                                            </span>
                                        </div>

                                        <div className="grid grid-cols-2 gap-3 text-xs">
                                            <div>
                                                <div className="text-slate-500 mb-1">Volume</div>
                                                <div className="text-slate-200 font-semibold">
                                                    {sub.search_volume?.toLocaleString() || '0'}
                                                </div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 mb-1">CPC</div>
                                                <div className="text-slate-200 font-semibold">
                                                    ${sub.cpc?.toFixed(2) || '0.00'}
                                                </div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 mb-1">Difficulty</div>
                                                <div className={`font-semibold ${(sub.seo_difficulty || 0) > 60 ? 'text-red-400' :
                                                        (sub.seo_difficulty || 0) > 30 ? 'text-yellow-400' :
                                                            'text-green-400'
                                                    }`}>
                                                    {sub.seo_difficulty || '0'}
                                                </div>
                                            </div>
                                            <div>
                                                <div className="text-slate-500 mb-1">Offers</div>
                                                <div className="text-slate-200 font-semibold">
                                                    {sub.affiliate_offer_count || 0}
                                                </div>
                                            </div>
                                        </div>
                                    </motion.div>
                                ))}
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
                            className="p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-center"
                        >
                            {error}
                        </motion.div>
                    </div>
                )}
            </div>
        </div>
    )
}
