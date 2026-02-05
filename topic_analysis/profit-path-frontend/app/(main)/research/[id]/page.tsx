"use client"

import * as React from "react"
import { useParams, useRouter } from "next/navigation"
import { useAuth } from "@/context/auth-context"
import { researchTopicsService } from "@/lib/services/research-topics.service"
import { contentIdeasService } from "@/lib/services/content-ideas.service"
import { keywordsService } from "@/lib/services/keywords.service"
import { ResearchTopic, Subtopic } from "@/types/research"
import { ContentIdea } from "@/types/idea-burst"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import {
    ArrowLeft,
    Sparkles,
    BarChart3,
    Target,
    TrendingUp,
    FileText,
    Activity,
    DollarSign,
    Loader2,
    ListPlus,
    Zap
} from "lucide-react"

import { MarketIntelligenceDrawer } from "@/components/dashboard/market-intelligence-drawer"
import { affiliateResearchService, AffiliateProgram, AffiliateAnalysis } from "@/lib/services/affiliate-research.service"
import { subtopicsService } from "@/lib/services/subtopics.service"
import { SubtopicsTable } from "@/components/research/subtopics-table"
import { SubtopicsGrid } from "@/components/research/subtopics-grid"
import { AllKeywordsTable } from "@/components/research/all-keywords-table"
import { KeywordExpansionPanel } from "@/components/research/keyword-expansion-panel"
import { ContentTopicsPanel } from "@/components/research/content-topics-panel"

export default function ResearchDetailPage() {
    const { id } = useParams() as { id: string }
    const { user } = useAuth()
    const router = useRouter()

    const [topic, setTopic] = React.useState<ResearchTopic | null>(null)
    const [ideas, setIdeas] = React.useState<ContentIdea[]>([])
    const [programs, setPrograms] = React.useState<AffiliateProgram[]>([])
    const [analysis, setAnalysis] = React.useState<AffiliateAnalysis | null>(null)
    const [subtopics, setSubtopics] = React.useState<Subtopic[]>([])
    const [loading, setLoading] = React.useState(true)
    const [error, setError] = React.useState<string | null>(null)
    const [drawerOpen, setDrawerOpen] = React.useState(false)
    const [generating, setGenerating] = React.useState(false)
    const [expanding, setExpanding] = React.useState(false)
    const [enriching, setEnriching] = React.useState(false)
    const [keywordsUpdated, setKeywordsUpdated] = React.useState<number>(0)
    const [viewMode, setViewMode] = React.useState<"grid" | "table">("grid")
    const [selectedSubtopics, setSelectedSubtopics] = React.useState<Set<string>>(new Set())

    const toggleSelection = (id: string) => {
        const newSelection = new Set(selectedSubtopics)
        if (newSelection.has(id)) {
            newSelection.delete(id)
        } else {
            newSelection.add(id)
        }
        setSelectedSubtopics(newSelection)
    }

    const toggleAll = () => {
        if (selectedSubtopics.size === subtopics.length) {
            setSelectedSubtopics(new Set())
        } else {
            setSelectedSubtopics(new Set(subtopics.map(s => s.id)))
        }
    }

    const handleGenerateSubtopics = async () => {
        try {
            setGenerating(true)
            const newSubtopics = await subtopicsService.generateSubtopics(id)
            setSubtopics(newSubtopics)
        } catch (err) {
            console.error("Failed to generate subtopics:", err)
            // Optional: show toast error
        } finally {
            setGenerating(false)
        }
    }

    React.useEffect(() => {
        if (id && user) {
            loadTopicData()
        }
    }, [id, user])

    const stats = React.useMemo(() => {
        if (!subtopics.length) return { volume: 0, difficulty: 0, offers: 0, potential: "Low" };
        const vol = subtopics.reduce((acc, s) => acc + (s.search_volume || 0), 0);
        const diff = subtopics.reduce((acc, s) => acc + (s.seo_difficulty || 0), 0) / subtopics.length;
        const offers = subtopics.reduce((acc, s) => {
            const count = s.monetization_data?.offers?.length ?? s.affiliate_offer_count ?? 0;
            return acc + count;
        }, 0);

        let potential = "Low";
        if (vol > 10000 && offers > 5) potential = "High";
        else if (vol > 1000 || offers > 0) potential = "Medium";

        return { volume: vol, difficulty: Math.round(diff), offers, potential };
    }, [subtopics]);

    const handleDelete = async (subtopicId: string) => {
        try {
            await subtopicsService.deleteSubtopic(id, subtopicId);
            setSubtopics(prev => prev.filter(s => s.id !== subtopicId));
        } catch (err) {
            console.error("Failed to delete subtopic", err);
        }
    }

    const handleEnrichSubtopic = async (subtopicId: string) => {
        try {
            const enrichedSubtopic = await subtopicsService.enrichSubtopic(id, subtopicId)
            setSubtopics(prev => prev.map(s => s.id === subtopicId ? enrichedSubtopic : s))
        } catch (err) {
            console.error("Failed to enrich subtopic:", err)
        }
    }

    const handleVerify = async (subtopicId: string) => {
        // We now use manual enrichment as the primary "verification" step
        await handleEnrichSubtopic(subtopicId)
    }

    const loadTopicData = async () => {
        try {
            setLoading(true)
            const [topicData, ideasData, subtopicsData] = await Promise.all([
                researchTopicsService.getResearchTopic(id),
                contentIdeasService.getContentIdeas(id, user!.id),
                subtopicsService.getSubtopics(id)
            ])
            setTopic(topicData)
            setIdeas(ideasData)
            setSubtopics(subtopicsData)
            setLoading(false)

            // Search for affiliate programs based on topic title (Non-blocking)
            if (topicData.title) {
                affiliateResearchService.searchAffiliatePrograms({
                    search_term: topicData.title,
                    user_id: user!.id
                }).then(affiliateResult => {
                    setPrograms(affiliateResult.programs)
                    if (affiliateResult.analysis) {
                        setAnalysis(affiliateResult.analysis)
                    }
                }).catch(err => {
                    console.error("Background affiliate search failed:", err)
                })
            }
        } catch (err) {
            console.error("Failed to load research data:", err)
            setError("Failed to load research details")
            setLoading(false)
        }
    }

    if (loading) {
        return (
            <div className="space-y-8 animate-in fade-in duration-500">
                <Skeleton className="h-8 w-48" />
                <div className="grid gap-6 md:grid-cols-3">
                    <Skeleton className="h-32 rounded-xl" />
                    <Skeleton className="h-32 rounded-xl" />
                    <Skeleton className="h-32 rounded-xl" />
                </div>
                <Skeleton className="h-[400px] rounded-xl" />
            </div>
        )
    }

    if (error || !topic) {
        return (
            <div className="flex flex-col items-center justify-center min-h-[60vh] text-center">
                <div className="bg-red-50 dark:bg-red-950/20 p-4 rounded-full mb-4">
                    <Activity className="h-8 w-8 text-red-500" />
                </div>
                <h2 className="text-2xl font-bold mb-2">Something went wrong</h2>
                <p className="text-muted-foreground mb-6 max-w-md">{error || "Could not find this research topic"}</p>
                <Button onClick={() => router.push("/")}>Back to Dashboard</Button>
            </div>
        )
    }

    const handleEnrichment = async () => {
        try {
            setEnriching(true)
            const enrichedSubtopics = await subtopicsService.enrichSubtopics(id)
            setSubtopics(enrichedSubtopics)
        } catch (err) {
            console.error("Failed to enrich subtopics:", err)
        } finally {
            setEnriching(false)
        }
    }

    const handleExpandAllKeywords = async () => {
        try {
            setExpanding(true)
            await keywordsService.expandAllKeywords(id)
            setKeywordsUpdated(Date.now())

            // Re-fetch subtopics to update keyword counts in the table
            const updatedSubtopics = await subtopicsService.getSubtopics(id)
            setSubtopics(updatedSubtopics)

        } catch (err) {
            console.error("Failed to expand all keywords:", err)
        } finally {
            setExpanding(false)
        }
    }

    const isBusy = generating || expanding || enriching;

    return (
        <div className="space-y-8 pb-12 animate-in slide-in-from-bottom-4 duration-700">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div className="space-y-1">
                    <Button
                        variant="ghost"
                        size="sm"
                        className="mb-2 -ml-2 text-muted-foreground hover:text-foreground"
                        onClick={() => router.push("/")}
                    >
                        <ArrowLeft className="mr-2 h-4 w-4" />
                        Back to Dashboard
                    </Button>
                    <div className="flex items-center gap-3">
                        <h1 className="text-3xl font-bold tracking-tight">{topic.title}</h1>
                        <Badge variant="outline" className="capitalize">{topic.status}</Badge>
                        <Button
                            variant="ghost"
                            size="icon"
                            className="text-muted-foreground hover:text-destructive transition-colors ml-2"
                            onClick={async () => {
                                if (confirm("Are you sure you want to delete this topic? This cannot be undone.")) {
                                    try {
                                        await researchTopicsService.deleteResearchTopic(id);
                                        router.push("/");
                                    } catch (e) {
                                        console.error("Failed to delete topic:", e);
                                    }
                                }
                            }}
                        >
                            <span className="sr-only">Delete Topic</span>
                            <svg
                                xmlns="http://www.w3.org/2000/svg"
                                width="24"
                                height="24"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                className="h-4 w-4"
                            >
                                <path d="M3 6h18" />
                                <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
                                <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
                            </svg>
                        </Button>
                    </div>
                </div>

                <div className="flex gap-3">
                    {/* Always allow decomposing/re-decomposing */}
                    <Button
                        variant={subtopics.length === 0 ? "default" : "secondary"}
                        onClick={handleGenerateSubtopics}
                        disabled={isBusy}
                    >
                        {generating ? (
                            <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Decomposing...
                            </>
                        ) : (
                            <>
                                Decompose Topic
                                <Zap className="ml-2 h-4 w-4" />
                            </>
                        )}
                    </Button>

                    {subtopics.length > 0 && (
                        <Button
                            size="lg"
                            className="bg-emerald-600 hover:bg-emerald-700 text-white shadow-lg shadow-emerald-500/20 group transition-all"
                            onClick={handleEnrichment}
                            disabled={isBusy}
                        >
                            {enriching ? (
                                <>
                                    <Activity className="mr-2 h-4 w-4 animate-spin" />
                                    Finding Offers...
                                </>
                            ) : (
                                <>
                                    Find Affiliate Offers
                                    <DollarSign className="ml-2 h-4 w-4" />
                                </>
                            )}
                        </Button>
                    )}
                </div>
            </div>

            {/* Quick Stats */}
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                <Card className="border-zinc-200 dark:border-zinc-800 bg-background/50 backdrop-blur-sm">
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Total Volume</CardTitle>
                        <FileText className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{stats.volume.toLocaleString()}</div>
                        <p className="text-xs text-muted-foreground mt-1">Monthly Searches</p>
                    </CardContent>
                </Card>
                <Card className="border-zinc-200 dark:border-zinc-800 bg-background/50 backdrop-blur-sm">
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Total Opportunities</CardTitle>
                        <TrendingUp className="h-4 w-4 text-blue-500" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold font-mono">{stats.offers}</div>
                        <p className="text-xs text-muted-foreground mt-1">Affiliate Offers</p>
                    </CardContent>
                </Card>
                <Card className="border-zinc-200 dark:border-zinc-800 bg-background/50 backdrop-blur-sm">
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Avg. Difficulty</CardTitle>
                        <Target className="h-4 w-4 text-amber-500" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold font-mono">{stats.difficulty}</div>
                        <p className="text-xs text-muted-foreground mt-1">Keyword Difficulty</p>
                    </CardContent>
                </Card>
                <Card className="border-zinc-200 dark:border-zinc-800 bg-background/50 backdrop-blur-sm">
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Potential</CardTitle>
                        <BarChart3 className="h-4 w-4 text-green-500" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold font-mono">{stats.potential}</div>
                        <p className="text-xs text-muted-foreground mt-1">Overall Viability</p>
                    </CardContent>
                </Card>
            </div>

            {/* Unified Bulk Action Toolbar */}
            {selectedSubtopics.size > 0 && (
                <div className="flex items-center gap-2 p-3 bg-purple-50 dark:bg-purple-900/10 border border-purple-100 dark:border-purple-800 rounded-xl animate-in fade-in slide-in-from-top-2 shadow-sm">
                    <div className="bg-purple-100 dark:bg-purple-800/50 p-2 rounded-full">
                        <Sparkles className="h-4 w-4 text-purple-600 dark:text-purple-400" />
                    </div>
                    <div>
                        <span className="text-sm font-medium block">{selectedSubtopics.size} subtopics selected</span>
                        <span className="text-xs text-muted-foreground">Ready to generate content ideas</span>
                    </div>
                    <Button
                        size="sm"
                        className="ml-auto gap-2 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 text-white border-0 shadow-md"
                        onClick={() => router.push(`/research/${id}/burst?ids=${Array.from(selectedSubtopics).join(',')}`)}
                    >
                        Generate Ideas
                        <Sparkles className="h-3 w-3" />
                    </Button>
                </div>
            )}

            {/* Subtopics Grid/Table */}
            <Card className="border-zinc-200 dark:border-zinc-800">
                <CardHeader>
                    <div className="flex items-center justify-between">
                        <div>
                            <CardTitle>Content Opportunities</CardTitle>
                            {subtopics.length === 0 ? (
                                <div className="mt-2">
                                    <Button
                                        variant="outline"
                                        className="gap-2"
                                        onClick={handleGenerateSubtopics}
                                        disabled={isBusy}
                                    >
                                        Decompose Topic
                                        <Zap className="h-4 w-4" />
                                    </Button>
                                </div>
                            ) : (
                                <CardDescription>{`${subtopics.length} opportunities found`}</CardDescription>
                            )}
                        </div>
                        {subtopics.length > 0 && (
                            <div className="flex items-center gap-2">
                                {/* Grid View Select All Trigger - optional if we want it explicit in header */}
                                {viewMode === "grid" && (
                                    <Button
                                        variant="outline"
                                        size="sm"
                                        onClick={toggleAll}
                                        className="mr-2"
                                    >
                                        {selectedSubtopics.size === subtopics.length ? "Deselect All" : "Select All"}
                                    </Button>
                                )}

                                <div className="flex items-center gap-2 border rounded-md p-1">
                                    <Button
                                        variant={viewMode === "grid" ? "secondary" : "ghost"}
                                        size="sm"
                                        onClick={() => setViewMode("grid")}
                                        className="px-3"
                                    >
                                        <svg
                                            xmlns="http://www.w3.org/2000/svg"
                                            width="16"
                                            height="16"
                                            viewBox="0 0 24 24"
                                            fill="none"
                                            stroke="currentColor"
                                            strokeWidth="2"
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            className="mr-1.5"
                                        >
                                            <rect width="7" height="7" x="3" y="3" rx="1" />
                                            <rect width="7" height="7" x="14" y="3" rx="1" />
                                            <rect width="7" height="7" x="14" y="14" rx="1" />
                                            <rect width="7" height="7" x="3" y="14" rx="1" />
                                        </svg>
                                        Grid
                                    </Button>
                                    <Button
                                        variant={viewMode === "table" ? "secondary" : "ghost"}
                                        size="sm"
                                        onClick={() => setViewMode("table")}
                                        className="px-3"
                                    >
                                        <svg
                                            xmlns="http://www.w3.org/2000/svg"
                                            width="16"
                                            height="16"
                                            viewBox="0 0 24 24"
                                            fill="none"
                                            stroke="currentColor"
                                            strokeWidth="2"
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                            className="mr-1.5"
                                        >
                                            <path d="M12 3v18" />
                                            <rect width="18" height="18" x="3" y="3" rx="2" />
                                            <path d="M3 9h18" />
                                            <path d="M3 15h18" />
                                        </svg>
                                        Table
                                    </Button>
                                </div>
                            </div>
                        )}
                    </div>
                </CardHeader>
                <CardContent>
                    {viewMode === "grid" ? (
                        <SubtopicsGrid
                            subtopics={subtopics}
                            selectedSubtopics={selectedSubtopics}
                            onToggle={toggleSelection}
                        />
                    ) : (
                        <SubtopicsTable
                            subtopics={subtopics}
                            topicId={id}
                            selectedSubtopics={selectedSubtopics}
                            onToggle={toggleSelection}
                            onToggleAll={toggleAll}
                            onDelete={handleDelete}
                            onEnrich={handleEnrichSubtopic}
                        />
                    )}
                </CardContent>
            </Card>

            {/* Content Ideas Preview */}
            {ideas.length > 0 && (
                <Card className="border-zinc-200 dark:border-zinc-800">
                    <CardHeader>
                        <div className="flex items-center justify-between">
                            <div>
                                <CardTitle>Content Ideas</CardTitle>
                                <CardDescription>{`${ideas.length} ideas generated`}</CardDescription>
                            </div>
                            <Button
                                onClick={() => router.push(`/research/${id}/burst`)}
                                className="gap-2"
                            >
                                View All & Publish
                                <Sparkles className="h-4 w-4" />
                            </Button>
                        </div>
                    </CardHeader>
                    <CardContent>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {ideas.slice(0, 6).map((idea) => (
                                <div
                                    key={idea.id}
                                    className="p-4 border rounded-lg hover:border-primary/50 transition-colors cursor-pointer"
                                    onClick={() => router.push(`/research/${id}/burst`)}
                                >
                                    <div className="flex items-start justify-between mb-2">
                                        <Badge variant="outline" className="text-xs">
                                            {idea.content_type === 'blog' ? 'Blog Post' : 'Software'}
                                        </Badge>
                                    </div>
                                    <h3 className="font-semibold text-sm mb-2 line-clamp-2">
                                        {idea.title}
                                    </h3>
                                    <p className="text-xs text-muted-foreground line-clamp-2">
                                        {idea.description}
                                    </p>
                                    {idea.keywords && idea.keywords.length > 0 && (
                                        <div className="mt-2 flex flex-wrap gap-1">
                                            {idea.keywords.slice(0, 3).map((kw, i) => (
                                                <span
                                                    key={i}
                                                    className="text-xs bg-muted px-2 py-0.5 rounded"
                                                >
                                                    {kw}
                                                </span>
                                            ))}
                                            {idea.keywords.length > 3 && (
                                                <span className="text-xs text-muted-foreground">
                                                    +{idea.keywords.length - 3}
                                                </span>
                                            )}
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                        {ideas.length > 6 && (
                            <div className="mt-4 text-center">
                                <Button
                                    variant="outline"
                                    onClick={() => router.push(`/research/${id}/burst`)}
                                    className="gap-2"
                                >
                                    View {ideas.length - 6} More Ideas
                                    <Sparkles className="h-4 w-4" />
                                </Button>
                            </div>
                        )}
                    </CardContent>
                </Card>
            )}


            <AllKeywordsTable
                topicId={id as string}
                refreshTrigger={keywordsUpdated}
            />

            {/* Content Topics */}
            {subtopics.length > 0 && (
                <ContentTopicsPanel topicId={id} />
            )}

            <MarketIntelligenceDrawer
                isOpen={drawerOpen}
                onClose={() => setDrawerOpen(false)}
                keyword={topic.title}
                programs={programs}
            />
        </div>
    )
}
