"use client"

import * as React from "react"
import { useParams, useRouter, useSearchParams } from "next/navigation"
import { IdeaCard } from "@/components/burst/idea-card"
import { FloatingActionBar } from "@/components/burst/floating-action-bar"
import { MasonryGrid } from "@/components/burst/masonry-grid"
import { Button } from "@/components/ui/button"
import { ArrowLeft, Sparkles, Loader2, Plus } from "lucide-react"
import { useAuth } from "@/context/auth-context"
import { researchTopicsService } from "@/lib/services/research-topics.service"
import { contentIdeasService } from "@/lib/services/content-ideas.service"
import { subtopicsService } from "@/lib/services/subtopics.service"
import { ResearchTopic } from "@/types/research"
import { ContentIdea } from "@/types/idea-burst"
import { Skeleton } from "@/components/ui/skeleton"
import { toast } from "sonner"
import { IdeaDetailModal } from "@/components/burst/idea-detail-modal"

export default function IdeaBurstPage() {
    const { id } = useParams() as { id: string }
    const router = useRouter()
    const searchParams = useSearchParams()
    const { user } = useAuth()

    const [topic, setTopic] = React.useState<ResearchTopic | null>(null)
    const [ideas, setIdeas] = React.useState<ContentIdea[]>([])
    const [selectedIds, setSelectedIds] = React.useState<Set<string>>(new Set())
    const [loading, setLoading] = React.useState(true)
    const [generating, setGenerating] = React.useState(false)
    const [selectedSubtopicNames, setSelectedSubtopicNames] = React.useState<string[]>([])
    const [subtopics, setSubtopics] = React.useState<any[]>([])
    const [selectedIdeaForDetail, setSelectedIdeaForDetail] = React.useState<ContentIdea | null>(null)

    React.useEffect(() => {
        if (id && user) {
            loadData()
        }
    }, [id, user])

    const loadData = async () => {
        try {
            setLoading(true)
            const [topicData, ideasData, subtopicsData] = await Promise.all([
                researchTopicsService.getResearchTopic(id),
                contentIdeasService.getContentIdeas(id, user!.id),
                subtopicsService.getSubtopics(id)
            ])
            setTopic(topicData)
            setIdeas(Array.isArray(ideasData) ? ideasData : [])
            setSubtopics(subtopicsData)

            // Handle selected subtopics from query params
            const idsParam = searchParams.get('ids')
            if (idsParam) {
                const ids = idsParam.split(',')
                const selectedNames = subtopicsData
                    .filter(s => ids.includes(s.id))
                    .map(s => s.name)

                if (selectedNames.length > 0) {
                    setSelectedSubtopicNames(selectedNames)
                    toast.info(`Loaded ${selectedNames.length} selected subtopics for generation`)
                }
            }
        } catch (error) {
            console.error("Failed to load burst data:", error)
            toast.error("Failed to load ideas")
        } finally {
            setLoading(false)
        }
    }

    const handleGenerate = async () => {
        console.log("Handle Generate Clicked", {
            hasTopic: !!topic,
            hasUser: !!user,
            subtopicsCount: subtopics.length,
            selectedCount: selectedSubtopicNames.length
        });

        if (!topic) {
            toast.error("Topic data missing");
            return;
        }
        if (!user) {
            toast.error("User not authenticated");
            return;
        }

        try {
            setGenerating(true)

            // Determine target subtopics
            const targets = selectedSubtopicNames.length > 0
                ? subtopics.filter(s => selectedSubtopicNames.includes(s.name))
                : subtopics.slice(0, 3); // Default to first 3 if none selected

            toast.message("Generating magical ideas...", {
                description: `Analyzing ${targets.length} subtopics...`,
            })

            let newsIdeasCount = 0;

            // Iterate and generate per subtopic
            for (const subtopic of targets) {
                const result = await contentIdeasService.generateBurst({
                    topicId: topic.id,
                    subtopicName: subtopic.name,
                    keywords: subtopic.keywords || [],
                    affiliateOffers: [], // TODO: Fetch from context if available
                    userId: user.id
                })

                if (result.success) {
                    const newIdeas = [...result.blog_ideas, ...result.software_ideas];
                    setIdeas(prev => {
                        const existingIds = new Set(prev.map(i => i.id));
                        const uniqueNewIdeas = newIdeas.filter(i => !existingIds.has(i.id));
                        return [...uniqueNewIdeas, ...prev];
                    })
                    newsIdeasCount += newIdeas.length;
                }
            }

            if (newsIdeasCount > 0) {
                toast.success(`Generated ${newsIdeasCount} new ideas!`)
            } else {
                toast.warning("No ideas generated. Try different subtopics.")
            }

        } catch (error) {
            console.error("Generation failed:", error)
            toast.error("Generation failed. Please try again.")
        } finally {
            setGenerating(false)
        }
    }

    // Toggle selection
    const toggleSelection = (id: string) => {
        const newSet = new Set(selectedIds)
        if (newSet.has(id)) {
            newSet.delete(id)
        } else {
            newSet.add(id)
        }
        setSelectedIds(newSet)
    }

    const handleClear = () => {
        setSelectedIds(new Set())
    }

    // Toggle Select All
    const toggleSelectAll = () => {
        if (selectedIds.size === ideas.length) {
            setSelectedIds(new Set())
        } else {
            setSelectedIds(new Set(ideas.map(i => i.id)))
        }
    }

    const handlePublish = async () => {
        if (!user || selectedIds.size === 0) return

        try {
            const success = await contentIdeasService.publishContentIdeas(
                Array.from(selectedIds),
                user.id
            )

            if (success) {
                toast.success("Publication Successful!", {
                    description: `Successfully published ${selectedIds.size} content ideas.`
                })
                setSelectedIds(new Set())
            } else {
                toast.error("Failed to publish ideas. Please try again.")
            }
        } catch (error) {
            console.error("Publish failed:", error)
            toast.error("Failed to publish ideas")
        }
    }

    if (loading) {
        return (
            <div className="space-y-6 animate-pulse">
                <Skeleton className="h-12 w-1/3" />
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {[1, 2, 3, 4, 5, 6].map(i => <Skeleton key={i} className="h-48 rounded-xl" />)}
                </div>
            </div>
        )
    }

    const handleDeleteIdea = async (id: string, e: React.MouseEvent) => {
        e.stopPropagation();
        if (!user) return;

        try {
            const success = await contentIdeasService.deleteContentIdea(id, user.id);
            if (success) {
                setIdeas(prev => prev.filter(idea => idea.id !== id));
                toast.success("Idea deleted");
            } else {
                toast.error("Failed to delete idea");
            }
        } catch (error) {
            console.error("Delete failed:", error);
            toast.error("Failed to delete idea");
        }
    };

    const isAllSelected = ideas.length > 0 && selectedIds.size === ideas.length;

    return (
        <div className="space-y-6 pb-24">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8 animate-in slide-in-from-top duration-500">
                <div className="flex items-center gap-4">
                    <Button variant="ghost" size="icon" onClick={() => router.back()} className="rounded-full">
                        <ArrowLeft className="h-4 w-4" />
                    </Button>
                    <div>
                        <h1 className="text-3xl font-bold tracking-tight">Idea Burst</h1>
                        <p className="text-muted-foreground">
                            Topic: <span className="font-semibold text-primary">{topic?.title}</span>
                            {selectedSubtopicNames.length > 0 && (
                                <span className="ml-2 text-xs bg-primary/10 text-primary px-2 py-0.5 rounded-full">
                                    Focus: {selectedSubtopicNames.length} subtopics
                                </span>
                            )}
                        </p>
                    </div>
                </div>

                <div className="flex items-center gap-3">
                    {ideas.length > 0 && (
                        <Button
                            variant="outline"
                            onClick={toggleSelectAll}
                            className="hidden md:flex"
                        >
                            {isAllSelected ? "Deselect All" : "Select All"}
                        </Button>
                    )}

                    <Button
                        onClick={handleGenerate}
                        disabled={generating}
                        className="bg-zinc-900 border border-zinc-700 text-white hover:bg-zinc-800 shadow-xl group transition-all"
                    >
                        {generating ? (
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        ) : (
                            <Plus className="mr-2 h-4 w-4 group-hover:rotate-90 transition-transform" />
                        )}
                        Generate New Ideas
                    </Button>
                </div>
            </div>

            {ideas.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-32 text-center bg-zinc-50 dark:bg-zinc-900/40 rounded-3xl border border-dashed border-zinc-200 dark:border-zinc-800">
                    <div className="bg-primary/10 p-4 rounded-full mb-6">
                        <Sparkles className="h-12 w-12 text-primary" />
                    </div>
                    <h2 className="text-2xl font-bold mb-2">No ideas yet</h2>
                    <p className="text-muted-foreground max-w-sm mb-8">
                        Click the button above to generate a burst of content and software ideas for "{topic?.title}".
                    </p>
                    <Button size="lg" onClick={handleGenerate} disabled={generating}>
                        Start First Generation
                    </Button>
                </div>
            ) : (
                <MasonryGrid>
                    {ideas.map((idea) => (
                        <IdeaCard
                            key={idea.id}
                            idea={idea}
                            selected={selectedIds.has(idea.id)}
                            onSelect={(checked) => toggleSelection(idea.id)}
                            onDelete={handleDeleteIdea}
                        />
                    ))}
                </MasonryGrid>
            )}

            <IdeaDetailModal
                idea={selectedIdeaForDetail}
                isOpen={!!selectedIdeaForDetail}
                onClose={() => setSelectedIdeaForDetail(null)}
            />

            <FloatingActionBar
                selectedCount={selectedIds.size}
                onClear={handleClear}
                onPublish={handlePublish}
            />
        </div>
    )
}
