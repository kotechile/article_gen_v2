"use client"

import * as React from "react"
import { useAuth } from "@/context/auth-context"
import { researchTopicsService } from "@/lib/services/research-topics.service"
import type { ResearchTopic } from "@/types/research"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { Clock, Plus, Sparkles } from "lucide-react"
import { useRouter } from "next/navigation"
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

export default function ResearchPage() {
    const { user, isLoading: authLoading } = useAuth()
    const router = useRouter()
    const [topics, setTopics] = React.useState<ResearchTopic[]>([])
    const [loading, setLoading] = React.useState(true)
    const [error, setError] = React.useState<string | null>(null)

    // New topic state
    const [isDialogOpen, setIsDialogOpen] = React.useState(false)
    const [newTopicTitle, setNewTopicTitle] = React.useState("")
    const [creating, setCreating] = React.useState(false)

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
                size: 50 // Load more for the list view
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
        if (!newTopicTitle.trim()) return

        try {
            setCreating(true)
            const newTopic = await researchTopicsService.createResearchTopic({
                title: newTopicTitle,
                description: `Research topic: ${newTopicTitle}`
            })
            setTopics([newTopic, ...topics])
            setNewTopicTitle("")
            setIsDialogOpen(false)
            // Optional: Navigate to the new topic immediately
            router.push(`/research/${newTopic.id}`)
        } catch (err) {
            console.error('Failed to create topic:', err)
            setError('Failed to create research topic')
        } finally {
            setCreating(false)
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

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'completed': return 'default'
            case 'active': return 'secondary'
            default: return 'outline'
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
            <div className="flex items-center justify-center min-h-[60vh]">
                <Skeleton className="h-12 w-12 rounded-full" />
            </div>
        )
    }

    return (
        <div className="space-y-8 p-6">
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight">Research Projects</h1>
                    <p className="text-muted-foreground mt-2">
                        Manage your ongoing keyword research and content analysis projects.
                    </p>
                </div>
                <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
                    <DialogTrigger asChild>
                        <Button>
                            <Plus className="mr-2 h-4 w-4" /> New Project
                        </Button>
                    </DialogTrigger>
                    <DialogContent>
                        <DialogHeader>
                            <DialogTitle>Start New Research</DialogTitle>
                            <DialogDescription>
                                Enter a broad topic to begin your research process.
                            </DialogDescription>
                        </DialogHeader>
                        <div className="grid gap-4 py-4">
                            <div className="grid gap-2">
                                <Label htmlFor="topic">Topic</Label>
                                <Input
                                    id="topic"
                                    placeholder='e.g. "Sustainable Living"'
                                    value={newTopicTitle}
                                    onChange={(e) => setNewTopicTitle(e.target.value)}
                                    onKeyDown={(e) => e.key === 'Enter' && handleCreateTopic()}
                                />
                            </div>
                        </div>
                        <DialogFooter>
                            <Button variant="outline" onClick={() => setIsDialogOpen(false)}>Cancel</Button>
                            <Button onClick={handleCreateTopic} disabled={!newTopicTitle.trim() || creating}>
                                {creating ? "Creating..." : "Start Research"}
                            </Button>
                        </DialogFooter>
                    </DialogContent>
                </Dialog>
            </div>

            {error && (
                <div className="p-4 rounded-lg bg-red-50 dark:bg-red-950 text-red-600 dark:text-red-400">
                    {error}
                </div>
            )}

            {loading ? (
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                    {[1, 2, 3, 4, 5, 6].map((i) => (
                        <Card key={i}>
                            <CardHeader>
                                <Skeleton className="h-6 w-3/4" />
                                <Skeleton className="h-4 w-1/2 mt-2" />
                            </CardHeader>
                            <CardContent>
                                <Skeleton className="h-4 w-full" />
                            </CardContent>
                        </Card>
                    ))}
                </div>
            ) : topics.length === 0 ? (
                <Card className="border-dashed">
                    <CardContent className="flex flex-col items-center justify-center py-12 text-center">
                        <Sparkles className="h-12 w-12 text-muted-foreground mb-4" />
                        <h3 className="text-lg font-semibold mb-2">No projects yet</h3>
                        <p className="text-sm text-muted-foreground mb-4">
                            Start your first research project to discover profitable niches.
                        </p>
                        <Button onClick={() => setIsDialogOpen(true)}>
                            <Plus className="mr-2 h-4 w-4" /> Create Project
                        </Button>
                    </CardContent>
                </Card>
            ) : (
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                    {topics.map((topic) => (
                        <Card
                            key={topic.id}
                            className="hover:shadow-md transition-shadow cursor-pointer group relative"
                            onClick={() => router.push(`/research/${topic.id}`)}
                        >
                            <CardHeader className="pb-3">
                                <div className="flex items-start justify-between">
                                    <CardTitle className="text-lg line-clamp-1 pr-8">{topic.title}</CardTitle>
                                    <div className="flex items-center gap-2">
                                        <Badge variant={getStatusColor(topic.status)} className="capitalize">
                                            {topic.status}
                                        </Badge>
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            className="h-8 w-8 text-muted-foreground hover:text-destructive opacity-0 group-hover:opacity-100 transition-opacity"
                                            onClick={(e) => handleDeleteTopic(e, topic.id)}
                                        >
                                            <span className="sr-only">Delete</span>
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
                                <CardDescription className="flex items-center text-xs">
                                    <Clock className="h-3 w-3 mr-1" />
                                    {formatDate(topic.created_at)}
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <p className="text-sm text-muted-foreground line-clamp-2">
                                    {topic.description}
                                </p>
                            </CardContent>
                        </Card>
                    ))}
                </div>
            )}
        </div>
    )
}
