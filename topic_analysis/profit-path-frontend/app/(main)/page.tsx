"use client"

import * as React from "react"
import { useAuth } from "@/context/auth-context"
import { researchTopicsService } from "@/lib/services/research-topics.service"
import type { ResearchTopic } from "@/types/research"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { Search, Sparkles, Clock } from "lucide-react"
import { useRouter } from "next/navigation"

export default function DashboardPage() {
  const { user, isLoading: authLoading } = useAuth()
  const router = useRouter()
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
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="flex flex-col items-center justify-center py-16 px-4 text-center">
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-4 bg-gradient-to-r from-zinc-900 to-zinc-600 bg-clip-text text-transparent dark:from-white dark:to-zinc-500">
          Discover Your Next Profitable Niche
        </h1>
        <p className="text-lg text-muted-foreground mb-8 max-w-2xl">
          Combine SEO data, Google Trends, and affiliate offers in one powerful workflow
        </p>

        <div className="w-full max-w-2xl">
          <div className="relative">
            <Search className="absolute left-4 top-3.5 h-5 w-5 text-muted-foreground" />
            <Input
              placeholder='Enter a broad topic to begin research (e.g., "Sustainable Living")...'
              className="pl-11 pr-4 h-12 text-base bg-background"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleCreateTopic()}
            />
          </div>
          <Button
            className="w-full mt-4 h-12 bg-blue-600 hover:bg-blue-700 dark:bg-blue-600 dark:hover:bg-blue-700 text-white"
            onClick={handleCreateTopic}
            disabled={!searchTerm.trim()}
          >
            Start Research <Sparkles className="ml-2 h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Recent Projects */}
      <div className="space-y-4">
        <h2 className="text-2xl font-bold tracking-tight">Recent Projects</h2>

        {error && (
          <div className="p-4 rounded-lg bg-red-50 dark:bg-red-950 text-red-600 dark:text-red-400">
            {error}
          </div>
        )}

        {loading ? (
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {[1, 2, 3, 4].map((i) => (
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
              <p className="text-sm text-muted-foreground">
                Start your first research project by entering a topic above
              </p>
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
    </div>
  )
}
