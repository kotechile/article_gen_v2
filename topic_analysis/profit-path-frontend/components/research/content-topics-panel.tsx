"use client"

import * as React from "react"
import { ContentTopic } from "@/types/research"
import { keywordsService } from "@/lib/services/keywords.service"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Sparkles, Download, FileText, Target } from "lucide-react"

interface ContentTopicsPanelProps {
    topicId: string
}

export function ContentTopicsPanel({ topicId }: ContentTopicsPanelProps) {
    const [contentTopics, setContentTopics] = React.useState<ContentTopic[]>([])
    const [loading, setLoading] = React.useState(false)
    const [generating, setGenerating] = React.useState(false)

    // Load content topics on mount
    React.useEffect(() => {
        loadContentTopics()
    }, [topicId])

    const loadContentTopics = async () => {
        setLoading(true)
        try {
            const data = await keywordsService.getContentTopics(topicId)
            setContentTopics(data)
        } catch (error) {
            console.error("Failed to load content topics:", error)
        } finally {
            setLoading(false)
        }
    }

    const handleGenerate = async () => {
        setGenerating(true)
        try {
            await keywordsService.clusterKeywords(topicId)
            await loadContentTopics()
        } catch (error) {
            console.error("Failed to generate content topics:", error)
        } finally {
            setGenerating(false)
        }
    }

    const handleExportCSV = () => {
        if (contentTopics.length === 0) return

        const headers = ["Title", "Intent Type", "Profitability Score", "Created At"]
        const rows = contentTopics.map(topic => [
            topic.title,
            topic.intent_type ?? "Unknown",
            topic.estimated_profitability_score?.toFixed(2) ?? "N/A",
            new Date(topic.created_at).toLocaleDateString()
        ])

        const csvContent = [
            headers.join(","),
            ...rows.map(row => row.map(cell => `"${cell}"`).join(","))
        ].join("\n")

        const blob = new Blob([csvContent], { type: "text/csv" })
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement("a")
        a.href = url
        a.download = `content-topics-${topicId}.csv`
        a.click()
        window.URL.revokeObjectURL(url)
    }

    const getProfitabilityColor = (score?: number) => {
        if (!score) return "bg-gray-100 dark:bg-gray-800 text-gray-600"
        if (score > 5) return "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400"
        if (score > 2) return "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400"
        return "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400"
    }

    const getIntentColor = (intent?: string) => {
        if (!intent) return "bg-gray-100 dark:bg-gray-800"
        const lower = intent.toLowerCase()
        if (lower.includes("commercial")) return "bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400"
        if (lower.includes("transactional")) return "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400"
        if (lower.includes("informational")) return "bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400"
        return "bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400"
    }

    return (
        <Card className="border-zinc-200 dark:border-zinc-800">
            <CardHeader>
                <div className="flex items-center justify-between">
                    <div>
                        <CardTitle className="text-xl">Content Topics</CardTitle>
                        <CardDescription>
                            AI-generated blog topics clustered by keyword intent
                        </CardDescription>
                    </div>
                    <div className="flex gap-2">
                        {contentTopics.length > 0 && (
                            <Button
                                variant="outline"
                                onClick={handleExportCSV}
                            >
                                <Download className="mr-2 h-4 w-4" />
                                Export CSV
                            </Button>
                        )}
                        <Button
                            onClick={handleGenerate}
                            disabled={generating}
                            className="bg-blue-600 hover:bg-blue-700"
                        >
                            {generating ? (
                                <>
                                    <Sparkles className="mr-2 h-4 w-4 animate-spin" />
                                    Generating...
                                </>
                            ) : (
                                <>
                                    <Sparkles className="mr-2 h-4 w-4" />
                                    Generate Content Topics
                                </>
                            )}
                        </Button>
                    </div>
                </div>
            </CardHeader>

            <CardContent>
                {loading ? (
                    <div className="flex items-center justify-center py-8 text-muted-foreground">
                        Loading content topics...
                    </div>
                ) : contentTopics.length === 0 ? (
                    <div className="flex flex-col items-center justify-center py-12 text-center">
                        <FileText className="h-12 w-12 text-muted-foreground mb-4" />
                        <p className="text-muted-foreground">No content topics generated yet</p>
                        <p className="text-sm text-muted-foreground mt-2">
                            Generate subtopics and expand keywords first, then cluster them into content topics
                        </p>
                    </div>
                ) : (
                    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                        {contentTopics.map((topic) => (
                            <Card
                                key={topic.id}
                                className="border-zinc-200 dark:border-zinc-800 hover:shadow-md transition-shadow"
                            >
                                <CardHeader className="pb-3">
                                    <div className="flex items-start justify-between gap-2">
                                        <CardTitle className="text-base leading-tight">
                                            {topic.title}
                                        </CardTitle>
                                        <Target className="h-4 w-4 text-muted-foreground shrink-0 mt-1" />
                                    </div>
                                </CardHeader>
                                <CardContent className="space-y-3">
                                    <div className="flex items-center gap-2">
                                        <Badge className={getProfitabilityColor(topic.estimated_profitability_score)}>
                                            Score: {topic.estimated_profitability_score?.toFixed(2) ?? "N/A"}
                                        </Badge>
                                        <Badge className={getIntentColor(topic.intent_type)}>
                                            {topic.intent_type ?? "Unknown"}
                                        </Badge>
                                    </div>

                                    <div className="text-xs text-muted-foreground">
                                        <div className="flex items-center justify-between">
                                            <span>Keywords:</span>
                                            <span className="font-medium">
                                                {topic.supporting_keyword_ids?.length ?? 0} supporting
                                            </span>
                                        </div>
                                    </div>

                                    <div className="pt-2 text-xs text-muted-foreground">
                                        Created {new Date(topic.created_at).toLocaleDateString()}
                                    </div>
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                )}
            </CardContent>
        </Card>
    )
}
