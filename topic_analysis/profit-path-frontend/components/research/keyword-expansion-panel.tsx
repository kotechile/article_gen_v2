"use client"

import * as React from "react"
import { Keyword } from "@/types/research"
import { keywordsService } from "@/lib/services/keywords.service"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table"
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select"
import { subtopicsService } from "@/lib/services/subtopics.service"
import { Sparkles, TrendingUp, DollarSign, Target, ArrowUpDown } from "lucide-react"

interface KeywordExpansionPanelProps {
    topicId: string
    subtopicId: string
    subtopicName: string
    seedKeywords?: string[]
}

type SortField = "profitability_score" | "search_volume" | "difficulty" | "cpc"
type SortDirection = "asc" | "desc"

export function KeywordExpansionPanel({ topicId, subtopicId, subtopicName, seedKeywords }: KeywordExpansionPanelProps) {
    const [keywords, setKeywords] = React.useState<Keyword[]>([])
    const [loading, setLoading] = React.useState(false)
    const [expanding, setExpanding] = React.useState(false)
    const [sortField, setSortField] = React.useState<SortField>("profitability_score")
    const [sortDirection, setSortDirection] = React.useState<SortDirection>("desc")
    const [intentFilter, setIntentFilter] = React.useState<string>("all")

    // ... (rest of the component logic)

    // (Inside the JSX, locate the empty state)
    // I need to find the specific part to replace. 
    // Since I can't see the whole file efficiently in one go, I'll replace the Props definition first, 
    // and then do a second Replace for the JSX.

    // ERROR in Strategy: I can't do partial replace of the function body easily without seeing it.
    // I will replace the Interface and Function Signature first.


    // Load keywords on mount
    React.useEffect(() => {
        loadKeywords()
    }, [topicId, subtopicId])

    const loadKeywords = async () => {
        setLoading(true)
        try {
            const data = await keywordsService.getSubtopicKeywords(topicId, subtopicId)
            setKeywords(data)
        } catch (error) {
            console.error("Failed to load keywords:", error)
        } finally {
            setLoading(false)
        }
    }

    const handleExpand = async () => {
        setExpanding(true)
        try {
            await subtopicsService.expandSubtopicKeywords(topicId, subtopicId)
            await loadKeywords()
        } catch (error) {
            console.error("Failed to expand keywords:", error)
        } finally {
            setExpanding(false)
        }
    }

    const filteredKeywords = React.useMemo(() => {
        let filtered = keywords
        if (intentFilter !== "all") {
            filtered = keywords.filter(k => k.main_intent?.toLowerCase() === intentFilter.toLowerCase())
        }
        return filtered
    }, [keywords, intentFilter])

    const sortedKeywords = React.useMemo(() => {
        return [...filteredKeywords].sort((a, b) => {
            let aValue: number = 0;
            let bValue: number = 0;

            if (sortField === "difficulty") {
                aValue = a.keyword_difficulty ?? a.difficulty ?? 0;
                bValue = b.keyword_difficulty ?? b.difficulty ?? 0;
            } else if (sortField === "profitability_score") {
                aValue = a.profitability_score ?? 0;
                bValue = b.profitability_score ?? 0;
            } else {
                aValue = Number(a[sortField]) || 0;
                bValue = Number(b[sortField]) || 0;
            }

            return sortDirection === "asc" ? aValue - bValue : bValue - aValue;
        })
    }, [filteredKeywords, sortField, sortDirection])

    const toggleSort = (field: SortField) => {
        if (sortField === field) {
            setSortDirection(prev => prev === "asc" ? "desc" : "asc")
        } else {
            setSortField(field)
            setSortDirection("desc")
        }
    }

    const getProfitabilityColor = (score?: number) => {
        if (!score) return "bg-gray-100 dark:bg-gray-800 text-gray-600"
        if (score > 5) return "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400"
        if (score > 2) return "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400"
        return "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400"
    }

    const uniqueIntents = React.useMemo(() => {
        const intents = new Set(keywords.map(k => k.main_intent).filter(Boolean) as string[])
        return Array.from(intents)
    }, [keywords])

    return (
        <Card className="border-zinc-200 dark:border-zinc-800">
            <CardHeader>
                <div className="flex items-center justify-between">
                    <div>
                        <CardTitle className="text-xl">Keyword Expansion</CardTitle>
                        <CardDescription>
                            Expanded keywords for: <strong>{subtopicName}</strong>
                        </CardDescription>
                    </div>
                </div>

                {keywords.length > 0 && (
                    <div className="flex gap-2 mt-4">
                        <Select value={intentFilter} onValueChange={setIntentFilter}>
                            <SelectTrigger className="w-[200px]">
                                <SelectValue placeholder="Filter by intent" />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="all">All Intents</SelectItem>
                                {uniqueIntents.map(intent => (
                                    <SelectItem key={intent} value={intent.toLowerCase()}>
                                        {intent}
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    </div>
                )}
            </CardHeader>

            <CardContent>
                {loading ? (
                    <div className="flex items-center justify-center py-8 text-muted-foreground">
                        Loading keywords...
                    </div>
                ) : keywords.length === 0 ? (
                    <div className="flex flex-col items-center justify-center py-12 text-center">
                        <TrendingUp className="h-12 w-12 text-muted-foreground mb-4" />
                        <p className="text-muted-foreground font-medium">No keywords expanded yet</p>

                        {seedKeywords && seedKeywords.length > 0 && (
                            <div className="mt-4 max-w-md">
                                <p className="text-sm text-muted-foreground mb-3">
                                    Seed Keywords:
                                </p>
                                <div className="flex flex-wrap gap-2 justify-center">
                                    {seedKeywords.map((seed, i) => (
                                        <Badge key={i} variant="secondary" className="text-xs">
                                            {seed}
                                        </Badge>
                                    ))}
                                </div>
                            </div>
                        )}

                        <p className="text-sm text-muted-foreground mt-6">
                            Use "Expand All Keywords" at the top of the page to fetch data.
                        </p>
                    </div>
                ) : (
                    <div className="rounded-md border">
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead>Keyword</TableHead>
                                    <TableHead>
                                        <Button
                                            variant="ghost"
                                            size="sm"
                                            onClick={() => toggleSort("search_volume")}
                                            className="h-8 px-2"
                                        >
                                            Volume
                                            <ArrowUpDown className="ml-2 h-3 w-3" />
                                        </Button>
                                    </TableHead>
                                    <TableHead>
                                        <Button
                                            variant="ghost"
                                            size="sm"
                                            onClick={() => toggleSort("cpc")}
                                            className="h-8 px-2"
                                        >
                                            CPC
                                            <ArrowUpDown className="ml-2 h-3 w-3" />
                                        </Button>
                                    </TableHead>
                                    <TableHead>
                                        <Button
                                            variant="ghost"
                                            size="sm"
                                            onClick={() => toggleSort("difficulty")}
                                            className="h-8 px-2"
                                        >
                                            Difficulty
                                            <ArrowUpDown className="ml-2 h-3 w-3" />
                                        </Button>
                                    </TableHead>
                                    <TableHead>
                                        <Button
                                            variant="ghost"
                                            size="sm"
                                            onClick={() => toggleSort("profitability_score")}
                                            className="h-8 px-2"
                                        >
                                            Score
                                            <ArrowUpDown className="ml-2 h-3 w-3" />
                                        </Button>
                                    </TableHead>
                                    <TableHead>Intent</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {sortedKeywords.map((keyword) => (
                                    <TableRow key={keyword.id}>
                                        <TableCell className="font-medium">
                                            {keyword.keyword}
                                        </TableCell>
                                        <TableCell>
                                            {keyword.search_volume?.toLocaleString() ?? "-"}
                                        </TableCell>
                                        <TableCell>
                                            {keyword.cpc ? `$${keyword.cpc.toFixed(2)}` : "-"}
                                        </TableCell>
                                        <TableCell>
                                            <Badge variant="outline">
                                                {keyword.difficulty ?? "-"}
                                            </Badge>
                                        </TableCell>
                                        <TableCell>
                                            <Badge className={getProfitabilityColor(keyword.profitability_score)}>
                                                {keyword.profitability_score?.toFixed(2) ?? "N/A"}
                                            </Badge>
                                        </TableCell>
                                        <TableCell>
                                            <Badge variant="secondary">
                                                {keyword.main_intent ?? "Unknown"}
                                            </Badge>
                                        </TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </div>
                )}
            </CardContent>
        </Card>
    )
}
