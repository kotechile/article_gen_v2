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
import { Sparkles, TrendingUp, ArrowUpDown, Download, Filter } from "lucide-react"

interface AllKeywordsTableProps {
    topicId: string
    refreshTrigger?: number
}

type SortField = "profitability_score" | "search_volume" | "difficulty" | "cpc"
type SortDirection = "asc" | "desc"

export function AllKeywordsTable({ topicId, refreshTrigger }: AllKeywordsTableProps) {
    const [keywords, setKeywords] = React.useState<Keyword[]>([])
    const [loading, setLoading] = React.useState(false)
    const [sortField, setSortField] = React.useState<SortField>("profitability_score")
    const [sortDirection, setSortDirection] = React.useState<SortDirection>("desc")
    const [intentFilter, setIntentFilter] = React.useState<string>("all")

    // Load keywords on mount or refresh
    React.useEffect(() => {
        loadKeywords()
    }, [topicId, refreshTrigger])

    const loadKeywords = async () => {
        setLoading(true)
        try {
            const data = await keywordsService.getTopicKeywords(topicId)
            setKeywords(data)
        } catch (error) {
            console.error("Failed to load topic keywords:", error)
        } finally {
            setLoading(false)
        }
    }

    const uniqueIntents = React.useMemo(() => {
        const intents = new Set<string>()
        keywords.forEach(k => {
            if (k.main_intent) intents.add(k.main_intent)
        })
        return Array.from(intents)
    }, [keywords])

    const filteredKeywords = React.useMemo(() => {
        if (intentFilter === "all") return keywords
        return keywords.filter(k => k.main_intent?.toLowerCase() === intentFilter)
    }, [keywords, intentFilter])

    const sortedKeywords = React.useMemo(() => {
        return [...filteredKeywords].sort((a, b) => {
            let aValue: number = 0;
            let bValue: number = 0;

            // Handle strict property names for known fields
            if (sortField === "difficulty") {
                aValue = a.keyword_difficulty ?? a.difficulty ?? 0;
                bValue = b.keyword_difficulty ?? b.difficulty ?? 0;
            } else if (sortField === "profitability_score") {
                aValue = a.profitability_score ?? 0;
                bValue = b.profitability_score ?? 0;
            } else {
                // Generic fallback for search_volume, cpc
                aValue = Number(a[sortField]) || 0;
                bValue = Number(b[sortField]) || 0;
            }

            return sortDirection === "asc" ? aValue - bValue : bValue - aValue;
        })
    }, [filteredKeywords, sortField, sortDirection])

    const toggleSort = (field: SortField) => {
        if (sortField === field) {
            setSortDirection(sortDirection === "asc" ? "desc" : "asc")
        } else {
            setSortField(field)
            setSortDirection("desc")
        }
    }

    const getDifficultyColor = (kd?: number) => {
        if (kd === undefined) return "text-zinc-500"
        if (kd <= 30) return "text-green-600 dark:text-green-400 font-medium"
        if (kd <= 60) return "text-yellow-600 dark:text-yellow-400 font-medium"
        return "text-red-600 dark:text-red-400 font-medium"
    }

    const getVolumeColor = (vol?: number) => {
        if (!vol) return "text-zinc-500"
        if (vol >= 1000) return "text-green-600 dark:text-green-400 font-medium"
        if (vol >= 300) return "text-yellow-600 dark:text-yellow-400 font-medium"
        return "text-red-600 dark:text-red-400 font-medium"
    }

    const getScoreColor = (score?: number) => {
        if (!score) return "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400"
        if (score >= 1000) return "bg-green-100 text-green-700 hover:bg-green-100/80 dark:bg-green-900/30 dark:text-green-400"
        if (score >= 100) return "bg-yellow-100 text-yellow-700 hover:bg-yellow-100/80 dark:bg-yellow-900/30 dark:text-yellow-400"
        return "bg-red-100 text-red-700 hover:bg-red-100/80 dark:bg-red-900/30 dark:text-red-400"
    }

    if (loading && keywords.length === 0) {
        return (
            <Card>
                <CardContent className="py-12 flex justify-center">
                    <div className="flex items-center gap-2 text-muted-foreground">
                        <Sparkles className="h-4 w-4 animate-spin" />
                        Loading keywords...
                    </div>
                </CardContent>
            </Card>
        )
    }

    if (keywords.length === 0) {
        return null // Don't show if empty (e.g. before expansion)
    }

    return (
        <Card className="mt-8 border-zinc-200 dark:border-zinc-800">
            <CardHeader>
                <div className="flex items-center justify-between">
                    <div>
                        <CardTitle className="text-xl">All Expanded Keywords</CardTitle>
                        <CardDescription>
                            Aggregated keywords from all subtopics ({keywords.length} total)
                        </CardDescription>
                    </div>
                    <div className="flex gap-2">
                        <Select value={intentFilter} onValueChange={setIntentFilter}>
                            <SelectTrigger className="w-[180px]">
                                <Filter className="mr-2 h-4 w-4" />
                                <SelectValue placeholder="Filter Intent" />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="all">All Intents</SelectItem>
                                <SelectItem value="informational">Informational</SelectItem>
                                <SelectItem value="commercial">Commercial</SelectItem>
                                <SelectItem value="transactional">Transactional</SelectItem>
                                <SelectItem value="navigational">Navigational</SelectItem>
                            </SelectContent>
                        </Select>
                        <Button variant="outline" size="sm">
                            <Download className="mr-2 h-4 w-4" />
                            Export
                        </Button>
                    </div>
                </div>
            </CardHeader>
            <CardContent>
                <div className="rounded-md border">
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead className="w-[300px]">Keyword</TableHead>
                                <TableHead>
                                    <Button variant="ghost" size="sm" onClick={() => toggleSort("search_volume")}>
                                        Volume <ArrowUpDown className="ml-2 h-3 w-3" />
                                    </Button>
                                </TableHead>
                                <TableHead>
                                    <Button variant="ghost" size="sm" onClick={() => toggleSort("cpc")}>
                                        CPC <ArrowUpDown className="ml-2 h-3 w-3" />
                                    </Button>
                                </TableHead>
                                <TableHead>
                                    <Button variant="ghost" size="sm" onClick={() => toggleSort("difficulty")}>
                                        KD <ArrowUpDown className="ml-2 h-3 w-3" />
                                    </Button>
                                </TableHead>
                                <TableHead>
                                    <Button variant="ghost" size="sm" onClick={() => toggleSort("profitability_score")}>
                                        Score <ArrowUpDown className="ml-2 h-3 w-3" />
                                    </Button>
                                </TableHead>
                                <TableHead>Intent</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {sortedKeywords.map((keyword) => {
                                const kdValue = keyword.keyword_difficulty ?? keyword.difficulty;
                                return (
                                    <TableRow key={keyword.id}>
                                        <TableCell className="font-medium">{keyword.keyword}</TableCell>
                                        <TableCell>
                                            <span className={getVolumeColor(keyword.search_volume)}>
                                                {keyword.search_volume?.toLocaleString() ?? "-"}
                                            </span>
                                        </TableCell>
                                        <TableCell>{keyword.cpc ? `$${keyword.cpc.toFixed(2)}` : "-"}</TableCell>
                                        <TableCell>
                                            <span className={getDifficultyColor(kdValue)}>
                                                {kdValue ?? "-"}
                                            </span>
                                        </TableCell>
                                        <TableCell>
                                            <Badge variant="outline" className={`border-0 ${getScoreColor(keyword.profitability_score)}`}>
                                                {keyword.profitability_score?.toFixed(1) ?? "N/A"}
                                            </Badge>
                                        </TableCell>
                                        <TableCell>
                                            <Badge variant="secondary" className="capitalize">
                                                {keyword.main_intent ?? "Unknown"}
                                            </Badge>
                                        </TableCell>
                                    </TableRow>
                                )
                            })}
                        </TableBody>
                    </Table>
                </div>
            </CardContent>
        </Card>
    )
}
