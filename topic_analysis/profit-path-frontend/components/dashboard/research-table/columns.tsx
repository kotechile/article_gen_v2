"use client"

import { ColumnDef } from "@tanstack/react-table"
import { Badge } from "@/components/ui/badge"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { TrendSparkline } from "./trend-sparkline"
import { ViabilityScore } from "./viability-score"

export type ResearchData = {
    id: string
    topic: string
    searchVolume: number
    keywords: string[]
    trendData: number[]
    viability: number
    cpc: number
}

export const columns: ColumnDef<ResearchData>[] = [
    {
        accessorKey: "topic",
        header: "Topic",
        cell: ({ row }) => {
            return (
                <div className="flex flex-col">
                    <span className="font-semibold text-base">{row.getValue("topic")}</span>
                    <span className="text-xs text-muted-foreground">{row.original.searchVolume.toLocaleString()} monthly searches</span>
                </div>
            )
        },
    },
    {
        accessorKey: "keywords",
        header: "Keywords",
        cell: ({ row }) => {
            const keywords = row.original.keywords
            const visible = keywords.slice(0, 3)
            const remainder = keywords.length - 3

            return (
                <div className="flex flex-wrap gap-1 max-w-[200px]">
                    {visible.map((kw, i) => (
                        <Badge key={i} variant="secondary" className="bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-200 dark:hover:bg-zinc-700 font-normal">
                            {kw}
                        </Badge>
                    ))}
                    {remainder > 0 && (
                        <TooltipProvider>
                            <Tooltip>
                                <TooltipTrigger>
                                    <Badge variant="outline" className="text-xs text-muted-foreground hover:bg-muted cursor-help">
                                        +{remainder} more
                                    </Badge>
                                </TooltipTrigger>
                                <TooltipContent>
                                    <div className="flex flex-col gap-1">
                                        {keywords.slice(3).map((kw, i) => (
                                            <span key={i}>{kw}</span>
                                        ))}
                                    </div>
                                </TooltipContent>
                            </Tooltip>
                        </TooltipProvider>
                    )}
                </div>
            )
        },
    },
    {
        accessorKey: "trendData",
        header: "12-Month Trend",
        cell: ({ row }) => {
            return <TrendSparkline data={row.getValue("trendData")} />
        },
    },
    {
        accessorKey: "viability",
        header: "Viability",
        cell: ({ row }) => {
            return <ViabilityScore score={row.getValue("viability")} />
        },
    },
    {
        accessorKey: "cpc",
        header: "CPC",
        cell: ({ row }) => {
            return <span className="font-mono text-muted-foreground">${row.getValue<number>("cpc").toFixed(2)}</span>
        }
    }
]
