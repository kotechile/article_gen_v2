"use client"

import {
    Sheet,
    SheetContent,
    SheetDescription,
    SheetHeader,
    SheetTitle,
} from "@/components/ui/sheet"
import { Button } from "@/components/ui/button"
import { Loader2, TrendingUp, ExternalLink, Search, BarChart3 } from "lucide-react"
import Link from "next/link"

interface TrendAnalysis {
    search_volume?: number
    subtopic_name?: string
}

interface TrendAnalysisModalProps {
    isOpen: boolean
    onClose: () => void
    subtopicName: string
    trendData?: TrendAnalysis
    onAnalyze?: () => void
    isLoading?: boolean
}

export function TrendAnalysisModal({
    isOpen,
    onClose,
    subtopicName,
    trendData,
    onAnalyze,
    isLoading = false
}: TrendAnalysisModalProps) {

    const googleTrendsUrl = `https://trends.google.com/trends/explore?date=today%2012-m&geo=US&q=${encodeURIComponent(subtopicName)}`

    return (
        <Sheet open={isOpen} onOpenChange={onClose}>
            <SheetContent className="w-[400px] sm:w-[540px] overflow-y-auto">
                <SheetHeader>
                    <SheetTitle className="flex justify-between items-center">
                        <span>Market Analysis: {subtopicName}</span>
                    </SheetTitle>
                    <SheetDescription>
                        Analyze search volume and verify trends externally.
                    </SheetDescription>
                </SheetHeader>

                <div className="mt-8 flex flex-col gap-6">

                    {/* Action Button (Trigger Backend Enrichment) */}
                    {onAnalyze && (
                        <div className="p-6 bg-secondary/20 rounded-xl border border-border/50 flex flex-col items-center text-center gap-3">
                            <div className="p-3 bg-secondary/30 rounded-full">
                                <BarChart3 className="h-6 w-6 text-secondary-foreground" />
                            </div>
                            <div className="space-y-1">
                                <h4 className="font-semibold">Refresh Market Data</h4>
                                <p className="text-sm text-muted-foreground">
                                    Fetch latest Search Volume & Affiliate Offers.
                                </p>
                            </div>
                            <Button
                                onClick={onAnalyze}
                                disabled={isLoading}
                                className="w-full max-w-[200px] mt-2"
                            >
                                {isLoading ? (
                                    <>
                                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                        Analyzing...
                                    </>
                                ) : (
                                    <>
                                        Run Analysis
                                    </>
                                )}
                            </Button>
                        </div>
                    )}

                    <div className="grid grid-cols-1 gap-4">
                        {/* Search Volume Card */}
                        <div className="p-5 bg-card rounded-xl border shadow-sm flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg text-blue-600 dark:text-blue-400">
                                    <Search className="h-5 w-5" />
                                </div>
                                <div>
                                    <span className="text-xs text-muted-foreground uppercase font-semibold tracking-wider">Monthly Volume</span>
                                    <div className="text-2xl font-bold mt-0.5">
                                        {trendData?.search_volume
                                            ? trendData.search_volume.toLocaleString()
                                            : <span className="text-muted-foreground text-lg font-normal">-</span>}
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Google Trends Card */}
                        <div className="p-5 bg-card rounded-xl border shadow-sm flex items-center justify-between relative overflow-hidden group">
                            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-transparent to-orange-50/30 dark:to-orange-900/10 pointer-events-none" />
                            <div className="flex items-center gap-3 relative z-10">
                                <div className="p-2 bg-orange-100 dark:bg-orange-900/30 rounded-lg text-orange-600 dark:text-orange-400">
                                    <TrendingUp className="h-5 w-5" />
                                </div>
                                <div>
                                    <span className="text-xs text-muted-foreground uppercase font-semibold tracking-wider">Trend Stability</span>
                                    <h4 className="text-sm font-medium mt-0.5">Check on Google Trends</h4>
                                </div>
                            </div>
                            <Button variant="outline" size="sm" asChild className="relative z-10 hover:bg-orange-50 dark:hover:bg-orange-900/20 hover:text-orange-600 dark:hover:text-orange-400 border-orange-200 dark:border-orange-800 transition-colors">
                                <Link href={googleTrendsUrl} target="_blank" rel="noopener noreferrer">
                                    View Report <ExternalLink className="ml-2 h-3 w-3" />
                                </Link>
                            </Button>
                        </div>
                    </div>

                    <div className="text-xs text-muted-foreground text-center mt-4">
                        <p>We recommend verifying checking for a <strong>stable or rising 12-month trend</strong> before creating content.</p>
                    </div>

                </div>
            </SheetContent>
        </Sheet>
    )
}
