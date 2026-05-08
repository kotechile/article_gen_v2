import * as React from "react"
import { AlertCircle, BarChart3, ChevronDown, ChevronUp, FolderTree, RefreshCw, Search, Sparkles } from "lucide-react"
import { motion } from "framer-motion"

import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import type {
    TopicKeywordCandidate,
    TopicKeywordCluster,
    TopicKeywordResearchRun,
} from "@/types/research"

interface TopicKeywordResearchSummary {
    seedCount: number
    candidateCount: number
    totalCandidateCount?: number
    clusterCount: number
    generatedAt: string | null
}

interface TopicKeywordResearchPanelProps {
    topicId: string
    topicMode: 'keyword_first' | 'editorial_first' | 'hybrid'
    keywordResearchRun: TopicKeywordResearchRun | null
    keywordCandidates: TopicKeywordCandidate[]
    allKeywordCandidates: TopicKeywordCandidate[]
    keywordClusters: TopicKeywordCluster[]
    keywordResearchLoading: boolean
    runningKeywordResearch: boolean
    keywordResearchError: string | null
    generatingClusterIdeas: boolean
    selectedClusterIds: Set<string>
    canGenerateIdeas: boolean
    manualSeedInput: string
    keywordResearchSummary: TopicKeywordResearchSummary
    rebuildRunCount?: number
    latestRebuildRunLabel?: string | null
    latestRebuildRoute?: string | null
    latestRebuildRouteLabel?: string | null
    onOpenRebuild?: (() => void) | null
    onRefreshKeywordResearch: () => void
    onRunKeywordResearch: () => void
    onManualSeedInputChange: (value: string) => void
    onAddManualSeed: (keyword: string) => void
    onGenerateIdeasFromClusters: () => void
    onToggleClusterSelection: (clusterId: string) => void
    formatDateTime: (value?: string | null) => string
}

export function TopicKeywordResearchPanel({
    topicId,
    topicMode,
    keywordResearchRun,
    keywordCandidates,
    allKeywordCandidates,
    keywordClusters,
    keywordResearchLoading,
    runningKeywordResearch,
    keywordResearchError,
    generatingClusterIdeas,
    selectedClusterIds,
    canGenerateIdeas,
    manualSeedInput,
    keywordResearchSummary,
    rebuildRunCount = 0,
    latestRebuildRunLabel = null,
    latestRebuildRoute = null,
    latestRebuildRouteLabel = null,
    onOpenRebuild = null,
    onRefreshKeywordResearch,
    onRunKeywordResearch,
    onManualSeedInputChange,
    onAddManualSeed,
    onGenerateIdeasFromClusters,
    onToggleClusterSelection,
    formatDateTime,
}: TopicKeywordResearchPanelProps) {
    const [showAllKeywords, setShowAllKeywords] = React.useState(false)
    const topKeywordRows = keywordCandidates.slice(0, 8)
    const visibleAllKeywords = showAllKeywords ? allKeywordCandidates : allKeywordCandidates.slice(0, 20)
    const hasReadyRebuildRoute =
        latestRebuildRoute === 'article_ready' ||
        latestRebuildRoute === 'software_ready' ||
        latestRebuildRoute === 'article_plus_software'
    const [showLegacyKeywordFlow, setShowLegacyKeywordFlow] = React.useState(!hasReadyRebuildRoute)

    React.useEffect(() => {
        setShowLegacyKeywordFlow(!hasReadyRebuildRoute)
    }, [hasReadyRebuildRoute])

    return (
        <div className="max-w-7xl mx-auto mb-8">
            <div className="bg-muted/30 backdrop-blur-md border border-border rounded-2xl p-6">
                <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between mb-6">
                    <div>
                        <h2 className="text-xl font-semibold text-foreground">Topic Keyword Research</h2>
                        <p className="text-sm text-muted-foreground mt-1">
                            {hasReadyRebuildRoute
                                ? 'A rebuild run already marked this scope as ready, so this legacy topic-keyword flow is best used for comparison, recovery, or extra seed exploration.'
                                : topicMode === 'editorial_first'
                                ? 'This topic is editorial-first, so keyword research is optional. Use it when you want search evidence or manual seed exploration.'
                                : 'This topic-level pipeline turns ranked keywords into intent clusters, then turns the strongest clusters into article ideas and companion software opportunities.'}
                        </p>
                    </div>
                    <div className="flex items-center gap-2 flex-wrap">
                        {keywordResearchRun && (
                            <Button
                                onClick={onRefreshKeywordResearch}
                                disabled={keywordResearchLoading || !topicId}
                                variant="outline"
                                size="sm"
                                className="border-border hover:bg-muted"
                            >
                                {keywordResearchLoading ? (
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
                        <Button
                            onClick={onRunKeywordResearch}
                            disabled={runningKeywordResearch || !topicId}
                            size="sm"
                            variant={hasReadyRebuildRoute ? 'outline' : 'default'}
                            className={hasReadyRebuildRoute ? 'border-border hover:bg-muted' : 'bg-primary hover:bg-primary/90 text-primary-foreground'}
                        >
                            {runningKeywordResearch ? (
                                <>
                                    <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                                    Running...
                                </>
                            ) : (
                                <>
                                    <Search className="mr-2 h-4 w-4" />
                                    {hasReadyRebuildRoute ? 'Run Legacy Keyword Research' : 'Run Keyword Research'}
                                </>
                            )}
                        </Button>
                    </div>
                </div>

                {keywordResearchError && (
                    <div className="mb-5 rounded-xl border border-amber-500/20 bg-amber-500/10 p-4 text-sm text-amber-100">
                        <div className="flex items-start gap-2">
                            <AlertCircle className="mt-0.5 h-4 w-4 flex-shrink-0 text-amber-300" />
                            <span>{keywordResearchError}</span>
                        </div>
                    </div>
                )}

                {rebuildRunCount > 0 && (
                    <div className="mb-5 rounded-xl border border-primary/20 bg-primary/10 p-4">
                        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                            <div>
                                <div className="text-sm font-semibold text-foreground">Research Rebuild Available</div>
                                <p className="mt-1 text-xs text-muted-foreground">
                                    This topic’s category scope already has {rebuildRunCount} rebuild run{rebuildRunCount === 1 ? '' : 's'}.
                                    {latestRebuildRunLabel ? ` Latest run: ${latestRebuildRunLabel}.` : ''}
                                    {latestRebuildRouteLabel ? ` Strongest route: ${latestRebuildRouteLabel}.` : ''}
                                </p>
                                {hasReadyRebuildRoute && (
                                    <p className="mt-2 text-xs text-primary">
                                        Rebuild has already produced a ready route for this scope, so prefer that path before spending more time in the legacy keyword pipeline.
                                    </p>
                                )}
                            </div>
                            {onOpenRebuild && (
                                <Button
                                    type="button"
                                    variant="outline"
                                    size="sm"
                                    onClick={onOpenRebuild}
                                    className="border-primary/30 bg-background/40 hover:bg-background/60"
                                >
                                    <Sparkles className="mr-2 h-3 w-3" />
                                    Open Rebuild Run
                                </Button>
                            )}
                        </div>
                    </div>
                )}

                {hasReadyRebuildRoute && (
                    <div className="mb-5 rounded-xl border border-border bg-background/30 p-4">
                        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                            <div>
                                <div className="text-sm font-semibold text-foreground">Legacy Keyword Flow Hidden By Default</div>
                                <p className="mt-1 text-xs text-muted-foreground">
                                    Rebuild is already ready for this scope, so the legacy keyword flow is collapsed unless you want to compare results or rescue edge-case seeds.
                                </p>
                            </div>
                            <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                onClick={() => setShowLegacyKeywordFlow((current) => !current)}
                                className="border-border hover:bg-muted"
                            >
                                {showLegacyKeywordFlow ? (
                                    <>
                                        <ChevronUp className="mr-2 h-3 w-3" />
                                        Hide Legacy Flow
                                    </>
                                ) : (
                                    <>
                                        <ChevronDown className="mr-2 h-3 w-3" />
                                        Show Legacy Flow
                                    </>
                                )}
                            </Button>
                        </div>
                    </div>
                )}

                {showLegacyKeywordFlow && (
                    <>
                        <div className="mb-5 rounded-xl border border-border bg-muted/20 p-4">
                            <div className="text-sm font-medium text-foreground">Optional Manual Seeds</div>
                            <p className="mt-1 text-xs text-muted-foreground">
                                Leave this empty for fully automated seed generation. Add comma-separated or line-separated seeds when you want to rescue a hard topic, steer the run more directly, or rerun from an already discovered keyword below.
                            </p>
                            <textarea
                                value={manualSeedInput}
                                onChange={(event) => onManualSeedInputChange(event.target.value)}
                                placeholder="example: market spending by state, regional price sensitivity, consumer spending shifts"
                                className="mt-3 min-h-[84px] w-full rounded-lg border border-border bg-background/70 px-3 py-2 text-sm text-foreground outline-none transition focus:border-ring"
                            />
                        </div>

                        {!keywordResearchRun && !keywordResearchLoading ? (
                            <div className="rounded-xl border border-dashed border-border bg-muted/20 p-6 text-center">
                                <Search className="mx-auto mb-3 h-10 w-10 text-muted-foreground" />
                                <h3 className="text-base font-semibold text-foreground mb-2">No topic keyword research run yet</h3>
                                <p className="text-sm text-muted-foreground max-w-2xl mx-auto">
                                    Run the new keyword pipeline to discover candidate keywords, cluster them by intent, and prepare the next generation flow for article and software ideas.
                                </p>
                            </div>
                        ) : keywordResearchLoading ? (
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                {[1, 2, 3].map((item) => (
                                    <Skeleton key={`keyword-research-skeleton-${item}`} className="h-28 w-full rounded-xl" />
                                ))}
                            </div>
                        ) : (
                            <div className="space-y-6">
                        <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/10 p-4">
                            <div className="text-sm font-semibold text-foreground">Recommended workflow</div>
                            <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-muted-foreground">
                                {hasReadyRebuildRoute ? (
                                    <>
                                        <span className="rounded-full border border-emerald-500/20 bg-background/40 px-2 py-1">1. Review the rebuild-ready route first</span>
                                        <span className="rounded-full border border-emerald-500/20 bg-background/40 px-2 py-1">2. Use legacy keyword research only if you need comparison or extra seed expansion</span>
                                        <span className="rounded-full border border-emerald-500/20 bg-background/40 px-2 py-1">3. Publish or refine the best idea with rebuild evidence in mind</span>
                                    </>
                                ) : (
                                    <>
                                        <span className="rounded-full border border-emerald-500/20 bg-background/40 px-2 py-1">1. Refresh or rerun keyword research</span>
                                        <span className="rounded-full border border-emerald-500/20 bg-background/40 px-2 py-1">2. Select the strongest clusters</span>
                                        <span className="rounded-full border border-emerald-500/20 bg-background/40 px-2 py-1">3. Generate article ideas plus companion software ideas when tool potential is strong</span>
                                        <span className="rounded-full border border-emerald-500/20 bg-background/40 px-2 py-1">4. Publish the best ideas to Content Studio</span>
                                    </>
                                )}
                            </div>
                            <p className="mt-3 text-xs text-muted-foreground">
                                Each selected cluster is treated like a distinct user intent. The generator creates one focused article angle per cluster and may also create a companion software concept when that same cluster suggests calculator, planner, comparison, workflow, or app potential.
                            </p>
                        </div>

                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="bg-muted/20 border border-border rounded-xl p-5">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-sm text-muted-foreground">Seeds</span>
                                    <Search className="h-4 w-4 text-muted-foreground" />
                                </div>
                                <div className="text-2xl font-bold text-foreground">{keywordResearchSummary.seedCount.toLocaleString()}</div>
                                <div className="text-xs text-muted-foreground mt-1">Directional keyword starting points</div>
                            </motion.div>

                            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }} className="bg-muted/20 border border-border rounded-xl p-5">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-sm text-muted-foreground">Candidates</span>
                                    <BarChart3 className="h-4 w-4 text-muted-foreground" />
                                </div>
                                <div className="text-2xl font-bold text-foreground">{keywordResearchSummary.candidateCount.toLocaleString()}</div>
                                <div className="text-xs text-muted-foreground mt-1">
                                    Active keywords after filtering
                                    {keywordResearchSummary.totalCandidateCount && keywordResearchSummary.totalCandidateCount > keywordResearchSummary.candidateCount
                                        ? ` · ${keywordResearchSummary.totalCandidateCount} found total`
                                        : ''}
                                </div>
                            </motion.div>

                            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="bg-muted/20 border border-border rounded-xl p-5">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-sm text-muted-foreground">Clusters</span>
                                    <FolderTree className="h-4 w-4 text-muted-foreground" />
                                </div>
                                <div className="text-2xl font-bold text-foreground">{keywordResearchSummary.clusterCount.toLocaleString()}</div>
                                <div className="text-xs text-muted-foreground mt-1">Intent clusters ready for idea generation</div>
                            </motion.div>

                            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }} className="bg-muted/20 border border-border rounded-xl p-5">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-sm text-muted-foreground">Last Run</span>
                                    <RefreshCw className="h-4 w-4 text-muted-foreground" />
                                </div>
                                <div className="text-sm font-semibold text-foreground capitalize">{keywordResearchRun?.status || 'unknown'}</div>
                                <div className="text-xs text-muted-foreground mt-1">{formatDateTime(keywordResearchSummary.generatedAt)}</div>
                            </motion.div>
                        </div>

                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                            <div className="rounded-xl border border-border bg-muted/20 p-5">
                                <div className="mb-3 flex items-center justify-between gap-3">
                                    <div>
                                        <h3 className="text-sm font-semibold text-foreground">Top Keyword Opportunities</h3>
                                        <p className="mt-1 text-xs text-muted-foreground">
                                            Use strong existing keywords as manual seeds for your next run when the current topic drifts.
                                        </p>
                                    </div>
                                    <div className="flex items-center gap-2">
                                        {allKeywordCandidates.length > 0 && (
                                            <span className="text-xs text-muted-foreground">
                                                Showing {Math.min(topKeywordRows.length, 8)} of {allKeywordCandidates.length}
                                            </span>
                                        )}
                                        {allKeywordCandidates.length > 8 && (
                                            <Button
                                                type="button"
                                                variant="outline"
                                                size="sm"
                                                onClick={() => setShowAllKeywords((current) => !current)}
                                                className="h-7 border-border px-2 text-[11px] hover:bg-muted"
                                            >
                                                {showAllKeywords ? 'Hide Full List' : `View All ${allKeywordCandidates.length}`}
                                            </Button>
                                        )}
                                    </div>
                                </div>
                                <div className="space-y-3">
                                    {topKeywordRows.map((row) => (
                                        <div key={row.id} className="rounded-lg border border-border/70 bg-background/40 p-3">
                                            <div className="flex items-start justify-between gap-3">
                                                <div>
                                                    <div className="text-sm font-medium text-foreground">{row.keyword}</div>
                                                    <div className="text-xs text-muted-foreground mt-1">
                                                        Intent: {row.intent_label || 'unknown'}
                                                    </div>
                                                </div>
                                                <div className="flex flex-col items-end gap-2">
                                                    <span className="rounded-full border border-primary/20 bg-primary/10 px-2 py-1 text-[11px] text-primary">
                                                        Score {Math.round(Number(row.opportunity_score || 0))}
                                                    </span>
                                                    <Button
                                                        type="button"
                                                        variant="outline"
                                                        size="sm"
                                                        onClick={() => onAddManualSeed(row.keyword)}
                                                        className="h-7 border-border px-2 text-[11px] hover:bg-muted"
                                                    >
                                                        Use as Seed
                                                    </Button>
                                                </div>
                                            </div>
                                            <div className="mt-2 text-xs text-muted-foreground">
                                                {(row.search_volume || 0).toLocaleString()} volume
                                                {row.keyword_difficulty != null ? ` • KD ${Math.round(Number(row.keyword_difficulty || 0))}` : ''}
                                                {row.cpc != null ? ` • $${Number(row.cpc || 0).toFixed(2)} CPC` : ''}
                                            </div>
                                        </div>
                                    ))}
                                    {keywordCandidates.length === 0 && (
                                        <p className="text-sm text-muted-foreground">No keyword candidates saved yet.</p>
                                    )}
                                </div>

                                {allKeywordCandidates.length > 0 && (
                                    <div className="mt-5 rounded-lg border border-border/70 bg-background/30 p-3">
                                        <div className="mb-3 flex items-center justify-between gap-3">
                                            <div>
                                                <h4 className="text-sm font-medium text-foreground">All Discovered Keywords</h4>
                                                <p className="mt-1 text-xs text-muted-foreground">
                                                    Review every keyword returned by the run, including filtered keywords and their filter reasons.
                                                </p>
                                            </div>
                                            <span className="text-xs text-muted-foreground">
                                                Showing {visibleAllKeywords.length} of {allKeywordCandidates.length}
                                            </span>
                                        </div>
                                        <div className="space-y-2">
                                            {visibleAllKeywords.map((row) => {
                                                const isFiltered = Boolean(row.is_filtered_out)
                                                return (
                                                    <div key={`all-${row.id}`} className="rounded-lg border border-border/60 bg-background/40 p-3">
                                                        <div className="flex items-start justify-between gap-3">
                                                            <div>
                                                                <div className="text-sm font-medium text-foreground">{row.keyword}</div>
                                                                <div className="mt-1 flex flex-wrap gap-2 text-[11px] text-muted-foreground">
                                                                    <span>Intent: {row.intent_label || 'unknown'}</span>
                                                                    <span>{(row.search_volume || 0).toLocaleString()} volume</span>
                                                                    {row.keyword_difficulty != null && <span>KD {Math.round(Number(row.keyword_difficulty || 0))}</span>}
                                                                    {row.cpc != null && <span>${Number(row.cpc || 0).toFixed(2)} CPC</span>}
                                                                </div>
                                                                {isFiltered && row.filter_reason && (
                                                                    <div className="mt-2 text-[11px] text-amber-300">
                                                                        Filtered out: {String(row.filter_reason).replace(/_/g, ' ')}
                                                                    </div>
                                                                )}
                                                            </div>
                                                            <div className="flex flex-col items-end gap-2">
                                                                <span className={`rounded-full border px-2 py-1 text-[11px] ${
                                                                    isFiltered
                                                                        ? 'border-amber-500/20 bg-amber-500/10 text-amber-300'
                                                                        : 'border-emerald-500/20 bg-emerald-500/10 text-emerald-400'
                                                                }`}>
                                                                    {isFiltered ? 'Filtered' : 'Active'}
                                                                </span>
                                                                <Button
                                                                    type="button"
                                                                    variant="outline"
                                                                    size="sm"
                                                                    onClick={() => onAddManualSeed(row.keyword)}
                                                                    className="h-7 border-border px-2 text-[11px] hover:bg-muted"
                                                                >
                                                                    Use as Seed
                                                                </Button>
                                                            </div>
                                                        </div>
                                                    </div>
                                                )
                                            })}
                                        </div>
                                        {!showAllKeywords && allKeywordCandidates.length > 20 && (
                                            <div className="mt-3 flex justify-center">
                                                <Button
                                                    type="button"
                                                    variant="outline"
                                                    size="sm"
                                                    onClick={() => setShowAllKeywords(true)}
                                                    className="border-border hover:bg-muted"
                                                >
                                                    View Remaining {allKeywordCandidates.length - 20} Keywords
                                                </Button>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>

                            <div className="rounded-xl border border-border bg-muted/20 p-5">
                                <div className="mb-3 flex items-center justify-between gap-3">
                                    <h3 className="text-sm font-semibold text-foreground">Top Intent Clusters</h3>
                                    <div className="flex items-center gap-2">
                                        {keywordClusters.length > 0 && (
                                            <span className="text-xs text-muted-foreground">
                                                {selectedClusterIds.size} selected
                                            </span>
                                        )}
                                        {keywordClusters.length > 0 && (
                                            <Button
                                                onClick={onGenerateIdeasFromClusters}
                                                disabled={generatingClusterIdeas || selectedClusterIds.size === 0 || !canGenerateIdeas}
                                                variant="outline"
                                                size="sm"
                                                className="border-emerald-500/30 bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/15"
                                            >
                                                {generatingClusterIdeas ? (
                                                    <>
                                                        <RefreshCw className="mr-2 h-3 w-3 animate-spin" />
                                                        Generating...
                                                    </>
                                                ) : (
                                                    <>
                                                        <Sparkles className="mr-2 h-3 w-3" />
                                                        Generate Ideas
                                                    </>
                                                )}
                                            </Button>
                                        )}
                                    </div>
                                </div>
                                {keywordClusters.length > 0 && (
                                    <div className="mb-3 rounded-lg border border-emerald-500/20 bg-emerald-500/10 p-3">
                                        <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                                            <div className="text-xs text-muted-foreground">
                                                {selectedClusterIds.size > 0
                                                    ? `${selectedClusterIds.size} cluster${selectedClusterIds.size === 1 ? '' : 's'} selected. Generate article and software ideas from the selected intent clusters.`
                                                    : 'Select one or more clusters below, then generate article and software ideas from that selection.'}
                                            </div>
                                            <Button
                                                onClick={onGenerateIdeasFromClusters}
                                                disabled={generatingClusterIdeas || selectedClusterIds.size === 0 || !canGenerateIdeas}
                                                className="bg-emerald-600 hover:bg-emerald-500 text-white"
                                                size="sm"
                                            >
                                                {generatingClusterIdeas ? (
                                                    <>
                                                        <RefreshCw className="mr-2 h-3 w-3 animate-spin" />
                                                        Generating Ideas...
                                                    </>
                                                ) : (
                                                    <>
                                                        <Sparkles className="mr-2 h-3 w-3" />
                                                        Generate Article + Software Ideas From {selectedClusterIds.size} Cluster{selectedClusterIds.size === 1 ? '' : 's'}
                                                    </>
                                                )}
                                            </Button>
                                        </div>
                                    </div>
                                )}
                                <div className="space-y-3">
                                    {keywordClusters.slice(0, 4).map((cluster) => (
                                        <button
                                            key={cluster.id}
                                            type="button"
                                            onClick={() => onToggleClusterSelection(cluster.id)}
                                            className={`w-full rounded-lg border p-3 text-left transition ${
                                                selectedClusterIds.has(cluster.id)
                                                    ? 'border-emerald-500/40 bg-emerald-500/10'
                                                    : 'border-border/70 bg-background/40 hover:border-ring/40'
                                            }`}
                                        >
                                            <div className="flex items-start justify-between gap-3">
                                                <div>
                                                    <div className="text-sm font-medium text-foreground">{cluster.cluster_name}</div>
                                                    <div className="text-xs text-muted-foreground mt-1">
                                                        Primary keyword: {cluster.primary_keyword || 'Not set'}
                                                    </div>
                                                </div>
                                                <span className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-2 py-1 text-[11px] text-emerald-400">
                                                    {Math.round(Number(cluster.opportunity_score || 0))} score
                                                </span>
                                            </div>
                                            <div className="mt-2 text-[11px] text-muted-foreground">
                                                {selectedClusterIds.has(cluster.id) ? 'Selected for generation' : 'Click to select this cluster'}
                                            </div>
                                            {cluster.secondary_keywords_json?.length > 0 && (
                                                <div className="mt-2 flex flex-wrap gap-1.5">
                                                    {cluster.secondary_keywords_json.slice(0, 4).map((keyword) => (
                                                        <span key={`${cluster.id}-${keyword}`} className="rounded-full border border-border bg-muted/40 px-2 py-0.5 text-[11px] text-muted-foreground">
                                                            {keyword}
                                                        </span>
                                                    ))}
                                                </div>
                                            )}
                                        </button>
                                    ))}
                                    {keywordClusters.length === 0 && (
                                        <p className="text-sm text-muted-foreground">No clusters saved yet.</p>
                                    )}
                                </div>
                            </div>
                        </div>
                            </div>
                        )}
                    </>
                )}
            </div>
        </div>
    )
}
