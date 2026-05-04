import { LibraryBig, RefreshCw, Sparkles, X } from "lucide-react"

import { Button } from "@/components/ui/button"
import type { ContentIdea } from "@/types/idea-burst"

type GeneratedIdeaTypeFilter = 'all' | 'blog' | 'software'
type GeneratedIdeaStatusFilter = 'all' | 'draft' | 'published'
type GeneratedIdeaSort = 'score' | 'volume' | 'difficulty' | 'recent'

interface GeneratedIdeasPanelProps {
    generatedClusterIdeas: ContentIdea[]
    visibleGeneratedClusterIdeas: ContentIdea[]
    selectedGeneratedIdeaIds: Set<string>
    publishingGeneratedIdeas: boolean
    generatedIdeaTypeFilter: GeneratedIdeaTypeFilter
    generatedIdeaStatusFilter: GeneratedIdeaStatusFilter
    generatedIdeaSort: GeneratedIdeaSort
    setGeneratedIdeaTypeFilter: (value: GeneratedIdeaTypeFilter) => void
    setGeneratedIdeaStatusFilter: (value: GeneratedIdeaStatusFilter) => void
    setGeneratedIdeaSort: (value: GeneratedIdeaSort) => void
    activeGeneratedIdea: ContentIdea | null
    activeGeneratedIdeaIndex: number
    hasPreviousGeneratedIdea: boolean
    hasNextGeneratedIdea: boolean
    canPublish: boolean
    openGeneratedIdeaDetail: (idea: ContentIdea) => void
    closeGeneratedIdeaDetail: () => void
    toggleGeneratedIdeaSelection: (ideaId: string) => void
    handlePublishGeneratedIdeas: () => void
    handlePublishSingleGeneratedIdea: (ideaId: string) => void
    handleSelectVisibleGeneratedIdeas: () => void
    handleSelectTopGeneratedIdeas: (count: number) => void
    handleClearGeneratedIdeaSelection: () => void
    openPreviousGeneratedIdea: () => void
    openNextGeneratedIdea: () => void
}

const formatMetricValue = (value?: number | null, decimals = 0) => {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return 'N/A'
    }
    if (decimals > 0) {
        return Number(value).toLocaleString(undefined, {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals,
        })
    }
    return Number(value).toLocaleString()
}

const isIdeaPublished = (idea: ContentIdea) =>
    Boolean(
        idea.published ||
        idea.published_to_titles ||
        idea.titles_record_id ||
        idea.status?.toLowerCase() === 'published'
    )

export function GeneratedIdeasPanel({
    generatedClusterIdeas,
    visibleGeneratedClusterIdeas,
    selectedGeneratedIdeaIds,
    publishingGeneratedIdeas,
    generatedIdeaTypeFilter,
    generatedIdeaStatusFilter,
    generatedIdeaSort,
    setGeneratedIdeaTypeFilter,
    setGeneratedIdeaStatusFilter,
    setGeneratedIdeaSort,
    activeGeneratedIdea,
    activeGeneratedIdeaIndex,
    hasPreviousGeneratedIdea,
    hasNextGeneratedIdea,
    canPublish,
    openGeneratedIdeaDetail,
    closeGeneratedIdeaDetail,
    toggleGeneratedIdeaSelection,
    handlePublishGeneratedIdeas,
    handlePublishSingleGeneratedIdea,
    handleSelectVisibleGeneratedIdeas,
    handleSelectTopGeneratedIdeas,
    handleClearGeneratedIdeaSelection,
    openPreviousGeneratedIdea,
    openNextGeneratedIdea,
}: GeneratedIdeasPanelProps) {
    return (
        <>
            <div className="max-w-7xl mx-auto mb-8">
                <div className="bg-muted/30 backdrop-blur-md border border-border rounded-2xl p-6">
                    <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between mb-6">
                        <div>
                            <h2 className="text-xl font-semibold text-foreground">Generated Ideas From Keyword Clusters</h2>
                            <p className="text-sm text-muted-foreground mt-1">
                                Review the ideas created by the new topic-level keyword pipeline and publish the strongest ones into Content Studio.
                            </p>
                        </div>
                        {generatedClusterIdeas.length > 0 && (
                            <Button
                                onClick={handlePublishGeneratedIdeas}
                                disabled={publishingGeneratedIdeas || selectedGeneratedIdeaIds.size === 0 || !canPublish}
                                className="bg-emerald-600 hover:bg-emerald-500 text-white"
                            >
                                {publishingGeneratedIdeas ? (
                                    <>
                                        <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                                        Publishing...
                                    </>
                                ) : (
                                    <>
                                        <LibraryBig className="mr-2 h-4 w-4" />
                                        Publish {selectedGeneratedIdeaIds.size} Idea{selectedGeneratedIdeaIds.size === 1 ? '' : 's'}
                                    </>
                                )}
                            </Button>
                        )}
                    </div>

                    {generatedClusterIdeas.length === 0 ? (
                        <div className="rounded-xl border border-dashed border-border bg-muted/20 p-6 text-center">
                            <Sparkles className="mx-auto mb-3 h-10 w-10 text-muted-foreground" />
                            <h3 className="text-base font-semibold text-foreground mb-2">No cluster-generated ideas yet</h3>
                            <p className="text-sm text-muted-foreground max-w-2xl mx-auto">
                                Run topic keyword research, select clusters, and generate ideas to start using the new pipeline end to end.
                            </p>
                        </div>
                    ) : (
                        <div className="space-y-4">
                            <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                                <div className="flex items-center gap-3">
                                    <span className="text-sm text-muted-foreground">
                                        {visibleGeneratedClusterIdeas.length} visible idea{visibleGeneratedClusterIdeas.length === 1 ? '' : 's'}
                                        {visibleGeneratedClusterIdeas.length !== generatedClusterIdeas.length ? ` of ${generatedClusterIdeas.length}` : ''}
                                    </span>
                                    <span className="text-xs text-muted-foreground">
                                        {selectedGeneratedIdeaIds.size} selected
                                    </span>
                                </div>
                                <div className="flex flex-wrap items-center gap-2">
                                    <select
                                        value={generatedIdeaStatusFilter}
                                        onChange={(event) => setGeneratedIdeaStatusFilter(event.target.value as GeneratedIdeaStatusFilter)}
                                        className="rounded-lg border border-border bg-background/70 px-3 py-2 text-xs text-foreground outline-none transition focus:border-ring"
                                    >
                                        <option value="draft">Drafts Only</option>
                                        <option value="published">Published Only</option>
                                        <option value="all">All Statuses</option>
                                    </select>
                                    <select
                                        value={generatedIdeaTypeFilter}
                                        onChange={(event) => setGeneratedIdeaTypeFilter(event.target.value as GeneratedIdeaTypeFilter)}
                                        className="rounded-lg border border-border bg-background/70 px-3 py-2 text-xs text-foreground outline-none transition focus:border-ring"
                                    >
                                        <option value="all">All Types</option>
                                        <option value="blog">Articles Only</option>
                                        <option value="software">Software Only</option>
                                    </select>
                                    <select
                                        value={generatedIdeaSort}
                                        onChange={(event) => setGeneratedIdeaSort(event.target.value as GeneratedIdeaSort)}
                                        className="rounded-lg border border-border bg-background/70 px-3 py-2 text-xs text-foreground outline-none transition focus:border-ring"
                                    >
                                        <option value="score">Sort: Highest Score</option>
                                        <option value="volume">Sort: Highest Volume</option>
                                        <option value="difficulty">Sort: Lowest Difficulty</option>
                                        <option value="recent">Sort: Newest First</option>
                                    </select>
                                </div>
                            </div>

                            <div className="flex flex-wrap items-center gap-2">
                                <Button
                                    type="button"
                                    variant="outline"
                                    size="sm"
                                    onClick={handleSelectVisibleGeneratedIdeas}
                                    disabled={visibleGeneratedClusterIdeas.length === 0}
                                    className="border-border hover:bg-muted"
                                >
                                    Select Visible
                                </Button>
                                <Button
                                    type="button"
                                    variant="outline"
                                    size="sm"
                                    onClick={() => handleSelectTopGeneratedIdeas(5)}
                                    disabled={visibleGeneratedClusterIdeas.length === 0}
                                    className="border-border hover:bg-muted"
                                >
                                    Select Top 5
                                </Button>
                                <Button
                                    type="button"
                                    variant="outline"
                                    size="sm"
                                    onClick={handleClearGeneratedIdeaSelection}
                                    disabled={selectedGeneratedIdeaIds.size === 0}
                                    className="border-border hover:bg-muted"
                                >
                                    Clear Selection
                                </Button>
                            </div>

                            {visibleGeneratedClusterIdeas.length === 0 ? (
                                <div className="rounded-xl border border-dashed border-border bg-muted/20 p-6 text-center">
                                    <h3 className="text-base font-semibold text-foreground mb-2">No ideas match this view</h3>
                                    <p className="text-sm text-muted-foreground max-w-2xl mx-auto">
                                        Change the type filter or sorting to see more generated ideas.
                                    </p>
                                </div>
                            ) : (
                                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                                    {visibleGeneratedClusterIdeas.map((idea) => {
                                        const metadata = (idea.idea_metadata || {}) as any
                                        const topicKeywordResearch = metadata?.topic_keyword_research || {}
                                        const isSelected = selectedGeneratedIdeaIds.has(idea.id)
                                        const published = isIdeaPublished(idea)
                                        const keywordMetricCount = Object.keys((idea.keyword_metrics || {}) as Record<string, unknown>).length

                                        return (
                                            <button
                                                key={idea.id}
                                                type="button"
                                                onClick={() => openGeneratedIdeaDetail(idea)}
                                                className={`w-full rounded-xl border p-4 text-left transition ${
                                                    isSelected
                                                        ? 'border-emerald-500/40 bg-emerald-500/10'
                                                        : 'border-border bg-background/40 hover:border-ring/40'
                                                }`}
                                            >
                                                <div className="flex items-start justify-between gap-3">
                                                    <div>
                                                        <div className="text-sm font-semibold text-foreground">{idea.title}</div>
                                                        <div className="mt-1 text-xs text-muted-foreground">
                                                            {idea.content_type === 'software' ? 'Software Idea' : 'Article Idea'}
                                                            {topicKeywordResearch?.cluster_name ? ` • ${topicKeywordResearch.cluster_name}` : ''}
                                                        </div>
                                                    </div>
                                                    <div className="flex flex-col items-end gap-1">
                                                        {published ? (
                                                            <span className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-2 py-1 text-[11px] text-emerald-400">
                                                                Published
                                                            </span>
                                                        ) : (
                                                            <span className="rounded-full border border-primary/20 bg-primary/10 px-2 py-1 text-[11px] text-primary">
                                                                Draft
                                                            </span>
                                                        )}
                                                        <button
                                                            type="button"
                                                            onClick={(event) => {
                                                                event.stopPropagation()
                                                                toggleGeneratedIdeaSelection(idea.id)
                                                            }}
                                                            className={`rounded-full border px-2 py-1 text-[11px] transition ${
                                                                isSelected
                                                                    ? 'border-emerald-500/20 bg-emerald-500/10 text-emerald-400'
                                                                    : 'border-border bg-background/60 text-muted-foreground hover:bg-muted/50'
                                                            }`}
                                                        >
                                                            {isSelected ? 'Selected' : 'Select'}
                                                        </button>
                                                    </div>
                                                </div>
                                                {idea.description && (
                                                    <p className="mt-3 text-sm text-muted-foreground line-clamp-3">
                                                        {idea.description}
                                                    </p>
                                                )}
                                                <div className="mt-3 flex flex-wrap gap-1.5">
                                                    {(idea.primary_keywords || []).slice(0, 4).map((keyword) => (
                                                        <span
                                                            key={`${idea.id}-${keyword}`}
                                                            className="rounded-full border border-border bg-muted/40 px-2 py-0.5 text-[11px] text-muted-foreground"
                                                        >
                                                            {keyword}
                                                        </span>
                                                    ))}
                                                </div>
                                                <div className="mt-3 flex flex-wrap gap-1.5">
                                                    <span className="rounded-full border border-border bg-muted/30 px-2 py-0.5 text-[11px] text-muted-foreground">
                                                        Volume {formatMetricValue(idea.total_search_volume)}
                                                    </span>
                                                    <span className="rounded-full border border-border bg-muted/30 px-2 py-0.5 text-[11px] text-muted-foreground">
                                                        Difficulty {formatMetricValue(idea.average_difficulty, 1)}
                                                    </span>
                                                    <span className="rounded-full border border-border bg-muted/30 px-2 py-0.5 text-[11px] text-muted-foreground">
                                                        CPC ${formatMetricValue(idea.average_cpc, 2)}
                                                    </span>
                                                    <span className="rounded-full border border-border bg-muted/30 px-2 py-0.5 text-[11px] text-muted-foreground">
                                                        Metrics {keywordMetricCount}
                                                    </span>
                                                </div>
                                                <div className="mt-3 text-[11px] text-muted-foreground">
                                                    {isSelected ? 'Selected for publishing' : 'Open to review'}
                                                    {idea.opportunity_score ? ` • Score ${idea.opportunity_score}` : ''}
                                                </div>
                                            </button>
                                        )
                                    })}
                                </div>
                            )}
                        </div>
                    )}
                </div>
            </div>

            {activeGeneratedIdea && (
                <div className="fixed inset-0 z-50 flex items-end justify-end bg-black/50 backdrop-blur-sm md:items-stretch">
                    <button
                        type="button"
                        aria-label="Close generated idea detail"
                        onClick={closeGeneratedIdeaDetail}
                        className="absolute inset-0"
                    />
                    <div className="relative z-10 flex h-[88vh] w-full max-w-2xl flex-col border-l border-border bg-background shadow-2xl">
                        <div className="flex items-start justify-between gap-4 border-b border-border px-6 py-5">
                            <div>
                                <div className="text-xs uppercase tracking-wide text-muted-foreground">
                                    {activeGeneratedIdea.content_type === 'software' ? 'Software Idea' : 'Article Idea'}
                                </div>
                                <h3 className="mt-1 text-xl font-semibold text-foreground">
                                    {activeGeneratedIdea.title}
                                </h3>
                                {activeGeneratedIdeaIndex >= 0 && (
                                    <div className="mt-2 text-xs text-muted-foreground">
                                        Idea {activeGeneratedIdeaIndex + 1} of {visibleGeneratedClusterIdeas.length} in current view
                                    </div>
                                )}
                                <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-muted-foreground">
                                    {((activeGeneratedIdea.idea_metadata || {}) as any)?.topic_keyword_research?.cluster_name && (
                                        <span className="rounded-full border border-border bg-muted/30 px-2 py-1">
                                            {((activeGeneratedIdea.idea_metadata || {}) as any).topic_keyword_research.cluster_name}
                                        </span>
                                    )}
                                    {activeGeneratedIdea.opportunity_score != null && (
                                        <span className="rounded-full border border-border bg-muted/30 px-2 py-1">
                                            Score {activeGeneratedIdea.opportunity_score}
                                        </span>
                                    )}
                                    <span className="rounded-full border border-border bg-muted/30 px-2 py-1">
                                        Volume {formatMetricValue(activeGeneratedIdea.total_search_volume)}
                                    </span>
                                    <span className="rounded-full border border-border bg-muted/30 px-2 py-1">
                                        Difficulty {formatMetricValue(activeGeneratedIdea.average_difficulty, 1)}
                                    </span>
                                    <span className="rounded-full border border-border bg-muted/30 px-2 py-1">
                                        CPC ${formatMetricValue(activeGeneratedIdea.average_cpc, 2)}
                                    </span>
                                </div>
                            </div>
                            <Button
                                type="button"
                                variant="ghost"
                                size="icon"
                                onClick={closeGeneratedIdeaDetail}
                                className="text-muted-foreground hover:text-foreground"
                            >
                                <X className="h-5 w-5" />
                            </Button>
                        </div>

                        <div className="flex-1 overflow-y-auto px-6 py-5">
                            {activeGeneratedIdea.description && (
                                <div className="mb-6">
                                    <h4 className="mb-2 text-sm font-semibold text-foreground">Summary</h4>
                                    <p className="text-sm leading-6 text-muted-foreground">
                                        {activeGeneratedIdea.description}
                                    </p>
                                </div>
                            )}

                            <div className="mb-6">
                                <h4 className="mb-2 text-sm font-semibold text-foreground">Keywords</h4>
                                <div className="flex flex-wrap gap-2">
                                    {(activeGeneratedIdea.primary_keywords || activeGeneratedIdea.keywords || []).map((keyword) => (
                                        <span
                                            key={`${activeGeneratedIdea.id}-detail-${keyword}`}
                                            className="rounded-full border border-border bg-muted/30 px-2.5 py-1 text-xs text-muted-foreground"
                                        >
                                            {keyword}
                                        </span>
                                    ))}
                                </div>
                            </div>

                            <div className="mb-6 grid grid-cols-2 gap-3">
                                <div className="rounded-xl border border-border bg-muted/20 p-4">
                                    <div className="text-xs uppercase tracking-wide text-muted-foreground">Metric Coverage</div>
                                    <div className="mt-1 text-lg font-semibold text-foreground">
                                        {Object.keys((activeGeneratedIdea.keyword_metrics || {}) as Record<string, unknown>).length}
                                    </div>
                                    <div className="text-xs text-muted-foreground mt-1">Keywords with attached metrics</div>
                                </div>
                                <div className="rounded-xl border border-border bg-muted/20 p-4">
                                    <div className="text-xs uppercase tracking-wide text-muted-foreground">Status</div>
                                    <div className="mt-1 text-lg font-semibold text-foreground">
                                        {isIdeaPublished(activeGeneratedIdea) ? 'Published' : 'Draft'}
                                    </div>
                                    <div className="text-xs text-muted-foreground mt-1">Current publishing state</div>
                                </div>
                            </div>

                            <div className="mb-6">
                                <h4 className="mb-3 text-sm font-semibold text-foreground">Keyword Metrics</h4>
                                {Object.entries(activeGeneratedIdea.keyword_metrics || {}).length === 0 ? (
                                    <div className="rounded-xl border border-dashed border-border bg-muted/20 p-4 text-sm text-muted-foreground">
                                        No keyword-level metrics were attached to this idea.
                                    </div>
                                ) : (
                                    <div className="space-y-2">
                                        {Object.entries(activeGeneratedIdea.keyword_metrics || {}).map(([keyword, metrics]) => (
                                            <div
                                                key={`${activeGeneratedIdea.id}-metric-${keyword}`}
                                                className="rounded-xl border border-border bg-muted/20 p-4"
                                            >
                                                <div className="text-sm font-medium text-foreground">{keyword}</div>
                                                <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-muted-foreground">
                                                    <span className="rounded-full border border-border bg-background/50 px-2 py-1">
                                                        Volume {formatMetricValue(metrics.search_volume)}
                                                    </span>
                                                    <span className="rounded-full border border-border bg-background/50 px-2 py-1">
                                                        Difficulty {formatMetricValue(metrics.keyword_difficulty, 1)}
                                                    </span>
                                                    <span className="rounded-full border border-border bg-background/50 px-2 py-1">
                                                        CPC ${formatMetricValue(metrics.cpc, 2)}
                                                    </span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>

                        <div className="border-t border-border px-6 py-4">
                            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                                <div className="text-xs text-muted-foreground">
                                    Review the metrics, then publish this idea directly or add it to the current batch.
                                </div>
                                <div className="flex items-center gap-2">
                                    <Button
                                        type="button"
                                        variant="outline"
                                        onClick={openPreviousGeneratedIdea}
                                        disabled={!hasPreviousGeneratedIdea}
                                        className="border-border hover:bg-muted"
                                    >
                                        Previous
                                    </Button>
                                    <Button
                                        type="button"
                                        variant="outline"
                                        onClick={openNextGeneratedIdea}
                                        disabled={!hasNextGeneratedIdea}
                                        className="border-border hover:bg-muted"
                                    >
                                        Next Draft
                                    </Button>
                                    <Button
                                        type="button"
                                        variant="outline"
                                        onClick={closeGeneratedIdeaDetail}
                                        className="border-border hover:bg-muted"
                                    >
                                        Close
                                    </Button>
                                    {!isIdeaPublished(activeGeneratedIdea) && (
                                        <Button
                                            type="button"
                                            onClick={() => handlePublishSingleGeneratedIdea(activeGeneratedIdea.id)}
                                            disabled={publishingGeneratedIdeas}
                                            className="bg-emerald-600 hover:bg-emerald-500 text-white"
                                        >
                                            {publishingGeneratedIdeas ? 'Publishing...' : 'Publish This Idea'}
                                        </Button>
                                    )}
                                    <Button
                                        type="button"
                                        onClick={() => toggleGeneratedIdeaSelection(activeGeneratedIdea.id)}
                                        disabled={isIdeaPublished(activeGeneratedIdea)}
                                        className={
                                            selectedGeneratedIdeaIds.has(activeGeneratedIdea.id)
                                                ? 'bg-emerald-600 hover:bg-emerald-500 text-white'
                                                : 'bg-primary hover:bg-primary/90 text-primary-foreground'
                                        }
                                    >
                                        {isIdeaPublished(activeGeneratedIdea)
                                            ? 'Already Published'
                                            : selectedGeneratedIdeaIds.has(activeGeneratedIdea.id)
                                            ? 'Selected for Publish'
                                            : 'Select for Publish'}
                                    </Button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </>
    )
}
