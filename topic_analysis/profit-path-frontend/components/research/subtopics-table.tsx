'use client'

import * as React from 'react'
import { useRouter } from 'next/navigation'
import {
    ArrowUpDown,
    Check,
    ChevronDown,
    ChevronRight,
    Loader2,
    MoreHorizontal,
    Search,
    Sparkles,
    TrendingUp,
    TrendingDown,
    Minus,
    ExternalLink,
    LineChart,
    DollarSign,
    Pencil,
    Trash2
} from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Checkbox } from '@/components/ui/checkbox'
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { SparklineChart } from './sparkline-chart'
import { ViabilityScore } from './viability-score'
import { Subtopic } from '@/types/research'

interface SubtopicsTableProps {
    subtopics: Subtopic[]
    topicId: string
    selectedSubtopics: Set<string>
    onToggle: (id: string) => void
    onToggleAll: () => void
    onGenerate?: () => void
    onVerify?: (id: string) => void
    onEnrich?: (id: string) => Promise<void>
    onEdit?: (id: string) => void
    onDelete?: (id: string) => void
    isGenerating?: boolean
    keywordsUpdated?: number
}

type SortField = 'name' | 'search_volume' | 'seo_difficulty' | 'viability_score' | 'trend_direction' | 'affiliate_offer_count'
type SortDirection = 'asc' | 'desc'

import { AffiliateOffersModal } from './affiliate-offers-modal'


export function SubtopicsTable({
    subtopics,
    topicId,
    selectedSubtopics,
    onToggle,
    onToggleAll,
    onGenerate,
    onVerify,
    onEnrich,
    onEdit,
    onDelete,
    isGenerating = false,
    keywordsUpdated
}: SubtopicsTableProps) {
    const router = useRouter()
    const [sortField, setSortField] = React.useState<SortField>('viability_score')
    const [sortDirection, setSortDirection] = React.useState<SortDirection>('desc')
    // Local selection state removed in favor of props
    const [enrichingIds, setEnrichingIds] = React.useState<Set<string>>(new Set())
    const [expandedSubtopic, setExpandedSubtopic] = React.useState<string | null>(null)

    // State for Affiliate Offers Modal
    const [offersModalOpen, setOffersModalOpen] = React.useState(false)
    const [selectedSubtopicForOffers, setSelectedSubtopicForOffers] = React.useState<Subtopic | null>(null)

    // Sync selectedSubtopicForOffers with subtopics prop to ensure fresh data in modal (Fixes stale state after enrichment)
    React.useEffect(() => {
        if (selectedSubtopicForOffers) {
            const updatedSubtopic = subtopics.find(s => s.id === selectedSubtopicForOffers.id)
            if (updatedSubtopic) {
                setSelectedSubtopicForOffers(updatedSubtopic)
            }
        }
    }, [subtopics, selectedSubtopicForOffers?.id])

    const toggleExpand = (id: string, e: React.MouseEvent) => {
        e.stopPropagation()
        setExpandedSubtopic(expandedSubtopic === id ? null : id)
    }

    const handleSort = (field: SortField) => {
        if (sortField === field) {
            setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
        } else {
            setSortField(field)
            setSortDirection('desc')
        }
    }

    const handleEnrichClick = async (e: React.MouseEvent, id: string) => {
        e.stopPropagation()
        if (!onEnrich) return

        setEnrichingIds(prev => new Set(prev).add(id))
        try {
            await onEnrich(id)
        } finally {
            setEnrichingIds(prev => {
                const next = new Set(prev)
                next.delete(id)
                return next
            })
        }
    }

    const handleOffersClick = (e: React.MouseEvent, subtopic: Subtopic) => {
        e.stopPropagation()
        // Only open if there are offers or if we want to show empty state explicitly
        setSelectedSubtopicForOffers(subtopic)
        setOffersModalOpen(true)
    }

    const sortedSubtopics = React.useMemo(() => {
        return [...subtopics].sort((a, b) => {
            let aValue: any = a[sortField]
            let bValue: any = b[sortField]

            // Handle nulls
            if (aValue === null) aValue = -1
            if (bValue === null) bValue = -1

            if (typeof aValue === 'string') {
                return sortDirection === 'asc'
                    ? aValue.localeCompare(bValue)
                    : bValue.localeCompare(aValue)
            }

            return sortDirection === 'asc'
                ? aValue - bValue
                : bValue - aValue
        })
    }, [subtopics, sortField, sortDirection])

    const getSEOBadgeVariant = (difficulty: number | null) => {
        if (difficulty === null) return 'secondary'
        if (difficulty <= 30) return 'default' // Green (Easy)
        if (difficulty <= 60) return 'secondary' // Orange (Moderate)
        return 'destructive' // Red (Hard)
    }

    const getSEOLabel = (difficulty: number | null) => {
        if (difficulty === null) return 'Unknown'
        if (difficulty <= 30) return 'Easy'
        if (difficulty <= 60) return 'Moderate'
        return 'Hard'
    }

    const formatNumber = (num: number | null) => {
        if (num === null || num === undefined) return '-'
        if (num >= 1000) {
            return `${(num / 1000).toFixed(1)}k`
        }
        return num.toString()
    }

    const formatCPC = (cpc: number | null) => {
        if (cpc === null) return '-'
        return `$${cpc.toFixed(2)}`
    }

    const getSparklineDirection = (direction: string | null): 'up' | 'down' | 'neutral' => {
        if (direction === 'up' || direction === 'down') return direction;
        return 'neutral';
    }

    const getStatus = (subtopic: Subtopic) => {
        if (enrichingIds.has(subtopic.id)) return 'enriching'
        if (subtopic.search_volume === null) return 'pending_enrichment'
        return 'complete'
    }

    if (subtopics.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center py-24 text-center">
                <div className="bg-muted rounded-full p-4 mb-4">
                    <TrendingUp className="h-8 w-8 text-muted-foreground" />
                </div>
                <h3 className="text-lg font-medium mb-2">No Subtopics Yet</h3>
                <p className="text-sm text-muted-foreground max-w-sm">
                    Subtopics will appear here once trend analysis is complete.
                </p>
                {onGenerate && (
                    <div className="mt-4">
                        <Button
                            onClick={onGenerate}
                            disabled={isGenerating}
                            size="lg"
                            className="bg-blue-600 hover:bg-blue-700 text-white shadow-lg shadow-blue-500/20"
                        >
                            {isGenerating ? (
                                <>
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                    Analyzing Topic...
                                </>
                            ) : (
                                <>
                                    <Sparkles className="mr-2 h-4 w-4" />
                                    Decompose Topic
                                </>
                            )}
                        </Button>
                    </div>
                )}
            </div>
        )
    }

    return (
        <div className="space-y-4">
            {/* Toolbar removed - hoisted to page.tsx */}

            <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 overflow-hidden bg-background">
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead className="bg-muted/30 border-b border-zinc-200 dark:border-zinc-800">
                            <tr>
                                <th className="w-[40px]"></th>
                                <th className="text-left p-4 w-[50px]"><Checkbox checked={selectedSubtopics.size === subtopics.length && subtopics.length > 0} onChange={onToggleAll} /></th>
                                <th
                                    className="text-left p-4 font-medium cursor-pointer hover:bg-muted/50 transition-colors"
                                    onClick={() => handleSort('name')}
                                >
                                    <div className="flex items-center gap-2">
                                        Subtopic Name
                                        <ArrowUpDown className="h-4 w-4 text-muted-foreground" />
                                    </div>
                                </th>

                                <th
                                    className="text-left p-4 font-medium cursor-pointer hover:bg-muted/50 transition-colors"
                                    onClick={() => handleSort('search_volume')}
                                >
                                    <div className="flex items-center gap-2">
                                        Volume & CPC
                                        <ArrowUpDown className="h-4 w-4 text-muted-foreground" />
                                    </div>
                                </th>
                                <th
                                    className="text-left p-4 font-medium cursor-pointer hover:bg-muted/50 transition-colors"
                                    onClick={() => handleSort('seo_difficulty')}
                                >
                                    <div className="flex items-center gap-2">
                                        SEO
                                        <ArrowUpDown className="h-4 w-4 text-muted-foreground" />
                                    </div>
                                </th>
                                <th
                                    className="text-left p-4 font-medium cursor-pointer hover:bg-muted/50 transition-colors hidden sm:table-cell"
                                    onClick={() => handleSort('affiliate_offer_count')}
                                >
                                    <div className="flex items-center gap-2">
                                        Offers
                                        <ArrowUpDown className="h-4 w-4 text-muted-foreground" />
                                    </div>
                                </th>
                                <th
                                    className="text-left p-4 font-medium cursor-pointer hover:bg-muted/50 transition-colors"
                                    onClick={() => handleSort('viability_score')}
                                >
                                    <div className="flex items-center gap-2">
                                        Viability
                                        <ArrowUpDown className="h-4 w-4 text-muted-foreground" />
                                    </div>
                                </th>
                                <th className="text-right p-4 font-medium w-[100px]">
                                    Actions
                                </th>
                            </tr>
                        </thead>
                        <tbody>
                            {sortedSubtopics.map((subtopic) => {
                                const hasTrendData = subtopic.interest_over_time && subtopic.interest_over_time.length > 0

                                // STRICT COUNT: Only show actual offers found, not categories
                                // If we have specific offers data, use that length
                                // Otherwise fallback to the persisted count (which should be 0 if no offers found)
                                // We explicitly avoid falling back to 'affiliate_categories' which caused the "2 vs 0" confusion
                                const offerCount = subtopic.monetization_data?.offers?.length ?? subtopic.affiliate_offer_count ?? 0

                                const hasOffers = offerCount > 0

                                return (
                                    <React.Fragment key={subtopic.id}>
                                        <tr
                                            className={`
                                            border-b border-zinc-200 dark:border-zinc-800 transition-colors
                                            hover:bg-muted/5 cursor-pointer
                                            ${selectedSubtopics.has(subtopic.id) ? 'bg-purple-50 dark:bg-purple-900/10' : ''}
                                            ${expandedSubtopic === subtopic.id ? 'bg-muted/10' : ''}
                                        `}
                                            onClick={(e) => toggleExpand(subtopic.id, e)}
                                        >
                                            <td className="p-4 w-[40px]">
                                                <Button variant="ghost" size="sm" className="h-6 w-6 p-0 hover:bg-muted/20">
                                                    {expandedSubtopic === subtopic.id ? (
                                                        <ChevronDown className="h-4 w-4" />
                                                    ) : (
                                                        <ChevronRight className="h-4 w-4" />
                                                    )}
                                                </Button>
                                            </td>
                                            <td className="p-4" onClick={(e) => e.stopPropagation()}>
                                                <Checkbox
                                                    checked={selectedSubtopics.has(subtopic.id)}
                                                    onChange={() => onToggle(subtopic.id)}
                                                />
                                            </td>
                                            <td className="p-4">
                                                <div className="flex items-center gap-2">
                                                    <div className="font-medium">{subtopic.name}</div>
                                                    {getStatus(subtopic) === 'enriching' && (
                                                        <Badge variant="outline" className="text-xs h-5 px-1 py-0 gap-1 border-blue-200 text-blue-700">
                                                            <Loader2 className="h-3 w-3 animate-spin" />
                                                            Fetching...
                                                        </Badge>
                                                    )}
                                                    {getStatus(subtopic) === 'pending_enrichment' && (
                                                        <Badge variant="outline" className="text-xs h-5 px-1 py-0 gap-1 text-muted-foreground opacity-60">
                                                            Pending Market Data
                                                        </Badge>
                                                    )}
                                                </div>
                                                <div className="flex flex-wrap gap-1 mt-1">
                                                    {(subtopic.keywords || []).slice(0, 2).map((keyword: any, i: any) => (
                                                        <span key={i} className="text-xs text-muted-foreground bg-secondary/10 px-1 rounded">
                                                            {typeof keyword === 'string' ? keyword : keyword.keyword}
                                                        </span>
                                                    ))}
                                                    {(subtopic.keywords || []).length > 2 && (
                                                        <span className="text-xs text-muted-foreground">+{(subtopic.keywords || []).length - 2}</span>
                                                    )}
                                                </div>
                                            </td>

                                            <td className="p-4">
                                                <div className="space-y-1">
                                                    <div className="font-medium">{formatNumber(subtopic.search_volume)}</div>
                                                    <div className="text-xs text-muted-foreground">{formatCPC(subtopic.cpc)}</div>
                                                </div>
                                            </td>
                                            <td className="p-4">
                                                <div className="space-y-1">
                                                    <Badge variant={getSEOBadgeVariant(subtopic.seo_difficulty)}>
                                                        {getSEOLabel(subtopic.seo_difficulty)}
                                                    </Badge>
                                                    {subtopic.seo_difficulty !== null && (
                                                        <div className="text-xs text-muted-foreground pl-1">
                                                            KD: {subtopic.seo_difficulty}
                                                        </div>
                                                    )}
                                                </div>
                                            </td>
                                            <td
                                                className="p-4 hidden sm:table-cell group"
                                                onClick={(e) => handleOffersClick(e, subtopic)}
                                            >
                                                <div className={`font-medium transition-colors ${hasOffers ? 'text-blue-600 group-hover:text-blue-800 cursor-pointer underline decoration-dotted underline-offset-4' : ''}`}>
                                                    {offerCount}
                                                </div>
                                            </td>
                                            <td className="p-4">
                                                <ViabilityScore score={subtopic.viability_score || 0} size={50} />
                                            </td>
                                            <td className="p-4 text-right">
                                                <div className="flex items-center justify-end gap-1">
                                                    <a
                                                        href={`https://trends.google.com/trends/explore?date=today%2012-m&geo=US&q=${encodeURIComponent(subtopic.name)}`}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        className="inline-flex items-center justify-center rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 hover:bg-accent hover:text-accent-foreground h-8 w-8 text-muted-foreground hover:text-blue-500"
                                                        title="Analyze on Google Trends"
                                                        onClick={(e) => e.stopPropagation()}
                                                    >
                                                        <LineChart className="h-4 w-4" />
                                                    </a>

                                                    {onEdit && (
                                                        <Button
                                                            variant="ghost"
                                                            size="icon"
                                                            className="h-8 w-8 text-muted-foreground hover:text-foreground"
                                                            onClick={(e) => { e.stopPropagation(); onEdit(subtopic.id); }}
                                                            title="Edit Subtopic"
                                                        >
                                                            <Pencil className="h-4 w-4" />
                                                        </Button>
                                                    )}
                                                    {onDelete && (
                                                        <Button
                                                            variant="ghost"
                                                            size="icon"
                                                            className="h-8 w-8 text-muted-foreground hover:text-destructive"
                                                            onClick={(e) => { e.stopPropagation(); onDelete(subtopic.id); }}
                                                            title="Remove Subtopic"
                                                        >
                                                            <Trash2 className="h-4 w-4" />
                                                        </Button>
                                                    )}
                                                </div>
                                            </td>
                                        </tr>

                                        {/* Expanded Row */}
                                        {expandedSubtopic === subtopic.id && (
                                            <tr className="bg-muted/5 animate-in fade-in zoom-in-95 duration-200">
                                                <td colSpan={9} className="p-0">
                                                    <div className="border-t border-zinc-200 dark:border-zinc-800 bg-muted/10 p-4 pl-12 shadow-inner">
                                                        <div className="rounded-md border border-zinc-200 dark:border-zinc-800 bg-background overflow-hidden">
                                                            <div className="bg-muted/50 px-4 py-2 border-b border-zinc-200 dark:border-zinc-800 flex justify-between items-center">
                                                                <h4 className="text-xs font-semibold text-muted-foreground flex items-center gap-2">
                                                                    <Sparkles className="h-3 w-3" />
                                                                    Verified Keywords ({(subtopic.keywords || []).length})
                                                                </h4>
                                                            </div>
                                                            <table className="w-full text-sm">
                                                                <thead className="bg-muted/20">
                                                                    <tr>
                                                                        <th className="text-left p-2 pl-4 font-medium text-xs text-muted-foreground w-1/3">Keyword</th>
                                                                        <th className="text-left p-2 font-medium text-xs text-muted-foreground">Vol</th>
                                                                        <th className="text-left p-2 font-medium text-xs text-muted-foreground">CPC</th>
                                                                        <th className="text-left p-2 font-medium text-xs text-muted-foreground">KD</th>
                                                                        <th className="text-left p-2 font-medium text-xs text-muted-foreground">Comp</th>
                                                                        <th className="text-left p-2 font-medium text-xs text-muted-foreground">Intent</th>
                                                                    </tr>
                                                                </thead>
                                                                <tbody>
                                                                    {(subtopic.keywords || []).map((kw: any, idx) => {
                                                                        // Handle both string and object formats gracefully
                                                                        const isObj = typeof kw === 'object' && kw !== null
                                                                        const text = isObj ? kw.keyword : kw
                                                                        const vol = isObj ? kw.search_volume : '-'
                                                                        const cpc = isObj ? kw.cpc : '-'
                                                                        const kd = isObj ? kw.keyword_difficulty : '-'
                                                                        const comp = isObj ? kw.competition : '-'
                                                                        const intent = isObj ? kw.intent : '-'

                                                                        return (
                                                                            <tr key={idx} className="border-b border-zinc-100 dark:border-zinc-800 last:border-0 hover:bg-muted/20">
                                                                                <td className="p-2 pl-4 font-medium">{text}</td>
                                                                                <td className="p-2 text-muted-foreground">{vol !== '-' ? formatNumber(vol) : vol}</td>
                                                                                <td className="p-2 text-muted-foreground">{cpc !== '-' ? formatCPC(cpc) : cpc}</td>
                                                                                <td className="p-2">
                                                                                    <Badge variant={typeof kd === 'number' ? getSEOBadgeVariant(kd) : 'secondary'} className="text-[10px] h-5 px-1">
                                                                                        {typeof kd === 'number' ? kd : '-'}
                                                                                    </Badge>
                                                                                </td>
                                                                                <td className="p-2 text-xs text-muted-foreground">{typeof comp === 'number' ? comp?.toFixed(2) : comp}</td>
                                                                                <td className="p-2 text-xs text-muted-foreground capitalize">
                                                                                    {intent !== '-' && (
                                                                                        <Badge variant="outline" className="text-[10px] h-5 font-normal opacity-70">
                                                                                            {intent}
                                                                                        </Badge>
                                                                                    )}
                                                                                </td>
                                                                            </tr>
                                                                        )
                                                                    })}
                                                                </tbody>
                                                            </table>
                                                        </div>
                                                    </div>
                                                </td>
                                            </tr>
                                        )}
                                    </React.Fragment>
                                )
                            })}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Affiliate Offers Modal */}
            <AffiliateOffersModal
                isOpen={offersModalOpen}
                onClose={() => setOffersModalOpen(false)}
                subtopicName={selectedSubtopicForOffers?.name || ''}
                offers={selectedSubtopicForOffers?.monetization_data?.offers || []}
                onRetry={() => {
                    if (selectedSubtopicForOffers && onEnrich) {
                        handleEnrichClick({ stopPropagation: () => { } } as React.MouseEvent, selectedSubtopicForOffers.id)
                    } else if (selectedSubtopicForOffers && onVerify) {
                        // Fallback to onVerify if onEnrich is not explicitly passed (though handleEnrichClick uses onEnrich)
                        onVerify(selectedSubtopicForOffers.id)
                    }
                }}
                isLoading={selectedSubtopicForOffers ? enrichingIds.has(selectedSubtopicForOffers.id) : false}
            />


        </div>
    )
}
