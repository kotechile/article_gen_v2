"use client"

import {
    Sheet,
    SheetContent,
    SheetDescription,
    SheetHeader,
    SheetTitle,
} from "@/components/ui/sheet"
import { Badge } from "@/components/ui/badge"
import { ExternalLink, DollarSign, Building } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface AffiliateOffer {
    program_name?: string
    name?: string // LinkUp fallback
    company_name?: string
    description?: string
    website_url?: string
    link?: string // LinkUp fallback
    network_name?: string
    network?: string // LinkUp fallback
    commission_rate?: string | number
    commission?: string | number // LinkUp fallback
    affiliate_link?: string
    content_opportunities?: string[]
}

interface AffiliateOffersModalProps {
    isOpen: boolean
    onClose: () => void
    subtopicName: string
    offers: AffiliateOffer[]
    onRetry?: () => void
    isLoading?: boolean
}

export function AffiliateOffersModal({
    isOpen,
    onClose,
    subtopicName,
    offers,
    onRetry,
    isLoading = false
}: AffiliateOffersModalProps) {
    return (
        <Sheet open={isOpen} onOpenChange={onClose}>
            <SheetContent className="w-[400px] sm:w-[540px] overflow-y-auto">
                <SheetHeader>
                    <SheetTitle className="flex justify-between items-center">
                        <span>Affiliate Offers: {subtopicName}</span>
                        {onRetry && (
                            <button
                                onClick={onRetry}
                                disabled={isLoading}
                                className="text-xs px-3 py-1 bg-primary text-primary-foreground rounded hover:opacity-90 disabled:opacity-50"
                            >
                                {isLoading ? "Searching..." : "Find More Offers"}
                            </button>
                        )}
                    </SheetTitle>
                    <SheetDescription>
                        Discovered {offers.length} potential affiliate programs for this niche.
                    </SheetDescription>
                </SheetHeader>

                <div className="mt-6 flex flex-col gap-4">
                    {offers.length === 0 ? (
                        <div className="text-center py-8 text-muted-foreground flex flex-col items-center gap-4">
                            <p>No offers data available yet.</p>
                            {onRetry && (
                                <button
                                    onClick={onRetry}
                                    disabled={isLoading}
                                    className="px-4 py-2 bg-secondary text-secondary-foreground rounded hover:opacity-80 disabled:opacity-50 font-medium text-sm"
                                >
                                    {isLoading ? "Searching..." : "Try Detailed Research"}
                                </button>
                            )}
                        </div>
                    ) : (
                        offers.map((offer, index) => (
                            <Card key={index} className="overflow-hidden">
                                <CardHeader className="bg-muted/50 p-4 pb-2">
                                    <div className="flex justify-between items-start gap-2">
                                        <CardTitle className="text-base font-bold text-primary">
                                            {offer.program_name || offer.name || offer.company_name || "Affiliate Program"}
                                        </CardTitle>
                                        {(offer.network_name || offer.network) && (
                                            <Badge variant="outline" className="shrink-0 text-[10px] uppercase">
                                                {offer.network_name || offer.network}
                                            </Badge>
                                        )}
                                    </div>
                                </CardHeader>
                                <CardContent className="p-4 pt-3 flex flex-col gap-3">
                                    {offer.description && (
                                        <p className="text-sm text-foreground/80 line-clamp-3">
                                            {offer.description}
                                        </p>
                                    )}

                                    <div className="flex flex-wrap gap-3 text-sm">
                                        {(offer.commission_rate || offer.commission) && (
                                            <div className="flex items-center gap-1.5 text-green-600 font-medium bg-green-50 px-2 py-1 rounded">
                                                <DollarSign className="w-3.5 h-3.5" />
                                                <span>{offer.commission_rate || offer.commission}</span>
                                            </div>
                                        )}
                                        {offer.company_name && offer.company_name !== offer.program_name && (
                                            <div className="flex items-center gap-1.5 text-muted-foreground bg-muted/40 px-2 py-1 rounded">
                                                <Building className="w-3.5 h-3.5" />
                                                <span>{offer.company_name}</span>
                                            </div>
                                        )}
                                    </div>

                                    {offer.content_opportunities && offer.content_opportunities.length > 0 && (
                                        <div className="mt-1">
                                            <p className="text-xs font-semibold mb-1 text-muted-foreground">Content Angles:</p>
                                            <div className="flex flex-wrap gap-1">
                                                {offer.content_opportunities.map((opp, i) => (
                                                    <span key={i} className="text-[10px] bg-secondary text-secondary-foreground px-1.5 py-0.5 rounded">
                                                        {opp}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {(offer.website_url || offer.affiliate_link || offer.link) && (
                                        <a
                                            href={offer.website_url || offer.affiliate_link || offer.link}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="mt-2 text-xs flex items-center gap-1 text-blue-600 hover:underline"
                                        >
                                            Visit Program Page <ExternalLink className="w-3 h-3" />
                                        </a>
                                    )}
                                </CardContent>
                            </Card>
                        ))
                    )}
                </div>
            </SheetContent>
        </Sheet>
    )
}
