"use client"

import {
    Sheet,
    SheetContent,
    SheetDescription,
    SheetHeader,
    SheetTitle,
} from "@/components/ui/sheet"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ExternalLink, DollarSign, Activity } from "lucide-react"

interface AffiliateProgram {
    name: string
    trustScore: number
    commission: string
    url: string
}

interface MarketIntelligenceDrawerProps {
    isOpen: boolean
    onClose: () => void
    keyword: string
    programs: AffiliateProgram[]
}

export function MarketIntelligenceDrawer({ isOpen, onClose, keyword, programs }: MarketIntelligenceDrawerProps) {
    return (
        <Sheet open={isOpen} onOpenChange={onClose}>
            <SheetContent className="w-[400px] sm:w-[540px] border-l-zinc-800 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
                <SheetHeader className="mb-6">
                    <SheetTitle className="text-xl">Market Intelligence</SheetTitle>
                    <SheetDescription>
                        Affiliate opportunities for <span className="text-foreground font-medium">"{keyword}"</span>
                    </SheetDescription>
                </SheetHeader>

                <div className="space-y-4">
                    <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">Top Programs</h3>

                    {programs.map((program, idx) => (
                        <Card key={idx} className="bg-zinc-900/50 border-zinc-800 hover:border-zinc-700 transition-colors">
                            <CardHeader className="pb-2">
                                <div className="flex justify-between items-start">
                                    <div>
                                        <CardTitle className="text-base">{program.name}</CardTitle>
                                        <CardDescription className="flex items-center mt-1">
                                            <DollarSign className="w-3 h-3 mr-1 text-green-500" />
                                            {program.commission}
                                        </CardDescription>
                                    </div>
                                    <Badge variant={program.trustScore >= 90 ? "default" : "secondary"}>
                                        {program.trustScore}% Match
                                    </Badge>
                                </div>
                            </CardHeader>
                            <CardContent>
                                <Button variant="outline" size="sm" className="w-full gap-2 border-zinc-700 hover:bg-zinc-800">
                                    Visit Program <ExternalLink className="w-3 h-3" />
                                </Button>
                            </CardContent>
                        </Card>
                    ))}
                </div>
            </SheetContent>
        </Sheet>
    )
}
