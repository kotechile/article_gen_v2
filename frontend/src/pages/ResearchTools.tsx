import * as React from 'react'
import {
    Loader2, Search, Globe, ListChecks,
    Download, BarChart3, Wrench
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { LineChart, Line, ResponsiveContainer, YAxis } from 'recharts'
import { researchToolsService } from '@/services/research-tools.service'
import type { KeywordResult } from '@/services/research-tools.service'

export function ResearchTools() {
    const [activeTab, setActiveTab] = React.useState('bulk')
    
    // Bulk Metrics State
    const [bulkKeywordsText, setBulkKeywordsText] = React.useState('')
    const [bulkLoading, setBulkLoading] = React.useState(false)
    const [bulkResults, setBulkResults] = React.useState<KeywordResult[]>([])
    const [bulkError, setBulkError] = React.useState<string | null>(null)

    // Website Keywords State
    const [websiteDomain, setWebsiteDomain] = React.useState('')
    const [websiteLoading, setWebsiteLoading] = React.useState(false)
    const [websiteResults, setWebsiteResults] = React.useState<KeywordResult[]>([])
    const [websiteError, setWebsiteError] = React.useState<string | null>(null)

    // Related Keywords State
    const [seedKeyword, setSeedKeyword] = React.useState('')
    const [relatedLoading, setRelatedLoading] = React.useState(false)
    const [relatedResults, setRelatedResults] = React.useState<KeywordResult[]>([])
    const [relatedError, setRelatedError] = React.useState<string | null>(null)

    const handleBulkSearch = async () => {
        if (!bulkKeywordsText.trim()) return
        setBulkLoading(true)
        setBulkError(null)
        try {
            const list = bulkKeywordsText.split('\n').map(k => k.trim()).filter(Boolean)
            if (list.length === 0) return
            const res = await researchToolsService.getBulkMetrics(list)
            setBulkResults(res)
        } catch (err: any) {
            setBulkError(err.message || 'Failed to fetch bulk metrics')
        } finally {
            setBulkLoading(false)
        }
    }

    const handleWebsiteSearch = async () => {
        if (!websiteDomain.trim()) return
        setWebsiteLoading(true)
        setWebsiteError(null)
        try {
            const res = await researchToolsService.getWebsiteKeywords(websiteDomain.trim())
            setWebsiteResults(res)
        } catch (err: any) {
            setWebsiteError(err.message || 'Failed to fetch website keywords')
        } finally {
            setWebsiteLoading(false)
        }
    }

    const handleRelatedSearch = async () => {
        if (!seedKeyword.trim()) return
        setRelatedLoading(true)
        setRelatedError(null)
        try {
            const res = await researchToolsService.getRelatedKeywords(seedKeyword.trim())
            setRelatedResults(res)
        } catch (err: any) {
            setRelatedError(err.message || 'Failed to fetch related keywords')
        } finally {
            setRelatedLoading(false)
        }
    }

    const downloadCsv = (data: KeywordResult[], filename: string) => {
        if (!data || data.length === 0) return
        const headers = ['Keyword', 'Volume', 'KD', 'CPC']
        const rows = data.map(k => [
            `"${k.keyword?.replace(/"/g, '""')}"`,
            k.search_volume || 0,
            k.keyword_difficulty || 0,
            k.cpc || 0
        ])
        const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n')
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
        const url = URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.setAttribute('href', url)
        link.setAttribute('download', filename)
        link.style.visibility = 'hidden'
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
    }

    const renderResultsTable = (results: KeywordResult[]) => {
        if (!results || results.length === 0) return null

        return (
            <div className="mt-8 bg-muted/20 dark:bg-black/40 border border-border rounded-2xl overflow-hidden">
                <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm">
                        <thead>
                            <tr className="border-b border-border bg-muted/30 dark:bg-white/5">
                                <th className="p-4 font-black uppercase tracking-widest text-xs text-muted-foreground">Keyword</th>
                                <th className="p-4 font-black uppercase tracking-widest text-xs text-muted-foreground text-right">Volume</th>
                                <th className="p-4 font-black uppercase tracking-widest text-xs text-muted-foreground text-right">KD</th>
                                <th className="p-4 font-black uppercase tracking-widest text-xs text-muted-foreground text-right">CPC</th>
                                <th className="p-4 font-black uppercase tracking-widest text-xs text-muted-foreground">Intent</th>
                                <th className="p-4 font-black uppercase tracking-widest text-xs text-muted-foreground w-32">Trend</th>
                            </tr>
                        </thead>
                        <tbody>
                            {results.map((kw, idx) => {
                                const chartData = (kw.monthly_searches || [])
                                    .slice()
                                    .reverse()
                                    .map((m: any) => ({
                                        name: `${m.month}/${m.year}`,
                                        val: m.search_volume
                                    }))

                                return (
                                    <tr key={idx} className="border-b border-border hover:bg-muted/30 dark:hover:bg-white/5">
                                        <td className="p-4 font-medium text-foreground">{kw.keyword}</td>
                                        <td className="p-4 text-right text-emerald-600 dark:text-emerald-400 font-bold">{kw.search_volume?.toLocaleString()}</td>
                                        <td className="p-4 text-right text-amber-600 dark:text-amber-400 font-bold">{kw.keyword_difficulty || '-'}</td>
                                        <td className="p-4 text-right">${kw.cpc?.toFixed(2) || '0.00'}</td>
                                        <td className="p-4">
                                            <Badge variant="outline" className="text-[10px] uppercase border-indigo-500/30 text-indigo-400">
                                                {kw.intent || 'unknown'}
                                            </Badge>
                                        </td>
                                        <td className="p-4 h-12 w-32">
                                            {chartData.length > 0 ? (
                                                <ResponsiveContainer width="100%" height="100%">
                                                    <LineChart data={chartData}>
                                                        <YAxis domain={['dataMin', 'dataMax']} hide />
                                                        <Line type="monotone" dataKey="val" stroke="#8b5cf6" strokeWidth={2} dot={false} isAnimationActive={false} />
                                                    </LineChart>
                                                </ResponsiveContainer>
                                            ) : (
                                                <span className="text-xs text-muted-foreground">No data</span>
                                            )}
                                        </td>
                                    </tr>
                                )
                            })}
                        </tbody>
                    </table>
                </div>
            </div>
        )
    }

    return (
        <div className="container mx-auto py-8 max-w-6xl">
            <div className="mb-8">
                <h1 className="text-3xl font-black tracking-tight flex items-center gap-3">
                    <Wrench className="h-8 w-8 text-indigo-500" />
                    Research Tools
                </h1>
                <p className="text-muted-foreground mt-2">
                    Standalone tools for quick keyword discovery and competitor analysis.
                </p>
            </div>

            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
                <TabsList className="grid w-full grid-cols-3 mb-8">
                    <TabsTrigger value="bulk" className="flex items-center gap-2">
                        <ListChecks className="w-4 h-4" /> Bulk Metrics
                    </TabsTrigger>
                    <TabsTrigger value="website" className="flex items-center gap-2">
                        <Globe className="w-4 h-4" /> Website Keywords
                    </TabsTrigger>
                    <TabsTrigger value="related" className="flex items-center gap-2">
                        <BarChart3 className="w-4 h-4" /> Related Keywords
                    </TabsTrigger>
                </TabsList>

                {/* BULK METRICS TAB */}
                <TabsContent value="bulk" className="space-y-6">
                    <div className="bg-white/5 border border-border rounded-3xl p-8">
                        <h3 className="text-sm font-black uppercase tracking-widest text-foreground mb-4">Search a List of Keywords</h3>
                        <p className="text-sm text-muted-foreground mb-6">Enter up to 500 keywords, one per line, to retrieve search volume, KD, and CPC metrics.</p>
                        <div className="flex gap-4">
                            <textarea
                                placeholder="e.g.&#10;buy coffee beans&#10;best coffee maker&#10;how to brew espresso"
                                value={bulkKeywordsText}
                                onChange={(e: any) => setBulkKeywordsText(e.target.value)}
                                className="flex-1 bg-muted/30 dark:bg-black/50 border border-border text-foreground min-h-[150px] p-4 rounded-xl text-sm"
                            />
                        </div>
                        <div className="mt-6 flex justify-between items-center">
                            {bulkError && <p className="text-red-400 text-sm">{bulkError}</p>}
                            <div className="ml-auto flex gap-4">
                                {bulkResults.length > 0 && (
                                    <Button variant="outline" onClick={() => downloadCsv(bulkResults, 'bulk_metrics.csv')}>
                                        <Download className="w-4 h-4 mr-2" />
                                        Export CSV
                                    </Button>
                                )}
                                <Button 
                                    onClick={handleBulkSearch}
                                    disabled={bulkLoading || !bulkKeywordsText}
                                    className="bg-indigo-500 hover:bg-indigo-600 text-white font-black"
                                >
                                    {bulkLoading ? <Loader2 className="w-5 h-5 animate-spin mr-2" /> : <Search className="w-5 h-5 mr-2" />}
                                    Get Metrics
                                </Button>
                            </div>
                        </div>
                    </div>
                    {renderResultsTable(bulkResults)}
                </TabsContent>

                {/* WEBSITE KEYWORDS TAB */}
                <TabsContent value="website" className="space-y-6">
                    <div className="bg-white/5 border border-border rounded-3xl p-8">
                        <h3 className="text-sm font-black uppercase tracking-widest text-foreground mb-4">Find Keywords from Website</h3>
                        <p className="text-sm text-muted-foreground mb-6">Enter a competitor's domain or specific URL to see what keywords they rank for.</p>
                        <div className="flex gap-4">
                            <Input
                                placeholder="e.g. example.com or example.com/blog-post"
                                value={websiteDomain}
                                onChange={(e: any) => setWebsiteDomain(e.target.value)}
                                className="flex-1 bg-muted/30 dark:bg-black/50 border-border text-foreground text-lg py-6"
                                onKeyDown={(e) => e.key === 'Enter' && handleWebsiteSearch()}
                            />
                            <Button 
                                onClick={handleWebsiteSearch}
                                disabled={websiteLoading || !websiteDomain}
                                className="h-auto px-8 bg-indigo-500 hover:bg-indigo-600 text-white font-black"
                            >
                                {websiteLoading ? <Loader2 className="w-5 h-5 animate-spin mr-2" /> : <Search className="w-5 h-5 mr-2" />}
                                Analyze
                            </Button>
                        </div>
                        {websiteError && <p className="mt-4 text-red-400 text-sm">{websiteError}</p>}
                        
                        {websiteResults.length > 0 && (
                             <div className="mt-6 flex justify-end">
                                <Button variant="outline" size="sm" onClick={() => downloadCsv(websiteResults, `${websiteDomain}_keywords.csv`)}>
                                    <Download className="w-4 h-4 mr-2" />
                                    Export CSV
                                </Button>
                             </div>
                        )}
                    </div>
                    {renderResultsTable(websiteResults)}
                </TabsContent>

                {/* RELATED KEYWORDS TAB */}
                <TabsContent value="related" className="space-y-6">
                    <div className="bg-white/5 border border-border rounded-3xl p-8">
                        <h3 className="text-sm font-black uppercase tracking-widest text-foreground mb-4">Search Related Keywords</h3>
                        <p className="text-sm text-muted-foreground mb-6">Enter a seed topic to generate hundreds of related long-tail keyword ideas.</p>
                        <div className="flex gap-4">
                            <Input
                                placeholder="e.g. credit cards"
                                value={seedKeyword}
                                onChange={(e: any) => setSeedKeyword(e.target.value)}
                                className="flex-1 bg-muted/30 dark:bg-black/50 border-border text-foreground text-lg py-6"
                                onKeyDown={(e) => e.key === 'Enter' && handleRelatedSearch()}
                            />
                            <Button 
                                onClick={handleRelatedSearch}
                                disabled={relatedLoading || !seedKeyword}
                                className="h-auto px-8 bg-indigo-500 hover:bg-indigo-600 text-white font-black"
                            >
                                {relatedLoading ? <Loader2 className="w-5 h-5 animate-spin mr-2" /> : <Search className="w-5 h-5 mr-2" />}
                                Find Ideas
                            </Button>
                        </div>
                        {relatedError && <p className="mt-4 text-red-400 text-sm">{relatedError}</p>}

                        {relatedResults.length > 0 && (
                             <div className="mt-6 flex justify-end">
                                <Button variant="outline" size="sm" onClick={() => downloadCsv(relatedResults, `${seedKeyword}_related.csv`)}>
                                    <Download className="w-4 h-4 mr-2" />
                                    Export CSV
                                </Button>
                             </div>
                        )}
                    </div>
                    {renderResultsTable(relatedResults)}
                </TabsContent>
            </Tabs>
        </div>
    )
}
