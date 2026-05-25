import * as React from 'react'
import { Rocket, Loader2, Target } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { researchPipelineService } from '@/services/research-pipeline.service'

export function ResearchPipeline() {
    const [queryText, setQueryText] = React.useState('')
    const [loading, setLoading] = React.useState(false)
    const [clusters, setClusters] = React.useState<any[]>([])
    const [error, setError] = React.useState<string | null>(null)

    const handleRunPipeline = async () => {
        if (!queryText) return
        setLoading(true)
        setError(null)
        try {
            const data = await researchPipelineService.runPipeline(queryText)
            setClusters(data)
        } catch (err: any) {
            setError(err.message || 'An error occurred while running the pipeline.')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="flex flex-col h-full bg-background overflow-y-auto p-8 max-w-7xl mx-auto w-full">
            <div className="mb-8 space-y-2">
                <h1 className="text-3xl font-black tracking-tight flex items-center gap-3">
                    <Rocket className="h-8 w-8 text-indigo-500" />
                    End-to-End Research Pipeline
                </h1>
                <p className="text-muted-foreground">
                    Discover highly-profitable keywords through SERP and autocomplete expansion, automatically clustered into actionable topics.
                </p>
            </div>

            <div className="bg-white/5 border border-border rounded-3xl p-8 mb-8">
                <h3 className="text-sm font-black uppercase tracking-widest text-foreground mb-4">Seed Topic</h3>
                <div className="flex gap-4">
                    <Input
                        placeholder="e.g. espresso machines"
                        value={queryText}
                        onChange={(e) => setQueryText(e.target.value)}
                        className="flex-1 bg-black/50 border-white/10 text-lg py-6"
                        onKeyDown={(e) => e.key === 'Enter' && handleRunPipeline()}
                    />
                    <Button 
                        onClick={handleRunPipeline}
                        disabled={loading || !queryText}
                        className="h-auto px-8 bg-indigo-500 hover:bg-indigo-600 text-white font-black"
                    >
                        {loading ? <Loader2 className="w-5 h-5 animate-spin mr-2" /> : <Rocket className="w-5 h-5 mr-2" />}
                        Run Full Pipeline
                    </Button>
                </div>
                {error && <div className="mt-4 text-red-400 text-sm">{error}</div>}
            </div>

            {clusters.length > 0 && (
                <div className="space-y-6">
                    <h3 className="text-xl font-black text-foreground">Discovered Clusters ({clusters.length})</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                        {clusters.map((cluster, idx) => {
                            const title = cluster.subtopic_name || cluster.cluster_title || 'Unknown Cluster'
                            const intent = cluster.primary_intent || cluster.cluster_intent || 'mixed'
                            const keywords = cluster.keywords || []
                            
                            // Calculate metrics
                            const totalVol = keywords.reduce((acc: number, kw: any) => acc + (kw.search_volume || 0), 0)
                            const maxKd = keywords.reduce((acc: number, kw: any) => Math.max(acc, kw.keyword_difficulty || 0), 0)
                            
                            return (
                                <div key={idx} className="bg-black/40 border border-white/10 rounded-2xl p-6 flex flex-col hover:border-indigo-500/50 transition-colors">
                                    <div className="flex justify-between items-start mb-4">
                                        <h4 className="text-lg font-black leading-tight flex-1 pr-4">{title}</h4>
                                        <Badge variant="outline" className="text-[10px] uppercase tracking-widest border-indigo-500/30 text-indigo-400 whitespace-nowrap">
                                            {intent}
                                        </Badge>
                                    </div>
                                    
                                    <div className="flex gap-6 mb-6">
                                        <div>
                                            <div className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1">Total Vol</div>
                                            <div className="font-black text-emerald-400">{totalVol.toLocaleString()}</div>
                                        </div>
                                        <div>
                                            <div className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1">Max KD</div>
                                            <div className="font-black text-amber-400">{maxKd}</div>
                                        </div>
                                        <div>
                                            <div className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1">Keywords</div>
                                            <div className="font-black">{keywords.length}</div>
                                        </div>
                                    </div>
                                    
                                    <div className="flex-1">
                                        <div className="text-[10px] uppercase tracking-widest text-muted-foreground mb-2">Profitable Keywords</div>
                                        <div className="flex flex-wrap gap-2">
                                            {keywords.slice(0, 5).map((kw: any, i: number) => (
                                                <Badge key={i} variant="secondary" className="bg-white/5 hover:bg-white/10 border-white/5 font-medium text-xs">
                                                    {kw.keyword} <span className="text-emerald-400/70 ml-1 text-[10px]">{kw.search_volume}v</span>
                                                </Badge>
                                            ))}
                                            {keywords.length > 5 && (
                                                <Badge variant="secondary" className="bg-white/5 border-white/5 font-medium text-xs opacity-50">
                                                    +{keywords.length - 5} more
                                                </Badge>
                                            )}
                                        </div>
                                    </div>

                                    <div className="mt-6 pt-6 border-t border-white/5">
                                        <Button className="w-full bg-white/5 hover:bg-indigo-500 hover:text-white transition-colors text-foreground font-bold">
                                            <Target className="w-4 h-4 mr-2" />
                                            Select for Article
                                        </Button>
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                </div>
            )}
        </div>
    )
}
