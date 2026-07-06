import * as React from 'react'
import { Rocket, Loader2, Target, ArrowRight, Table as TableIcon, Check } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { researchPipelineService } from '@/services/research-pipeline.service'
import { LineChart, Line, ResponsiveContainer, YAxis } from 'recharts'
import { useNavigate } from 'react-router-dom'
import { supabase } from '@/lib/supabase'
import { useAuth } from '@/context/auth-context'
import { useProject } from '@/context/project-context'
import { toast } from 'sonner'

export function ResearchPipeline() {
    const navigate = useNavigate()
    const { user } = useAuth()
    const { activeProject } = useProject()
    
    const [queryText, setQueryText] = React.useState('')
    const [loading, setLoading] = React.useState(false)
    const [step, setStep] = React.useState<1 | 2 | 3>(1)
    
    const [keywords, setKeywords] = React.useState<any[]>([])
    const [clusters, setClusters] = React.useState<any[]>([])
    const [error, setError] = React.useState<string | null>(null)
    
    const [selectedClusters, setSelectedClusters] = React.useState<Record<number, boolean>>({})
    const [selectingClusterIdx, setSelectingClusterIdx] = React.useState<number | null>(null)

    const handleSelectForArticle = async (cluster: any, idx: number) => {
        console.log('[ResearchPipeline] handleSelectForArticle button clicked', {
            idx,
            cluster,
            user,
            activeProject
        })
        if (!user) {
            console.warn('[ResearchPipeline] handleSelectForArticle: user is null')
            toast.error('You must be logged in to select an article.')
            return
        }
        if (!activeProject) {
            console.warn('[ResearchPipeline] handleSelectForArticle: activeProject is null')
            toast.error('Please select a project first.')
            return
        }

        setSelectingClusterIdx(idx)
        try {
            const clusterKws = cluster.keywords || []
            const sortedKws = [...clusterKws].sort((a: any, b: any) => (b.search_volume || 0) - (a.search_volume || 0))
            const primaryKeywordObj = sortedKws[0]
            const primaryKeyword = primaryKeywordObj?.keyword || ''
            
            const secondaryKeywords = clusterKws
                .map((kw: any) => kw.keyword)
                .filter((kwStr: string) => kwStr !== primaryKeyword)

            const title = cluster.subtopic_name || cluster.cluster_title || 'Untitled Article'

            const selected_keyword_metrics_json = {
                primary: primaryKeywordObj ? {
                    keyword: primaryKeywordObj.keyword,
                    search_volume: primaryKeywordObj.search_volume || 0,
                    keyword_difficulty: primaryKeywordObj.keyword_difficulty || 0,
                    cpc: primaryKeywordObj.cpc || 0,
                } : null,
                secondary: clusterKws
                    .filter((kw: any) => kw.keyword !== primaryKeyword)
                    .map((kw: any) => ({
                        keyword: kw.keyword,
                        search_volume: kw.search_volume || 0,
                        keyword_difficulty: kw.keyword_difficulty || 0,
                        cpc: kw.cpc || 0,
                    })),
                secondaries: clusterKws
                    .filter((kw: any) => kw.keyword !== primaryKeyword)
                    .map((kw: any) => ({
                        keyword: kw.keyword,
                        search_volume: kw.search_volume || 0,
                        keyword_difficulty: kw.keyword_difficulty || 0,
                        cpc: kw.cpc || 0,
                    }))
            }

            console.log('[ResearchPipeline] Inserting new article into Titles...', {
                title,
                primaryKeyword,
                secondaryKeywords,
                domain: activeProject.domain || '',
                user_id: user.id
            })

            const { error: insertError } = await supabase
                .from('Titles')
                .insert([{
                    user_id: user.id,
                    Title: title,
                    userDescription: cluster.description || cluster.primary_user_outcome || cluster.outcome || `Keywords: ${clusterKws.map((k: any) => k.keyword).join(', ')}`,
                    status: 'New',
                    dateCreatedOn: new Date().toISOString(),
                    primary_keyword: primaryKeyword,
                    primary_keywords: primaryKeyword ? JSON.stringify([primaryKeyword]) : '[]',
                    secondary_keywords: JSON.stringify(secondaryKeywords),
                    secondary_keywords_json: secondaryKeywords,
                    Keywords: clusterKws.map((k: any) => k.keyword).join(', '),
                    keyword_candidates_json: clusterKws.map((k: any) => k.keyword),
                    domain: activeProject.domain || '',
                    selected_keyword_metrics_json: selected_keyword_metrics_json,
                }])

            if (insertError) {
                console.error('[ResearchPipeline] Supabase insert error:', insertError)
                throw insertError
            }

            console.log('[ResearchPipeline] Successfully inserted article to queue!')

            setSelectedClusters(prev => ({ ...prev, [idx]: true }))
            toast.success(`Successfully added "${title}" to your article queue!`, {
                action: {
                    label: 'View Queue',
                    onClick: () => navigate('/my-articles')
                }
            })
        } catch (err: any) {
            console.error('[ResearchPipeline] Failed to select article:', err)
            toast.error(err.message || 'Failed to select article.')
        } finally {
            setSelectingClusterIdx(null)
        }
    }

    const handleExtract = async () => {
        if (!queryText) return
        setLoading(true)
        setError(null)
        try {
            const data = await researchPipelineService.extractKeywords(queryText)
            setKeywords(data.keywords || [])
            setStep(2)
        } catch (err: any) {
            setError(err.message || 'An error occurred during extraction.')
        } finally {
            setLoading(false)
        }
    }

    const handleCluster = async () => {
        if (keywords.length === 0) return
        setLoading(true)
        setError(null)
        try {
            const data = await researchPipelineService.clusterKeywords(keywords)
            setClusters(data || [])
            setStep(3)
        } catch (err: any) {
            setError(err.message || 'An error occurred during clustering.')
        } finally {
            setLoading(false)
        }
    }

    return (
        <div className="flex flex-col h-full bg-background overflow-y-auto p-8 max-w-[1400px] mx-auto w-full">
            <div className="mb-8 space-y-2 flex justify-between items-end">
                <div>
                    <h1 className="text-3xl font-black tracking-tight flex items-center gap-3">
                        <Rocket className="h-8 w-8 text-indigo-500" />
                        End-to-End Research Pipeline
                    </h1>
                    <p className="text-muted-foreground mt-2">
                        Discover highly-profitable keywords, analyze metrics, and cluster them into actionable topics.
                    </p>
                </div>
                
                <div className="flex gap-2">
                    <Badge variant={step >= 1 ? "default" : "outline"} className={step >= 1 ? "bg-indigo-500" : ""}>1. Seed</Badge>
                    <ArrowRight className="w-4 h-4 text-muted-foreground self-center" />
                    <Badge variant={step >= 2 ? "default" : "outline"} className={step >= 2 ? "bg-indigo-500" : ""}>2. Validate</Badge>
                    <ArrowRight className="w-4 h-4 text-muted-foreground self-center" />
                    <Badge variant={step >= 3 ? "default" : "outline"} className={step >= 3 ? "bg-indigo-500" : ""}>3. Cluster</Badge>
                </div>
            </div>

            {step === 1 && (
                <div className="bg-white/5 border border-border rounded-3xl p-8 mb-8">
                    <h3 className="text-sm font-black uppercase tracking-widest text-foreground mb-4">Seed Topic</h3>
                    <div className="flex gap-4">
                        <Input
                            placeholder="e.g. hidden costs of owning a home"
                            value={queryText}
                            onChange={(e) => setQueryText(e.target.value)}
                            className="flex-1 bg-muted/30 dark:bg-black/50 border-border text-foreground text-lg py-6"
                            onKeyDown={(e) => e.key === 'Enter' && handleExtract()}
                        />
                        <Button 
                            onClick={handleExtract}
                            disabled={loading || !queryText}
                            className="h-auto px-8 bg-indigo-500 hover:bg-indigo-600 text-white font-black"
                        >
                            {loading ? <Loader2 className="w-5 h-5 animate-spin mr-2" /> : <Rocket className="w-5 h-5 mr-2" />}
                            Extract Keywords
                        </Button>
                    </div>
                    {error && <div className="mt-4 text-red-400 text-sm">{error}</div>}
                </div>
            )}

            {step === 2 && (
                <div className="space-y-6">
                    <div className="flex justify-between items-center bg-white/5 border border-border rounded-2xl p-6">
                        <div>
                            <h3 className="text-xl font-black text-foreground">Validated Keywords</h3>
                            <p className="text-sm text-muted-foreground mt-1">Found {keywords.length} profitable keywords for "{queryText}". Saved to database.</p>
                        </div>
                        <Button 
                            onClick={handleCluster}
                            disabled={loading || keywords.length === 0}
                            className="bg-emerald-500 hover:bg-emerald-600 text-white font-black"
                        >
                            {loading ? <Loader2 className="w-5 h-5 animate-spin mr-2" /> : <TableIcon className="w-5 h-5 mr-2" />}
                            Generate Clusters
                        </Button>
                    </div>

                    <div className="bg-muted/20 dark:bg-black/40 border border-border rounded-2xl overflow-hidden">
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
                                    {keywords.map((kw, idx) => {
                                        // Sort trend by month chronologically if needed, assuming it's returned sorted
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
                                                <td className="p-4 text-right text-amber-600 dark:text-amber-400 font-bold">{kw.keyword_difficulty}</td>
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
                </div>
            )}

            {step === 3 && (
                <div className="space-y-6">
                    <div className="flex justify-between items-center">
                        <h3 className="text-xl font-black text-foreground">Discovered Clusters ({clusters.length})</h3>
                        <Button variant="outline" onClick={() => setStep(1)} className="border-white/10">Start New Search</Button>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                        {clusters.map((cluster, idx) => {
                            const title = cluster.subtopic_name || cluster.cluster_title || 'Unknown Cluster'
                            const intent = cluster.primary_intent || cluster.cluster_intent || 'mixed'
                            const clusterKws = cluster.keywords || []
                            
                            const totalVol = clusterKws.reduce((acc: number, kw: any) => acc + (kw.search_volume || 0), 0)
                            const maxKd = clusterKws.reduce((acc: number, kw: any) => Math.max(acc, kw.keyword_difficulty || 0), 0)
                            
                            return (
                                <div key={idx} className="bg-muted/20 dark:bg-black/40 border border-border rounded-2xl p-6 flex flex-col hover:border-indigo-500/50 transition-colors">
                                    <div className="flex justify-between items-start mb-4">
                                        <h4 className="text-lg font-black leading-tight flex-1 pr-4">{title}</h4>
                                        <Badge variant="outline" className="text-[10px] uppercase tracking-widest border-indigo-500/30 text-indigo-400 whitespace-nowrap">
                                            {intent}
                                        </Badge>
                                    </div>
                                    
                                    {cluster.description && (
                                        <p className="text-xs text-muted-foreground mb-4 leading-relaxed line-clamp-3">
                                            {cluster.description}
                                        </p>
                                    )}

                                    <div className="flex gap-6 mb-6">
                                        <div>
                                            <div className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1">Total Vol</div>
                                            <div className="font-black text-emerald-600 dark:text-emerald-400">{totalVol.toLocaleString()}</div>
                                        </div>
                                        <div>
                                            <div className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1">Max KD</div>
                                            <div className="font-black text-amber-600 dark:text-amber-400">{maxKd}</div>
                                        </div>
                                        <div>
                                            <div className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1">Keywords</div>
                                            <div className="font-black">{clusterKws.length}</div>
                                        </div>
                                    </div>
                                    
                                    <div className="flex-1">
                                        <div className="text-[10px] uppercase tracking-widest text-muted-foreground mb-2">Profitable Keywords</div>
                                        <div className="flex flex-wrap gap-2">
                                            {clusterKws.slice(0, 5).map((kw: any, i: number) => (
                                                <Badge key={i} variant="secondary" className="bg-muted/50 hover:bg-muted dark:bg-white/5 dark:hover:bg-white/10 border-border font-medium text-xs text-foreground">
                                                    {kw.keyword} <span className="text-emerald-600/70 dark:text-emerald-400/70 ml-1 text-[10px]">{kw.search_volume}v</span>
                                                </Badge>
                                            ))}
                                            {clusterKws.length > 5 && (
                                                <Badge variant="secondary" className="bg-muted/50 dark:bg-white/5 border-border font-medium text-xs opacity-50 text-foreground">
                                                    +{clusterKws.length - 5} more
                                                </Badge>
                                            )}
                                        </div>
                                    </div>

                                    <div className="mt-6 pt-6 border-t border-border">
                                        <Button
                                            onClick={() => handleSelectForArticle(cluster, idx)}
                                            disabled={loading || selectingClusterIdx === idx || selectedClusters[idx]}
                                            className={`w-full transition-colors font-bold ${
                                                selectedClusters[idx]
                                                    ? 'bg-emerald-500 hover:bg-emerald-600 text-white'
                                                    : 'bg-muted/50 dark:bg-white/5 hover:bg-indigo-500 hover:text-white text-foreground'
                                            }`}
                                        >
                                            {selectingClusterIdx === idx ? (
                                                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                            ) : selectedClusters[idx] ? (
                                                <Check className="w-4 h-4 mr-2" />
                                            ) : (
                                                <Target className="w-4 h-4 mr-2" />
                                            )}
                                            {selectedClusters[idx] ? 'Added to Queue' : 'Select for Article'}
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
