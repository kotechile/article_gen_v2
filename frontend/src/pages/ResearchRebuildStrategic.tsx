import * as React from 'react'
import { useSearchParams } from 'react-router-dom'

import { useProject } from '@/context/project-context'
import { supabase } from '@/lib/supabase'
import { researchRebuildService } from '@/services/research-rebuild.service'
import type { ProjectCategory } from '@/types/command-center'
import type {
    ResearchRebuildDataforseoSearch,
    ResearchRebuildJob,
    ResearchKeywordCluster,
    ResearchStrategyRun,
    ResearchStrategyRunDetail,
    ResearchTopicBet,
} from '@/types/research-rebuild'

type LookupType = 'related_keywords' | 'keyword_overview' | 'serp'

function scoreLabel(value?: number | null) {
    const score = Number(value || 0)
    if (score >= 0.75) return 'Strong'
    if (score >= 0.55) return 'Promising'
    if (score >= 0.35) return 'Mixed'
    return 'Weak'
}

function toneClass(value?: number | null) {
    const score = Number(value || 0)
    if (score >= 0.75) return 'text-emerald-300'
    if (score >= 0.55) return 'text-sky-300'
    if (score >= 0.35) return 'text-amber-300'
    return 'text-rose-300'
}

function routeLabel(route?: string | null) {
    if (!route) return 'Not decided'
    return route.replaceAll('_', ' ')
}

function groupClustersByBet(clusters: ResearchKeywordCluster[]) {
    return clusters.reduce<Record<string, ResearchKeywordCluster[]>>((acc, cluster) => {
        const key = cluster.bet_id
        acc[key] = acc[key] || []
        acc[key].push(cluster)
        return acc
    }, {})
}

export function ResearchRebuildStrategicPage() {
    const { activeProject, projects, setActiveProject } = useProject()
    const [searchParams, setSearchParams] = useSearchParams()

    const [categories, setCategories] = React.useState<ProjectCategory[]>([])
    const [topics, setTopics] = React.useState<ResearchRebuildJob[]>([])
    const [runs, setRuns] = React.useState<ResearchStrategyRun[]>([])
    const [runDetail, setRunDetail] = React.useState<ResearchStrategyRunDetail | null>(null)
    const [searchHistory, setSearchHistory] = React.useState<ResearchRebuildDataforseoSearch[]>([])

    const [primaryCategoryId, setPrimaryCategoryId] = React.useState(searchParams.get('primary_category_id') || '')
    const [secondaryCategoryId, setSecondaryCategoryId] = React.useState(searchParams.get('secondary_category_id') || '')
    const [topicDraft, setTopicDraft] = React.useState('')
    const [selectedTopicId, setSelectedTopicId] = React.useState(searchParams.get('topic_id') || '')
    const [selectedClusterId, setSelectedClusterId] = React.useState('')
    const [lookupType, setLookupType] = React.useState<LookupType>('related_keywords')
    const [lookupQuery, setLookupQuery] = React.useState('')
    const [lookupKeywords, setLookupKeywords] = React.useState('')
    const [topicsModalOpen, setTopicsModalOpen] = React.useState(false)
    const [supportingToolsOpen, setSupportingToolsOpen] = React.useState(false)
    const [isLoading, setIsLoading] = React.useState(false)
    const [isRemovingTopic, setIsRemovingTopic] = React.useState<string | null>(null)
    const [error, setError] = React.useState<string | null>(null)

    const projectId = searchParams.get('project_id') || activeProject?.id || ''

    React.useEffect(() => {
        const incomingProjectId = searchParams.get('project_id')
        if (!incomingProjectId || !projects.length || activeProject?.id === incomingProjectId) {
            return
        }
        const match = projects.find((project) => project.id === incomingProjectId) || null
        if (match) {
            setActiveProject(match)
        }
    }, [activeProject?.id, projects, searchParams, setActiveProject])

    React.useEffect(() => {
        if (!projectId || searchParams.get('project_id') === projectId) {
            return
        }
        const next = new URLSearchParams(searchParams)
        next.set('project_id', projectId)
        setSearchParams(next, { replace: true })
    }, [projectId, searchParams, setSearchParams])

    React.useEffect(() => {
        if (!projectId) return
        let cancelled = false
        const loadCategories = async () => {
            const { data, error: categoryError } = await supabase
                .from('project_categories')
                .select('id,project_id,user_id,name,description,slug,level,parent_category_id,sort_order,created_at,updated_at')
                .eq('project_id', projectId)
                .order('sort_order', { ascending: true })
            if (cancelled) return
            if (categoryError) {
                setError(categoryError.message)
                setCategories([])
                return
            }
            setCategories((data as ProjectCategory[]) || [])
        }
        loadCategories()
        return () => {
            cancelled = true
        }
    }, [projectId])

    const primaryCategories = React.useMemo(
        () => categories.filter((category) => Number(category.level) === 1),
        [categories],
    )
    const secondaryCategories = React.useMemo(
        () => categories.filter((category) => String(category.parent_category_id || '') === primaryCategoryId),
        [categories, primaryCategoryId],
    )
    const primaryCategory = primaryCategories.find((category) => category.id === primaryCategoryId) || null
    const secondaryCategory = secondaryCategories.find((category) => category.id === secondaryCategoryId) || null

    const syncScopeParams = React.useCallback((nextTopicId?: string) => {
        const next = new URLSearchParams(searchParams)
        if (projectId) next.set('project_id', projectId)
        if (primaryCategoryId) next.set('primary_category_id', primaryCategoryId)
        else next.delete('primary_category_id')
        if (secondaryCategoryId) next.set('secondary_category_id', secondaryCategoryId)
        else next.delete('secondary_category_id')
        if (nextTopicId) next.set('topic_id', nextTopicId)
        else next.delete('topic_id')
        setSearchParams(next, { replace: true })
    }, [primaryCategoryId, projectId, searchParams, secondaryCategoryId, setSearchParams])

    React.useEffect(() => {
        if (!projectId) return
        const currentPrimary = searchParams.get('primary_category_id') || ''
        const currentSecondary = searchParams.get('secondary_category_id') || ''
        const currentTopic = searchParams.get('topic_id') || ''
        if (
            currentPrimary === primaryCategoryId &&
            currentSecondary === secondaryCategoryId &&
            currentTopic === selectedTopicId
        ) {
            return
        }
        syncScopeParams(selectedTopicId || '')
    }, [primaryCategoryId, projectId, searchParams, secondaryCategoryId, selectedTopicId, syncScopeParams])

    const loadTopicsAndRuns = React.useCallback(async () => {
        if (!projectId) return
        setError(null)
        const [topicResponse, runResponse] = await Promise.all([
            researchRebuildService.listJobs({
                project_id: projectId,
                primary_category_id: primaryCategoryId || undefined,
                secondary_category_id: secondaryCategoryId || undefined,
            }),
            researchRebuildService.listStrategyRuns({
                project_id: projectId,
                primary_category_id: primaryCategoryId || undefined,
                secondary_category_id: secondaryCategoryId || undefined,
                limit: 50,
            }),
        ])
        setTopics(topicResponse.items || [])
        setRuns(runResponse.items || [])
    }, [primaryCategoryId, projectId, secondaryCategoryId])

    const loadRunDetail = React.useCallback(async (runId: string) => {
        const detail = await researchRebuildService.getStrategyRun(runId)
        setRunDetail(detail)
        setSelectedTopicId(detail.topic?.id || '')
        setSelectedClusterId(detail.run.selected_cluster_id || detail.clusters[0]?.id || '')
        syncScopeParams(detail.topic?.id || '')
    }, [syncScopeParams])

    const loadSearchHistory = React.useCallback(async () => {
        if (!projectId) return
        const response = await researchRebuildService.listDataforseoSearches({
            project_id: projectId,
            user_job_id: selectedTopicId || undefined,
            limit: 10,
        })
        setSearchHistory(response.items || [])
    }, [projectId, selectedTopicId])

    React.useEffect(() => {
        if (!projectId) return
        loadTopicsAndRuns().catch((loadError: unknown) => {
            setError(loadError instanceof Error ? loadError.message : 'Failed to load strategic research state.')
        })
    }, [loadTopicsAndRuns, projectId])

    React.useEffect(() => {
        loadSearchHistory().catch(() => undefined)
    }, [loadSearchHistory])

    React.useEffect(() => {
        const runId = searchParams.get('run_id')
        if (runId) {
            loadRunDetail(runId).catch((loadError: unknown) => {
                setError(loadError instanceof Error ? loadError.message : 'Failed to load strategy run.')
            })
            return
        }
        if (!selectedTopicId || !runs.length) return
        const latestRun = runs.find((run) => run.topic_id === selectedTopicId)
        if (latestRun) {
            loadRunDetail(latestRun.id).catch(() => undefined)
        }
    }, [loadRunDetail, runs, searchParams, selectedTopicId])

    const latestRunByTopic = React.useMemo(() => {
        const mapping = new Map<string, ResearchStrategyRun>()
        for (const run of runs) {
            if (!mapping.has(run.topic_id)) {
                mapping.set(run.topic_id, run)
            }
        }
        return mapping
    }, [runs])

    const handleCreateTopic = async () => {
        if (!projectId || !topicDraft.trim()) return
        setIsLoading(true)
        setError(null)
        try {
            const job = await researchRebuildService.createJob({
                project_id: projectId,
                primary_category_id: primaryCategoryId || undefined,
                secondary_category_id: secondaryCategoryId || undefined,
                job_text: topicDraft.trim(),
                job_type_hint: 'hybrid',
                website_context_snapshot: {
                    primary_category_name: primaryCategory?.name || null,
                    secondary_category_name: secondaryCategory?.name || null,
                },
            })
            setTopics((current) => [job, ...current])
            setTopicDraft('')
            setSelectedTopicId(job.id)
            syncScopeParams(job.id)
        } catch (createError) {
            setError(createError instanceof Error ? createError.message : 'Failed to save topic.')
        } finally {
            setIsLoading(false)
        }
    }

    const handleRunTopic = async (topicId?: string) => {
        const resolvedTopicId = topicId || selectedTopicId
        if (!projectId || !resolvedTopicId) return
        setIsLoading(true)
        setError(null)
        try {
            const detail = await researchRebuildService.createStrategyRun({
                project_id: projectId,
                primary_category_id: primaryCategoryId || undefined,
                secondary_category_id: secondaryCategoryId || undefined,
                topic_id: resolvedTopicId,
            })
            setRunDetail(detail)
            setSelectedTopicId(detail.topic?.id || resolvedTopicId)
            setSelectedClusterId(detail.run.selected_cluster_id || detail.clusters[0]?.id || '')
            await loadTopicsAndRuns()
            syncScopeParams(detail.topic?.id || resolvedTopicId)
        } catch (runError) {
            setError(runError instanceof Error ? runError.message : 'Failed to run strategy.')
        } finally {
            setIsLoading(false)
        }
    }

    const handleRemoveTopic = async (topicId: string) => {
        const previousTopics = topics
        setIsRemovingTopic(topicId)
        setTopics((current) => current.filter((topic) => topic.id !== topicId))
        if (selectedTopicId === topicId) {
            setSelectedTopicId('')
            setRunDetail(null)
            syncScopeParams('')
        }
        try {
            await researchRebuildService.archiveJob(topicId)
            await loadTopicsAndRuns()
        } catch (removeError) {
            setTopics(previousTopics)
            setError(removeError instanceof Error ? removeError.message : 'Failed to remove topic.')
        } finally {
            setIsRemovingTopic(null)
        }
    }

    const handleRerunStage = async (stage: 'trends' | 'serp' | 'competitor_mining') => {
        if (!runDetail) return
        setIsLoading(true)
        setError(null)
        try {
            const detail = await researchRebuildService.rerunStrategyStage(runDetail.run.id, { stage })
            setRunDetail(detail)
            setSelectedClusterId(detail.run.selected_cluster_id || detail.clusters[0]?.id || '')
            await loadTopicsAndRuns()
        } catch (rerunError) {
            setError(rerunError instanceof Error ? rerunError.message : 'Failed to rerun stage.')
        } finally {
            setIsLoading(false)
        }
    }

    const handleSelectCluster = async (clusterId?: string) => {
        if (!runDetail) return
        const resolvedClusterId = clusterId || selectedClusterId
        if (!resolvedClusterId) return
        setIsLoading(true)
        setError(null)
        try {
            const detail = await researchRebuildService.selectStrategyCluster(runDetail.run.id, {
                cluster_id: resolvedClusterId,
            })
            setRunDetail(detail)
            setSelectedClusterId(resolvedClusterId)
            await loadTopicsAndRuns()
        } catch (selectError) {
            setError(selectError instanceof Error ? selectError.message : 'Failed to select cluster.')
        } finally {
            setIsLoading(false)
        }
    }

    const handleManualLookup = async () => {
        if (!projectId) return
        const keywords = lookupKeywords
            .split('\n')
            .map((value) => value.trim())
            .filter(Boolean)
        if (lookupType === 'keyword_overview' && !keywords.length) {
            setError('Paste one keyword per line for keyword overview.')
            return
        }
        if (lookupType !== 'keyword_overview' && !lookupQuery.trim()) {
            setError('Enter a query before running the lookup.')
            return
        }
        setIsLoading(true)
        setError(null)
        try {
            await researchRebuildService.runDataforseoSearch({
                project_id: projectId,
                user_job_id: selectedTopicId || undefined,
                primary_category_id: primaryCategoryId || undefined,
                secondary_category_id: secondaryCategoryId || undefined,
                search_type: lookupType,
                query_text: lookupQuery.trim() || undefined,
                keywords: lookupType === 'keyword_overview' ? keywords : undefined,
            })
            await loadSearchHistory()
        } catch (lookupError) {
            setError(lookupError instanceof Error ? lookupError.message : 'Failed to run lookup.')
        } finally {
            setIsLoading(false)
        }
    }

    const clustersByBet = React.useMemo(() => groupClustersByBet(runDetail?.clusters || []), [runDetail?.clusters])
    const finalOutcome = runDetail?.final_selection?.generated_outcome?.outcome_metadata as Record<string, unknown> | undefined

    return (
        <div className="min-h-screen bg-[#05070c] text-white">
            <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-6 py-8">
                <header className="rounded-[28px] border border-white/10 bg-[#0d1018] px-6 py-5 shadow-[0_0_0_1px_rgba(255,255,255,0.02)]">
                    <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
                        <div>
                            <div className="text-xs uppercase tracking-[0.35em] text-sky-300/80">Research Rebuild</div>
                            <h1 className="mt-2 text-3xl font-semibold tracking-tight">Competitive SERP Mining</h1>
                            <p className="mt-2 max-w-3xl text-sm text-slate-300">
                                Start from a topic, pressure-test a few article-angle bets, and only spend keyword credits after the SERP proves the angle is worth chasing.
                            </p>
                        </div>
                        <button
                            type="button"
                            onClick={() => setTopicsModalOpen(true)}
                            className="rounded-full border border-slate-600 bg-slate-900 px-4 py-2 text-sm font-medium text-slate-100 transition hover:border-sky-400 hover:text-white"
                        >
                            Manage Saved Topics
                        </button>
                    </div>
                </header>

                {error ? (
                    <div className="rounded-2xl border border-rose-500/30 bg-rose-950/30 px-4 py-3 text-sm text-rose-200">
                        {error}
                    </div>
                ) : null}

                <section className="rounded-[28px] border border-white/10 bg-[#0f1320] p-6">
                    <div className="mb-5 flex items-center justify-between">
                        <div>
                            <div className="text-xs uppercase tracking-[0.35em] text-sky-300/70">Step 1</div>
                            <h2 className="mt-2 text-2xl font-semibold">Select category and define the topic</h2>
                        </div>
                        {isLoading ? <div className="text-sm text-slate-400">Working…</div> : null}
                    </div>

                    <div className="grid gap-4 lg:grid-cols-2">
                        <label className="flex flex-col gap-2">
                            <span className="text-xs uppercase tracking-[0.28em] text-slate-400">Primary category</span>
                            <select
                                value={primaryCategoryId}
                                onChange={(event) => {
                                    setPrimaryCategoryId(event.target.value)
                                    setSecondaryCategoryId('')
                                }}
                                className="h-14 rounded-2xl border border-slate-700 bg-[#151b2a] px-4 text-base text-white outline-none transition focus:border-sky-400"
                            >
                                <option value="">Select primary category</option>
                                {primaryCategories.map((category) => (
                                    <option key={category.id} value={category.id}>{category.name}</option>
                                ))}
                            </select>
                        </label>
                        <label className="flex flex-col gap-2">
                            <span className="text-xs uppercase tracking-[0.28em] text-slate-400">Sub-category</span>
                            <select
                                value={secondaryCategoryId}
                                onChange={(event) => setSecondaryCategoryId(event.target.value)}
                                className="h-14 rounded-2xl border border-slate-700 bg-[#151b2a] px-4 text-base text-white outline-none transition focus:border-sky-400"
                            >
                                <option value="">Select sub-category</option>
                                {secondaryCategories.map((category) => (
                                    <option key={category.id} value={category.id}>{category.name}</option>
                                ))}
                            </select>
                        </label>
                    </div>

                    <div className="mt-4 rounded-2xl border border-slate-700 bg-[#131827]">
                        <div className="border-b border-slate-700 px-4 py-3 text-xs uppercase tracking-[0.3em] text-slate-400">
                            Category context
                        </div>
                        <div className="max-h-52 overflow-y-auto px-4 py-4 text-sm text-slate-300">
                            <div className="font-medium text-white">{primaryCategory?.name || 'No primary category selected'}</div>
                            <p className="mt-1 whitespace-pre-wrap text-slate-400">{primaryCategory?.description || 'Choose a primary category to anchor the strategic research run.'}</p>
                            <div className="mt-4 font-medium text-white">{secondaryCategory?.name || 'No sub-category selected'}</div>
                            <p className="mt-1 whitespace-pre-wrap text-slate-400">{secondaryCategory?.description || 'Optional: select a sub-category to narrow the angle bets and competitor mining.'}</p>
                        </div>
                    </div>

                    <div className="mt-5 grid gap-4 lg:grid-cols-[1fr_auto_auto]">
                        <textarea
                            value={topicDraft}
                            onChange={(event) => setTopicDraft(event.target.value)}
                            rows={3}
                            placeholder="Example: Eco-friendly home improvements and property value"
                            className="min-h-[112px] rounded-2xl border border-slate-700 bg-[#151b2a] px-4 py-3 text-base text-white outline-none transition placeholder:text-slate-500 focus:border-sky-400"
                        />
                        <button
                            type="button"
                            onClick={handleCreateTopic}
                            disabled={!projectId || !topicDraft.trim() || isLoading}
                            className="rounded-2xl border border-slate-600 bg-slate-900 px-5 py-3 text-sm font-semibold text-slate-100 transition hover:border-sky-400 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            Save Topic
                        </button>
                        <button
                            type="button"
                            onClick={() => handleRunTopic()}
                            disabled={!projectId || !selectedTopicId || isLoading}
                            className="rounded-2xl bg-sky-500 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            Run Strategy
                        </button>
                    </div>

                    <div className="mt-5 rounded-2xl border border-slate-700 bg-[#121826] px-4 py-4">
                        <div className="flex items-center justify-between gap-4">
                            <div>
                                <div className="text-xs uppercase tracking-[0.28em] text-slate-400">Saved topics</div>
                                <div className="mt-2 text-sm text-slate-300">Choose one topic at a time, then run the full article-angle screening workflow.</div>
                            </div>
                            <div className="text-sm text-slate-400">{topics.length} topic{topics.length === 1 ? '' : 's'}</div>
                        </div>
                        <div className="mt-4 flex flex-wrap gap-2">
                            {topics.slice(0, 8).map((topic) => {
                                const latestRun = latestRunByTopic.get(topic.id)
                                const selected = selectedTopicId === topic.id
                                return (
                                    <button
                                        type="button"
                                        key={topic.id}
                                        onClick={() => {
                                            setSelectedTopicId(topic.id)
                                            syncScopeParams(topic.id)
                                            if (latestRun) {
                                                loadRunDetail(latestRun.id).catch(() => undefined)
                                            } else {
                                                setRunDetail(null)
                                            }
                                        }}
                                        className={`rounded-full border px-4 py-2 text-left text-sm transition ${selected ? 'border-sky-400 bg-sky-500/15 text-white' : 'border-slate-700 bg-[#171d2c] text-slate-300 hover:border-slate-500'}`}
                                    >
                                        {topic.job_text}
                                        {latestRun ? <span className="ml-2 text-xs text-slate-400">• {routeLabel(latestRun.winning_route)}</span> : null}
                                    </button>
                                )
                            })}
                        </div>
                    </div>
                </section>

                <section className="rounded-[28px] border border-white/10 bg-[#0f1320] p-6">
                    <div className="mb-5 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                        <div>
                            <div className="text-xs uppercase tracking-[0.35em] text-sky-300/70">Step 2</div>
                            <h2 className="mt-2 text-2xl font-semibold">Review article-angle bets and screening</h2>
                            <p className="mt-2 text-sm text-slate-300">Only the most article-friendly bets should survive into competitor mining and cluster selection.</p>
                        </div>
                        {runDetail ? (
                            <div className="flex flex-wrap gap-2">
                                <button type="button" onClick={() => handleRerunStage('trends')} className="rounded-full border border-slate-600 bg-slate-900 px-4 py-2 text-sm text-slate-100 hover:border-sky-400">Rerun trends</button>
                                <button type="button" onClick={() => handleRerunStage('serp')} className="rounded-full border border-slate-600 bg-slate-900 px-4 py-2 text-sm text-slate-100 hover:border-sky-400">Rerun SERP</button>
                                <button type="button" onClick={() => handleRerunStage('competitor_mining')} className="rounded-full border border-slate-600 bg-slate-900 px-4 py-2 text-sm text-slate-100 hover:border-sky-400">Rerun mining</button>
                            </div>
                        ) : null}
                    </div>

                    {!runDetail ? (
                        <div className="rounded-2xl border border-dashed border-slate-700 bg-[#111725] px-5 py-10 text-sm text-slate-400">
                            Save a topic and run the strategy to generate angle bets, probe queries, and cluster candidates.
                        </div>
                    ) : (
                        <div className="grid gap-4">
                            {runDetail.bets.map((bet: ResearchTopicBet) => {
                                const betClusters = clustersByBet[bet.id] || []
                                const reasonCodes = bet.reason_codes || []
                                const probes = runDetail.probe_queries.filter((probe) => probe.bet_id === bet.id)
                                return (
                                    <article key={bet.id} className={`rounded-3xl border p-5 ${bet.status === 'survived' ? 'border-emerald-500/30 bg-[#101a18]' : 'border-slate-700 bg-[#121826]'}`}>
                                        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                                            <div className="max-w-3xl">
                                                <div className="text-xs uppercase tracking-[0.3em] text-slate-400">{bet.status === 'survived' ? 'Survived' : 'Killed'} bet</div>
                                                <h3 className="mt-2 text-xl font-semibold text-white">{bet.bet_text}</h3>
                                                <p className="mt-2 text-sm text-slate-300">{bet.searcher_problem || 'No searcher problem captured.'}</p>
                                                <div className="mt-4 flex flex-wrap gap-2 text-xs text-slate-300">
                                                    <span className="rounded-full border border-slate-600 px-3 py-1">{bet.article_format || 'format pending'}</span>
                                                    <span className="rounded-full border border-slate-600 px-3 py-1">{bet.commercial_angle || 'commercial angle pending'}</span>
                                                    <span className="rounded-full border border-slate-600 px-3 py-1">{bet.route_hint || 'route pending'}</span>
                                                </div>
                                            </div>
                                            <div className="grid min-w-[220px] gap-2 rounded-2xl border border-slate-700 bg-[#0b111d] p-4 text-sm">
                                                <div className="flex items-center justify-between">
                                                    <span className="text-slate-400">Trend</span>
                                                    <span className={toneClass(bet.trend_score)}>{scoreLabel(bet.trend_score)}</span>
                                                </div>
                                                <div className="flex items-center justify-between">
                                                    <span className="text-slate-400">Articleability</span>
                                                    <span className={toneClass(bet.serp_articleability_score)}>{scoreLabel(bet.serp_articleability_score)}</span>
                                                </div>
                                                <div className="flex items-center justify-between">
                                                    <span className="text-slate-400">SERP weakness</span>
                                                    <span className={toneClass(bet.serp_weakness_score)}>{scoreLabel(bet.serp_weakness_score)}</span>
                                                </div>
                                                <div className="flex items-center justify-between">
                                                    <span className="text-slate-400">Article fit</span>
                                                    <span className={toneClass(bet.article_fit_score)}>{scoreLabel(bet.article_fit_score)}</span>
                                                </div>
                                            </div>
                                        </div>

                                        <div className="mt-5 grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
                                            <div className="rounded-2xl border border-slate-700 bg-[#0e1421] p-4">
                                                <div className="text-xs uppercase tracking-[0.28em] text-slate-400">Probe queries</div>
                                                <div className="mt-3 grid gap-2">
                                                    {probes.map((probe) => (
                                                        <div key={probe.id} className="rounded-xl border border-slate-700 bg-[#141a29] px-3 py-3 text-sm text-slate-200">
                                                            <div className="font-medium text-white">{probe.query_text}</div>
                                                            <div className="mt-2 text-xs text-slate-400">
                                                                {probe.serp_classification || 'SERP not classified yet'}
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                                {reasonCodes.length ? (
                                                    <div className="mt-4 flex flex-wrap gap-2">
                                                        {reasonCodes.map((code) => (
                                                            <span key={code} className="rounded-full border border-slate-600 px-3 py-1 text-xs text-slate-300">
                                                                {code.replaceAll('_', ' ')}
                                                            </span>
                                                        ))}
                                                    </div>
                                                ) : null}
                                            </div>

                                            <div className="rounded-2xl border border-slate-700 bg-[#0e1421] p-4">
                                                <div className="text-xs uppercase tracking-[0.28em] text-slate-400">Clusters</div>
                                                {betClusters.length ? (
                                                    <div className="mt-3 grid gap-3">
                                                        {betClusters.map((cluster) => (
                                                            <button
                                                                type="button"
                                                                key={cluster.id}
                                                                onClick={() => setSelectedClusterId(cluster.id)}
                                                                className={`rounded-2xl border p-4 text-left transition ${selectedClusterId === cluster.id ? 'border-sky-400 bg-sky-500/10' : 'border-slate-700 bg-[#141a29] hover:border-slate-500'}`}
                                                            >
                                                                <div className="flex items-start justify-between gap-4">
                                                                    <div>
                                                                        <div className="font-medium text-white">{cluster.cluster_name}</div>
                                                                        <div className="mt-1 text-sm text-slate-300">{cluster.primary_keyword_candidate || 'No primary keyword yet'}</div>
                                                                    </div>
                                                                    <div className={`text-sm font-medium ${toneClass(cluster.opportunity_score)}`}>
                                                                        {Math.round(Number(cluster.opportunity_score || 0) * 100)}%
                                                                    </div>
                                                                </div>
                                                                <div className="mt-3 text-xs text-slate-400">
                                                                    {cluster.secondary_keywords_json?.length || 0} secondaries • {cluster.supporting_competitor_urls_json?.length || 0} competitor URLs
                                                                </div>
                                                            </button>
                                                        ))}
                                                    </div>
                                                ) : (
                                                    <div className="mt-3 text-sm text-slate-400">
                                                        {bet.status === 'survived' ? 'This bet survived, but competitor mining has not produced a strong cluster yet.' : 'Killed bets do not proceed into cluster mining.'}
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                    </article>
                                )
                            })}
                        </div>
                    )}
                </section>

                <section className="rounded-[28px] border border-white/10 bg-[#0f1320] p-6">
                    <div className="mb-5 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                        <div>
                            <div className="text-xs uppercase tracking-[0.35em] text-sky-300/70">Step 3</div>
                            <h2 className="mt-2 text-2xl font-semibold">Choose the winning cluster and define the article</h2>
                            <p className="mt-2 text-sm text-slate-300">The final decision is cluster-first: choose the angle with the best SERP fit, competitor support, and commercial depth.</p>
                        </div>
                        <button
                            type="button"
                            onClick={() => handleSelectCluster()}
                            disabled={!runDetail || !selectedClusterId || isLoading}
                            className="rounded-2xl bg-emerald-400 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            Lock cluster and generate output
                        </button>
                    </div>

                    {runDetail ? (
                        <div className="grid gap-5 xl:grid-cols-[0.95fr_1.05fr]">
                            <div className="rounded-3xl border border-slate-700 bg-[#101726] p-5">
                                <div className="text-xs uppercase tracking-[0.28em] text-slate-400">Winning route</div>
                                <div className="mt-2 text-2xl font-semibold text-white">{routeLabel(runDetail.run.winning_route)}</div>
                                <div className="mt-2 text-sm text-slate-300">
                                    Confidence {Math.round(Number(runDetail.run.confidence_score || 0) * 100)}%
                                </div>
                                <div className="mt-5 grid gap-3">
                                    {(runDetail.clusters || []).map((cluster) => (
                                        <div key={cluster.id} className={`rounded-2xl border p-4 ${selectedClusterId === cluster.id ? 'border-sky-400 bg-sky-500/10' : 'border-slate-700 bg-[#141a29]'}`}>
                                            <div className="flex items-start justify-between gap-4">
                                                <div>
                                                    <div className="font-medium text-white">{cluster.cluster_name}</div>
                                                    <div className="mt-1 text-sm text-slate-300">{cluster.primary_keyword_candidate}</div>
                                                </div>
                                                <div className={`text-sm ${toneClass(cluster.opportunity_score)}`}>
                                                    {Math.round(Number(cluster.opportunity_score || 0) * 100)}%
                                                </div>
                                            </div>
                                            <div className="mt-3 grid gap-2 text-xs text-slate-400">
                                                <div>SERP weakness: {Math.round(Number(cluster.serp_weakness_score || 0) * 100)}%</div>
                                                <div>Competitor support: {Math.round(Number(cluster.competitor_support_score || 0) * 100)}%</div>
                                                <div>Commercial value: {Math.round(Number(cluster.commercial_value_score || 0) * 100)}%</div>
                                            </div>
                                            {cluster.supporting_competitor_urls_json?.length ? (
                                                <div className="mt-3 flex flex-wrap gap-2">
                                                    {cluster.supporting_competitor_urls_json.slice(0, 3).map((url) => (
                                                        <a key={url} href={url} target="_blank" rel="noreferrer" className="truncate rounded-full border border-slate-600 px-3 py-1 text-xs text-slate-200 hover:border-sky-400">
                                                            {url}
                                                        </a>
                                                    ))}
                                                </div>
                                            ) : null}
                                        </div>
                                    ))}
                                </div>
                            </div>

                            <div className="rounded-3xl border border-slate-700 bg-[#101726] p-5">
                                <div className="text-xs uppercase tracking-[0.28em] text-slate-400">Final output</div>
                                {finalOutcome ? (
                                    <div className="mt-4">
                                        <h3 className="text-2xl font-semibold text-white">{String(finalOutcome.title || 'Untitled output')}</h3>
                                        <div className="mt-2 text-sm text-slate-300">
                                            Primary keyword: <span className="font-medium text-white">{String(finalOutcome.primary_keyword || '')}</span>
                                        </div>
                                        <div className="mt-2 text-sm text-slate-300">
                                            Slug: <span className="font-mono text-slate-100">{String(finalOutcome.slug || '')}</span>
                                        </div>
                                        <div className="mt-4 flex flex-wrap gap-2">
                                            {Array.isArray(finalOutcome.secondary_keywords) ? finalOutcome.secondary_keywords.map((keyword) => (
                                                <span key={String(keyword)} className="rounded-full border border-slate-600 px-3 py-1 text-xs text-slate-200">
                                                    {String(keyword)}
                                                </span>
                                            )) : null}
                                        </div>
                                        <div className="mt-5 rounded-2xl border border-slate-700 bg-[#0d1320] p-4">
                                            <div className="text-xs uppercase tracking-[0.28em] text-slate-400">Outline</div>
                                            <ul className="mt-3 grid gap-2 text-sm text-slate-200">
                                                {Array.isArray(finalOutcome.outline) ? finalOutcome.outline.map((item) => (
                                                    <li key={String(item)} className="rounded-xl border border-slate-700 bg-[#131a29] px-3 py-2">
                                                        {String(item)}
                                                    </li>
                                                )) : null}
                                            </ul>
                                        </div>
                                        <div className="mt-4 text-sm text-slate-300">
                                            {String(finalOutcome.rationale || '')}
                                        </div>
                                    </div>
                                ) : (
                                    <div className="mt-4 rounded-2xl border border-dashed border-slate-700 bg-[#111725] px-4 py-8 text-sm text-slate-400">
                                        Choose a surviving cluster, then lock it to generate the title, slug, keyword map, and outline.
                                    </div>
                                )}
                            </div>
                        </div>
                    ) : (
                        <div className="rounded-2xl border border-dashed border-slate-700 bg-[#111725] px-5 py-10 text-sm text-slate-400">
                            No strategic run loaded yet.
                        </div>
                    )}
                </section>

                <section className="rounded-[28px] border border-white/10 bg-[#0f1320] p-6">
                    <button
                        type="button"
                        onClick={() => setSupportingToolsOpen((current) => !current)}
                        className="flex w-full items-center justify-between text-left"
                    >
                        <div>
                            <div className="text-xs uppercase tracking-[0.35em] text-slate-400">Supporting tools</div>
                            <div className="mt-2 text-lg font-semibold text-white">Manual DataForSEO lookups</div>
                        </div>
                        <div className="text-sm text-slate-400">{supportingToolsOpen ? 'Hide' : 'Show'}</div>
                    </button>

                    {supportingToolsOpen ? (
                        <div className="mt-5 grid gap-5 xl:grid-cols-[0.9fr_1.1fr]">
                            <div className="rounded-2xl border border-slate-700 bg-[#111826] p-4">
                                <div className="grid gap-3">
                                    <label className="flex flex-col gap-2">
                                        <span className="text-xs uppercase tracking-[0.28em] text-slate-400">Lookup type</span>
                                        <select value={lookupType} onChange={(event) => setLookupType(event.target.value as LookupType)} className="h-12 rounded-xl border border-slate-700 bg-[#151b2a] px-3 text-white outline-none focus:border-sky-400">
                                            <option value="related_keywords">Related keywords</option>
                                            <option value="keyword_overview">Keyword overview</option>
                                            <option value="serp">SERP snapshot</option>
                                        </select>
                                    </label>
                                    <label className="flex flex-col gap-2">
                                        <span className="text-xs uppercase tracking-[0.28em] text-slate-400">Query</span>
                                        <input value={lookupQuery} onChange={(event) => setLookupQuery(event.target.value)} className="h-12 rounded-xl border border-slate-700 bg-[#151b2a] px-3 text-white outline-none focus:border-sky-400" placeholder="Exact seed or probe query" />
                                    </label>
                                    {lookupType === 'keyword_overview' ? (
                                        <label className="flex flex-col gap-2">
                                            <span className="text-xs uppercase tracking-[0.28em] text-slate-400">Keywords</span>
                                            <textarea value={lookupKeywords} onChange={(event) => setLookupKeywords(event.target.value)} rows={5} className="rounded-xl border border-slate-700 bg-[#151b2a] px-3 py-3 text-white outline-none focus:border-sky-400" placeholder="One keyword per line" />
                                        </label>
                                    ) : null}
                                    <button type="button" onClick={handleManualLookup} className="rounded-xl bg-slate-100 px-4 py-3 text-sm font-semibold text-slate-950 hover:bg-white">
                                        Run and save lookup
                                    </button>
                                </div>
                            </div>
                            <div className="rounded-2xl border border-slate-700 bg-[#111826] p-4">
                                <div className="text-xs uppercase tracking-[0.28em] text-slate-400">Recent lookup history</div>
                                <div className="mt-3 grid gap-3">
                                    {searchHistory.map((row) => (
                                        <div key={row.id} className="rounded-xl border border-slate-700 bg-[#151b2a] p-4">
                                            <div className="flex items-start justify-between gap-3">
                                                <div>
                                                    <div className="font-medium text-white">{row.query_text || row.search_type}</div>
                                                    <div className="mt-1 text-xs uppercase tracking-[0.24em] text-slate-400">{row.search_type.replaceAll('_', ' ')}</div>
                                                </div>
                                                <div className="text-xs text-slate-500">{row.searched_at ? new Date(row.searched_at).toLocaleDateString() : ''}</div>
                                            </div>
                                            <div className="mt-3 text-sm text-slate-300">
                                                {(row.result_summary_json as Record<string, unknown> | undefined)?.result_count ? `${String((row.result_summary_json as Record<string, unknown>).result_count)} results captured` : 'Saved for later mining'}
                                            </div>
                                        </div>
                                    ))}
                                    {!searchHistory.length ? (
                                        <div className="rounded-xl border border-dashed border-slate-700 bg-[#151b2a] px-4 py-6 text-sm text-slate-400">
                                            No saved supporting lookups yet.
                                        </div>
                                    ) : null}
                                </div>
                            </div>
                        </div>
                    ) : null}
                </section>
            </div>

            {topicsModalOpen ? (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4">
                    <div className="max-h-[80vh] w-full max-w-3xl overflow-hidden rounded-[28px] border border-slate-700 bg-[#0f1320] shadow-2xl">
                        <div className="flex items-center justify-between border-b border-slate-700 px-6 py-4">
                            <div>
                                <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Saved topics</div>
                                <div className="mt-2 text-xl font-semibold text-white">Manage topics in scope</div>
                            </div>
                            <button type="button" onClick={() => setTopicsModalOpen(false)} className="rounded-full border border-slate-600 px-3 py-2 text-sm text-slate-200 hover:border-slate-400">Close</button>
                        </div>
                        <div className="max-h-[60vh] overflow-y-auto px-6 py-5">
                            <div className="grid gap-3">
                                {topics.map((topic) => {
                                    const latestRun = latestRunByTopic.get(topic.id)
                                    return (
                                        <div key={topic.id} className="rounded-2xl border border-slate-700 bg-[#131827] p-4">
                                            <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                                                <div>
                                                    <div className="font-medium text-white">{topic.job_text}</div>
                                                    <div className="mt-1 text-sm text-slate-400">
                                                        {latestRun ? `Latest route: ${routeLabel(latestRun.winning_route)}` : 'No strategy run yet'}
                                                    </div>
                                                </div>
                                                <div className="flex gap-2">
                                                    <button type="button" onClick={() => {
                                                        setSelectedTopicId(topic.id)
                                                        setTopicsModalOpen(false)
                                                        syncScopeParams(topic.id)
                                                    }} className="rounded-full border border-slate-600 px-4 py-2 text-sm text-slate-100 hover:border-sky-400">Use topic</button>
                                                    <button type="button" onClick={() => handleRemoveTopic(topic.id)} disabled={isRemovingTopic === topic.id} className="rounded-full border border-rose-500/40 px-4 py-2 text-sm text-rose-200 hover:border-rose-400 disabled:opacity-50">
                                                        {isRemovingTopic === topic.id ? 'Removing…' : 'Remove'}
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    )
                                })}
                            </div>
                        </div>
                    </div>
                </div>
            ) : null}
        </div>
    )
}

export default ResearchRebuildStrategicPage
