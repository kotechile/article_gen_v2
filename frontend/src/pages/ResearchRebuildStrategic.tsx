import * as React from 'react'
import { useSearchParams } from 'react-router-dom'

import { useProject } from '@/context/project-context'
import { supabase } from '@/lib/supabase'
import { researchRebuildService } from '@/services/research-rebuild.service'
import type { ProjectCategory } from '@/types/command-center'
import type {
    ResearchFeasibleKeywordOpportunity,
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

function groupCompetitorPagesByBet(pages: ResearchStrategyRunDetail['competitor_pages']) {
    return pages.reduce<Record<string, ResearchStrategyRunDetail['competitor_pages']>>((acc, page) => {
        const key = page.bet_id
        acc[key] = acc[key] || []
        acc[key].push(page)
        return acc
    }, {})
}

function clusterMetadata(cluster?: ResearchKeywordCluster | null) {
    return (cluster?.cluster_metadata as Record<string, unknown> | undefined) || {}
}

export function ResearchRebuildStrategicPage() {
    const { activeProject, projects, setActiveProject } = useProject()
    const [searchParams, setSearchParams] = useSearchParams()
    const skipNextAutoSelectRef = React.useRef(false)
    const lastLoadedRunIdRef = React.useRef<string>('')
    const primaryCategoryParam = searchParams.get('primary_category_id') || ''
    const secondaryCategoryParam = searchParams.get('secondary_category_id') || ''
    const topicIdParam = searchParams.get('topic_id') || ''
    const runIdParam = searchParams.get('run_id') || ''

    const [categories, setCategories] = React.useState<ProjectCategory[]>([])
    const [topics, setTopics] = React.useState<ResearchRebuildJob[]>([])
    const [runs, setRuns] = React.useState<ResearchStrategyRun[]>([])
    const [runDetail, setRunDetail] = React.useState<ResearchStrategyRunDetail | null>(null)
    const [searchHistory, setSearchHistory] = React.useState<ResearchRebuildDataforseoSearch[]>([])
    const [feasibleKeywords, setFeasibleKeywords] = React.useState<ResearchFeasibleKeywordOpportunity[]>([])

    const [primaryCategoryId, setPrimaryCategoryId] = React.useState(primaryCategoryParam)
    const [secondaryCategoryId, setSecondaryCategoryId] = React.useState(secondaryCategoryParam)
    const [topicDraft, setTopicDraft] = React.useState('')
    const [selectedTopicId, setSelectedTopicId] = React.useState(topicIdParam)
    const [selectedClusterId, setSelectedClusterId] = React.useState('')
    const [lookupType, setLookupType] = React.useState<LookupType>('related_keywords')
    const [lookupQuery, setLookupQuery] = React.useState('')
    const [lookupKeywords, setLookupKeywords] = React.useState('')
    const [topicsModalOpen, setTopicsModalOpen] = React.useState(false)
    const [supportingToolsOpen, setSupportingToolsOpen] = React.useState(false)
    const [isLoading, setIsLoading] = React.useState(false)
    const [loadingLabel, setLoadingLabel] = React.useState('Working through the strategic research flow…')
    const [runningTopicId, setRunningTopicId] = React.useState<string | null>(null)
    const [isRemovingTopic, setIsRemovingTopic] = React.useState<string | null>(null)
    const [error, setError] = React.useState<string | null>(null)
    const [success, setSuccess] = React.useState<string | null>(null)
    const [mutatingOutcomeAction, setMutatingOutcomeAction] = React.useState<'release' | 'persist' | 'dismiss' | null>(null)

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

    const applyScopeSelection = React.useCallback((payload: {
        topicId?: string
        primaryCategoryId?: string
        secondaryCategoryId?: string
    }) => {
        const nextPrimary = payload.primaryCategoryId || ''
        const nextSecondary = payload.secondaryCategoryId || ''
        const nextTopic = payload.topicId || ''
        setPrimaryCategoryId(nextPrimary)
        setSecondaryCategoryId(nextSecondary)
        setSelectedTopicId(nextTopic)
    }, [])

    const syncScopeParams = React.useCallback((nextTopicId?: string, nextRunId?: string) => {
        setSearchParams((currentParams) => {
            const next = new URLSearchParams(currentParams)
            if (projectId) next.set('project_id', projectId)
            if (primaryCategoryId) next.set('primary_category_id', primaryCategoryId)
            else next.delete('primary_category_id')
            if (secondaryCategoryId) next.set('secondary_category_id', secondaryCategoryId)
            else next.delete('secondary_category_id')
            if (nextTopicId) next.set('topic_id', nextTopicId)
            else next.delete('topic_id')
            if (nextRunId) next.set('run_id', nextRunId)
            else next.delete('run_id')
            return next
        }, { replace: true })
    }, [primaryCategoryId, projectId, secondaryCategoryId, setSearchParams])

    React.useEffect(() => {
        if (!projectId) return
        const selectedRun = runDetail?.run.id || ''
        if (
            primaryCategoryParam === primaryCategoryId &&
            secondaryCategoryParam === secondaryCategoryId &&
            topicIdParam === selectedTopicId &&
            runIdParam === selectedRun
        ) {
            return
        }
        syncScopeParams(selectedTopicId || '', selectedRun)
    }, [primaryCategoryId, projectId, runDetail?.run.id, secondaryCategoryId, selectedTopicId, syncScopeParams, primaryCategoryParam, secondaryCategoryParam, topicIdParam, runIdParam])

    const loadTopicsAndRuns = React.useCallback(async () => {
        if (!projectId) return
        setError(null)
        setSuccess(null)
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
        if (!runId) return
        if (lastLoadedRunIdRef.current === runId && runDetail?.run.id === runId) {
            return
        }
        lastLoadedRunIdRef.current = runId
        const detail = await researchRebuildService.getStrategyRun(runId)
        setRunDetail(detail)
        applyScopeSelection({
            topicId: detail.topic?.id || '',
            primaryCategoryId: String(detail.run.primary_category_id || detail.topic?.primary_category_id || ''),
            secondaryCategoryId: String(detail.run.secondary_category_id || detail.topic?.secondary_category_id || ''),
        })
        setSelectedClusterId(detail.run.selected_cluster_id || detail.clusters[0]?.id || '')
        syncScopeParams(detail.topic?.id || '', detail.run.id)
    }, [applyScopeSelection, runDetail?.run.id, syncScopeParams])

    const loadSearchHistory = React.useCallback(async () => {
        if (!projectId) return
        const response = await researchRebuildService.listDataforseoSearches({
            project_id: projectId,
            user_job_id: selectedTopicId || undefined,
            limit: 10,
        })
        setSearchHistory(response.items || [])
    }, [projectId, selectedTopicId])

    const loadFeasibleKeywords = React.useCallback(async () => {
        if (!projectId) return
        const response = await researchRebuildService.listFeasibleKeywords({
            project_id: projectId,
            primary_category_id: primaryCategoryId || undefined,
            secondary_category_id: secondaryCategoryId || undefined,
            include_used: false,
            limit: 200,
        })
        setFeasibleKeywords(response.items || [])
    }, [primaryCategoryId, projectId, secondaryCategoryId])

    React.useEffect(() => {
        if (!projectId) return
        loadTopicsAndRuns().catch((loadError: unknown) => {
            setError(loadError instanceof Error ? loadError.message : 'Failed to load strategic research state.')
        })
    }, [loadTopicsAndRuns, projectId])

    React.useEffect(() => {
        if (!projectId) return
        loadFeasibleKeywords().catch((loadError: unknown) => {
            setError(loadError instanceof Error ? loadError.message : 'Failed to load feasible keywords.')
        })
    }, [loadFeasibleKeywords, projectId])

    React.useEffect(() => {
        if (!topics.length) return
        if (skipNextAutoSelectRef.current) {
            skipNextAutoSelectRef.current = false
            return
        }
        if (selectedTopicId && topics.some((topic) => topic.id === selectedTopicId)) {
            return
        }

        if (topicIdParam && topics.some((topic) => topic.id === topicIdParam)) {
            const matchedTopic = topics.find((topic) => topic.id === topicIdParam)
            applyScopeSelection({
                topicId: topicIdParam,
                primaryCategoryId: String(matchedTopic?.primary_category_id || ''),
                secondaryCategoryId: String(matchedTopic?.secondary_category_id || ''),
            })
            return
        }

        const latestRunInScope = runs[0]
        if (latestRunInScope?.topic_id && topics.some((topic) => topic.id === latestRunInScope.topic_id)) {
            applyScopeSelection({
                topicId: latestRunInScope.topic_id,
                primaryCategoryId: String(latestRunInScope.primary_category_id || ''),
                secondaryCategoryId: String(latestRunInScope.secondary_category_id || ''),
            })
            return
        }

        applyScopeSelection({
            topicId: topics[0].id,
            primaryCategoryId: String(topics[0].primary_category_id || ''),
            secondaryCategoryId: String(topics[0].secondary_category_id || ''),
        })
    }, [applyScopeSelection, runs, selectedTopicId, topics, topicIdParam])

    React.useEffect(() => {
        if (!selectedTopicId || !topics.length) return
        const matchedTopic = topics.find((topic) => topic.id === selectedTopicId)
        if (!matchedTopic) return

        const nextPrimary = String(matchedTopic.primary_category_id || '')
        const nextSecondary = String(matchedTopic.secondary_category_id || '')
        if (!primaryCategoryId && nextPrimary) {
            setPrimaryCategoryId(nextPrimary)
        }
        if (!secondaryCategoryId && nextSecondary) {
            setSecondaryCategoryId(nextSecondary)
        }
    }, [primaryCategoryId, secondaryCategoryId, selectedTopicId, topics])

    React.useEffect(() => {
        loadSearchHistory().catch(() => undefined)
    }, [loadSearchHistory])

    React.useEffect(() => {
        if (runIdParam) {
            loadRunDetail(runIdParam).catch((loadError: unknown) => {
                lastLoadedRunIdRef.current = ''
                setError(loadError instanceof Error ? loadError.message : 'Failed to load strategy run.')
            })
            return
        }
        if (!runs.length) return

        const topicIdToLoad = selectedTopicId || topicIdParam || runs[0]?.topic_id || ''
        if (!topicIdToLoad) return

        const latestRun = runs.find((run) => run.topic_id === topicIdToLoad)
        if (latestRun) {
            loadRunDetail(latestRun.id).catch(() => undefined)
        } else {
            lastLoadedRunIdRef.current = ''
            setRunDetail(null)
            setSelectedClusterId('')
            syncScopeParams(topicIdToLoad, '')
        }
    }, [loadRunDetail, runs, selectedTopicId, syncScopeParams, runIdParam, topicIdParam])

    const latestRunByTopic = React.useMemo(() => {
        const mapping = new Map<string, ResearchStrategyRun>()
        for (const run of runs) {
            if (!mapping.has(run.topic_id)) {
                mapping.set(run.topic_id, run)
            }
        }
        return mapping
    }, [runs])

    const selectedTopic = React.useMemo(
        () => topics.find((topic) => topic.id === selectedTopicId) || null,
        [selectedTopicId, topics],
    )

    const handleCreateTopic = async () => {
        if (!projectId || !topicDraft.trim()) return
        setIsLoading(true)
        setLoadingLabel('Saving topic…')
        setError(null)
        setSuccess(null)
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
            applyScopeSelection({
                topicId: job.id,
                primaryCategoryId: String(job.primary_category_id || primaryCategoryId || ''),
                secondaryCategoryId: String(job.secondary_category_id || secondaryCategoryId || ''),
            })
            syncScopeParams(job.id)
            await loadFeasibleKeywords()
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
        setRunningTopicId(resolvedTopicId)
        setLoadingLabel('Generating seed queries, checking SERPs, and mining competitor URLs…')
        setError(null)
        setSuccess(null)
        try {
            const detail = await researchRebuildService.createStrategyRun({
                project_id: projectId,
                primary_category_id: primaryCategoryId || undefined,
                secondary_category_id: secondaryCategoryId || undefined,
                topic_id: resolvedTopicId,
            })
            setRunDetail(detail)
            applyScopeSelection({
                topicId: detail.topic?.id || resolvedTopicId,
                primaryCategoryId: String(detail.run.primary_category_id || detail.topic?.primary_category_id || primaryCategoryId || ''),
                secondaryCategoryId: String(detail.run.secondary_category_id || detail.topic?.secondary_category_id || secondaryCategoryId || ''),
            })
            setSelectedClusterId(detail.run.selected_cluster_id || detail.clusters[0]?.id || '')
            await loadTopicsAndRuns()
            await loadFeasibleKeywords()
            syncScopeParams(detail.topic?.id || resolvedTopicId, detail.run.id)
        } catch (runError) {
            setError(runError instanceof Error ? runError.message : 'Failed to run strategy.')
        } finally {
            setRunningTopicId(null)
            setIsLoading(false)
        }
    }

    const handleRemoveTopic = async (topicId: string) => {
        const previousTopics = topics
        setIsRemovingTopic(topicId)
        setTopics((current) => current.filter((topic) => topic.id !== topicId))
        if (selectedTopicId === topicId) {
            skipNextAutoSelectRef.current = true
            applyScopeSelection({
                topicId: '',
                primaryCategoryId,
                secondaryCategoryId,
            })
            setRunDetail(null)
            syncScopeParams('')
        }
        try {
            await researchRebuildService.archiveJob(topicId)
            await loadTopicsAndRuns()
            await loadFeasibleKeywords()
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
        setLoadingLabel(`Rerunning ${stage.replace('_', ' ')}…`)
        setError(null)
        setSuccess(null)
        try {
            const detail = await researchRebuildService.rerunStrategyStage(runDetail.run.id, { stage })
            setRunDetail(detail)
            setSelectedClusterId(detail.run.selected_cluster_id || detail.clusters[0]?.id || '')
            await loadTopicsAndRuns()
            await loadFeasibleKeywords()
        } catch (rerunError) {
            setError(rerunError instanceof Error ? rerunError.message : 'Failed to rerun stage.')
        } finally {
            setIsLoading(false)
        }
    }

    const handleSelectCluster = async (clusterId?: string) => {
        if (!runDetail) return
        const resolvedClusterId = clusterId || selectedClusterId
        if (isArticleRoute && !resolvedClusterId) return
        setIsLoading(true)
        setLoadingLabel(
            isSoftwareRoute
                ? 'Generating the software output from the winning bet…'
                : isEditorialRoute
                    ? 'Generating the editorial output from the winning bet…'
                    : 'Locking the cluster and generating the final output…',
        )
        setError(null)
        setSuccess(null)
        try {
            const detail = await researchRebuildService.selectStrategyCluster(runDetail.run.id, {
                cluster_id: resolvedClusterId || undefined,
            })
            setRunDetail(detail)
            setSelectedClusterId(resolvedClusterId || detail.run.selected_cluster_id || detail.clusters[0]?.id || '')
            await loadTopicsAndRuns()
            await loadFeasibleKeywords()
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
        setLoadingLabel('Running and saving the supporting DataForSEO lookup…')
        setError(null)
        setSuccess(null)
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
    const competitorPagesByBet = React.useMemo(() => groupCompetitorPagesByBet(runDetail?.competitor_pages || []), [runDetail?.competitor_pages])
    const finalOutcome = runDetail?.final_selection?.generated_outcome?.outcome_metadata as Record<string, unknown> | undefined
    const winningRoute = runDetail?.run.winning_route || null
    const selectedBet = React.useMemo(
        () => runDetail?.bets.find((bet) => bet.id === runDetail?.run.selected_bet_id) || null,
        [runDetail],
    )
    const selectedCluster = React.useMemo(
        () => runDetail?.clusters.find((cluster) => cluster.id === selectedClusterId) || runDetail?.clusters[0] || null,
        [runDetail, selectedClusterId],
    )
    const selectedClusterMetadata = React.useMemo(() => clusterMetadata(selectedCluster), [selectedCluster])
    const isArticleRoute = winningRoute === 'article_ready'
    const isSoftwareRoute = winningRoute === 'software_ready'
    const isEditorialRoute = winningRoute === 'editorial_only'
    const canFinalizeSelection = Boolean(
        runDetail && (
            (isArticleRoute && selectedClusterId) ||
            ((isSoftwareRoute || isEditorialRoute) && selectedBet)
        ),
    )
    const finalizeLabel = isSoftwareRoute
        ? 'Generate software output'
        : isEditorialRoute
            ? 'Generate editorial output'
            : 'Lock keyword and generate output'
    const generatedOutcome = runDetail?.final_selection?.generated_outcome || null
    const generatedOutcomeId = generatedOutcome?.id ? String(generatedOutcome.id) : null
    const isDismissed = String(runDetail?.run.status || '') === 'dismissed'
    const softwareName = String(finalOutcome?.product_name || finalOutcome?.title || selectedBet?.bet_text || 'Untitled software concept')
    const softwareSearchAngle = String(finalOutcome?.primary_keyword || '')
    const softwareKeywords = Array.isArray(finalOutcome?.secondary_keywords) ? finalOutcome.secondary_keywords.map(String) : []
    const softwareCoreWorkflow = Array.isArray(finalOutcome?.core_workflow) ? finalOutcome.core_workflow.map(String) : []
    const softwareFeatures = Array.isArray(finalOutcome?.key_features) ? finalOutcome.key_features.map(String) : []
    const softwareInputs = Array.isArray(finalOutcome?.inputs) ? finalOutcome.inputs.map(String) : []
    const softwareOutputs = Array.isArray(finalOutcome?.outputs) ? finalOutcome.outputs.map(String) : []
    const softwareMvpScope = Array.isArray(finalOutcome?.mvp_scope) ? finalOutcome.mvp_scope.map(String) : []
    const hasDetailedSoftwareSpec = Boolean(
        String(finalOutcome?.software_concept || '').trim()
        || softwareCoreWorkflow.length
        || softwareFeatures.length
        || softwareInputs.length
        || softwareOutputs.length
        || softwareMvpScope.length,
    )
    const keywordOpportunities = React.useMemo(
        () => (runDetail?.clusters || []).filter((cluster) => String(cluster.cluster_type || '') === 'keyword_opportunity'),
        [runDetail?.clusters],
    )

    const handleReleaseSoftwareIdea = async () => {
        if (!generatedOutcomeId) return
        setMutatingOutcomeAction('release')
        setError(null)
        setSuccess(null)
        try {
            await researchRebuildService.releaseSoftwareOutcome(generatedOutcomeId)
            const refreshed = await researchRebuildService.getStrategyRun(runDetail!.run.id)
            setRunDetail(refreshed)
            setSuccess('Software outcome sent to Software Ideas.')
        } catch (releaseError) {
            setError(releaseError instanceof Error ? releaseError.message : 'Failed to release software outcome.')
        } finally {
            setMutatingOutcomeAction(null)
        }
    }

    const handleSendToContentStudio = async () => {
        if (!generatedOutcomeId || !projectId) return
        setMutatingOutcomeAction('persist')
        setError(null)
        setSuccess(null)
        try {
            await researchRebuildService.persistOutcomeToContentIdea(generatedOutcomeId, {
                project_id: projectId,
                topic_id: selectedTopicId || undefined,
                category_context: {
                    project_id: projectId,
                    primary_category_id: primaryCategoryId || null,
                    secondary_category_id: secondaryCategoryId || null,
                    primary_category_name: primaryCategory?.name || null,
                    secondary_category_name: secondaryCategory?.name || null,
                    category_path: [primaryCategory?.name, secondaryCategory?.name].filter(Boolean).join(' / ') || null,
                },
            })
            const refreshed = await researchRebuildService.getStrategyRun(runDetail!.run.id)
            setRunDetail(refreshed)
            setSuccess('Outcome sent to Content Studio.')
        } catch (persistError) {
            setError(persistError instanceof Error ? persistError.message : 'Failed to send outcome to Content Studio.')
        } finally {
            setMutatingOutcomeAction(null)
        }
    }

    const handleDismissRun = async () => {
        if (!runDetail) return
        setMutatingOutcomeAction('dismiss')
        setError(null)
        setSuccess(null)
        try {
            const refreshed = await researchRebuildService.dismissStrategyRun(runDetail.run.id, {
                reason: 'not_pursuing',
            })
            setRunDetail(refreshed)
            await loadTopicsAndRuns()
            await loadFeasibleKeywords()
            setSuccess('Marked as not pursuing. You can rerun the strategy later if you want to revisit it.')
        } catch (dismissError) {
            setError(dismissError instanceof Error ? dismissError.message : 'Failed to mark this opportunity as not pursuing.')
        } finally {
            setMutatingOutcomeAction(null)
        }
    }

    const handleOpenKeywordOpportunity = async (item: ResearchFeasibleKeywordOpportunity) => {
        setError(null)
        setSuccess(null)
        if (runDetail?.run.id === item.run_id) {
            setSelectedClusterId(item.id)
            return
        }
        setIsLoading(true)
        setLoadingLabel('Opening the source run for this keyword…')
        try {
            await loadRunDetail(item.run_id)
            setSelectedClusterId(item.id)
            applyScopeSelection({
                topicId: item.topic_id,
                primaryCategoryId: String(item.primary_category_id || ''),
                secondaryCategoryId: String(item.secondary_category_id || ''),
            })
            syncScopeParams(item.topic_id, item.run_id)
        } catch (openError) {
            setError(openError instanceof Error ? openError.message : 'Failed to open the source run for this keyword.')
        } finally {
            setIsLoading(false)
        }
    }

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

                {success ? (
                    <div className="rounded-2xl border border-emerald-500/30 bg-emerald-950/20 px-4 py-3 text-sm text-emerald-200">
                        {success}
                    </div>
                ) : null}

                {isLoading ? (
                    <div className="rounded-[24px] border border-sky-400/30 bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.18),transparent_45%),#08111d] px-5 py-4 shadow-[0_0_40px_rgba(56,189,248,0.12)]">
                        <div className="flex items-center gap-4">
                            <div className="relative flex h-12 w-12 items-center justify-center">
                                <div className="absolute h-12 w-12 rounded-full border border-sky-400/30 animate-ping" />
                                <div className="absolute h-8 w-8 rounded-full border-2 border-sky-300/80 border-t-transparent animate-spin" />
                                <div className="h-2.5 w-2.5 rounded-full bg-sky-300" />
                            </div>
                            <div>
                                <div className="text-xs uppercase tracking-[0.32em] text-sky-300/80">Research engine active</div>
                                <div className="mt-1 text-base font-medium text-white">{loadingLabel}</div>
                                <div className="mt-1 text-sm text-slate-300">This can take a moment because the app is generating seed queries, screening SERPs, mining competitor URLs, and saving the evidence.</div>
                            </div>
                        </div>
                    </div>
                ) : null}

                <section className="rounded-[28px] border border-white/10 bg-[#0f1320] p-6">
                    <div className="mb-5 flex items-center justify-between">
                        <div>
                            <div className="text-xs uppercase tracking-[0.35em] text-sky-300/70">Step 1</div>
                            <h2 className="mt-2 text-2xl font-semibold">Select category and define the topic</h2>
                        </div>
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
                            <p className="mt-1 whitespace-pre-wrap text-slate-400">{secondaryCategory?.description || 'Optional: select a sub-category to narrow the seed queries and competitor mining.'}</p>
                        </div>
                    </div>

                    <div className="mt-5 grid gap-4 lg:grid-cols-[1fr_auto]">
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
                    </div>

                    <div className="mt-5 rounded-2xl border border-slate-700 bg-[#121826] px-4 py-4">
                        <div className="flex items-center justify-between gap-4">
                            <div>
                                <div className="text-xs uppercase tracking-[0.28em] text-slate-400">Saved topics</div>
                                <div className="mt-2 text-sm text-slate-300">Choose one topic at a time, then run the seed-query, SERP, and competitor-keyword workflow.</div>
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
                                            applyScopeSelection({
                                                topicId: topic.id,
                                                primaryCategoryId: String(latestRun?.primary_category_id || topic.primary_category_id || ''),
                                                secondaryCategoryId: String(latestRun?.secondary_category_id || topic.secondary_category_id || ''),
                                            })
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
                        {selectedTopic ? (
                            <div className="mt-4 flex flex-col gap-3 rounded-2xl border border-sky-400/30 bg-sky-500/10 px-4 py-4 lg:flex-row lg:items-center lg:justify-between">
                                <div>
                                    <div className="text-xs uppercase tracking-[0.28em] text-sky-300/80">Selected topic</div>
                                    <div className="mt-2 text-base font-medium text-white">{selectedTopic.job_text}</div>
                                    <div className="mt-1 text-sm text-slate-300">
                                        {latestRunByTopic.get(selectedTopic.id)
                                            ? `Latest route: ${routeLabel(latestRunByTopic.get(selectedTopic.id)?.winning_route)}`
                                            : 'No strategy run yet for this topic.'}
                                    </div>
                                </div>
                                <div className="flex flex-wrap gap-3">
                                    <button
                                        type="button"
                                        onClick={() => handleRemoveTopic(selectedTopic.id)}
                                        disabled={isLoading || isRemovingTopic === selectedTopic.id}
                                        className="rounded-2xl border border-slate-600 bg-slate-900 px-5 py-3 text-sm font-semibold text-slate-100 transition hover:border-rose-400 hover:text-white disabled:cursor-not-allowed disabled:opacity-50"
                                    >
                                        {isRemovingTopic === selectedTopic.id ? 'Deleting seed…' : 'Delete Seed'}
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => handleRunTopic(selectedTopic.id)}
                                        disabled={!projectId || isLoading}
                                        className="inline-flex items-center justify-center gap-2 rounded-2xl bg-sky-400 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-sky-300 disabled:cursor-not-allowed disabled:opacity-50"
                                    >
                                        {runningTopicId === selectedTopic.id ? (
                                            <>
                                                <span className="inline-block h-4 w-4 rounded-full border-2 border-slate-950/30 border-t-slate-950 animate-spin" />
                                                Screening seed queries…
                                            </>
                                        ) : (
                                            'Run Strategy For This Topic'
                                        )}
                                    </button>
                                </div>
                            </div>
                        ) : null}
                    </div>
                </section>

                <section className="rounded-[28px] border border-white/10 bg-[#0f1320] p-6">
                    <div className="mb-5 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                        <div>
                            <div className="text-xs uppercase tracking-[0.35em] text-sky-300/70">Step 2</div>
                            <h2 className="mt-2 text-2xl font-semibold">Review seed queries and SERP screening</h2>
                            <p className="mt-2 text-sm text-slate-300">The engine now starts from simple search seeds, runs SERPs directly, picks the best competitor URLs, and lets competitor-ranked keywords drive article discovery.</p>
                        </div>
                        {runDetail ? (
                            <div className="flex flex-wrap gap-2">
                                <button type="button" onClick={() => handleRerunStage('trends')} className="rounded-full border border-slate-600 bg-slate-900 px-4 py-2 text-sm text-slate-100 hover:border-sky-400">Rerun trends</button>
                                <button type="button" onClick={() => handleRerunStage('serp')} className="rounded-full border border-slate-600 bg-slate-900 px-4 py-2 text-sm text-slate-100 hover:border-sky-400">Rerun SERP</button>
                                <button type="button" onClick={() => handleRerunStage('competitor_mining')} className="rounded-full border border-slate-600 bg-slate-900 px-4 py-2 text-sm text-slate-100 hover:border-sky-400">Rerun mining</button>
                            </div>
                        ) : null}
                    </div>

                    <div className="mb-5 rounded-2xl border border-slate-700 bg-[#111827] px-4 py-4">
                        <div className="text-xs uppercase tracking-[0.28em] text-slate-400">How Step 2 works</div>
                        <div className="mt-3 grid gap-3 text-sm text-slate-300 lg:grid-cols-3">
                            <div>
                                <div className="font-medium text-white">1. The app generates seed queries</div>
                                <p className="mt-1">From your saved topic, the app creates a handful of simple Google-style search seeds. The topic is only the starting point, not the final keyword target.</p>
                            </div>
                            <div>
                                <div className="font-medium text-white">2. Each seed goes straight to SERP</div>
                                <p className="mt-1">Each seed is tested directly in Google-style SERPs. The goal is to surface strong competitor article URLs first, then mine those URLs for ranked keywords.</p>
                            </div>
                            <div>
                                <div className="font-medium text-white">3. Mining-first screening</div>
                                <p className="mt-1"><span className="text-emerald-300">Most seeds</span> now continue into competitor mining if the SERP returns usable article or comparison pages. <span className="text-rose-300">Killed seeds</span> only appear when the SERP looks overwhelmingly unusable, such as service-, ecommerce-, or pure product-dominant results.</p>
                            </div>
                        </div>
                    </div>

                    {!runDetail ? (
                        <div className="rounded-2xl border border-dashed border-slate-700 bg-[#111725] px-5 py-10 text-sm text-slate-400">
                            Save a topic and run the strategy to generate seed queries, SERP results, recurring competitor domains, and keyword opportunities.
                        </div>
                    ) : (
                        <div className="grid gap-4">
                            {runDetail.bets.map((bet: ResearchTopicBet) => {
                                const betClusters = clustersByBet[bet.id] || []
                                const betCompetitors = competitorPagesByBet[bet.id] || []
                                const reasonCodes = bet.reason_codes || []
                                const probes = runDetail.probe_queries.filter((probe) => probe.bet_id === bet.id)
                                return (
                                    <article key={bet.id} className={`rounded-3xl border p-5 ${bet.status === 'survived' ? 'border-emerald-500/30 bg-[#101a18]' : 'border-slate-700 bg-[#121826]'}`}>
                                        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                                            <div className="max-w-3xl">
                                                <div className="text-xs uppercase tracking-[0.3em] text-slate-400">{bet.status === 'survived' ? 'Survived' : 'Killed'} seed</div>
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

                                        <div className="mt-5 grid gap-4 xl:grid-cols-[0.9fr_0.8fr_1.05fr]">
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
                                                <div className="text-xs uppercase tracking-[0.28em] text-slate-400">Competitor domains</div>
                                                {betCompetitors.length ? (
                                                    <div className="mt-3 grid gap-3">
                                                        {betCompetitors.map((page) => {
                                                            const metadata = (page.page_metadata || {}) as Record<string, unknown>
                                                            const analysisTarget = String(metadata.analysis_target || page.domain || page.url || '')
                                                            const domainHits = Number(metadata.domain_hits || 0)
                                                            return (
                                                                <div key={page.id} className="rounded-2xl border border-slate-700 bg-[#141a29] p-4">
                                                                    <div className="flex items-start justify-between gap-4">
                                                                        <div>
                                                                            <div className="font-medium text-white">{analysisTarget}</div>
                                                                            <div className="mt-1 text-sm text-slate-300">{page.title || page.url}</div>
                                                                        </div>
                                                                        <div className="text-xs text-slate-400">
                                                                            {domainHits > 1 ? `${domainHits} SERP appearances` : '1 SERP appearance'}
                                                                        </div>
                                                                    </div>
                                                                    {page.url ? (
                                                                        <a href={page.url} target="_blank" rel="noreferrer" className="mt-3 inline-flex max-w-full truncate rounded-full border border-slate-600 px-3 py-1 text-xs text-slate-200 hover:border-sky-400">
                                                                            {page.url}
                                                                        </a>
                                                                    ) : null}
                                                                </div>
                                                            )
                                                        })}
                                                    </div>
                                                ) : (
                                                    <div className="mt-3 text-sm text-slate-400">
                                                        {bet.status === 'survived' ? 'The seed survived, but no beatable recurring competitor domain was selected yet.' : 'This seed was rejected before mining because the SERP looked overwhelmingly unusable.'}
                                                    </div>
                                                )}
                                            </div>

                                            <div className="rounded-2xl border border-slate-700 bg-[#0e1421] p-4">
                                                <div className="text-xs uppercase tracking-[0.28em] text-slate-400">Keyword opportunities</div>
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
                                                                        <div className="font-medium text-white">{cluster.primary_keyword_candidate || cluster.cluster_name || 'No primary keyword yet'}</div>
                                                                        <div className="mt-1 text-sm text-slate-300">
                                                                            {String((cluster.cluster_metadata as Record<string, unknown> | undefined)?.source_domain || 'Unknown competitor domain')}
                                                                        </div>
                                                                    </div>
                                                                    <div className={`text-sm font-medium ${toneClass(cluster.opportunity_score)}`}>
                                                                        {Math.round(Number(cluster.opportunity_score || 0) * 100)}%
                                                                    </div>
                                                                </div>
                                                                <div className="mt-3 grid gap-2 text-xs text-slate-400 sm:grid-cols-2">
                                                                    <div>Volume: {String((cluster.cluster_metadata as Record<string, unknown> | undefined)?.search_volume || 'n/a')}</div>
                                                                    <div>KD: {String((cluster.cluster_metadata as Record<string, unknown> | undefined)?.keyword_difficulty || 'n/a')}</div>
                                                                    <div>Intent: {String((cluster.cluster_metadata as Record<string, unknown> | undefined)?.intent || 'n/a')}</div>
                                                                    <div>Competitor rank: {String((cluster.cluster_metadata as Record<string, unknown> | undefined)?.median_rank || cluster.median_rank || 'n/a')}</div>
                                                                </div>
                                                                {cluster.supporting_competitor_urls_json?.length ? (
                                                                    <div className="mt-3 flex flex-wrap gap-2">
                                                                        {cluster.supporting_competitor_urls_json.slice(0, 2).map((url) => (
                                                                            <a key={url} href={url} target="_blank" rel="noreferrer" className="truncate rounded-full border border-slate-600 px-3 py-1 text-xs text-slate-200 hover:border-sky-400">
                                                                                {url}
                                                                            </a>
                                                                        ))}
                                                                    </div>
                                                                ) : null}
                                                            </button>
                                                        ))}
                                                    </div>
                                                ) : (
                                                    <div className="mt-3 text-sm text-slate-400">
                                                        {bet.status === 'survived' ? 'This seed reached competitor mining, but no feasible keyword opportunity was surfaced yet.' : 'This seed was rejected before keyword mining because the SERP looked overwhelmingly unusable.'}
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
                    <div className="mb-5 flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
                        <div>
                            <div className="text-xs uppercase tracking-[0.35em] text-sky-300/70">Keyword Bank</div>
                            <h2 className="mt-2 text-2xl font-semibold">Feasible keywords found</h2>
                            <p className="mt-2 text-sm text-slate-300">
                                Keep every realistic keyword opportunity from this category scope visible. Deleting a seed clears the clutter in Step 2, but the collected keyword evidence stays here for later use.
                            </p>
                        </div>
                        <div className="text-sm text-slate-400">
                            {feasibleKeywords.length} unused keyword{feasibleKeywords.length === 1 ? '' : 's'}
                        </div>
                    </div>

                    {!projectId ? (
                        <div className="rounded-2xl border border-dashed border-slate-700 bg-[#111725] px-5 py-8 text-sm text-slate-400">
                            Select a project to load the feasible keyword repository.
                        </div>
                    ) : feasibleKeywords.length === 0 ? (
                        <div className="rounded-2xl border border-dashed border-slate-700 bg-[#111725] px-5 py-8 text-sm text-slate-400">
                            No unused feasible keyword opportunities were found for this scope yet.
                        </div>
                    ) : (
                        <div className="overflow-hidden rounded-2xl border border-slate-700 bg-[#101726]">
                            <div className="hidden grid-cols-[minmax(0,2.2fr)_100px_80px_110px_120px_minmax(0,1.5fr)_140px] gap-3 border-b border-slate-700 px-4 py-3 text-[11px] uppercase tracking-[0.26em] text-slate-400 lg:grid">
                                <div>Keyword</div>
                                <div>Volume</div>
                                <div>KD</div>
                                <div>Intent</div>
                                <div>Competitor</div>
                                <div>Source URL</div>
                                <div className="text-right">Action</div>
                            </div>
                            <div className="divide-y divide-slate-800">
                                {feasibleKeywords.map((item) => {
                                    const sourceUrl = String(item.source_url || item.supporting_competitor_urls?.[0] || '')
                                    const sourceDomain = String(item.source_domain || 'Unknown')
                                    const isSelected = selectedClusterId === item.id && runDetail?.run.id === item.run_id

                                    return (
                                        <div key={item.id} className={`px-4 py-4 ${isSelected ? 'bg-sky-500/8' : ''}`}>
                                            <div className="grid gap-3 lg:grid-cols-[minmax(0,2.2fr)_100px_80px_110px_120px_minmax(0,1.5fr)_140px] lg:items-start">
                                                <div className="min-w-0">
                                                    <div className="font-medium text-white">{item.keyword || 'Untitled keyword'}</div>
                                                    <div className="mt-1 text-xs text-slate-400">
                                                        {item.topic_text || 'Unknown topic'}
                                                        {' · '}
                                                        Score {Math.round(Number(item.opportunity_score || 0) * 100)}%
                                                        {' · '}
                                                        Competitor rank {String(item.competitor_rank || 'n/a')}
                                                    </div>
                                                </div>
                                                <div className="text-sm text-slate-200">
                                                    <span className="mr-2 text-xs uppercase tracking-[0.2em] text-slate-500 lg:hidden">Volume</span>
                                                    {String(item.search_volume || 'n/a')}
                                                </div>
                                                <div className="text-sm text-slate-200">
                                                    <span className="mr-2 text-xs uppercase tracking-[0.2em] text-slate-500 lg:hidden">KD</span>
                                                    {String(item.keyword_difficulty || 'n/a')}
                                                </div>
                                                <div className="text-sm text-slate-200">
                                                    <span className="mr-2 text-xs uppercase tracking-[0.2em] text-slate-500 lg:hidden">Intent</span>
                                                    {String(item.intent || 'n/a')}
                                                </div>
                                                <div className="text-sm text-slate-200">
                                                    <span className="mr-2 text-xs uppercase tracking-[0.2em] text-slate-500 lg:hidden">Competitor</span>
                                                    {sourceDomain}
                                                </div>
                                                <div className="min-w-0 text-sm">
                                                    <span className="mr-2 text-xs uppercase tracking-[0.2em] text-slate-500 lg:hidden">Source URL</span>
                                                    {sourceUrl ? (
                                                        <a
                                                            href={sourceUrl}
                                                            target="_blank"
                                                            rel="noreferrer"
                                                            className="block truncate text-sky-300 transition hover:text-sky-200"
                                                            title={sourceUrl}
                                                        >
                                                            {sourceUrl}
                                                        </a>
                                                    ) : (
                                                        <span className="text-slate-500">No source URL</span>
                                                    )}
                                                </div>
                                                <div className="flex justify-end lg:justify-end">
                                                    <button
                                                        type="button"
                                                        onClick={() => handleOpenKeywordOpportunity(item)}
                                                        className={`rounded-full border px-3 py-1.5 text-xs font-semibold transition ${isSelected ? 'border-sky-400 bg-sky-500/15 text-white' : 'border-slate-600 bg-slate-900 text-slate-100 hover:border-sky-400'}`}
                                                    >
                                                        {isSelected ? 'Selected' : runDetail?.run.id === item.run_id ? 'Use this keyword' : 'Open source run'}
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    )
                                })}
                            </div>
                        </div>
                    )}
                </section>

                <section className="rounded-[28px] border border-white/10 bg-[#0f1320] p-6">
                    <div className="mb-5 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                        <div>
                            <div className="text-xs uppercase tracking-[0.35em] text-sky-300/70">Step 3</div>
                            <h2 className="mt-2 text-2xl font-semibold">
                                {isSoftwareRoute
                                    ? 'Finalize the winning software opportunity'
                                    : isEditorialRoute
                                        ? 'Finalize the winning editorial angle'
                                        : 'Choose the winning keyword and define the article'}
                            </h2>
                            <p className="mt-2 text-sm text-slate-300">
                                {isSoftwareRoute
                                    ? 'This run looks more like a software-first opportunity, so Step 3 finalizes the winning bet instead of asking for an article cluster.'
                                    : isEditorialRoute
                                        ? 'This run looks more like an editorial opportunity, so Step 3 finalizes the winning angle directly.'
                                        : 'Pick the strongest individual keyword you can realistically beat, using the competitor domain, page, and metric evidence surfaced above.'}
                            </p>
                        </div>
                        <div className="flex flex-wrap items-center gap-3">
                            <button
                                type="button"
                                onClick={handleDismissRun}
                                disabled={!runDetail || mutatingOutcomeAction !== null}
                                className="rounded-2xl border border-slate-600 bg-slate-900 px-5 py-3 text-sm font-semibold text-slate-100 transition hover:border-rose-400 hover:text-white disabled:cursor-not-allowed disabled:opacity-50"
                            >
                                {mutatingOutcomeAction === 'dismiss' ? 'Marking as not pursuing…' : 'Not pursuing this'}
                            </button>
                            <button
                                type="button"
                                onClick={() => handleSelectCluster()}
                                disabled={!canFinalizeSelection || isLoading || isDismissed}
                                className="rounded-2xl bg-emerald-400 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:opacity-50"
                            >
                                {finalizeLabel}
                            </button>
                        </div>
                    </div>

                    {runDetail ? (
                        <div className="grid gap-5 xl:grid-cols-[0.95fr_1.05fr]">
                            <div className="rounded-3xl border border-slate-700 bg-[#101726] p-5">
                                <div className="text-xs uppercase tracking-[0.28em] text-slate-400">Winning route</div>
                                <div className="mt-2 text-2xl font-semibold text-white">{routeLabel(runDetail.run.winning_route)}</div>
                                <div className="mt-2 text-sm text-slate-300">
                                    Confidence {Math.round(Number(runDetail.run.confidence_score || 0) * 100)}%
                                </div>
                                {isArticleRoute ? (
                                    <div className="mt-5 grid gap-3">
                                        {keywordOpportunities.map((cluster) => (
                                            <div key={cluster.id} className={`rounded-2xl border p-4 ${selectedClusterId === cluster.id ? 'border-sky-400 bg-sky-500/10' : 'border-slate-700 bg-[#141a29]'}`}>
                                                <div className="flex items-start justify-between gap-4">
                                                    <div>
                                                        <div className="font-medium text-white">{cluster.primary_keyword_candidate || cluster.cluster_name}</div>
                                                        <div className="mt-1 text-sm text-slate-300">
                                                            {String(clusterMetadata(cluster).source_domain || 'Unknown competitor domain')}
                                                        </div>
                                                    </div>
                                                    <div className={`text-sm ${toneClass(cluster.opportunity_score)}`}>
                                                        {Math.round(Number(cluster.opportunity_score || 0) * 100)}%
                                                    </div>
                                                </div>
                                                <div className="mt-3 grid gap-2 text-xs text-slate-400">
                                                    <div>SERP weakness: {Math.round(Number(cluster.serp_weakness_score || 0) * 100)}%</div>
                                                    <div>Competitor support: {Math.round(Number(cluster.competitor_support_score || 0) * 100)}%</div>
                                                    <div>Commercial value: {Math.round(Number(cluster.commercial_value_score || 0) * 100)}%</div>
                                                    <div>Volume: {String(clusterMetadata(cluster).search_volume || 'n/a')}</div>
                                                    <div>KD: {String(clusterMetadata(cluster).keyword_difficulty || 'n/a')}</div>
                                                    <div>Intent: {String(clusterMetadata(cluster).intent || 'n/a')}</div>
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
                                ) : selectedBet ? (
                                    <div className="mt-5 rounded-2xl border border-emerald-500/30 bg-[#141a29] p-4">
                                        <div className="text-xs uppercase tracking-[0.28em] text-emerald-300/80">
                                            {isSoftwareRoute ? 'Winning software bet' : 'Winning editorial bet'}
                                        </div>
                                        <div className="mt-2 text-xl font-semibold text-white">{selectedBet.bet_text}</div>
                                        <p className="mt-2 text-sm text-slate-300">{selectedBet.searcher_problem || 'No searcher problem captured.'}</p>
                                        <div className="mt-4 flex flex-wrap gap-2 text-xs text-slate-300">
                                            <span className="rounded-full border border-slate-600 px-3 py-1">{selectedBet.article_format || 'format pending'}</span>
                                            <span className="rounded-full border border-slate-600 px-3 py-1">{selectedBet.commercial_angle || 'commercial angle pending'}</span>
                                            <span className="rounded-full border border-slate-600 px-3 py-1">{selectedBet.route_hint || 'route pending'}</span>
                                        </div>
                                        <div className="mt-4 grid gap-2 text-xs text-slate-400">
                                            <div>Articleability: {Math.round(Number(selectedBet.serp_articleability_score || 0) * 100)}%</div>
                                            <div>SERP weakness: {Math.round(Number(selectedBet.serp_weakness_score || 0) * 100)}%</div>
                                            <div>Article fit: {Math.round(Number(selectedBet.article_fit_score || 0) * 100)}%</div>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="mt-5 rounded-2xl border border-dashed border-slate-700 bg-[#111725] px-4 py-8 text-sm text-slate-400">
                                        No winning selection is available for this route yet.
                                    </div>
                                )}
                            </div>

                            <div className="rounded-3xl border border-slate-700 bg-[#101726] p-5">
                                <div className="text-xs uppercase tracking-[0.28em] text-slate-400">Final output</div>
                                {finalOutcome ? (
                                    <div className="mt-4">
                                        <h3 className="text-2xl font-semibold text-white">{isSoftwareRoute ? softwareName : String(finalOutcome.title || 'Untitled output')}</h3>
                                        {isSoftwareRoute ? (
                                            <>
                                                {isDismissed ? (
                                                    <div className="mt-3 rounded-2xl border border-amber-500/30 bg-amber-950/20 px-4 py-3 text-sm text-amber-100">
                                                        This opportunity is marked as not pursuing right now. You can still review it here or rerun the strategy later.
                                                    </div>
                                                ) : null}
                                                <div className="mt-3 text-base text-slate-200">
                                                    {String(finalOutcome.software_concept || finalOutcome.description || finalOutcome.rationale || '')}
                                                </div>
                                                <div className="mt-4 grid gap-3 text-sm text-slate-300 lg:grid-cols-2">
                                                    <div>
                                                        Target user: <span className="font-medium text-white">{String(finalOutcome.target_user || 'Not specified')}</span>
                                                    </div>
                                                    <div>
                                                        Slug: <span className="font-mono text-slate-100">{String(finalOutcome.slug || '')}</span>
                                                    </div>
                                                </div>
                                                {softwareSearchAngle ? (
                                                    <div className="mt-3 text-sm text-slate-300">
                                                        Search angle: <span className="font-medium text-white">{softwareSearchAngle}</span>
                                                    </div>
                                                ) : null}
                                                <div className="mt-3 text-sm text-slate-300">
                                                    User problem: <span className="text-white">{String(finalOutcome.user_problem || '')}</span>
                                                </div>
                                                <div className="mt-4 flex flex-wrap gap-2">
                                                    {softwareKeywords.map((keyword) => (
                                                        <span key={String(keyword)} className="rounded-full border border-slate-600 px-3 py-1 text-xs text-slate-200">
                                                            {String(keyword)}
                                                        </span>
                                                    ))}
                                                </div>
                                                {!hasDetailedSoftwareSpec ? (
                                                    <div className="mt-5 rounded-2xl border border-dashed border-slate-700 bg-[#111725] px-4 py-4 text-sm text-slate-400">
                                                        This software result is still too thin. Regenerate the software output to get a fuller product workflow, feature set, inputs, outputs, and MVP scope.
                                                    </div>
                                                ) : null}
                                                {softwareCoreWorkflow.length || softwareFeatures.length ? (
                                                    <div className="mt-5 grid gap-4 lg:grid-cols-2">
                                                        {softwareCoreWorkflow.length ? (
                                                            <div className="rounded-2xl border border-slate-700 bg-[#0d1320] p-4">
                                                                <div className="text-xs uppercase tracking-[0.28em] text-slate-400">Core workflow</div>
                                                                <ul className="mt-3 grid gap-2 text-sm text-slate-200">
                                                                    {softwareCoreWorkflow.map((item) => (
                                                                        <li key={String(item)} className="rounded-xl border border-slate-700 bg-[#131a29] px-3 py-2">
                                                                            {String(item)}
                                                                        </li>
                                                                    ))}
                                                                </ul>
                                                            </div>
                                                        ) : null}
                                                        {softwareFeatures.length ? (
                                                            <div className="rounded-2xl border border-slate-700 bg-[#0d1320] p-4">
                                                                <div className="text-xs uppercase tracking-[0.28em] text-slate-400">Key features</div>
                                                                <ul className="mt-3 grid gap-2 text-sm text-slate-200">
                                                                    {softwareFeatures.map((item) => (
                                                                        <li key={String(item)} className="rounded-xl border border-slate-700 bg-[#131a29] px-3 py-2">
                                                                            {String(item)}
                                                                        </li>
                                                                    ))}
                                                                </ul>
                                                            </div>
                                                        ) : null}
                                                    </div>
                                                ) : null}
                                                {softwareInputs.length || softwareOutputs.length ? (
                                                    <div className="mt-4 grid gap-4 lg:grid-cols-2">
                                                        {softwareInputs.length ? (
                                                            <div className="rounded-2xl border border-slate-700 bg-[#0d1320] p-4">
                                                                <div className="text-xs uppercase tracking-[0.28em] text-slate-400">Inputs</div>
                                                                <ul className="mt-3 grid gap-2 text-sm text-slate-200">
                                                                    {softwareInputs.map((item) => (
                                                                        <li key={String(item)} className="rounded-xl border border-slate-700 bg-[#131a29] px-3 py-2">
                                                                            {String(item)}
                                                                        </li>
                                                                    ))}
                                                                </ul>
                                                            </div>
                                                        ) : null}
                                                        {softwareOutputs.length ? (
                                                            <div className="rounded-2xl border border-slate-700 bg-[#0d1320] p-4">
                                                                <div className="text-xs uppercase tracking-[0.28em] text-slate-400">Outputs</div>
                                                                <ul className="mt-3 grid gap-2 text-sm text-slate-200">
                                                                    {softwareOutputs.map((item) => (
                                                                        <li key={String(item)} className="rounded-xl border border-slate-700 bg-[#131a29] px-3 py-2">
                                                                            {String(item)}
                                                                        </li>
                                                                    ))}
                                                                </ul>
                                                            </div>
                                                        ) : null}
                                                    </div>
                                                ) : null}
                                                {softwareMvpScope.length ? (
                                                    <div className="mt-4 rounded-2xl border border-slate-700 bg-[#0d1320] p-4">
                                                        <div className="text-xs uppercase tracking-[0.28em] text-slate-400">MVP scope</div>
                                                        <ul className="mt-3 grid gap-2 text-sm text-slate-200">
                                                            {softwareMvpScope.map((item) => (
                                                                <li key={String(item)} className="rounded-xl border border-slate-700 bg-[#131a29] px-3 py-2">
                                                                    {String(item)}
                                                                </li>
                                                            ))}
                                                        </ul>
                                                    </div>
                                                ) : null}
                                                <div className="mt-4 rounded-2xl border border-slate-700 bg-[#0d1320] p-4 text-sm text-slate-300">
                                                    <div className="text-xs uppercase tracking-[0.28em] text-slate-400">Build notes</div>
                                                    <div className="mt-2">{String(finalOutcome.build_notes || finalOutcome.rationale || '')}</div>
                                                </div>
                                            </>
                                        ) : (
                                            <>
                                                <div className="mt-2 text-sm text-slate-300">
                                                    Primary keyword: <span className="font-medium text-white">{String(finalOutcome.primary_keyword || '')}</span>
                                                </div>
                                                <div className="mt-2 text-sm text-slate-300">
                                                    Competitor domain: <span className="font-medium text-white">{String(finalOutcome.source_competitor_domain || 'Not captured')}</span>
                                                </div>
                                                {String(finalOutcome.source_competitor_url || '').trim() ? (
                                                    <div className="mt-2 text-sm text-slate-300">
                                                        Competitor URL:{' '}
                                                        <a
                                                            href={String(finalOutcome.source_competitor_url)}
                                                            target="_blank"
                                                            rel="noreferrer"
                                                            className="font-medium text-sky-300 underline-offset-2 hover:underline"
                                                        >
                                                            {String(finalOutcome.source_competitor_url)}
                                                        </a>
                                                    </div>
                                                ) : null}
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
                                                {Array.isArray(finalOutcome.competitor_urls_used) && finalOutcome.competitor_urls_used.length ? (
                                                    <div className="mt-4 rounded-2xl border border-slate-700 bg-[#0d1320] p-4">
                                                        <div className="text-xs uppercase tracking-[0.28em] text-slate-400">Competitor URLs used</div>
                                                        <div className="mt-3 flex flex-wrap gap-2">
                                                            {finalOutcome.competitor_urls_used.map((url) => (
                                                                <a
                                                                    key={String(url)}
                                                                    href={String(url)}
                                                                    target="_blank"
                                                                    rel="noreferrer"
                                                                    className="truncate rounded-full border border-slate-600 px-3 py-1 text-xs text-slate-200 hover:border-sky-400"
                                                                >
                                                                    {String(url)}
                                                                </a>
                                                            ))}
                                                        </div>
                                                    </div>
                                                ) : null}
                                            </>
                                        )}

                                        <div className="mt-5 flex flex-wrap gap-3">
                                            {isSoftwareRoute ? (
                                                <>
                                                    <button
                                                        type="button"
                                                        onClick={handleReleaseSoftwareIdea}
                                                        disabled={!generatedOutcomeId || mutatingOutcomeAction !== null || isDismissed}
                                                        className="rounded-2xl bg-emerald-400 px-4 py-3 text-sm font-semibold text-slate-950 transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:opacity-50"
                                                    >
                                                        {mutatingOutcomeAction === 'release' ? 'Sending to Software Ideas…' : 'Send to Software Ideas'}
                                                    </button>
                                                    <button
                                                        type="button"
                                                        onClick={handleSendToContentStudio}
                                                        disabled={!generatedOutcomeId || mutatingOutcomeAction !== null || isDismissed}
                                                        className="rounded-2xl border border-slate-600 bg-slate-900 px-4 py-3 text-sm font-semibold text-slate-100 transition hover:border-sky-400 disabled:cursor-not-allowed disabled:opacity-50"
                                                    >
                                                        {mutatingOutcomeAction === 'persist' ? 'Sending to Content Studio…' : 'Send to Content Studio'}
                                                    </button>
                                                </>
                                            ) : (
                                                <button
                                                    type="button"
                                                    onClick={handleSendToContentStudio}
                                                    disabled={!generatedOutcomeId || mutatingOutcomeAction !== null}
                                                    className="rounded-2xl bg-emerald-400 px-4 py-3 text-sm font-semibold text-slate-950 transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:opacity-50"
                                                >
                                                    {mutatingOutcomeAction === 'persist' ? 'Sending to Content Studio…' : 'Send to Content Studio'}
                                                </button>
                                            )}
                                        </div>
                                    </div>
                                ) : (
                                    <div className="mt-4 rounded-2xl border border-dashed border-slate-700 bg-[#111725] px-4 py-8 text-sm text-slate-400">
                                        {isSoftwareRoute
                                            ? 'This run is software-first. Generate the software output to create the title, slug, workflow framing, and rationale from the winning bet.'
                                                : isEditorialRoute
                                                    ? 'This run is editorial-first. Generate the editorial output to create the title, angle, and rationale from the winning bet.'
                                                    : selectedCluster
                                                        ? (
                                                            <div className="space-y-3">
                                                                <div>Lock the selected keyword opportunity to generate the title, slug, keyword map, and outline.</div>
                                                                <div className="rounded-2xl border border-sky-400/20 bg-sky-500/5 px-4 py-3 text-slate-300">
                                                                    <div className="text-xs uppercase tracking-[0.24em] text-sky-300/80">Selected keyword</div>
                                                                    <div className="mt-2 font-medium text-white">
                                                                        {selectedCluster.primary_keyword_candidate || selectedCluster.cluster_name}
                                                                    </div>
                                                                    <div className="mt-1 text-sm">
                                                                        Competitor domain: <span className="text-white">{String(selectedClusterMetadata.source_domain || 'Unknown')}</span>
                                                                    </div>
                                                                    {String(selectedClusterMetadata.source_url || selectedCluster.supporting_competitor_urls_json?.[0] || '').trim() ? (
                                                                        <a
                                                                            href={String(selectedClusterMetadata.source_url || selectedCluster.supporting_competitor_urls_json?.[0] || '')}
                                                                            target="_blank"
                                                                            rel="noreferrer"
                                                                            className="mt-2 inline-flex max-w-full truncate text-sky-300 hover:text-sky-200"
                                                                        >
                                                                            {String(selectedClusterMetadata.source_url || selectedCluster.supporting_competitor_urls_json?.[0] || '')}
                                                                        </a>
                                                                    ) : null}
                                                                </div>
                                                            </div>
                                                        )
                                                    : 'No feasible keyword opportunity is available yet for this run.'}
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
                                                        applyScopeSelection({
                                                            topicId: topic.id,
                                                            primaryCategoryId: String(latestRun?.primary_category_id || topic.primary_category_id || ''),
                                                            secondaryCategoryId: String(latestRun?.secondary_category_id || topic.secondary_category_id || ''),
                                                        })
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
