import * as React from 'react'
import { 
    Loader2, Rocket, Wrench, CheckCircle2, Sparkles, ChevronDown, Search,
    ChevronUp, XCircle, RefreshCw, Ban, Copy, FileText, 
    Gauge, Target, Layers, Globe, ExternalLink, Info, Filter, ArrowRight,
    ChevronLeft, ChevronRight
} from 'lucide-react'
import { cn } from '@/lib/utils'

import { useAuth } from '@/context/auth-context'
import { useProject } from '@/context/project-context'
import { supabase } from '@/lib/supabase'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { researchRebuildService } from '@/services/research-rebuild.service'
import { useLocation, useNavigate, useSearchParams } from 'react-router-dom'
import type {
    ResearchRebuildInternalLinkCandidate,
    ResearchRebuildDataforseoSearch,
    ResearchRebuildJob,
    ResearchRebuildValidationRun,
    ResearchRebuildWorkflowJobResult,
    ResearchRebuildWorkflowRunSummary,
} from '@/types/research-rebuild'

type ProjectCategory = {
    id: string
    name: string
    level: number
    description?: string | null
    parent_category_id: string | null
}

type ResearchRebuildMode = 'jobs' | 'opportunities'

type ResearchRebuildProps = {
    mode?: ResearchRebuildMode
}

const WORKFLOW_BATCH_SIZE = 4
const DATAFORSEO_SEARCH_TYPES = [
    { value: 'expansion_funnel', label: 'Expansion Funnel (Data Expansion)' },
    { value: 'keyword_suggestions', label: 'Keyword Suggestions' },
    { value: 'related_keywords', label: 'Related Keywords' },
    { value: 'keyword_overview', label: 'Keyword Overview' },
    { value: 'serp', label: 'SERP Snapshot' },
] as const

export function ResearchRebuild({ mode = 'jobs' }: ResearchRebuildProps) {
    const { user } = useAuth()
    const { activeProject, projects, setActiveProject } = useProject()
    const navigate = useNavigate()
    const location = useLocation()
    const [searchParams] = useSearchParams()
    const isJobsPage = mode === 'jobs'
    const isOpportunitiesPage = mode === 'opportunities'
    const [jobs, setJobs] = React.useState<ResearchRebuildJob[]>([])
    const [workflowResults, setWorkflowResults] = React.useState<ResearchRebuildWorkflowJobResult[]>([])
    const [projectCategories, setProjectCategories] = React.useState<ProjectCategory[]>([])
    const [primaryCategoryId, setPrimaryCategoryId] = React.useState('')
    const [secondaryCategoryId, setSecondaryCategoryId] = React.useState('')
    const [workflowPage, setWorkflowPage] = React.useState(0)
    const [workflowPageSize, setWorkflowPageSize] = React.useState(10)
    const [workflowTotalJobs, setWorkflowTotalJobs] = React.useState(0)
    const [workflowRuns, setWorkflowRuns] = React.useState<ResearchRebuildWorkflowRunSummary[]>([])
    const [routeFilter, setRouteFilter] = React.useState('all')
    const [candidateSearch, setCandidateSearch] = React.useState('')
    const [manualJobText, setManualJobText] = React.useState('')
    const [articleTitleDraft, setArticleTitleDraft] = React.useState('')
    const [articlePrimaryKeyword, setArticlePrimaryKeyword] = React.useState('')
    const [articleSecondaryKeywordsText, setArticleSecondaryKeywordsText] = React.useState('')
    const [selectedLookupJobId, setSelectedLookupJobId] = React.useState('')
    const [lookupSearchType, setLookupSearchType] = React.useState<'related_keywords' | 'keyword_suggestions' | 'expansion_funnel' | 'keyword_overview' | 'serp'>('expansion_funnel')
    const [lookupKeywordsText, setLookupKeywordsText] = React.useState('')
    const [runningLookup, setRunningLookup] = React.useState(false)
    const [lookupProgressText, setLookupProgressText] = React.useState<string | null>(null)
    const [savingCandidateKey, setSavingCandidateKey] = React.useState<string | null>(null)
    const [dataforseoSearches, setDataforseoSearches] = React.useState<ResearchRebuildDataforseoSearch[]>([])
    const [activeSearchRecord, setActiveSearchRecord] = React.useState<ResearchRebuildDataforseoSearch | null>(null)
    const [focusArea, setFocusArea] = React.useState('')
    const [avoidGuidance, setAvoidGuidance] = React.useState('')
    const [startFreshBatch, setStartFreshBatch] = React.useState(true)
    const [loadingJobs, setLoadingJobs] = React.useState(false)
    const [loadingWorkflowArtifacts, setLoadingWorkflowArtifacts] = React.useState(false)
    const [generatingJobs, setGeneratingJobs] = React.useState(false)
    const [creatingManualJob, setCreatingManualJob] = React.useState(false)
    const [runningWorkflow, setRunningWorkflow] = React.useState(false)
    const [showCategoryDescriptions, setShowCategoryDescriptions] = React.useState(false)
    const [showSavedTopicsModal, setShowSavedTopicsModal] = React.useState(false)
    const [showEasyWinsModal, setShowEasyWinsModal] = React.useState(false)
    const [showAllMetricsModal, setShowAllMetricsModal] = React.useState(false)
    const [mutatingOutcomeIds, setMutatingOutcomeIds] = React.useState<Set<string>>(new Set())
    const [expandedCandidateIds, setExpandedCandidateIds] = React.useState<Set<string>>(new Set())
    const [archivingJobIds, setArchivingJobIds] = React.useState<Set<string>>(new Set())
    const [rejectingCandidateIds, setRejectingCandidateIds] = React.useState<Set<string>>(new Set())
    const [refreshingValidationIds, setRefreshingValidationIds] = React.useState<Set<string>>(new Set())
    const [refreshingAllStale, setRefreshingAllStale] = React.useState(false)
    const [error, setError] = React.useState<string | null>(null)
    const [success, setSuccess] = React.useState<string | null>(null)

    const incomingProjectId = searchParams.get('project_id') || ''
    const incomingPrimaryCategoryId = searchParams.get('primary_category_id') || ''
    const incomingSecondaryCategoryId = searchParams.get('secondary_category_id') || ''
    const incomingWorkflowRunId = searchParams.get('workflow_run_id') || ''
    const incomingBatchId = searchParams.get('batch_id') || ''
    const [workflowRunFilter, setWorkflowRunFilter] = React.useState(incomingWorkflowRunId || 'all')
    const [activeBatchId, setActiveBatchId] = React.useState(incomingBatchId)

    const primaryCategories = React.useMemo(
        () => projectCategories.filter((category) => category.level === 1),
        [projectCategories],
    )

    const secondaryCategories = React.useMemo(
        () => projectCategories.filter((category) => category.parent_category_id === primaryCategoryId),
        [projectCategories, primaryCategoryId],
    )

    const activeTopics = React.useMemo(
        () => jobs.filter((job) => job.status !== 'archived' && job.status !== 'rejected'),
        [jobs],
    )

    React.useEffect(() => {
        if (!selectedLookupJobId && jobs.length > 0) {
            setSelectedLookupJobId(jobs[0].id)
        }
    }, [jobs, selectedLookupJobId])

    const filteredWorkflowResults = React.useMemo(() => workflowResults, [workflowResults])

    const staleValidationTargets = React.useMemo(
        () =>
            filteredWorkflowResults.flatMap((jobResult) =>
                jobResult.candidates
                    .filter((candidateResult) => candidateResult.validation_run.freshness_state !== 'fresh')
                    .map((candidateResult) => ({
                        candidateId: candidateResult.candidate.id,
                        validationRunId: candidateResult.validation_run.id,
                    })),
            ),
        [filteredWorkflowResults],
    )

    const activeSearchPreviewItems = React.useMemo(() => {
        const topItems = activeSearchRecord?.result_summary_json?.top_items
        return Array.isArray(topItems) ? topItems as Array<Record<string, unknown>> : []
    }, [activeSearchRecord])

    const buildScopedPath = React.useCallback((targetMode: ResearchRebuildMode, runId?: string | null) => {
        const params = new URLSearchParams()
        if (activeProject?.id) {
            params.set('project_id', activeProject.id)
        }
        if (primaryCategoryId) {
            params.set('primary_category_id', primaryCategoryId)
        }
        if (secondaryCategoryId) {
            params.set('secondary_category_id', secondaryCategoryId)
        }
        const effectiveRunId = runId ?? (workflowRunFilter !== 'all' ? workflowRunFilter : '')
        if (effectiveRunId) {
            params.set('workflow_run_id', effectiveRunId)
        }
        if (activeBatchId) {
            params.set('batch_id', activeBatchId)
        }
        const query = params.toString()
        return `/research-rebuild/${targetMode}${query ? `?${query}` : ''}`
    }, [activeBatchId, activeProject?.id, primaryCategoryId, secondaryCategoryId, workflowRunFilter])

    const currentViewUrl = React.useMemo(() => {
        const path = buildScopedPath(mode)
        if (typeof window === 'undefined') {
            return path
        }
        return `${window.location.origin}${path}`
    }, [buildScopedPath, mode])

    const focusStorageKey = React.useMemo(
        () =>
            [
                'research-rebuild-brief',
                activeProject?.id || incomingProjectId || 'no-project',
                primaryCategoryId || incomingPrimaryCategoryId || 'no-primary',
                secondaryCategoryId || incomingSecondaryCategoryId || 'no-secondary',
            ].join(':'),
        [
            activeProject?.id,
            incomingPrimaryCategoryId,
            incomingProjectId,
            incomingSecondaryCategoryId,
            primaryCategoryId,
            secondaryCategoryId,
        ],
    )

    React.useEffect(() => {
        if (!user?.id || !activeProject?.id) return

        const loadProjectCategories = async () => {
            const { data, error: categoryError } = await supabase
                .from('project_categories')
                .select('id, name, level, description, parent_category_id')
                .eq('user_id', user.id)
                .eq('project_id', activeProject.id)
                .order('level', { ascending: true })
                .order('sort_order', { ascending: true })
                .order('name', { ascending: true })

            if (categoryError) {
                console.error('Failed to load project categories:', categoryError)
                return
            }

            setProjectCategories((data || []) as ProjectCategory[])
        }

        void loadProjectCategories()
    }, [activeProject?.id, user?.id])

    React.useEffect(() => {
        if (!incomingProjectId || projects.length === 0) return
        if (activeProject?.id === incomingProjectId) return
        const matchedProject = projects.find((project) => project.id === incomingProjectId)
        if (matchedProject) {
            setActiveProject(matchedProject)
        }
    }, [activeProject?.id, incomingProjectId, projects, setActiveProject])

    React.useEffect(() => {
        if (incomingPrimaryCategoryId) {
            setPrimaryCategoryId((current) => current || incomingPrimaryCategoryId)
        }
        if (incomingSecondaryCategoryId) {
            setSecondaryCategoryId((current) => current || incomingSecondaryCategoryId)
        }
        if (incomingWorkflowRunId) {
            setWorkflowRunFilter((current) => (current === 'all' ? incomingWorkflowRunId : current))
        }
        setActiveBatchId(incomingBatchId)
    }, [incomingBatchId, incomingPrimaryCategoryId, incomingSecondaryCategoryId, incomingWorkflowRunId])

    React.useEffect(() => {
        if (typeof window === 'undefined') return
        const raw = window.sessionStorage.getItem(focusStorageKey)
        if (!raw) {
            setFocusArea('')
            setAvoidGuidance('')
            return
        }
        try {
            const parsed = JSON.parse(raw) as { focusArea?: string; avoidGuidance?: string }
            setFocusArea(parsed.focusArea || '')
            setAvoidGuidance(parsed.avoidGuidance || '')
        } catch {
            setFocusArea('')
            setAvoidGuidance('')
        }
    }, [focusStorageKey])

    React.useEffect(() => {
        if (typeof window === 'undefined') return
        window.sessionStorage.setItem(
            focusStorageKey,
            JSON.stringify({
                focusArea,
                avoidGuidance,
            }),
        )
    }, [avoidGuidance, focusArea, focusStorageKey])

    React.useEffect(() => {
        const params = new URLSearchParams()
        if (activeProject?.id) {
            params.set('project_id', activeProject.id)
        }
        if (primaryCategoryId) {
            params.set('primary_category_id', primaryCategoryId)
        }
        if (secondaryCategoryId) {
            params.set('secondary_category_id', secondaryCategoryId)
        }
        if (workflowRunFilter !== 'all') {
            params.set('workflow_run_id', workflowRunFilter)
        }
        if (activeBatchId) {
            params.set('batch_id', activeBatchId)
        }

        const targetPathname = mode === 'jobs' ? '/research-rebuild/jobs' : '/research-rebuild/opportunities'
        const targetSearch = params.toString() ? `?${params.toString()}` : ''
        if (location.pathname === targetPathname && location.search === targetSearch) {
            return
        }

        navigate(
            {
                pathname: targetPathname,
                search: targetSearch,
            },
            { replace: true },
        )
    }, [activeBatchId, activeProject?.id, location.pathname, location.search, mode, navigate, primaryCategoryId, secondaryCategoryId, workflowRunFilter])

    React.useEffect(() => {
        setWorkflowPage(0)
    }, [activeProject?.id, primaryCategoryId, secondaryCategoryId, activeBatchId, workflowRunFilter, routeFilter, candidateSearch])

    React.useEffect(() => {
        if (workflowRunFilter === 'all') return
        if (workflowRuns.some((run) => run.workflow_run_id === workflowRunFilter)) return
        setWorkflowRunFilter('all')
    }, [workflowRunFilter, workflowRuns])

    const refreshPageContext = React.useCallback(async (options?: {
        workflowRunId?: string
        workflowPage?: number
        batchId?: string
    }) => {
        if (!activeProject?.id) return

        const nextWorkflowRunFilter = options?.workflowRunId ?? workflowRunFilter
        const nextWorkflowPage = options?.workflowPage ?? workflowPage
        const nextBatchId = options?.batchId ?? activeBatchId

        try {
            setLoadingJobs(true)
            setLoadingWorkflowArtifacts(true)
            const response = await researchRebuildService.getPageContext({
                project_id: activeProject.id,
                primary_category_id: primaryCategoryId || undefined,
                secondary_category_id: secondaryCategoryId || undefined,
                batch_id: nextBatchId || undefined,
                workflow_run_id: nextWorkflowRunFilter !== 'all' ? nextWorkflowRunFilter : undefined,
                route: routeFilter !== 'all' ? routeFilter : undefined,
                search: candidateSearch.trim() || undefined,
                limit: workflowPageSize,
                offset: nextWorkflowPage * workflowPageSize,
                run_limit: 25,
            })
            setJobs(response.jobs?.items || [])
            setWorkflowRuns(response.workflow?.runs || [])
            setWorkflowResults(response.workflow?.snapshot?.items || [])
            setWorkflowTotalJobs(response.workflow?.snapshot?.total_jobs ?? response.workflow?.snapshot?.count ?? 0)
        } catch (err) {
            console.error('Failed to load rebuild page context:', err)
            setError('Failed to load rebuild page context.')
        } finally {
            setLoadingJobs(false)
            setLoadingWorkflowArtifacts(false)
        }
    }, [activeBatchId, activeProject?.id, primaryCategoryId, secondaryCategoryId, workflowRunFilter, routeFilter, candidateSearch, workflowPage, workflowPageSize])

    React.useEffect(() => {
        void refreshPageContext()
    }, [refreshPageContext])

    const refreshSearchHistory = React.useCallback(async () => {
        if (!activeProject?.id) {
            setDataforseoSearches([])
            return
        }
        try {
            const response = await researchRebuildService.listDataforseoSearches({
                project_id: activeProject.id,
                user_job_id: selectedLookupJobId || undefined,
                limit: 12,
            })
            setDataforseoSearches(response.items || [])
            setActiveSearchRecord((current) => current || response.items?.[0] || null)
        } catch (err) {
            console.error('Failed to load DataForSEO search history:', err)
        }
    }, [activeProject?.id, selectedLookupJobId])

    React.useEffect(() => {
        void refreshSearchHistory()
    }, [refreshSearchHistory])

    React.useEffect(() => {
        if (!showSavedTopicsModal && !showEasyWinsModal) return
        const handleEscape = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                setShowSavedTopicsModal(false)
                setShowEasyWinsModal(false)
            }
        }
        window.addEventListener('keydown', handleEscape)
        return () => window.removeEventListener('keydown', handleEscape)
    }, [showSavedTopicsModal, showEasyWinsModal])

    const primaryCategory = primaryCategories.find((category) => category.id === primaryCategoryId)
    const secondaryCategory = secondaryCategories.find((category) => category.id === secondaryCategoryId)
    const latestWorkflowRunId = workflowRuns[0]?.workflow_run_id || ''
    const jobsPagePath = buildScopedPath('jobs')
    const opportunitiesPagePath = buildScopedPath('opportunities', latestWorkflowRunId || undefined)

    const handleGenerateJobs = async () => {
        if (!activeProject?.id) return
        try {
            setGeneratingJobs(true)
            setError(null)
            setSuccess(null)
            const response = await researchRebuildService.generateJobs({
                project_id: activeProject.id,
                primary_category_id: primaryCategoryId || undefined,
                secondary_category_id: secondaryCategoryId || undefined,
                project_name: activeProject.app_name || activeProject.domain || 'Project',
                website_description: activeProject.site_description || activeProject.websiteDescription || '',
                primary_category_name: primaryCategory?.name,
                primary_category_description: primaryCategory?.description || undefined,
                secondary_category_name: secondaryCategory?.name,
                secondary_category_description: secondaryCategory?.description || undefined,
                focus_area: focusArea.trim() || undefined,
                avoid_guidance: avoidGuidance.trim() || undefined,
                count: 12,
                archive_existing_in_scope: startFreshBatch,
            })
            const nextBatchId = response.batch_id || ''
            setWorkflowResults([])
            setWorkflowRuns([])
            setWorkflowTotalJobs(0)
            setWorkflowRunFilter('all')
            setWorkflowPage(0)
            setActiveBatchId(nextBatchId)
            if (response.count > 0) {
                const clearedPrefix = (response.cleared_count || 0) > 0
                    ? `Removed ${response.cleared_count} earlier active job${response.cleared_count === 1 ? '' : 's'} and `
                    : ''
                setSuccess(`${clearedPrefix}generated ${response.count} fresh job${response.count === 1 ? '' : 's'}.`)
            } else {
                setError(response.message || 'No distinct jobs were found for this focus area.')
            }
            await refreshPageContext({
                batchId: nextBatchId || undefined,
                workflowRunId: 'all',
                workflowPage: 0,
            })
        } catch (err) {
            console.error('Failed to generate jobs:', err)
            setError('Failed to generate jobs.')
        } finally {
            setGeneratingJobs(false)
        }
    }

    const handleCreateManualJob = async () => {
        if (!activeProject?.id || !manualJobText.trim()) return
        try {
            setCreatingManualJob(true)
            setError(null)
            setSuccess(null)
            const createdJob = await researchRebuildService.createJob({
                project_id: activeProject.id,
                primary_category_id: primaryCategoryId || undefined,
                secondary_category_id: secondaryCategoryId || undefined,
                job_text: manualJobText.trim(),
                job_type_hint: 'hybrid',
                website_context_snapshot: {
                    project_name: activeProject.app_name || activeProject.domain || 'Project',
                    website_description: activeProject.site_description || activeProject.websiteDescription || '',
                },
            })
            setManualJobText('')
            setActiveBatchId('')
            setWorkflowRunFilter('all')
            setSelectedLookupJobId(createdJob.id)
            setSuccess('Topic saved to the manual workflow.')
            await refreshPageContext({
                batchId: '',
                workflowRunId: 'all',
                workflowPage: 0,
            })
        } catch (err) {
            console.error('Failed to create manual job:', err)
            setError(err instanceof Error ? err.message : 'Failed to create manual job.')
        } finally {
            setCreatingManualJob(false)
        }
    }

    const handleUseKeywordForArticle = (keyword: string) => {
        const cleaned = keyword.trim()
        if (!cleaned) return
        setArticlePrimaryKeyword(cleaned)
        setArticleSecondaryKeywordsText((current) => {
            const existing = Array.from(
                new Set(
                    current
                        .split(/[\n,]+/)
                        .map((item) => item.trim())
                        .filter(Boolean),
                ),
            )
            if (!existing.includes(cleaned)) {
                existing.push(cleaned)
            }
            return existing.join('\n')
        })
        if (!articleTitleDraft.trim()) {
            setArticleTitleDraft(cleaned)
        }
    }

    const parseLookupKeywords = React.useCallback(() => {
        return Array.from(
            new Set(
                lookupKeywordsText
                    .split(/[\n,]+/)
                    .map((item) => item.trim())
                    .filter(Boolean),
            ),
        )
    }, [lookupKeywordsText])

    const handleRunDataforseoSearch = async () => {
        if (!activeProject?.id) return
        if (lookupSearchType === 'keyword_overview' && parseLookupKeywords().length === 0) {
            setError('Add at least one keyword for keyword overview lookups.')
            return
        }
        const selectedTopic = activeTopics.find(t => t.id === selectedLookupJobId);
        const queryTextToUse = selectedTopic ? selectedTopic.job_text : '';

        if (lookupSearchType !== 'keyword_overview' && !queryTextToUse.trim()) {
            setError('Select a topic to use as the search query before running a lookup.')
            return
        }

        try {
            setRunningLookup(true)
            if (lookupSearchType === 'expansion_funnel') {
                setLookupProgressText('Extracting SERP... Found 12 Core Angles')
            } else {
                setLookupProgressText(null)
            }
            setError(null)
            setSuccess(null)
            const item = await researchRebuildService.runDataforseoSearch({
                project_id: activeProject.id,
                user_job_id: selectedLookupJobId || undefined,
                primary_category_id: primaryCategoryId || undefined,
                secondary_category_id: secondaryCategoryId || undefined,
                search_type: lookupSearchType,
                query_text: lookupSearchType === 'keyword_overview' ? undefined : queryTextToUse.trim(),
                keywords: lookupSearchType === 'keyword_overview' ? parseLookupKeywords() : undefined,
                limit: lookupSearchType === 'serp' ? 10 : 25,
            })
            setActiveSearchRecord(item)
            await refreshSearchHistory()
            setSuccess('DataForSEO lookup saved to history.')
        } catch (err) {
            console.error('Failed to run DataForSEO search:', err)
            setError(err instanceof Error ? err.message : 'Failed to run DataForSEO search.')
        } finally {
            setRunningLookup(false)
            setLookupProgressText(null)
        }
    }

    React.useEffect(() => {
        if (!runningLookup || lookupSearchType !== 'expansion_funnel') return
        const timer = setTimeout(() => {
            setLookupProgressText('Expanding Database... Analyzing 642 variations')
        }, 2500)
        return () => clearTimeout(timer)
    }, [runningLookup, lookupSearchType])

    const handleSaveArticleDraft = async () => {
        if (!activeProject?.id || !selectedLookupJobId) {
            setError('Create or select a topic first.')
            return
        }
        if (!articleTitleDraft.trim()) {
            setError('Add an article title before saving.')
            return
        }

        const keywordList = Array.from(
            new Set(
                [
                    articlePrimaryKeyword,
                    ...articleSecondaryKeywordsText
                        .split(/[\n,]+/)
                        .map((item) => item.trim()),
                ].filter(Boolean),
            ),
        )

        try {
            setSavingCandidateKey('article-draft')
            setError(null)
            setSuccess(null)
            await researchRebuildService.createCandidate({
                project_id: activeProject.id,
                user_job_id: selectedLookupJobId,
                candidate_type: 'seo_article',
                candidate_text: articleTitleDraft.trim(),
                normalized_candidate_text: articleTitleDraft.trim().toLowerCase(),
                candidate_metadata: {
                    creation_source: 'manual_article_draft',
                    primary_keyword: articlePrimaryKeyword || null,
                    category_context: {
                        project_id: activeProject.id,
                        primary_category_id: primaryCategoryId || null,
                        secondary_category_id: secondaryCategoryId || null,
                        primary_category_name: primaryCategory?.name || null,
                        secondary_category_name: secondaryCategory?.name || null,
                    },
                },
                source_keywords_json: keywordList,
            })
            setSuccess('Article draft saved. You can now continue to validation.')
        } catch (err) {
            console.error('Failed to save article draft:', err)
            setError(err instanceof Error ? err.message : 'Failed to save article draft.')
        } finally {
            setSavingCandidateKey(null)
        }
    }

    const handleArchiveJob = async (jobId: string) => {
        setArchivingJobIds((current) => new Set(current).add(jobId))
        const remainingTopics = jobs.filter((job) => job.id !== jobId)
        const nextSelectedTopicId = selectedLookupJobId === jobId
            ? (remainingTopics[0]?.id || '')
            : selectedLookupJobId

        setJobs(remainingTopics)
        setWorkflowResults((current) => current.filter((result) => result.job?.id !== jobId))
        setWorkflowTotalJobs((current) => Math.max(0, current - 1))
        setSelectedLookupJobId(nextSelectedTopicId)
        if (selectedLookupJobId === jobId) {
            setActiveSearchRecord(null)
            setDataforseoSearches([])
        }

        try {
            await researchRebuildService.archiveJob(jobId)
            setSuccess('Topic removed from the active workflow.')
        } catch (err) {
            console.error('Failed to archive job:', err)
            await refreshPageContext()
            setError('Failed to remove topic.')
        } finally {
            setArchivingJobIds((current) => {
                const next = new Set(current)
                next.delete(jobId)
                return next
            })
        }
    }

    const chunkJobIds = React.useCallback((jobIds: string[]) => {
        if (jobIds.length <= WORKFLOW_BATCH_SIZE) return [jobIds]
        const batches: string[][] = []
        for (let index = 0; index < jobIds.length; index += WORKFLOW_BATCH_SIZE) {
            batches.push(jobIds.slice(index, index + WORKFLOW_BATCH_SIZE))
        }
        return batches
    }, [])

    const executeWorkflow = async (jobIds: string[], successLabel: string) => {
        if (!activeProject?.id || jobIds.length === 0) return
        try {
            setRunningWorkflow(true)
            setError(null)
            setSuccess(null)
            const batches = chunkJobIds(jobIds)
            let lastWorkflowRunId = workflowRunFilter
            for (const batch of batches) {
                const response = await researchRebuildService.runWorkflow({
                    project_id: activeProject.id,
                    user_job_ids: batch,
                })
                if (response.workflow_run_id) {
                    lastWorkflowRunId = response.workflow_run_id
                }
            }
            setWorkflowPage(0)
            if (batches.length === 1 && lastWorkflowRunId && lastWorkflowRunId !== 'all') {
                setWorkflowRunFilter(lastWorkflowRunId)
            } else if (batches.length > 1) {
                setWorkflowRunFilter('all')
            }
            await refreshPageContext({
                workflowRunId: batches.length === 1 ? lastWorkflowRunId || workflowRunFilter : 'all',
                workflowPage: 0,
            })
            if (batches.length > 1) {
                setSuccess(`${successLabel} Processed in ${batches.length} smaller batches to avoid request timeouts.`)
            } else {
                setSuccess(successLabel)
            }
        } catch (err) {
            console.error('Failed to run rebuild workflow:', err)
            const message = err instanceof Error ? err.message.toLowerCase() : ''
            if (message.includes('timeout')) {
                setError('Rebuild workflow timed out in the browser before the server finished. Try again now that the request timeout has been extended.')
            } else {
                setError('Failed to run rebuild workflow.')
            }
        } finally {
            setRunningWorkflow(false)
        }
    }

    const handleRunWorkflow = async () => {
        await executeWorkflow(
            activeTopics.map((job) => job.id),
            `Workflow ran for ${activeTopics.length} saved topic${activeTopics.length === 1 ? '' : 's'}.`,
        )
    }

    const handleReleaseSoftware = async (outcomeId: string) => {
        setMutatingOutcomeIds((current) => new Set(current).add(outcomeId))
        try {
            await researchRebuildService.releaseSoftwareOutcome(outcomeId)
            await refreshPageContext()
            setSuccess('Software outcome released to Released Software.')
        } catch (err) {
            console.error('Failed to release software outcome:', err)
            setError(err instanceof Error ? err.message : 'Failed to release software outcome.')
        } finally {
            setMutatingOutcomeIds((current) => {
                const next = new Set(current)
                next.delete(outcomeId)
                return next
            })
        }
    }

    const handlePersistOutcome = async (outcomeId: string) => {
        if (!activeProject?.id) return
        setMutatingOutcomeIds((current) => new Set(current).add(outcomeId))
        try {
            await researchRebuildService.persistOutcomeToContentIdea(outcomeId, {
                project_id: activeProject.id,
                category_context: {
                    project_id: activeProject.id,
                    primary_category_id: primaryCategoryId || null,
                    secondary_category_id: secondaryCategoryId || null,
                    primary_category_name: primaryCategory?.name || null,
                    secondary_category_name: secondaryCategory?.name || null,
                    category_path: [primaryCategory?.name, secondaryCategory?.name].filter(Boolean).join(' / ') || null,
                },
            })
            await refreshPageContext()
            setSuccess('Outcome sent to Content Studio.')
        } catch (err) {
            console.error('Failed to send outcome to Content Studio:', err)
            setError('Failed to send outcome to Content Studio.')
        } finally {
            setMutatingOutcomeIds((current) => {
                const next = new Set(current)
                next.delete(outcomeId)
                return next
            })
        }
    }

    const toggleCandidateExpansion = (candidateId: string) => {
        setExpandedCandidateIds((current) => {
            const next = new Set(current)
            if (next.has(candidateId)) {
                next.delete(candidateId)
            } else {
                next.add(candidateId)
            }
            return next
        })
    }

    const handleRejectCandidate = async (candidateId: string, reason: string) => {
        setRejectingCandidateIds((current) => new Set(current).add(candidateId))
        try {
            await researchRebuildService.rejectCandidate(candidateId, {
                rejection_reason_tags: [reason],
            })
            await refreshPageContext()
            setSuccess('Candidate rejected.')
        } catch (err) {
            console.error('Failed to reject candidate:', err)
            setError('Failed to reject candidate.')
        } finally {
            setRejectingCandidateIds((current) => {
                const next = new Set(current)
                next.delete(candidateId)
                return next
            })
        }
    }

    const handleRefreshValidation = async (validationRunId: string) => {
        setRefreshingValidationIds((current) => new Set(current).add(validationRunId))
        try {
            await researchRebuildService.refreshValidation({
                validation_run_id: validationRunId,
                ttl_days: 14,
                freshness_state: 'fresh',
            })
            await refreshPageContext()
            setSuccess('Validation freshness refreshed.')
        } catch (err) {
            console.error('Failed to refresh validation:', err)
            setError('Failed to refresh validation.')
        } finally {
            setRefreshingValidationIds((current) => {
                const next = new Set(current)
                next.delete(validationRunId)
                return next
            })
        }
    }

    const handleRefreshAllStaleValidations = async () => {
        if (staleValidationTargets.length === 0) return

        setRefreshingAllStale(true)
        try {
            await Promise.all(
                staleValidationTargets.map(async ({ validationRunId }) =>
                    researchRebuildService.refreshValidation({
                        validation_run_id: validationRunId,
                        ttl_days: 14,
                        freshness_state: 'fresh',
                    }),
                ),
            )
            await refreshPageContext()
            setSuccess(`Refreshed ${staleValidationTargets.length} stale validation${staleValidationTargets.length === 1 ? '' : 's'}.`)
        } catch (err) {
            console.error('Failed to refresh stale validations:', err)
            setError('Failed to refresh stale validations.')
        } finally {
            setRefreshingAllStale(false)
        }
    }

    const handleCopyCurrentViewLink = async () => {
        try {
            if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
                await navigator.clipboard.writeText(currentViewUrl)
                setSuccess('Copied rebuild view link.')
                return
            }
            setError('Clipboard is not available in this browser.')
        } catch (err) {
            console.error('Failed to copy rebuild view link:', err)
            setError('Failed to copy rebuild view link.')
        }
    }

    return (
        <div className="min-h-screen bg-background text-foreground selection:bg-indigo-500/30">
            {/* Premium Header */}
            <header className="sticky top-0 z-50 border-b border-border bg-card shadow-[0_12px_30px_rgba(0,0,0,0.35)]">
                <div className="mx-auto w-full px-4 lg:px-8 px-6 py-4 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <div className="bg-gradient-to-tr from-indigo-600 to-indigo-400 p-2.5 rounded-2xl shadow-[0_0_20px_rgba(99,102,241,0.4)]">
                            <Sparkles className="h-5 w-5 text-foreground" />
                        </div>
                        <div>
                            <h1 className="text-xl font-black tracking-tight text-foreground flex items-center gap-2">
                                Research <span className="text-indigo-400">{isJobsPage ? 'Setup' : 'Results'}</span>
                                <Badge variant="outline" className="text-[10px] bg-indigo-500/10 border-indigo-500/20 text-indigo-400 font-black uppercase">v2.0</Badge>
                            </h1>
                            <p className="text-[10px] text-muted-foreground font-black uppercase tracking-[0.25em]">
                                {isJobsPage ? 'Step 1 of 2 · Job Discovery' : 'Step 2 of 2 · Opportunity Validation'}
                            </p>
                        </div>
                    </div>
                    
                    {/* Visual Flow Indicator */}
                    <div className="hidden lg:flex flex-1 justify-center items-center gap-4">
                        <FlowStep 
                            number="1" 
                            label="Discover" 
                            active={isJobsPage} 
                            icon={<Layers className="h-3.5 w-3.5" />} 
                            onClick={() => navigate(jobsPagePath)}
                        />
                        <div className="w-4 h-px bg-border" />
                        <FlowStep 
                            number="2" 
                            label="Validate" 
                            active={isOpportunitiesPage} 
                            icon={<Target className="h-3.5 w-3.5" />} 
                            onClick={() => navigate(opportunitiesPagePath)}
                            disabled={activeTopics.length === 0}
                        />
                        <div className="w-4 h-px bg-border" />
                        <FlowStep 
                            number="3" 
                            label="Promote" 
                            active={false} 
                            icon={<Rocket className="h-3.5 w-3.5" />} 
                            disabled={true}
                        />
                    </div>

                    <div className="flex items-center gap-4">
                        <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 bg-white/5 rounded-xl border border-border">
                            <div className="w-2 h-2 rounded-full bg-emerald-500" />
                            <span className="text-[10px] font-black text-muted-foreground uppercase tracking-widest">Live Engine</span>
                        </div>
                        <Button 
                            variant="ghost" 
                            size="sm" 
                            className="rounded-xl h-10 px-4 text-muted-foreground hover:text-foreground hover:bg-white/5 border border-border transition-all"
                            onClick={handleCopyCurrentViewLink}
                        >
                            <Copy className="mr-2 h-3.5 w-3.5" />
                            <span className="text-xs font-bold uppercase tracking-wider">Share Link</span>
                        </Button>
                    </div>
                </div>
            </header>

            <main className="mx-auto w-full px-4 lg:px-8 p-6 lg:p-10">
                {/* Notifications */}
                <div className="fixed bottom-10 right-10 z-[100] space-y-4 max-w-md">
                    {error && (
                        <div className="flex items-start gap-4 bg-red-500/10 border border-red-500/20 text-red-200 px-5 py-4 rounded-2xl backdrop-blur-3xl animate-in slide-in-from-bottom-5 shadow-2xl">
                            <div className="bg-red-500/20 p-1.5 rounded-lg"><XCircle className="h-5 w-5 text-red-400" /></div>
                            <div className="flex-1">
                                <h4 className="text-xs font-bold uppercase tracking-widest text-red-400 mb-1">Error Encountered</h4>
                                <p className="text-[13px] font-medium leading-relaxed opacity-90">{error}</p>
                            </div>
                            <button onClick={() => setError(null)} className="text-muted-foreground hover:text-foreground transition-colors"><XCircle className="h-4 w-4" /></button>
                        </div>
                    )}
                    {success && (
                        <div className="flex items-start gap-4 bg-emerald-500/10 border border-emerald-500/20 text-emerald-200 px-5 py-4 rounded-2xl backdrop-blur-3xl animate-in slide-in-from-bottom-5 shadow-2xl">
                            <div className="bg-emerald-500/20 p-1.5 rounded-lg"><CheckCircle2 className="h-5 w-5 text-emerald-400" /></div>
                            <div className="flex-1">
                                <h4 className="text-xs font-bold uppercase tracking-widest text-emerald-400 mb-1">Success</h4>
                                <p className="text-[13px] font-medium leading-relaxed opacity-90">{success}</p>
                            </div>
                            <button onClick={() => setSuccess(null)} className="text-muted-foreground hover:text-foreground transition-colors"><XCircle className="h-4 w-4" /></button>
                        </div>
                    )}
                </div>

                <div
                    className={cn(
                        "grid gap-10 items-start relative",
                        isJobsPage ? "mx-auto w-full lg:grid-cols-1" : "xl:grid-cols-[1fr,400px]",
                    )}
                >
                    {/* Background Tints */}
                    {!isJobsPage && (
                        <>
                            <div className="absolute inset-y-0 left-0 w-[400px] bg-indigo-500/[0.02] rounded-[3rem] -ml-4 -my-4 pointer-events-none hidden lg:block" />
                            <div className="absolute inset-y-0 left-[440px] right-0 bg-emerald-500/[0.01] rounded-[3rem] -mr-4 -my-4 pointer-events-none hidden lg:block" />
                        </>
                    )}
                    {/* LEFT PANEL: CONFIGURATION & DISCOVERY */}
                    <aside className={cn("space-y-8", isJobsPage ? "mx-auto w-full w-full" : "lg:sticky lg:top-32")}>
                        <div className="px-2 space-y-4 relative z-10">
                             <h2 className="text-[11px] font-black text-indigo-400 uppercase tracking-[0.4em] flex items-center gap-3">
                                <span className="flex items-center justify-center w-6 h-6 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-[10px]">01</span>
                                {isJobsPage ? 'Discovery Phase' : 'Research Context'}
                            </h2>
                            <div>
                                <h3 className="text-xl font-black text-foreground tracking-tight mb-2">
                                    {isJobsPage ? 'Capture User Intent' : 'Keep Context Visible'}
                                </h3>
                                <p className="text-muted-foreground text-xs font-medium leading-relaxed mb-6">
                                    {isJobsPage
                                        ? 'Follow a simple 3-step flow: define a topic, research keywords, then save an article draft with assigned keywords.'
                                        : 'Review the approved jobs and scope while you validate only the saved opportunities on the second step.'}
                                </p>
                            </div>
                        </div>

                        <section className={cn("bg-card border border-border rounded-[2.5rem] shadow-2xl overflow-hidden", isOpportunitiesPage && "hidden")}>
                            

                            <div className="p-8 space-y-8">
                                <div>
                                    <div className="flex items-center gap-3">
                                        <span className="flex h-8 w-8 items-center justify-center rounded-full border border-indigo-500/30 bg-indigo-500/15 text-xs font-black text-indigo-300">1</span>
                                        <div>
                                            <h4 className="text-base font-black text-foreground">Select Domain / Category And Define A Topic</h4>
                                            <p className="text-sm text-muted-foreground">Pick the category scope, then write the exact topic you want to research.</p>
                                        </div>
                                    </div>

                                    <div className="grid gap-4 md:grid-cols-2">
                                        <SelectField 
                                            label="Primary Domain Category" 
                                            value={primaryCategoryId} 
                                            onChange={(val) => {
                                                setPrimaryCategoryId(val)
                                                setSecondaryCategoryId('')
                                                setActiveBatchId('')
                                                setWorkflowRunFilter('all')
                                                setWorkflowResults([])
                                                setWorkflowRuns([])
                                                setWorkflowTotalJobs(0)
                                            }}
                                            options={[{ value: '', label: 'Select Primary Category' }, ...primaryCategories.map(c => ({ value: c.id, label: c.name }))]} 
                                        />
                                        <SelectField 
                                            label="Target Sub-Category" 
                                            value={secondaryCategoryId} 
                                            onChange={(val) => {
                                                setSecondaryCategoryId(val)
                                                setActiveBatchId('')
                                                setWorkflowRunFilter('all')
                                                setWorkflowResults([])
                                                setWorkflowRuns([])
                                                setWorkflowTotalJobs(0)
                                            }}
                                            options={[{ value: '', label: 'Select Sub-category' }, ...secondaryCategories.map(c => ({ value: c.id, label: c.name }))]} 
                                        />
                                    </div>

                                    <div className="rounded-2xl border border-border bg-card mt-6">
                                        <button
                                            type="button"
                                            onClick={() => setShowCategoryDescriptions((current) => !current)}
                                            className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left"
                                        >
                                            <div>
                                                <div className="text-[10px] font-black uppercase tracking-[0.25em] text-muted-foreground">
                                                    Category Descriptions
                                                </div>
                                                <p className="mt-1 text-xs text-muted-foreground">
                                                    Expand to review the selected category and sub-category details.
                                                </p>
                                            </div>
                                            {showCategoryDescriptions ? <ChevronUp className="h-4 w-4 text-muted-foreground" /> : <ChevronDown className="h-4 w-4 text-muted-foreground" />}
                                        </button>
                                        {showCategoryDescriptions && (
                                            <div className="max-h-56 space-y-4 overflow-y-auto border-t border-border px-4 py-4">
                                                <div className="rounded-2xl border border-indigo-400/15 bg-indigo-500/[0.06] px-4 py-3">
                                                    <div className="text-[10px] font-black uppercase tracking-[0.22em] text-indigo-200/90">
                                                        Category
                                                    </div>
                                                    <div className="mt-2 text-sm font-semibold text-foreground">
                                                        {primaryCategory?.name || 'No category selected'}
                                                    </div>
                                                    <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                                                        {primaryCategory?.description || 'Select a primary category to see its description here.'}
                                                    </p>
                                                </div>
                                                <div className="rounded-2xl border border-emerald-400/15 bg-emerald-500/[0.06] px-4 py-3">
                                                    <div className="text-[10px] font-black uppercase tracking-[0.22em] text-emerald-200/90">
                                                        Sub-Category
                                                    </div>
                                                    <div className="mt-2 text-sm font-semibold text-foreground">
                                                        {secondaryCategory?.name || 'No sub-category selected'}
                                                    </div>
                                                    <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                                                        {secondaryCategory?.description || 'Select a sub-category to see its description here.'}
                                                    </p>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                    <div className="space-y-2">
                                        <label className="text-[10px] font-black uppercase tracking-[0.25em] text-muted-foreground">
                                            Topic
                                        </label>
                                        <textarea
                                            value={manualJobText}
                                            onChange={(e) => setManualJobText(e.target.value)}
                                            placeholder="Example: Best expired domains for local SEO lead generation"
                                            className="min-h-[88px] w-full rounded-2xl border border-border bg-card px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground outline-none transition focus:border-indigo-400/60 focus:ring-2 focus:ring-indigo-500/20"
                                        />
                                    </div>
                                    <div className="flex flex-col gap-3 sm:flex-row">
                                        <Button
                                            className="flex-1 bg-primary text-primary-foreground hover:bg-primary/90 h-12 rounded-2xl font-black uppercase tracking-[0.15em] text-[11px]"
                                            onClick={handleCreateManualJob}
                                            disabled={creatingManualJob || !manualJobText.trim()}
                                        >
                                            {creatingManualJob ? <Loader2 className="mr-3 h-4 w-4 animate-spin" /> : <ArrowRight className="mr-3 h-4 w-4" />}
                                            Save Topic
                                        </Button>
                                        <Button
                                            type="button"
                                            variant="outline"
                                            className="h-12 rounded-2xl border-border bg-card px-5 text-[11px] font-black uppercase tracking-[0.15em] text-card-foreground hover:bg-white/[0.06]"
                                            onClick={() => setShowSavedTopicsModal(true)}
                                        >
                                            View Saved Topics
                                        </Button>
                                    </div>
                                </div>

                                <div className="rounded-[2rem] border border-emerald-400/20 bg-emerald-50/50 dark:bg-emerald-500/10 p-6 space-y-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]">
                                    <div className="flex items-center gap-3">
                                        <span className="flex h-8 w-8 items-center justify-center rounded-full border border-emerald-500/30 bg-emerald-500/15 text-xs font-black text-emerald-300">2</span>
                                        <div>
                                            <h4 className="text-base font-black text-foreground">Use SEO Tools To Find Keywords</h4>
                                            <p className="text-sm text-muted-foreground">Run a manual lookup, inspect the results, then choose the keyword you want to use.</p>
                                        </div>
                                    </div>

                                    <SelectField
                                        label="Topic"
                                        value={selectedLookupJobId}
                                        onChange={setSelectedLookupJobId}
                                        options={[
                                            { value: '', label: 'Select Topic' },
                                            ...activeTopics.map((job) => ({ value: job.id, label: job.job_text })),
                                        ]}
                                    />

                                    <SelectField
                                        label="Lookup Type"
                                        value={lookupSearchType}
                                        onChange={(value) => setLookupSearchType(value as 'related_keywords' | 'keyword_overview' | 'serp')}
                                        options={DATAFORSEO_SEARCH_TYPES.map((item) => ({ value: item.value, label: item.label }))}
                                    />

                                    {lookupSearchType === 'keyword_overview' && (
                                        <div className="space-y-2">
                                            <label className="text-[10px] font-black uppercase tracking-[0.25em] text-muted-foreground">
                                                Keywords
                                            </label>
                                            <textarea
                                                value={lookupKeywordsText}
                                                onChange={(e) => setLookupKeywordsText(e.target.value)}
                                                placeholder="Paste one keyword per line or comma separated"
                                                className="min-h-[110px] w-full rounded-2xl border border-border bg-card px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground outline-none transition focus:border-emerald-400/60 focus:ring-2 focus:ring-emerald-500/20"
                                            />
                                        </div>
                                    )}

                                    <Button
                                        className="w-full bg-emerald-500 text-black hover:bg-emerald-400 h-12 rounded-2xl font-black uppercase tracking-[0.15em] text-[11px]"
                                        onClick={handleRunDataforseoSearch}
                                        disabled={runningLookup || !activeProject?.id}
                                    >
                                        {runningLookup ? (
                                            <>
                                                <Loader2 className="mr-3 h-4 w-4 animate-spin" />
                                                {lookupProgressText || 'Running Lookup...'}
                                            </>
                                        ) : (
                                            <>
                                                <Search className="mr-3 h-4 w-4" />
                                                Run And Save Lookup
                                            </>
                                        )}
                                    </Button>

                                    <div className="rounded-2xl border border-border bg-card p-4">
                                        <div className="mb-3 flex items-center justify-between gap-3">
                                            <span className="text-[10px] font-black uppercase tracking-[0.22em] text-muted-foreground">Keyword Results</span>
                                            {activeSearchRecord && (
                                                <span className="text-[10px] text-muted-foreground">
                                                    {(activeSearchRecord.result_summary_json?.result_count as number | undefined) ?? activeSearchPreviewItems.length} results
                                                </span>
                                            )}
                                        </div>
                                        {!activeSearchRecord ? (
                                            <p className="text-xs text-muted-foreground">Run a lookup to inspect results here.</p>
                                        ) : (
                                            <div className="space-y-3">
                                                <div className="rounded-2xl border border-border bg-white/[0.03] px-4 py-3">
                                                    <div className="text-[10px] font-black uppercase tracking-[0.18em] text-muted-foreground">
                                                        {activeSearchRecord.search_type.replace('_', ' ')}
                                                    </div>
                                                    <p className="mt-2 text-sm text-card-foreground">{activeSearchRecord.query_text}</p>
                                                    {activeSearchRecord.search_type === 'expansion_funnel' && (
                                                            <div className="flex gap-2 mt-3">
                                                                <Badge 
                                                                    onClick={() => setShowEasyWinsModal(true)} 
                                                                    className="cursor-pointer bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 px-3 py-1 text-xs border border-emerald-500/30 shadow-[0_0_15px_rgba(16,185,129,0.1)] transition-all hover:shadow-[0_0_20px_rgba(16,185,129,0.2)]"
                                                                >
                                                                    <Sparkles className="mr-1.5 h-3.5 w-3.5" />
                                                                    Contains {activeSearchPreviewItems.length} Easy-Wins
                                                                </Badge>
                                                                <Badge 
                                                                    onClick={() => setShowAllMetricsModal(true)} 
                                                                    className="cursor-pointer bg-indigo-500/20 text-indigo-400 hover:bg-indigo-500/30 px-3 py-1 text-xs border border-indigo-500/30 shadow-[0_0_15px_rgba(99,102,241,0.1)] transition-all hover:shadow-[0_0_20px_rgba(99,102,241,0.2)]"
                                                                >
                                                                    <Layers className="mr-1.5 h-3.5 w-3.5" />
                                                                    View All Metrics
                                                                </Badge>
                                                            </div>
                                                    )}
                                                </div>
                                                <div className="max-h-[260px] space-y-2 overflow-y-auto pr-1">
                                                    {activeSearchPreviewItems.length === 0 ? (
                                                        <p className="text-xs text-muted-foreground">This search did not return preview items.</p>
                                                    ) : (
                                                        activeSearchPreviewItems.slice(0, 8).map((item, index) => {
                                                            const keyword = String(item.keyword || item.title || item.url || `Result ${index + 1}`)
                                                            return (
                                                                <div key={`${activeSearchRecord.id}-${index}`} className="rounded-2xl border border-border bg-white/[0.03] px-4 py-3">
                                                                    <div className="flex items-start justify-between gap-3">
                                                                        <div className="min-w-0 flex-1">
                                                                            <p className="text-sm text-card-foreground">{keyword}</p>
                                                                            <p className="mt-1 text-[11px] text-muted-foreground">
                                                                                {[item.search_volume ? `Vol ${item.search_volume}` : null, item.keyword_difficulty ? `KD ${item.keyword_difficulty}` : null, item.cpc ? `CPC ${item.cpc}` : null]
                                                                                    .filter(Boolean)
                                                                                    .join(' · ') || String(item.snippet || '')}
                                                                            </p>
                                                                        </div>
                                                                        <button
                                                                            type="button"
                                                                            onClick={() => handleUseKeywordForArticle(String(item.keyword || activeSearchRecord.query_text || keyword))}
                                                                            className="rounded-xl border border-border bg-white/[0.05] px-3 py-2 text-[10px] font-black uppercase tracking-[0.12em] text-foreground transition hover:bg-white/[0.1]"
                                                                        >
                                                                            Use Keyword
                                                                        </button>
                                                                    </div>
                                                                </div>
                                                            )
                                                        })
                                                    )}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>

                                <div className="rounded-[2rem] border border-amber-400/20 bg-amber-50/50 dark:bg-amber-500/10 p-6 space-y-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]">
                                    <div className="flex items-center gap-3">
                                        <span className="flex h-8 w-8 items-center justify-center rounded-full border border-amber-500/30 bg-amber-500/15 text-xs font-black text-amber-300">3</span>
                                        <div>
                                            <h4 className="text-base font-black text-foreground">Define Article Title And Assign Keywords</h4>
                                            <p className="text-sm text-muted-foreground">Write the title you want, assign the primary keyword, then save the article draft for validation.</p>
                                        </div>
                                    </div>

                                    <div className="space-y-2">
                                        <label className="text-[10px] font-black uppercase tracking-[0.25em] text-muted-foreground">
                                            Article Title
                                        </label>
                                        <Input
                                            value={articleTitleDraft}
                                            onChange={(e) => setArticleTitleDraft(e.target.value)}
                                            placeholder="Example: Best Expired Domains for Local SEO in 2026"
                                            className="bg-card border-border rounded-xl text-xs h-12 text-foreground focus:ring-amber-500/30"
                                        />
                                    </div>

                                    <div className="space-y-2">
                                        <label className="text-[10px] font-black uppercase tracking-[0.25em] text-muted-foreground">
                                            Primary Keyword
                                        </label>
                                        <Input
                                            value={articlePrimaryKeyword}
                                            onChange={(e) => setArticlePrimaryKeyword(e.target.value)}
                                            placeholder="Pick one keyword from Step 2 or type it here"
                                            className="bg-card border-border rounded-xl text-xs h-12 text-foreground focus:ring-amber-500/30"
                                        />
                                    </div>

                                    <div className="space-y-2">
                                        <label className="text-[10px] font-black uppercase tracking-[0.25em] text-muted-foreground">
                                            Assigned Keywords
                                        </label>
                                        <textarea
                                            value={articleSecondaryKeywordsText}
                                            onChange={(e) => setArticleSecondaryKeywordsText(e.target.value)}
                                            placeholder="One keyword per line"
                                            className="min-h-[110px] w-full rounded-2xl border border-border bg-card px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground outline-none transition focus:border-amber-500/40 focus:ring-2 focus:ring-amber-500/20"
                                        />
                                    </div>

                                    <Button
                                        className="w-full bg-amber-400 text-black hover:bg-amber-300 h-12 rounded-2xl font-black uppercase tracking-[0.15em] text-[11px]"
                                        onClick={handleSaveArticleDraft}
                                        disabled={savingCandidateKey === 'article-draft' || !selectedLookupJobId}
                                    >
                                        {savingCandidateKey === 'article-draft' ? <Loader2 className="mr-3 h-4 w-4 animate-spin" /> : <FileText className="mr-3 h-4 w-4" />}
                                        Save Article Draft
                                    </Button>
                                </div>

                                <details className="rounded-[2rem] border border-border bg-card p-6">
                                    <summary className="cursor-pointer text-[10px] font-black uppercase tracking-[0.22em] text-muted-foreground">
                                        Optional AI Batch Tools
                                    </summary>
                                    <div className="mt-5 space-y-5">
                                        <div className="space-y-2">
                                            <label className="text-[10px] font-black uppercase tracking-[0.25em] text-muted-foreground">
                                                Focus Area For AI Batch
                                            </label>
                                            <textarea
                                                value={focusArea}
                                                onChange={(e) => setFocusArea(e.target.value)}
                                                placeholder="Example: privacy-first PKM workflows, second-brain tools for structured thinking, AI research efficiency for solo founders"
                                                className="min-h-[88px] w-full rounded-2xl border border-border bg-muted px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground outline-none transition focus:border-indigo-500/40 focus:ring-2 focus:ring-indigo-500/20"
                                            />
                                        </div>
                                        <div className="space-y-2">
                                            <label className="text-[10px] font-black uppercase tracking-[0.25em] text-muted-foreground">
                                                Avoid In AI Batch
                                            </label>
                                            <Input
                                                value={avoidGuidance}
                                                onChange={(e) => setAvoidGuidance(e.target.value)}
                                                placeholder="Example: generic productivity advice, enterprise use cases, broad AI news"
                                                className="bg-muted border-border rounded-xl text-xs h-12 text-foreground focus:ring-indigo-500/30"
                                            />
                                        </div>
                                        <button
                                            type="button"
                                            onClick={() => setStartFreshBatch((current) => !current)}
                                            className="flex items-start gap-3 rounded-2xl border border-border bg-white/[0.04] px-4 py-4 text-left transition hover:bg-white/[0.07]"
                                        >
                                            <span className={cn(
                                                "mt-0.5 inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full border transition-all",
                                                startFreshBatch
                                                    ? "border-emerald-500/30 bg-emerald-500/15 text-emerald-400"
                                                    : "border-border bg-white/[0.03] text-muted-foreground",
                                            )}>
                                                {startFreshBatch ? <CheckCircle2 className="h-3.5 w-3.5" /> : <Info className="h-3.5 w-3.5" />}
                                            </span>
                                            <div className="space-y-1">
                                                <p className="text-[10px] font-black uppercase tracking-[0.22em] text-muted-foreground">
                                                    Start Fresh Batch
                                                </p>
                                                <p className="text-[11px] leading-relaxed text-muted-foreground">
                                                    {startFreshBatch
                                                        ? "Archive the current draft and approved jobs in this category scope before generating a cleaner batch."
                                                        : "Keep the current active jobs visible and add the next generation on top of them."}
                                                </p>
                                            </div>
                                        </button>
                                        <Button 
                                            className="w-full bg-primary text-primary-foreground hover:bg-indigo-50 h-12 rounded-2xl font-black uppercase tracking-[0.15em] text-[11px] transition-all"
                                            onClick={handleGenerateJobs} 
                                            disabled={generatingJobs || !activeProject?.id}
                                        >
                                            {generatingJobs ? <Loader2 className="mr-3 h-4 w-4 animate-spin" /> : <Sparkles className="mr-3 h-4 w-4 text-indigo-600" />}
                                            Generate AI Topics
                                        </Button>
                                    </div>
                                </details>
                            </div>

                            <div className="border-t border-border bg-card">
                                <div className="flex flex-col gap-4 p-6 md:flex-row md:items-center md:justify-between">
                                    <div>
                                        <div className="flex items-center gap-2">
                                            <Layers className="h-3.5 w-3.5 text-muted-foreground" />
                                            <span className="text-[10px] font-black uppercase tracking-[0.22em] text-muted-foreground">Saved Topics</span>
                                        </div>
                                        <p className="mt-2 text-sm text-muted-foreground">
                                            {activeTopics.length} saved topic{activeTopics.length === 1 ? '' : 's'} in this scope.
                                        </p>
                                    </div>
                                    <div className="flex gap-2">
                                        <Button
                                            type="button"
                                            variant="outline"
                                            className="rounded-xl border-border bg-white/[0.03] px-4 text-[10px] font-black uppercase tracking-[0.15em] text-card-foreground hover:bg-white/[0.07]"
                                            onClick={() => setShowSavedTopicsModal(true)}
                                        >
                                            Manage Topics
                                        </Button>
                                    </div>
                                </div>
                            </div>
                        </section>

                        {isOpportunitiesPage && (
                            <section className="bg-card border border-white/5 rounded-[2.5rem] shadow-2xl overflow-hidden">
                                <div className="p-8 border-b border-white/5 bg-gradient-to-br from-white/[0.03] to-transparent">
                                    <h3 className="text-sm font-black text-foreground uppercase tracking-widest flex items-center gap-3">
                                        <Globe className="h-4 w-4 text-indigo-400" />
                                        Validation Context
                                    </h3>
                                </div>
                                <div className="p-8 space-y-6">
                                    <div className="grid gap-4 md:grid-cols-2">
                                        <ContextStat label="Primary Category" value={primaryCategory?.name || 'Not selected'} />
                                        <ContextStat label="Sub-Category" value={secondaryCategory?.name || 'Not selected'} />
                                    </div>
                                    <ContextBlock label="Focus Area" value={focusArea || 'No focus area supplied for this batch.'} />
                                    <ContextBlock label="Avoid Guidance" value={avoidGuidance || 'No avoid guidance supplied for this batch.'} />
                                    <div className="grid gap-4 md:grid-cols-2">
                                        <ContextStat label="Saved Topics" value={String(activeTopics.length)} />
                                        <ContextStat label="Workflow Runs" value={String(workflowRuns.length)} />
                                    </div>
                                    <div className="rounded-2xl border border-white/5 bg-black/30 p-5">
                                        <div className="mb-3 flex items-center justify-between">
                                            <span className="text-[10px] font-black uppercase tracking-[0.25em] text-muted-foreground">Saved Topic Set</span>
                                            <Button
                                                variant="outline"
                                                size="sm"
                                                className="border-border bg-white/[0.03] hover:bg-white/[0.08]"
                                                onClick={() => navigate(jobsPagePath)}
                                            >
                                                Back To Jobs
                                            </Button>
                                        </div>
                                        <div className="space-y-3">
                                            {activeTopics.length === 0 ? (
                                                <p className="text-sm text-muted-foreground">No saved topics yet. Go back to Jobs and create the topics you want to validate.</p>
                                            ) : (
                                                activeTopics.slice(0, 6).map((job) => (
                                                    <div key={job.id} className="rounded-2xl border border-white/5 bg-white/[0.02] px-4 py-3 text-sm text-muted-foreground">
                                                        {job.job_text}
                                                    </div>
                                                ))
                                            )}
                                        </div>
                                    </div>
                                </div>
                            </section>
                        )}

                        {/* Phase 02 Launcher */}
                        <div className="relative group">
                            <div className="absolute -inset-0.5 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-[2.5rem] blur opacity-20 group-hover:opacity-40 transition-opacity" />
                            <div className="relative bg-gradient-to-br from-indigo-50 dark:from-[#121218] to-card dark:to-[#0d0d0f] p-10 rounded-[2.5rem] border border-border shadow-2xl overflow-hidden">
                                <div className="absolute top-0 right-0 p-6 opacity-5 group-hover:scale-125 group-hover:opacity-10 transition-all duration-700 pointer-events-none">
                                    <Rocket className="h-32 w-32 text-foreground" />
                                </div>
                                <h3 className="text-[11px] font-black text-indigo-400 uppercase tracking-[0.4em] mb-4">
                                    {isJobsPage ? 'Phase 02: Validate' : 'Step Actions'}
                                </h3>
                                <h4 className="text-foreground font-black text-xl mb-3 tracking-tight leading-tight">
                                    {isJobsPage ? <>Opportunity <br />Verification Engine</> : <>Run Or Refresh <br />Validation</>}
                                </h4>
                                <p className="text-muted-foreground text-[13px] mb-10 leading-relaxed font-medium">
                                    {isJobsPage
                                        ? <>Move forward with <span className="text-foreground font-bold">{activeTopics.length} saved topic{activeTopics.length === 1 ? '' : 's'}</span>, then validate only the opportunities you manually saved from your searches.</>
                                        : <>Use the saved topics on the left to run or rerun validation for saved opportunities, then inspect the strongest outcomes on this page.</>}
                                </p>
                                
                                <Button 
                                    className={cn(
                                        "w-full h-14 rounded-2xl font-black uppercase tracking-widest text-[11px] transition-all active:scale-[0.97] border",
                                        activeTopics.length > 0 
                                            ? "bg-indigo-500 text-foreground hover:bg-indigo-400 shadow-[0_0_25px_rgba(99,102,241,0.4)] border-indigo-400/50" 
                                            : "bg-white/5 text-muted-foreground border-white/5 cursor-not-allowed"
                                    )}
                                    onClick={isJobsPage ? () => navigate(opportunitiesPagePath) : handleRunWorkflow}
                                    disabled={(isOpportunitiesPage && runningWorkflow) || activeTopics.length === 0 || !activeProject?.id}
                                >
                                    {isOpportunitiesPage && runningWorkflow
                                        ? <Loader2 className="mr-3 h-5 w-5 animate-spin" />
                                        : <Rocket className={cn("mr-3 h-5 w-5 fill-current", activeTopics.length > 0 ? "animate-bounce" : "")} />}
                                    {activeTopics.length > 0
                                        ? (isJobsPage ? `Continue With ${activeTopics.length} Saved Topic${activeTopics.length === 1 ? '' : 's'}` : `Validate Saved Opportunities`)
                                        : 'Select Candidates First'}
                                </Button>
                            </div>
                        </div>
                    </aside>

                    {/* RIGHT PANEL: ANALYSIS RESULTS */}
                    {isJobsPage ? null : (
                    <div className="space-y-10">
                        {/* Phase 03 Header */}
                        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 px-4 relative z-10">
                            <div className="max-w-xl space-y-4">
                                <h2 className="text-[11px] font-black text-emerald-400 uppercase tracking-[0.3em] flex items-center gap-3">
                                    <span className="flex items-center justify-center w-6 h-6 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-[10px]">03</span>
                                    Intelligence Phase
                                </h2>
                                <div>
                                    <h3 className="text-3xl font-black text-foreground tracking-tight leading-tight">High-Achievability Outcomes</h3>
                                    <p className="text-muted-foreground text-[13px] font-medium mt-3 leading-relaxed mb-6">
                                        Strategic opportunities derived from validated search data. Every outcome represents a winnable path identified by the engine.
                                    </p>
                                </div>
                            </div>
                            
                            <div className="flex flex-wrap items-center gap-3">
                                <div className="px-4 py-2 bg-white/5 rounded-2xl border border-border flex items-center gap-2.5">
                                    <Layers className="h-4 w-4 text-muted-foreground" />
                                    <span className="text-xs font-black text-muted-foreground uppercase tracking-tighter">{workflowTotalJobs} Jobs</span>
                                </div>
                                <div className="px-4 py-2 bg-white/5 rounded-2xl border border-border flex items-center gap-2.5">
                                    <Target className="h-4 w-4 text-muted-foreground" />
                                    <span className="text-xs font-black text-muted-foreground uppercase tracking-tighter">{workflowResults.reduce((s, j) => s + j.candidates.length, 0)} Outcomes</span>
                                </div>
                            </div>
                        </div>

                        <section className="bg-card border border-white/5 rounded-[3rem] p-1 shadow-2xl min-h-[800px] flex flex-col relative overflow-hidden">
                            {/* Inner Background Glow */}
                            <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-indigo-500/5 blur-[120px] rounded-full -translate-y-1/2 translate-x-1/3 pointer-events-none" />

                            {/* Control Bar */}
                            <div className="p-8 border-b border-white/5 bg-gradient-to-b from-white/[0.02] to-transparent relative z-10">
                                <div className="flex flex-col xl:flex-row xl:items-center justify-between gap-6">
                                    <div className="flex flex-wrap items-center gap-3">
                                        {staleValidationTargets.length > 0 && (
                                            <button 
                                                onClick={handleRefreshAllStaleValidations} 
                                                disabled={refreshingAllStale} 
                                                className="px-5 py-2.5 bg-amber-500/10 border border-amber-500/20 rounded-2xl text-amber-400 text-xs font-black uppercase tracking-widest flex items-center gap-3 hover:bg-amber-500/20 transition-all shadow-xl shadow-amber-500/10"
                                            >
                                                {refreshingAllStale ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
                                                Refresh {staleValidationTargets.length} Stale Results
                                            </button>
                                        )}
                                        <div className="h-10 px-4 bg-white/5 rounded-2xl border border-border flex items-center gap-3">
                                            <Filter className="h-3.5 w-3.5 text-muted-foreground" />
                                            <span className="text-[10px] font-black text-muted-foreground uppercase tracking-widest">Active Filters:</span>
                                            <Badge variant="secondary" className="bg-white/10 text-muted-foreground text-[9px] uppercase tracking-tighter">{routeFilter === 'all' ? 'No Route Filter' : routeFilter.replace('_', ' ')}</Badge>
                                        </div>
                                    </div>

                                    {/* Advanced Filters Container */}
                                    <div className="flex flex-wrap items-center gap-2.5 bg-black/60 p-2 rounded-[1.5rem] border border-white/5 shadow-inner">
                                        <div className="flex items-center">
                                            <FilterSelect value={workflowRunFilter} onChange={setWorkflowRunFilter} options={[{ value: 'all', label: 'All History' }, ...workflowRuns.map(r => ({ value: r.workflow_run_id, label: formatWorkflowRunShortLabel(r) }))]} />
                                            <div className="w-px h-6 bg-white/10 mx-2" />
                                            <FilterSelect value={routeFilter} onChange={setRouteFilter} options={[{ value: 'all', label: 'All Routes' }, { value: 'article_ready', label: 'Articles' }, { value: 'software_ready', label: 'Software' }, { value: 'rejected_low_achievability', label: 'Rejected' }]} />
                                        </div>
                                        <div className="w-px h-6 bg-white/10 mx-1 hidden sm:block" />
                                        <div className="relative group flex-1 sm:flex-none">
                                            <Filter className="absolute left-4 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground group-focus-within:text-indigo-400 transition-colors" />
                                            <Input 
                                                value={candidateSearch} 
                                                onChange={(e) => setCandidateSearch(e.target.value)} 
                                                placeholder="Keyword search..." 
                                                className="h-11 min-w-[200px] w-full bg-transparent border-none text-[13px] font-medium focus-visible:ring-0 pl-11 pr-4 placeholder:text-slate-700" 
                                            />
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Opportunity Stream */}
                            <div className="p-8 lg:p-12 space-y-12 flex-1 relative z-10">
                                {loadingWorkflowArtifacts ? (
                                    <div className="py-52 flex flex-col items-center justify-center gap-6 text-muted-foreground">
                                        <div className="relative">
                                            <div className="absolute inset-0 animate-ping bg-indigo-500/20 rounded-full scale-150" />
                                            <div className="bg-indigo-500/10 p-6 rounded-full border border-indigo-500/20 relative z-10">
                                                <Loader2 className="h-12 w-12 animate-spin text-indigo-500" />
                                            </div>
                                        </div>
                                        <div className="text-center space-y-2">
                                            <p className="text-sm font-black uppercase tracking-[0.3em] text-foreground">Synthesizing Analysis</p>
                                            <p className="text-xs font-medium text-muted-foreground italic">Compiling live SERP snapshots and internal hooks...</p>
                                        </div>
                                    </div>
                                ) : filteredWorkflowResults.length === 0 ? (
                                    <div className="py-40 text-center bg-white/[0.01] border border-dashed border-border rounded-[4rem] group">
                                        <div className="bg-indigo-500/10 w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-8 group-hover:scale-110 transition-transform duration-700 border border-indigo-500/20">
                                            <Target className="h-10 w-10 text-indigo-400" />
                                        </div>
                                        <h3 className="text-foreground font-black text-2xl tracking-tight mb-4">Awaiting Intelligence</h3>
                                        <p className="text-muted-foreground text-[15px] font-medium max-w-sm mx-auto leading-relaxed mb-8">
                                            No opportunities found for the current filter. Initiate a <b>Validation Run</b> in Phase 02 to discover new outcomes.
                                        </p>
                                        <div className="flex justify-center gap-4">
                                            <div className="px-4 py-2 bg-white/5 rounded-xl border border-border text-[10px] font-black uppercase tracking-widest text-muted-foreground">
                                                Search Active: {candidateSearch || 'None'}
                                            </div>
                                            <div className="px-4 py-2 bg-white/5 rounded-xl border border-border text-[10px] font-black uppercase tracking-widest text-muted-foreground">
                                                Run: {workflowRunFilter === 'all' ? 'All History' : 'Selected Run'}
                                            </div>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="space-y-16">
                                        {filteredWorkflowResults.map((jobResult) => (
                                            <div key={jobResult.job_id} className="group/job animate-in fade-in slide-in-from-bottom-8 duration-700">
                                                {/* Job Context Header */}
                                                <div className="flex flex-col md:flex-row md:items-center gap-6 mb-10 pb-4 border-b border-white/5 relative">
                                                    <div className="flex items-center gap-4 relative z-10">
                                                        <div className="bg-indigo-500/10 text-indigo-400 h-10 px-4 rounded-xl flex items-center justify-center text-[10px] font-black tracking-widest border border-indigo-500/20 shadow-lg shadow-indigo-500/5 uppercase">
                                                            Job Node
                                                        </div>
                                                        <h3 className="text-lg font-black text-foreground tracking-tight leading-tight max-w-2xl">{jobResult.job?.job_text}</h3>
                                                    </div>
                                                    <div className="flex-1 h-px bg-white/5 hidden md:block" />
                                                    <Badge variant="outline" className="text-[10px] border-border text-muted-foreground uppercase tracking-widest px-3 py-1 self-start md:self-center">
                                                        {jobResult.candidates.length} Outcomes Identified
                                                    </Badge>
                                                    
                                                    {/* Flow Connector Line */}
                                                    <div className="absolute left-[3.5rem] top-12 bottom-[-2.5rem] w-px bg-gradient-to-b from-indigo-500/30 to-transparent hidden xl:block" />
                                                </div>

                                                {/* Candidate Cards Grid */}
                                                <div className="grid gap-8">
                                                    {jobResult.candidates.map((candidateData) => (
                                                        <OpportunityCard 
                                                            key={candidateData.candidate.id}
                                                            data={candidateData}
                                                            isExpanded={expandedCandidateIds.has(candidateData.candidate.id)}
                                                            onToggle={() => toggleCandidateExpansion(candidateData.candidate.id)}
                                                            onPersist={handlePersistOutcome}
                                                            onRelease={handleReleaseSoftware}
                                                            onReject={handleRejectCandidate}
                                                            onRefresh={handleRefreshValidation}
                                                            isMutating={mutatingOutcomeIds.has(candidateData.generated_outcome.id)}
                                                            isRejecting={rejectingCandidateIds.has(candidateData.candidate.id)}
                                                            isRefreshing={refreshingValidationIds.has(candidateData.validation_run.id)}
                                                        />
                                                    ))}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>

                            {/* Pagination Footer */}
                            {workflowTotalJobs > 0 && (
                                <div className="mt-auto bg-black/40 border-t border-white/5 p-8 relative z-10">
                                    <div className="mx-auto w-full flex items-center justify-between">
                                        <div className="text-[11px] text-muted-foreground font-black uppercase tracking-[0.2em]">
                                            Job View <span className="text-foreground bg-white/10 px-2 py-0.5 rounded ml-2">{workflowPage * workflowPageSize + 1}—{Math.min((workflowPage + 1) * workflowPageSize, workflowTotalJobs)}</span> <span className="mx-2 opacity-30">/</span> {workflowTotalJobs} Total
                                        </div>
                                        <div className="flex items-center gap-6">
                                            <div className="flex items-center gap-3">
                                                <span className="text-[10px] font-black text-muted-foreground uppercase tracking-widest">Page Size:</span>
                                                <select value={String(workflowPageSize)} onChange={(e) => { setWorkflowPageSize(Number(e.target.value)); setWorkflowPage(0); }} className="bg-white/5 border border-border rounded-xl text-[10px] font-black px-4 h-10 uppercase text-muted-foreground focus:ring-1 focus:ring-indigo-500/50 outline-none hover:bg-white/10 transition-all cursor-pointer">
                                                    <option value="5">05 / PAGE</option>
                                                    <option value="10">10 / PAGE</option>
                                                    <option value="25">25 / PAGE</option>
                                                </select>
                                            </div>
                                            <div className="flex gap-2">
                                                <Button 
                                                    variant="outline" 
                                                    size="icon" 
                                                    className="rounded-xl h-11 w-11 border-border bg-white/5 hover:bg-white/10 transition-all disabled:opacity-20" 
                                                    disabled={workflowPage === 0} 
                                                    onClick={() => setWorkflowPage(p => p - 1)}
                                                >
                                                    <ChevronLeft className="h-5 w-5" />
                                                </Button>
                                                <Button 
                                                    variant="outline" 
                                                    size="icon" 
                                                    className="rounded-xl h-11 w-11 border-border bg-white/5 hover:bg-white/10 transition-all disabled:opacity-20" 
                                                    disabled={(workflowPage + 1) * workflowPageSize >= workflowTotalJobs} 
                                                    onClick={() => setWorkflowPage(p => p + 1)}
                                                >
                                                    <ChevronRight className="h-5 w-5" />
                                                </Button>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </section>
                    </div>
                    )}
                </div>
            </main>

            {showSavedTopicsModal && (
                <div className="fixed inset-0 z-[120] flex items-center justify-center bg-black/70 px-4 py-6 backdrop-blur-sm">
                    <div className="w-full max-w-3xl rounded-[2rem] border border-border bg-card shadow-[0_30px_90px_rgba(0,0,0,0.55)]">
                        <div className="flex items-center justify-between border-b border-border px-6 py-5">
                            <div>
                                <h3 className="text-lg font-black text-foreground">Saved Topics</h3>
                                <p className="mt-1 text-sm text-muted-foreground">Review, approve, reject, or remove topics without crowding the main workflow.</p>
                            </div>
                            <button
                                type="button"
                                onClick={() => setShowSavedTopicsModal(false)}
                                className="rounded-xl border border-border bg-white/[0.03] p-2 text-muted-foreground transition hover:bg-white/[0.08] hover:text-foreground"
                            >
                                <XCircle className="h-4 w-4" />
                            </button>
                        </div>
                        <div className="max-h-[70vh] overflow-y-auto px-6 py-6">
                            {loadingJobs ? (
                                <div className="flex min-h-[240px] flex-col items-center justify-center gap-4">
                                    <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                                    <p className="text-sm text-muted-foreground">Loading saved topics...</p>
                                </div>
                            ) : jobs.length === 0 ? (
                                <div className="flex min-h-[240px] flex-col items-center justify-center gap-3 text-center">
                                    <Info className="h-8 w-8 text-muted-foreground" />
                                    <p className="text-sm font-semibold text-muted-foreground">No saved topics yet.</p>
                                    <p className="max-w-sm text-sm text-muted-foreground">Create a topic in Step 1 and it will appear here for later review and cleanup.</p>
                                </div>
                            ) : (
                                <div className="space-y-3">
                                    {jobs.map((job) => (
                                        <div key={job.id} className="rounded-2xl border border-border bg-card p-4">
                                            <div className="flex items-start justify-between gap-4">
                                                <div className="min-w-0 flex-1">
                                                    <div className="flex flex-wrap items-center gap-2">
                                                        {(job.primary_category_id || job.secondary_category_id) && (
                                                            <span className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
                                                                Scoped Topic
                                                            </span>
                                                        )}
                                                    </div>
                                                    <p className="mt-3 text-sm leading-relaxed text-foreground">{job.job_text}</p>
                                                </div>
                                                <div className="flex shrink-0 flex-wrap justify-end gap-2">
                                                    <button
                                                        type="button"
                                                        onClick={() => {
                                                            setSelectedLookupJobId(job.id)
                                                            setShowSavedTopicsModal(false)
                                                        }}
                                                        className="rounded-xl border border-emerald-500/20 bg-emerald-500/10 px-4 py-2.5 text-xs font-bold text-emerald-400 transition hover:bg-emerald-500/20 hover:text-emerald-300"
                                                    >
                                                        Use Topic
                                                    </button>
                                                    <button
                                                        type="button"
                                                        disabled={archivingJobIds.has(job.id)}
                                                        onClick={() => handleArchiveJob(job.id)}
                                                        className="rounded-xl border border-rose-500/20 bg-rose-500/10 px-4 py-2.5 text-xs font-bold text-rose-400 transition hover:bg-rose-500/20 hover:text-rose-300 disabled:opacity-50"
                                                    >
                                                        {archivingJobIds.has(job.id) ? (
                                                            <Loader2 className="h-4 w-4 animate-spin" />
                                                        ) : (
                                                            'Remove'
                                                        )}
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}

            {showEasyWinsModal && (
                <div className="fixed inset-0 z-[120] flex items-stretch justify-end bg-black/70 backdrop-blur-sm">
                    <div className="w-full max-w-xl border-l border-border bg-card shadow-[-30px_0_90px_rgba(0,0,0,0.55)] flex flex-col h-full animate-in slide-in-from-right duration-300">
                        <div className="flex items-center justify-between border-b border-border px-6 py-5 shrink-0 bg-card">
                            <div>
                                <h3 className="text-lg font-black text-emerald-400 flex items-center gap-2">
                                    <Sparkles className="h-5 w-5" />
                                    Easy-Wins Discovery
                                </h3>
                                <p className="mt-1 text-sm text-muted-foreground">Showing {activeSearchPreviewItems.length} highly viable keywords (KD &lt; 30, Vol &gt; 20).</p>
                            </div>
                            <button
                                type="button"
                                onClick={() => setShowEasyWinsModal(false)}
                                className="rounded-xl border border-border bg-white/[0.03] p-2 text-muted-foreground transition hover:bg-white/[0.08] hover:text-foreground"
                            >
                                <XCircle className="h-4 w-4" />
                            </button>
                        </div>
                        <div className="flex-1 overflow-y-auto px-6 py-6 bg-background">
                            <div className="space-y-3">
                                {activeSearchPreviewItems.map((item, index) => {
                                    const keyword = String((item as any).keyword || (item as any).title || (item as any).url || `Result ${index + 1}`)
                                    return (
                                        <div key={`modal-${activeSearchRecord?.id}-${index}`} className="rounded-2xl border border-emerald-500/20 bg-emerald-500/[0.02] px-4 py-4 transition hover:bg-emerald-500/[0.05]">
                                            <div className="flex items-start justify-between gap-3">
                                                <div className="min-w-0 flex-1">
                                                    <p className="text-[15px] font-semibold text-card-foreground">{keyword}</p>
                                                    <div className="mt-2 flex items-center gap-3">
                                                        {(item as any).keyword_difficulty != null && (
                                                            <span className="flex items-center gap-1.5 rounded-lg border border-emerald-500/30 bg-emerald-500/10 px-2 py-1 text-[10px] font-bold text-emerald-400">
                                                                <Target className="h-3 w-3" />
                                                                KD {(item as any).keyword_difficulty}
                                                            </span>
                                                        )}
                                                        {(item as any).search_volume != null && (
                                                            <span className="flex items-center gap-1.5 rounded-lg border border-blue-500/30 bg-blue-500/10 px-2 py-1 text-[10px] font-bold text-blue-400">
                                                                <Gauge className="h-3 w-3" />
                                                                VOL {(item as any).search_volume}
                                                            </span>
                                                        )}
                                                        {(item as any).cpc != null && (
                                                            <span className="flex items-center gap-1.5 rounded-lg border border-amber-500/30 bg-amber-500/10 px-2 py-1 text-[10px] font-bold text-amber-400">
                                                                CPC ${(item as any).cpc}
                                                            </span>
                                                        )}
                                                    </div>
                                                </div>
                                                <button
                                                    type="button"
                                                    onClick={() => {
                                                        handleUseKeywordForArticle(keyword)
                                                        setShowEasyWinsModal(false)
                                                    }}
                                                    className="rounded-xl border border-emerald-500/40 bg-emerald-500/20 px-3 py-2 text-[10px] font-black uppercase tracking-[0.12em] text-emerald-300 transition hover:bg-emerald-500/30 hover:text-emerald-200"
                                                >
                                                    Use
                                                </button>
                                            </div>
                                        </div>
                                    )
                                })}
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {showAllMetricsModal && activeSearchRecord && (
                <div className="fixed inset-0 z-[120] flex items-center justify-center bg-black/70 backdrop-blur-sm p-4 md:p-8">
                    <div className="w-full max-w-[1400px] h-full max-h-[90vh] flex flex-col rounded-[2rem] border border-border bg-card shadow-[0_30px_90px_rgba(0,0,0,0.55)] animate-in fade-in zoom-in-95 duration-300">
                        <div className="flex items-center justify-between border-b border-border px-8 py-6 shrink-0 bg-card rounded-t-[2rem]">
                            <div>
                                <h3 className="text-2xl font-black text-foreground flex items-center gap-3">
                                    <Layers className="h-6 w-6 text-indigo-400" />
                                    Full Keyword Metrics
                                </h3>
                                <p className="mt-1.5 text-sm text-muted-foreground">Showing {(activeSearchRecord.result_summary_json?.top_items as any[])?.length || 0} keywords for "{activeSearchRecord.query_text}".</p>
                            </div>
                            <button
                                type="button"
                                onClick={() => setShowAllMetricsModal(false)}
                                className="rounded-xl border border-border bg-white/[0.03] p-2.5 text-muted-foreground transition hover:bg-white/[0.08] hover:text-foreground"
                            >
                                <XCircle className="h-5 w-5" />
                            </button>
                        </div>
                        <div className="flex-1 overflow-auto bg-background p-6 rounded-b-[2rem]">
                            <div className="min-w-[800px]">
                                <table className="w-full text-left border-collapse">
                                    <thead>
                                        <tr className="border-b border-border">
                                            <th className="py-3 px-4 text-xs font-black uppercase tracking-wider text-muted-foreground">Keyword</th>
                                            <th className="py-3 px-4 text-xs font-black uppercase tracking-wider text-muted-foreground text-right">Volume</th>
                                            <th className="py-3 px-4 text-xs font-black uppercase tracking-wider text-muted-foreground text-right">KD</th>
                                            <th className="py-3 px-4 text-xs font-black uppercase tracking-wider text-muted-foreground text-right">CPC</th>
                                            <th className="py-3 px-4 text-xs font-black uppercase tracking-wider text-muted-foreground">Intent</th>
                                            <th className="py-3 px-4 text-xs font-black uppercase tracking-wider text-muted-foreground text-center">Action</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-border">
                                        {((activeSearchRecord.result_summary_json?.top_items as any[]) || []).map((item, index) => {
                                            const keyword = String(item.keyword || item.title || item.url || `Result ${index + 1}`)
                                            const isEasyWin = (item.keyword_difficulty || 100) < 30 && (item.search_volume || 0) > 20
                                            return (
                                                <tr key={`full-metric-${index}`} className="hover:bg-muted/50 transition-colors">
                                                    <td className="py-3 px-4">
                                                        <div className="flex items-center gap-2">
                                                            <span className="font-medium text-foreground">{keyword}</span>
                                                            {isEasyWin && (
                                                                <span className="bg-emerald-500/20 text-emerald-400 text-[9px] font-black uppercase px-1.5 py-0.5 rounded">Easy Win</span>
                                                            )}
                                                        </div>
                                                    </td>
                                                    <td className="py-3 px-4 text-right font-mono text-sm text-muted-foreground">{item.search_volume ?? '-'}</td>
                                                    <td className="py-3 px-4 text-right">
                                                        {item.keyword_difficulty != null ? (
                                                            <span className={`inline-flex items-center justify-center rounded-md px-2 py-0.5 text-xs font-bold ${item.keyword_difficulty < 30 ? 'bg-emerald-500/20 text-emerald-400' : 'bg-muted text-muted-foreground'}`}>
                                                                {item.keyword_difficulty}
                                                            </span>
                                                        ) : '-'}
                                                    </td>
                                                    <td className="py-3 px-4 text-right font-mono text-sm text-muted-foreground">{item.cpc ? `$${item.cpc}` : '-'}</td>
                                                    <td className="py-3 px-4 text-xs text-muted-foreground">{(item.intent || '-').substring(0, 20)}</td>
                                                    <td className="py-3 px-4 text-center">
                                                        <button
                                                            onClick={() => {
                                                                handleUseKeywordForArticle(keyword)
                                                                setShowAllMetricsModal(false)
                                                            }}
                                                            className="rounded-lg border border-border bg-card px-3 py-1.5 text-[10px] font-black uppercase text-foreground hover:bg-muted hover:border-muted-foreground transition"
                                                        >
                                                            Use
                                                        </button>
                                                    </td>
                                                </tr>
                                            )
                                        })}
                                        {((activeSearchRecord.result_summary_json?.top_items as any[]) || []).length === 0 && (
                                            <tr>
                                                <td colSpan={6} className="py-12 text-center text-muted-foreground">No keyword data available for this lookup.</td>
                                            </tr>
                                        )}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

// --- SUB-COMPONENTS ---

function FlowStep({ number, label, active, icon, onClick, disabled }: { number: string; label: string; active: boolean; icon: React.ReactNode; onClick?: () => void; disabled?: boolean }) {
    return (
        <button 
            type="button"
            onClick={onClick}
            disabled={disabled}
            className={cn(
                "flex items-center gap-3 transition-all duration-500 text-left outline-none", 
                active ? "opacity-100 translate-y-0" : "opacity-50 translate-y-1",
                onClick && !disabled ? "cursor-pointer hover:opacity-80 hover:-translate-y-0.5" : "",
                disabled ? "cursor-not-allowed opacity-40" : ""
            )}
        >
            <div className={cn(
                "w-8 h-8 rounded-xl flex items-center justify-center text-xs font-black border transition-all duration-500", 
                active 
                    ? "bg-indigo-500 border-indigo-400 text-white shadow-[0_0_20px_rgba(99,102,241,0.5)] rotate-0" 
                    : "bg-black/5 dark:bg-white/5 border-border text-slate-500 dark:text-muted-foreground rotate-[-5deg]"
            )}>
                {number}
            </div>
            <div className="flex flex-col">
                <span className={cn("text-[11px] font-black uppercase tracking-[0.15em] leading-none mb-1", active ? "text-foreground" : "text-slate-500 dark:text-muted-foreground")}>{label}</span>
                <span className={cn("flex items-center gap-1.5 text-[9px] font-bold uppercase tracking-widest", active ? "text-indigo-500 dark:text-indigo-400" : "text-slate-400 dark:text-slate-600")}>
                    {icon} 
                    {active ? 'Current Phase' : 'Pipeline'}
                </span>
            </div>
        </button>
    )
}

function SelectField({ label, value, onChange, options }: { label: string; value: string; onChange: (v: string) => void; options: { value: string; label: string }[] }) {
    return (
        <div className="space-y-2.5">
            <label className="pl-1 text-[10px] font-black uppercase tracking-[0.2em] text-muted-foreground">{label}</label>
            <div className="relative group">
                <select 
                    value={value} 
                    onChange={(e) => onChange(e.target.value)} 
                    className="h-12 w-full cursor-pointer appearance-none rounded-2xl border border-border bg-card px-4 text-xs font-medium text-foreground outline-none transition-all hover:bg-white/[0.06] focus:ring-2 focus:ring-indigo-500/40"
                >
                    {options.map(opt => <option key={opt.value} value={opt.value} className="bg-card py-3">{opt.label}</option>)}
                </select>
                <div className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 text-muted-foreground transition-colors group-hover:text-card-foreground">
                    <ChevronDown className="h-4 w-4" />
                </div>
            </div>
        </div>
    )
}

function FilterSelect({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: { value: string; label: string }[] }) {
    return (
        <div className="relative group">
            <select 
                value={value} 
                onChange={(e) => onChange(e.target.value)} 
                className="bg-transparent border-none text-[11px] font-black uppercase tracking-[0.2em] text-muted-foreground focus:ring-0 px-4 h-9 outline-none hover:text-foreground transition-all cursor-pointer appearance-none"
            >
                {options.map(opt => <option key={opt.value} value={opt.value} className="bg-card">{opt.label}</option>)}
            </select>
        </div>
    )
}

const ROUTE_CONFIG: Record<string, { label: string; color: string; accent: string; icon: any; description: string }> = {
    article_ready: { 
        label: 'Article Ready', 
        color: 'blue', 
        accent: '#3b82f6', 
        icon: FileText,
        description: 'Validated for high-ranking search potential.'
    },
    software_ready: { 
        label: 'Software Intent', 
        color: 'violet', 
        accent: '#8b5cf6', 
        icon: Wrench,
        description: 'Repeatable workflow pattern identified.'
    },
    editorial_only: { 
        label: 'Editorial', 
        color: 'amber', 
        accent: '#f59e0b', 
        icon: Sparkles,
        description: 'High strategic value, low search volume.'
    },
    article_plus_software: { 
        label: 'Hybrid Opportunity', 
        color: 'indigo', 
        accent: '#6366f1', 
        icon: Rocket,
        description: 'Multi-path potential for growth.'
    },
    needs_more_keyword_validation: { 
        label: 'Manual Review', 
        color: 'orange', 
        accent: '#f97316', 
        icon: Target,
        description: 'Data signals are mixed. Review manually.'
    },
    rejected_low_achievability: { 
        label: 'Low Potential', 
        color: 'slate', 
        accent: '#475569', 
        icon: Ban,
        description: 'Unwinnable SERP or off-brand intent.'
    },
    software_backlog_low_feasibility: { 
        label: 'Feasibility Backlog', 
        color: 'slate', 
        accent: '#475569', 
        icon: Layers,
        description: 'Strong intent but high build complexity.'
    }
}

function OpportunityCard({ data, isExpanded, onToggle, onPersist, onRelease, onReject, onRefresh, isMutating, isRejecting, isRefreshing }: any) {
    const { candidate, validation_run, routing_decision, keyword_pack, internal_link_candidates, generated_outcome } = data
    const route = routing_decision.route as string
    const config = ROUTE_CONFIG[route] || { label: route, color: 'slate', accent: '#475569', icon: Info, description: 'Analysis outcome pending.' }
    const Icon = config.icon

    const colorClasses: Record<string, string> = {
        blue: 'border-blue-500/30 bg-blue-500/10 text-blue-400 shadow-blue-500/10',
        violet: 'border-violet-500/30 bg-violet-500/10 text-violet-400 shadow-violet-500/10',
        amber: 'border-amber-500/30 bg-amber-500/10 text-amber-400 shadow-amber-500/10',
        indigo: 'border-indigo-500/30 bg-indigo-500/10 text-indigo-400 shadow-indigo-500/10',
        orange: 'border-orange-500/30 bg-orange-500/10 text-orange-400 shadow-orange-500/10',
        slate: 'border-border bg-white/5 text-muted-foreground'
    }

    const achievability = validation_run.achievability_score || 0

    return (
        <div 
            className={cn(
                "rounded-[2.5rem] border transition-all duration-700 relative overflow-hidden group", 
                isExpanded 
                    ? "bg-white/[0.05] border-border ring-1 ring-white/5 shadow-2xl" 
                    : "bg-white/[0.02] border-white/5 hover:border-border hover:bg-white/[0.03] hover:shadow-xl"
            )}
        >
            {/* Decision Stamp Watermark */}
            <div className="absolute right-10 top-1/2 -translate-y-1/2 opacity-[0.03] select-none pointer-events-none">
                <span className="text-8xl font-black uppercase tracking-tighter italic">
                    {route === 'software_ready' ? 'Software' : route === 'article_ready' ? 'Article' : 'Outcome'}
                </span>
            </div>

            {/* Status Accent Bar */}
            <div 
                className="absolute left-0 top-0 bottom-0 w-1.5 transition-all duration-500 z-10" 
                style={{ backgroundColor: config.accent, opacity: isExpanded ? 1 : 0.4 }} 
            />

            <div className="p-8">
                <div className="flex flex-col xl:flex-row xl:items-start justify-between gap-8">
                    <div className="flex-1 space-y-6">
                        <div className="flex flex-wrap items-center gap-3">
                            <div className={cn("px-4 py-1.5 rounded-xl border text-[10px] font-black uppercase tracking-widest flex items-center gap-2.5 shadow-xl transition-all duration-500", colorClasses[config.color])}>
                                <Icon className="h-3.5 w-3.5" />
                                {config.label}
                            </div>
                            <div className="flex items-center gap-2 px-3 py-1.5 bg-black/40 rounded-xl border border-white/5">
                                <Gauge className="h-3.5 w-3.5 text-indigo-500" />
                                <span className="text-[10px] font-black text-muted-foreground uppercase tracking-[0.2em]">Efficiency: {Math.round(achievability * 100)}%</span>
                            </div>
                            {validation_run.freshness_state !== 'fresh' && (
                                <Badge variant="outline" className="text-[9px] border-amber-500/20 text-amber-500 bg-amber-500/5 uppercase font-black tracking-widest px-2.5 py-1">Stale Data</Badge>
                            )}
                        </div>

                        <div>
                            <h4 className="text-2xl font-black text-foreground tracking-tight leading-tight group-hover:text-indigo-400 transition-colors duration-500 mb-3">{candidate.candidate_text}</h4>
                            <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
                                <div className="flex items-center gap-2.5 text-xs font-bold">
                                    <span className="text-muted-foreground uppercase tracking-widest text-[9px] font-black">Core Keyword:</span>
                                    <span className="text-indigo-400 bg-indigo-500/10 px-3 py-1 rounded-lg border border-indigo-500/20">{keyword_pack.primary_keyword || 'Intent Discovery'}</span>
                                </div>
                                <div className="text-[12px] text-muted-foreground font-medium">{config.description}</div>
                            </div>
                        </div>
                    </div>

                    <div className="flex items-center gap-3 self-start xl:self-center bg-black/40 p-2 rounded-2xl border border-white/5 shadow-inner">
                        {route !== 'rejected_low_achievability' && (
                             <>
                                {generated_outcome.outcome_type === 'software' ? (
                                    <Button 
                                        size="sm" 
                                        className="bg-indigo-500 hover:bg-indigo-400 text-foreground rounded-xl h-11 px-6 font-black text-[11px] uppercase tracking-widest transition-all active:scale-[0.95] shadow-lg shadow-indigo-500/20 group/btn" 
                                        onClick={() => onRelease(generated_outcome.id)} 
                                        disabled={isMutating}
                                    >
                                        {isMutating ? <Loader2 className="h-4 w-4 animate-spin mr-3" /> : <Rocket className="h-4 w-4 mr-3 group-hover/btn:translate-y-[-2px] transition-transform" />}
                                        Release Tool
                                    </Button>
                                ) : (
                                    <Button 
                                        size="sm" 
                                        className="bg-primary text-primary-foreground hover:bg-indigo-50 rounded-xl h-11 px-6 font-black text-[11px] uppercase tracking-widest transition-all active:scale-[0.95] shadow-xl group/btn" 
                                        onClick={() => onPersist(generated_outcome.id)} 
                                        disabled={isMutating}
                                    >
                                        {isMutating ? <Loader2 className="h-4 w-4 animate-spin mr-3" /> : <Sparkles className="h-4 w-4 mr-3 text-indigo-600 group-hover/btn:scale-110 transition-transform" />}
                                        Send to Content Studio
                                    </Button>
                                )}
                                <div className="w-px h-8 bg-white/10 mx-1" />
                                <button 
                                    onClick={() => onReject(candidate.id, 'weak_serp')} 
                                    disabled={isRejecting} 
                                    className="h-11 w-11 flex items-center justify-center rounded-xl bg-red-500/10 text-red-500 border border-red-500/20 hover:bg-red-500 hover:text-foreground transition-all shadow-xl shadow-red-500/10"
                                    title="Reject Analysis"
                                >
                                    {isRejecting ? <Loader2 className="h-5 w-5 animate-spin" /> : <Ban className="h-5 w-5" />}
                                </button>
                             </>
                        )}
                        <button 
                            onClick={() => onRefresh(validation_run.id)} 
                            disabled={isRefreshing} 
                            className="h-11 w-11 flex items-center justify-center rounded-xl bg-white/5 text-muted-foreground border border-border hover:bg-white/10 hover:text-foreground transition-all"
                            title="Refresh Validation"
                        >
                            {isRefreshing ? <Loader2 className="h-5 w-5 animate-spin" /> : <RefreshCw className="h-5 w-5" />}
                        </button>
                        <button 
                            onClick={onToggle} 
                            className={cn(
                                "h-11 px-4 flex items-center justify-center gap-2 rounded-xl border transition-all text-[11px] font-black uppercase tracking-widest",
                                isExpanded ? "bg-primary text-primary-foreground border-white shadow-xl" : "bg-white/5 text-muted-foreground border-border hover:bg-white/10"
                            )}
                        >
                            {isExpanded ? 'Details' : 'Inspect'}
                            {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                        </button>
                    </div>
                </div>

                {/* Achievability Visual Meter */}
                <div className="mt-10 flex flex-col md:flex-row md:items-center gap-8">
                    <div className="flex-1">
                        <div className="flex items-center justify-between mb-3 px-1">
                            <span className="text-[10px] font-black text-muted-foreground uppercase tracking-[0.25em]">Achievability Gradient</span>
                            <span className={cn(
                                "text-[11px] font-black uppercase tracking-widest",
                                achievability > 0.6 ? "text-emerald-400" : achievability > 0.3 ? "text-amber-400" : "text-red-400"
                            )}>
                                {achievability > 0.6 ? 'High Probability' : achievability > 0.3 ? 'Moderate Effort' : 'Low Win Probability'}
                            </span>
                        </div>
                        <div className="h-2.5 bg-white/5 rounded-full overflow-hidden border border-white/5 p-0.5 shadow-inner">
                            <div 
                                className={cn("h-full rounded-full transition-all duration-[1.5s] ease-out", 
                                    achievability > 0.6 ? "bg-gradient-to-r from-emerald-600 to-emerald-400 shadow-[0_0_20px_rgba(16,185,129,0.4)]" : 
                                    achievability > 0.3 ? "bg-gradient-to-r from-amber-600 to-amber-400" : "bg-gradient-to-r from-red-600 to-red-400"
                                )}
                                style={{ width: `${Math.max(4, Math.min(100, achievability * 100))}%` }}
                            />
                        </div>
                    </div>
                    <div className="grid grid-cols-4 gap-12 shrink-0 border-l border-border pl-12">
                        <ScoreMini label="Intent" value={validation_run.intent_match_score} color="indigo" />
                        <ScoreMini label="SERP" value={validation_run.serp_weakness_score} color="emerald" />
                        <ScoreMini label="Gap" value={validation_run.serp_gap_score} color="violet" />
                        <ScoreMini label="Ease" value={getValidationMetric(data.validation_run, 'keyword_difficulty') !== undefined ? 1 - (Number(getValidationMetric(data.validation_run, 'keyword_difficulty')) / 100) : 0} color="amber" />
                    </div>
                </div>
            </div>

            {isExpanded && (
                <div className="px-8 pb-12 pt-6 border-t border-border animate-in fade-in slide-in-from-top-4 duration-500 bg-white/[0.02]">
                    <div className="grid lg:grid-cols-2 gap-16 mt-8">
                        {/* SERP Evidence */}
                        <div className="space-y-8">
                            <div className="flex items-center justify-between">
                                <h5 className="text-[11px] font-black uppercase tracking-[0.3em] text-muted-foreground flex items-center gap-3">
                                    <Globe className="h-4 w-4 text-indigo-500" /> SERP Competitive Evidence
                                </h5>
                                <span className="text-[9px] font-black text-slate-700 uppercase tracking-widest">Top 4 Results</span>
                            </div>
                            <div className="space-y-4">
                                {getSerpRows(validation_run).length > 0 ? (
                                    getSerpRows(validation_run).slice(0, 4).map((row: any, i: number) => (
                                        <div key={i} className="group/row bg-black/40 border border-white/5 rounded-2xl p-5 hover:bg-black/60 hover:border-indigo-500/30 transition-all duration-300 shadow-sm relative overflow-hidden">
                                            <div className="absolute top-0 right-0 p-3 opacity-5 group-hover/row:opacity-10 transition-opacity">
                                                <ExternalLink className="h-12 w-12 text-foreground" />
                                            </div>
                                            <div className="flex items-start justify-between gap-4 mb-3">
                                                <div className="flex items-start gap-4">
                                                    <div className="mt-0.5 w-6 h-6 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-[11px] font-black text-indigo-400 group-hover/row:bg-indigo-500 group-hover/row:text-foreground transition-all shadow-lg">{i + 1}</div>
                                                    <p className="text-[14px] font-black text-card-foreground group-hover/row:text-foreground transition-colors leading-snug">{row.title || 'Competitor Snapshot'}</p>
                                                </div>
                                                <a href={row.url} target="_blank" rel="noreferrer" className="shrink-0 text-muted-foreground hover:text-indigo-400 p-2 bg-white/5 rounded-xl transition-all border border-white/5 hover:border-indigo-500/20">
                                                    <ExternalLink className="h-4 w-4" />
                                                </a>
                                            </div>
                                            <div className="flex items-center gap-3 pl-10">
                                                <img 
                                                    src={`https://www.google.com/s2/favicons?domain=${new URL(row.url).hostname}&sz=32`} 
                                                    alt="icon" 
                                                    className="w-4 h-4 rounded shadow-sm opacity-60 grayscale group-hover/row:grayscale-0 group-hover/row:opacity-100 transition-all"
                                                />
                                                <p className="text-[10px] text-muted-foreground font-bold uppercase tracking-widest">{new URL(row.url).hostname}</p>
                                            </div>
                                        </div>
                                    ))
                                ) : (
                                    <div className="py-20 flex flex-col items-center justify-center bg-white/[0.01] border border-dashed border-border rounded-[2.5rem]">
                                        <Ban className="h-10 w-10 text-slate-800 mb-4" />
                                        <p className="text-[10px] font-black uppercase tracking-widest text-slate-700">No SERP data collected</p>
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Internal Hooks & Analysis Details */}
                        <div className="space-y-12">
                            {/* Existing content matches */}
                            <div className="space-y-8">
                                <h5 className="text-[11px] font-black uppercase tracking-[0.3em] text-muted-foreground flex items-center gap-3">
                                    <Layers className="h-4 w-4 text-emerald-500" /> Existing Content Matches
                                </h5>
                                <div className="space-y-4">
                                    {(internal_link_candidates || []).length > 0 ? (
                                        internal_link_candidates.slice(0, 3).map((link: any) => {
                                            const meta = getInternalLinkMetadata(link)
                                            return (
                                                <div key={link.id} className="bg-black/40 border border-white/5 rounded-2xl p-5 hover:bg-black/60 hover:border-emerald-500/30 transition-all shadow-sm">
                                                    <p className="text-[14px] font-black text-card-foreground mb-4 leading-tight">{String(meta.matched_title || 'Target Neighborhood')}</p>
                                                    <div className="flex items-center justify-between">
                                                        <Badge variant="outline" className="text-[10px] font-black uppercase text-emerald-400 bg-emerald-500/10 border-emerald-500/20 tracking-widest px-3">
                                                            {link.link_role.replace('_', ' ')}
                                                        </Badge>
                                                        <div className="flex items-center gap-4">
                                                            <div className="w-24 h-2 bg-white/5 rounded-full overflow-hidden border border-white/5 p-px">
                                                                <div className="h-full bg-emerald-500 rounded-full shadow-[0_0_10px_rgba(16,185,129,0.5)]" style={{ width: `${(link.match_score || 0) * 100}%` }} />
                                                            </div>
                                                            <span className="text-[11px] text-muted-foreground font-black uppercase tabular-nums">{Math.round((link.match_score || 0) * 100)}% Match</span>
                                                        </div>
                                                    </div>
                                                </div>
                                            )
                                        })
                                    ) : (
                                        <div className="py-20 flex flex-col items-center justify-center bg-white/[0.01] border border-dashed border-border rounded-[2.5rem]">
                                            <div className="bg-white/5 p-4 rounded-full mb-4">
                                                <Layers className="h-8 w-8 text-slate-800" />
                                            </div>
                                            <p className="text-[10px] font-black uppercase tracking-widest text-slate-700 text-center px-10">No strong existing-content matches were detected for internal linking.</p>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Secondary Keywords */}
                            <div className="space-y-6">
                                <h5 className="text-[11px] font-black uppercase tracking-[0.3em] text-muted-foreground flex items-center gap-3">
                                    <Target className="h-4 w-4 text-indigo-400" /> Semantic Expansion
                                </h5>
                                <div className="flex flex-wrap gap-2.5">
                                    {(keyword_pack.secondary_keywords || []).map((kw: any, i: number) => (
                                        <div key={i} className="px-4 py-2 bg-white/5 border border-border rounded-xl text-[11px] font-bold text-muted-foreground hover:text-foreground hover:bg-white/10 transition-all cursor-default">
                                            {typeof kw === 'string' ? kw : kw.keyword}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    {/* Reason Codes & Tags */}
                    <div className="mt-12 pt-8 border-t border-border flex flex-col md:flex-row md:items-center justify-between gap-6">
                        <div className="flex flex-wrap items-center gap-3">
                            <span className="text-[10px] font-black uppercase tracking-[0.3em] text-muted-foreground mr-2">Analysis Intelligence Tags:</span>
                            {(validation_run.validation_reason_codes || []).map((code: string) => (
                                <span key={code} className="px-3 py-1.5 bg-white/5 border border-border rounded-xl text-[10px] font-black uppercase tracking-widest text-muted-foreground group-hover:text-indigo-400 transition-colors">
                                    {code.replace('_', ' ')}
                                </span>
                            ))}
                        </div>
                        <div className="text-[10px] font-bold text-slate-700 uppercase tracking-widest italic">
                            Validated via {keyword_pack.secondary_keywords?.length || 0} Secondary Vectors
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

function ScoreMini({ label, value, color = 'indigo' }: { label: string; value?: number | null; color?: string }) {
    const percentage = Math.max(4, Math.min(100, (value || 0) * 100))
    const colorMap: Record<string, string> = {
        indigo: 'bg-indigo-500',
        emerald: 'bg-emerald-500',
        violet: 'bg-violet-500',
        amber: 'bg-amber-500'
    }
    
    return (
        <div className="flex flex-col gap-2.5 min-w-[60px]">
            <span className="text-[9px] font-black uppercase tracking-[0.2em] text-muted-foreground leading-none">{label}</span>
            <div className="space-y-2">
                <span className="text-[13px] font-black text-foreground tabular-nums">{(value || 0).toFixed(2)}</span>
                <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
                    <div 
                        className={cn("h-full rounded-full transition-all duration-1000", colorMap[color])}
                        style={{ width: `${percentage}%` }}
                    />
                </div>
            </div>
        </div>
    )
}

function ContextStat({ label, value }: { label: string; value: string }) {
    return (
        <div className="rounded-2xl border border-white/5 bg-black/30 p-4">
            <div className="text-[10px] font-black uppercase tracking-[0.25em] text-muted-foreground">{label}</div>
            <div className="mt-2 text-sm font-semibold text-card-foreground">{value}</div>
        </div>
    )
}

function ContextBlock({ label, value }: { label: string; value: string }) {
    return (
        <div className="rounded-2xl border border-white/5 bg-black/30 p-4">
            <div className="text-[10px] font-black uppercase tracking-[0.25em] text-muted-foreground">{label}</div>
            <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{value}</p>
        </div>
    )
}

function getValidationMetric(validationRun: ResearchRebuildValidationRun, key: string): any {
    const metadata = validationRun.validation_metadata as Record<string, any>
    return metadata?.[key]
}

function getSerpRows(validationRun: ResearchRebuildValidationRun): any[] {
    const metadata = validationRun.validation_metadata as Record<string, any>
    return metadata?.serp_rows || []
}

function getInternalLinkMetadata(link: ResearchRebuildInternalLinkCandidate): any {
    return link.match_metadata || {}
}



function formatWorkflowRunShortLabel(run: ResearchRebuildWorkflowRunSummary): string {
    const date = run.started_at ? new Date(run.started_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) : '??'
    return `${date} • ${run.candidate_count} Targets`
}

export function ResearchRebuildJobsPage() {
    return <ResearchRebuild mode="jobs" />
}

export function ResearchRebuildOpportunitiesPage() {
    return <ResearchRebuild mode="opportunities" />
}

export default ResearchRebuild
