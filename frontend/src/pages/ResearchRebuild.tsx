import * as React from 'react'
import { Loader2, Rocket, Wrench, CheckCircle2, Sparkles, BookOpen, ChevronDown, ChevronUp, XCircle, RefreshCw, Ban, Copy, ArrowUpRight } from 'lucide-react'

import { useAuth } from '@/context/auth-context'
import { useProject } from '@/context/project-context'
import { supabase } from '@/lib/supabase'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { researchRebuildService } from '@/services/research-rebuild.service'
import { useNavigate, useSearchParams } from 'react-router-dom'
import type {
    ResearchRebuildInternalLinkCandidate,
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

export function ResearchRebuild() {
    const { user } = useAuth()
    const { activeProject, projects, setActiveProject } = useProject()
    const navigate = useNavigate()
    const [searchParams] = useSearchParams()
    const [jobs, setJobs] = React.useState<ResearchRebuildJob[]>([])
    const [workflowResults, setWorkflowResults] = React.useState<ResearchRebuildWorkflowJobResult[]>([])
    const [projectCategories, setProjectCategories] = React.useState<ProjectCategory[]>([])
    const [primaryCategoryId, setPrimaryCategoryId] = React.useState('')
    const [secondaryCategoryId, setSecondaryCategoryId] = React.useState('')
    const [workflowPage, setWorkflowPage] = React.useState(0)
    const [workflowPageSize, setWorkflowPageSize] = React.useState(10)
    const [workflowTotalJobs, setWorkflowTotalJobs] = React.useState(0)
    const [jobStatusFilter, setJobStatusFilter] = React.useState('all')
    const [workflowRuns, setWorkflowRuns] = React.useState<ResearchRebuildWorkflowRunSummary[]>([])
    const [routeFilter, setRouteFilter] = React.useState('all')
    const [candidateTypeFilter, setCandidateTypeFilter] = React.useState('all')
    const [outcomeTypeFilter, setOutcomeTypeFilter] = React.useState('all')
    const [candidateSearch, setCandidateSearch] = React.useState('')
    const [manualJobText, setManualJobText] = React.useState('')
    const [loadingJobs, setLoadingJobs] = React.useState(false)
    const [loadingWorkflowArtifacts, setLoadingWorkflowArtifacts] = React.useState(false)
    const [generatingJobs, setGeneratingJobs] = React.useState(false)
    const [runningWorkflow, setRunningWorkflow] = React.useState(false)
    const [mutatingOutcomeIds, setMutatingOutcomeIds] = React.useState<Set<string>>(new Set())
    const [expandedCandidateIds, setExpandedCandidateIds] = React.useState<Set<string>>(new Set())
    const [rejectingJobIds, setRejectingJobIds] = React.useState<Set<string>>(new Set())
    const [rejectingCandidateIds, setRejectingCandidateIds] = React.useState<Set<string>>(new Set())
    const [refreshingValidationIds, setRefreshingValidationIds] = React.useState<Set<string>>(new Set())
    const [refreshingAllStale, setRefreshingAllStale] = React.useState(false)
    const [error, setError] = React.useState<string | null>(null)
    const [success, setSuccess] = React.useState<string | null>(null)

    const incomingProjectId = searchParams.get('project_id') || ''
    const incomingPrimaryCategoryId = searchParams.get('primary_category_id') || ''
    const incomingSecondaryCategoryId = searchParams.get('secondary_category_id') || ''
    const incomingWorkflowRunId = searchParams.get('workflow_run_id') || ''
    const [workflowRunFilter, setWorkflowRunFilter] = React.useState(incomingWorkflowRunId || 'all')

    const primaryCategories = React.useMemo(
        () => projectCategories.filter((category) => category.level === 1),
        [projectCategories],
    )

    const secondaryCategories = React.useMemo(
        () => projectCategories.filter((category) => category.parent_category_id === primaryCategoryId),
        [projectCategories, primaryCategoryId],
    )

    const approvedJobs = React.useMemo(
        () => jobs.filter((job) => job.status === 'approved'),
        [jobs],
    )

    const jobStatusCounts = React.useMemo(() => {
        const counts = new Map<string, number>()
        for (const job of jobs) {
            counts.set(job.status, (counts.get(job.status) || 0) + 1)
        }
        return counts
    }, [jobs])

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

    const routeCounts = React.useMemo(() => {
        const counts = new Map<string, number>()
        for (const jobResult of filteredWorkflowResults) {
            for (const candidateResult of jobResult.candidates) {
                const route = candidateResult.routing_decision.route
                counts.set(route, (counts.get(route) || 0) + 1)
            }
        }
        return counts
    }, [filteredWorkflowResults])

    const selectedWorkflowRun = React.useMemo(
        () => workflowRuns.find((run) => run.workflow_run_id === workflowRunFilter) || null,
        [workflowRunFilter, workflowRuns],
    )

    const latestWorkflowRun = React.useMemo(
        () => workflowRuns[0] || null,
        [workflowRuns],
    )

    const comparisonWorkflowRun = React.useMemo(() => {
        if (!selectedWorkflowRun || !latestWorkflowRun) return null
        if (selectedWorkflowRun.workflow_run_id === latestWorkflowRun.workflow_run_id) return null
        return latestWorkflowRun
    }, [latestWorkflowRun, selectedWorkflowRun])

    const currentViewUrl = React.useMemo(() => {
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

        const query = params.toString()
        const path = `/research-rebuild${query ? `?${query}` : ''}`
        if (typeof window === 'undefined') {
            return path
        }
        return `${window.location.origin}${path}`
    }, [activeProject?.id, primaryCategoryId, secondaryCategoryId, workflowRunFilter])

    const visibleJobIds = React.useMemo(
        () => filteredWorkflowResults.map((jobResult) => jobResult.job_id),
        [filteredWorkflowResults],
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
    }, [incomingPrimaryCategoryId, incomingSecondaryCategoryId, incomingWorkflowRunId])

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

        const nextQuery = params.toString()
        const currentQuery = searchParams.toString()
        if (nextQuery === currentQuery) {
            return
        }

        navigate(
            {
                pathname: '/research-rebuild',
                search: nextQuery ? `?${nextQuery}` : '',
            },
            { replace: true },
        )
    }, [activeProject?.id, navigate, primaryCategoryId, searchParams, secondaryCategoryId, workflowRunFilter])

    React.useEffect(() => {
        setWorkflowPage(0)
    }, [activeProject?.id, primaryCategoryId, secondaryCategoryId, jobStatusFilter, workflowRunFilter, routeFilter, candidateTypeFilter, outcomeTypeFilter, candidateSearch])

    React.useEffect(() => {
        if (workflowRunFilter === 'all') return
        if (workflowRuns.some((run) => run.workflow_run_id === workflowRunFilter)) return
        setWorkflowRunFilter('all')
    }, [workflowRunFilter, workflowRuns])

    const refreshPageContext = React.useCallback(async (options?: {
        workflowRunId?: string
        workflowPage?: number
    }) => {
        if (!activeProject?.id) return

        const nextWorkflowRunFilter = options?.workflowRunId ?? workflowRunFilter
        const nextWorkflowPage = options?.workflowPage ?? workflowPage

        try {
            setLoadingJobs(true)
            setLoadingWorkflowArtifacts(true)
            const response = await researchRebuildService.getPageContext({
                project_id: activeProject.id,
                primary_category_id: primaryCategoryId || undefined,
                secondary_category_id: secondaryCategoryId || undefined,
                job_status: jobStatusFilter !== 'all' ? jobStatusFilter : undefined,
                workflow_run_id: nextWorkflowRunFilter !== 'all' ? nextWorkflowRunFilter : undefined,
                route: routeFilter !== 'all' ? routeFilter : undefined,
                candidate_type: candidateTypeFilter !== 'all' ? candidateTypeFilter : undefined,
                outcome_type: outcomeTypeFilter !== 'all' ? outcomeTypeFilter : undefined,
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
    }, [activeProject?.id, primaryCategoryId, secondaryCategoryId, jobStatusFilter, workflowRunFilter, routeFilter, candidateTypeFilter, outcomeTypeFilter, candidateSearch, workflowPage, workflowPageSize])

    React.useEffect(() => {
        void refreshPageContext()
    }, [refreshPageContext])

    const primaryCategory = primaryCategories.find((category) => category.id === primaryCategoryId)
    const secondaryCategory = secondaryCategories.find((category) => category.id === secondaryCategoryId)

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
                count: 12,
            })
            setSuccess(`Generated ${response.count} jobs.`)
            await refreshPageContext()
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
            setError(null)
            setSuccess(null)
            const created = await researchRebuildService.createJob({
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
            setSuccess('Manual job created.')
            if (created?.id) {
                await refreshPageContext()
            }
        } catch (err) {
            console.error('Failed to create manual job:', err)
            setError('Failed to create manual job.')
        }
    }

    const handleApproveJob = async (jobId: string) => {
        try {
            await researchRebuildService.approveJob(jobId)
            await refreshPageContext()
        } catch (err) {
            console.error('Failed to approve job:', err)
            setError('Failed to approve job.')
        }
    }

    const handleRejectJob = async (jobId: string, reason: string) => {
        setRejectingJobIds((current) => new Set(current).add(jobId))
        try {
            await researchRebuildService.rejectJob(jobId, {
                rejection_reason_tags: [reason],
            })
            await refreshPageContext()
        } catch (err) {
            console.error('Failed to reject job:', err)
            setError('Failed to reject job.')
        } finally {
            setRejectingJobIds((current) => {
                const next = new Set(current)
                next.delete(jobId)
                return next
            })
        }
    }

    const executeWorkflow = async (jobIds: string[], successLabel: string) => {
        if (!activeProject?.id || jobIds.length === 0) return
        try {
            setRunningWorkflow(true)
            setError(null)
            setSuccess(null)
            const response = await researchRebuildService.runWorkflow({
                project_id: activeProject.id,
                user_job_ids: jobIds,
            })
            setWorkflowPage(0)
            if (response.workflow_run_id) {
                setWorkflowRunFilter(response.workflow_run_id)
            }
            await refreshPageContext({
                workflowRunId: response.workflow_run_id || workflowRunFilter,
                workflowPage: 0,
            })
            setSuccess(successLabel)
        } catch (err) {
            console.error('Failed to run rebuild workflow:', err)
            setError('Failed to run rebuild workflow.')
        } finally {
            setRunningWorkflow(false)
        }
    }

    const handleRunWorkflow = async () => {
        await executeWorkflow(
            approvedJobs.map((job) => job.id),
            `Workflow ran for ${approvedJobs.length} approved job${approvedJobs.length === 1 ? '' : 's'}.`,
        )
    }

    const handleRerunVisibleJobs = async () => {
        await executeWorkflow(
            visibleJobIds,
            `Workflow re-ran for ${visibleJobIds.length} visible job${visibleJobIds.length === 1 ? '' : 's'}.`,
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
            setError('Failed to release software outcome.')
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
            setSuccess('Outcome persisted to Content Ideas.')
        } catch (err) {
            console.error('Failed to persist outcome to content ideas:', err)
            setError('Failed to persist outcome to Content Ideas.')
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
        <div className="min-h-screen bg-background">
            <div className="mx-auto max-w-7xl px-8 py-10 lg:py-14">
                <div className="mb-8">
                    <h1 className="text-2xl font-semibold tracking-tight text-foreground">Research Rebuild</h1>
                    <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
                        Experimental job-first workflow for discovering article and software opportunities before they reach the legacy
                        content pipeline.
                    </p>
                </div>

                {error && (
                    <div className="mb-4 rounded-xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-200">
                        {error}
                    </div>
                )}
                {success && (
                    <div className="mb-4 rounded-xl border border-emerald-500/20 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-200">
                        {success}
                    </div>
                )}

                <div className="grid gap-6 xl:grid-cols-[420px,1fr]">
                    <section className="rounded-3xl border border-border bg-card/70 p-6 backdrop-blur">
                        <div className="mb-4 flex items-center gap-2">
                            <Sparkles className="h-5 w-5 text-amber-300" />
                            <h2 className="text-lg font-semibold text-foreground">Job Discovery</h2>
                        </div>

                        <div className="space-y-4">
                            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-1">
                                <div>
                                    <label className="mb-1 block text-xs uppercase tracking-wide text-muted-foreground">Primary category</label>
                                    <select
                                        value={primaryCategoryId}
                                        onChange={(e) => {
                                            setPrimaryCategoryId(e.target.value)
                                            setSecondaryCategoryId('')
                                        }}
                                        className="h-10 w-full rounded-xl border border-border bg-background px-3 text-sm text-foreground"
                                    >
                                        <option value="">All</option>
                                        {primaryCategories.map((category) => (
                                            <option key={category.id} value={category.id}>
                                                {category.name}
                                            </option>
                                        ))}
                                    </select>
                                </div>
                                <div>
                                    <label className="mb-1 block text-xs uppercase tracking-wide text-muted-foreground">Secondary category</label>
                                    <select
                                        value={secondaryCategoryId}
                                        onChange={(e) => setSecondaryCategoryId(e.target.value)}
                                        className="h-10 w-full rounded-xl border border-border bg-background px-3 text-sm text-foreground"
                                    >
                                        <option value="">All</option>
                                        {secondaryCategories.map((category) => (
                                            <option key={category.id} value={category.id}>
                                                {category.name}
                                            </option>
                                        ))}
                                    </select>
                                </div>
                                <div>
                                    <label className="mb-1 block text-xs uppercase tracking-wide text-muted-foreground">Job status</label>
                                    <select
                                        value={jobStatusFilter}
                                        onChange={(e) => setJobStatusFilter(e.target.value)}
                                        className="h-10 w-full rounded-xl border border-border bg-background px-3 text-sm text-foreground"
                                    >
                                        <option value="all">All statuses</option>
                                        <option value="draft">Draft</option>
                                        <option value="approved">Approved</option>
                                        <option value="rejected">Rejected</option>
                                    </select>
                                </div>
                            </div>

                            <div className="flex flex-wrap gap-3">
                                <Button onClick={handleGenerateJobs} disabled={generatingJobs || !activeProject?.id}>
                                    {generatingJobs ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Sparkles className="mr-2 h-4 w-4" />}
                                    Generate Jobs
                                </Button>
                                <Button
                                    variant="secondary"
                                    onClick={handleRunWorkflow}
                                    disabled={runningWorkflow || approvedJobs.length === 0 || !activeProject?.id}
                                >
                                    {runningWorkflow ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Rocket className="mr-2 h-4 w-4" />}
                                    Run Workflow
                                </Button>
                                {Array.from(jobStatusCounts.entries())
                                    .sort((left, right) => right[1] - left[1])
                                    .map(([status, count]) => (
                                        <Badge key={status} variant="secondary">
                                            {status}: {count}
                                        </Badge>
                                    ))}
                            </div>

                            <div className="space-y-2">
                                <label className="block text-xs uppercase tracking-wide text-muted-foreground">Manual job</label>
                                <div className="flex gap-2">
                                    <Input
                                        value={manualJobText}
                                        onChange={(e) => setManualJobText(e.target.value)}
                                        placeholder="I need to compare ... / calculate ... / track ..."
                                    />
                                    <Button variant="outline" onClick={handleCreateManualJob} disabled={!manualJobText.trim()}>
                                        Add
                                    </Button>
                                </div>
                            </div>
                        </div>

                        <div className="mt-6 space-y-3">
                            <div className="flex items-center justify-between">
                                <h3 className="text-sm font-medium text-foreground">Jobs</h3>
                                {loadingJobs && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
                            </div>
                            <div className="space-y-3">
                                {jobs.map((job) => (
                                    <div key={job.id} className="rounded-2xl border border-border bg-background/70 p-4">
                                        <div className="flex items-start justify-between gap-3">
                                            <div>
                                                <p className="text-sm font-medium text-foreground">{job.job_text}</p>
                                                <p className="mt-1 text-xs text-muted-foreground">
                                                    {job.job_type_hint || 'hybrid'} · {job.status}
                                                </p>
                                            </div>
                                            <div className="flex flex-wrap gap-2">
                                                {job.status !== 'approved' && job.status !== 'rejected' && (
                                                    <Button size="sm" variant="outline" onClick={() => handleApproveJob(job.id)}>
                                                        Approve
                                                    </Button>
                                                )}
                                                {job.status !== 'rejected' && (
                                                    <Button
                                                        size="sm"
                                                        variant="ghost"
                                                        disabled={rejectingJobIds.has(job.id)}
                                                        onClick={() => handleRejectJob(job.id, 'too_broad')}
                                                    >
                                                        {rejectingJobIds.has(job.id) ? (
                                                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                                        ) : (
                                                            <XCircle className="mr-2 h-4 w-4" />
                                                        )}
                                                        Reject
                                                    </Button>
                                                )}
                                            </div>
                                        </div>
                                        {job.rejection_reason_tags && job.rejection_reason_tags.length > 0 && (
                                            <div className="mt-3 flex flex-wrap gap-2">
                                                {job.rejection_reason_tags.map((tag) => (
                                                    <Badge key={tag} variant="secondary">{tag}</Badge>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                ))}
                                {!loadingJobs && jobs.length === 0 && (
                                    <div className="rounded-2xl border border-dashed border-border bg-background/30 p-6 text-sm text-muted-foreground">
                                        No rebuild jobs yet.
                                    </div>
                                )}
                            </div>
                        </div>
                    </section>

                    <section className="rounded-3xl border border-border bg-card/70 p-6 backdrop-blur">
                        <div className="mb-4 flex items-center gap-2">
                            <CheckCircle2 className="h-5 w-5 text-emerald-300" />
                            <h2 className="text-lg font-semibold text-foreground">Workflow Results</h2>
                            {loadingWorkflowArtifacts && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
                        </div>

                        <div className="mb-4 flex flex-wrap items-center gap-3">
                            <Badge variant="secondary">
                                {filteredWorkflowResults.reduce((sum, jobResult) => sum + jobResult.candidates.length, 0)} candidates
                            </Badge>
                            <Badge variant="secondary">{workflowTotalJobs} workflow jobs</Badge>
                            <Badge variant="secondary">{staleValidationTargets.length} stale validations</Badge>
                            {Array.from(routeCounts.entries())
                                .sort((left, right) => right[1] - left[1])
                                .slice(0, 3)
                                .map(([route, count]) => (
                                    <Badge key={route} variant="secondary">
                                        {route.replaceAll('_', ' ')}: {count}
                                    </Badge>
                                ))}
                            <Button
                                size="sm"
                                variant="outline"
                                onClick={handleRefreshAllStaleValidations}
                                disabled={refreshingAllStale || staleValidationTargets.length === 0}
                            >
                                {refreshingAllStale ? (
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                ) : (
                                    <RefreshCw className="mr-2 h-4 w-4" />
                                )}
                                Refresh Stale
                            </Button>
                        </div>

                        {workflowRuns.length > 0 && (
                            <div className="mb-5">
                                <p className="mb-2 text-xs uppercase tracking-wide text-muted-foreground">Recent Runs</p>
                                <div className="flex flex-wrap gap-2">
                                    <Button
                                        size="sm"
                                        variant={workflowRunFilter === 'all' ? 'secondary' : 'outline'}
                                        onClick={() => setWorkflowRunFilter('all')}
                                    >
                                        All runs
                                    </Button>
                                    {workflowRuns.slice(0, 6).map((run) => (
                                        <Button
                                            key={run.workflow_run_id}
                                            size="sm"
                                            variant={workflowRunFilter === run.workflow_run_id ? 'secondary' : 'outline'}
                                            onClick={() => setWorkflowRunFilter(run.workflow_run_id)}
                                        >
                                            {formatWorkflowRunShortLabel(run)}
                                        </Button>
                                    ))}
                                </div>
                            </div>
                        )}

                        {selectedWorkflowRun && (
                            <div className="mb-5 rounded-2xl border border-border bg-background/40 p-4">
                                <div className="flex flex-wrap items-center justify-between gap-3">
                                    <div>
                                        <p className="text-sm font-medium text-foreground">Selected Run</p>
                                        <p className="mt-1 text-xs text-muted-foreground">
                                            {formatWorkflowRunLabel(selectedWorkflowRun)}
                                        </p>
                                    </div>
                                    <div className="flex flex-wrap items-center gap-2">
                                        {comparisonWorkflowRun && (
                                            <Button
                                                size="sm"
                                                variant="outline"
                                                onClick={() => setWorkflowRunFilter(comparisonWorkflowRun.workflow_run_id)}
                                            >
                                                <ArrowUpRight className="mr-2 h-4 w-4" />
                                                Latest Run
                                            </Button>
                                        )}
                                        <Button
                                            size="sm"
                                            variant="outline"
                                            onClick={handleCopyCurrentViewLink}
                                        >
                                            <Copy className="mr-2 h-4 w-4" />
                                            Copy Link
                                        </Button>
                                        <Button
                                            size="sm"
                                            variant="outline"
                                            onClick={handleRerunVisibleJobs}
                                            disabled={runningWorkflow || visibleJobIds.length === 0}
                                        >
                                            {runningWorkflow ? (
                                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                            ) : (
                                                <Rocket className="mr-2 h-4 w-4" />
                                            )}
                                            Rerun Visible Jobs
                                        </Button>
                                    </div>
                                </div>
                                <div className="mt-3 flex flex-wrap gap-2">
                                    <Badge variant="secondary">{selectedWorkflowRun.job_count} jobs</Badge>
                                    <Badge variant="secondary">{selectedWorkflowRun.candidate_count} candidates</Badge>
                                    {Object.entries(selectedWorkflowRun.route_counts || {})
                                        .sort((left, right) => right[1] - left[1])
                                        .slice(0, 3)
                                        .map(([route, count]) => (
                                            <Badge key={route} variant="secondary">
                                                {route.replaceAll('_', ' ')}: {count}
                                            </Badge>
                                        ))}
                                </div>
                            </div>
                        )}

                        {selectedWorkflowRun && comparisonWorkflowRun && (
                            <div className="mb-5 rounded-2xl border border-border bg-background/30 p-4">
                                <p className="text-sm font-medium text-foreground">Compare To Latest Run</p>
                                <p className="mt-1 text-xs text-muted-foreground">
                                    Latest: {formatWorkflowRunLabel(comparisonWorkflowRun)}
                                </p>
                                <div className="mt-3 grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                                    <ComparisonCard
                                        label="Jobs"
                                        current={selectedWorkflowRun.job_count}
                                        baseline={comparisonWorkflowRun.job_count}
                                    />
                                    <ComparisonCard
                                        label="Candidates"
                                        current={selectedWorkflowRun.candidate_count}
                                        baseline={comparisonWorkflowRun.candidate_count}
                                    />
                                    <ComparisonCard
                                        label="Article Ready"
                                        current={(selectedWorkflowRun.route_counts || {}).article_ready || 0}
                                        baseline={(comparisonWorkflowRun.route_counts || {}).article_ready || 0}
                                    />
                                    <ComparisonCard
                                        label="Software Ready"
                                        current={(selectedWorkflowRun.route_counts || {}).software_ready || 0}
                                        baseline={(comparisonWorkflowRun.route_counts || {}).software_ready || 0}
                                    />
                                </div>
                            </div>
                        )}

                        <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
                            <div className="text-xs text-muted-foreground">
                                Showing jobs {workflowTotalJobs === 0 ? 0 : workflowPage * workflowPageSize + 1}-
                                {Math.min((workflowPage + 1) * workflowPageSize, workflowTotalJobs)} of {workflowTotalJobs}
                            </div>
                            <div className="flex items-center gap-2">
                                <label className="text-xs uppercase tracking-wide text-muted-foreground">Page size</label>
                                <select
                                    value={String(workflowPageSize)}
                                    onChange={(e) => {
                                        setWorkflowPageSize(Number(e.target.value))
                                        setWorkflowPage(0)
                                    }}
                                    className="h-9 rounded-xl border border-border bg-background px-3 text-sm text-foreground"
                                >
                                    <option value="5">5</option>
                                    <option value="10">10</option>
                                    <option value="20">20</option>
                                    <option value="50">50</option>
                                </select>
                                <Button
                                    size="sm"
                                    variant="outline"
                                    onClick={() => setWorkflowPage((current) => Math.max(0, current - 1))}
                                    disabled={workflowPage === 0 || loadingWorkflowArtifacts}
                                >
                                    Previous
                                </Button>
                                <Button
                                    size="sm"
                                    variant="outline"
                                    onClick={() => setWorkflowPage((current) => current + 1)}
                                    disabled={loadingWorkflowArtifacts || (workflowPage + 1) * workflowPageSize >= workflowTotalJobs}
                                >
                                    Next
                                </Button>
                            </div>
                        </div>

                        <div className="mb-5 grid gap-3 md:grid-cols-2 xl:grid-cols-5">
                            <div>
                                <label className="mb-1 block text-xs uppercase tracking-wide text-muted-foreground">Workflow run</label>
                                <select
                                    value={workflowRunFilter}
                                    onChange={(e) => setWorkflowRunFilter(e.target.value)}
                                    className="h-10 w-full rounded-xl border border-border bg-background px-3 text-sm text-foreground"
                                >
                                    <option value="all">All runs</option>
                                    {workflowRuns.map((run) => (
                                        <option key={run.workflow_run_id} value={run.workflow_run_id}>
                                            {formatWorkflowRunLabel(run)}
                                        </option>
                                    ))}
                                </select>
                            </div>
                            <div>
                                <label className="mb-1 block text-xs uppercase tracking-wide text-muted-foreground">Route</label>
                                <select
                                    value={routeFilter}
                                    onChange={(e) => setRouteFilter(e.target.value)}
                                    className="h-10 w-full rounded-xl border border-border bg-background px-3 text-sm text-foreground"
                                >
                                    <option value="all">All routes</option>
                                    <option value="article_ready">Article Ready</option>
                                    <option value="software_ready">Software Ready</option>
                                    <option value="article_plus_software">Article + Software</option>
                                    <option value="editorial_only">Editorial Only</option>
                                    <option value="software_backlog_low_feasibility">Software Backlog</option>
                                    <option value="needs_more_keyword_validation">Needs Validation</option>
                                    <option value="rejected_low_achievability">Rejected</option>
                                </select>
                            </div>
                            <div>
                                <label className="mb-1 block text-xs uppercase tracking-wide text-muted-foreground">Candidate type</label>
                                <select
                                    value={candidateTypeFilter}
                                    onChange={(e) => setCandidateTypeFilter(e.target.value)}
                                    className="h-10 w-full rounded-xl border border-border bg-background px-3 text-sm text-foreground"
                                >
                                    <option value="all">All types</option>
                                    <option value="seo_article">SEO Article</option>
                                    <option value="software">Software</option>
                                    <option value="editorial">Editorial</option>
                                </select>
                            </div>
                            <div>
                                <label className="mb-1 block text-xs uppercase tracking-wide text-muted-foreground">Outcome type</label>
                                <select
                                    value={outcomeTypeFilter}
                                    onChange={(e) => setOutcomeTypeFilter(e.target.value)}
                                    className="h-10 w-full rounded-xl border border-border bg-background px-3 text-sm text-foreground"
                                >
                                    <option value="all">All outcomes</option>
                                    <option value="article">Article</option>
                                    <option value="software">Software</option>
                                    <option value="editorial">Editorial</option>
                                </select>
                            </div>
                            <div>
                                <label className="mb-1 block text-xs uppercase tracking-wide text-muted-foreground">Search</label>
                                <Input
                                    value={candidateSearch}
                                    onChange={(e) => setCandidateSearch(e.target.value)}
                                    placeholder="keyword, job, route..."
                                />
                            </div>
                        </div>

                        <div className="space-y-5">
                            {filteredWorkflowResults.map((jobResult) => (
                                <div key={jobResult.job_id} className="rounded-2xl border border-border bg-background/50 p-4">
                                    <div className="mb-3">
                                        <p className="text-sm font-medium text-foreground">
                                            {jobResult.job?.job_text || `Job ${jobResult.job_id}`}
                                        </p>
                                        <p className="mt-1 text-xs text-muted-foreground">{jobResult.job_id}</p>
                                    </div>
                                    <div className="space-y-3">
                                        {jobResult.candidates.map(({ candidate, validation_run, routing_decision, keyword_pack, internal_link_candidates, generated_outcome }) => (
                                            <div key={candidate.id} className="rounded-2xl border border-border bg-background/70 p-4">
                                                <div className="flex items-start justify-between gap-3">
                                                    <div>
                                                        <p className="text-sm font-semibold text-foreground">{candidate.candidate_text}</p>
                                                        <p className="mt-1 text-xs text-muted-foreground">
                                                            {candidate.candidate_type} · route: {routing_decision.route}
                                                        </p>
                                                        <div className="mt-2 flex flex-wrap gap-2">
                                                            <Badge variant="secondary">{candidate.candidate_type}</Badge>
                                                            <Badge variant="secondary">{routing_decision.route}</Badge>
                                                            <Badge variant="secondary">{keyword_pack.keyword_pack_status}</Badge>
                                                        </div>
                                                    </div>
                                                    <div className="flex flex-wrap items-center gap-2">
                                                        {generated_outcome.outcome_type === 'software' ? (
                                                            <Button
                                                                size="sm"
                                                                variant="outline"
                                                                onClick={() => handleReleaseSoftware(generated_outcome.id)}
                                                                disabled={mutatingOutcomeIds.has(generated_outcome.id)}
                                                            >
                                                                {mutatingOutcomeIds.has(generated_outcome.id) ? (
                                                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                                                ) : (
                                                                    <Wrench className="mr-2 h-4 w-4" />
                                                                )}
                                                                Release
                                                            </Button>
                                                        ) : (
                                                            <Button
                                                                size="sm"
                                                                variant="outline"
                                                                onClick={() => handlePersistOutcome(generated_outcome.id)}
                                                                disabled={mutatingOutcomeIds.has(generated_outcome.id)}
                                                            >
                                                                {mutatingOutcomeIds.has(generated_outcome.id) ? (
                                                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                                                ) : (
                                                                    <BookOpen className="mr-2 h-4 w-4" />
                                                                )}
                                                                Persist
                                                            </Button>
                                                        )}
                                                        <Button
                                                            size="sm"
                                                            variant="ghost"
                                                            disabled={rejectingCandidateIds.has(candidate.id)}
                                                            onClick={() => handleRejectCandidate(candidate.id, 'weak_serp')}
                                                        >
                                                            {rejectingCandidateIds.has(candidate.id) ? (
                                                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                                            ) : (
                                                                <Ban className="mr-2 h-4 w-4" />
                                                            )}
                                                            Reject
                                                        </Button>
                                                        <Button
                                                            size="sm"
                                                            variant="ghost"
                                                            disabled={refreshingValidationIds.has(validation_run.id)}
                                                            onClick={() => handleRefreshValidation(validation_run.id)}
                                                        >
                                                            {refreshingValidationIds.has(validation_run.id) ? (
                                                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                                            ) : (
                                                                <RefreshCw className="mr-2 h-4 w-4" />
                                                            )}
                                                            Refresh
                                                        </Button>
                                                        <Button size="sm" variant="ghost" onClick={() => toggleCandidateExpansion(candidate.id)}>
                                                            {expandedCandidateIds.has(candidate.id) ? (
                                                                <ChevronUp className="h-4 w-4" />
                                                            ) : (
                                                                <ChevronDown className="h-4 w-4" />
                                                            )}
                                                        </Button>
                                                    </div>
                                                </div>
                                                <div className="mt-3 grid gap-3 md:grid-cols-3">
                                                    <div className="rounded-xl border border-border bg-background/50 p-3">
                                                        <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Achievability</p>
                                                        <p className="mt-1 text-lg font-semibold text-foreground">
                                                            {typeof validation_run.achievability_score === 'number'
                                                                ? validation_run.achievability_score.toFixed(2)
                                                                : 'n/a'}
                                                        </p>
                                                    </div>
                                                    <div className="rounded-xl border border-border bg-background/50 p-3">
                                                        <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Primary keyword</p>
                                                        <p className="mt-1 text-sm font-medium text-foreground">
                                                            {keyword_pack.primary_keyword || 'n/a'}
                                                        </p>
                                                    </div>
                                                    <div className="rounded-xl border border-border bg-background/50 p-3">
                                                        <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Outcome</p>
                                                        <p className="mt-1 text-sm font-medium text-foreground">
                                                            {generated_outcome.outcome_type} · {generated_outcome.status}
                                                        </p>
                                                    </div>
                                                </div>

                                                {expandedCandidateIds.has(candidate.id) && (
                                                    <div className="mt-4 space-y-4">
                                                        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                                                            <ScoreCard label="Intent Match" value={validation_run.intent_match_score} />
                                                            <ScoreCard label="SERP Weakness" value={validation_run.serp_weakness_score} />
                                                            <ScoreCard label="SERP Gap" value={validation_run.serp_gap_score} />
                                                            <ScoreCard label="Feasibility" value={validation_run.feasibility_score} />
                                                        </div>

                                                        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                                                            <MetricCard label="Search Volume" value={getValidationMetric(validation_run, 'search_volume')} />
                                                            <MetricCard label="CPC" value={getValidationMetric(validation_run, 'cpc')} />
                                                            <MetricCard label="Keyword Difficulty" value={getValidationMetric(validation_run, 'keyword_difficulty')} />
                                                            <MetricCard label="Freshness" value={validation_run.freshness_state} />
                                                        </div>

                                                        <div className="rounded-xl border border-border bg-background/50 p-4">
                                                            <p className="text-xs uppercase tracking-wide text-muted-foreground">Reason Codes</p>
                                                            <div className="mt-2 flex flex-wrap gap-2">
                                                                {(validation_run.validation_reason_codes || []).map((code) => (
                                                                    <Badge key={code} variant="secondary">{code}</Badge>
                                                                ))}
                                                            </div>
                                                        </div>

                                                        <div className="rounded-xl border border-border bg-background/50 p-4">
                                                            <p className="text-xs uppercase tracking-wide text-muted-foreground">SERP Evidence</p>
                                                            <div className="mt-3 space-y-2">
                                                                {getSerpRows(validation_run).slice(0, 5).map((row, index) => (
                                                                    <div key={`${candidate.id}-serp-${index}`} className="rounded-lg border border-border/60 bg-background/70 p-3">
                                                                        <p className="text-sm font-medium text-foreground">{String(row.title || 'Untitled result')}</p>
                                                                        <p className="mt-1 text-xs text-muted-foreground">{String(row.url || '')}</p>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        </div>

                                                        <div className="rounded-xl border border-border bg-background/50 p-4">
                                                            <p className="text-xs uppercase tracking-wide text-muted-foreground">Internal Link Hooks</p>
                                                            <div className="mt-3 space-y-2">
                                                                {(internal_link_candidates || []).length > 0 ? (
                                                                    (internal_link_candidates || []).slice(0, 5).map((linkRow) => {
                                                                        const matchMetadata = getInternalLinkMetadata(linkRow)
                                                                        const matchedLink = typeof matchMetadata.matched_link === 'string' ? matchMetadata.matched_link : null
                                                                        const matchedTitle = typeof matchMetadata.matched_title === 'string' ? matchMetadata.matched_title : 'Matched post'

                                                                        return (
                                                                        <div key={linkRow.id} className="rounded-lg border border-border/60 bg-background/70 p-3">
                                                                            <p className="text-sm font-medium text-foreground">
                                                                                {matchedTitle}
                                                                            </p>
                                                                            <p className="mt-1 text-xs text-muted-foreground">
                                                                                {linkRow.link_role} · score {typeof linkRow.match_score === 'number' ? linkRow.match_score.toFixed(2) : 'n/a'}
                                                                            </p>
                                                                            {matchedLink && (
                                                                                <a
                                                                                    href={matchedLink}
                                                                                    target="_blank"
                                                                                    rel="noreferrer"
                                                                                    className="mt-1 inline-block text-xs text-blue-300 underline-offset-2 hover:underline"
                                                                                >
                                                                                    Open source post
                                                                                </a>
                                                                            )}
                                                                        </div>
                                                                        )
                                                                    })
                                                                ) : (
                                                                    <p className="text-sm text-muted-foreground">No internal-link matches found yet.</p>
                                                                )}
                                                            </div>
                                                        </div>
                                                    </div>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            ))}

                            {filteredWorkflowResults.length === 0 && (
                                <div className="rounded-2xl border border-dashed border-border bg-background/30 p-10 text-sm text-muted-foreground">
                                    No workflow results match the current filters yet.
                                </div>
                            )}
                        </div>
                    </section>
                </div>
            </div>
        </div>
    )
}

export default ResearchRebuild

function ScoreCard({ label, value }: { label: string; value?: number | null }) {
    return (
        <div className="rounded-xl border border-border bg-background/50 p-3">
            <p className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</p>
            <p className="mt-1 text-lg font-semibold text-foreground">
                {typeof value === 'number' ? value.toFixed(2) : 'n/a'}
            </p>
        </div>
    )
}

function MetricCard({ label, value }: { label: string; value?: number | string | null }) {
    return (
        <div className="rounded-xl border border-border bg-background/50 p-3">
            <p className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</p>
            <p className="mt-1 text-sm font-semibold text-foreground">
                {value === null || value === undefined || value === '' ? 'n/a' : String(value)}
            </p>
        </div>
    )
}

function ComparisonCard({ label, current, baseline }: { label: string; current: number; baseline: number }) {
    const delta = current - baseline
    const deltaLabel = delta === 0 ? 'No change' : `${delta > 0 ? '+' : ''}${delta}`
    const deltaTone = delta > 0 ? 'text-emerald-300' : delta < 0 ? 'text-amber-300' : 'text-muted-foreground'

    return (
        <div className="rounded-xl border border-border bg-background/50 p-3">
            <p className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</p>
            <p className="mt-1 text-lg font-semibold text-foreground">{current}</p>
            <p className={`mt-1 text-xs ${deltaTone}`}>
                vs latest {baseline} · {deltaLabel}
            </p>
        </div>
    )
}

function getValidationMetric(
    validationRun: ResearchRebuildValidationRun,
    key: string,
): string | number | null | undefined {
    const metadata = validationRun.validation_metadata as Record<string, unknown> | undefined
    const value = metadata?.[key]
    if (typeof value === 'string' || typeof value === 'number' || value === null || value === undefined) {
        return value
    }
    return undefined
}

function getSerpRows(validationRun: ResearchRebuildValidationRun): Array<Record<string, unknown>> {
    const metadata = validationRun.validation_metadata as Record<string, unknown> | undefined
    const rows = metadata?.serp_rows
    return Array.isArray(rows) ? rows.filter((row): row is Record<string, unknown> => typeof row === 'object' && row !== null) : []
}

function getInternalLinkMetadata(linkCandidate: ResearchRebuildInternalLinkCandidate): Record<string, unknown> {
    return (linkCandidate.match_metadata as Record<string, unknown> | undefined) || {}
}

function formatWorkflowRunLabel(run: ResearchRebuildWorkflowRunSummary): string {
    const startedAt = run.started_at ? new Date(run.started_at).toLocaleString() : 'Unknown start'
    return `${startedAt} · ${run.job_count} jobs · ${run.candidate_count} candidates`
}

function formatWorkflowRunShortLabel(run: ResearchRebuildWorkflowRunSummary): string {
    const startedAt = run.started_at ? new Date(run.started_at).toLocaleDateString() : 'Unknown'
    return `${startedAt} · ${run.job_count}j · ${run.candidate_count}c`
}
