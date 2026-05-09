import * as React from 'react'
import { 
    Loader2, Rocket, Wrench, CheckCircle2, Sparkles, ChevronDown, 
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

type ResearchRebuildMode = 'jobs' | 'opportunities'

type ResearchRebuildProps = {
    mode?: ResearchRebuildMode
}

export function ResearchRebuild({ mode = 'jobs' }: ResearchRebuildProps) {
    const { user } = useAuth()
    const { activeProject, projects, setActiveProject } = useProject()
    const navigate = useNavigate()
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
    const [jobStatusFilter, setJobStatusFilter] = React.useState('all')
    const [workflowRuns, setWorkflowRuns] = React.useState<ResearchRebuildWorkflowRunSummary[]>([])
    const [routeFilter, setRouteFilter] = React.useState('all')
    const [candidateSearch, setCandidateSearch] = React.useState('')
    const [manualJobText, setManualJobText] = React.useState('')
    const [focusArea, setFocusArea] = React.useState('')
    const [avoidGuidance, setAvoidGuidance] = React.useState('')
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
        const query = params.toString()
        return `/research-rebuild/${targetMode}${query ? `?${query}` : ''}`
    }, [activeProject?.id, primaryCategoryId, secondaryCategoryId, workflowRunFilter])

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
    }, [incomingPrimaryCategoryId, incomingSecondaryCategoryId, incomingWorkflowRunId])

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

        const relevantKeys = ['project_id', 'primary_category_id', 'secondary_category_id', 'workflow_run_id']
        const hasChanged = relevantKeys.some(key => params.get(key) !== searchParams.get(key))
        
        // Also check if there are extra keys in searchParams that we should strip
        const hasExtraKeys = Array.from(searchParams.keys()).some(key => !relevantKeys.includes(key))

        if (!hasChanged && !hasExtraKeys) {
            return
        }

        navigate(
            {
                pathname: mode === 'jobs' ? '/research-rebuild/jobs' : '/research-rebuild/opportunities',
                search: params.toString() ? `?${params.toString()}` : '',
            },
            { replace: true },
        )
    }, [activeProject?.id, mode, navigate, primaryCategoryId, searchParams, secondaryCategoryId, workflowRunFilter])

    React.useEffect(() => {
        setWorkflowPage(0)
    }, [activeProject?.id, primaryCategoryId, secondaryCategoryId, jobStatusFilter, workflowRunFilter, routeFilter, candidateSearch])

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
    }, [activeProject?.id, primaryCategoryId, secondaryCategoryId, jobStatusFilter, workflowRunFilter, routeFilter, candidateSearch, workflowPage, workflowPageSize])

    React.useEffect(() => {
        void refreshPageContext()
    }, [refreshPageContext])

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
            approvedJobs.map((job) => job.id),
            `Workflow ran for ${approvedJobs.length} approved job${approvedJobs.length === 1 ? '' : 's'}.`,
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
        <div className="min-h-screen bg-[#08080a] text-slate-200 selection:bg-indigo-500/30">
            {/* Premium Header */}
            <header className="border-b border-white/5 bg-[#0d0d0f]/90 backdrop-blur-2xl sticky top-0 z-50">
                <div className="mx-auto max-w-[1600px] px-6 py-4 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <div className="bg-gradient-to-tr from-indigo-600 to-indigo-400 p-2.5 rounded-2xl shadow-[0_0_20px_rgba(99,102,241,0.4)]">
                            <Sparkles className="h-5 w-5 text-white" />
                        </div>
                        <div>
                            <h1 className="text-xl font-black tracking-tight text-white flex items-center gap-2">
                                Research <span className="text-indigo-400">{isJobsPage ? 'Setup' : 'Results'}</span>
                                <Badge variant="outline" className="text-[10px] bg-indigo-500/10 border-indigo-500/20 text-indigo-400 font-black uppercase">v2.0</Badge>
                            </h1>
                            <p className="text-[10px] text-slate-500 font-black uppercase tracking-[0.25em]">
                                {isJobsPage ? 'Step 1 of 2 · Job Discovery' : 'Step 2 of 2 · Opportunity Validation'}
                            </p>
                        </div>
                    </div>
                    
                    {/* Visual Flow Indicator */}
                    <div className="hidden xl:flex items-center gap-8 absolute left-1/2 -translate-x-1/2">
                        <FlowStep number="1" label="Discover" active={isJobsPage || jobs.length > 0} icon={<Layers className="h-3.5 w-3.5" />} />
                        <div className="w-8 h-px bg-white/10" />
                        <FlowStep number="2" label="Validate" active={isOpportunitiesPage || workflowResults.length > 0} icon={<Target className="h-3.5 w-3.5" />} />
                        <div className="w-8 h-px bg-white/10" />
                        <FlowStep number="3" label="Promote" active={workflowResults.some(r => r.candidates.some(c => c.generated_outcome.status === 'persisted'))} icon={<Rocket className="h-3.5 w-3.5" />} />
                    </div>

                    <div className="flex items-center gap-4">
                        <div className="hidden lg:flex items-center gap-2 rounded-2xl border border-white/10 bg-white/[0.03] px-3 py-2">
                            <button
                                type="button"
                                onClick={() => navigate(jobsPagePath)}
                                className={cn(
                                    "rounded-xl px-3 py-1.5 text-[10px] font-black uppercase tracking-[0.2em] transition-all",
                                    isJobsPage ? "bg-white text-black" : "text-slate-400 hover:text-white",
                                )}
                            >
                                1. Jobs
                            </button>
                            <button
                                type="button"
                                onClick={() => navigate(opportunitiesPagePath)}
                                disabled={approvedJobs.length === 0}
                                className={cn(
                                    "rounded-xl px-3 py-1.5 text-[10px] font-black uppercase tracking-[0.2em] transition-all",
                                    isOpportunitiesPage ? "bg-indigo-500 text-white" : "text-slate-400 hover:text-white",
                                    approvedJobs.length === 0 && "cursor-not-allowed opacity-40 hover:text-slate-400",
                                )}
                            >
                                2. Opportunities
                            </button>
                        </div>
                        <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 bg-white/5 rounded-xl border border-white/10">
                            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                            <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Live Engine</span>
                        </div>
                        <Button 
                            variant="ghost" 
                            size="sm" 
                            className="rounded-xl h-10 px-4 text-slate-400 hover:text-white hover:bg-white/5 border border-white/10 transition-all"
                            onClick={handleCopyCurrentViewLink}
                        >
                            <Copy className="mr-2 h-3.5 w-3.5" />
                            <span className="text-xs font-bold uppercase tracking-wider">Share Link</span>
                        </Button>
                    </div>
                </div>
            </header>

            <main className="mx-auto max-w-[1600px] p-6 lg:p-10">
                {/* Notifications */}
                <div className="fixed bottom-10 right-10 z-[100] space-y-4 max-w-md">
                    {error && (
                        <div className="flex items-start gap-4 bg-red-500/10 border border-red-500/20 text-red-200 px-5 py-4 rounded-2xl backdrop-blur-3xl animate-in slide-in-from-bottom-5 shadow-2xl">
                            <div className="bg-red-500/20 p-1.5 rounded-lg"><XCircle className="h-5 w-5 text-red-400" /></div>
                            <div className="flex-1">
                                <h4 className="text-xs font-bold uppercase tracking-widest text-red-400 mb-1">Error Encountered</h4>
                                <p className="text-[13px] font-medium leading-relaxed opacity-90">{error}</p>
                            </div>
                            <button onClick={() => setError(null)} className="text-slate-500 hover:text-white transition-colors"><XCircle className="h-4 w-4" /></button>
                        </div>
                    )}
                    {success && (
                        <div className="flex items-start gap-4 bg-emerald-500/10 border border-emerald-500/20 text-emerald-200 px-5 py-4 rounded-2xl backdrop-blur-3xl animate-in slide-in-from-bottom-5 shadow-2xl">
                            <div className="bg-emerald-500/20 p-1.5 rounded-lg"><CheckCircle2 className="h-5 w-5 text-emerald-400" /></div>
                            <div className="flex-1">
                                <h4 className="text-xs font-bold uppercase tracking-widest text-emerald-400 mb-1">Success</h4>
                                <p className="text-[13px] font-medium leading-relaxed opacity-90">{success}</p>
                            </div>
                            <button onClick={() => setSuccess(null)} className="text-slate-500 hover:text-white transition-colors"><XCircle className="h-4 w-4" /></button>
                        </div>
                    )}
                </div>

                <div
                    className={cn(
                        "grid gap-10 items-start relative",
                        isJobsPage ? "mx-auto max-w-5xl lg:grid-cols-1" : "lg:grid-cols-[400px,1fr]",
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
                    <aside className={cn("space-y-8", isJobsPage ? "mx-auto w-full max-w-4xl" : "lg:sticky lg:top-32")}>
                        <div className="px-2 space-y-4 relative z-10">
                             <h2 className="text-[11px] font-black text-indigo-400 uppercase tracking-[0.4em] flex items-center gap-3">
                                <span className="flex items-center justify-center w-6 h-6 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-[10px]">01</span>
                                {isJobsPage ? 'Discovery Phase' : 'Research Context'}
                            </h2>
                            <div>
                                <h3 className="text-xl font-black text-white tracking-tight mb-2">
                                    {isJobsPage ? 'Capture User Intent' : 'Keep Context Visible'}
                                </h3>
                                <p className="text-slate-500 text-xs font-medium leading-relaxed mb-6">
                                    {isJobsPage
                                        ? 'Define the niche and capture category-specific user jobs to fuel the research engine.'
                                        : 'Review the approved jobs and scope while you validate opportunities on the second step.'}
                                </p>
                                
                                <PhaseGuide 
                                    title={isJobsPage ? 'Getting Started' : 'Step Reminder'}
                                    description={isJobsPage
                                        ? "Define your niche below. Use 'Generate AI-Driven Jobs' to discover what users are looking for in this category."
                                        : "This page is for validation and decision-making. Go back to Jobs if you need to reshape the batch before running validation again."}
                                    color="indigo"
                                />
                            </div>
                        </div>

                        <section className={cn("bg-[#0d0d0f] border border-white/5 rounded-[2.5rem] shadow-2xl overflow-hidden", isOpportunitiesPage && "hidden")}>
                            <div className="p-8 border-b border-white/5 bg-gradient-to-br from-white/[0.03] to-transparent">
                                <h3 className="text-sm font-black text-white uppercase tracking-widest flex items-center gap-3">
                                    <Globe className="h-4 w-4 text-indigo-400" />
                                    Niche Definition
                                </h3>
                            </div>

                            <div className="p-8 space-y-8">
                                <div className="space-y-5">
                                    <SelectField 
                                        label="Primary Domain Category" 
                                        value={primaryCategoryId} 
                                        onChange={(val) => { setPrimaryCategoryId(val); setSecondaryCategoryId(''); }}
                                        options={[{ value: '', label: 'Select Primary Category' }, ...primaryCategories.map(c => ({ value: c.id, label: c.name }))]} 
                                    />
                                    <SelectField 
                                        label="Target Sub-Category" 
                                        value={secondaryCategoryId} 
                                        onChange={setSecondaryCategoryId}
                                        options={[{ value: '', label: 'Select Sub-category' }, ...secondaryCategories.map(c => ({ value: c.id, label: c.name }))]} 
                                    />
                                    <div className="space-y-2">
                                        <label className="text-[10px] font-black uppercase tracking-[0.25em] text-slate-500">
                                            Focus Area For This Batch
                                        </label>
                                        <textarea
                                            value={focusArea}
                                            onChange={(e) => setFocusArea(e.target.value)}
                                            placeholder="Example: privacy-first PKM workflows, second-brain tools for structured thinking, AI research efficiency for solo founders"
                                            className="min-h-[88px] w-full rounded-2xl border border-white/10 bg-white/[0.03] px-4 py-3 text-sm text-slate-200 placeholder:text-slate-600 outline-none transition focus:border-indigo-500/40 focus:ring-2 focus:ring-indigo-500/20"
                                        />
                                        <p className="text-[11px] text-slate-500 leading-relaxed">
                                            Use this to tell the generator which slice of the category you want to explore. It guides the batch without creating a job directly.
                                        </p>
                                    </div>
                                    <div className="space-y-2">
                                        <label className="text-[10px] font-black uppercase tracking-[0.25em] text-slate-500">
                                            Avoid In This Batch
                                        </label>
                                        <Input
                                            value={avoidGuidance}
                                            onChange={(e) => setAvoidGuidance(e.target.value)}
                                            placeholder="Example: generic productivity advice, enterprise use cases, broad AI news"
                                            className="bg-white/[0.03] border-white/10 rounded-xl text-xs h-12 focus:ring-indigo-500/30"
                                        />
                                        <p className="text-[11px] text-slate-500 leading-relaxed">
                                            Optional steering to suppress patterns you do not want repeated in the generated jobs.
                                        </p>
                                    </div>
                                </div>

                                <div className="pt-2 flex flex-col gap-4">
                                    <Button 
                                        className="w-full bg-white text-black hover:bg-indigo-50 h-14 rounded-2xl font-black uppercase tracking-[0.15em] text-[11px] transition-all active:scale-[0.97] shadow-xl group"
                                        onClick={handleGenerateJobs} 
                                        disabled={generatingJobs || !activeProject?.id}
                                    >
                                        {generatingJobs ? <Loader2 className="mr-3 h-5 w-5 animate-spin" /> : <Sparkles className="mr-3 h-5 w-5 text-indigo-600 group-hover:scale-110 transition-transform" />}
                                        Generate AI-Driven Jobs
                                    </Button>
                                    
                                    <div className="relative py-2">
                                        <div className="absolute inset-0 flex items-center"><span className="w-full border-t border-white/5" /></div>
                                        <div className="relative flex justify-center text-[9px] uppercase tracking-[0.3em]"><span className="bg-[#0d0d0f] px-6 text-slate-600 font-black">Or Manual Entry</span></div>
                                    </div>

                                    <div className="flex gap-2">
                                        <Input
                                            value={manualJobText}
                                            onChange={(e) => setManualJobText(e.target.value)}
                                            placeholder="Example: I need to calculate X..."
                                            className="bg-white/[0.03] border-white/10 rounded-xl text-xs h-12 focus:ring-indigo-500/30"
                                        />
                                        <Button 
                                            variant="secondary" 
                                            size="icon"
                                            onClick={handleCreateManualJob} 
                                            disabled={!manualJobText.trim()}
                                            className="rounded-xl h-12 w-12 shrink-0 bg-white/5 hover:bg-white/10"
                                        >
                                            <ArrowRight className="h-5 w-5" />
                                        </Button>
                                    </div>
                                </div>
                            </div>

                            {/* Job Feed */}
                            <div className="bg-black/40 border-t border-white/5">
                                <div className="p-6 flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                        <Layers className="h-3.5 w-3.5 text-slate-500" />
                                        <span className="text-[10px] font-black text-slate-400 uppercase tracking-[0.2em]">Job Pipeline</span>
                                    </div>
                                    <div className="flex gap-1.5 p-1 bg-white/5 rounded-lg">
                                        {['approved', 'draft'].map(s => (
                                            <button 
                                                key={s} 
                                                onClick={() => setJobStatusFilter(s)} 
                                                className={cn(
                                                    "px-3 py-1 rounded-md text-[9px] font-black uppercase transition-all", 
                                                    jobStatusFilter === s ? "bg-white text-black shadow-lg" : "text-slate-500 hover:text-slate-300"
                                                )}
                                            >
                                                {s} <span className="opacity-50 ml-1">{jobStatusCounts.get(s) || 0}</span>
                                            </button>
                                        ))}
                                    </div>
                                </div>
                                <div className="max-h-[440px] overflow-y-auto px-6 pb-8 space-y-3 custom-scrollbar">
                                    {loadingJobs ? (
                                        <div className="py-20 text-center flex flex-col items-center gap-4">
                                            <Loader2 className="h-6 w-6 animate-spin text-slate-700" />
                                            <p className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-600">Retrieving Jobs...</p>
                                        </div>
                                    ) : jobs.length === 0 ? (
                                        <div className="py-20 text-center flex flex-col items-center gap-3">
                                            <Info className="h-8 w-8 text-slate-800" />
                                            <p className="text-slate-600 font-bold text-xs">No active jobs found.</p>
                                            <p className="text-slate-700 text-[10px] uppercase tracking-widest max-w-[200px]">Generate or add jobs to start the pipeline.</p>
                                        </div>
                                    ) : (
                                        jobs.map((job) => (
                                            <div 
                                                key={job.id} 
                                                className={cn(
                                                    "group relative rounded-2xl p-4 border transition-all duration-300", 
                                                    job.status === 'approved' 
                                                        ? "bg-indigo-500/[0.04] border-indigo-500/10 hover:border-indigo-500/20" 
                                                        : "bg-white/[0.02] border-white/5 hover:border-white/10"
                                                )}
                                            >
                                                <div className="flex items-start justify-between gap-4">
                                                    <p className={cn("text-[13px] leading-relaxed font-medium transition-colors", job.status === 'approved' ? "text-slate-200" : "text-slate-400")}>
                                                        {job.job_text}
                                                    </p>
                                                    <div className="flex flex-col gap-1.5 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                                                        {job.status !== 'approved' && job.status !== 'rejected' && (
                                                            <button 
                                                                onClick={() => handleApproveJob(job.id)} 
                                                                className="p-2 rounded-xl bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500 hover:text-white transition-all shadow-xl shadow-emerald-500/10"
                                                                title="Approve for Validation"
                                                            >
                                                                <CheckCircle2 className="h-4 w-4" />
                                                            </button>
                                                        )}
                                                        {job.status !== 'rejected' && (
                                                            <button 
                                                                onClick={() => handleRejectJob(job.id, 'off_brand')} 
                                                                disabled={rejectingJobIds.has(job.id)} 
                                                                className="p-2 rounded-xl bg-red-500/10 text-red-400 hover:bg-red-500 hover:text-white transition-all shadow-xl shadow-red-500/10"
                                                                title="Reject Job"
                                                            >
                                                                {rejectingJobIds.has(job.id) ? <Loader2 className="h-4 w-4 animate-spin" /> : <Ban className="h-4 w-4" />}
                                                            </button>
                                                        )}
                                                    </div>
                                                </div>
                                            </div>
                                        ))
                                    )}
                                </div>
                            </div>
                        </section>

                        {isOpportunitiesPage && (
                            <section className="bg-[#0d0d0f] border border-white/5 rounded-[2.5rem] shadow-2xl overflow-hidden">
                                <div className="p-8 border-b border-white/5 bg-gradient-to-br from-white/[0.03] to-transparent">
                                    <h3 className="text-sm font-black text-white uppercase tracking-widest flex items-center gap-3">
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
                                        <ContextStat label="Approved Jobs" value={String(approvedJobs.length)} />
                                        <ContextStat label="Workflow Runs" value={String(workflowRuns.length)} />
                                    </div>
                                    <div className="rounded-2xl border border-white/5 bg-black/30 p-5">
                                        <div className="mb-3 flex items-center justify-between">
                                            <span className="text-[10px] font-black uppercase tracking-[0.25em] text-slate-500">Approved Job Set</span>
                                            <Button
                                                variant="outline"
                                                size="sm"
                                                className="border-white/10 bg-white/[0.03] hover:bg-white/[0.08]"
                                                onClick={() => navigate(jobsPagePath)}
                                            >
                                                Back To Jobs
                                            </Button>
                                        </div>
                                        <div className="space-y-3">
                                            {approvedJobs.length === 0 ? (
                                                <p className="text-sm text-slate-500">No approved jobs yet. Go back to Jobs and approve the ones worth validating.</p>
                                            ) : (
                                                approvedJobs.slice(0, 6).map((job) => (
                                                    <div key={job.id} className="rounded-2xl border border-white/5 bg-white/[0.02] px-4 py-3 text-sm text-slate-300">
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
                            <div className="relative bg-gradient-to-br from-[#121218] to-[#0d0d0f] p-10 rounded-[2.5rem] border border-white/10 shadow-2xl overflow-hidden">
                                <div className="absolute top-0 right-0 p-6 opacity-5 group-hover:scale-125 group-hover:opacity-10 transition-all duration-700 pointer-events-none">
                                    <Rocket className="h-32 w-32 text-white" />
                                </div>
                                <h3 className="text-[11px] font-black text-indigo-400 uppercase tracking-[0.4em] mb-4">
                                    {isJobsPage ? 'Phase 02: Validate' : 'Step Actions'}
                                </h3>
                                <h4 className="text-white font-black text-xl mb-3 tracking-tight leading-tight">
                                    {isJobsPage ? <>Opportunity <br />Verification Engine</> : <>Run Or Refresh <br />Validation</>}
                                </h4>
                                <p className="text-slate-400 text-[13px] mb-10 leading-relaxed font-medium">
                                    {isJobsPage
                                        ? <>Analyze <span className="text-white font-bold">{approvedJobs.length} approved candidates</span> against real-time SERP weakness and keyword feasibility.</>
                                        : <>Use the approved jobs on the left to run or rerun validation, then inspect the strongest opportunities on this page.</>}
                                </p>
                                
                                <Button 
                                    className={cn(
                                        "w-full h-14 rounded-2xl font-black uppercase tracking-widest text-[11px] transition-all active:scale-[0.97] border",
                                        approvedJobs.length > 0 
                                            ? "bg-indigo-500 text-white hover:bg-indigo-400 shadow-[0_0_25px_rgba(99,102,241,0.4)] border-indigo-400/50" 
                                            : "bg-white/5 text-slate-500 border-white/5 cursor-not-allowed"
                                    )}
                                    onClick={isJobsPage ? () => navigate(opportunitiesPagePath) : handleRunWorkflow}
                                    disabled={(isOpportunitiesPage && runningWorkflow) || approvedJobs.length === 0 || !activeProject?.id}
                                >
                                    {isOpportunitiesPage && runningWorkflow
                                        ? <Loader2 className="mr-3 h-5 w-5 animate-spin" />
                                        : <Rocket className={cn("mr-3 h-5 w-5 fill-current", approvedJobs.length > 0 ? "animate-bounce" : "")} />}
                                    {approvedJobs.length > 0
                                        ? (isJobsPage ? `Continue With ${approvedJobs.length} Approved Jobs` : `Validate ${approvedJobs.length} Candidates`)
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
                                    <h3 className="text-3xl font-black text-white tracking-tight leading-tight">High-Achievability Outcomes</h3>
                                    <p className="text-slate-500 text-[13px] font-medium mt-3 leading-relaxed mb-6">
                                        Strategic opportunities derived from validated search data. Every outcome represents a winnable path identified by the engine.
                                    </p>
                                    
                                    <PhaseGuide 
                                        title="Making Decisions"
                                        description="Inspect opportunities to see SERP evidence. 'Promote' winning ideas to start creating content, or 'Release' software tools."
                                        color="emerald"
                                    />
                                </div>
                            </div>
                            
                            <div className="flex flex-wrap items-center gap-3">
                                <div className="px-4 py-2 bg-white/5 rounded-2xl border border-white/10 flex items-center gap-2.5">
                                    <Layers className="h-4 w-4 text-slate-500" />
                                    <span className="text-xs font-black text-slate-300 uppercase tracking-tighter">{workflowTotalJobs} Jobs</span>
                                </div>
                                <div className="px-4 py-2 bg-white/5 rounded-2xl border border-white/10 flex items-center gap-2.5">
                                    <Target className="h-4 w-4 text-slate-500" />
                                    <span className="text-xs font-black text-slate-300 uppercase tracking-tighter">{workflowResults.reduce((s, j) => s + j.candidates.length, 0)} Outcomes</span>
                                </div>
                            </div>
                        </div>

                        <section className="bg-[#0d0d0f] border border-white/5 rounded-[3rem] p-1 shadow-2xl min-h-[800px] flex flex-col relative overflow-hidden">
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
                                        <div className="h-10 px-4 bg-white/5 rounded-2xl border border-white/10 flex items-center gap-3">
                                            <Filter className="h-3.5 w-3.5 text-slate-600" />
                                            <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Active Filters:</span>
                                            <Badge variant="secondary" className="bg-white/10 text-slate-300 text-[9px] uppercase tracking-tighter">{routeFilter === 'all' ? 'No Route Filter' : routeFilter.replace('_', ' ')}</Badge>
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
                                            <Filter className="absolute left-4 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-600 group-focus-within:text-indigo-400 transition-colors" />
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
                                    <div className="py-52 flex flex-col items-center justify-center gap-6 text-slate-600">
                                        <div className="relative">
                                            <div className="absolute inset-0 animate-ping bg-indigo-500/20 rounded-full scale-150" />
                                            <div className="bg-indigo-500/10 p-6 rounded-full border border-indigo-500/20 relative z-10">
                                                <Loader2 className="h-12 w-12 animate-spin text-indigo-500" />
                                            </div>
                                        </div>
                                        <div className="text-center space-y-2">
                                            <p className="text-sm font-black uppercase tracking-[0.3em] text-white">Synthesizing Analysis</p>
                                            <p className="text-xs font-medium text-slate-500 italic">Compiling live SERP snapshots and internal hooks...</p>
                                        </div>
                                    </div>
                                ) : filteredWorkflowResults.length === 0 ? (
                                    <div className="py-40 text-center bg-white/[0.01] border border-dashed border-white/10 rounded-[4rem] group">
                                        <div className="bg-indigo-500/10 w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-8 group-hover:scale-110 transition-transform duration-700 border border-indigo-500/20">
                                            <Target className="h-10 w-10 text-indigo-400" />
                                        </div>
                                        <h3 className="text-white font-black text-2xl tracking-tight mb-4">Awaiting Intelligence</h3>
                                        <p className="text-slate-500 text-[15px] font-medium max-w-sm mx-auto leading-relaxed mb-8">
                                            No opportunities found for the current filter. Initiate a <b>Validation Run</b> in Phase 02 to discover new outcomes.
                                        </p>
                                        <div className="flex justify-center gap-4">
                                            <div className="px-4 py-2 bg-white/5 rounded-xl border border-white/10 text-[10px] font-black uppercase tracking-widest text-slate-500">
                                                Search Active: {candidateSearch || 'None'}
                                            </div>
                                            <div className="px-4 py-2 bg-white/5 rounded-xl border border-white/10 text-[10px] font-black uppercase tracking-widest text-slate-500">
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
                                                        <h3 className="text-lg font-black text-white tracking-tight leading-tight max-w-2xl">{jobResult.job?.job_text}</h3>
                                                    </div>
                                                    <div className="flex-1 h-px bg-white/5 hidden md:block" />
                                                    <Badge variant="outline" className="text-[10px] border-white/10 text-slate-500 uppercase tracking-widest px-3 py-1 self-start md:self-center">
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
                                    <div className="mx-auto max-w-4xl flex items-center justify-between">
                                        <div className="text-[11px] text-slate-500 font-black uppercase tracking-[0.2em]">
                                            Job View <span className="text-white bg-white/10 px-2 py-0.5 rounded ml-2">{workflowPage * workflowPageSize + 1}—{Math.min((workflowPage + 1) * workflowPageSize, workflowTotalJobs)}</span> <span className="mx-2 opacity-30">/</span> {workflowTotalJobs} Total
                                        </div>
                                        <div className="flex items-center gap-6">
                                            <div className="flex items-center gap-3">
                                                <span className="text-[10px] font-black text-slate-600 uppercase tracking-widest">Page Size:</span>
                                                <select value={String(workflowPageSize)} onChange={(e) => { setWorkflowPageSize(Number(e.target.value)); setWorkflowPage(0); }} className="bg-white/5 border border-white/10 rounded-xl text-[10px] font-black px-4 h-10 uppercase text-slate-300 focus:ring-1 focus:ring-indigo-500/50 outline-none hover:bg-white/10 transition-all cursor-pointer">
                                                    <option value="5">05 / PAGE</option>
                                                    <option value="10">10 / PAGE</option>
                                                    <option value="25">25 / PAGE</option>
                                                </select>
                                            </div>
                                            <div className="flex gap-2">
                                                <Button 
                                                    variant="outline" 
                                                    size="icon" 
                                                    className="rounded-xl h-11 w-11 border-white/10 bg-white/5 hover:bg-white/10 transition-all disabled:opacity-20" 
                                                    disabled={workflowPage === 0} 
                                                    onClick={() => setWorkflowPage(p => p - 1)}
                                                >
                                                    <ChevronLeft className="h-5 w-5" />
                                                </Button>
                                                <Button 
                                                    variant="outline" 
                                                    size="icon" 
                                                    className="rounded-xl h-11 w-11 border-white/10 bg-white/5 hover:bg-white/10 transition-all disabled:opacity-20" 
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
        </div>
    )
}

// --- SUB-COMPONENTS ---

function FlowStep({ number, label, active, icon }: { number: string; label: string; active: boolean; icon: React.ReactNode }) {
    return (
        <div className={cn("flex items-center gap-4 transition-all duration-500", active ? "opacity-100 translate-y-0" : "opacity-30 translate-y-1")}>
            <div className={cn(
                "w-9 h-9 rounded-2xl flex items-center justify-center text-xs font-black border transition-all duration-500", 
                active 
                    ? "bg-indigo-500 border-indigo-400 text-white shadow-[0_0_20px_rgba(99,102,241,0.5)] rotate-0" 
                    : "bg-white/5 border-white/10 text-slate-500 rotate-[-5deg]"
            )}>
                {number}
            </div>
            <div className="flex flex-col">
                <span className={cn("text-[11px] font-black uppercase tracking-[0.15em] leading-none mb-1", active ? "text-white" : "text-slate-600")}>{label}</span>
                <span className={cn("flex items-center gap-1.5 text-[9px] font-bold uppercase tracking-widest", active ? "text-indigo-400" : "text-slate-700")}>
                    {active ? <CheckCircle2 className="h-3 w-3" /> : icon} 
                    {active ? 'Complete' : 'Pipeline'}
                </span>
            </div>
        </div>
    )
}

function SelectField({ label, value, onChange, options }: { label: string; value: string; onChange: (v: string) => void; options: { value: string; label: string }[] }) {
    return (
        <div className="space-y-2.5">
            <label className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] pl-1">{label}</label>
            <div className="relative group">
                <select 
                    value={value} 
                    onChange={(e) => onChange(e.target.value)} 
                    className="w-full bg-white/[0.03] border border-white/10 rounded-2xl h-12 px-4 text-xs font-medium text-slate-200 focus:ring-2 focus:ring-indigo-500/40 outline-none hover:bg-white/[0.06] transition-all appearance-none cursor-pointer"
                >
                    {options.map(opt => <option key={opt.value} value={opt.value} className="bg-[#0d0d0f] py-3">{opt.label}</option>)}
                </select>
                <div className="absolute right-4 top-1/2 -translate-y-1/2 pointer-events-none text-slate-500 group-hover:text-slate-300 transition-colors">
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
                className="bg-transparent border-none text-[11px] font-black uppercase tracking-[0.2em] text-slate-400 focus:ring-0 px-4 h-9 outline-none hover:text-white transition-all cursor-pointer appearance-none"
            >
                {options.map(opt => <option key={opt.value} value={opt.value} className="bg-[#0d0d0f]">{opt.label}</option>)}
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
        slate: 'border-white/10 bg-white/5 text-slate-500'
    }

    const achievability = validation_run.achievability_score || 0

    return (
        <div 
            className={cn(
                "rounded-[2.5rem] border transition-all duration-700 relative overflow-hidden group", 
                isExpanded 
                    ? "bg-white/[0.05] border-white/10 ring-1 ring-white/5 shadow-2xl" 
                    : "bg-white/[0.02] border-white/5 hover:border-white/10 hover:bg-white/[0.03] hover:shadow-xl"
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
                                <span className="text-[10px] font-black text-slate-300 uppercase tracking-[0.2em]">Efficiency: {Math.round(achievability * 100)}%</span>
                            </div>
                            {validation_run.freshness_state !== 'fresh' && (
                                <Badge variant="outline" className="text-[9px] border-amber-500/20 text-amber-500 bg-amber-500/5 uppercase font-black tracking-widest px-2.5 py-1">Stale Data</Badge>
                            )}
                        </div>

                        <div>
                            <h4 className="text-2xl font-black text-white tracking-tight leading-tight group-hover:text-indigo-400 transition-colors duration-500 mb-3">{candidate.candidate_text}</h4>
                            <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
                                <div className="flex items-center gap-2.5 text-xs font-bold">
                                    <span className="text-slate-500 uppercase tracking-widest text-[9px] font-black">Core Keyword:</span>
                                    <span className="text-indigo-400 bg-indigo-500/10 px-3 py-1 rounded-lg border border-indigo-500/20">{keyword_pack.primary_keyword || 'Intent Discovery'}</span>
                                </div>
                                <div className="text-[12px] text-slate-400 font-medium">{config.description}</div>
                            </div>
                        </div>
                    </div>

                    <div className="flex items-center gap-3 self-start xl:self-center bg-black/40 p-2 rounded-2xl border border-white/5 shadow-inner">
                        {route !== 'rejected_low_achievability' && (
                             <>
                                {generated_outcome.outcome_type === 'software' ? (
                                    <Button 
                                        size="sm" 
                                        className="bg-indigo-500 hover:bg-indigo-400 text-white rounded-xl h-11 px-6 font-black text-[11px] uppercase tracking-widest transition-all active:scale-[0.95] shadow-lg shadow-indigo-500/20 group/btn" 
                                        onClick={() => onRelease(generated_outcome.id)} 
                                        disabled={isMutating}
                                    >
                                        {isMutating ? <Loader2 className="h-4 w-4 animate-spin mr-3" /> : <Rocket className="h-4 w-4 mr-3 group-hover/btn:translate-y-[-2px] transition-transform" />}
                                        Release Tool
                                    </Button>
                                ) : (
                                    <Button 
                                        size="sm" 
                                        className="bg-white text-black hover:bg-indigo-50 rounded-xl h-11 px-6 font-black text-[11px] uppercase tracking-widest transition-all active:scale-[0.95] shadow-xl group/btn" 
                                        onClick={() => onPersist(generated_outcome.id)} 
                                        disabled={isMutating}
                                    >
                                        {isMutating ? <Loader2 className="h-4 w-4 animate-spin mr-3" /> : <Sparkles className="h-4 w-4 mr-3 text-indigo-600 group-hover/btn:scale-110 transition-transform" />}
                                        Promote Idea
                                    </Button>
                                )}
                                <div className="w-px h-8 bg-white/10 mx-1" />
                                <button 
                                    onClick={() => onReject(candidate.id, 'weak_serp')} 
                                    disabled={isRejecting} 
                                    className="h-11 w-11 flex items-center justify-center rounded-xl bg-red-500/10 text-red-500 border border-red-500/20 hover:bg-red-500 hover:text-white transition-all shadow-xl shadow-red-500/10"
                                    title="Reject Analysis"
                                >
                                    {isRejecting ? <Loader2 className="h-5 w-5 animate-spin" /> : <Ban className="h-5 w-5" />}
                                </button>
                             </>
                        )}
                        <button 
                            onClick={() => onRefresh(validation_run.id)} 
                            disabled={isRefreshing} 
                            className="h-11 w-11 flex items-center justify-center rounded-xl bg-white/5 text-slate-500 border border-white/10 hover:bg-white/10 hover:text-white transition-all"
                            title="Refresh Validation"
                        >
                            {isRefreshing ? <Loader2 className="h-5 w-5 animate-spin" /> : <RefreshCw className="h-5 w-5" />}
                        </button>
                        <button 
                            onClick={onToggle} 
                            className={cn(
                                "h-11 px-4 flex items-center justify-center gap-2 rounded-xl border transition-all text-[11px] font-black uppercase tracking-widest",
                                isExpanded ? "bg-white text-black border-white shadow-xl" : "bg-white/5 text-slate-400 border-white/10 hover:bg-white/10"
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
                            <span className="text-[10px] font-black text-slate-600 uppercase tracking-[0.25em]">Achievability Gradient</span>
                            <span className={cn(
                                "text-[11px] font-black uppercase tracking-widest",
                                achievability > 0.6 ? "text-emerald-400" : achievability > 0.3 ? "text-amber-400" : "text-red-400"
                            )}>
                                {achievability > 0.6 ? 'High Probability' : achievability > 0.3 ? 'Moderate Effort' : 'High Risk'}
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
                    <div className="grid grid-cols-4 gap-12 shrink-0 border-l border-white/10 pl-12">
                        <ScoreMini label="Intent" value={validation_run.intent_match_score} color="indigo" />
                        <ScoreMini label="SERP" value={validation_run.serp_weakness_score} color="emerald" />
                        <ScoreMini label="Gap" value={validation_run.serp_gap_score} color="violet" />
                        <ScoreMini label="Ease" value={getValidationMetric(data.validation_run, 'keyword_difficulty') !== undefined ? 1 - (Number(getValidationMetric(data.validation_run, 'keyword_difficulty')) / 100) : 0} color="amber" />
                    </div>
                </div>
            </div>

            {isExpanded && (
                <div className="px-8 pb-12 pt-6 border-t border-white/10 animate-in fade-in slide-in-from-top-4 duration-500 bg-white/[0.02]">
                    <div className="grid lg:grid-cols-2 gap-16 mt-8">
                        {/* SERP Evidence */}
                        <div className="space-y-8">
                            <div className="flex items-center justify-between">
                                <h5 className="text-[11px] font-black uppercase tracking-[0.3em] text-slate-500 flex items-center gap-3">
                                    <Globe className="h-4 w-4 text-indigo-500" /> SERP Competitive Evidence
                                </h5>
                                <span className="text-[9px] font-black text-slate-700 uppercase tracking-widest">Top 4 Results</span>
                            </div>
                            <div className="space-y-4">
                                {getSerpRows(validation_run).length > 0 ? (
                                    getSerpRows(validation_run).slice(0, 4).map((row: any, i: number) => (
                                        <div key={i} className="group/row bg-black/40 border border-white/5 rounded-2xl p-5 hover:bg-black/60 hover:border-indigo-500/30 transition-all duration-300 shadow-sm relative overflow-hidden">
                                            <div className="absolute top-0 right-0 p-3 opacity-5 group-hover/row:opacity-10 transition-opacity">
                                                <ExternalLink className="h-12 w-12 text-white" />
                                            </div>
                                            <div className="flex items-start justify-between gap-4 mb-3">
                                                <div className="flex items-start gap-4">
                                                    <div className="mt-0.5 w-6 h-6 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-[11px] font-black text-indigo-400 group-hover/row:bg-indigo-500 group-hover/row:text-white transition-all shadow-lg">{i + 1}</div>
                                                    <p className="text-[14px] font-black text-slate-200 group-hover/row:text-white transition-colors leading-snug">{row.title || 'Competitor Snapshot'}</p>
                                                </div>
                                                <a href={row.url} target="_blank" rel="noreferrer" className="shrink-0 text-slate-500 hover:text-indigo-400 p-2 bg-white/5 rounded-xl transition-all border border-white/5 hover:border-indigo-500/20">
                                                    <ExternalLink className="h-4 w-4" />
                                                </a>
                                            </div>
                                            <div className="flex items-center gap-3 pl-10">
                                                <img 
                                                    src={`https://www.google.com/s2/favicons?domain=${new URL(row.url).hostname}&sz=32`} 
                                                    alt="icon" 
                                                    className="w-4 h-4 rounded shadow-sm opacity-60 grayscale group-hover/row:grayscale-0 group-hover/row:opacity-100 transition-all"
                                                />
                                                <p className="text-[10px] text-slate-600 font-bold uppercase tracking-widest">{new URL(row.url).hostname}</p>
                                            </div>
                                        </div>
                                    ))
                                ) : (
                                    <div className="py-20 flex flex-col items-center justify-center bg-white/[0.01] border border-dashed border-white/10 rounded-[2.5rem]">
                                        <Ban className="h-10 w-10 text-slate-800 mb-4" />
                                        <p className="text-[10px] font-black uppercase tracking-widest text-slate-700">No SERP data collected</p>
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Internal Hooks & Analysis Details */}
                        <div className="space-y-12">
                            {/* Infrastructure Section */}
                            <div className="space-y-8">
                                <h5 className="text-[11px] font-black uppercase tracking-[0.3em] text-slate-500 flex items-center gap-3">
                                    <Layers className="h-4 w-4 text-emerald-500" /> Infrastructure Hooks
                                </h5>
                                <div className="space-y-4">
                                    {(internal_link_candidates || []).length > 0 ? (
                                        internal_link_candidates.slice(0, 3).map((link: any) => {
                                            const meta = getInternalLinkMetadata(link)
                                            return (
                                                <div key={link.id} className="bg-black/40 border border-white/5 rounded-2xl p-5 hover:bg-black/60 hover:border-emerald-500/30 transition-all shadow-sm">
                                                    <p className="text-[14px] font-black text-slate-200 mb-4 leading-tight">{String(meta.matched_title || 'Target Neighborhood')}</p>
                                                    <div className="flex items-center justify-between">
                                                        <Badge variant="outline" className="text-[10px] font-black uppercase text-emerald-400 bg-emerald-500/10 border-emerald-500/20 tracking-widest px-3">
                                                            {link.link_role.replace('_', ' ')}
                                                        </Badge>
                                                        <div className="flex items-center gap-4">
                                                            <div className="w-24 h-2 bg-white/5 rounded-full overflow-hidden border border-white/5 p-px">
                                                                <div className="h-full bg-emerald-500 rounded-full shadow-[0_0_10px_rgba(16,185,129,0.5)]" style={{ width: `${(link.match_score || 0) * 100}%` }} />
                                                            </div>
                                                            <span className="text-[11px] text-slate-500 font-black uppercase tabular-nums">{Math.round((link.match_score || 0) * 100)}% Match</span>
                                                        </div>
                                                    </div>
                                                </div>
                                            )
                                        })
                                    ) : (
                                        <div className="py-20 flex flex-col items-center justify-center bg-white/[0.01] border border-dashed border-white/10 rounded-[2.5rem]">
                                            <div className="bg-white/5 p-4 rounded-full mb-4">
                                                <Layers className="h-8 w-8 text-slate-800" />
                                            </div>
                                            <p className="text-[10px] font-black uppercase tracking-widest text-slate-700 text-center px-10">No existing content matches detected for linking.</p>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Secondary Keywords */}
                            <div className="space-y-6">
                                <h5 className="text-[11px] font-black uppercase tracking-[0.3em] text-slate-500 flex items-center gap-3">
                                    <Target className="h-4 w-4 text-indigo-400" /> Semantic Expansion
                                </h5>
                                <div className="flex flex-wrap gap-2.5">
                                    {(keyword_pack.secondary_keywords || []).map((kw: any, i: number) => (
                                        <div key={i} className="px-4 py-2 bg-white/5 border border-white/10 rounded-xl text-[11px] font-bold text-slate-300 hover:text-white hover:bg-white/10 transition-all cursor-default">
                                            {typeof kw === 'string' ? kw : kw.keyword}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    {/* Reason Codes & Tags */}
                    <div className="mt-12 pt-8 border-t border-white/10 flex flex-col md:flex-row md:items-center justify-between gap-6">
                        <div className="flex flex-wrap items-center gap-3">
                            <span className="text-[10px] font-black uppercase tracking-[0.3em] text-slate-600 mr-2">Analysis Intelligence Tags:</span>
                            {(validation_run.validation_reason_codes || []).map((code: string) => (
                                <span key={code} className="px-3 py-1.5 bg-white/5 border border-white/10 rounded-xl text-[10px] font-black uppercase tracking-widest text-slate-400 group-hover:text-indigo-400 transition-colors">
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
            <span className="text-[9px] font-black uppercase tracking-[0.2em] text-slate-500 leading-none">{label}</span>
            <div className="space-y-2">
                <span className="text-[13px] font-black text-white tabular-nums">{(value || 0).toFixed(2)}</span>
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
            <div className="text-[10px] font-black uppercase tracking-[0.25em] text-slate-500">{label}</div>
            <div className="mt-2 text-sm font-semibold text-slate-200">{value}</div>
        </div>
    )
}

function ContextBlock({ label, value }: { label: string; value: string }) {
    return (
        <div className="rounded-2xl border border-white/5 bg-black/30 p-4">
            <div className="text-[10px] font-black uppercase tracking-[0.25em] text-slate-500">{label}</div>
            <p className="mt-2 text-sm leading-relaxed text-slate-300">{value}</p>
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

function PhaseGuide({ title, description, color = 'indigo' }: { title: string; description: string; color?: 'indigo' | 'emerald' | 'purple' }) {
    const [isVisible, setIsVisible] = React.useState(true)
    if (!isVisible) return null

    const colors = {
        indigo: 'bg-indigo-500/10 border-indigo-500/20 text-indigo-200 shadow-indigo-500/5',
        emerald: 'bg-emerald-500/10 border-emerald-500/20 text-emerald-200 shadow-emerald-500/5',
        purple: 'bg-purple-500/10 border-purple-500/20 text-purple-200 shadow-purple-500/5',
    }

    return (
        <div className={cn("p-5 rounded-2xl border flex items-start gap-4 mb-8 animate-in fade-in slide-in-from-left-4 duration-700", colors[color])}>
            <div className="bg-white/10 p-2 rounded-xl mt-0.5">
                <Info className="h-4 w-4" />
            </div>
            <div className="flex-1">
                <h5 className="text-[11px] font-black uppercase tracking-widest mb-1">{title}</h5>
                <p className="text-[12px] font-medium leading-relaxed opacity-80">{description}</p>
            </div>
            <button onClick={() => setIsVisible(false)} className="text-white/20 hover:text-white transition-colors">
                <XCircle className="h-4 w-4" />
            </button>
        </div>
    )
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
