import * as React from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Loader2, Plus, Trash2, CheckCircle2, Globe2, Layers3, BookOpenText, Info, X } from 'lucide-react'
import { toast } from 'sonner'
import { useProject } from '@/context/project-context'
import { useAuth } from '@/context/auth-context'
import { commandCenterService } from '@/services/command-center.service'
import { researchTopicsService } from '@/services/research-topics.service'
import type { Project } from '@/types'
import type { ProjectCategory, TopicCandidate, TopicDraft } from '@/types/command-center'
import type { ResearchTopic } from '@/types/research'
import { ScrollArea } from '@/components/ui/scroll-area'

const inputSelectClasses =
    "h-11 w-full rounded-xl border border-border bg-muted/50 pl-10 pr-4 text-sm text-foreground outline-none transition focus:border-blue-400/30 hover:border-border disabled:cursor-not-allowed disabled:opacity-50"
const smallSelectClasses =
    "h-10 rounded-lg border border-border bg-muted/50 px-3 text-sm text-foreground outline-none transition focus:border-blue-400/30 hover:border-border disabled:cursor-not-allowed disabled:opacity-50"

type TopicInputMode = 'ai' | 'news' | 'manual'

function getProjectDescription(project: Project | null) {
    return project?.site_description || project?.websiteDescription || 'Choose a website to load its research workspace.'
}

export function Landing() {
    const navigate = useNavigate()
    const { user } = useAuth()
    const { projects, activeProject, setActiveProject, isLoading: projectLoading } = useProject()

    const [categories, setCategories] = React.useState<ProjectCategory[]>([])
    const [primaryCategoryId, setPrimaryCategoryId] = React.useState('')
    const [secondaryCategoryId, setSecondaryCategoryId] = React.useState('')
    const [topicCandidates, setTopicCandidates] = React.useState<TopicCandidate[]>([])
    const [selectedTopicIds, setSelectedTopicIds] = React.useState<Set<string>>(new Set())
    const [manualTopic, setManualTopic] = React.useState('')
    const [topicInputMode, setTopicInputMode] = React.useState<TopicInputMode>('ai')
    const [manualModalOpen, setManualModalOpen] = React.useState(false)
    const [showSiteInfo, setShowSiteInfo] = React.useState(false)

    const [categoryLoading, setCategoryLoading] = React.useState(false)
    const [topicsLoading, setTopicsLoading] = React.useState(false)
    const [aiLoading, setAiLoading] = React.useState(false)
    const [newsLoading, setNewsLoading] = React.useState(false)
    const [manualLoading, setManualLoading] = React.useState(false)
    const [startLoading, setStartLoading] = React.useState(false)
    const [researchTopicByTitle, setResearchTopicByTitle] = React.useState<Record<string, ResearchTopic>>({})
    const workspaceKeyRef = React.useRef<string | null>(null)

    const primaryCategories = React.useMemo(
        () => categories.filter((category) => category.level === 1),
        [categories],
    )

    const secondaryCategories = React.useMemo(
        () => categories.filter((category) => category.level === 2 && category.parent_category_id === primaryCategoryId),
        [categories, primaryCategoryId],
    )

    const activePrimaryCategory = React.useMemo(
        () => primaryCategories.find((category) => category.id === primaryCategoryId) || null,
        [primaryCategories, primaryCategoryId],
    )

    const activeSecondaryCategory = React.useMemo(
        () => secondaryCategories.find((category) => category.id === secondaryCategoryId) || null,
        [secondaryCategories, secondaryCategoryId],
    )

    React.useEffect(() => {
        const loadCategories = async () => {
            if (!activeProject) {
                setCategories([])
                setPrimaryCategoryId('')
                setSecondaryCategoryId('')
                return
            }

            setCategoryLoading(true)

            try {
                const loadedCategories = await commandCenterService.listCategories(activeProject.id)
                setCategories(loadedCategories)

                const firstPrimary = loadedCategories.find((category) => category.level === 1)
                const firstSecondary = loadedCategories.find(
                    (category) => category.level === 2 && category.parent_category_id === firstPrimary?.id,
                )

                setPrimaryCategoryId(firstPrimary?.id || '')
                setSecondaryCategoryId(firstSecondary?.id || '')
            } catch (error) {
                console.error(error)
                const msg = (error as any)?.message || ''
                if (msg.toLowerCase().includes('project_categories') || msg.toLowerCase().includes('relation')) {
                    toast.error('Categories are not set up yet. Apply the Supabase migration for the command center tables.')
                } else {
                    toast.error('Unable to load categories for this project.')
                }
            } finally {
                setCategoryLoading(false)
            }
        }

        loadCategories()
    }, [activeProject])

    React.useEffect(() => {
        if (!secondaryCategories.length) {
            setSecondaryCategoryId('')
            return
        }

        const exists = secondaryCategories.some((category) => category.id === secondaryCategoryId)
        if (!exists) {
            setSecondaryCategoryId(secondaryCategories[0].id)
        }
    }, [secondaryCategories, secondaryCategoryId])

    React.useEffect(() => {
        const loadTopics = async () => {
            if (!activeProject || !secondaryCategoryId) {
                setTopicCandidates([])
                setSelectedTopicIds(new Set())
                workspaceKeyRef.current = null
                return
            }

            setTopicsLoading(true)
            const workspaceKey = `${activeProject.id}:${secondaryCategoryId}`

            try {
                const loadedTopics = await commandCenterService.listTopicCandidates(activeProject.id, secondaryCategoryId)
                setTopicCandidates((current) => {
                    // For the same workspace, append/merge instead of replacing.
                    // On workspace switch (project/subcategory), replace with fresh results.
                    if (workspaceKeyRef.current !== workspaceKey) {
                        workspaceKeyRef.current = workspaceKey
                        return loadedTopics
                    }

                    const byId = new Map<string, TopicCandidate>()
                    current.forEach((topic) => byId.set(topic.id, topic))
                    loadedTopics.forEach((topic) => byId.set(topic.id, topic))
                    return Array.from(byId.values())
                })
            } catch (error) {
                console.error(error)
                const msg = (error as any)?.message || ''
                if (msg.toLowerCase().includes('project_topic_candidates') || msg.toLowerCase().includes('relation')) {
                    toast.error('Topic workspace is not set up yet. Apply the Supabase migration for the command center tables.')
                } else {
                    toast.error('Unable to load starter topics for this category.')
                }
            } finally {
                setTopicsLoading(false)
            }
        }

        loadTopics()
    }, [activeProject, secondaryCategoryId])

    React.useEffect(() => {
        const normalizeTitle = (value: string) => value.trim().toLowerCase()

        const loadResearchTopicStatuses = async () => {
            if (!activeProject || !primaryCategoryId) {
                setResearchTopicByTitle({})
                return
            }

            try {
                let page = 1
                const size = 100
                const allItems: ResearchTopic[] = []

                while (true) {
                    const response = await researchTopicsService.listResearchTopics({
                        page,
                        size,
                        project_id: activeProject.id,
                        primary_category_id: primaryCategoryId,
                        secondary_category_id: secondaryCategoryId || undefined,
                        order_by: 'created_at',
                        order_direction: 'desc',
                    })
                    allItems.push(...(response.items || []))
                    if (!response.has_next) break
                    page += 1
                }

                const byTitle: Record<string, ResearchTopic> = {}
                for (const item of allItems) {
                    const key = normalizeTitle(item.title || '')
                    if (!key || byTitle[key]) continue
                    byTitle[key] = item
                }
                setResearchTopicByTitle(byTitle)
            } catch (error) {
                console.error('Failed to load research topic statuses for landing page:', error)
                setResearchTopicByTitle({})
            }
        }

        void loadResearchTopicStatuses()
    }, [activeProject?.id, primaryCategoryId, secondaryCategoryId])

    React.useEffect(() => {
        setSelectedTopicIds((current) => {
            const next = new Set<string>()
            topicCandidates.forEach((topic) => {
                if (current.has(topic.id)) {
                    next.add(topic.id)
                }
            })
            return next
        })
    }, [topicCandidates])

    const addTopicsToWorkspace = async (drafts: TopicDraft[], source: 'ai' | 'news') => {
        if (!activeProject || !user || !activePrimaryCategory || !activeSecondaryCategory) {
            return
        }

        const existingTitles = new Set(topicCandidates.map((topic) => topic.title.trim().toLowerCase()))
        const cleaned = drafts
            .map((draft) => ({
                ...draft,
                title: draft.title.trim(),
            }))
            .filter((draft, index, all) => draft.title.length > 0 && all.findIndex((item) => item.title.toLowerCase() === draft.title.toLowerCase()) === index)
            .filter((draft) => !existingTitles.has(draft.title.toLowerCase()))

        if (!cleaned.length) {
            toast.message('Those topics are already in the workspace.')
            return
        }

        const inserted = await commandCenterService.createTopicCandidates(
            cleaned.map((draft) => ({
                project_id: activeProject.id,
                user_id: user.id,
                primary_category_id: activePrimaryCategory.id,
                secondary_category_id: activeSecondaryCategory.id,
                title: draft.title,
                rationale: draft.rationale ?? null,
                intent_bucket: draft.intent_bucket ?? null,
                decision_focus: draft.decision_focus ?? null,
                angle_question: draft.angle_question ?? null,
                value_layer_tags: draft.value_layer_tags ?? null,
                related_terms: draft.related_terms ?? null,
                source_signals: draft.source_signals ?? null,
                topic_source: source,
                source_label: commandCenterService.getSourceLabel(source),
            })),
        )

        setTopicCandidates((current) => [...current, ...inserted])
        setSelectedTopicIds((current) => {
            const next = new Set(current)
            inserted.forEach((topic) => next.add(topic.id))
            return next
        })
        toast.success(`${inserted.length} topics added to the workspace.`)
    }

    const handleGenerateAiTopics = async () => {
        if (!activeProject || !activePrimaryCategory) {
            return
        }

        setAiLoading(true)
        try {
            const titles = await commandCenterService.generateAiTopics({
                project: activeProject,
                primaryCategory: activePrimaryCategory,
                secondaryCategory: activeSecondaryCategory,
            })
            await addTopicsToWorkspace(titles, 'ai')
        } catch (error: any) {
            console.error(error)
            toast.error(error?.response?.data?.error || error?.message || 'AI topic generation failed.')
        } finally {
            setAiLoading(false)
        }
    }

    const handleGenerateNewsTopics = async () => {
        if (!activeProject || !activePrimaryCategory) {
            return
        }

        setNewsLoading(true)
        try {
            const titles = await commandCenterService.generateNewsTopics({
                project: activeProject,
                primaryCategory: activePrimaryCategory,
                secondaryCategory: activeSecondaryCategory,
            })
            await addTopicsToWorkspace(titles, 'news')
        } catch (error: any) {
            console.error(error)
            toast.error(error?.response?.data?.error || error?.message || 'News-based topic generation failed.')
        } finally {
            setNewsLoading(false)
        }
    }

    const handleAddManualTopic = async () => {
        if (!activeProject || !user || !activePrimaryCategory || !activeSecondaryCategory || !manualTopic.trim()) {
            return false
        }

        const normalizedTitle = manualTopic.trim().toLowerCase()
        if (topicCandidates.some((topic) => topic.title.trim().toLowerCase() === normalizedTitle)) {
            toast.message('That topic is already in the workspace.')
            return false
        }

        setManualLoading(true)
        try {
            const inserted = await commandCenterService.createTopicCandidate({
                project_id: activeProject.id,
                user_id: user.id,
                primary_category_id: activePrimaryCategory.id,
                secondary_category_id: activeSecondaryCategory.id,
                title: manualTopic.trim(),
                topic_source: 'manual',
                source_label: commandCenterService.getSourceLabel('manual'),
            })

            setTopicCandidates((current) => [...current, inserted])
            setSelectedTopicIds((current) => new Set(current).add(inserted.id))
            setManualTopic('')
            return true
        } catch (error) {
            console.error(error)
            toast.error('Unable to save that topic.')
            return false
        } finally {
            setManualLoading(false)
        }
    }

    const handleRemoveTopic = async (topicId: string) => {
        try {
            await commandCenterService.deleteTopicCandidate(topicId)
            setTopicCandidates((current) => current.filter((topic) => topic.id !== topicId))
            setSelectedTopicIds((current) => {
                const next = new Set(current)
                next.delete(topicId)
                return next
            })
        } catch (error) {
            console.error(error)
            toast.error('Unable to remove this topic.')
        }
    }

    const toggleTopic = (topicId: string) => {
        setSelectedTopicIds((current) => {
            const next = new Set(current)
            if (next.has(topicId)) {
                next.delete(topicId)
            } else {
                next.add(topicId)
            }
            return next
        })
    }

    const handleSelectAll = () => {
        if (selectedTopicIds.size === topicCandidates.length) {
            setSelectedTopicIds(new Set())
            return
        }

        setSelectedTopicIds(new Set(topicCandidates.map((topic) => topic.id)))
    }

    const handleStartResearch = async () => {
        if (!activeProject || !user || !activePrimaryCategory || selectedTopicIds.size === 0) {
            return
        }

        const selectedTopics = topicCandidates.filter((topic) => selectedTopicIds.has(topic.id))

        setStartLoading(true)
        try {
            const created = await commandCenterService.startResearch({
                project: activeProject,
                userId: user.id,
                primaryCategory: activePrimaryCategory,
                secondaryCategory: activeSecondaryCategory,
                topics: selectedTopics,
            })

            toast.success(`${created.length} research ${created.length === 1 ? 'topic' : 'topics'} started.`)

            if (created.length === 1) {
                navigate(`/research/${created[0].id}`)
            } else {
                navigate('/research')
            }
        } catch (error) {
            console.error(error)
            toast.error('Unable to start research right now.')
        } finally {
            setStartLoading(false)
        }
    }

    const handleAddItemsToWorkspace = async () => {
        if (topicInputMode === 'ai') {
            await handleGenerateAiTopics()
            return
        }

        if (topicInputMode === 'news') {
            await handleGenerateNewsTopics()
            return
        }

        setManualModalOpen(true)
    }

    const selectionLocked = !activeProject || categoryLoading || projectLoading
    const addItemsLoading = aiLoading || newsLoading || manualLoading

    return (
        <div className="min-h-screen bg-background">
            <div className="mx-auto max-w-4xl px-6 py-10 lg:py-16">

                {/* Page header */}
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.25 }}
                >
                    <h1 className="text-2xl font-semibold tracking-tight text-foreground">
                        New Research
                    </h1>
                    <p className="mt-2 text-sm text-muted-foreground">
                        Select a website, narrow the category, add topics, and start.
                    </p>
                </motion.div>

                {/* Configuration */}
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.25, delay: 0.04 }}
                    className="mt-10 space-y-6"
                >
                    {/* Website */}
                    <div>
                        <label className="block">
                            <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                                Website
                            </span>
                            <div className="mt-2 flex items-center gap-2">
                                <div className="relative flex-1">
                                    <Globe2 className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                                    <select
                                        className={inputSelectClasses}
                                        value={activeProject?.id || ''}
                                        onChange={(event) => {
                                            const nextProject = projects.find((project) => project.id === event.target.value) || null
                                            setActiveProject(nextProject)
                                        }}
                                    >
                                        <option value="">Select a website</option>
                                        {projects.map((project) => (
                                            <option key={project.id} value={project.id}>
                                                {project.domain || project.app_name || 'Untitled Project'}
                                            </option>
                                        ))}
                                    </select>
                                </div>

                                <button
                                    type="button"
                                    onClick={() => setShowSiteInfo(!showSiteInfo)}
                                    disabled={!activeProject}
                                    className="inline-flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-border bg-muted/50 text-muted-foreground transition hover:border-border hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
                                    aria-label={showSiteInfo ? 'Hide website details' : 'Show website details'}
                                >
                                    {showSiteInfo ? <X className="h-4 w-4" /> : <Info className="h-4 w-4" />}
                                </button>
                            </div>
                        </label>

                        <AnimatePresence>
                            {showSiteInfo && activeProject && (
                                <motion.p
                                    initial={{ opacity: 0, height: 0 }}
                                    animate={{ opacity: 1, height: 'auto' }}
                                    exit={{ opacity: 0, height: 0 }}
                                    transition={{ duration: 0.2 }}
                                    className="mt-2 overflow-hidden text-sm leading-6 text-muted-foreground"
                                >
                                    {getProjectDescription(activeProject)}
                                </motion.p>
                            )}
                        </AnimatePresence>
                    </div>

                    {/* Categories */}
                    <div className="grid gap-4 sm:grid-cols-2">
                        <label className="block">
                            <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                                Category
                            </span>
                            <div className="mt-2 relative">
                                <Layers3 className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                                <select
                                    className={inputSelectClasses}
                                    value={primaryCategoryId}
                                    onChange={(event) => setPrimaryCategoryId(event.target.value)}
                                    disabled={selectionLocked || primaryCategories.length === 0}
                                >
                                    <option value="">Select a category</option>
                                    {primaryCategories.map((category) => (
                                        <option key={category.id} value={category.id}>
                                            {category.name}
                                        </option>
                                    ))}
                                </select>
                            </div>
                        </label>

                        <label className="block">
                            <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                                Subcategory
                            </span>
                            <div className="mt-2 relative">
                                <BookOpenText className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                                <select
                                    className={inputSelectClasses}
                                    value={secondaryCategoryId}
                                    onChange={(event) => setSecondaryCategoryId(event.target.value)}
                                    disabled={selectionLocked || secondaryCategories.length === 0}
                                >
                                    <option value="">Select a subcategory</option>
                                    {secondaryCategories.map((category) => (
                                        <option key={category.id} value={category.id}>
                                            {category.name}
                                        </option>
                                    ))}
                                </select>
                            </div>
                        </label>
                    </div>
                </motion.div>

                {/* Research Topics */}
                <motion.section
                    initial={{ opacity: 0, y: 14 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: 0.08 }}
                    className="mt-14"
                >
                    <div className="flex items-start justify-between gap-4">
                        <div>
                            <h2 className="text-lg font-medium text-foreground">
                                Research Topics
                            </h2>
                            <p className="mt-1 text-xs text-muted-foreground">
                                {topicCandidates.length > 0
                                    ? `${selectedTopicIds.size} of ${topicCandidates.length} selected`
                                    : 'Add topics below to get started'}
                            </p>
                        </div>

                        <div className="flex items-center gap-2">
                            <select
                                value={topicInputMode}
                                onChange={(event) => setTopicInputMode(event.target.value as TopicInputMode)}
                                disabled={selectionLocked || !activePrimaryCategory}
                                className={smallSelectClasses}
                            >
                                <option value="ai">AI Generated</option>
                                <option value="news">Hot in the News</option>
                                <option value="manual">Manual Entry</option>
                            </select>

                            <button
                                type="button"
                                onClick={handleAddItemsToWorkspace}
                                disabled={selectionLocked || !activePrimaryCategory || addItemsLoading}
                                className="inline-flex h-10 items-center gap-1.5 rounded-lg border border-border bg-muted/50 px-3.5 text-sm text-muted-foreground transition hover:border-border hover:bg-muted hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
                            >
                                {addItemsLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />}
                                <span>Add</span>
                            </button>
                        </div>
                    </div>

                    <div className="mt-4 border-t border-border" />

                    {/* Topic list */}
                    <div className="mt-4 pb-24">
                        {topicsLoading ? (
                            <div className="space-y-2">
                                {[1, 2, 3].map((item) => (
                                    <div key={item} className="h-10 animate-pulse rounded-lg bg-muted" />
                                ))}
                            </div>
                        ) : topicCandidates.length === 0 ? (
                            <div className="py-14 text-center">
                                <p className="text-sm text-muted-foreground">
                                    No topics yet. Choose a source and add some above.
                                </p>
                            </div>
                        ) : (
                            <ScrollArea className="max-h-[400px]">
                                <div className="divide-y divide-border pb-8">
                                    {topicCandidates.map((topic) => {
                                        const checked = selectedTopicIds.has(topic.id)
                                        const statusTopic = researchTopicByTitle[topic.title.trim().toLowerCase()]
                                        const subtopicsCount = Number(statusTopic?.subtopics_count || 0)
                                        const ideasCount = Number(statusTopic?.content_ideas_count || 0)
                                        const inLibraryCount = Number(statusTopic?.in_library_count || 0)
                                        const isFullyResearched = Boolean(statusTopic?.all_subtopics_researched)
                                        const hasAnyProgress = subtopicsCount > 0 || ideasCount > 0 || inLibraryCount > 0 || Boolean(statusTopic?.has_underlying_data)
                                        const statusLabel = isFullyResearched
                                            ? 'Researched'
                                            : hasAnyProgress
                                                ? 'Researching'
                                                : 'Not Started'
                                        const statusClass = isFullyResearched
                                            ? 'text-emerald-500 dark:text-emerald-400'
                                            : hasAnyProgress
                                                ? 'text-amber-500 dark:text-amber-400'
                                                : 'text-muted-foreground'
                                        return (
                                            <div
                                                key={topic.id}
                                                className={`flex items-center gap-3 py-2.5 transition ${
                                                    checked ? 'bg-primary/[0.05]' : ''
                                                }`}
                                            >
                                                <input
                                                    type="checkbox"
                                                    checked={checked}
                                                    onChange={() => toggleTopic(topic.id)}
                                                    className="h-4 w-4 shrink-0 rounded border-border bg-transparent text-primary focus:ring-ring focus:ring-offset-0"
                                                />
                                                <span className={`min-w-0 flex-1 truncate text-sm ${checked ? 'text-foreground' : 'text-muted-foreground'}`}>
                                                    {topic.title}
                                                </span>
                                                <span className={`shrink-0 text-xs font-medium ${statusClass}`}>
                                                    {statusLabel}
                                                </span>
                                                <button
                                                    type="button"
                                                    onClick={(event) => {
                                                        event.preventDefault()
                                                        void handleRemoveTopic(topic.id)
                                                    }}
                                                    className="shrink-0 rounded-md p-1 text-muted-foreground transition hover:bg-muted hover:text-destructive"
                                                    aria-label={`Remove ${topic.title}`}
                                                >
                                                    <Trash2 className="h-3.5 w-3.5" />
                                                </button>
                                            </div>
                                        )
                                    })}
                                </div>
                            </ScrollArea>
                        )}
                    </div>

                    {/* Actions */}
                    <div className="sticky bottom-4 z-20 mt-4 flex items-center justify-between rounded-xl border border-border bg-background/95 px-3 py-3 shadow-lg backdrop-blur supports-[backdrop-filter]:bg-background/85">
                        <button
                            type="button"
                            onClick={handleSelectAll}
                            disabled={topicCandidates.length === 0}
                            className="inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-xs text-muted-foreground transition hover:text-foreground disabled:cursor-not-allowed disabled:opacity-40"
                        >
                            <CheckCircle2 className="h-3.5 w-3.5" />
                            {selectedTopicIds.size === topicCandidates.length && topicCandidates.length > 0 ? 'Deselect all' : 'Select all'}
                        </button>

                        <button
                            type="button"
                            onClick={handleStartResearch}
                            disabled={selectedTopicIds.size === 0 || startLoading || !activeProject}
                            className="inline-flex h-11 items-center gap-2 rounded-xl bg-primary px-5 text-sm font-medium text-primary-foreground transition hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-40"
                        >
                            {startLoading ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                                `Start Research${selectedTopicIds.size > 0 ? ` (${selectedTopicIds.size})` : ''}`
                            )}
                        </button>
                    </div>
                </motion.section>
            </div>

            {/* Manual topic modal */}
            {manualModalOpen && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 px-4 backdrop-blur-sm">
                    <div className="w-full max-w-md rounded-2xl border border-border bg-background p-6">
                        <div className="flex items-center justify-between">
                            <p className="text-base font-medium text-foreground">Add Manual Topic</p>
                            <button
                                type="button"
                                onClick={() => setManualModalOpen(false)}
                                className="rounded-lg p-1.5 text-muted-foreground transition hover:bg-muted hover:text-foreground"
                            >
                                <X className="h-4 w-4" />
                            </button>
                        </div>

                        <div className="mt-5 space-y-4">
                            <input
                                autoFocus
                                value={manualTopic}
                                onChange={(event) => setManualTopic(event.target.value)}
                                onKeyDown={(event) => {
                                    if (event.key === 'Enter') {
                                        void handleAddManualTopic()
                                    }
                                }}
                                placeholder="Enter a research topic"
                                className="h-11 w-full rounded-xl border border-border bg-muted/50 px-4 text-sm text-foreground outline-none transition placeholder:text-muted-foreground focus:border-ring/50"
                            />

                            <div className="flex justify-end gap-2">
                                <button
                                    type="button"
                                    onClick={() => setManualModalOpen(false)}
                                    className="h-10 rounded-xl border border-border px-4 text-sm text-muted-foreground transition hover:bg-muted hover:text-foreground"
                                >
                                    Cancel
                                </button>
                                <button
                                    type="button"
                                    onClick={async () => {
                                        const added = await handleAddManualTopic()
                                        if (added) {
                                            setManualModalOpen(false)
                                        }
                                    }}
                                    disabled={!manualTopic.trim() || manualLoading || selectionLocked}
                                    className="inline-flex h-10 items-center gap-1.5 rounded-xl bg-primary px-4 text-sm font-medium text-primary-foreground transition hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-40"
                                >
                                    {manualLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />}
                                    <span>Add Topic</span>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}
