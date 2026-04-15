import * as React from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Loader2, Plus, Sparkles, Newspaper, Trash2, CheckCircle2, Globe2, Layers3, BookOpenText } from 'lucide-react'
import { toast } from 'sonner'
import { useProject } from '@/context/project-context'
import { useAuth } from '@/context/auth-context'
import { commandCenterService } from '@/services/command-center.service'
import type { Project } from '@/types'
import type { ProjectCategory, TopicCandidate } from '@/types/command-center'

const selectClasses = "h-14 w-full rounded-2xl border border-white/10 bg-[#0d1728]/90 px-4 text-sm text-white outline-none transition focus:border-blue-400/40"
const panelClasses = "rounded-[28px] border border-white/8 bg-white/[0.04] shadow-[0_30px_80px_rgba(4,10,24,0.45)] backdrop-blur-xl"

function getProjectLabel(project: Project | null) {
    return project?.domain || project?.app_name || 'Select a website'
}

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

    const [categoryLoading, setCategoryLoading] = React.useState(false)
    const [topicsLoading, setTopicsLoading] = React.useState(false)
    const [aiLoading, setAiLoading] = React.useState(false)
    const [newsLoading, setNewsLoading] = React.useState(false)
    const [manualLoading, setManualLoading] = React.useState(false)
    const [startLoading, setStartLoading] = React.useState(false)

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
                toast.error('Unable to load categories for this project.')
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
                return
            }

            setTopicsLoading(true)

            try {
                const loadedTopics = await commandCenterService.listTopicCandidates(activeProject.id, secondaryCategoryId)
                setTopicCandidates(loadedTopics)
            } catch (error) {
                console.error(error)
                toast.error('Unable to load starter topics for this category.')
            } finally {
                setTopicsLoading(false)
            }
        }

        loadTopics()
    }, [activeProject, secondaryCategoryId])

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

    const addTopicsToWorkspace = async (titles: string[], source: 'ai' | 'news') => {
        if (!activeProject || !user || !activePrimaryCategory || !activeSecondaryCategory) {
            return
        }

        const existingTitles = new Set(topicCandidates.map((topic) => topic.title.trim().toLowerCase()))
        const cleaned = titles
            .map((title) => title.trim())
            .filter((title, index, all) => title.length > 0 && all.findIndex((item) => item.toLowerCase() === title.toLowerCase()) === index)
            .filter((title) => !existingTitles.has(title.toLowerCase()))

        if (!cleaned.length) {
            toast.message('Those topics are already in the workspace.')
            return
        }

        const inserted = await commandCenterService.createTopicCandidates(
            cleaned.map((title) => ({
                project_id: activeProject.id,
                user_id: user.id,
                primary_category_id: activePrimaryCategory.id,
                secondary_category_id: activeSecondaryCategory.id,
                title,
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
            return
        }

        const normalizedTitle = manualTopic.trim().toLowerCase()
        if (topicCandidates.some((topic) => topic.title.trim().toLowerCase() === normalizedTitle)) {
            toast.message('That topic is already in the workspace.')
            return
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
        } catch (error) {
            console.error(error)
            toast.error('Unable to save that topic.')
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

    const selectionLocked = !activeProject || categoryLoading || projectLoading

    return (
        <div className="relative min-h-screen overflow-hidden bg-[#07111e]">
            <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(56,189,248,0.16),transparent_24%),radial-gradient(circle_at_bottom_left,rgba(59,130,246,0.12),transparent_28%),linear-gradient(180deg,#07111e_0%,#091426_100%)]" />

            <div className="relative mx-auto max-w-7xl px-4 pb-12 pt-6 sm:px-6 lg:px-10 lg:pt-10">
                <motion.div
                    initial={{ opacity: 0, y: 18 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.35 }}
                    className="mb-8"
                >
                    <p className="mb-3 text-[11px] uppercase tracking-[0.34em] text-slate-500">Research Workflow</p>
                    <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                        <div className="max-w-3xl">
                            <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl lg:text-5xl">
                                Build the next research queue without the noise.
                            </h1>
                            <p className="mt-3 max-w-2xl text-sm leading-7 text-slate-400 sm:text-base">
                                Choose the website, narrow the category, shape the topic list, and launch one or many research runs from a single workspace.
                            </p>
                        </div>

                        <div className="rounded-3xl border border-white/8 bg-white/[0.035] px-5 py-4 text-sm text-slate-300">
                            <p className="text-[11px] uppercase tracking-[0.28em] text-slate-500">Active Website</p>
                            <p className="mt-2 text-lg font-medium text-white">{getProjectLabel(activeProject)}</p>
                            <p className="mt-2 max-w-md text-sm text-slate-400">{getProjectDescription(activeProject)}</p>
                        </div>
                    </div>
                </motion.div>

                <motion.section
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4, delay: 0.05 }}
                    className={`${panelClasses} mb-8 p-5 sm:p-7`}
                >
                    <div className="grid gap-4 xl:grid-cols-[1.15fr_0.85fr_0.85fr]">
                        <label className="space-y-2">
                            <span className="text-sm font-medium text-slate-300">Website</span>
                            <div className="relative">
                                <Globe2 className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                                <select
                                    className={`${selectClasses} pl-11`}
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
                        </label>

                        <label className="space-y-2">
                            <span className="text-sm font-medium text-slate-300">Category</span>
                            <div className="relative">
                                <Layers3 className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                                <select
                                    className={`${selectClasses} pl-11`}
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

                        <label className="space-y-2">
                            <span className="text-sm font-medium text-slate-300">Subcategory</span>
                            <div className="relative">
                                <BookOpenText className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                                <select
                                    className={`${selectClasses} pl-11`}
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
                </motion.section>

                <motion.section
                    initial={{ opacity: 0, y: 24 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.45, delay: 0.1 }}
                    className="mb-8 grid gap-4 lg:grid-cols-[0.95fr_0.95fr_1.1fr]"
                >
                    <button
                        type="button"
                        onClick={handleGenerateAiTopics}
                        disabled={selectionLocked || !activePrimaryCategory || aiLoading}
                        className={`${panelClasses} flex min-h-[180px] flex-col items-start justify-between p-6 text-left transition hover:border-blue-400/20 hover:bg-blue-500/[0.06] disabled:cursor-not-allowed disabled:opacity-50`}
                    >
                        <div className="rounded-2xl bg-blue-500/15 p-3 text-blue-200">
                            {aiLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Sparkles className="h-5 w-5" />}
                        </div>
                        <div>
                            <p className="text-lg font-medium text-white">AI Generated Topics</p>
                            <p className="mt-2 text-sm leading-6 text-slate-400">
                                Pull in broad topics aligned with the selected website and category path.
                            </p>
                        </div>
                    </button>

                    <button
                        type="button"
                        onClick={handleGenerateNewsTopics}
                        disabled={selectionLocked || !activePrimaryCategory || newsLoading}
                        className={`${panelClasses} flex min-h-[180px] flex-col items-start justify-between p-6 text-left transition hover:border-cyan-300/20 hover:bg-cyan-400/[0.05] disabled:cursor-not-allowed disabled:opacity-50`}
                    >
                        <div className="rounded-2xl bg-cyan-400/15 p-3 text-cyan-100">
                            {newsLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : <Newspaper className="h-5 w-5" />}
                        </div>
                        <div>
                            <p className="text-lg font-medium text-white">Hot in the News</p>
                            <p className="mt-2 text-sm leading-6 text-slate-400">
                                Use the trend engine to surface timely topics with current momentum.
                            </p>
                        </div>
                    </button>

                    <div className={`${panelClasses} flex min-h-[180px] flex-col justify-between p-6`}>
                        <div>
                            <p className="text-lg font-medium text-white">Manual Topic</p>
                            <p className="mt-2 text-sm leading-6 text-slate-400">
                                Add your own topic to the workspace, then include it in the next batch.
                            </p>
                        </div>

                        <div className="mt-6 flex flex-col gap-3 sm:flex-row">
                            <input
                                value={manualTopic}
                                onChange={(event) => setManualTopic(event.target.value)}
                                onKeyDown={(event) => {
                                    if (event.key === 'Enter') {
                                        void handleAddManualTopic()
                                    }
                                }}
                                placeholder="Type a broad topic"
                                className="h-14 flex-1 rounded-2xl border border-white/10 bg-[#0d1728]/90 px-4 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-blue-400/40"
                                disabled={selectionLocked}
                            />
                            <button
                                type="button"
                                onClick={handleAddManualTopic}
                                disabled={selectionLocked || !manualTopic.trim() || manualLoading}
                                className="inline-flex h-14 items-center justify-center rounded-2xl bg-white px-5 text-sm font-medium text-slate-950 transition hover:bg-slate-200 disabled:cursor-not-allowed disabled:opacity-50"
                            >
                                {manualLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
                            </button>
                        </div>
                    </div>
                </motion.section>

                <motion.section
                    initial={{ opacity: 0, y: 24 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.45, delay: 0.15 }}
                    className={`${panelClasses} overflow-hidden`}
                >
                    <div className="border-b border-white/8 px-5 py-5 sm:px-7">
                        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                            <div>
                                <p className="text-sm font-medium text-white">Topic Workspace</p>
                                <p className="mt-2 text-sm text-slate-400">
                                    Preloaded topics appear here. Add more from AI, news, or your own manual input, then choose the ones to start.
                                </p>
                            </div>

                            <div className="flex flex-wrap gap-2 text-xs text-slate-400">
                                {activePrimaryCategory && (
                                    <span className="rounded-full border border-white/10 px-3 py-1.5">
                                        {activePrimaryCategory.name}
                                    </span>
                                )}
                                {activeSecondaryCategory && (
                                    <span className="rounded-full border border-blue-400/15 bg-blue-500/10 px-3 py-1.5 text-blue-100">
                                        {activeSecondaryCategory.name}
                                    </span>
                                )}
                            </div>
                        </div>
                    </div>

                    <div className="px-5 py-5 sm:px-7">
                        <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                            <button
                                type="button"
                                onClick={handleSelectAll}
                                disabled={topicCandidates.length === 0}
                                className="inline-flex items-center gap-2 rounded-full border border-white/10 px-4 py-2 text-xs uppercase tracking-[0.24em] text-slate-400 transition hover:border-white/15 hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
                            >
                                <CheckCircle2 className="h-4 w-4" />
                                {selectedTopicIds.size === topicCandidates.length && topicCandidates.length > 0 ? 'Clear selection' : 'Select all'}
                            </button>

                            <p className="text-sm text-slate-400">
                                {selectedTopicIds.size} selected of {topicCandidates.length}
                            </p>
                        </div>

                        {topicsLoading ? (
                            <div className="space-y-3">
                                {[1, 2, 3].map((item) => (
                                    <div key={item} className="h-20 animate-pulse rounded-3xl bg-white/[0.035]" />
                                ))}
                            </div>
                        ) : topicCandidates.length === 0 ? (
                            <div className="rounded-[26px] border border-dashed border-white/10 bg-[#0c1525] px-6 py-14 text-center">
                                <p className="text-lg font-medium text-white">No topics in this workspace yet.</p>
                                <p className="mx-auto mt-3 max-w-xl text-sm leading-7 text-slate-400">
                                    Start with the seeded topics for another category, or generate fresh options from AI and news for this selection.
                                </p>
                            </div>
                        ) : (
                            <div className="space-y-3">
                                {topicCandidates.map((topic) => {
                                    const checked = selectedTopicIds.has(topic.id)
                                    return (
                                        <label
                                            key={topic.id}
                                            className={`flex cursor-pointer items-center gap-4 rounded-[26px] border px-4 py-4 transition sm:px-5 ${
                                                checked
                                                    ? 'border-blue-400/30 bg-blue-500/[0.09]'
                                                    : 'border-white/8 bg-[#0c1525] hover:border-white/14 hover:bg-white/[0.045]'
                                            }`}
                                        >
                                            <input
                                                type="checkbox"
                                                checked={checked}
                                                onChange={() => toggleTopic(topic.id)}
                                                className="h-5 w-5 rounded border-white/15 bg-transparent text-blue-400 focus:ring-blue-400"
                                            />

                                            <div className="min-w-0 flex-1">
                                                <p className="truncate text-sm font-medium text-white sm:text-base">{topic.title}</p>
                                                <div className="mt-2 flex flex-wrap gap-2">
                                                    <span className="rounded-full border border-white/8 bg-white/[0.04] px-3 py-1 text-xs text-slate-400">
                                                        {topic.source_label || commandCenterService.getSourceLabel(topic.topic_source)}
                                                    </span>
                                                </div>
                                            </div>

                                            <button
                                                type="button"
                                                onClick={(event) => {
                                                    event.preventDefault()
                                                    void handleRemoveTopic(topic.id)
                                                }}
                                                className="inline-flex h-10 w-10 items-center justify-center rounded-2xl border border-white/8 text-slate-500 transition hover:border-red-400/20 hover:bg-red-500/10 hover:text-red-200"
                                                aria-label={`Remove ${topic.title}`}
                                            >
                                                <Trash2 className="h-4 w-4" />
                                            </button>
                                        </label>
                                    )
                                })}
                            </div>
                        )}

                        <div className="mt-6 border-t border-white/8 pt-6">
                            <button
                                type="button"
                                onClick={handleStartResearch}
                                disabled={selectedTopicIds.size === 0 || startLoading || !activeProject}
                                className="inline-flex h-14 w-full items-center justify-center rounded-[22px] bg-[linear-gradient(135deg,#4ea7ff_0%,#3478f6_45%,#2558d7_100%)] px-6 text-base font-medium text-white shadow-[0_18px_40px_rgba(37,88,215,0.4)] transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50"
                            >
                                {startLoading ? <Loader2 className="h-5 w-5 animate-spin" /> : `Start Research on ${selectedTopicIds.size || ''} Selected Topic${selectedTopicIds.size === 1 ? '' : 's'}`}
                            </button>
                        </div>
                    </div>
                </motion.section>
            </div>
        </div>
    )
}
