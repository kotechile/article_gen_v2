import * as React from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'

import { useProject } from '@/context/project-context'
import {
    buildRelevantPages,
    buildWarehouseKeywordCandidates,
    clusterWarehouseKeywords,
    mapDomainCategories,
    mergeOverviewMetrics,
    normalizeDomainInput,
    recommendWarehouseScope,
    scoreDomainFit,
    splitTopicInput,
    type CategoryIndexRow,
    type DomainCategoryRow,
    type WarehouseCluster,
    type WarehouseRelevantPage,
    type WarehouseScope,
} from '@/lib/domainKeywordWarehouse'
import { supabase } from '@/lib/supabase'
import { researchRebuildService } from '@/services/research-rebuild.service'
import { researchTopicsService } from '@/services/research-topics.service'
import { topicKeywordResearchService } from '@/services/topic-keyword-research.service'
import type { TopicKeywordCandidate, TopicKeywordCluster, TopicKeywordResearchRun, ResearchTopic } from '@/types/research'

type ProjectCategory = {
    id: string
    name: string
    description?: string | null
    level: number
    parent_category_id?: string | null
}

type CountryOption = { label: string; locationCode: number }
type LanguageOption = { label: string; code: string }
type DeviceOption = 'Desktop' | 'Mobile'
type ResearchScope = 'focused' | 'expanded'

type ProbePreview = {
    label: 'Practical' | 'ROI' | 'Question'
    query: string
}

type WorkflowMode = 'article_serp' | 'domain_warehouse'

const COUNTRY_OPTIONS: CountryOption[] = [
    { label: 'United States', locationCode: 2840 },
    { label: 'United Kingdom', locationCode: 2826 },
    { label: 'Canada', locationCode: 2124 },
    { label: 'Australia', locationCode: 2036 },
]

const LANGUAGE_OPTIONS: LanguageOption[] = [
    { label: 'English', code: 'en' },
    { label: 'Spanish', code: 'es' },
    { label: 'French', code: 'fr' },
    { label: 'German', code: 'de' },
]

function safeNumber(value: unknown, fallback = 0) {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : fallback
}

function normalizeText(value: string) {
    return value.trim().replace(/\s+/g, ' ')
}

function slugTokens(value: string) {
    return normalizeText(value)
        .toLowerCase()
        .replace(/[^a-z0-9\s]/g, ' ')
        .split(' ')
        .filter(Boolean)
}

function dedupePreviewQueries(queries: string[]) {
    const seen = new Set<string>()
    return queries.filter((query) => {
        const normalized = normalizeText(query).toLowerCase()
        if (!normalized || seen.has(normalized)) return false
        seen.add(normalized)
        return true
    })
}

function buildProbePreview(topicIdea: string): ProbePreview[] {
    const normalized = normalizeText(topicIdea)
    const tokens = slugTokens(topicIdea)
    const base = tokens.slice(0, 10).join(' ')
    const shortBase = base || normalized.toLowerCase()
    const practical = shortBase || 'topic opportunity'

    const isValueTopic = /(value|roi|worth|investment|resale|return|appreciation)/.test(shortBase)
    const isImprovementTopic = /(increase|improve|upgrades|improvements|renovation|renovations|energy|efficient|eco|green|solar|insulation)/.test(shortBase)
    const isBuyingTopic = /\bbuy\b|\bbuyer\b|\binvest\b|\binvesting\b|\breal estate\b|\bproperty\b|\bhousing\b/.test(shortBase)

    let roiProbe = shortBase
    if (!isValueTopic) {
        if (/\bbuy\b/.test(shortBase)) {
            roiProbe = normalizeText(shortBase.replace(/\bbuy\b/, 'invest in'))
        } else if (/\bbest\b/.test(shortBase) || isBuyingTopic) {
            roiProbe = `${shortBase} for investment`
        } else if (/\bcompare\b|\bcomparison\b|\bvs\b|\bversus\b/.test(shortBase)) {
            roiProbe = `${shortBase} comparison`
        } else if (isImprovementTopic) {
            roiProbe = `${shortBase} roi`
        } else {
            roiProbe = `${shortBase} worth it`
        }
    }

    let questionProbe = `what are the best ${shortBase}`
    if (isImprovementTopic || isValueTopic) {
        questionProbe = `are ${shortBase} worth it`
    } else if (isBuyingTopic) {
        questionProbe = /\bbest\b/.test(shortBase)
            ? `what are ${shortBase}`
            : `what are the best ${shortBase}`
    } else if (shortBase) {
        questionProbe = `how to choose ${shortBase}`
    }

    const [finalPractical, finalRoi, finalQuestion] = dedupePreviewQueries([
        practical,
        roiProbe,
        questionProbe,
        `${shortBase} guide`,
        `${shortBase} comparison`,
    ])

    return [
        { label: 'Practical', query: normalizeText(finalPractical || practical) },
        { label: 'ROI', query: normalizeText(finalRoi || roiProbe || practical) },
        { label: 'Question', query: normalizeText(finalQuestion || questionProbe || practical) },
    ]
}

function strengthLabel(domain: string) {
    const normalized = String(domain || '').toLowerCase()
    if (!normalized) return 'Unknown'
    if (normalized.includes('.gov') || normalized.includes('wikipedia') || normalized.includes('forbes')) return 'Very High'
    if (normalized.includes('bobvila') || normalized.includes('homedepot') || normalized.includes('betterhomes')) return 'Strong'
    if (normalized.includes('reddit') || normalized.includes('medium')) return 'Medium'
    return 'Low/Med'
}

function pageTypeLabel(url: string, domain: string) {
    const target = `${url} ${domain}`.toLowerCase()
    if (domain.includes('.gov')) return 'Gov'
    if (target.includes('/blog/') || target.includes('/article') || target.includes('/guide')) return 'Article'
    if (target.includes('/news/')) return 'Article'
    if (target.includes('/resources/')) return 'Article'
    return 'Blog'
}

function useLabel(entry: { domain?: string; query_hits?: number; content_gap_score?: number }) {
    const domain = String(entry.domain || '').toLowerCase()
    const hits = safeNumber(entry.query_hits)
    const gaps = safeNumber(entry.content_gap_score)
    if (domain.includes('.gov') || domain.includes('forbes') || domain.includes('wikipedia') || domain.includes('reddit')) {
        return { label: 'No', tone: 'text-rose-300 border-rose-500/20 bg-rose-500/10' }
    }
    if (hits >= 2 && gaps >= 1) {
        return { label: 'Yes', tone: 'text-emerald-300 border-emerald-500/20 bg-emerald-500/10' }
    }
    if (hits >= 2) {
        return { label: 'Yes', tone: 'text-emerald-300 border-emerald-500/20 bg-emerald-500/10' }
    }
    return { label: 'Maybe', tone: 'text-amber-300 border-amber-500/20 bg-amber-500/10' }
}

function topicStatusLabel(score: number, passed: boolean) {
    if (!passed || score < 45) return 'Reject Topic'
    if (score >= 80) return 'Strong Opportunity'
    if (score >= 65) return 'Worth Exploring'
    if (score >= 50) return 'Weak Opportunity'
    return 'Reject Topic'
}

function clusterRecommendation(score: number) {
    if (score >= 80) return 'Create article now'
    if (score >= 65) return 'Good candidate'
    if (score >= 50) return 'Save for later'
    return 'Reject'
}

function keywordIntentLabel(intent?: string | null) {
    const value = String(intent || '').toLowerCase()
    if (value.includes('commercial')) return 'Invest'
    if (value.includes('utility')) return 'Utility'
    if (value.includes('informational')) return 'Info'
    return value ? value : 'Info'
}

function categoryPathLabel(primary: ProjectCategory | null, secondary: ProjectCategory | null) {
    return [primary?.name, secondary?.name].filter(Boolean).join(' / ')
}

function parseCategoryIndexRows(search: { response_payload?: Record<string, any> | null; result_summary_json?: Record<string, any> | null } | null | undefined): CategoryIndexRow[] {
    const responseRows = (((search?.response_payload || {}) as Record<string, any>).response?.tasks?.[0]?.result || []) as CategoryIndexRow[]
    if (Array.isArray(responseRows) && responseRows.length) return responseRows
    const summaryRows = (((search?.result_summary_json || {}) as Record<string, any>).top_items || []) as CategoryIndexRow[]
    return Array.isArray(summaryRows) ? summaryRows : []
}

function parseSummaryRows<T extends Record<string, any>>(search: { result_summary_json?: Record<string, any> | null } | null | undefined): T[] {
    const rows = (((search?.result_summary_json || {}) as Record<string, any>).top_items || []) as T[]
    return Array.isArray(rows) ? rows : []
}

export function ResearchRebuildStrategicPage() {
    const navigate = useNavigate()
    const { activeProject, projects, setActiveProject } = useProject()
    const [searchParams, setSearchParams] = useSearchParams()
    const [workflowMode, setWorkflowMode] = React.useState<WorkflowMode>(
        searchParams.get('workflow') === 'domain_warehouse' ? 'domain_warehouse' : 'article_serp',
    )

    const [categories, setCategories] = React.useState<ProjectCategory[]>([])
    const [recentTopics, setRecentTopics] = React.useState<ResearchTopic[]>([])
    const [currentTopic, setCurrentTopic] = React.useState<ResearchTopic | null>(null)
    const [currentRun, setCurrentRun] = React.useState<TopicKeywordResearchRun | null>(null)
    const [allKeywordCandidates, setAllKeywordCandidates] = React.useState<TopicKeywordCandidate[]>([])
    const [clusters, setClusters] = React.useState<TopicKeywordCluster[]>([])

    const [topicIdea, setTopicIdea] = React.useState('')
    const [primaryCategoryId, setPrimaryCategoryId] = React.useState(searchParams.get('primary_category_id') || '')
    const [secondaryCategoryId, setSecondaryCategoryId] = React.useState(searchParams.get('secondary_category_id') || '')
    const [countryLabel, setCountryLabel] = React.useState<CountryOption['label']>('United States')
    const [languageLabel, setLanguageLabel] = React.useState<LanguageOption['label']>('English')
    const [searchEngine, setSearchEngine] = React.useState('Google')
    const [device, setDevice] = React.useState<DeviceOption>('Desktop')
    const [researchScope, setResearchScope] = React.useState<ResearchScope>('focused')
    const [competitorDomain, setCompetitorDomain] = React.useState(searchParams.get('competitor_domain') || '')
    const [allowedTopicsInput, setAllowedTopicsInput] = React.useState(
        'green homes, energy efficiency, resale value, renovation ROI, smart home, solar, insulation, heat pumps',
    )
    const [excludedTopicsInput, setExcludedTopicsInput] = React.useState(
        'jobs, coupons, tools, unrelated product reviews, brand terms, local contractors',
    )
    const [warehouseCategories, setWarehouseCategories] = React.useState<DomainCategoryRow[]>([])
    const [warehousePages, setWarehousePages] = React.useState<WarehouseRelevantPage[]>([])
    const [warehouseClusters, setWarehouseClusters] = React.useState<WarehouseCluster[]>([])
    const [warehouseScope, setWarehouseScope] = React.useState<WarehouseScope>('reject')
    const [warehouseFit, setWarehouseFit] = React.useState<{
        fitScore: number
        categoryMatch: number
        contentTypeMatch: number
        repeatedCompetitorSignal: number
        lowNoise: number
        passed: boolean
    } | null>(null)
    const [warehouseSummary, setWarehouseSummary] = React.useState<{
        rawKeywordCount: number
        shortlistedKeywordCount: number
        usefulKeywordRate: number
        apiCalls: number
        savedTopicId?: string | null
        selectedTargets: string[]
    } | null>(null)
    const [includeKeepOnly, setIncludeKeepOnly] = React.useState(true)
    const [hideBrands, setHideBrands] = React.useState(true)
    const [onlyLowKd, setOnlyLowKd] = React.useState(false)
    const [onlyHighCpc, setOnlyHighCpc] = React.useState(false)
    const [onlyArticleIntent, setOnlyArticleIntent] = React.useState(false)
    const [seenOnMultiplePages, setSeenOnMultiplePages] = React.useState(false)
    const [selectedClusterId, setSelectedClusterId] = React.useState('')
    const [isLoading, setIsLoading] = React.useState(false)
    const [error, setError] = React.useState<string | null>(null)
    const [success, setSuccess] = React.useState<string | null>(null)

    const projectId = searchParams.get('project_id') || activeProject?.id || ''

    React.useEffect(() => {
        const incomingProjectId = searchParams.get('project_id')
        if (!incomingProjectId || !projects.length || activeProject?.id === incomingProjectId) return
        const match = projects.find((project) => project.id === incomingProjectId) || null
        if (match) setActiveProject(match)
    }, [activeProject?.id, projects, searchParams, setActiveProject])

    React.useEffect(() => {
        if (!projectId || searchParams.get('project_id') === projectId) return
        const next = new URLSearchParams(searchParams)
        next.set('project_id', projectId)
        setSearchParams(next, { replace: true })
    }, [projectId, searchParams, setSearchParams])

    React.useEffect(() => {
        const next = new URLSearchParams(searchParams)
        if (workflowMode === 'domain_warehouse') next.set('workflow', 'domain_warehouse')
        else next.delete('workflow')
        if (competitorDomain.trim()) next.set('competitor_domain', normalizeDomainInput(competitorDomain))
        else next.delete('competitor_domain')
        if (next.toString() !== searchParams.toString()) {
            setSearchParams(next, { replace: true })
        }
    }, [competitorDomain, searchParams, setSearchParams, workflowMode])

    React.useEffect(() => {
        if (!projectId) return
        let cancelled = false
        const loadCategories = async () => {
            const { data, error: categoryError } = await supabase
                .from('project_categories')
                .select('id,name,description,level,parent_category_id')
                .eq('project_id', projectId)
                .order('level', { ascending: true })
                .order('sort_order', { ascending: true })
                .order('name', { ascending: true })
            if (cancelled) return
            if (categoryError) {
                setCategories([])
                setError(categoryError.message)
                return
            }
            setCategories((data as ProjectCategory[]) || [])
        }
        void loadCategories()
        return () => {
            cancelled = true
        }
    }, [projectId])

    const loadRecentTopics = React.useCallback(async () => {
        if (!projectId) return
        try {
            const response = await researchTopicsService.listResearchTopics({
                order_by: 'created_at',
                order_direction: 'desc',
                page: 1,
                size: 12,
                project_id: projectId,
                primary_category_id: primaryCategoryId || undefined,
                secondary_category_id: secondaryCategoryId || undefined,
            })
            setRecentTopics(response.items || [])
        } catch (loadError) {
            console.error('Failed to load recent research topics', loadError)
        }
    }, [primaryCategoryId, projectId, secondaryCategoryId])

    React.useEffect(() => {
        void loadRecentTopics()
    }, [loadRecentTopics])

    const primaryCategories = React.useMemo(
        () => categories.filter((category) => Number(category.level) === 1),
        [categories],
    )
    const secondaryCategories = React.useMemo(
        () => categories.filter((category) => Number(category.level) === 2 && String(category.parent_category_id || '') === primaryCategoryId),
        [categories, primaryCategoryId],
    )
    const primaryCategory = primaryCategories.find((category) => category.id === primaryCategoryId) || null
    const secondaryCategory = secondaryCategories.find((category) => category.id === secondaryCategoryId) || null
    const selectedCountry = COUNTRY_OPTIONS.find((country) => country.label === countryLabel) || COUNTRY_OPTIONS[0]
    const selectedLanguage = LANGUAGE_OPTIONS.find((language) => language.label === languageLabel) || LANGUAGE_OPTIONS[0]
    const probePreview = React.useMemo(() => buildProbePreview(topicIdea), [topicIdea])

    const serpGate = React.useMemo(() => {
        return ((currentRun?.raw_data_json || {}) as Record<string, any>).serp_gate || {}
    }, [currentRun])
    const competitorPages = React.useMemo(() => {
        return ((((currentRun?.raw_data_json || {}) as Record<string, any>).competitor_pages) || []) as Array<Record<string, any>>
    }, [currentRun])
    const controlledExpansion = React.useMemo(() => {
        return ((((currentRun?.raw_data_json || {}) as Record<string, any>).controlled_expansion) || []) as Array<Record<string, any>>
    }, [currentRun])
    const runSummary = React.useMemo(() => {
        return ((currentRun?.summary_json || {}) as Record<string, any>) || {}
    }, [currentRun])

    const opportunityScore = React.useMemo(() => {
        const serpWeakness = safeNumber(runSummary.serp_weakness_score) * 100
        const repeatedDomains = Array.isArray(runSummary.top_repeated_domains) ? runSummary.top_repeated_domains.length : 0
        const signalCount = safeNumber(runSummary.serp_signal_count)
        const clusterBoost = clusters.length ? Math.min(20, safeNumber(clusters[0]?.opportunity_score)) : 0
        return Math.max(0, Math.min(100, Math.round(serpWeakness * 0.55 + repeatedDomains * 6 + signalCount * 6 + clusterBoost * 0.15)))
    }, [clusters, runSummary])

    const scoreStatus = topicStatusLabel(opportunityScore, Boolean(runSummary.serp_opportunity_passed))

    const opportunitySignals = React.useMemo(() => {
        const signals = serpGate.signals || {}
        return [
            { label: 'Small sites ranking', passed: Boolean(signals.niche_sites_present) },
            { label: 'Usable content pages visible', passed: Boolean(signals.article_friendly_results) },
            { label: 'Repeated domains found', passed: Boolean(signals.stable_competitor_set) },
            { label: 'Search intent is consistent', passed: Boolean(signals.consistent_intent) },
            { label: 'Some authority sites present', passed: !Boolean(signals.not_authority_dominated) },
            { label: 'SERP has weak or outdated pages', passed: Boolean(signals.weak_pages_present) },
        ]
    }, [serpGate])

    const filteredKeywords = React.useMemo(() => {
        return allKeywordCandidates.filter((row) => {
            const trend = (row.trend_json || {}) as Record<string, any>
            const sourceUrlCount = Array.isArray(trend.source_urls) ? trend.source_urls.length : trend.source_url ? 1 : 0
            const keepStatus = !row.is_filtered_out && safeNumber(row.opportunity_score) >= 50
            if (includeKeepOnly && !keepStatus) return false
            if (hideBrands && String(row.filter_reason || '').includes('brand')) return false
            if (onlyLowKd && safeNumber(row.keyword_difficulty, 999) > 35) return false
            if (onlyHighCpc && safeNumber(row.cpc) < 1) return false
            if (onlyArticleIntent && !['informational', 'commercial_investigation'].includes(String(row.intent_label || ''))) return false
            if (seenOnMultiplePages && sourceUrlCount < 2) return false
            return true
        })
    }, [allKeywordCandidates, hideBrands, includeKeepOnly, onlyArticleIntent, onlyHighCpc, onlyLowKd, seenOnMultiplePages])

    const selectedCluster = React.useMemo(() => {
        return clusters.find((cluster) => cluster.id === selectedClusterId) || clusters[0] || null
    }, [clusters, selectedClusterId])

    React.useEffect(() => {
        if (!selectedClusterId && clusters[0]?.id) {
            setSelectedClusterId(clusters[0].id)
        }
    }, [clusters, selectedClusterId])

    const finalArticle = React.useMemo(() => {
        if (!selectedCluster) return null
        const score = safeNumber(selectedCluster.opportunity_score)
        const secondaryKeywords = (selectedCluster.secondary_keywords_json || []).slice(0, 6)
        const primaryKeyword = String(selectedCluster.primary_keyword || selectedCluster.cluster_name || '')
        const title = selectedCluster.article_angle
            ? selectedCluster.article_angle.replace(/^What\s+/i, '').replace(/^Best ways to evaluate\s+/i, '').replace(/^Practical tools, workflows, and calculators built around\s+/i, '').trim()
            : primaryKeyword
        return {
            title: title
                ? title.charAt(0).toUpperCase() + title.slice(1)
                : primaryKeyword,
            primaryKeyword,
            secondaryKeywords,
            score,
            recommendation: clusterRecommendation(score),
        }
    }, [selectedCluster])

    const decisionSidebarScores = React.useMemo(() => {
        const topClusterScore = safeNumber(selectedCluster?.opportunity_score)
        const topKeyword = filteredKeywords[0]
        const kd = topKeyword ? Math.max(0, 100 - safeNumber(topKeyword.keyword_difficulty, 100)) : 0
        const volume = topKeyword ? Math.min(100, Math.log10(safeNumber(topKeyword.search_volume) + 1) / Math.log10(10001) * 100) : 0
        const commercial = topKeyword ? Math.min(100, safeNumber(topKeyword.cpc) * 20) : 0
        const topicalFit = topKeyword ? safeNumber(topKeyword.topical_fit_score) : 0
        const serp = safeNumber(runSummary.serp_weakness_score) * 100
        return {
            serp: Math.round(serp),
            kd: Math.round(kd),
            volume: Math.round(volume),
            commercial: Math.round(commercial),
            topicalFit: Math.round(topicalFit),
            final: Math.round(topClusterScore || opportunityScore),
        }
    }, [filteredKeywords, opportunityScore, runSummary.serp_weakness_score, selectedCluster?.opportunity_score])

    const loadTopicRun = React.useCallback(async (topic: ResearchTopic) => {
        try {
            setIsLoading(true)
            setError(null)
            setCurrentTopic(topic)
            setTopicIdea(topic.title || '')
            const run = await topicKeywordResearchService.getLatestRun(topic.id)
            setCurrentRun(run)
            const [, allKeywords, clusterResponse] = await Promise.all([
                topicKeywordResearchService.listKeywords(topic.id, run.id, false),
                topicKeywordResearchService.listKeywords(topic.id, run.id, true),
                topicKeywordResearchService.listClusters(topic.id, run.id),
            ])
            setAllKeywordCandidates(allKeywords.items || [])
            setClusters(clusterResponse.items || [])
            setSuccess('Loaded the latest SERP opportunity analysis for this topic.')
        } catch (loadError) {
            console.error('Failed to load topic run', loadError)
            setError(loadError instanceof Error ? loadError.message : 'Failed to load the latest analysis.')
        } finally {
            setIsLoading(false)
        }
    }, [])

    const handleAnalyze = async () => {
        if (!projectId || !topicIdea.trim() || !primaryCategoryId) return
        setIsLoading(true)
        setError(null)
        setSuccess(null)
        try {
            const topic = await researchTopicsService.createResearchTopic({
                title: normalizeText(topicIdea),
                description: normalizeText(topicIdea),
                project_id: projectId,
                primary_category_id: primaryCategoryId,
                secondary_category_id: secondaryCategoryId || null,
                topic_mode: 'keyword_first',
                keyword_viability_label: 'medium',
                topic_generation_metadata: {
                    target_country: countryLabel,
                    target_language: languageLabel,
                    search_engine: searchEngine,
                    device,
                    research_scope: researchScope,
                },
            })
            setCurrentTopic(topic)
            const runResult = await topicKeywordResearchService.runTopicKeywordResearch(topic.id, {
                replace_existing: true,
                filters: {
                    target_location_code: selectedCountry.locationCode,
                    target_language_code: selectedLanguage.code,
                    target_device: device.toLowerCase(),
                    research_scope: researchScope,
                },
            })
            const run = runResult.run
            setCurrentRun(run)
            const [, allKeywords, clusterResponse] = await Promise.all([
                topicKeywordResearchService.listKeywords(topic.id, run.id, false),
                topicKeywordResearchService.listKeywords(topic.id, run.id, true),
                topicKeywordResearchService.listClusters(topic.id, run.id),
            ])
            setAllKeywordCandidates(allKeywords.items || [])
            setClusters(clusterResponse.items || [])
            setSearchParams((currentParams) => {
                const next = new URLSearchParams(currentParams)
                next.set('project_id', projectId)
                next.set('primary_category_id', primaryCategoryId)
                if (secondaryCategoryId) next.set('secondary_category_id', secondaryCategoryId)
                else next.delete('secondary_category_id')
                next.set('topic_id', topic.id)
                return next
            }, { replace: true })
            void loadRecentTopics()
            setSuccess('SERP opportunity analyzed. Review the competitor evidence before deciding whether to create the article.')
        } catch (analyzeError) {
            console.error('Failed to analyze SERP opportunity', analyzeError)
            setError(analyzeError instanceof Error ? analyzeError.message : 'Failed to analyze SERP opportunity.')
        } finally {
            setIsLoading(false)
        }
    }

    const openTopicDetail = () => {
        if (!currentTopic?.id) return
        navigate(`/research/${currentTopic.id}`)
    }

    const handleRunWarehouse = async () => {
        if (!projectId || !primaryCategoryId || !competitorDomain.trim()) return
        const normalizedDomain = normalizeDomainInput(competitorDomain)
        const allowedTopics = splitTopicInput(allowedTopicsInput)
        const excludedTopics = splitTopicInput(excludedTopicsInput)
        const siteCategory = categoryPathLabel(primaryCategory, secondaryCategory) || primaryCategory?.name || ''

        setIsLoading(true)
        setError(null)
        setSuccess(null)
        setWarehouseCategories([])
        setWarehousePages([])
        setWarehouseClusters([])
        setWarehouseSummary(null)

        try {
            const [categoryIndexSearch, domainCategorySearch, relevantPagesSearch] = await Promise.all([
                researchRebuildService.runDataforseoSearch({
                    project_id: projectId,
                    primary_category_id: primaryCategoryId,
                    secondary_category_id: secondaryCategoryId || undefined,
                    search_type: 'category_index',
                    force_refresh: false,
                }),
                researchRebuildService.runDataforseoSearch({
                    project_id: projectId,
                    primary_category_id: primaryCategoryId,
                    secondary_category_id: secondaryCategoryId || undefined,
                    search_type: 'categories_for_domain',
                    target: normalizedDomain,
                    language_code: selectedLanguage.code,
                    location_code: selectedCountry.locationCode,
                    limit: 12,
                    extra: {
                        include_subcategories: false,
                        item_types: ['organic'],
                        order_by: ['metrics.organic.count,desc'],
                    },
                }),
                researchRebuildService.runDataforseoSearch({
                    project_id: projectId,
                    primary_category_id: primaryCategoryId,
                    secondary_category_id: secondaryCategoryId || undefined,
                    search_type: 'relevant_pages',
                    target: normalizedDomain,
                    language_code: selectedLanguage.code,
                    location_code: selectedCountry.locationCode,
                    limit: 20,
                    extra: {
                        item_types: ['organic'],
                        historical_serp_mode: 'live',
                        ignore_synonyms: true,
                        filters: [['metrics.organic.count', '>=', 5]],
                        order_by: ['metrics.organic.etv,desc'],
                    },
                }),
            ])

            const categoryIndexRows = parseCategoryIndexRows(categoryIndexSearch)
            const domainCategoryRows = mapDomainCategories(parseSummaryRows(domainCategorySearch), categoryIndexRows)
            const relevantPages = buildRelevantPages(parseSummaryRows(relevantPagesSearch), [siteCategory, ...allowedTopics], excludedTopics)
            const fit = scoreDomainFit({
                domainCategories: domainCategoryRows,
                relevantPages,
                siteCategory,
                allowedTopics,
                excludedTopics,
            })
            const scopeRecommendation = recommendWarehouseScope(fit.fitScore, relevantPages)

            setWarehouseCategories(domainCategoryRows)
            setWarehousePages(relevantPages)
            setWarehouseFit(fit)
            setWarehouseScope(scopeRecommendation)

            if (!fit.passed) {
                setWarehouseSummary({
                    rawKeywordCount: 0,
                    shortlistedKeywordCount: 0,
                    usefulKeywordRate: 0,
                    apiCalls: 2,
                    selectedTargets: [],
                })
                setSuccess(`Domain fit score ${fit.fitScore}/100. This domain was rejected before keyword harvesting to save credits.`)
                return
            }

            const includedPages = relevantPages.filter((page) => page.include)
            const selectedTargets =
                scopeRecommendation === 'whole_domain'
                    ? [normalizedDomain]
                    : includedPages.slice(0, scopeRecommendation === 'exact_pages' ? 3 : 4).map((page) => page.url)

            const rankedRows: Array<Record<string, unknown>> = []
            let apiCalls = 2

            if (scopeRecommendation === 'whole_domain') {
                const offsets = [0, 100, 200]
                for (const offset of offsets) {
                    const rankedSearch = await researchRebuildService.runDataforseoSearch({
                        project_id: projectId,
                        primary_category_id: primaryCategoryId,
                        secondary_category_id: secondaryCategoryId || undefined,
                        search_type: 'ranked_keywords',
                        target: normalizedDomain,
                        language_code: selectedLanguage.code,
                        location_code: selectedCountry.locationCode,
                        limit: 100,
                        extra: {
                            offset,
                            item_types: ['organic'],
                            historical_serp_mode: 'live',
                            ignore_synonyms: true,
                            filters: [
                                ['ranked_serp_element.serp_item.rank_group', '<=', 30],
                                'and',
                                ['keyword_data.keyword_info.search_volume', '>=', 30],
                            ],
                            order_by: [
                                'ranked_serp_element.serp_item.rank_group,asc',
                                'keyword_data.keyword_info.search_volume,desc',
                            ],
                        },
                    })
                    apiCalls += 1
                    const batchRows = parseSummaryRows<Record<string, unknown>>(rankedSearch)
                    rankedRows.push(...batchRows)
                    const batchCandidates = buildWarehouseKeywordCandidates({
                        rows: batchRows,
                        siteCategory,
                        allowedTopics,
                        excludedTopics,
                        sourceDomain: normalizedDomain,
                    })
                    const usefulRate = batchCandidates.length
                        ? (batchCandidates.filter((item) => !item.rejected).length / batchCandidates.length) * 100
                        : 0
                    if ((offset > 0 && usefulRate < 20) || rankedRows.length >= 500) break
                }
            } else {
                for (const target of selectedTargets) {
                    const rankedSearch = await researchRebuildService.runDataforseoSearch({
                        project_id: projectId,
                        primary_category_id: primaryCategoryId,
                        secondary_category_id: secondaryCategoryId || undefined,
                        search_type: 'ranked_keywords',
                        target,
                        language_code: selectedLanguage.code,
                        location_code: selectedCountry.locationCode,
                        limit: 100,
                        extra: {
                            offset: 0,
                            item_types: ['organic'],
                            historical_serp_mode: 'live',
                            ignore_synonyms: true,
                            filters: [
                                ['ranked_serp_element.serp_item.rank_group', '<=', 30],
                                'and',
                                ['keyword_data.keyword_info.search_volume', '>=', 30],
                            ],
                            order_by: [
                                'ranked_serp_element.serp_item.rank_group,asc',
                                'keyword_data.keyword_info.search_volume,desc',
                            ],
                        },
                    })
                    apiCalls += 1
                    rankedRows.push(...parseSummaryRows<Record<string, unknown>>(rankedSearch))
                }
            }

            const dedupedRows = Array.from(
                new Map(
                    rankedRows.map((row) => {
                        const key = `${String(row.keyword || '').toLowerCase()}::${String(row.url || '')}`
                        return [key, row]
                    }),
                ).values(),
            )

            const rawCandidates = buildWarehouseKeywordCandidates({
                rows: dedupedRows,
                siteCategory,
                allowedTopics,
                excludedTopics,
                sourceDomain: normalizedDomain,
            })
            const shortlisted = rawCandidates
                .filter((item) => !item.rejected)
                .sort((a, b) => b.warehouseScore - a.warehouseScore || b.searchVolume - a.searchVolume)

            let enrichedCandidates = shortlisted
            if (shortlisted.length) {
                const overviewSearch = await researchRebuildService.runDataforseoSearch({
                    project_id: projectId,
                    primary_category_id: primaryCategoryId,
                    secondary_category_id: secondaryCategoryId || undefined,
                    search_type: 'keyword_overview',
                    keywords: shortlisted.slice(0, 120).map((item) => item.keyword),
                    language_code: selectedLanguage.code,
                    location_code: selectedCountry.locationCode,
                })
                apiCalls += 1
                enrichedCandidates = mergeOverviewMetrics(shortlisted, parseSummaryRows<Record<string, unknown>>(overviewSearch))
                    .sort((a, b) => b.warehouseScore - a.warehouseScore || b.searchVolume - a.searchVolume)
            }

            const clusters = clusterWarehouseKeywords(enrichedCandidates.slice(0, 150))
            const usefulKeywordRate = rawCandidates.length
                ? Math.round((shortlisted.length / rawCandidates.length) * 100)
                : 0

            const savedTopic = await researchTopicsService.createResearchTopic({
                title: `Warehouse: ${normalizedDomain}`,
                description: `Competitor domain keyword warehouse for ${normalizedDomain}`,
                project_id: projectId,
                primary_category_id: primaryCategoryId,
                secondary_category_id: secondaryCategoryId || null,
                topic_mode: 'keyword_first',
                keyword_viability_label: 'medium',
                topic_source: 'competitor_domain_warehouse',
                topic_generation_metadata: {
                    workflow_type: 'domain_keyword_warehouse',
                    competitor_domain: normalizedDomain,
                    target_country: countryLabel,
                    target_language: languageLabel,
                    site_category: siteCategory,
                    allowed_topics: allowedTopics,
                    excluded_topics: excludedTopics,
                    fit_summary: fit,
                    scope_recommendation: scopeRecommendation,
                    selected_targets: selectedTargets,
                    domain_categories: domainCategoryRows,
                    relevant_pages: relevantPages,
                    keyword_candidates: enrichedCandidates.slice(0, 150),
                    clusters,
                    useful_keyword_rate: usefulKeywordRate,
                },
            })

            setWarehouseClusters(clusters)
            setWarehouseSummary({
                rawKeywordCount: rawCandidates.length,
                shortlistedKeywordCount: enrichedCandidates.length,
                usefulKeywordRate,
                apiCalls,
                savedTopicId: savedTopic.id,
                selectedTargets,
            })
            setSuccess(`Stored ${clusters.length} future article clusters from ${normalizedDomain}. These are warehouse opportunities, not final article approvals.`)
        } catch (warehouseError) {
            console.error('Failed to run domain keyword warehouse', warehouseError)
            setError(warehouseError instanceof Error ? warehouseError.message : 'Failed to run domain keyword warehouse.')
        } finally {
            setIsLoading(false)
        }
    }

    const workflowToggle = (
        <div className="grid gap-3 md:grid-cols-2">
            <button
                type="button"
                onClick={() => setWorkflowMode('article_serp')}
                className={`rounded-[24px] border p-5 text-left transition ${workflowMode === 'article_serp' ? 'border-sky-400 bg-sky-500/10' : 'border-slate-700 bg-[#111a28] hover:border-slate-500'}`}
            >
                <div className="text-xs uppercase tracking-[0.28em] text-sky-300/75">Workflow A</div>
                <div className="mt-2 text-xl font-semibold text-white">Find Article Keyword from SERP</div>
                <p className="mt-2 text-sm text-slate-300">Use the current topic-first flow to test one article opportunity and decide whether it deserves a full SEO brief.</p>
            </button>
            <button
                type="button"
                onClick={() => setWorkflowMode('domain_warehouse')}
                className={`rounded-[24px] border p-5 text-left transition ${workflowMode === 'domain_warehouse' ? 'border-emerald-400 bg-emerald-500/10' : 'border-slate-700 bg-[#111a28] hover:border-slate-500'}`}
            >
                <div className="text-xs uppercase tracking-[0.28em] text-emerald-300/75">Workflow B</div>
                <div className="mt-2 text-xl font-semibold text-white">Harvest Future Keywords from Competitor Domain</div>
                <p className="mt-2 text-sm text-slate-300">Mine a relevant niche competitor, cluster the useful keywords, and save them as future opportunities ready for later SERP validation.</p>
            </button>
        </div>
    )

    if (workflowMode === 'domain_warehouse') {
        const categoryPath = categoryPathLabel(primaryCategory, secondaryCategory)
        return (
            <div className="min-h-screen bg-[#06101a] text-white">
                <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-6 py-8">
                    <header className="rounded-[28px] border border-slate-800 bg-[radial-gradient(circle_at_top_left,rgba(16,185,129,0.16),transparent_35%),linear-gradient(180deg,#0d1624_0%,#0a1019_100%)] p-6 shadow-[0_20px_80px_rgba(2,8,23,0.35)]">
                        <div className="text-xs uppercase tracking-[0.35em] text-emerald-300/80">Research</div>
                        <h1 className="mt-3 text-3xl font-semibold tracking-tight">Domain Keyword Warehouse</h1>
                        <p className="mt-3 max-w-4xl text-sm text-slate-300">
                            Build a future opportunity database from a competitor domain. This flow does not approve articles directly; it stores clusters that should go through SERP validation later.
                        </p>
                    </header>

                    {workflowToggle}

                    {error ? (
                        <div className="rounded-2xl border border-rose-500/30 bg-rose-950/30 px-4 py-3 text-sm text-rose-200">{error}</div>
                    ) : null}

                    {success ? (
                        <div className="rounded-2xl border border-emerald-500/30 bg-emerald-950/20 px-4 py-3 text-sm text-emerald-200">{success}</div>
                    ) : null}

                    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_320px]">
                        <div className="flex flex-col gap-6">
                            <section className="rounded-[28px] border border-slate-800 bg-[#0c1420] p-6">
                                <div className="mb-5">
                                    <div className="text-xs uppercase tracking-[0.3em] text-emerald-300/75">Step 1</div>
                                    <h2 className="mt-2 text-2xl font-semibold">Input Screen</h2>
                                </div>

                                <div className="grid gap-4 md:grid-cols-2">
                                    <label className="flex flex-col gap-2 md:col-span-2">
                                        <span className="text-sm text-slate-300">Competitor Domain</span>
                                        <input
                                            value={competitorDomain}
                                            onChange={(event) => setCompetitorDomain(event.target.value)}
                                            placeholder="attainablehome.com"
                                            className="h-12 rounded-2xl border border-slate-700 bg-[#141d2c] px-4 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-emerald-400"
                                        />
                                    </label>

                                    <label className="flex flex-col gap-2">
                                        <span className="text-sm text-slate-300">Website Category</span>
                                        <select
                                            value={primaryCategoryId}
                                            onChange={(event) => {
                                                setPrimaryCategoryId(event.target.value)
                                                setSecondaryCategoryId('')
                                            }}
                                            className="h-12 rounded-2xl border border-slate-700 bg-[#141d2c] px-4 text-sm text-white outline-none transition focus:border-emerald-400"
                                        >
                                            <option value="">Select category</option>
                                            {primaryCategories.map((category) => (
                                                <option key={category.id} value={category.id}>{category.name}</option>
                                            ))}
                                        </select>
                                    </label>

                                    <label className="flex flex-col gap-2">
                                        <span className="text-sm text-slate-300">Subcategory</span>
                                        <select
                                            value={secondaryCategoryId}
                                            onChange={(event) => setSecondaryCategoryId(event.target.value)}
                                            className="h-12 rounded-2xl border border-slate-700 bg-[#141d2c] px-4 text-sm text-white outline-none transition focus:border-emerald-400"
                                        >
                                            <option value="">Select subcategory</option>
                                            {secondaryCategories.map((category) => (
                                                <option key={category.id} value={category.id}>{category.name}</option>
                                            ))}
                                        </select>
                                    </label>

                                    <label className="flex flex-col gap-2">
                                        <span className="text-sm text-slate-300">Target Country</span>
                                        <select value={countryLabel} onChange={(event) => setCountryLabel(event.target.value)} className="h-12 rounded-2xl border border-slate-700 bg-[#141d2c] px-4 text-sm text-white outline-none transition focus:border-emerald-400">
                                            {COUNTRY_OPTIONS.map((country) => (
                                                <option key={country.label} value={country.label}>{country.label}</option>
                                            ))}
                                        </select>
                                    </label>

                                    <label className="flex flex-col gap-2">
                                        <span className="text-sm text-slate-300">Language</span>
                                        <select value={languageLabel} onChange={(event) => setLanguageLabel(event.target.value)} className="h-12 rounded-2xl border border-slate-700 bg-[#141d2c] px-4 text-sm text-white outline-none transition focus:border-emerald-400">
                                            {LANGUAGE_OPTIONS.map((language) => (
                                                <option key={language.label} value={language.label}>{language.label}</option>
                                            ))}
                                        </select>
                                    </label>

                                    <label className="flex flex-col gap-2 md:col-span-2">
                                        <span className="text-sm text-slate-300">Allowed Topics</span>
                                        <textarea
                                            value={allowedTopicsInput}
                                            onChange={(event) => setAllowedTopicsInput(event.target.value)}
                                            rows={2}
                                            className="rounded-2xl border border-slate-700 bg-[#141d2c] px-4 py-3 text-sm text-white outline-none transition focus:border-emerald-400"
                                        />
                                    </label>

                                    <label className="flex flex-col gap-2 md:col-span-2">
                                        <span className="text-sm text-slate-300">Excluded Topics</span>
                                        <textarea
                                            value={excludedTopicsInput}
                                            onChange={(event) => setExcludedTopicsInput(event.target.value)}
                                            rows={2}
                                            className="rounded-2xl border border-slate-700 bg-[#141d2c] px-4 py-3 text-sm text-white outline-none transition focus:border-emerald-400"
                                        />
                                    </label>
                                </div>

                                <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                                    <button
                                        type="button"
                                        onClick={handleRunWarehouse}
                                        disabled={!projectId || !primaryCategoryId || !competitorDomain.trim() || isLoading}
                                        className="inline-flex items-center justify-center rounded-2xl bg-emerald-400 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:opacity-50"
                                    >
                                        {isLoading ? 'Harvesting…' : 'Harvest Future Keywords'}
                                    </button>
                                    <div className="text-sm text-slate-400">Keyword mode: Future content discovery</div>
                                </div>
                            </section>

                            <section className="rounded-[28px] border border-slate-800 bg-[#0c1420] p-6">
                                <div className="mb-4">
                                    <div className="text-xs uppercase tracking-[0.3em] text-emerald-300/75">Step 2</div>
                                    <h2 className="mt-2 text-2xl font-semibold">Domain Fit Test</h2>
                                </div>

                                <div className="rounded-[24px] border border-slate-700 bg-[#111a28] p-5">
                                    <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                                        <div>
                                            <div className="text-sm text-slate-300">Domain Fit Score</div>
                                            <div className="mt-2 text-4xl font-semibold text-white">{warehouseFit?.fitScore ?? 0} / 100</div>
                                            <div className="mt-2 text-lg text-emerald-200">
                                                {warehouseFit ? (warehouseFit.passed ? 'Harvest domain' : 'Reject before harvesting') : 'Run the fit test'}
                                            </div>
                                        </div>
                                        <div className="grid gap-2 text-sm text-slate-300">
                                            <div>Topical category match: {warehouseFit?.categoryMatch ?? 0}</div>
                                            <div>Content-type match: {warehouseFit?.contentTypeMatch ?? 0}</div>
                                            <div>Repeated competitor signal: {warehouseFit?.repeatedCompetitorSignal ?? 0}</div>
                                            <div>Niche focus / low noise: {warehouseFit?.lowNoise ?? 0}</div>
                                        </div>
                                    </div>
                                    <div className="mt-4 rounded-2xl border border-slate-700 bg-[#0f1928] p-4 text-sm text-slate-300">
                                        Recommended scope: <span className="font-medium text-white">{warehouseScope.replace('_', ' ')}</span>
                                        <div className="mt-2">Site category: {categoryPath || primaryCategory?.name || 'Select a category'}</div>
                                    </div>
                                </div>
                            </section>

                            <section className="rounded-[28px] border border-slate-800 bg-[#0c1420] p-6">
                                <div className="mb-4">
                                    <div className="text-xs uppercase tracking-[0.3em] text-emerald-300/75">Step 3</div>
                                    <h2 className="mt-2 text-2xl font-semibold">Relevant Pages and Categories</h2>
                                </div>

                                <div className="grid gap-6 xl:grid-cols-[320px_minmax(0,1fr)]">
                                    <div className="rounded-2xl border border-slate-700 bg-[#111a28] p-4">
                                        <div className="text-sm font-medium text-white">Top domain categories</div>
                                        <div className="mt-3 grid gap-3">
                                            {warehouseCategories.slice(0, 6).map((row, index) => (
                                                <div key={`${row.category_names.join('-')}-${index}`} className="rounded-2xl border border-slate-700 bg-[#0f1928] p-3">
                                                    <div className="text-sm font-medium text-white">{row.category_names.join(' / ')}</div>
                                                    <div className="mt-2 text-xs text-slate-300">Keywords: {row.organic_count || 0} • Top 30: {(row.pos_4_10 || 0) + (row.pos_11_20 || 0) + (row.pos_21_30 || 0)}</div>
                                                </div>
                                            ))}
                                            {!warehouseCategories.length ? <div className="text-sm text-slate-400">No category fit data yet.</div> : null}
                                        </div>
                                    </div>

                                    <div className="overflow-hidden rounded-2xl border border-slate-700 bg-[#111a28]">
                                        <div className="hidden grid-cols-[minmax(0,2fr)_100px_100px_110px_90px] gap-3 border-b border-slate-700 px-4 py-3 text-[11px] uppercase tracking-[0.26em] text-slate-400 lg:grid">
                                            <div>Page</div>
                                            <div>Traffic</div>
                                            <div>Keywords</div>
                                            <div>Topic Match</div>
                                            <div>Use?</div>
                                        </div>
                                        <div className="divide-y divide-slate-800">
                                            {warehousePages.slice(0, 12).map((page) => (
                                                <div key={page.url} className="grid gap-2 px-4 py-4 lg:grid-cols-[minmax(0,2fr)_100px_100px_110px_90px] lg:items-center">
                                                    <div>
                                                        <div className="font-medium text-white">{page.title || page.url}</div>
                                                        <div className="mt-1 text-xs text-slate-400">{page.url}</div>
                                                    </div>
                                                    <div className="text-sm text-slate-300">{Math.round(page.traffic || 0)}</div>
                                                    <div className="text-sm text-slate-300">{page.organicCount}</div>
                                                    <div className="text-sm text-slate-300">{page.topicMatchLabel}</div>
                                                    <div className="text-sm text-slate-300">{page.include ? 'Yes' : 'No'}</div>
                                                </div>
                                            ))}
                                            {!warehousePages.length ? <div className="px-4 py-6 text-sm text-slate-400">Relevant pages will appear after the domain fit step.</div> : null}
                                        </div>
                                    </div>
                                </div>
                            </section>

                            <section className="rounded-[28px] border border-slate-800 bg-[#0c1420] p-6">
                                <div className="mb-4">
                                    <div className="text-xs uppercase tracking-[0.3em] text-emerald-300/75">Step 4</div>
                                    <h2 className="mt-2 text-2xl font-semibold">Stored Warehouse Opportunities</h2>
                                </div>

                                <div className="grid gap-4 md:grid-cols-3">
                                    <div className="rounded-2xl border border-slate-700 bg-[#111a28] p-4">
                                        <div className="text-sm text-slate-300">Future article clusters</div>
                                        <div className="mt-2 text-3xl font-semibold text-white">{warehouseClusters.length}</div>
                                    </div>
                                    <div className="rounded-2xl border border-slate-700 bg-[#111a28] p-4">
                                        <div className="text-sm text-slate-300">Useful keyword rate</div>
                                        <div className="mt-2 text-3xl font-semibold text-white">{warehouseSummary?.usefulKeywordRate ?? 0}%</div>
                                    </div>
                                    <div className="rounded-2xl border border-slate-700 bg-[#111a28] p-4">
                                        <div className="text-sm text-slate-300">Estimated API calls used</div>
                                        <div className="mt-2 text-3xl font-semibold text-white">{warehouseSummary?.apiCalls ?? 0}</div>
                                    </div>
                                </div>

                                <div className="mt-5 grid gap-4">
                                    {warehouseClusters.slice(0, 12).map((cluster) => (
                                        <div key={cluster.id} className="rounded-3xl border border-slate-700 bg-[#111a28] p-5">
                                            <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                                                <div>
                                                    <div className="text-xs uppercase tracking-[0.24em] text-slate-400">{cluster.priority} Priority</div>
                                                    <h3 className="mt-2 text-xl font-semibold text-white">{cluster.clusterName}</h3>
                                                    <div className="mt-2 text-sm text-slate-300">
                                                        Primary keyword: <span className="font-medium text-white">{cluster.primaryKeyword}</span>
                                                    </div>
                                                </div>
                                                <div className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-3 py-1 text-xs text-emerald-200">
                                                    {cluster.status}
                                                </div>
                                            </div>
                                            <div className="mt-4 flex flex-wrap gap-2">
                                                {cluster.supportingKeywords.slice(0, 6).map((keyword) => (
                                                    <span key={`${cluster.id}-${keyword}`} className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-200">{keyword}</span>
                                                ))}
                                            </div>
                                            <div className="mt-4 grid gap-2 text-sm text-slate-300 md:grid-cols-4">
                                                <div>Keywords: <span className="font-medium text-white">{1 + cluster.supportingKeywords.length}</span></div>
                                                <div>Avg KD: <span className="font-medium text-white">{cluster.avgKd ?? 'n/a'}</span></div>
                                                <div>Volume Potential: <span className="font-medium text-white">{cluster.volumePotential}</span></div>
                                                <div>Warehouse Score: <span className="font-medium text-white">{cluster.warehouseScore}</span></div>
                                            </div>
                                        </div>
                                    ))}
                                    {!warehouseClusters.length ? (
                                        <div className="rounded-2xl border border-dashed border-slate-700 bg-[#111725] px-5 py-8 text-sm text-slate-400">
                                            Run a competitor harvest to create clustered future opportunities for later SERP validation.
                                        </div>
                                    ) : null}
                                </div>
                            </section>
                        </div>

                        <aside className="flex flex-col gap-6 xl:sticky xl:top-6 xl:self-start">
                            <section className="rounded-[28px] border border-slate-800 bg-[#0c1420] p-6">
                                <div className="text-xs uppercase tracking-[0.3em] text-emerald-300/75">Warehouse Rules</div>
                                <h2 className="mt-2 text-xl font-semibold">What gets stored</h2>
                                <div className="mt-4 grid gap-3 text-sm text-slate-300">
                                    <div className="rounded-2xl border border-slate-700 bg-[#111a28] p-3">Keep keywords when topical fit, article intent, rank 4–30, and volume thresholds line up.</div>
                                    <div className="rounded-2xl border border-slate-700 bg-[#111a28] p-3">Reject branded, local-service, ecommerce, or off-topic phrases before enrichment.</div>
                                    <div className="rounded-2xl border border-slate-700 bg-[#111a28] p-3">Clusters stay as <span className="font-medium text-white">Ready for SERP validation</span> or <span className="font-medium text-white">Future opportunity</span>.</div>
                                </div>
                                <div className="mt-4 rounded-2xl border border-slate-700 bg-[#111a28] p-4 text-sm text-slate-300">
                                    Saved topic id: <span className="font-medium text-white">{warehouseSummary?.savedTopicId || 'Not saved yet'}</span>
                                </div>
                            </section>
                        </aside>
                    </div>
                </div>
            </div>
        )
    }

    return (
        <div className="min-h-screen bg-[#06101a] text-white">
            <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-6 py-8">
                <header className="rounded-[28px] border border-slate-800 bg-[radial-gradient(circle_at_top_left,rgba(14,165,233,0.16),transparent_35%),linear-gradient(180deg,#0d1624_0%,#0a1019_100%)] p-6 shadow-[0_20px_80px_rgba(2,8,23,0.35)]">
                    <div className="text-xs uppercase tracking-[0.35em] text-sky-300/80">Research</div>
                    <h1 className="mt-3 text-3xl font-semibold tracking-tight">SERP Opportunity Keyword Finder</h1>
                    <p className="mt-3 max-w-4xl text-sm text-slate-300">
                        Move from topic idea to one article opportunity. The system evaluates whether the topic is worth pursuing before it spends credits on broader keyword harvesting.
                    </p>
                </header>

                {workflowToggle}

                {error ? (
                    <div className="rounded-2xl border border-rose-500/30 bg-rose-950/30 px-4 py-3 text-sm text-rose-200">{error}</div>
                ) : null}

                {success ? (
                    <div className="rounded-2xl border border-emerald-500/30 bg-emerald-950/20 px-4 py-3 text-sm text-emerald-200">{success}</div>
                ) : null}

                <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_320px]">
                    <div className="flex flex-col gap-6">
                        <section className="rounded-[28px] border border-slate-800 bg-[#0c1420] p-6">
                            <div className="mb-5 flex items-center justify-between">
                                <div>
                                    <div className="text-xs uppercase tracking-[0.3em] text-sky-300/75">Step 1</div>
                                    <h2 className="mt-2 text-2xl font-semibold">Topic Input Panel</h2>
                                </div>
                                <div className="text-sm text-slate-400">{activeProject?.domain || activeProject?.app_name || 'Select a project'}</div>
                            </div>

                            <div className="grid gap-4 md:grid-cols-2">
                                <label className="flex flex-col gap-2">
                                    <span className="text-sm text-slate-300">Website Category</span>
                                    <select
                                        value={primaryCategoryId}
                                        onChange={(event) => {
                                            setPrimaryCategoryId(event.target.value)
                                            setSecondaryCategoryId('')
                                        }}
                                        className="h-12 rounded-2xl border border-slate-700 bg-[#141d2c] px-4 text-sm text-white outline-none transition focus:border-sky-400"
                                    >
                                        <option value="">Select category</option>
                                        {primaryCategories.map((category) => (
                                            <option key={category.id} value={category.id}>{category.name}</option>
                                        ))}
                                    </select>
                                </label>

                                <label className="flex flex-col gap-2">
                                    <span className="text-sm text-slate-300">Subcategory</span>
                                    <select
                                        value={secondaryCategoryId}
                                        onChange={(event) => setSecondaryCategoryId(event.target.value)}
                                        className="h-12 rounded-2xl border border-slate-700 bg-[#141d2c] px-4 text-sm text-white outline-none transition focus:border-sky-400"
                                    >
                                        <option value="">Select subcategory</option>
                                        {secondaryCategories.map((category) => (
                                            <option key={category.id} value={category.id}>{category.name}</option>
                                        ))}
                                    </select>
                                </label>

                                <label className="flex flex-col gap-2 md:col-span-2">
                                    <span className="text-sm text-slate-300">Topic Idea</span>
                                    <textarea
                                        value={topicIdea}
                                        onChange={(event) => setTopicIdea(event.target.value)}
                                        rows={3}
                                        placeholder="Eco-friendly upgrades that increase property value"
                                        className="min-h-[96px] rounded-2xl border border-slate-700 bg-[#141d2c] px-4 py-3 text-sm text-white outline-none transition placeholder:text-slate-500 focus:border-sky-400"
                                    />
                                </label>

                                <label className="flex flex-col gap-2">
                                    <span className="text-sm text-slate-300">Target Country</span>
                                    <select value={countryLabel} onChange={(event) => setCountryLabel(event.target.value)} className="h-12 rounded-2xl border border-slate-700 bg-[#141d2c] px-4 text-sm text-white outline-none transition focus:border-sky-400">
                                        {COUNTRY_OPTIONS.map((country) => (
                                            <option key={country.label} value={country.label}>{country.label}</option>
                                        ))}
                                    </select>
                                </label>

                                <label className="flex flex-col gap-2">
                                    <span className="text-sm text-slate-300">Language</span>
                                    <select value={languageLabel} onChange={(event) => setLanguageLabel(event.target.value)} className="h-12 rounded-2xl border border-slate-700 bg-[#141d2c] px-4 text-sm text-white outline-none transition focus:border-sky-400">
                                        {LANGUAGE_OPTIONS.map((language) => (
                                            <option key={language.label} value={language.label}>{language.label}</option>
                                        ))}
                                    </select>
                                </label>

                                <label className="flex flex-col gap-2">
                                    <span className="text-sm text-slate-300">Search Engine</span>
                                    <select value={searchEngine} onChange={(event) => setSearchEngine(event.target.value)} className="h-12 rounded-2xl border border-slate-700 bg-[#141d2c] px-4 text-sm text-white outline-none transition focus:border-sky-400">
                                        <option value="Google">Google</option>
                                    </select>
                                </label>

                                <label className="flex flex-col gap-2">
                                    <span className="text-sm text-slate-300">Device</span>
                                    <select value={device} onChange={(event) => setDevice(event.target.value as DeviceOption)} className="h-12 rounded-2xl border border-slate-700 bg-[#141d2c] px-4 text-sm text-white outline-none transition focus:border-sky-400">
                                        <option value="Desktop">Desktop</option>
                                        <option value="Mobile">Mobile</option>
                                    </select>
                                </label>

                                <label className="flex flex-col gap-2 md:col-span-2">
                                    <span className="text-sm text-slate-300">Research Mode</span>
                                    <select
                                        value={researchScope}
                                        onChange={(event) => setResearchScope(event.target.value as ResearchScope)}
                                        className="h-12 rounded-2xl border border-slate-700 bg-[#141d2c] px-4 text-sm text-white outline-none transition focus:border-sky-400"
                                    >
                                        <option value="focused">Focused: stay close to this topic</option>
                                        <option value="expanded">Expanded nearby: find easier adjacent opportunities</option>
                                    </select>
                                    <div className="text-xs text-slate-400">
                                        {researchScope === 'focused'
                                            ? 'Keeps the article opportunity tightly tied to your input topic.'
                                            : 'Uses the topic as a starting point and allows nearby category-relevant keyword clusters if they look easier to rank.'}
                                    </div>
                                </label>
                            </div>

                            <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                                <button
                                    type="button"
                                    onClick={handleAnalyze}
                                    disabled={!projectId || !primaryCategoryId || !topicIdea.trim() || isLoading}
                                    className="inline-flex items-center justify-center rounded-2xl bg-sky-400 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:bg-sky-300 disabled:cursor-not-allowed disabled:opacity-50"
                                >
                                    {isLoading ? 'Analyzing…' : 'Analyze SERP Opportunity'}
                                </button>
                                <div className="text-sm text-slate-400">
                                    This will run 3 SERP probe searches before spending credits on keyword expansion.
                                </div>
                            </div>

                            <div className="mt-5 rounded-2xl border border-slate-700 bg-[#111a28] p-4 text-sm text-slate-300">
                                <div className="font-medium text-white">Category context</div>
                                <div className="mt-2">{primaryCategory?.name || 'No category selected'}</div>
                                <div className="mt-1 text-slate-400">{primaryCategory?.description || 'Choose a category to anchor the topic.'}</div>
                                <div className="mt-3">{secondaryCategory?.name || 'No subcategory selected'}</div>
                                <div className="mt-1 text-slate-400">{secondaryCategory?.description || 'Optional: add a subcategory to tighten the probe intent.'}</div>
                            </div>
                        </section>

                        <section className="rounded-[28px] border border-slate-800 bg-[#0c1420] p-6">
                            <div className="mb-4 flex items-center justify-between">
                                <div>
                                    <div className="text-xs uppercase tracking-[0.3em] text-sky-300/75">Step 2</div>
                                    <h2 className="mt-2 text-2xl font-semibold">SERP Probe Preview</h2>
                                </div>
                                <div className="rounded-full border border-slate-700 bg-[#141d2c] px-3 py-1 text-xs text-slate-300">
                                    Estimated API calls: 3
                                </div>
                            </div>

                            <div className="grid gap-3">
                                {probePreview.map((probe, index) => (
                                    <div key={`${probe.label}-${index}`} className="rounded-2xl border border-slate-700 bg-[#111a28] p-4">
                                        <div className="flex items-center justify-between gap-3">
                                            <div className="text-sm font-medium text-white">{index + 1}. {probe.label} intent</div>
                                            <span className="rounded-full border border-sky-500/20 bg-sky-500/10 px-3 py-1 text-xs text-sky-200">{probe.label}</span>
                                        </div>
                                        <div className="mt-2 text-base text-slate-200">{probe.query}</div>
                                    </div>
                                ))}
                            </div>
                        </section>

                        <section className="rounded-[28px] border border-slate-800 bg-[#0c1420] p-6">
                            <div className="mb-4">
                                <div className="text-xs uppercase tracking-[0.3em] text-sky-300/75">Step 3</div>
                                <h2 className="mt-2 text-2xl font-semibold">SERP Opportunity Score Card</h2>
                            </div>

                            <div className="rounded-[24px] border border-slate-700 bg-[linear-gradient(135deg,rgba(56,189,248,0.12),rgba(15,23,42,0.75))] p-5">
                                <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                                    <div>
                                        <div className="text-sm text-slate-300">SERP Opportunity Score</div>
                                        <div className="mt-2 text-4xl font-semibold text-white">{opportunityScore} / 100</div>
                                        <div className="mt-2 text-lg text-sky-200">Status: {scoreStatus}</div>
                                    </div>
                                    <div className="grid gap-2 text-sm">
                                        {opportunitySignals.map((signal) => (
                                            <div key={signal.label} className="text-slate-200">
                                                {signal.label === 'Some authority sites present'
                                                    ? signal.passed ? '⚠️' : '✅'
                                                    : signal.passed ? '✅' : '❌'} {signal.label}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </section>

                        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_320px]">
                            <section className="rounded-[28px] border border-slate-800 bg-[#0c1420] p-6">
                                <div className="mb-4">
                                    <div className="text-xs uppercase tracking-[0.3em] text-sky-300/75">Step 4</div>
                                    <h2 className="mt-2 text-2xl font-semibold">SERP Competitor Overlap</h2>
                                </div>

                                <div className="overflow-hidden rounded-2xl border border-slate-700 bg-[#111a28]">
                                    <div className="hidden grid-cols-[minmax(0,1.5fr)_100px_100px_110px_110px_90px] gap-3 border-b border-slate-700 px-4 py-3 text-[11px] uppercase tracking-[0.26em] text-slate-400 lg:grid">
                                        <div>Domain</div>
                                        <div>Times Seen</div>
                                        <div>Best Rank</div>
                                        <div>Page Type</div>
                                        <div>Strength</div>
                                        <div>Use?</div>
                                    </div>
                                    <div className="divide-y divide-slate-800">
                                        {competitorPages.length ? competitorPages.map((entry) => {
                                            const useState = useLabel(entry)
                                            const domain = String(entry.domain || '')
                                            return (
                                                <div key={`${domain}-${entry.url}`} className="grid gap-2 px-4 py-4 lg:grid-cols-[minmax(0,1.5fr)_100px_100px_110px_110px_90px] lg:items-center">
                                                    <div className="font-medium text-white">{domain || 'Unknown domain'}</div>
                                                    <div className="text-sm text-slate-300">{safeNumber(entry.query_hits || entry.domain_hits || 0)}</div>
                                                    <div className="text-sm text-slate-300">{safeNumber(entry.best_rank || 0) || 'n/a'}</div>
                                                    <div className="text-sm text-slate-300">{pageTypeLabel(String(entry.url || ''), domain)}</div>
                                                    <div className="text-sm text-slate-300">{strengthLabel(domain)}</div>
                                                    <div>
                                                        <span className={`inline-flex rounded-full border px-2 py-1 text-xs ${useState.tone}`}>{useState.label}</span>
                                                    </div>
                                                </div>
                                            )
                                        }) : (
                                            <div className="px-4 py-6 text-sm text-slate-400">
                                                Run an analysis to see repeated competitors across the 3 probe searches.
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </section>

                            <aside className="rounded-[28px] border border-slate-800 bg-[#0c1420] p-6 xl:sticky xl:top-6 xl:self-start">
                                <div className="text-xs uppercase tracking-[0.3em] text-sky-300/75">API Usage Plan</div>
                                <h2 className="mt-2 text-xl font-semibold">Credit control</h2>
                                <div className="mt-4 grid gap-3 text-sm text-slate-300">
                                    <div className="rounded-2xl border border-slate-700 bg-[#111a28] p-3">
                                        <div className="font-medium text-white">Step 1: SERP probes</div>
                                        <div className="mt-1">Used: 3 calls</div>
                                    </div>
                                    <div className="rounded-2xl border border-slate-700 bg-[#111a28] p-3">
                                        <div className="font-medium text-white">Step 2: Competitor keyword harvesting</div>
                                        <div className="mt-1">Planned: {Math.max(0, competitorPages.length)} URL-level Ranked Keywords calls</div>
                                    </div>
                                    <div className="rounded-2xl border border-slate-700 bg-[#111a28] p-3">
                                        <div className="font-medium text-white">Step 3: Keyword expansion</div>
                                        <div className="mt-1">Planned: {Math.max(0, controlledExpansion.length || Math.min(5, competitorPages.length))} Keyword Suggestions calls</div>
                                    </div>
                                    <div className="rounded-2xl border border-slate-700 bg-[#111a28] p-3">
                                        <div className="font-medium text-white">Step 4: Keyword metrics</div>
                                        <div className="mt-1">Planned: 1 Bulk KD call</div>
                                        <div>Planned: 1 Keyword Overview batch</div>
                                    </div>
                                </div>
                                <div className="mt-4 rounded-2xl border border-emerald-500/20 bg-emerald-500/10 p-4 text-sm text-emerald-100">
                                    Estimated total: {3 + competitorPages.length + Math.max(controlledExpansion.length || Math.min(5, competitorPages.length), 0) + 2} calls
                                    <div className="mt-2 text-emerald-200/90">Avoided broad keyword expansion: 300+ possible wasted keyword checks</div>
                                </div>
                            </aside>
                        </div>

                        <section className="rounded-[28px] border border-slate-800 bg-[#0c1420] p-6">
                            <div className="mb-4">
                                <div className="text-xs uppercase tracking-[0.3em] text-sky-300/75">Step 5</div>
                                <h2 className="mt-2 text-2xl font-semibold">Selected Competitor Pages</h2>
                            </div>
                            <div className="grid gap-3">
                                {competitorPages.length ? competitorPages.map((entry) => {
                                    const useState = useLabel(entry)
                                        const reason =
                                            useState.label === 'Yes'
                                                ? 'Niche site, article format, and repeated across probe searches.'
                                            : useState.label === 'Maybe'
                                                ? 'Topical match is good, but the domain still needs manual review.'
                                                : 'Too strong, too broad, or not a realistic competitor for harvesting.'
                                    return (
                                        <div key={`page-${entry.url}`} className="rounded-2xl border border-slate-700 bg-[#111a28] p-4">
                                            <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                                                <div>
                                                    <div className="text-sm font-medium text-white">{entry.url}</div>
                                                    <div className="mt-2 text-sm text-slate-300">
                                                        {useState.label === 'Yes' ? '✅' : useState.label === 'Maybe' ? '⚠️' : '❌'} {reason}
                                                    </div>
                                                </div>
                                                <span className={`inline-flex rounded-full border px-3 py-1 text-xs ${useState.tone}`}>{useState.label}</span>
                                            </div>
                                        </div>
                                    )
                                }) : (
                                    <div className="rounded-2xl border border-dashed border-slate-700 bg-[#111725] px-5 py-8 text-sm text-slate-400">
                                        No competitor pages selected yet.
                                    </div>
                                )}
                            </div>
                        </section>

                        <section className="rounded-[28px] border border-slate-800 bg-[#0c1420] p-6">
                            <div className="mb-4 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                                <div>
                                    <div className="text-xs uppercase tracking-[0.3em] text-sky-300/75">Step 6</div>
                                    <h2 className="mt-2 text-2xl font-semibold">Harvested Keyword Candidates</h2>
                                </div>
                                <div className="flex flex-wrap gap-2 text-xs">
                                    <button type="button" onClick={() => setIncludeKeepOnly((value) => !value)} className={`rounded-full border px-3 py-1 ${includeKeepOnly ? 'border-emerald-500/20 bg-emerald-500/10 text-emerald-200' : 'border-slate-700 text-slate-300'}`}>Only Keep</button>
                                    <button type="button" onClick={() => setOnlyLowKd((value) => !value)} className={`rounded-full border px-3 py-1 ${onlyLowKd ? 'border-sky-500/20 bg-sky-500/10 text-sky-200' : 'border-slate-700 text-slate-300'}`}>Low KD</button>
                                    <button type="button" onClick={() => setOnlyHighCpc((value) => !value)} className={`rounded-full border px-3 py-1 ${onlyHighCpc ? 'border-amber-500/20 bg-amber-500/10 text-amber-200' : 'border-slate-700 text-slate-300'}`}>High CPC</button>
                                    <button type="button" onClick={() => setOnlyArticleIntent((value) => !value)} className={`rounded-full border px-3 py-1 ${onlyArticleIntent ? 'border-violet-500/20 bg-violet-500/10 text-violet-200' : 'border-slate-700 text-slate-300'}`}>Article Intent</button>
                                    <button type="button" onClick={() => setSeenOnMultiplePages((value) => !value)} className={`rounded-full border px-3 py-1 ${seenOnMultiplePages ? 'border-cyan-500/20 bg-cyan-500/10 text-cyan-200' : 'border-slate-700 text-slate-300'}`}>Seen on 2+ pages</button>
                                    <button type="button" onClick={() => setHideBrands((value) => !value)} className={`rounded-full border px-3 py-1 ${hideBrands ? 'border-rose-500/20 bg-rose-500/10 text-rose-200' : 'border-slate-700 text-slate-300'}`}>Hide Brands</button>
                                </div>
                            </div>

                            <div className="overflow-hidden rounded-2xl border border-slate-700 bg-[#111a28]">
                                <div className="hidden grid-cols-[minmax(0,2fr)_90px_110px_90px_70px_70px_90px_90px] gap-3 border-b border-slate-700 px-4 py-3 text-[11px] uppercase tracking-[0.26em] text-slate-400 lg:grid">
                                    <div>Keyword</div>
                                    <div>Source Pages</div>
                                    <div>Best Rank</div>
                                    <div>Volume</div>
                                    <div>KD</div>
                                    <div>CPC</div>
                                    <div>Intent</div>
                                    <div>Status</div>
                                </div>
                                <div className="divide-y divide-slate-800">
                                    {filteredKeywords.slice(0, 40).map((row) => {
                                        const trend = (row.trend_json || {}) as Record<string, any>
                                        const sourcePages = Array.isArray(trend.source_urls) ? trend.source_urls.length : trend.source_url ? 1 : 0
                                        const bestRank = sourcePages ? Math.max(1, Math.round(20 - safeNumber(trend.competitor_support_score) * 10)) : 'n/a'
                                        const status = row.is_filtered_out ? 'Reject' : safeNumber(row.opportunity_score) >= 65 ? 'Keep' : 'Review'
                                        return (
                                            <div key={row.id} className="grid gap-2 px-4 py-4 lg:grid-cols-[minmax(0,2fr)_90px_110px_90px_70px_70px_90px_90px] lg:items-center">
                                                <div className="font-medium text-white">{row.keyword}</div>
                                                <div className="text-sm text-slate-300">{sourcePages}</div>
                                                <div className="text-sm text-slate-300">{bestRank}</div>
                                                <div className="text-sm text-slate-300">{safeNumber(row.search_volume) || 'n/a'}</div>
                                                <div className="text-sm text-slate-300">{row.keyword_difficulty != null ? Math.round(safeNumber(row.keyword_difficulty)) : 'n/a'}</div>
                                                <div className="text-sm text-slate-300">{row.cpc != null ? Number(row.cpc).toFixed(2) : 'n/a'}</div>
                                                <div className="text-sm text-slate-300">{keywordIntentLabel(row.intent_label)}</div>
                                                <div className="text-sm text-slate-300">{status}</div>
                                            </div>
                                        )
                                    })}
                                    {!filteredKeywords.length ? (
                                        <div className="px-4 py-6 text-sm text-slate-400">No curated keywords match the current filters yet.</div>
                                    ) : null}
                                </div>
                            </div>
                        </section>

                        <section className="rounded-[28px] border border-slate-800 bg-[#0c1420] p-6">
                            <div className="mb-4">
                                <div className="text-xs uppercase tracking-[0.3em] text-sky-300/75">Step 7</div>
                                <h2 className="mt-2 text-2xl font-semibold">Keyword Cluster Cards</h2>
                            </div>
                            <div className="grid gap-4">
                                {clusters.length ? clusters.map((cluster, index) => {
                                    const score = Math.round(safeNumber(cluster.opportunity_score))
                                    const recommendation = clusterRecommendation(score)
                                    return (
                                        <button
                                            type="button"
                                            key={cluster.id}
                                            onClick={() => setSelectedClusterId(cluster.id)}
                                            className={`rounded-3xl border p-5 text-left transition ${selectedClusterId === cluster.id ? 'border-sky-400 bg-sky-500/10' : 'border-slate-700 bg-[#111a28] hover:border-slate-500'}`}
                                        >
                                            <div className="text-xs uppercase tracking-[0.24em] text-slate-400">Cluster {index + 1}</div>
                                            <h3 className="mt-2 text-xl font-semibold text-white">{cluster.cluster_name}</h3>
                                            <div className="mt-4 text-sm text-slate-300">
                                                Primary keyword: <span className="font-medium text-white">{cluster.primary_keyword || cluster.cluster_name}</span>
                                            </div>
                                            <div className="mt-4 flex flex-wrap gap-2">
                                                {(cluster.secondary_keywords_json || []).slice(0, 4).map((keyword) => (
                                                    <span key={`${cluster.id}-${keyword}`} className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-200">{keyword}</span>
                                                ))}
                                            </div>
                                            <div className="mt-4 grid gap-2 md:grid-cols-[160px_1fr_180px] md:items-center">
                                                <div className="text-sm text-slate-300">Opportunity Score: <span className="font-medium text-white">{score} / 100</span></div>
                                                <div className="text-sm text-slate-300">Recommendation: <span className="font-medium text-white">{recommendation}</span></div>
                                                <div className="text-right">
                                                    <span className="inline-flex rounded-full border border-slate-700 bg-[#162132] px-3 py-1 text-xs text-slate-200">
                                                        {recommendation === 'Create article now' ? 'Generate Article Brief' : recommendation === 'Good candidate' ? 'Review SERP' : 'Save Candidate'}
                                                    </span>
                                                </div>
                                            </div>
                                        </button>
                                    )
                                }) : (
                                    <div className="rounded-2xl border border-dashed border-slate-700 bg-[#111725] px-5 py-8 text-sm text-slate-400">
                                        No keyword clusters yet. If the topic fails the gate, the page stops before cluster creation.
                                    </div>
                                )}
                            </div>
                        </section>

                        <section className="rounded-[28px] border border-slate-800 bg-[#0c1420] p-6">
                            <div className="mb-4">
                                <div className="text-xs uppercase tracking-[0.3em] text-sky-300/75">Step 8</div>
                                <h2 className="mt-2 text-2xl font-semibold">Final Article Recommendation</h2>
                            </div>

                            {finalArticle ? (
                                <div className="rounded-[24px] border border-emerald-500/20 bg-[linear-gradient(135deg,rgba(16,185,129,0.12),rgba(17,24,39,0.82))] p-5">
                                    <div className="text-sm text-slate-300">Recommended Article</div>
                                    <h3 className="mt-2 text-3xl font-semibold text-white">{finalArticle.title}</h3>
                                    <div className="mt-4 text-sm text-slate-300">
                                        Primary Keyword: <span className="font-medium text-white">{finalArticle.primaryKeyword}</span>
                                    </div>
                                    <div className="mt-4 flex flex-wrap gap-2">
                                        {finalArticle.secondaryKeywords.map((keyword) => (
                                            <span key={keyword} className="rounded-full border border-slate-700 bg-[#102235] px-3 py-1 text-xs text-slate-100">{keyword}</span>
                                        ))}
                                    </div>
                                    <div className="mt-5 grid gap-2 text-sm text-slate-200">
                                        <div>- Small niche sites already rank in the top 10</div>
                                        <div>- SERP has multiple article-style pages</div>
                                        <div>- Keyword difficulty looks achievable for this cluster</div>
                                        <div>- CPC suggests commercial value</div>
                                        <div>- Search intent matches your website category</div>
                                        <div>- One article can cover several closely related queries</div>
                                    </div>
                                    <div className="mt-5 rounded-2xl border border-slate-700 bg-[#0f1928] p-4 text-sm text-slate-200">
                                        <div className="font-medium text-white">Recommended Content Angle</div>
                                        <div className="mt-2">
                                            {selectedCluster?.article_angle || 'Focus on practical buyer value, achievable ROI, and a tight angle that matches the proven competitor cluster.'}
                                        </div>
                                    </div>
                                    <div className="mt-5 flex flex-wrap gap-3">
                                        <button type="button" onClick={openTopicDetail} className="rounded-2xl bg-emerald-400 px-4 py-3 text-sm font-semibold text-slate-950 transition hover:bg-emerald-300">
                                            Generate Full SEO Brief
                                        </button>
                                        <button type="button" onClick={openTopicDetail} className="rounded-2xl border border-slate-600 bg-slate-900 px-4 py-3 text-sm font-semibold text-slate-100 transition hover:border-sky-400">
                                            Save Opportunity
                                        </button>
                                        <button type="button" onClick={() => { setTopicIdea(''); setCurrentRun(null); setAllKeywordCandidates([]); setClusters([]); setCurrentTopic(null); }} className="rounded-2xl border border-slate-600 bg-slate-900 px-4 py-3 text-sm font-semibold text-slate-100 transition hover:border-rose-400">
                                            Reject and Analyze Another Topic
                                        </button>
                                    </div>
                                </div>
                            ) : (
                                <div className="rounded-2xl border border-dashed border-slate-700 bg-[#111725] px-5 py-8 text-sm text-slate-400">
                                    Analyze a topic and surface at least one viable cluster to get a final article recommendation.
                                </div>
                            )}
                        </section>

                        <section className="rounded-[28px] border border-slate-800 bg-[#0c1420] p-6">
                            <div className="mb-4">
                                <div className="text-xs uppercase tracking-[0.3em] text-sky-300/75">Recent Analyses</div>
                                <h2 className="mt-2 text-2xl font-semibold">Load a previous topic</h2>
                            </div>
                            <div className="flex flex-wrap gap-3">
                                {recentTopics
                                    .filter((topic) => topic.topic_generation_metadata?.workflow_type !== 'domain_keyword_warehouse')
                                    .map((topic) => (
                                    <button
                                        type="button"
                                        key={topic.id}
                                        onClick={() => void loadTopicRun(topic)}
                                        className={`rounded-full border px-4 py-2 text-left text-sm transition ${currentTopic?.id === topic.id ? 'border-sky-400 bg-sky-500/10 text-white' : 'border-slate-700 bg-[#111a28] text-slate-300 hover:border-slate-500'}`}
                                    >
                                        {topic.title}
                                    </button>
                                ))}
                                {!recentTopics.length ? (
                                    <div className="text-sm text-slate-400">No saved analyses yet for this scope.</div>
                                ) : null}
                            </div>
                        </section>
                    </div>

                    <aside className="flex flex-col gap-6 xl:sticky xl:top-6 xl:self-start">
                        <section className="rounded-[28px] border border-slate-800 bg-[#0c1420] p-6">
                            <div className="text-xs uppercase tracking-[0.3em] text-sky-300/75">Decision Logic</div>
                            <h2 className="mt-2 text-xl font-semibold">Why this topic passed</h2>
                            <div className="mt-4 grid gap-3 text-sm text-slate-300">
                                <div className="flex items-center justify-between rounded-2xl border border-slate-700 bg-[#111a28] px-4 py-3">
                                    <span>SERP Weakness</span>
                                    <span className="font-medium text-white">{decisionSidebarScores.serp} / 100</span>
                                </div>
                                <div className="flex items-center justify-between rounded-2xl border border-slate-700 bg-[#111a28] px-4 py-3">
                                    <span>Keyword Difficulty</span>
                                    <span className="font-medium text-white">{decisionSidebarScores.kd} / 100</span>
                                </div>
                                <div className="flex items-center justify-between rounded-2xl border border-slate-700 bg-[#111a28] px-4 py-3">
                                    <span>Search Volume</span>
                                    <span className="font-medium text-white">{decisionSidebarScores.volume} / 100</span>
                                </div>
                                <div className="flex items-center justify-between rounded-2xl border border-slate-700 bg-[#111a28] px-4 py-3">
                                    <span>Commercial Value</span>
                                    <span className="font-medium text-white">{decisionSidebarScores.commercial} / 100</span>
                                </div>
                                <div className="flex items-center justify-between rounded-2xl border border-slate-700 bg-[#111a28] px-4 py-3">
                                    <span>Topical Fit</span>
                                    <span className="font-medium text-white">{decisionSidebarScores.topicalFit} / 100</span>
                                </div>
                            </div>
                            <div className="mt-4 rounded-2xl border border-emerald-500/20 bg-emerald-500/10 px-4 py-4">
                                <div className="text-sm text-slate-300">Final Score</div>
                                <div className="mt-1 text-3xl font-semibold text-white">{decisionSidebarScores.final} / 100</div>
                            </div>
                            <div className="mt-4 rounded-2xl border border-slate-700 bg-[#111a28] p-4 text-sm text-slate-300">
                                <div className="font-medium text-white">Formula</div>
                                <div className="mt-2">
                                    Final Score = SERP Weakness + Keyword Difficulty + Volume + CPC / Ad Competition + Topical Fit
                                </div>
                            </div>
                        </section>
                    </aside>
                </div>
            </div>
        </div>
    )
}

export default ResearchRebuildStrategicPage
