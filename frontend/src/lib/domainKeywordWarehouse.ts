export type WarehouseScope = 'exact_pages' | 'relevant_sections' | 'whole_domain' | 'reject'

export type CategoryIndexRow = {
    category_code: number
    category_name: string
    category_code_parent?: number | null
}

export type DomainCategoryRow = {
    category_codes: number[]
    category_names: string[]
    organic_count: number
    organic_etv: number
    pos_4_10: number
    pos_11_20: number
    pos_21_30: number
}

export type WarehouseRelevantPage = {
    url: string
    title: string
    traffic: number
    organicCount: number
    positionOpportunityCount: number
    topicMatchScore: number
    topicMatchLabel: 'High' | 'Medium' | 'Low'
    pageType: 'Article' | 'Guide' | 'Comparison' | 'FAQ' | 'Mixed'
    include: boolean
    sectionKey: string
}

export type WarehouseKeywordCandidate = {
    keyword: string
    normalizedKeyword: string
    sourceUrl: string
    sourceDomain: string
    competitorRank: number
    searchVolume: number
    keywordDifficulty: number | null
    cpc: number | null
    paidCompetition: number | null
    intent: string
    topicalScore: number
    articlePotentialScore: number
    rankSignalScore: number
    volumeScore: number
    warehouseScore: number
    rejected: boolean
    rejectionReasons: string[]
}

export type WarehouseCluster = {
    id: string
    clusterName: string
    primaryKeyword: string
    supportingKeywords: string[]
    sourceUrls: string[]
    avgKd: number | null
    volumePotential: number
    warehouseScore: number
    priority: 'High' | 'Medium' | 'Low'
    status: 'Ready for SERP validation' | 'Future opportunity'
}

const ARTICLE_TERMS = ['how', 'guide', 'ideas', 'tips', 'ways', 'why', 'what', 'compare', 'comparison', 'vs', 'versus', 'roi', 'worth', 'increase', 'improve', 'value', 'faq']
const NEGATIVE_TERMS = ['jobs', 'job', 'coupon', 'coupons', 'near me', 'reddit', 'pdf', 'manual', 'parts', 'replacement parts', 'login', 'phone number']
const ECOMMERCE_TERMS = ['buy', 'price', 'for sale', 'shop', 'amazon', 'walmart', 'deal', 'deals']
const LOCAL_TERMS = ['near me', 'contractor', 'contractors', 'installer', 'installers', 'company', 'companies', 'service', 'services']
const BRAND_TERMS = ['forbes', 'reddit', 'facebook', 'youtube', 'tiktok', 'amazon', 'bob vila', 'home depot', 'lowes']
const GENERIC_CLUSTER_STOPWORDS = new Set([
    'a', 'an', 'and', 'are', 'best', 'can', 'does', 'for', 'from', 'guide', 'home', 'how',
    'in', 'increase', 'improve', 'is', 'it', 'of', 'on', 'or', 'the', 'to', 'value', 'what',
    'why', 'with', 'worth',
])

const CLUSTER_SYNONYMS: Record<string, string> = {
    eco: 'green',
    ecofriendly: 'green',
    sustainable: 'green',
    upgrades: 'improvements',
    upgrade: 'improvement',
    renovations: 'renovation',
    renovationsroi: 'renovation',
    resale: 'value',
    efficiency: 'efficient',
    windows: 'window',
    panels: 'panel',
}

const TOPIC_ALIASES: Record<string, string[]> = {
    productivity: ['focus', 'workflow', 'routine', 'routines', 'habit', 'habits', 'planning', 'task', 'tasks', 'output'],
    efficiency: ['focus', 'workflow', 'system', 'systems', 'optimize', 'optimization', 'faster'],
    automation: ['automate', 'automated', 'workflow', 'zapier', 'integration', 'integrations'],
    ai: ['artificial', 'intelligence', 'assistant', 'assistants', 'llm', 'agents', 'agentic'],
    agent: ['agents', 'assistant', 'assistants', 'automation', 'workflow'],
    agents: ['agent', 'assistant', 'assistants', 'automation', 'workflow'],
    wellness: ['health', 'sleep', 'mindfulness', 'energy', 'stress'],
    lifestyle: ['routine', 'routines', 'life', 'habits', 'wellbeing', 'wellness'],
    thinking: ['decision', 'decisions', 'clarity', 'mindset', 'focus'],
    work: ['career', 'team', 'office', 'productivity', 'workflow'],
    daily: ['routine', 'routines', 'habit', 'habits', 'everyday'],
}

function normalizeText(value: string) {
    return String(value || '').trim().replace(/\s+/g, ' ')
}

export function normalizeDomainInput(value: string) {
    const cleaned = normalizeText(value)
        .replace(/^https?:\/\//i, '')
        .replace(/^www\./i, '')
        .replace(/\/+$/, '')
    return cleaned.split('/')[0]
}

export function splitTopicInput(value: string) {
    return normalizeText(value)
        .split(/\s*,\s*|\n+/)
        .map((item) => normalizeText(item))
        .filter(Boolean)
}

function tokenize(value: string) {
    return normalizeText(value)
        .toLowerCase()
        .replace(/[^a-z0-9\s]/g, ' ')
        .split(/\s+/)
        .filter(Boolean)
}

function normalizeToken(token: string) {
    const cleaned = token.toLowerCase().replace(/[^a-z0-9]/g, '')
    if (!cleaned) return ''
    const synonym = CLUSTER_SYNONYMS[cleaned]
    if (synonym) return synonym
    if (cleaned.endsWith('ies')) return `${cleaned.slice(0, -3)}y`
    if (cleaned.endsWith('s') && cleaned.length > 4) return cleaned.slice(0, -1)
    return cleaned
}

function expandTopicVocabulary(topics: string[]) {
    const expanded = new Set<string>()
    for (const topic of topics) {
        for (const token of tokenize(topic)) {
            const normalized = normalizeToken(token)
            if (!normalized) continue
            expanded.add(normalized)
            for (const alias of TOPIC_ALIASES[normalized] || []) {
                const normalizedAlias = normalizeToken(alias)
                if (normalizedAlias) expanded.add(normalizedAlias)
            }
        }
    }
    return expanded
}

function titleCase(value: string) {
    return normalizeText(value)
        .split(' ')
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ')
}

function keywordVolumeScore(volume: number) {
    if (volume <= 0) return 0
    const capped = Math.min(volume, 5000)
    return Math.min(100, Math.round((Math.log10(capped + 1) / Math.log10(5001)) * 100))
}

function keywordRankSignal(rank: number) {
    if (rank <= 0) return 0
    if (rank <= 10) return 95
    if (rank <= 20) return 78
    if (rank <= 30) return 62
    return 35
}

function articlePotential(keyword: string, intent: string) {
    const lower = keyword.toLowerCase()
    let score = 25
    if (ARTICLE_TERMS.some((term) => lower.includes(term))) score += 35
    if (intent.includes('informational')) score += 30
    if (intent.includes('commercial')) score += 18
    if (ECOMMERCE_TERMS.some((term) => lower.includes(term))) score -= 25
    if (LOCAL_TERMS.some((term) => lower.includes(term))) score -= 20
    return Math.max(0, Math.min(100, score))
}

function extractSectionKey(rawUrl: string) {
    try {
        const url = new URL(rawUrl)
        const parts = url.pathname.split('/').filter(Boolean)
        if (parts.length === 0) return '/'
        return `/${parts[0]}/`
    } catch {
        return '/'
    }
}

function detectPageType(url: string, title: string): WarehouseRelevantPage['pageType'] {
    const target = `${url} ${title}`.toLowerCase()
    if (target.includes(' vs ') || target.includes(' versus ') || target.includes('/compare')) return 'Comparison'
    if (target.includes('faq') || target.includes('questions')) return 'FAQ'
    if (target.includes('guide') || target.includes('/guide')) return 'Guide'
    if (ARTICLE_TERMS.some((term) => target.includes(term))) return 'Article'
    if (title.trim()) return 'Mixed'
    return 'Mixed'
}

function topicMatchLabel(score: number): WarehouseRelevantPage['topicMatchLabel'] {
    if (score >= 55) return 'High'
    if (score >= 25) return 'Medium'
    return 'Low'
}

function topicMatchScore(text: string, topicVocabulary: string[], excludedVocabulary: string[]) {
    const haystack = tokenize(text).map(normalizeToken).filter(Boolean)
    if (!haystack.length) return 0
    const joined = ` ${haystack.join(' ')} `
    const expandedVocabulary = expandTopicVocabulary(topicVocabulary)
    const expandedExcludedVocabulary = expandTopicVocabulary(excludedVocabulary)
    const positiveHits = topicVocabulary.reduce((count, item) => {
        const normalized = normalizeText(item).toLowerCase()
        if (!normalized) return count
        return count + (joined.includes(` ${normalized} `) || joined.includes(normalized) ? 1 : 0)
    }, 0)
    const negativeHits = excludedVocabulary.reduce((count, item) => {
        const normalized = normalizeText(item).toLowerCase()
        if (!normalized) return count
        return count + (joined.includes(` ${normalized} `) || joined.includes(normalized) ? 1 : 0)
    }, 0)
    const tokenHits = haystack.filter((token) => expandedVocabulary.has(token)).length
    const uniqueTokenHits = new Set(haystack.filter((token) => expandedVocabulary.has(token))).size
    const negativeTokenHits = new Set(haystack.filter((token) => expandedExcludedVocabulary.has(token))).size
    const raw = Math.min(100, positiveHits * 20 + tokenHits * 8 + uniqueTokenHits * 14 - negativeHits * 20 - negativeTokenHits * 10)
    return Math.max(0, raw)
}

export function mapDomainCategories(items: Array<Record<string, unknown>>, categories: CategoryIndexRow[]) {
    const categoryMap = new Map<number, string>()
    for (const row of categories) {
        categoryMap.set(Number(row.category_code), row.category_name)
    }
    return items.map((item) => {
        const codes = Array.isArray(item.category_codes) ? item.category_codes.map((value) => Number(value)).filter(Number.isFinite) : []
        return {
            category_codes: codes,
            category_names: codes.map((code) => categoryMap.get(code) || `Category ${code}`),
            organic_count: Number(item.organic_count || 0),
            organic_etv: Number(item.organic_etv || 0),
            pos_4_10: Number(item.pos_4_10 || 0),
            pos_11_20: Number(item.pos_11_20 || 0),
            pos_21_30: Number(item.pos_21_30 || 0),
        } satisfies DomainCategoryRow
    })
}

export function buildRelevantPages(items: Array<Record<string, unknown>>, topicVocabulary: string[], excludedVocabulary: string[]) {
    return items.map((item) => {
        const url = String(item.url || '')
        const title = String(item.title || '')
        const score = topicMatchScore(`${url} ${title}`, topicVocabulary, excludedVocabulary)
        const label = topicMatchLabel(score)
        const pageType = detectPageType(url, title)
        const organicCount = Number(item.organic_count || 0)
        const positionOpportunityCount = Number(item.pos_4_10 || 0) + Number(item.pos_11_20 || 0) + Number(item.pos_21_30 || 0)
        return {
            url,
            title,
            traffic: Number(item.traffic || 0),
            organicCount,
            positionOpportunityCount,
            topicMatchScore: score,
            topicMatchLabel: label,
            pageType,
            include: score >= 30 && ['Article', 'Guide', 'Comparison', 'FAQ', 'Mixed'].includes(pageType),
            sectionKey: extractSectionKey(url),
        } satisfies WarehouseRelevantPage
    })
}

export function scoreDomainFit(params: {
    domainCategories: DomainCategoryRow[]
    relevantPages: WarehouseRelevantPage[]
    siteCategory: string
    allowedTopics: string[]
    excludedTopics: string[]
}) {
    const topicVocabulary = [params.siteCategory, ...params.allowedTopics].filter(Boolean)
    const excludedVocabulary = params.excludedTopics.filter(Boolean)
    const categoryRows = params.domainCategories.slice(0, 8)
    const categoryMatchFromRows = categoryRows.length
        ? categoryRows.reduce((sum, row) => {
            const rowScore = topicMatchScore(row.category_names.join(' '), topicVocabulary, excludedVocabulary)
            const weight = Math.max(1, row.organic_count)
            return sum + rowScore * weight
        }, 0) / categoryRows.reduce((sum, row) => sum + Math.max(1, row.organic_count), 0)
        : 0
    const categoryMatchFromPages = params.relevantPages.length
        ? params.relevantPages.reduce((sum, page) => {
            const weight = Math.max(1, page.organicCount)
            return sum + page.topicMatchScore * weight
        }, 0) / params.relevantPages.reduce((sum, page) => sum + Math.max(1, page.organicCount), 0)
        : 0
    const categoryMatch = categoryRows.length ? categoryMatchFromRows : categoryMatchFromPages
    const contentTypeMatch = params.relevantPages.length
        ? (params.relevantPages.filter((page) => page.include || page.topicMatchScore >= 35).length / params.relevantPages.length) * 100
        : 0
    const repeatedCompetitorSignal = params.relevantPages.length
        ? (params.relevantPages.filter((page) => page.positionOpportunityCount >= 10 || page.organicCount >= 25).length / params.relevantPages.length) * 100
        : 0
    const lowNoise = params.relevantPages.length
        ? 100 - (params.relevantPages.filter((page) => page.topicMatchScore < 25).length / params.relevantPages.length) * 100
        : 0
    const fitScore = Math.round(
        categoryMatch * 0.4 +
        contentTypeMatch * 0.25 +
        repeatedCompetitorSignal * 0.2 +
        lowNoise * 0.15
    )
    return {
        categoryMatch: Math.round(categoryMatch),
        contentTypeMatch: Math.round(contentTypeMatch),
        repeatedCompetitorSignal: Math.round(repeatedCompetitorSignal),
        lowNoise: Math.round(lowNoise),
        fitScore,
        passed: fitScore >= 60,
    }
}

export function recommendWarehouseScope(fitScore: number, relevantPages: WarehouseRelevantPage[]): WarehouseScope {
    if (fitScore < 60) return 'reject'
    const included = relevantPages.filter((page) => page.include)
    const includedRatio = relevantPages.length ? included.length / relevantPages.length : 0
    if (fitScore >= 80 && includedRatio >= 0.65) return 'whole_domain'
    if (fitScore >= 68 && includedRatio >= 0.4) return 'relevant_sections'
    return 'exact_pages'
}

export function buildWarehouseKeywordCandidates(params: {
    rows: Array<Record<string, unknown>>
    siteCategory: string
    allowedTopics: string[]
    excludedTopics: string[]
    sourceDomain: string
}) {
    const topicVocabulary = [params.siteCategory, ...params.allowedTopics].filter(Boolean)
    const excludedVocabulary = [...params.excludedTopics, ...NEGATIVE_TERMS]
    return params.rows.map((row) => {
        const keyword = normalizeText(String(row.keyword || ''))
        const searchVolume = Number(row.search_volume || 0)
        const intent = String(row.intent || '').toLowerCase()
        const rank = Number(row.rank_group || row.rank_absolute || 0)
        const topicalScore = topicMatchScore(keyword, topicVocabulary, excludedVocabulary)
        const articlePotentialScore = articlePotential(keyword, intent)
        const rankSignalScore = keywordRankSignal(rank)
        const volumeScore = keywordVolumeScore(searchVolume)
        const kd = row.keyword_difficulty == null ? null : Number(row.keyword_difficulty)
        const paidCompetition = row.competition_index == null ? null : Number(row.competition_index)
        const cpc = row.cpc == null ? null : Number(row.cpc)
        const rejectionReasons: string[] = []
        if (!keyword) rejectionReasons.push('missing_keyword')
        if (searchVolume < 30) rejectionReasons.push('low_volume')
        if (rank < 4 || rank > 30) rejectionReasons.push('rank_outside_target')
        if (topicalScore < 50) rejectionReasons.push('low_topical_fit')
        if (articlePotentialScore < 45) rejectionReasons.push('weak_article_intent')
        if (NEGATIVE_TERMS.some((term) => keyword.includes(term))) rejectionReasons.push('negative_term')
        if (LOCAL_TERMS.some((term) => keyword.includes(term))) rejectionReasons.push('local_service_intent')
        if (BRAND_TERMS.some((term) => keyword.includes(term))) rejectionReasons.push('brand_or_nav')
        if (ECOMMERCE_TERMS.some((term) => keyword.includes(term)) && !ARTICLE_TERMS.some((term) => keyword.includes(term))) {
            rejectionReasons.push('ecommerce_intent')
        }
        const kdScore = kd == null ? 60 : Math.max(0, 100 - Math.min(100, kd * 2))
        const cpcScore = cpc == null ? 20 : Math.min(100, cpc * 20)
        const warehouseScore = Math.round(
            topicalScore * 0.25 +
            rankSignalScore * 0.2 +
            kdScore * 0.2 +
            volumeScore * 0.15 +
            cpcScore * 0.1 +
            articlePotentialScore * 0.1
        )
        return {
            keyword,
            normalizedKeyword: keyword.toLowerCase(),
            sourceUrl: String(row.url || ''),
            sourceDomain: params.sourceDomain,
            competitorRank: rank,
            searchVolume,
            keywordDifficulty: kd,
            cpc,
            paidCompetition,
            intent,
            topicalScore,
            articlePotentialScore,
            rankSignalScore,
            volumeScore,
            warehouseScore,
            rejected: rejectionReasons.length > 0,
            rejectionReasons,
        } satisfies WarehouseKeywordCandidate
    })
}

export function mergeOverviewMetrics(
    candidates: WarehouseKeywordCandidate[],
    overviewRows: Array<Record<string, unknown>>,
) {
    const metricMap = new Map<string, Record<string, unknown>>()
    for (const row of overviewRows) {
        const keyword = normalizeText(String(row.keyword || '')).toLowerCase()
        if (keyword) metricMap.set(keyword, row)
    }
    return candidates.map((candidate) => {
        const row = metricMap.get(candidate.normalizedKeyword)
        if (!row) return candidate
        const kd = row.keyword_difficulty == null ? candidate.keywordDifficulty : Number(row.keyword_difficulty)
        const cpc = row.cpc == null ? candidate.cpc : Number(row.cpc)
        const paidCompetition = row.competition_index == null ? candidate.paidCompetition : Number(row.competition_index)
        const searchVolume = row.search_volume == null ? candidate.searchVolume : Number(row.search_volume)
        const intent = String(row.intent || candidate.intent || '').toLowerCase()
        const kdScore = kd == null ? 60 : Math.max(0, 100 - Math.min(100, kd * 2))
        const cpcScore = cpc == null ? 20 : Math.min(100, cpc * 20)
        const volumeScore = keywordVolumeScore(searchVolume)
        const warehouseScore = Math.round(
            candidate.topicalScore * 0.25 +
            candidate.rankSignalScore * 0.2 +
            kdScore * 0.2 +
            volumeScore * 0.15 +
            cpcScore * 0.1 +
            candidate.articlePotentialScore * 0.1
        )
        return {
            ...candidate,
            keywordDifficulty: kd,
            cpc,
            paidCompetition,
            searchVolume,
            intent,
            volumeScore,
            warehouseScore,
        }
    })
}

function familyToken(word: string) {
    const normalized = word.replace(/[^a-z0-9]/g, '')
    if (!normalized) return ''
    const synonym = CLUSTER_SYNONYMS[normalized]
    if (synonym) return synonym
    if (normalized.endsWith('ies')) return `${normalized.slice(0, -3)}y`
    if (normalized.endsWith('s') && normalized.length > 4) return normalized.slice(0, -1)
    return normalized
}

function buildClusterKey(keyword: string) {
    const parts = tokenize(keyword)
        .map(familyToken)
        .filter((token) => token && !GENERIC_CLUSTER_STOPWORDS.has(token))
    return parts.slice(0, 4).sort().join('|')
}

export function clusterWarehouseKeywords(candidates: WarehouseKeywordCandidate[]) {
    const grouped = new Map<string, WarehouseKeywordCandidate[]>()
    for (const candidate of candidates.filter((item) => !item.rejected)) {
        const key = buildClusterKey(candidate.keyword) || candidate.normalizedKeyword
        const bucket = grouped.get(key) || []
        bucket.push(candidate)
        grouped.set(key, bucket)
    }

    const clusters: WarehouseCluster[] = []
    Array.from(grouped.entries()).forEach(([key, bucket], index) => {
        const sorted = [...bucket].sort((a, b) => b.warehouseScore - a.warehouseScore || b.searchVolume - a.searchVolume)
        const primary = sorted[0]
        const avgKd = sorted.some((row) => row.keywordDifficulty != null)
            ? Math.round(sorted.reduce((sum, row) => sum + (row.keywordDifficulty || 0), 0) / sorted.filter((row) => row.keywordDifficulty != null).length)
            : null
        const warehouseScore = Math.round(sorted.reduce((sum, row) => sum + row.warehouseScore, 0) / sorted.length)
        const volumePotential = sorted.reduce((sum, row) => sum + row.searchVolume, 0)
        const priority: WarehouseCluster['priority'] = warehouseScore >= 75 ? 'High' : warehouseScore >= 60 ? 'Medium' : 'Low'
        const status: WarehouseCluster['status'] = avgKd != null && avgKd <= 35 && warehouseScore >= 65
            ? 'Ready for SERP validation'
            : 'Future opportunity'
        clusters.push({
            id: `${key || 'cluster'}-${index + 1}`,
            clusterName: titleCase(primary.keyword),
            primaryKeyword: primary.keyword,
            supportingKeywords: sorted.slice(1, 8).map((row) => row.keyword),
            sourceUrls: Array.from(new Set(sorted.map((row) => row.sourceUrl).filter(Boolean))).slice(0, 12),
            avgKd,
            volumePotential,
            warehouseScore,
            priority,
            status,
        })
    })

    return clusters.sort((a, b) => b.warehouseScore - a.warehouseScore || b.volumePotential - a.volumePotential)
}
