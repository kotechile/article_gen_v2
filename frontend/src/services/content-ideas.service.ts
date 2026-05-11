import { apiClient } from '../api-client';
import { supabase } from '../lib/supabase';
import {
    type ContentIdea,
    type ContentIdeaGenerationRequest,
    type ContentIdeaGenerationResponse,
    type DFSKeywordRow,
    type DFSParsedOutput,
    type IdeaBurstResponse,
} from '../types/idea-burst';

type KeywordSelectionMetrics = {
    volume?: number | null;
    difficulty?: number | null;
    cpc?: number | null;
};

type KeywordMetricRecord = {
    keyword: string;
    search_volume: number | null;
    keyword_difficulty: number | null;
    cpc: number | null;
    metric_source?: string;
    is_estimated?: boolean;
    intent?: string | null;
};

function safeJsonParse<T>(value: unknown, fallback: T): T {
    if (value === null || value === undefined) return fallback;
    if (typeof value === 'object') return value as T;
    if (typeof value !== 'string') return fallback;
    try {
        return (JSON.parse(value) as T) ?? fallback;
    } catch {
        return fallback;
    }
}

function normalizeKeywordKey(input: string): string {
    return String(input || '').trim().toLowerCase();
}

function extractKeywordValuesLoose(value: unknown, depth = 0): string[] {
    if (depth > 5 || value === null || value === undefined) return [];

    if (Array.isArray(value)) {
        return value.flatMap((item) => extractKeywordValuesLoose(item, depth + 1));
    }

    if (typeof value === 'object') {
        const maybeKeyword = (value as any)?.keyword;
        if (typeof maybeKeyword === 'string') {
            return extractKeywordValuesLoose(maybeKeyword, depth + 1);
        }
        return [];
    }

    if (typeof value !== 'string') return [];

    const raw = value.trim();
    if (!raw) return [];

    let cleaned = raw;
    while (
        (cleaned.startsWith('[') && cleaned.endsWith(']')) ||
        (cleaned.startsWith('"') && cleaned.endsWith('"')) ||
        (cleaned.startsWith("'") && cleaned.endsWith("'"))
    ) {
        cleaned = cleaned.slice(1, -1).trim();
        if (!cleaned) break;
    }
    cleaned = cleaned.replace(/^[\[\]"']+/, '').replace(/[\[\]"']+$/, '').trim();
    if (!cleaned) return [];

    try {
        const parsed = JSON.parse(raw);
        if (typeof parsed === 'string' && parsed !== raw) {
            return extractKeywordValuesLoose(parsed, depth + 1);
        }
        if (Array.isArray(parsed)) {
            return parsed.flatMap((item) => extractKeywordValuesLoose(item, depth + 1));
        }
    } catch {
        // Fall through to manual parsing.
    }

    const parts = cleaned
        .split(',')
        .map((part) => part.trim().replace(/^[\[\]"']+|[\[\]"']+$/g, '').trim())
        .filter(Boolean);
    return parts.length > 1 ? parts : [cleaned.replace(/^[\[\]"']+|[\[\]"']+$/g, '').trim()];
}

function normalizeKeywordSelectionLoose(primaryKeyword: string, secondaryKeywords: string[]) {
    const primary = extractKeywordValuesLoose(primaryKeyword)[0] || '';
    const secondarySource = extractKeywordValuesLoose(secondaryKeywords);
    const secondary = Array.from(
        new Set(
            secondarySource
                .map((k) => String(k || '').trim())
                .filter(Boolean),
        ),
    ).filter((k) => k !== primary);
    const all = primary ? [primary, ...secondary] : secondary;
    return { primary, secondary, all };
}

function hasMetricValue(value: number | null | undefined): boolean {
    return typeof value === 'number' && Number.isFinite(value);
}

function hasAnyMetric(metric: Partial<KeywordMetricRecord> | null | undefined): boolean {
    if (!metric) return false;
    return hasMetricValue(metric.search_volume) || hasMetricValue(metric.keyword_difficulty) || hasMetricValue(metric.cpc);
}

function buildMetricRecord(keyword: string, metric: any, fallbackSource = 'manual_keyword_intelligence'): KeywordMetricRecord {
    const searchVolume = Number(metric?.search_volume);
    const keywordDifficulty = Number(metric?.keyword_difficulty);
    const cpc = Number(metric?.cpc);
    const metricRecord: KeywordMetricRecord = {
        keyword,
        search_volume: Number.isFinite(searchVolume) ? searchVolume : null,
        keyword_difficulty: Number.isFinite(keywordDifficulty) ? keywordDifficulty : null,
        cpc: Number.isFinite(cpc) ? cpc : null,
        metric_source: typeof metric?.metric_source === 'string' && metric.metric_source.trim()
            ? metric.metric_source.trim()
            : hasAnyMetric({
                search_volume: Number.isFinite(searchVolume) ? searchVolume : null,
                keyword_difficulty: Number.isFinite(keywordDifficulty) ? keywordDifficulty : null,
                cpc: Number.isFinite(cpc) ? cpc : null,
            })
                ? 'dataforseo_exact'
                : fallbackSource,
        is_estimated: typeof metric?.is_estimated === 'boolean'
            ? metric.is_estimated
            : !hasAnyMetric({
                search_volume: Number.isFinite(searchVolume) ? searchVolume : null,
                keyword_difficulty: Number.isFinite(keywordDifficulty) ? keywordDifficulty : null,
                cpc: Number.isFinite(cpc) ? cpc : null,
            }),
        intent: typeof metric?.intent === 'string' ? metric.intent : null,
    };
    return metricRecord;
}

function buildRow(
    kd: any,
    type: 'seed' | 'related',
    depth: number,
    relatedKeywords: string[] | null,
): DFSKeywordRow {
    const ki = kd?.keyword_info ?? {};
    const kp = kd?.keyword_properties ?? {};
    const si = kd?.search_intent_info ?? {};
    const monthly: any[] = Array.isArray(ki.monthly_searches) ? ki.monthly_searches : [];
    const sortedMonthly = [...monthly].sort((a, b) => {
        const da = a.year * 100 + a.month;
        const db = b.year * 100 + b.month;
        return da - db;
    });

    return {
        keyword: kd?.keyword ?? '',
        type,
        depth,
        search_volume: ki.search_volume ?? null,
        competition: ki.competition ?? null,
        competition_level: ki.competition_level ?? null,
        cpc: ki.cpc ?? null,
        keyword_difficulty: kp.keyword_difficulty ?? null,
        main_intent: si.main_intent ?? null,
        foreign_intents: Array.isArray(si.foreign_intent) ? si.foreign_intent : null,
        monthly_searches: sortedMonthly,
        search_volume_trend: ki.search_volume_trend ?? null,
        low_top_of_page_bid: ki.low_top_of_page_bid ?? null,
        high_top_of_page_bid: ki.high_top_of_page_bid ?? null,
        se_results_count: kd?.serp_info?.se_results_count ?? null,
        related_keywords: relatedKeywords,
    };
}

function parseDataForSEOOutput(raw: unknown): DFSParsedOutput | null {
    const parsed = safeJsonParse<any>(raw, null);
    if (!parsed || typeof parsed !== 'object') return null;

    if (parsed.rows && Array.isArray(parsed.rows)) {
        return parsed as DFSParsedOutput;
    }

    const tasks: any[] = Array.isArray(parsed.tasks) ? parsed.tasks : [];
    const task = tasks[0];
    if (!task) return null;

    const results: any[] = Array.isArray(task.result) ? task.result : [];
    const result = results[0];
    if (!result) return null;

    const rows: DFSKeywordRow[] = [];
    const seedData = result.seed_keyword_data;
    if (seedData) {
        rows.push(buildRow(seedData, 'seed', 0, null));
    }

    const items: any[] = Array.isArray(result.items) ? result.items : [];
    items.forEach((item: any) => {
        const kd = item?.keyword_data;
        if (!kd) return;
        if (kd.keyword === seedData?.keyword) return;
        rows.push(buildRow(kd, 'related', item.depth ?? 1, item.related_keywords ?? null));
    });

    return {
        seed_keyword: result.seed_keyword ?? seedData?.keyword ?? '',
        total_count: result.total_count ?? rows.length,
        items_count: result.items_count ?? rows.length,
        rows,
    };
}

function mergeRowMetrics(existingRow: any, incomingRow: any): any {
    const merged = { ...(existingRow || {}), ...(incomingRow || {}) };
    for (const field of ['search_volume', 'keyword_difficulty', 'cpc'] as const) {
        const incomingValue = incomingRow?.[field];
        const existingValue = existingRow?.[field];
        merged[field] = hasMetricValue(incomingValue)
            ? incomingValue
            : hasMetricValue(existingValue)
                ? existingValue
                : null;
    }
    if (!merged.metric_source) {
        merged.metric_source = hasAnyMetric(merged) ? 'dataforseo_exact' : 'manual_keyword_intelligence';
    }
    if (typeof merged.is_estimated !== 'boolean') {
        merged.is_estimated = !hasAnyMetric(merged);
    }
    return merged;
}

function parsedMetricMap(parsed: DFSParsedOutput | null): Map<string, KeywordMetricRecord> {
    const map = new Map<string, KeywordMetricRecord>();
    for (const row of parsed?.rows ?? []) {
        const keyword = String(row.keyword || '').trim();
        const key = normalizeKeywordKey(keyword);
        if (!keyword || !key) continue;
        map.set(key, buildMetricRecord(keyword, row, 'manual_keyword_intelligence'));
    }
    return map;
}

function selectedMetricMap(selected: unknown): Map<string, KeywordMetricRecord> {
    const map = new Map<string, KeywordMetricRecord>();
    const payload = safeJsonParse<any>(selected, {});
    const primaryKeyword = String(payload?.primary?.keyword || '').trim();
    if (primaryKeyword) {
        map.set(normalizeKeywordKey(primaryKeyword), buildMetricRecord(primaryKeyword, payload.primary, 'manual_keyword_intelligence'));
    }
    if (Array.isArray(payload?.secondary)) {
        payload.secondary.forEach((row: any) => {
            const keyword = String(row?.keyword || '').trim();
            if (!keyword) return;
            map.set(normalizeKeywordKey(keyword), buildMetricRecord(keyword, row, 'manual_keyword_intelligence'));
        });
    }
    return map;
}

function mergeMetricMaps(...maps: Array<Map<string, KeywordMetricRecord>>): Map<string, KeywordMetricRecord> {
    const merged = new Map<string, KeywordMetricRecord>();
    for (const map of maps) {
        map.forEach((metric, key) => {
            const existing = merged.get(key);
            if (!existing) {
                merged.set(key, metric);
                return;
            }
            merged.set(key, mergeRowMetrics(existing, metric));
        });
    }
    return merged;
}

function buildMergedRawOutput(
    existingRaw: unknown,
    incomingRaw: unknown,
    selectedKeywords: string[],
    metricMap: Map<string, KeywordMetricRecord>,
    seedKeyword: string,
): DFSParsedOutput | undefined {
    const existingParsed = parseDataForSEOOutput(existingRaw);
    const incomingParsed = parseDataForSEOOutput(incomingRaw);
    const rowMap = new Map<string, any>();

    const ingestRows = (rows: any[] | undefined) => {
        for (const row of rows ?? []) {
            const keyword = String(row?.keyword || '').trim();
            const key = normalizeKeywordKey(keyword);
            if (!keyword || !key) continue;
            rowMap.set(key, mergeRowMetrics(rowMap.get(key), { ...row, keyword }));
        }
    };

    ingestRows(existingParsed?.rows);
    ingestRows(incomingParsed?.rows);

    selectedKeywords.forEach((keyword, index) => {
        const key = normalizeKeywordKey(keyword);
        if (!key) return;
        const metric = metricMap.get(key);
        const existing = rowMap.get(key);
        rowMap.set(key, mergeRowMetrics(existing, {
            ...(existing || {}),
            keyword,
            type: existing?.type ?? (index === 0 ? 'seed' : 'related'),
            depth: existing?.depth ?? (index === 0 ? 0 : 1),
            related_keywords: existing?.related_keywords ?? null,
            search_volume: metric?.search_volume ?? existing?.search_volume ?? null,
            keyword_difficulty: metric?.keyword_difficulty ?? existing?.keyword_difficulty ?? null,
            cpc: metric?.cpc ?? existing?.cpc ?? null,
            metric_source: metric?.metric_source ?? existing?.metric_source,
            is_estimated: metric?.is_estimated ?? existing?.is_estimated,
            main_intent: existing?.main_intent ?? metric?.intent ?? null,
            monthly_searches: existing?.monthly_searches ?? [],
        }));
    });

    if (rowMap.size === 0) return undefined;

    const keywordOrder = [
        ...selectedKeywords,
        ...Array.from(rowMap.values()).map((row) => String(row.keyword || '').trim()),
    ];
    const seen = new Set<string>();
    const orderedRows = keywordOrder
        .map((keyword) => {
            const key = normalizeKeywordKey(keyword);
            if (!key || seen.has(key)) return null;
            seen.add(key);
            return rowMap.get(key) ?? null;
        })
        .filter(Boolean) as DFSKeywordRow[];

    return {
        seed_keyword: seedKeyword || incomingParsed?.seed_keyword || existingParsed?.seed_keyword || selectedKeywords[0] || '',
        total_count: orderedRows.length,
        items_count: orderedRows.length,
        rows: orderedRows,
    };
}

function buildSelectedKeywordMetricsJson(
    primary: string,
    secondary: string[],
    metricMap: Map<string, KeywordMetricRecord>,
    now: string,
): any {
    const primaryMetric = metricMap.get(normalizeKeywordKey(primary)) ?? buildMetricRecord(primary, {}, 'manual_keyword_intelligence');
    const secondaryMetrics = secondary.map((keyword) => {
        const metric = metricMap.get(normalizeKeywordKey(keyword)) ?? buildMetricRecord(keyword, {}, 'manual_keyword_intelligence');
        return {
            keyword,
            search_volume: metric.search_volume,
            keyword_difficulty: metric.keyword_difficulty,
            cpc: metric.cpc,
            metric_source: metric.metric_source ?? (hasAnyMetric(metric) ? 'dataforseo_exact' : 'manual_keyword_intelligence'),
            is_estimated: typeof metric.is_estimated === 'boolean' ? metric.is_estimated : !hasAnyMetric(metric),
        };
    });

    return {
        primary: {
            keyword: primary,
            search_volume: primaryMetric.search_volume,
            keyword_difficulty: primaryMetric.keyword_difficulty,
            cpc: primaryMetric.cpc,
            metric_source: primaryMetric.metric_source ?? (hasAnyMetric(primaryMetric) ? 'dataforseo_exact' : 'manual_keyword_intelligence'),
            is_estimated: typeof primaryMetric.is_estimated === 'boolean' ? primaryMetric.is_estimated : !hasAnyMetric(primaryMetric),
            intent: primaryMetric.intent ?? 'informational',
        },
        secondary: secondaryMetrics,
        candidate_count: [primary, ...secondary].filter(Boolean).length,
        source: hasAnyMetric(primaryMetric) ? 'dataforseo_exact' : 'manual_keyword_intelligence',
        generated_at: now,
    };
}

export function mergeKeywordSelectionState<T extends Record<string, any>>(
    prev: T,
    primaryKeyword: string,
    secondaryKeywords: string[],
    metrics?: KeywordSelectionMetrics,
    rawOutput?: any,
): T {
    const { primary, secondary, all } = normalizeKeywordSelectionLoose(primaryKeyword, secondaryKeywords);
    if (!primary) return prev;

    const existingRawMap = parsedMetricMap(parseDataForSEOOutput(prev.raw_dataforseo_output));
    const incomingRawMap = parsedMetricMap(parseDataForSEOOutput(rawOutput));
    const existingSelectedMap = selectedMetricMap(prev.selected_keyword_metrics_json);
    const directMetricMap = new Map<string, KeywordMetricRecord>();
    directMetricMap.set(
        normalizeKeywordKey(primary),
        buildMetricRecord(primary, {
            search_volume: metrics?.volume ?? null,
            keyword_difficulty: metrics?.difficulty ?? null,
            cpc: metrics?.cpc ?? null,
        }, 'manual_keyword_intelligence'),
    );
    const mergedMetricMap = mergeMetricMaps(existingRawMap, incomingRawMap, existingSelectedMap, directMetricMap);
    const mergedRawOutput = buildMergedRawOutput(prev.raw_dataforseo_output, rawOutput, all, mergedMetricMap, primary);
    const selectedKeywordMetricsJson = buildSelectedKeywordMetricsJson(primary, secondary, mergedMetricMap, new Date().toISOString());
    const primaryMetric = selectedKeywordMetricsJson.primary;

    return {
        ...prev,
        Keywords: all.join(', '),
        primary_keywords: [primary],
        secondary_keywords: secondary,
        search_phrase: primary,
        primary_keyword: primary,
        secondary_keywords_json: secondary,
        keyword_candidates_json: all,
        keyword_research_status: 'ready',
        keyword_research_source: selectedKeywordMetricsJson.source,
        keyword_selection_source: 'keyword_intelligence_modal',
        selected_keyword_search_volume: primaryMetric.search_volume ?? prev.selected_keyword_search_volume,
        selected_keyword_difficulty: primaryMetric.keyword_difficulty ?? prev.selected_keyword_difficulty,
        selected_keyword_metrics_json: selectedKeywordMetricsJson,
        raw_dataforseo_output: mergedRawOutput ?? prev.raw_dataforseo_output,
        keyword_research_generated_at: new Date().toISOString(),
        last_updated: new Date().toISOString(),
    };
}

class ContentIdeasService {
    private extractKeywordValues(value: unknown, depth = 0): string[] {
        if (depth > 5 || value === null || value === undefined) return [];

        if (Array.isArray(value)) {
            return value.flatMap((item) => this.extractKeywordValues(item, depth + 1));
        }

        if (typeof value === 'object') {
            const maybeKeyword = (value as any)?.keyword;
            if (typeof maybeKeyword === 'string') {
                return this.extractKeywordValues(maybeKeyword, depth + 1);
            }
            return [];
        }

        if (typeof value !== 'string') return [];

        const raw = value.trim();
        if (!raw) return [];

        // Heuristic: remove surrounding brackets/quotes, including malformed fragments like ["keyword
        let cleaned = raw;
        while ((cleaned.startsWith('[') && cleaned.endsWith(']')) || (cleaned.startsWith('"') && cleaned.endsWith('"')) || (cleaned.startsWith("'") && cleaned.endsWith("'"))) {
            cleaned = cleaned.slice(1, -1).trim();
            if (!cleaned) break;
        }
        cleaned = cleaned.replace(/^[\[\]"']+/, '').replace(/[\[\]"']+$/, '').trim();
        if (!cleaned) return [];

        try {
            const parsed = JSON.parse(raw);
            if (typeof parsed === 'string' && parsed !== raw) {
                return this.extractKeywordValues(parsed, depth + 1);
            }
            if (Array.isArray(parsed)) {
                return parsed.flatMap(item => this.extractKeywordValues(item, depth + 1));
            }
        } catch {
            // Continue with manual parsing
        }

        const parts = cleaned
            .split(',')
            .map((part) => part.trim().replace(/^[\[\]"']+|[\[\]"']+$/g, '').trim())
            .filter(Boolean);
        return parts.length > 1 ? parts : [cleaned.replace(/^[\[\]"']+|[\[\]"']+$/g, '').trim()];
    }

    private normalizeKeywordSelection(primaryKeyword: string, secondaryKeywords: string[]) {
        const primary = this.extractKeywordValues(primaryKeyword)[0] || '';
        const secondarySource = this.extractKeywordValues(secondaryKeywords);
        const secondary = Array.from(
            new Set(
                secondarySource
                    .map((k) => String(k || '').trim())
                    .filter(Boolean)
            )
        ).filter((k) => k !== primary);
        const all = primary ? [primary, ...secondary] : secondary;
        return { primary, secondary, all };
    }
    /**
     * Generate content ideas based on subtopics and keywords
     */
    async generateContentIdeas(request: ContentIdeaGenerationRequest): Promise<ContentIdeaGenerationResponse> {
        console.info('[ContentIdeas] generateContentIdeas request', {
            topic_id: request.topic_id,
            user_id: request.user_id,
            subtopics: request.subtopics?.length || 0,
        });
        return await apiClient.post<ContentIdeaGenerationResponse>('/content-ideas/generate', request);
    }

    /**
     * Generate Idea Burst for specific subtopic (New Flow)
     */
    async generateBurst(request: {
        topicId: string;
        subtopicName: string;
        keywords: Array<string | {
            keyword?: string;
            term?: string;
            search_volume?: number | null;
            keyword_difficulty?: number | null;
            difficulty?: number | null;
            cpc?: number | null;
        }>;
        affiliateOffers: string[];
        userId: string;
        intentBucket?: string | null;
        decisionFocus?: string | null;
        angleQuestion?: string | null;
        valueLayerTags?: string[] | null;
        clusterType?: string | null;
        primaryUserOutcome?: string | null;
        serpIntentMatch?: string | null;
        toolPotentialScore?: number | null;
    }): Promise<IdeaBurstResponse> {
        return await apiClient.post('/research-topics/idea-burst', {
            user_id: request.userId,
            topic_id: request.topicId,
            subtopic: request.subtopicName,
            keywords: request.keywords,
            affiliate_offers: request.affiliateOffers,
            intent_bucket: request.intentBucket,
            decision_focus: request.decisionFocus,
            angle_question: request.angleQuestion,
            value_layer_tags: request.valueLayerTags,
            cluster_type: request.clusterType,
            primary_user_outcome: request.primaryUserOutcome,
            serp_intent_match: request.serpIntentMatch,
            tool_potential_score: request.toolPotentialScore,
        });
    }

    /**
     * Get content ideas for a topic
     */
    async getContentIdeas(
        topicId: string,
        userId: string,
        contentType?: string
    ): Promise<ContentIdea[]> {
        try {
            console.info('[ContentIdeas] list request', { topicId, userId, contentType: contentType || 'all' });
            const data = await apiClient.post<ContentIdea[]>('/content-ideas/list', {
                topic_id: topicId,
                user_id: userId,
                content_type: contentType,
            });

            if (!Array.isArray(data)) {
                console.error('Content ideas API returned non-array:', data);
                return [];
            }
            console.info('[ContentIdeas] list response', {
                topicId,
                total: data.length,
                blog: data.filter((idea) => idea.content_type === 'blog').length,
                software: data.filter((idea) => idea.content_type === 'software').length,
            });
            return data || [];
        } catch (error) {
            console.error('Failed to get content ideas:', error);
            // Fallback to Supabase direct query
            return this.getContentIdeasFromSupabase(topicId, userId, contentType);
        }
    }

    /**
     * Fallback method to get content ideas directly from Supabase
     */
    private async getContentIdeasFromSupabase(
        topicId: string,
        userId: string,
        contentType?: string
    ): Promise<ContentIdea[]> {
        try {
            let query = supabase
                .from('content_ideas')
                .select('*')
                .eq('topic_id', topicId)
                .eq('user_id', userId);

            if (contentType) {
                query = query.eq('content_type', contentType);
            }

            const { data, error } = await query.order('created_at', { ascending: false });

            if (error) {
                console.error('Supabase query error:', error);
                return [];
            }

            return data as ContentIdea[] || [];
        } catch (error) {
            console.error('Failed to get content ideas from Supabase:', error);
            return [];
        }
    }

    /**
     * Delete a content idea
     */
    async deleteContentIdea(ideaId: string, userId: string): Promise<boolean> {
        try {
            console.info('[ContentIdeas] delete request', { ideaId, userId });
            await apiClient.delete(`/content-ideas/${ideaId}?user_id=${userId}`);
            console.info('[ContentIdeas] delete success', { ideaId });
            return true;
        } catch (error) {
            console.error('Failed to delete content idea:', error);
            return false;
        }
    }

    /**
     * Archive a content idea by switching status to "archived".
     */
    async archiveContentIdea(ideaId: string, userId: string): Promise<boolean> {
        return this.updateContentIdeaStatus(ideaId, userId, 'archived');
    }

    /**
     * Restore an archived content idea back to draft.
     */
    async restoreContentIdea(ideaId: string, userId: string): Promise<boolean> {
        return this.updateContentIdeaStatus(ideaId, userId, 'draft');
    }

    /**
     * Update status for a content idea.
     */
    async updateContentIdeaStatus(ideaId: string, userId: string, status: string): Promise<boolean> {
        try {
            const { error } = await supabase
                .from('content_ideas')
                .update({
                    status,
                    updated_at: new Date().toISOString(),
                })
                .eq('id', ideaId)
                .eq('user_id', userId);

            if (error) {
                console.error('[ContentIdeas] updateContentIdeaStatus error:', error);
                return false;
            }
            return true;
        } catch (err) {
            console.error('[ContentIdeas] updateContentIdeaStatus exception:', err);
            return false;
        }
    }

    /**
     * Persist user star rating for a content idea (0-5).
     */
    async updateContentIdeaRating(ideaId: string, userId: string, rating: number): Promise<boolean> {
        try {
            const nextRating = Math.max(0, Math.min(5, Number(rating || 0)));
            const { error } = await supabase
                .from('content_ideas')
                .update({
                    topic_rating: nextRating,
                    updated_at: new Date().toISOString(),
                })
                .eq('id', ideaId)
                .eq('user_id', userId);

            if (error) {
                console.error('[ContentIdeas] updateContentIdeaRating error:', error);
                return false;
            }
            return true;
        } catch (err) {
            console.error('[ContentIdeas] updateContentIdeaRating exception:', err);
            return false;
        }
    }

    /**
     * Get content ideas grouped by type
     */
    async getContentIdeasGrouped(
        topicId: string,
        userId: string
    ): Promise<{ blog: ContentIdea[]; software: ContentIdea[] }> {
        const allIdeas = await this.getContentIdeas(topicId, userId);

        return {
            blog: allIdeas.filter(idea => idea.content_type === 'blog'),
            software: allIdeas.filter(idea => idea.content_type === 'software'),
        };
    }

    /**
     * Publish content ideas to Titles
     */
    async publishContentIdeas(ideaIds: string[], userId: string): Promise<{
        success: boolean;
        publishedCount: number;
        publishedToTitlesCount: number;
        publishedToSoftwareCount: number;
        requestedCount: number;
        message?: string;
    }> {
        try {
            console.info('[ContentIdeas] publish request', {
                userId,
                ideaCount: ideaIds.length,
                ideaIds,
            });
            const result = await apiClient.post<any>('/content-ideas/publish', {
                idea_ids: ideaIds,
                user_id: userId
            });
            const normalized = {
                success: Boolean(result?.success),
                publishedCount: Number(result?.published_count || 0),
                publishedToTitlesCount: Number(result?.published_to_titles_count || 0),
                publishedToSoftwareCount: Number(result?.published_to_software_count || 0),
                requestedCount: Number(result?.requested_count || ideaIds.length),
                message: result?.message,
            };
            console.info('[ContentIdeas] publish success', { userId, ...normalized });
            return normalized;
        } catch (error) {
            console.error('Failed to publish content ideas:', error);
            return {
                success: false,
                publishedCount: 0,
                publishedToTitlesCount: 0,
                publishedToSoftwareCount: 0,
                requestedCount: ideaIds.length,
                message: 'Request failed',
            };
        }
    }

    /**
     * Enrich selected ideas with SEO metrics and affiliate offer signals.
     */
    async enrichContentIdeas(ideaIds: string[], userId: string): Promise<{
        success: boolean;
        requestedCount: number;
        enrichedCount: number;
        results: Array<{
            idea_id: string;
            status: 'enriched' | 'failed';
            reason?: string;
            metrics?: {
                total_search_volume: number;
                average_cpc: number;
                average_difficulty: number;
                affiliate_offer_count: number;
            };
            keywords_used?: string[];
            selected_primary_keyword?: string;
            restated_title?: string;
            title_restated?: boolean;
            keyword_metrics_map?: Record<string, {
                search_volume?: number;
                keyword_difficulty?: number;
                cpc?: number;
            }>;
            affiliate_offers_preview?: Array<{
                name?: string | null;
                network?: string | null;
                commission_rate?: string | null;
            }>;
            affiliate_search_status?: string | null;
            affiliate_search_error?: string | null;
        }>;
    }> {
        try {
            const result = await apiClient.post<any>('/content-ideas/enrich', {
                idea_ids: ideaIds,
                user_id: userId,
            });

            return {
                success: Boolean(result?.success),
                requestedCount: Number(result?.requested_count || ideaIds.length),
                enrichedCount: Number(result?.enriched_count || 0),
                results: Array.isArray(result?.results) ? result.results : [],
            };
        } catch (error) {
            console.error('Failed to enrich content ideas:', error);
            return {
                success: false,
                requestedCount: ideaIds.length,
                enrichedCount: 0,
                results: [],
            };
        }
    }

    /**
     * Persist primary and secondary keyword selections for a content idea.
     * Writes directly to Supabase so the update is instant and doesn't
     * require a Flask API round-trip.
     */
    async updateKeywordSelection(
        ideaId: string,
        userId: string,
        primaryKeyword: string,
        secondaryKeywords: string[],
        metrics?: {
            volume?: number | null;
            difficulty?: number | null;
            cpc?: number | null;
        },
        rawOutput?: any
    ): Promise<boolean> {
        try {
            const { primary, secondary, all } = this.normalizeKeywordSelection(primaryKeyword, secondaryKeywords);
            if (!primary) return false;
            const now = new Date().toISOString();

            console.log(`[ContentIdeasService] updateKeywordSelection for idea: ${ideaId}`, {
                primaryKeyword,
                secondaryCount: secondaryKeywords.length,
                hasRawOutput: !!rawOutput,
                rawOutputRows: rawOutput?.rows?.length
            });

            const { error: clearError } = await supabase
                .from('content_ideas')
                .update({
                    primary_keywords: [],
                    secondary_keywords: [],
                    keywords: [],
                    search_phrase: null,
                    updated_at: now,
                })
                .eq('id', ideaId)
                .eq('user_id', userId);

            if (clearError) {
                console.error('[ContentIdeasService] updateKeywordSelection clear step error:', clearError);
                return false;
            }

            const { error } = await supabase
                .from('content_ideas')
                .update({
                    // Force replacement semantics before setting the new values.
                    primary_keywords: [primary],
                    secondary_keywords: secondary,
                    keywords: all,
                    search_phrase: primary,
                    total_search_volume: metrics?.volume ?? undefined,
                    average_difficulty: metrics?.difficulty ?? undefined,
                    average_cpc: metrics?.cpc ?? undefined,
                    raw_dataforseo_output: rawOutput ?? undefined,
                    updated_at: now,
                })
                .eq('id', ideaId)
                .eq('user_id', userId);

            if (error) {
                console.error('[ContentIdeasService] updateKeywordSelection error:', error);
                return false;
            }
            console.log('[ContentIdeasService] updateKeywordSelection success');
            return true;
        } catch (err) {
            console.error('[ContentIdeasService] updateKeywordSelection exception:', err);
            return false;
        }
    }

    /**
     * Fetch related keywords for a custom seed keyword using the
     * /keyword-lab/related backend endpoint (DataForSEO Labs live).
     */
    async fetchRelatedKeywords(
        seedKeyword: string,
        excludeKeywords?: string[],
        limit = 20,
        filters?: {
            minSearchVolume?: number;
            maxKeywordDifficulty?: number;
        },
    ): Promise<{
        success: boolean;
        seed_keyword: string;
        keywords: Array<{
            keyword: string;
            search_volume: number;
            keyword_difficulty: number;
            cpc: number;
            opportunity: number;
            intent_label?: string | null;
            competition_level?: string | null;
        }>;
    }> {
        try {
            const result = await apiClient.post<any>('/content-ideas/keyword-lab/related', {
                seed_keyword: seedKeyword,
                exclude_keywords: excludeKeywords ?? [],
                limit,
                min_search_volume: filters?.minSearchVolume,
                max_keyword_difficulty: filters?.maxKeywordDifficulty,
            });
            return {
                success: Boolean(result?.success),
                seed_keyword: result?.seed_keyword ?? seedKeyword,
                keywords: Array.isArray(result?.keywords) ? result.keywords : [],
            };
        } catch (err) {
            console.error('[ContentIdeas] fetchRelatedKeywords error:', err);
            return { success: false, seed_keyword: seedKeyword, keywords: [] };
        }
    }
    /**
     * Persist keyword selections for a Titles record (Content Library).
     * Writes directly to the Titles table in Supabase.
     * Mirrors updateKeywordSelection but operates on the Titles table.
     */
    async updateTitleKeywordSelection(
        titleId: string,
        userId: string,
        primaryKeyword: string,
        secondaryKeywords: string[],
        metrics?: KeywordSelectionMetrics,
        rawOutput?: any
    ): Promise<boolean> {
        try {
            const { primary, secondary, all } = this.normalizeKeywordSelection(primaryKeyword, secondaryKeywords);
            if (!primary) return false;
            const now = new Date().toISOString();

            console.log(`[ContentIdeasService] updateTitleKeywordSelection for title: ${titleId}`, {
                primaryKeyword,
                secondaryCount: secondaryKeywords.length,
                hasRawOutput: !!rawOutput,
                rawOutputRows: rawOutput?.rows?.length
            });

            const { data: existingTitle, error: fetchError } = await supabase
                .from('Titles')
                .select('raw_dataforseo_output, selected_keyword_metrics_json, selected_keyword_search_volume, selected_keyword_difficulty, primary_keywords, secondary_keywords, secondary_keywords_json, Keywords, search_phrase, primary_keyword, keyword_candidates_json')
                .eq('id', titleId)
                .eq('user_id', userId)
                .maybeSingle();

            if (fetchError) {
                console.error('[ContentIdeasService] updateTitleKeywordSelection fetch step error:', fetchError);
                return false;
            }

            const existingRawMap = parsedMetricMap(parseDataForSEOOutput(existingTitle?.raw_dataforseo_output));
            const incomingRawMap = parsedMetricMap(parseDataForSEOOutput(rawOutput));
            const existingSelectedMap = selectedMetricMap(existingTitle?.selected_keyword_metrics_json);
            const directMetricMap = new Map<string, KeywordMetricRecord>();
            directMetricMap.set(
                normalizeKeywordKey(primary),
                buildMetricRecord(primary, {
                    search_volume: metrics?.volume ?? null,
                    keyword_difficulty: metrics?.difficulty ?? null,
                    cpc: metrics?.cpc ?? null,
                }, 'manual_keyword_intelligence'),
            );

            const mergedMetricMap = mergeMetricMaps(existingRawMap, incomingRawMap, existingSelectedMap, directMetricMap);
            const mergedRawOutput = buildMergedRawOutput(existingTitle?.raw_dataforseo_output, rawOutput, all, mergedMetricMap, primary);
            const selectedKeywordMetricsJson = buildSelectedKeywordMetricsJson(primary, secondary, mergedMetricMap, now);
            const primaryMetric = selectedKeywordMetricsJson.primary;

            const { error } = await supabase
                .from('Titles')
                .update({
                    Keywords: all.join(', '),
                    primary_keywords: [primary],
                    secondary_keywords: secondary,
                    search_phrase: primary,
                    primary_keyword: primary,
                    secondary_keywords_json: secondary,
                    keyword_candidates_json: all,
                    keyword_research_status: 'ready',
                    keyword_research_source: selectedKeywordMetricsJson.source,
                    keyword_selection_source: 'keyword_intelligence_modal',
                    selected_keyword_search_volume: primaryMetric.search_volume ?? existingTitle?.selected_keyword_search_volume ?? undefined,
                    selected_keyword_difficulty: primaryMetric.keyword_difficulty ?? existingTitle?.selected_keyword_difficulty ?? undefined,
                    selected_keyword_metrics_json: selectedKeywordMetricsJson,
                    raw_dataforseo_output: mergedRawOutput ?? existingTitle?.raw_dataforseo_output ?? undefined,
                    keyword_research_generated_at: now,
                    last_updated: now,
                })
                .eq('id', titleId)
                .eq('user_id', userId);

            if (error) {
                console.error('[ContentIdeasService] updateTitleKeywordSelection error:', error);
                return false;
            }
            console.log('[ContentIdeasService] updateTitleKeywordSelection success');
            return true;
        } catch (err) {
            console.error('[ContentIdeasService] updateTitleKeywordSelection exception:', err);
            return false;
        }
    }
}

export const contentIdeasService = new ContentIdeasService();
