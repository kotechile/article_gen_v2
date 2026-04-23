import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Sparkles, Lightbulb, Loader2, Check, Save, BookOpen, Code, Info, BarChart3, ChevronDown, ChevronUp, Key } from "lucide-react";
import { KeywordIntelligenceModal } from "./KeywordIntelligenceModal";
import { Button } from "@/components/ui/button";
import { contentIdeasService } from "@/services/content-ideas.service";
import type { ContentIdea } from "@/types/idea-burst";
import type { Subtopic } from "@/types/research";
import { useAuth } from "@/context/auth-context";

interface IdeaBurstModalProps {
    isOpen: boolean;
    onClose: () => void;
    subtopic: Subtopic | null;
    topicId: string;
    topicTitle: string;
    projectName?: string | null;
    categoryPath?: string | null;
}

interface KeywordMetricRow {
    keyword: string;
    search_volume: number | null;
    keyword_difficulty: number | null;
    cpc: number | null;
}

function normalizeKeywordKey(input: string): string {
    return (input || "")
        .trim()
        .toLowerCase()
        .replace(/&/g, " and ")
        .replace(/[^a-z0-9]+/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}

function coerceKeywordList(value: unknown): string[] {
    if (Array.isArray(value)) {
        return value.map((v) => String(v || "").trim()).filter(Boolean);
    }
    if (typeof value === "string") {
        const raw = value.trim();
        if (!raw) return [];
        try {
            const parsed = JSON.parse(raw);
            if (Array.isArray(parsed)) {
                return parsed.map((v) => String(v || "").trim()).filter(Boolean);
            }
        } catch {
            // Fall through to delimited parsing.
        }
        return raw
            .replace(/^\{|\}$/g, "")
            .split(/[\n,]+/)
            .map((v) => v.trim())
            .filter(Boolean);
    }
    return [];
}

function parseJsonLike<T>(value: unknown, fallback: T): T {
    if (value === null || value === undefined) return fallback;
    if (typeof value === "string") {
        const raw = value.trim();
        if (!raw) return fallback;
        try {
            const parsed = JSON.parse(raw);
            return (parsed as T) ?? fallback;
        } catch {
            return fallback;
        }
    }
    return value as T;
}

function extractIdeaKeywordMetricsMap(idea: ContentIdea): Map<string, KeywordMetricRow> {
    const map = new Map<string, KeywordMetricRow>();
    const ideaMetadata = parseJsonLike<Record<string, unknown>>((idea as any).idea_metadata, {});
    const fromColumn = parseJsonLike<unknown>((idea as any).keyword_metrics, {});
    const fromMetadata = parseJsonLike<unknown>((ideaMetadata as any)?.seo_offer_enrichment?.keyword_metrics, {});
    const rankedCandidates = parseJsonLike<unknown>((ideaMetadata as any)?.seo_offer_enrichment?.keyword_ranked_candidates, []);
    const pass2Candidates = parseJsonLike<unknown>((ideaMetadata as any)?.keyword_pass_2?.keyword_ranked_candidates, []);
    const rawDfsOutput = parseJsonLike<unknown>(
        (idea as any).raw_dataforseo_output ?? (idea as any).raw_supabase_output ?? (ideaMetadata as any)?.seo_offer_enrichment?.raw_dataforseo_output,
        {},
    );

    const ingestMetric = (keywordInput: unknown, metricInput: unknown) => {
        const keyword = String(keywordInput || "").trim();
        if (!keyword) return;
        const metric = parseJsonLike<Record<string, unknown>>(metricInput, {});
        const searchVolumeRaw = metric.search_volume;
        const keywordDifficultyRaw = metric.keyword_difficulty;
        const cpcRaw = metric.cpc;
        const searchVolume =
            typeof searchVolumeRaw === "number"
                ? searchVolumeRaw
                : (typeof searchVolumeRaw === "string" && searchVolumeRaw.trim() ? Number(searchVolumeRaw) : null);
        const keywordDifficulty =
            typeof keywordDifficultyRaw === "number"
                ? keywordDifficultyRaw
                : (typeof keywordDifficultyRaw === "string" && keywordDifficultyRaw.trim() ? Number(keywordDifficultyRaw) : null);
        const cpc =
            typeof cpcRaw === "number"
                ? cpcRaw
                : (typeof cpcRaw === "string" && cpcRaw.trim() ? Number(cpcRaw) : null);

        map.set(normalizeKeywordKey(keyword), {
            keyword,
            search_volume: Number.isFinite(searchVolume as number) ? (searchVolume as number) : null,
            keyword_difficulty: Number.isFinite(keywordDifficulty as number) ? (keywordDifficulty as number) : null,
            cpc: Number.isFinite(cpc as number) ? (cpc as number) : null,
        });
    };

    const ingestSource = (source: unknown) => {
        const parsedSource = parseJsonLike<unknown>(source, {});
        if (Array.isArray(parsedSource)) {
            parsedSource.forEach((row) => {
                if (!row || typeof row !== "object") return;
                ingestMetric((row as any).keyword || (row as any).term, row);
            });
            return;
        }
        if (parsedSource && typeof parsedSource === "object") {
            Object.entries(parsedSource as Record<string, unknown>).forEach(([rawKeyword, rawMetric]) => {
                ingestMetric(rawKeyword, rawMetric);
            });
        }
    };

    const ingestRawDataforSeo = (source: unknown) => {
        const parsed = parseJsonLike<any>(source, {});
        const tasks = Array.isArray(parsed?.tasks) ? parsed.tasks : [];
        tasks.forEach((task: any) => {
            const results = Array.isArray(task?.result) ? task.result : [];
            results.forEach((result: any) => {
                const seedKeywordData = result?.seed_keyword_data;
                if (seedKeywordData && typeof seedKeywordData === "object" && !Array.isArray(seedKeywordData)) {
                    const keyword = String(seedKeywordData?.keyword || "").trim();
                    if (keyword) {
                        ingestMetric(keyword, {
                            search_volume: seedKeywordData?.keyword_info?.search_volume,
                            keyword_difficulty: seedKeywordData?.keyword_properties?.keyword_difficulty,
                            cpc: seedKeywordData?.keyword_info?.cpc,
                        });
                    }
                }
                const items = Array.isArray(result?.items) ? result.items : [];
                items.forEach((item: any) => {
                    const keywordData = item?.keyword_data;
                    const keyword = String(keywordData?.keyword || "").trim();
                    if (!keyword) return;
                    ingestMetric(keyword, {
                        search_volume: keywordData?.keyword_info?.search_volume,
                        keyword_difficulty: keywordData?.keyword_properties?.keyword_difficulty,
                        cpc: keywordData?.keyword_info?.cpc,
                    });
                });
            });
        });
    };

    ingestSource(fromMetadata);
    ingestSource(fromColumn);
    ingestSource(rankedCandidates);
    ingestSource(pass2Candidates);
    ingestRawDataforSeo(rawDfsOutput);
    return map;
}

function extractKeywordMetricsMapFromUnknown(value: unknown): Map<string, KeywordMetricRow> {
    const map = new Map<string, KeywordMetricRow>();
    const parsed = parseJsonLike<unknown>(value, []);

    const ingestMetric = (keywordInput: unknown, metricInput: unknown) => {
        const keyword = String(keywordInput || "").trim();
        if (!keyword) return;
        const metric = parseJsonLike<Record<string, unknown>>(metricInput, {});
        const searchVolumeRaw = metric.search_volume ?? metric.volume;
        const keywordDifficultyRaw = metric.keyword_difficulty ?? metric.difficulty ?? metric.seo_difficulty;
        const cpcRaw = metric.cpc;
        const searchVolume =
            typeof searchVolumeRaw === "number"
                ? searchVolumeRaw
                : (typeof searchVolumeRaw === "string" && searchVolumeRaw.trim() ? Number(searchVolumeRaw) : null);
        const keywordDifficulty =
            typeof keywordDifficultyRaw === "number"
                ? keywordDifficultyRaw
                : (typeof keywordDifficultyRaw === "string" && keywordDifficultyRaw.trim() ? Number(keywordDifficultyRaw) : null);
        const cpc =
            typeof cpcRaw === "number"
                ? cpcRaw
                : (typeof cpcRaw === "string" && cpcRaw.trim() ? Number(cpcRaw) : null);

        map.set(normalizeKeywordKey(keyword), {
            keyword,
            search_volume: Number.isFinite(searchVolume as number) ? (searchVolume as number) : null,
            keyword_difficulty: Number.isFinite(keywordDifficulty as number) ? (keywordDifficulty as number) : null,
            cpc: Number.isFinite(cpc as number) ? (cpc as number) : null,
        });
    };

    if (Array.isArray(parsed)) {
        parsed.forEach((row) => {
            if (!row) return;
            if (typeof row === "string") {
                ingestMetric(row, { keyword: row });
                return;
            }
            if (typeof row === "object") {
                ingestMetric((row as any).keyword || (row as any).term || (row as any).seed_keyword, row);
            }
        });
        return map;
    }

    if (parsed && typeof parsed === "object") {
        Object.entries(parsed as Record<string, unknown>).forEach(([rawKeyword, rawMetric]) => {
            ingestMetric(rawKeyword, rawMetric);
        });
    }

    return map;
}

function resolveKeywordMetricRow(
    keyword: string,
    ideaMap: Map<string, KeywordMetricRow>,
    subtopicMap: Map<string, KeywordMetricRow>,
): KeywordMetricRow | undefined {
    const key = normalizeKeywordKey(keyword);
    const direct = ideaMap.get(key) || subtopicMap.get(key);
    if (direct) return direct;

    // Fuzzy fallback for punctuation/phrase variants (e.g., "s&p" vs "s p").
    const findFuzzy = (source: Map<string, KeywordMetricRow>) => {
        if (!key || key.length < 4) return undefined;
        for (const [candidateKey, row] of source.entries()) {
            if (!candidateKey) continue;
            if (candidateKey === key) return row;
            if (candidateKey.includes(key) || key.includes(candidateKey)) return row;
        }
        return undefined;
    };

    return findFuzzy(ideaMap) || findFuzzy(subtopicMap);
}

function resolveIdeaKeywords(idea: ContentIdea): string[] {
    const metadata = parseJsonLike<Record<string, any>>((idea as any).idea_metadata, {});
    const metadataKeywords = coerceKeywordList(metadata?.seo_offer_enrichment?.keywords_used);
    const metadataInputKeywords = coerceKeywordList(metadata?.input_keywords);
    const seedPackKeywords = coerceKeywordList(metadata?.keyword_seed_pack?.input_keywords);
    const ideaKeywords = coerceKeywordList((idea as any).keywords);
    const primaryKeywords = coerceKeywordList((idea as any).primary_keywords);
    const secondaryKeywords = coerceKeywordList((idea as any).secondary_keywords);
    const rankedCandidateKeywords = parseJsonLike<any[]>(metadata?.seo_offer_enrichment?.keyword_ranked_candidates, [])
        .map((row) => String(row?.keyword || "").trim())
        .filter(Boolean);
    const pass2CandidateKeywords = parseJsonLike<any[]>(metadata?.keyword_pass_2?.keyword_ranked_candidates, [])
        .map((row) => String(row?.keyword || "").trim())
        .filter(Boolean);
    const metricMapKeywords = Array.from(extractIdeaKeywordMetricsMap(idea).values()).map((row) => row.keyword);

    const merged = [
        ...metricMapKeywords,
        ...ideaKeywords,
        ...primaryKeywords,
        ...secondaryKeywords,
        ...metadataKeywords,
        ...metadataInputKeywords,
        ...seedPackKeywords,
        ...rankedCandidateKeywords,
        ...pass2CandidateKeywords,
    ];
    const deduped: string[] = [];
    const seen = new Set<string>();
    merged.forEach((kw) => {
        const norm = normalizeKeywordKey(kw);
        if (!norm || seen.has(norm)) return;
        seen.add(norm);
        deduped.push(kw);
    });
    return deduped;
}

function ideaHasExactKeywordMetrics(idea: ContentIdea): boolean {
    const map = extractIdeaKeywordMetricsMap(idea);
    for (const row of map.values()) {
        if ((row.search_volume || 0) > 0 || (row.keyword_difficulty || 0) > 0 || (row.cpc || 0) > 0) {
            return true;
        }
    }
    return false;
}

function getRawDataforSeoTrace(idea: ContentIdea): Record<string, unknown> | null {
    const direct = parseJsonLike<Record<string, unknown> | null>(
        (idea as any).raw_dataforseo_output ?? (idea as any).raw_supabase_output,
        null,
    );
    if (direct && typeof direct === "object" && Object.keys(direct).length > 0) {
        return direct;
    }
    const metadata = parseJsonLike<Record<string, any>>((idea as any).idea_metadata, {});
    const nested = parseJsonLike<Record<string, unknown> | null>(metadata?.seo_offer_enrichment?.raw_dataforseo_output, null);
    if (nested && typeof nested === "object" && Object.keys(nested).length > 0) {
        return nested;
    }
    return null;
}

function computeAggregateFromExactMap(
    keywords: string[],
    ideaMap: Map<string, KeywordMetricRow>,
    subtopicMap: Map<string, KeywordMetricRow>,
) {
    const rows = keywords
        .map((kw) => resolveKeywordMetricRow(kw, ideaMap, subtopicMap))
        .filter((row): row is KeywordMetricRow => Boolean(row));
    const totalVolume = rows.reduce((sum, row) => sum + Math.max(0, Number(row.search_volume || 0)), 0);
    const kdValues = rows.map((row) => Number(row.keyword_difficulty || 0)).filter((v) => v > 0);
    const cpcValues = rows.map((row) => Number(row.cpc || 0)).filter((v) => v > 0);
    return {
        totalVolume,
        avgDifficulty: kdValues.length ? kdValues.reduce((s, v) => s + v, 0) / kdValues.length : 0,
        avgCpc: cpcValues.length ? cpcValues.reduce((s, v) => s + v, 0) / cpcValues.length : 0,
    };
}

interface CachedIdeaBurst {
    blogIdeas: ContentIdea[];
    softwareIdeas: ContentIdea[];
    cachedAt: string;
}

function intentChipClass(intent?: string) {
    const value = (intent || "").toLowerCase();
    if (value.includes("transactional")) return "bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
    if (value.includes("commercial")) return "bg-amber-500/20 text-amber-300 border-amber-500/30";
    return "bg-blue-500/20 text-blue-300 border-blue-500/30";
}

function complexityChipClass(level?: string) {
    const value = (level || "").toLowerCase();
    if (value.includes("high")) return "bg-red-500/20 text-red-300 border-red-500/30";
    if (value.includes("medium")) return "bg-yellow-500/20 text-yellow-300 border-yellow-500/30";
    return "bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
}

function getRankFactors(idea: ContentIdea): Array<{ label: string; value: number }> {
    const breakdown = idea.ranking_breakdown;
    if (!breakdown) return [];
    const entries: Array<{ key: keyof NonNullable<ContentIdea["ranking_breakdown"]>; label: string }> = [
        { key: "search_opportunity", label: "Search" },
        { key: "intent_match", label: "Intent" },
        { key: "serp_intent_match", label: "SERP Fit" },
        { key: "viability", label: "Viability" },
        { key: "seo_ease", label: "SEO Ease" },
        { key: "tool_potential", label: "Tool Potential" },
        { key: "build_complexity_score", label: "Build Ease" },
    ];

    return entries
        .map(({ key, label }) => ({
            label,
            value: Number(breakdown[key] || 0),
        }))
        .filter((item) => item.value > 0);
}

function buildInternalLinkGroups(ideas: ContentIdea[]): Array<{ hook: string; count: number }> {
    const groups = new Map<string, number>();
    ideas.forEach((idea) => {
        const hook = (idea.internal_link_hook || "").trim();
        if (!hook) return;
        groups.set(hook, (groups.get(hook) || 0) + 1);
    });
    return Array.from(groups.entries())
        .map(([hook, count]) => ({ hook, count }))
        .sort((a, b) => b.count - a.count);
}

function isSentToContentLibrary(idea: ContentIdea): boolean {
    return Boolean(
        idea.published_to_titles ||
        idea.titles_record_id ||
        idea.published ||
        idea.status?.toLowerCase() === "published"
    );
}

export function IdeaBurstModal({ isOpen, onClose, subtopic, topicId, topicTitle, projectName, categoryPath }: IdeaBurstModalProps) {
    const { user } = useAuth();
    const ENABLE_IDEA_BURST_CACHE = false;
    const [loading, setLoading] = React.useState(false);
    const [error, setError] = React.useState<string | null>(null);
    const [blogIdeas, setBlogIdeas] = React.useState<ContentIdea[]>([]);
    const [softwareIdeas, setSoftwareIdeas] = React.useState<ContentIdea[]>([]);
    const [selectedBlogIdeas, setSelectedBlogIdeas] = React.useState<Set<string>>(new Set());
    const [selectedSoftwareIdeas, setSelectedSoftwareIdeas] = React.useState<Set<string>>(new Set());
    const [publishing, setPublishing] = React.useState(false);
    const [savingSoftware, setSavingSoftware] = React.useState(false);
    const [enrichingIdeas, setEnrichingIdeas] = React.useState(false);
    const [enrichResultMessage, setEnrichResultMessage] = React.useState<string | null>(null);
    const [published, setPublished] = React.useState(false);
    const [saved, setSaved] = React.useState(false);
    const [expandedMetrics, setExpandedMetrics] = React.useState<string | null>(null);
    const [loadedFromCache, setLoadedFromCache] = React.useState(false);
    const [loadedFromStored, setLoadedFromStored] = React.useState(false);
    const lastGeneratedKeyRef = React.useRef<string | null>(null);
    const autoEnrichAttemptedRef = React.useRef<Set<string>>(new Set());

    const cacheKey = React.useMemo(() => {
        if (!subtopic || !user) return null;
        return `ideaBurstCache:${topicId}:${subtopic.id}:${user.id}`;
    }, [topicId, subtopic?.id, user?.id]);

    React.useEffect(() => {
        if (!isOpen || !subtopic || !user) return;
        const generationKey = `${topicId}:${subtopic.id}:${user.id}`;
        if (lastGeneratedKeyRef.current === generationKey) return;

        let cancelled = false;
        const hydrateIdeas = async () => {
            // 1) Prefer persisted ideas for this topic/subtopic.
            try {
                const storedIdeas = await contentIdeasService.getContentIdeas(topicId, user.id);
                const normalizedSubtopic = (subtopic.name || "").trim().toLowerCase();
                const filtered = (storedIdeas || []).filter(
                    (idea) => (idea.subtopic || "").trim().toLowerCase() === normalizedSubtopic
                );
                if (!cancelled && filtered.length > 0) {
                    const nextBlogIdeas = filtered.filter((idea) => idea.content_type === "blog");
                    const nextSoftwareIdeas = filtered.filter((idea) => idea.content_type === "software");
                    setBlogIdeas(nextBlogIdeas);
                    setSoftwareIdeas(nextSoftwareIdeas);
                    setError(null);
                    setLoadedFromStored(true);
                    setLoadedFromCache(false);

                    if (ENABLE_IDEA_BURST_CACHE && cacheKey) {
                        try {
                            const payload: CachedIdeaBurst = {
                                blogIdeas: nextBlogIdeas,
                                softwareIdeas: nextSoftwareIdeas,
                                cachedAt: new Date().toISOString(),
                            };
                            localStorage.setItem(cacheKey, JSON.stringify(payload));
                        } catch (e) {
                            console.warn("Failed to persist idea burst cache:", e);
                        }
                    }
                    lastGeneratedKeyRef.current = generationKey;
                    autoEnrichIdeasWithoutKeywordMetrics([...nextBlogIdeas, ...nextSoftwareIdeas]);
                    return;
                }
            } catch (e) {
                console.warn("Failed to load stored ideas for subtopic:", e);
            }

            // 2) Fall back to local cache.
            if (ENABLE_IDEA_BURST_CACHE && cacheKey) {
                try {
                    const raw = localStorage.getItem(cacheKey);
                    if (raw) {
                        const parsed = JSON.parse(raw) as CachedIdeaBurst;
                        if (Array.isArray(parsed.blogIdeas) || Array.isArray(parsed.softwareIdeas)) {
                            const cachedBlogIdeas = Array.isArray(parsed.blogIdeas) ? parsed.blogIdeas : [];
                            const cachedSoftwareIdeas = Array.isArray(parsed.softwareIdeas) ? parsed.softwareIdeas : [];
                            setBlogIdeas(cachedBlogIdeas);
                            setSoftwareIdeas(cachedSoftwareIdeas);
                            setError(null);
                            setLoadedFromCache(true);
                            setLoadedFromStored(false);
                            lastGeneratedKeyRef.current = generationKey;
                            autoEnrichIdeasWithoutKeywordMetrics([...cachedBlogIdeas, ...cachedSoftwareIdeas]);
                            return;
                        }
                    }
                } catch (e) {
                    console.warn("Failed to read idea burst cache:", e);
                }
            }

            // 3) Generate only when no persisted ideas and no cache.
            setLoadedFromStored(false);
            setLoadedFromCache(false);
            lastGeneratedKeyRef.current = generationKey;
            await generateIdeas();
        };

        hydrateIdeas();
        return () => {
            cancelled = true;
        };
    }, [isOpen, topicId, subtopic?.id, subtopic?.name, user?.id, cacheKey]);

    React.useEffect(() => {
        if (!isOpen) {
            lastGeneratedKeyRef.current = null;
            autoEnrichAttemptedRef.current.clear();
        }
    }, [isOpen]);

    const safeValueLayerTags = React.useMemo(() => {
        if (!subtopic) return [] as string[];
        const raw = (subtopic as any).value_layer_tags;
        return Array.isArray(raw) ? raw.filter(Boolean) : [];
    }, [subtopic]);

    const subtopicKeywordMetrics = React.useMemo(() => {
        if (!subtopic) return new Map<string, KeywordMetricRow>();

        const merged = new Map<string, KeywordMetricRow>();
        [
            (subtopic as any).keywords,
            (subtopic as any)?.trend_analysis?.keyword_evidence,
            (subtopic as any)?.monetization_data?.keyword_evidence,
        ].forEach((source) => {
            const sourceMap = extractKeywordMetricsMapFromUnknown(source);
            sourceMap.forEach((value, key) => {
                const existing = merged.get(key);
                if (!existing) {
                    merged.set(key, value);
                    return;
                }
                merged.set(key, {
                    keyword: existing.keyword || value.keyword,
                    search_volume: existing.search_volume ?? value.search_volume,
                    keyword_difficulty: existing.keyword_difficulty ?? value.keyword_difficulty,
                    cpc: existing.cpc ?? value.cpc,
                });
            });
        });

        return merged;
    }, [subtopic]);

    const generateIdeas = async () => {
        if (!subtopic || !user) return;

        setLoading(true);
        setError(null);
        setLoadedFromCache(false);
        setLoadedFromStored(false);
        setBlogIdeas([]);
        setSoftwareIdeas([]);
        setSelectedBlogIdeas(new Set());
        setSelectedSoftwareIdeas(new Set());
        setPublished(false);
        setSaved(false);
        setEnrichResultMessage(null);

        try {
            const rawKeywords = parseJsonLike<any[]>((subtopic as any).keywords, []);
            const keywordPayload = Array.isArray(rawKeywords) && rawKeywords.length > 0
                ? rawKeywords
                : Array.from(subtopicKeywordMetrics.values()).map((row) => ({
                    keyword: row.keyword,
                    ...(row.search_volume !== null && row.search_volume !== undefined ? { search_volume: row.search_volume } : {}),
                    ...(row.keyword_difficulty !== null && row.keyword_difficulty !== undefined ? { keyword_difficulty: row.keyword_difficulty } : {}),
                    ...(row.cpc !== null && row.cpc !== undefined ? { cpc: row.cpc } : {}),
                }));

            const monetizationData = subtopic.monetization_data || {};
            const affiliateOffers = monetizationData.details?.affiliate_categories || [];

            const result = await contentIdeasService.generateBurst({
                topicId,
                subtopicName: subtopic.name,
                keywords: keywordPayload,
                affiliateOffers,
                userId: user.id,
                intentBucket: subtopic.intent_bucket,
                decisionFocus: subtopic.decision_focus,
                angleQuestion: subtopic.angle_question,
                valueLayerTags: safeValueLayerTags,
                clusterType: subtopic.cluster_type,
                primaryUserOutcome: subtopic.primary_user_outcome,
                serpIntentMatch: subtopic.serp_intent_match,
                toolPotentialScore: subtopic.tool_potential_score,
            });

            const nextBlogIdeas = result.blog_ideas || [];
            const nextSoftwareIdeas = result.software_ideas || [];
            setBlogIdeas(nextBlogIdeas);
            setSoftwareIdeas(nextSoftwareIdeas);
            const generatedCount = Number(result.generated_count ?? (nextBlogIdeas.length + nextSoftwareIdeas.length));
            const persistedCount = Number(result.persisted_count ?? generatedCount);
            const persistedIdeaIds = Array.isArray(result.persisted_idea_ids)
                ? result.persisted_idea_ids.filter((id): id is string => typeof id === "string" && id.length > 0)
                : [];

            if (persistedCount <= 0 && generatedCount > 0) {
                setError("Ideas were generated but were not saved to Supabase. Reloading will lose them until persistence is fixed.");
            } else if (persistedCount < generatedCount) {
                setError(result.persistence_warning || `Only ${persistedCount} of ${generatedCount} ideas were saved. Some may disappear on reload.`);
            }

            if (ENABLE_IDEA_BURST_CACHE && cacheKey) {
                try {
                    const payload: CachedIdeaBurst = {
                        blogIdeas: nextBlogIdeas,
                        softwareIdeas: nextSoftwareIdeas,
                        cachedAt: new Date().toISOString(),
                    };
                    localStorage.setItem(cacheKey, JSON.stringify(payload));
                } catch (e) {
                    console.warn("Failed to persist idea burst cache:", e);
                }
            }
            const autoEnrichIds = persistedIdeaIds.length > 0
                ? persistedIdeaIds
                : [...nextBlogIdeas, ...nextSoftwareIdeas].map((idea) => idea.id).filter(Boolean);
            if (autoEnrichIds.length > 0) {
                await runEnrichment(autoEnrichIds, { silent: true });
            }
        } catch (err: any) {
            console.error("Failed to generate ideas:", err);
            setError(err.message || "Failed to generate content ideas. Please try again.");
        } finally {
            setLoading(false);
        }
    };

    const toggleBlogSelection = (ideaId: string) => {
        setSelectedBlogIdeas(prev => {
            const newSet = new Set(prev);
            if (newSet.has(ideaId)) {
                newSet.delete(ideaId);
            } else {
                newSet.add(ideaId);
            }
            return newSet;
        });
    };

    const toggleSoftwareSelection = (ideaId: string) => {
        setSelectedSoftwareIdeas(prev => {
            const newSet = new Set(prev);
            if (newSet.has(ideaId)) {
                newSet.delete(ideaId);
            } else {
                newSet.add(ideaId);
            }
            return newSet;
        });
    };

    const handlePublishBlogs = async () => {
        if (!user || selectedBlogIdeas.size === 0) return;

        setPublishing(true);
        try {
            const ideaIds = Array.from(selectedBlogIdeas);
            const result = await contentIdeasService.publishContentIdeas(ideaIds, user.id);
            if (!result.success || (result.publishedToTitlesCount <= 0 && result.publishedCount <= 0)) {
                setError("Publish did not create any content items. Please refresh and try again.");
                return;
            }
            setPublished(true);
            // Keep user in-place so they can continue saving software ideas
            // without losing research topic context.
            setSelectedBlogIdeas(new Set());
            setTimeout(() => {
                setPublished(false);
            }, 2000);
        } catch (err) {
            console.error("Failed to publish ideas:", err);
            setError("Failed to publish ideas. Please try again.");
        } finally {
            setPublishing(false);
        }
    };

    const handleSaveSoftware = async () => {
        if (!user || selectedSoftwareIdeas.size === 0) return;

        setSavingSoftware(true);
        try {
            // Mark software ideas as saved (different status)
            const ideaIds = Array.from(selectedSoftwareIdeas);
            const result = await contentIdeasService.publishContentIdeas(ideaIds, user.id);
            if (!result.success || result.publishedCount <= 0) {
                setError("Save did not persist software ideas. Please refresh and try again.");
                return;
            }
            setSaved(true);
            setTimeout(() => {
                setSaved(false);
                setSelectedSoftwareIdeas(new Set());
            }, 2000);
        } catch (err) {
            console.error("Failed to save software ideas:", err);
            setError("Failed to save ideas. Please try again.");
        } finally {
            setSavingSoftware(false);
        }
    };

    const applyEnrichedMetrics = (ideaId: string, metrics?: {
        total_search_volume: number;
        average_cpc: number;
        average_difficulty: number;
        affiliate_offer_count: number;
    }, keywordMetricsMap?: Record<string, { search_volume?: number; keyword_difficulty?: number; cpc?: number }>, keywordsUsed?: string[], restatedTitle?: string, selectedPrimaryKeyword?: string) => {
        if (!metrics) return;
        setBlogIdeas((prev) => prev.map((idea) => (
            idea.id === ideaId
                ? {
                    ...idea,
                    total_search_volume: metrics.total_search_volume,
                    average_cpc: metrics.average_cpc,
                    average_difficulty: metrics.average_difficulty,
                    ...(restatedTitle ? { title: restatedTitle } : {}),
                    ...(selectedPrimaryKeyword ? { search_phrase: selectedPrimaryKeyword } : {}),
                    ...(Array.isArray(keywordsUsed) && keywordsUsed.length > 0 ? {
                        keywords: keywordsUsed,
                        primary_keywords: [selectedPrimaryKeyword || keywordsUsed[0]],
                        secondary_keywords: keywordsUsed.filter(k => k !== (selectedPrimaryKeyword || keywordsUsed[0])),
                    } : {}),
                    ...(keywordMetricsMap ? { keyword_metrics: keywordMetricsMap } : {}),
                }
                : idea
        )));
        setSoftwareIdeas((prev) => prev.map((idea) => (
            idea.id === ideaId
                ? {
                    ...idea,
                    total_search_volume: metrics.total_search_volume,
                    average_cpc: metrics.average_cpc,
                    average_difficulty: metrics.average_difficulty,
                    ...(restatedTitle ? { title: restatedTitle } : {}),
                    ...(selectedPrimaryKeyword ? { search_phrase: selectedPrimaryKeyword } : {}),
                    ...(Array.isArray(keywordsUsed) && keywordsUsed.length > 0 ? {
                        keywords: keywordsUsed,
                        primary_keywords: [selectedPrimaryKeyword || keywordsUsed[0]],
                        secondary_keywords: keywordsUsed.filter(k => k !== (selectedPrimaryKeyword || keywordsUsed[0])),
                    } : {}),
                    ...(keywordMetricsMap ? { keyword_metrics: keywordMetricsMap } : {}),
                }
                : idea
        )));
    };

    const runEnrichment = async (ideaIds: string[], options?: { silent?: boolean }) => {
        if (!user || ideaIds.length === 0) return;
        const silent = Boolean(options?.silent);

        try {
            setEnrichingIdeas(true);
            if (!silent) {
                setEnrichResultMessage(null);
                setError(null);
            }
            // Avoid gateway timeouts by enriching one idea per API request.
            // This guarantees per-idea persistence even if one request fails.
            const allResults: Array<{
                idea_id: string;
                status: "enriched" | "failed";
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
            }> = [];
            let enrichedCount = 0;

            for (const ideaId of ideaIds) {
                const singleResult = await contentIdeasService.enrichContentIdeas([ideaId], user.id);
                if (singleResult.success && singleResult.enrichedCount > 0) {
                    enrichedCount += singleResult.enrichedCount;
                }
                if (Array.isArray(singleResult.results) && singleResult.results.length > 0) {
                    allResults.push(...singleResult.results);
                } else {
                    allResults.push({
                        idea_id: ideaId,
                        status: "failed",
                        reason: "No response rows returned",
                    });
                }
            }

            if (enrichedCount <= 0) {
                if (!silent) {
                    setError("No ideas were enriched. Please try again.");
                }
                return;
            }

            allResults.forEach((item) => {
                if (item.status === "enriched") {
                    applyEnrichedMetrics(
                        item.idea_id,
                        item.metrics,
                        item.keyword_metrics_map,
                        item.keywords_used,
                        item.restated_title,
                        item.selected_primary_keyword
                    );
                }
            });

            const failedCount = Math.max(0, ideaIds.length - enrichedCount);
            if (!silent) {
                setEnrichResultMessage(
                    failedCount > 0
                        ? `Enriched ${enrichedCount} ideas (${failedCount} failed).`
                        : `Enriched ${enrichedCount} ideas.`
                );
            }
        } catch (err) {
            console.error("Failed to enrich selected ideas:", err);
            if (!silent) {
                setError("Failed to run SEO/Offers for selected ideas.");
            }
        } finally {
            setEnrichingIdeas(false);
        }
    };

    const handleEnrichSelectedIdeas = async (ideaIds: string[]) => {
        await runEnrichment(ideaIds, { silent: false });
    };

    const autoEnrichIdeasWithoutKeywordMetrics = async (ideas: ContentIdea[]) => {
        if (!user || !subtopic) return;
        const scopeKey = `${topicId}:${subtopic.id}:${user.id}`;
        if (autoEnrichAttemptedRef.current.has(scopeKey)) return;

        const missingIdeaIds = ideas
            .filter((idea) => !ideaHasExactKeywordMetrics(idea))
            .map((idea) => idea.id)
            .filter(Boolean);

        if (missingIdeaIds.length === 0) return;
        autoEnrichAttemptedRef.current.add(scopeKey);
        await runEnrichment(missingIdeaIds, { silent: true });
    };

    const handleClearCachedIdeas = () => {
        if (!cacheKey) return;
        try {
            localStorage.removeItem(cacheKey);
        } catch (e) {
            console.warn("Failed to clear cached idea burst:", e);
        }
        setLoadedFromCache(false);
        setLoadedFromStored(false);
        setBlogIdeas([]);
        setSoftwareIdeas([]);
        setSelectedBlogIdeas(new Set());
        setSelectedSoftwareIdeas(new Set());
        setExpandedMetrics(null);
        setError(null);
        lastGeneratedKeyRef.current = null;
    };

    const handleRegenerateIdeas = async () => {
        handleClearCachedIdeas();
        await generateIdeas();
    };

    const selectAllBlogs = () => {
        setSelectedBlogIdeas(new Set(blogIdeas.map(i => i.id)));
    };

    const deselectAllBlogs = () => {
        setSelectedBlogIdeas(new Set());
    };

    const selectAllSoftware = () => {
        setSelectedSoftwareIdeas(new Set(softwareIdeas.map(i => i.id)));
    };

    const deselectAllSoftware = () => {
        setSelectedSoftwareIdeas(new Set());
    };

    const toggleMetricsExpansion = (ideaId: string) => {
        setExpandedMetrics(prev => prev === ideaId ? null : ideaId);
    };

    const internalLinkGroups = React.useMemo(() => buildInternalLinkGroups(blogIdeas), [blogIdeas]);

    if (!isOpen || !subtopic) return null;

    const totalBlogIdeas = blogIdeas.length;
    const totalSoftwareIdeas = softwareIdeas.length;

    return (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="bg-slate-900 border border-white/10 rounded-2xl shadow-2xl max-w-5xl w-full max-h-[90vh] flex flex-col"
            >
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-white/10">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-indigo-500/20 rounded-xl">
                            <Sparkles className="w-5 h-5 text-indigo-400" />
                        </div>
                        <div>
                            <h2 className="text-lg font-bold text-white">Content Ideas</h2>
                            <p className="text-xs text-slate-500">
                                Generated for: <span className="text-indigo-400">{subtopic.name}</span>
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="text-slate-500 hover:text-white transition-colors text-xl leading-none"
                    >
                        <X className="w-6 h-6" />
                    </button>
                </div>
                <div className="px-6 py-3 border-b border-white/10 bg-white/5 flex flex-wrap items-center justify-between gap-2">
                    <div className="text-xs text-slate-400">
                        {loadedFromStored
                            ? "Loaded existing ideas already saved for this subtopic"
                            : loadedFromCache
                                ? "Loaded previously generated candidates"
                                : "Candidates are generated for this subtopic"}
                    </div>
                    <div className="flex items-center gap-2">
                        {ENABLE_IDEA_BURST_CACHE && cacheKey && (
                            <Button
                                onClick={handleClearCachedIdeas}
                                variant="ghost"
                                size="sm"
                                className="h-7 px-2 text-xs text-slate-300 hover:text-white"
                            >
                                Clear cached ideas
                            </Button>
                        )}
                        <Button
                            onClick={handleRegenerateIdeas}
                            variant="outline"
                            size="sm"
                            disabled={loading}
                            className="h-7 px-2 text-xs border-white/15"
                        >
                            {loading ? (
                                <>
                                    <Loader2 className="w-3 h-3 mr-1 animate-spin" />
                                    Regenerating...
                                </>
                            ) : (
                                "Regenerate"
                            )}
                        </Button>
                    </div>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-auto p-6">
                    {(subtopic.intent_bucket || subtopic.decision_focus || subtopic.angle_question || safeValueLayerTags.length > 0) && (
                        <div className="mb-4 rounded-xl border border-white/10 bg-white/5 p-3">
                            <div className="flex flex-wrap items-center gap-2 mb-2">
                                {subtopic.intent_bucket && (
                                    <span className={`text-[10px] px-2 py-0.5 rounded-full border ${intentChipClass(subtopic.intent_bucket)}`}>
                                        Intent: {subtopic.intent_bucket}
                                    </span>
                                )}
                                {safeValueLayerTags.map((tag, idx) => (
                                    <span key={`${tag}-${idx}`} className="text-[10px] px-2 py-0.5 rounded-full border border-indigo-500/30 bg-indigo-500/10 text-indigo-300">
                                        {tag}
                                    </span>
                                ))}
                            </div>
                            {subtopic.decision_focus && (
                                <p className="text-[11px] text-slate-300 mb-1">
                                    <span className="text-indigo-400 font-medium">Decision Focus:</span> {subtopic.decision_focus}
                                </p>
                            )}
                            {subtopic.angle_question && (
                                <p className="text-[11px] text-slate-300">
                                    <span className="text-indigo-400 font-medium">Angle Question:</span> {subtopic.angle_question}
                                </p>
                            )}
                        </div>
                    )}

                    {internalLinkGroups.length > 0 && (
                        <div className="mb-4 rounded-xl border border-indigo-500/20 bg-indigo-500/5 p-3">
                            <p className="text-[11px] text-indigo-300 font-medium mb-2">Internal Link Groups</p>
                            <div className="space-y-1">
                                {internalLinkGroups.slice(0, 4).map((group) => (
                                    <div key={group.hook} className="flex items-start justify-between gap-3 text-[11px]">
                                        <span className="text-slate-300 leading-snug">{group.hook}</span>
                                        <span className="text-indigo-300 bg-indigo-500/10 border border-indigo-500/20 rounded px-1.5 py-0.5 flex-shrink-0">
                                            {group.count}
                                        </span>
                                    </div>
                                ))}
                            </div>
                            {subtopic.decision_focus && (
                                <p className="mt-2 text-[10px] text-slate-400">
                                    Linked by decision focus: <span className="text-slate-300">{subtopic.decision_focus}</span>
                                </p>
                            )}
                        </div>
                    )}

                    {loading && (
                        <div className="py-16 flex flex-col items-center gap-4">
                            <Loader2 className="w-12 h-12 text-indigo-400 animate-spin" />
                            <p className="text-slate-400">Generating content ideas...</p>
                            <p className="text-xs text-slate-600">Analyzing keywords, trends, and monetization potential</p>
                        </div>
                    )}

                    {error && (
                        <div className="py-8 text-center">
                            <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4 mb-4">
                                <p className="text-red-400">{error}</p>
                            </div>
                            <Button onClick={generateIdeas} variant="outline" className="border-white/10">
                                Try Again
                            </Button>
                        </div>
                    )}

                    {!loading && !error && totalBlogIdeas === 0 && totalSoftwareIdeas === 0 && (
                        <div className="py-16 text-center">
                            <Lightbulb className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                            <p className="text-slate-400">No ideas generated.</p>
                            <Button onClick={generateIdeas} variant="outline" className="mt-4 border-white/10">
                                Regenerate
                            </Button>
                        </div>
                    )}

                    {!loading && !error && (totalBlogIdeas > 0 || totalSoftwareIdeas > 0) && (
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            {/* Blog Articles Column */}
                            {totalBlogIdeas > 0 && (
                                <div className="space-y-4">
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-2">
                                            <BookOpen className="w-4 h-4 text-blue-400" />
                                            <h3 className="text-sm font-semibold text-white">Blog Articles</h3>
                                            <span className="text-xs text-slate-500">({totalBlogIdeas})</span>
                                        </div>
                                        <div className="flex gap-1">
                                            <Button
                                                onClick={selectAllBlogs}
                                                variant="ghost"
                                                size="sm"
                                                className="text-xs text-slate-400 hover:text-white h-7 px-2"
                                            >
                                                Select All
                                            </Button>
                                            <Button
                                                onClick={deselectAllBlogs}
                                                variant="ghost"
                                                size="sm"
                                                className="text-xs text-slate-400 hover:text-white h-7 px-2"
                                            >
                                                Clear
                                            </Button>
                                        </div>
                                    </div>

                                    <p className="text-xs text-slate-500 flex items-center gap-1">
                                        <Info className="w-3 h-3" />
                                        These become articles in Content Studio
                                    </p>

                                    <div className="space-y-3">
                                        {blogIdeas.map((idea) => (
                                            <BlogIdeaCard
                                                key={idea.id}
                                                idea={idea}
                                                isSelected={selectedBlogIdeas.has(idea.id)}
                                                onToggle={() => toggleBlogSelection(idea.id)}
                                                isExpanded={expandedMetrics === idea.id}
                                                onToggleMetrics={() => toggleMetricsExpansion(idea.id)}
                                                mapContext={{
                                                    projectName: projectName || undefined,
                                                    categoryPath: categoryPath || undefined,
                                                    angleQuestion: subtopic.angle_question || undefined,
                                                    clusterName: subtopic.name || undefined,
                                                    topicTitle: topicTitle || undefined,
                                                }}
                                                keywordMetricsMap={subtopicKeywordMetrics}
                                                onKeywordSaved={(ideaId, primary, secondary, metrics) => {
                                                    applyEnrichedMetrics(
                                                        ideaId,
                                                        {
                                                            total_search_volume: metrics.volume || 0,
                                                            average_cpc: metrics.cpc || 0,
                                                            average_difficulty: metrics.difficulty || 0,
                                                            affiliate_offer_count: idea.viability_score || 0,
                                                        },
                                                        undefined,
                                                        [primary, ...secondary],
                                                        undefined,
                                                        primary
                                                    );
                                                }}
                                            />
                                        ))}
                                    </div>

                                    {/* Blog Actions */}
                                    <div className="pt-4 border-t border-white/10">
                                        <Button
                                            onClick={() => handleEnrichSelectedIdeas(Array.from(selectedBlogIdeas))}
                                            disabled={selectedBlogIdeas.size === 0 || enrichingIdeas}
                                            variant="outline"
                                            className="w-full mb-2 border-sky-500/30 text-sky-300 hover:bg-sky-500/10"
                                        >
                                            {enrichingIdeas ? (
                                                <>
                                                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                                    Running SEO/Offers...
                                                </>
                                            ) : (
                                                <>
                                                    <Sparkles className="w-4 h-4 mr-2" />
                                                    Run SEO/Offers ({selectedBlogIdeas.size})
                                                </>
                                            )}
                                        </Button>
                                        <Button
                                            onClick={handlePublishBlogs}
                                            disabled={selectedBlogIdeas.size === 0 || publishing || enrichingIdeas}
                                            className="w-full bg-indigo-600 hover:bg-indigo-700 text-white"
                                        >
                                            {publishing ? (
                                                <>
                                                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                                    Publishing...
                                                </>
                                            ) : (
                                                <>
                                                    <BookOpen className="w-4 h-4 mr-2" />
                                                    Publish to Content Studio ({selectedBlogIdeas.size})
                                                </>
                                            )}
                                        </Button>
                                    </div>
                                </div>
                            )}

                            {/* Software Tools Column */}
                            {totalSoftwareIdeas > 0 && (
                                <div className="space-y-4">
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-2">
                                            <Code className="w-4 h-4 text-amber-400" />
                                            <h3 className="text-sm font-semibold text-white">Software Tools to Build</h3>
                                            <span className="text-xs text-slate-500">({totalSoftwareIdeas})</span>
                                        </div>
                                        <div className="flex gap-1">
                                            <Button
                                                onClick={selectAllSoftware}
                                                variant="ghost"
                                                size="sm"
                                                className="text-xs text-slate-400 hover:text-white h-7 px-2"
                                            >
                                                Select All
                                            </Button>
                                            <Button
                                                onClick={deselectAllSoftware}
                                                variant="ghost"
                                                size="sm"
                                                className="text-xs text-slate-400 hover:text-white h-7 px-2"
                                            >
                                                Clear
                                            </Button>
                                        </div>
                                    </div>

                                    <p className="text-xs text-slate-500 flex items-center gap-1">
                                        <Info className="w-3 h-3" />
                                        Tools/features to develop for your website
                                    </p>

                                    <div className="space-y-3">
                                        {softwareIdeas.map((idea) => (
                                            <SoftwareIdeaCard
                                                key={idea.id}
                                                idea={idea}
                                                isSelected={selectedSoftwareIdeas.has(idea.id)}
                                                onToggle={() => toggleSoftwareSelection(idea.id)}
                                                isExpanded={expandedMetrics === idea.id}
                                                onToggleMetrics={() => toggleMetricsExpansion(idea.id)}
                                                mapContext={{
                                                    projectName: projectName || undefined,
                                                    categoryPath: categoryPath || undefined,
                                                    angleQuestion: subtopic.angle_question || undefined,
                                                    clusterName: subtopic.name || undefined,
                                                    topicTitle: topicTitle || undefined,
                                                }}
                                                keywordMetricsMap={subtopicKeywordMetrics}
                                            />
                                        ))}
                                    </div>

                                    {/* Software Actions */}
                                    <div className="pt-4 border-t border-white/10">
                                        <Button
                                            onClick={() => handleEnrichSelectedIdeas(Array.from(selectedSoftwareIdeas))}
                                            disabled={selectedSoftwareIdeas.size === 0 || enrichingIdeas}
                                            variant="outline"
                                            className="w-full mb-2 border-sky-500/30 text-sky-300 hover:bg-sky-500/10"
                                        >
                                            {enrichingIdeas ? (
                                                <>
                                                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                                    Running SEO/Offers...
                                                </>
                                            ) : (
                                                <>
                                                    <Sparkles className="w-4 h-4 mr-2" />
                                                    Run SEO/Offers ({selectedSoftwareIdeas.size})
                                                </>
                                            )}
                                        </Button>
                                        <Button
                                            onClick={handleSaveSoftware}
                                            disabled={selectedSoftwareIdeas.size === 0 || savingSoftware || enrichingIdeas}
                                            variant="outline"
                                            className="w-full border-amber-500/30 text-amber-400 hover:bg-amber-500/10"
                                        >
                                            {savingSoftware ? (
                                                <>
                                                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                                    Saving...
                                                </>
                                            ) : saved ? (
                                                <>
                                                    <Check className="w-4 h-4 mr-2" />
                                                    Saved!
                                                </>
                                            ) : (
                                                <>
                                                    <Save className="w-4 h-4 mr-2" />
                                                    Save for Later ({selectedSoftwareIdeas.size})
                                                </>
                                            )}
                                        </Button>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {published && (
                        <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="mt-6 bg-green-500/10 border border-green-500/20 rounded-xl p-4 text-center"
                        >
                            <Check className="w-6 h-6 text-green-400 mx-auto mb-2" />
                            <p className="text-green-400 font-medium">Articles published successfully!</p>
                            <p className="text-xs text-slate-500 mt-1">Redirecting to Content Studio...</p>
                        </motion.div>
                    )}

                    {enrichResultMessage && (
                        <motion.div
                            initial={{ opacity: 0, y: 8 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="mt-4 bg-sky-500/10 border border-sky-500/20 rounded-xl p-3 text-center"
                        >
                            <p className="text-sky-300 text-sm">{enrichResultMessage}</p>
                        </motion.div>
                    )}
                </div>

                {/* Footer */}
                {!loading && !error && (totalBlogIdeas > 0 || totalSoftwareIdeas > 0) && (
                    <div className="border-t border-white/10 p-4 flex items-center justify-between bg-white/5">
                        <div className="text-xs text-slate-500">
                            <span className="text-slate-400">Subtopic:</span> {subtopic.search_volume?.toLocaleString() || 0} monthly searches
                            {subtopic.cpc ? ` • $${subtopic.cpc.toFixed(2)} CPC` : ''}
                        </div>
                        <Button
                            onClick={onClose}
                            variant="ghost"
                            size="sm"
                            className="text-slate-400 hover:text-white"
                        >
                            Close
                        </Button>
                    </div>
                )}
            </motion.div>
        </div>
    );
}

// Blog Idea Card Component
interface BlogIdeaCardProps {
    idea: ContentIdea;
    isSelected: boolean;
    onToggle: () => void;
    isExpanded: boolean;
    onToggleMetrics: () => void;
    mapContext: {
        projectName?: string;
        categoryPath?: string;
        topicTitle?: string;
        angleQuestion?: string;
        clusterName?: string;
    };
    keywordMetricsMap: Map<string, KeywordMetricRow>;
    onKeywordSaved?: (
        ideaId: string,
        primary: string,
        secondary: string[],
        metrics: { volume: number | null; difficulty: number | null; cpc: number | null }
    ) => void;
}

function BlogIdeaCard({ idea, isSelected, onToggle, isExpanded: _isExpanded, onToggleMetrics: _onToggleMetrics, mapContext, keywordMetricsMap, onKeywordSaved }: BlogIdeaCardProps) {
    const [showKeywordModal, setShowKeywordModal] = React.useState(false);
    const keywords = resolveIdeaKeywords(idea);
    const rankFactors = getRankFactors(idea);
    const ideaKeywordMetricsMap = React.useMemo(() => extractIdeaKeywordMetricsMap(idea), [idea]);
    const hasAnyRealKeywordMetrics = keywords.some((kw: string) => {
        const row = resolveKeywordMetricRow(kw, ideaKeywordMetricsMap, keywordMetricsMap);
        return Boolean((row?.search_volume || 0) > 0 || (row?.keyword_difficulty || 0) > 0 || (row?.cpc || 0) > 0);
    });
    const exactAggregate = React.useMemo(
        () => computeAggregateFromExactMap(keywords, ideaKeywordMetricsMap, keywordMetricsMap),
        [keywords, ideaKeywordMetricsMap, keywordMetricsMap]
    );
    const totalSearchVolume = hasAnyRealKeywordMetrics
        ? (Number(idea.total_search_volume || 0) > 0 ? Number(idea.total_search_volume || 0) : exactAggregate.totalVolume)
        : exactAggregate.totalVolume;
    const averageDifficulty = hasAnyRealKeywordMetrics
        ? (Number(idea.average_difficulty || 0) > 0 ? Number(idea.average_difficulty || 0) : exactAggregate.avgDifficulty)
        : exactAggregate.avgDifficulty;
    const averageCpc = hasAnyRealKeywordMetrics
        ? (Number(idea.average_cpc || 0) > 0 ? Number(idea.average_cpc || 0) : exactAggregate.avgCpc)
        : exactAggregate.avgCpc;
    const rawTrace = React.useMemo(() => getRawDataforSeoTrace(idea), [idea]);
    const rawTraceAvailable = Boolean(rawTrace);

    return (
        <motion.div
            layout
            className={`rounded-xl border transition-all duration-200 ${
                isSelected
                    ? 'bg-indigo-500/10 border-indigo-500/50'
                    : 'bg-white/5 border-white/10 hover:border-white/20 hover:bg-white/8'
            }`}
        >
            {/* Main card content - clickable for selection */}
            <div
                onClick={onToggle}
                className="p-4 cursor-pointer"
            >
                <div className="flex items-start gap-3">
                    <div className={`w-5 h-5 rounded border flex items-center justify-center flex-shrink-0 mt-0.5 ${
                        isSelected
                            ? 'bg-indigo-500 border-indigo-500'
                            : 'border-slate-600'
                    }`}>
                        {isSelected && <Check className="w-3 h-3 text-white" />}
                    </div>
                    <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                            <h4 className={`font-medium text-sm ${isSelected ? 'text-indigo-300' : 'text-white'}`}>
                                {idea.title}
                            </h4>
                            {isSentToContentLibrary(idea) && (
                                <span className="inline-flex flex-shrink-0 items-center whitespace-nowrap rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium leading-none text-emerald-300">
                                    In Library
                                </span>
                            )}
                        </div>
                        {idea.description && (
                            <p className="text-xs text-slate-400 line-clamp-2 mb-2">{idea.description}</p>
                        )}

                        {/* Primary Keywords */}
                        {keywords.length > 0 && (
                            <div className="flex flex-wrap gap-1 mb-2">
                                {keywords.slice(0, 4).map((kw, idx) => (
                                    <span
                                        key={idx}
                                        className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] bg-blue-500/20 text-blue-300"
                                    >
                                        {kw}
                                    </span>
                                ))}
                                {keywords.length > 4 && (
                                    <span className="text-[10px] text-slate-500 px-1">
                                        +{keywords.length - 4} more
                                    </span>
                                )}
                            </div>
                        )}

                        {/* Metrics */}
                        <div className="flex flex-wrap items-center gap-3 text-[11px]">
                            {totalSearchVolume > 0 && (
                                <span className="flex items-center gap-1">
                                    <span className="text-blue-400">Vol:</span>
                                    <span className="text-slate-300">{Math.round(totalSearchVolume).toLocaleString()}</span>
                                </span>
                            )}
                            {averageDifficulty > 0 && (
                                <span className="flex items-center gap-1">
                                    <span className={averageDifficulty > 60 ? 'text-red-400' : averageDifficulty > 30 ? 'text-yellow-400' : 'text-green-400'}>
                                        KD:
                                    </span>
                                    <span className="text-slate-300">{Math.round(averageDifficulty)}</span>
                                </span>
                            )}
                            {averageCpc > 0 && (
                                <span className="flex items-center gap-1">
                                    <span className="text-emerald-400">CPC:</span>
                                    <span className="text-slate-300">${averageCpc.toFixed(2)}</span>
                                </span>
                            )}
                            {(idea.viability_score || 0) > 0 && (
                                <span className="flex items-center gap-1">
                                    <span className="text-indigo-400">Viability:</span>
                                    <span className="text-slate-300">{Math.round(idea.viability_score || 0)}%</span>
                                </span>
                            )}
                            {(idea.opportunity_score || 0) > 0 && (
                                <span className="flex items-center gap-1">
                                    <span className="text-violet-400">Opportunity:</span>
                                    <span className="text-slate-300">{Math.round(idea.opportunity_score || 0)}%</span>
                                </span>
                            )}
                        </div>

                        {/* Affiliate Hook - Full width */}
                        {idea.monetization_hook && (
                            <div className="mt-2 pt-2 border-t border-white/5">
                                <p className="text-[11px] text-amber-400/80">
                                    <span className="text-amber-500 font-medium">💰 Monetization:</span> {idea.monetization_hook}
                                </p>
                            </div>
                        )}

                        {/* Strategy Context */}
                        {(idea.article_format || idea.user_decision_helped || idea.internal_link_hook || idea.target_intent) && (
                            <div className="mt-2 space-y-1">
                                {idea.article_format && (
                                    <div className="text-[11px] text-slate-300 flex items-center gap-2">
                                        <span className="text-indigo-400 font-medium">Format:</span>
                                        <span className="px-1.5 py-0.5 rounded border bg-indigo-500/15 border-indigo-500/30 text-indigo-300 text-[10px]">
                                            {idea.article_format}
                                        </span>
                                    </div>
                                )}
                                {idea.target_intent && (
                                    <div className="text-[11px] text-slate-300 flex items-center gap-2">
                                        <span className="text-indigo-400 font-medium">Intent:</span>
                                        <span className={`px-1.5 py-0.5 rounded border text-[10px] ${intentChipClass(idea.target_intent)}`}>
                                            {idea.target_intent}
                                        </span>
                                    </div>
                                )}
                                {idea.user_decision_helped && (
                                    <p className="text-[11px] text-slate-300">
                                        <span className="text-indigo-400 font-medium">Decision Helped:</span> {idea.user_decision_helped}
                                    </p>
                                )}
                                {idea.internal_link_hook && (
                                    <p className="text-[11px] text-slate-300">
                                        <span className="text-indigo-400 font-medium">Internal Link Hook:</span> {idea.internal_link_hook}
                                    </p>
                                )}
                            </div>
                        )}

                        {rankFactors.length > 0 && (
                            <div className="mt-2 flex flex-wrap gap-1">
                                {rankFactors.slice(0, 4).map((factor) => (
                                    <span key={factor.label} className="text-[10px] px-1.5 py-0.5 rounded border border-violet-500/30 bg-violet-500/10 text-violet-300">
                                        {factor.label}: {Math.round(factor.value)}
                                    </span>
                                ))}
                            </div>
                        )}

                        <div className="mt-2 pt-2 border-t border-white/5 flex flex-wrap gap-1">
                            {mapContext.projectName && (
                                <span className="text-[10px] px-1.5 py-0.5 rounded border border-slate-500/30 bg-slate-500/10 text-slate-300">
                                    Project: {mapContext.projectName}
                                </span>
                            )}
                            {mapContext.categoryPath && (
                                <span className="text-[10px] px-1.5 py-0.5 rounded border border-slate-500/30 bg-slate-500/10 text-slate-300">
                                    Category: {mapContext.categoryPath}
                                </span>
                            )}
                            {mapContext.topicTitle && (
                                <span className="text-[10px] px-1.5 py-0.5 rounded border border-slate-500/30 bg-slate-500/10 text-slate-300">
                                    Topic: {mapContext.topicTitle}
                                </span>
                            )}
                            {mapContext.angleQuestion && (
                                <span className="text-[10px] px-1.5 py-0.5 rounded border border-slate-500/30 bg-slate-500/10 text-slate-300">
                                    Angle
                                </span>
                            )}
                            {mapContext.clusterName && (
                                <span className="text-[10px] px-1.5 py-0.5 rounded border border-slate-500/30 bg-slate-500/10 text-slate-300">
                                    Cluster: {mapContext.clusterName}
                                </span>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Keyword Intelligence Button */}
            <div className="border-t border-white/5">
                <button
                    onClick={(e) => {
                        e.stopPropagation();
                        setShowKeywordModal(true);
                    }}
                    className={`w-full px-4 py-2 flex items-center justify-center gap-2 text-[11px] transition-colors ${
                        rawTraceAvailable
                            ? 'text-indigo-400 hover:text-indigo-300 hover:bg-indigo-500/5'
                            : 'text-slate-600 hover:text-slate-400 hover:bg-white/3'
                    }`}
                >
                    <Key className="w-3 h-3" />
                    Keyword Intelligence
                    {rawTraceAvailable ? (
                        <span className="text-[9px] px-1.5 py-0.5 rounded bg-emerald-500/15 border border-emerald-500/30 text-emerald-300">Data available</span>
                    ) : (
                        <span className="text-[9px] px-1.5 py-0.5 rounded bg-amber-500/10 border border-amber-500/20 text-amber-400">Needs enrichment</span>
                    )}
                </button>
            </div>

            {/* Keyword Intelligence Modal */}
            <AnimatePresence>
                {showKeywordModal && (
                    <KeywordIntelligenceModal
                        isOpen={showKeywordModal}
                        onClose={() => setShowKeywordModal(false)}
                        idea={idea}
                        onSaved={(primary, secondary, metrics) => {
                            onKeywordSaved?.(idea.id, primary, secondary, metrics);
                        }}
                    />
                )}
            </AnimatePresence>
        </motion.div>
    );
}

// Software Idea Card Component
interface SoftwareIdeaCardProps {
    idea: ContentIdea;
    isSelected: boolean;
    onToggle: () => void;
    isExpanded: boolean;
    onToggleMetrics: () => void;
    mapContext: {
        projectName?: string;
        categoryPath?: string;
        topicTitle?: string;
        angleQuestion?: string;
        clusterName?: string;
    };
    keywordMetricsMap: Map<string, KeywordMetricRow>;
}

function SoftwareIdeaCard({ idea, isSelected, onToggle, isExpanded, onToggleMetrics, mapContext, keywordMetricsMap }: SoftwareIdeaCardProps) {
    const keywords = resolveIdeaKeywords(idea);
    const rankFactors = getRankFactors(idea);
    const ideaKeywordMetricsMap = React.useMemo(() => extractIdeaKeywordMetricsMap(idea), [idea]);
    const hasAnyRealKeywordMetrics = keywords.some((kw: string) => {
        const row = resolveKeywordMetricRow(kw, ideaKeywordMetricsMap, keywordMetricsMap);
        return Boolean((row?.search_volume || 0) > 0 || (row?.keyword_difficulty || 0) > 0 || (row?.cpc || 0) > 0);
    });
    const exactAggregate = React.useMemo(
        () => computeAggregateFromExactMap(keywords, ideaKeywordMetricsMap, keywordMetricsMap),
        [keywords, ideaKeywordMetricsMap, keywordMetricsMap]
    );
    const totalSearchVolume = hasAnyRealKeywordMetrics
        ? (Number(idea.total_search_volume || 0) > 0 ? Number(idea.total_search_volume || 0) : exactAggregate.totalVolume)
        : exactAggregate.totalVolume;
    const averageDifficulty = hasAnyRealKeywordMetrics
        ? (Number(idea.average_difficulty || 0) > 0 ? Number(idea.average_difficulty || 0) : exactAggregate.avgDifficulty)
        : exactAggregate.avgDifficulty;
    const averageCpc = hasAnyRealKeywordMetrics
        ? (Number(idea.average_cpc || 0) > 0 ? Number(idea.average_cpc || 0) : exactAggregate.avgCpc)
        : exactAggregate.avgCpc;
    const rawTrace = React.useMemo(() => getRawDataforSeoTrace(idea), [idea]);
    const rawTraceAvailable = Boolean(rawTrace);

    return (
        <motion.div
            layout
            className={`rounded-xl border transition-all duration-200 ${
                isSelected
                    ? 'bg-amber-500/10 border-amber-500/50'
                    : 'bg-white/5 border-white/10 hover:border-white/20 hover:bg-white/8'
            }`}
        >
            {/* Main card content - clickable for selection */}
            <div
                onClick={onToggle}
                className="p-4 cursor-pointer"
            >
                <div className="flex items-start gap-3">
                    <div className={`w-5 h-5 rounded border flex items-center justify-center flex-shrink-0 mt-0.5 ${
                        isSelected
                            ? 'bg-amber-500 border-amber-500'
                            : 'border-slate-600'
                    }`}>
                        {isSelected && <Check className="w-3 h-3 text-white" />}
                    </div>
                    <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                            <h4 className={`font-medium text-sm ${isSelected ? 'text-amber-300' : 'text-white'}`}>
                                {idea.title}
                            </h4>
                            {isSentToContentLibrary(idea) && (
                                <span className="inline-flex flex-shrink-0 items-center whitespace-nowrap rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium leading-none text-emerald-300">
                                    In Library
                                </span>
                            )}
                        </div>
                        {idea.description && (
                            <p className="text-xs text-slate-400 line-clamp-2 mb-2">{idea.description}</p>
                        )}

                        {/* Primary Keywords */}
                        {keywords.length > 0 && (
                            <div className="flex flex-wrap gap-1 mb-2">
                                {keywords.slice(0, 4).map((kw, idx) => (
                                    <span
                                        key={idx}
                                        className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] bg-amber-500/20 text-amber-300"
                                    >
                                        {kw}
                                    </span>
                                ))}
                                {keywords.length > 4 && (
                                    <span className="text-[10px] text-slate-500 px-1">
                                        +{keywords.length - 4} more
                                    </span>
                                )}
                            </div>
                        )}

                        {/* Metrics */}
                        <div className="flex flex-wrap items-center gap-3 text-[11px]">
                            {totalSearchVolume > 0 && (
                                <span className="flex items-center gap-1">
                                    <span className="text-blue-400">Demand:</span>
                                    <span className="text-slate-300">{Math.round(totalSearchVolume).toLocaleString()}/mo</span>
                                </span>
                            )}
                            {averageDifficulty > 0 && (
                                <span className="flex items-center gap-1">
                                    <span className={averageDifficulty > 60 ? 'text-red-400' : averageDifficulty > 30 ? 'text-yellow-400' : 'text-green-400'}>
                                        KD:
                                    </span>
                                    <span className="text-slate-300">{Math.round(averageDifficulty)}</span>
                                </span>
                            )}
                            {averageCpc > 0 && (
                                <span className="flex items-center gap-1">
                                    <span className="text-emerald-400">CPC:</span>
                                    <span className="text-slate-300">${averageCpc.toFixed(2)}</span>
                                </span>
                            )}
                            {(idea.viability_score || 0) > 0 && (
                                <span className="flex items-center gap-1">
                                    <span className="text-amber-400">Opportunity:</span>
                                    <span className="text-slate-300">{Math.round(idea.viability_score || 0)}%</span>
                                </span>
                            )}
                            {(idea.opportunity_score || 0) > 0 && (
                                <span className="flex items-center gap-1">
                                    <span className="text-violet-400">Rank:</span>
                                    <span className="text-slate-300">{Math.round(idea.opportunity_score || 0)}%</span>
                                </span>
                            )}
                        </div>

                        {/* Monetization Strategy */}
                        {idea.monetization_hook && (
                            <div className="mt-2 pt-2 border-t border-white/5">
                                <p className="text-[11px] text-emerald-400/80">
                                    <span className="text-emerald-500 font-medium">💡 Revenue Model:</span> {idea.monetization_hook}
                                </p>
                            </div>
                        )}

                        {/* Product Strategy Context */}
                        {(idea.product_type || idea.user_job_to_be_done || idea.build_complexity || idea.distribution_angle || idea.output_result) && (
                            <div className="mt-2 space-y-1">
                                {idea.product_type && (
                                    <div className="text-[11px] text-slate-300 flex items-center gap-2">
                                        <span className="text-amber-400 font-medium">Product Type:</span>
                                        <span className="px-1.5 py-0.5 rounded border bg-amber-500/15 border-amber-500/30 text-amber-300 text-[10px]">
                                            {idea.product_type}
                                        </span>
                                    </div>
                                )}
                                {idea.user_job_to_be_done && (
                                    <p className="text-[11px] text-slate-300">
                                        <span className="text-amber-400 font-medium">User Job:</span> {idea.user_job_to_be_done}
                                    </p>
                                )}
                                {idea.build_complexity && (
                                    <div className="text-[11px] text-slate-300 flex items-center gap-2">
                                        <span className="text-amber-400 font-medium">Build Complexity:</span>
                                        <span className={`px-1.5 py-0.5 rounded border text-[10px] ${complexityChipClass(idea.build_complexity)}`}>
                                            {idea.build_complexity}
                                        </span>
                                    </div>
                                )}
                                {idea.output_result && (
                                    <p className="text-[11px] text-slate-300">
                                        <span className="text-amber-400 font-medium">Output:</span> {idea.output_result}
                                    </p>
                                )}
                                {idea.distribution_angle && (
                                    <p className="text-[11px] text-slate-300">
                                        <span className="text-amber-400 font-medium">Distribution:</span> {idea.distribution_angle}
                                    </p>
                                )}
                            </div>
                        )}

                        {rankFactors.length > 0 && (
                            <div className="mt-2 flex flex-wrap gap-1">
                                {rankFactors.slice(0, 4).map((factor) => (
                                    <span key={factor.label} className="text-[10px] px-1.5 py-0.5 rounded border border-violet-500/30 bg-violet-500/10 text-violet-300">
                                        {factor.label}: {Math.round(factor.value)}
                                    </span>
                                ))}
                            </div>
                        )}

                        <div className="mt-2 pt-2 border-t border-white/5 flex flex-wrap gap-1">
                            {mapContext.projectName && (
                                <span className="text-[10px] px-1.5 py-0.5 rounded border border-slate-500/30 bg-slate-500/10 text-slate-300">
                                    Project: {mapContext.projectName}
                                </span>
                            )}
                            {mapContext.categoryPath && (
                                <span className="text-[10px] px-1.5 py-0.5 rounded border border-slate-500/30 bg-slate-500/10 text-slate-300">
                                    Category: {mapContext.categoryPath}
                                </span>
                            )}
                            {mapContext.topicTitle && (
                                <span className="text-[10px] px-1.5 py-0.5 rounded border border-slate-500/30 bg-slate-500/10 text-slate-300">
                                    Topic: {mapContext.topicTitle}
                                </span>
                            )}
                            {mapContext.angleQuestion && (
                                <span className="text-[10px] px-1.5 py-0.5 rounded border border-slate-500/30 bg-slate-500/10 text-slate-300">
                                    Angle
                                </span>
                            )}
                            {mapContext.clusterName && (
                                <span className="text-[10px] px-1.5 py-0.5 rounded border border-slate-500/30 bg-slate-500/10 text-slate-300">
                                    Cluster: {mapContext.clusterName}
                                </span>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Expandable Metrics Section */}
            {keywords.length > 0 && (
                <div className="border-t border-white/5">
                    <button
                        onClick={(e) => {
                            e.stopPropagation();
                            onToggleMetrics();
                        }}
                        className="w-full px-4 py-2 flex items-center justify-center gap-2 text-[11px] text-amber-400 hover:text-amber-300 hover:bg-amber-500/5 transition-colors"
                    >
                        <BarChart3 className="w-3 h-3" />
                        {isExpanded ? 'Hide Keyword Metrics' : 'View Keyword Metrics'}
                        {isExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                    </button>

                    <AnimatePresence>
                        {isExpanded && (
                            <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: 'auto', opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                transition={{ duration: 0.2 }}
                                className="overflow-hidden"
                            >
                                <div className="px-4 pb-4">
                                    <div className="bg-slate-800/50 rounded-lg overflow-hidden border border-white/5">
                                        <table className="w-full text-[11px]">
                                            <thead>
                                                <tr className="bg-slate-800/80 border-b border-white/5">
                                                    <th className="text-left px-3 py-2 text-slate-400 font-medium">Keyword</th>
                                                    <th className="text-right px-3 py-2 text-slate-400 font-medium">Volume</th>
                                                    <th className="text-right px-3 py-2 text-slate-400 font-medium">KD</th>
                                                    <th className="text-right px-3 py-2 text-slate-400 font-medium">CPC</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {keywords.map((kw, idx) => (
                                                    (() => {
                                                        const row = resolveKeywordMetricRow(kw, ideaKeywordMetricsMap, keywordMetricsMap);
                                                        const rowVolume = row?.search_volume ?? null;
                                                        const rowKD = row?.keyword_difficulty ?? null;
                                                        const rowCPC = row?.cpc ?? null;
                                                        return (
                                                    <tr key={idx} className="border-b border-white/5 last:border-0">
                                                        <td className="px-3 py-2 text-slate-300 truncate max-w-[120px]">{kw}</td>
                                                        <td className="px-3 py-2 text-right text-slate-300">
                                                            {rowVolume !== null ? rowVolume.toLocaleString() : '-'}
                                                        </td>
                                                        <td className="px-3 py-2 text-right">
                                                            <span className={(rowKD || 0) > 60 ? 'text-red-400' : (rowKD || 0) > 30 ? 'text-yellow-400' : 'text-green-400'}>
                                                                {rowKD !== null ? rowKD : '-'}
                                                            </span>
                                                        </td>
                                                        <td className="px-3 py-2 text-right text-slate-300">
                                                            {rowCPC !== null ? `$${rowCPC.toFixed(2)}` : '-'}
                                                        </td>
                                                    </tr>
                                                        );
                                                    })()
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                    <p className="text-[10px] text-slate-500 mt-2 text-center">
                                        {hasAnyRealKeywordMetrics
                                            ? "Note: Keyword rows show exact per-keyword metrics when available."
                                            : "Note: No exact per-keyword metrics yet. Run SEO/Offers to populate keyword-level data."}
                                    </p>
                                    <div className="mt-1 flex justify-center">
                                        <span className={`text-[10px] px-2 py-0.5 rounded border ${
                                            rawTraceAvailable
                                                ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
                                                : "border-amber-500/30 bg-amber-500/10 text-amber-300"
                                        }`}>
                                            Raw DFS Trace: {rawTraceAvailable ? "Available" : "Missing"}
                                        </span>
                                    </div>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            )}
        </motion.div>
    );
}
