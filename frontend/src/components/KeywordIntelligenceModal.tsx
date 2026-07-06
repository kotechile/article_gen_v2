import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    X,
    TrendingUp,
    TrendingDown,
    Minus,
    BarChart2,
    ChevronUp,
    Save,
    Loader2,
    Star,
    Target,
    AlertTriangle,
    RefreshCw,
    CheckCircle2,
    Plus,
    Search,
    Sparkles,
} from "lucide-react";
import {
    AreaChart,
    Area,
    XAxis,
    YAxis,
    Tooltip as RechartTooltip,
    ResponsiveContainer,
} from "recharts";
import { Button } from "@/components/ui/button";
import type { ContentIdea, DFSKeywordRow, DFSParsedOutput } from "@/types/idea-burst";
import { contentIdeasService } from "@/services/content-ideas.service";
import { useAuth } from "@/context/auth-context";

// ─── Parser ─────────────────────────────────────────────────────────────────

function safeJsonParse<T>(value: unknown, fallback: T): T {
    if (value === null || value === undefined) return fallback;
    if (typeof value === "object") return value as T;
    if (typeof value !== "string") return fallback;
    try {
        return (JSON.parse(value) as T) ?? fallback;
    } catch {
        return fallback;
    }
}

function parseDataForSEOOutput(raw: unknown): DFSParsedOutput | null {
    const parsed = safeJsonParse<any>(raw, null);
    if (!parsed || typeof parsed !== "object") return null;

    // Check if it's already in our flat format (e.g. from a previous manual save)
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

    // Seed keyword
    const seedData = result.seed_keyword_data;
    if (seedData) {
        rows.push(buildRow(seedData, "seed", 0, null));
    }

    // Related keywords (items[])
    const items: any[] = Array.isArray(result.items) ? result.items : [];
    items.forEach((item: any) => {
        const kd = item?.keyword_data;
        if (!kd) return;
        // Skip the seed keyword if it appears again in items
        if (kd.keyword === seedData?.keyword) return;
        rows.push(buildRow(kd, "related", item.depth ?? 1, item.related_keywords ?? null));
    });

    return {
        seed_keyword: result.seed_keyword ?? (seedData?.keyword ?? ""),
        total_count: result.total_count ?? rows.length,
        items_count: result.items_count ?? rows.length,
        rows,
    };
}

function buildRow(
    kd: any,
    type: "seed" | "related",
    depth: number,
    relatedKeywords: string[] | null
): DFSKeywordRow {
    const ki = kd?.keyword_info ?? {};
    const kp = kd?.keyword_properties ?? {};
    const si = kd?.search_intent_info ?? {};
    const monthly: any[] = Array.isArray(ki.monthly_searches) ? ki.monthly_searches : [];

    // Sort monthly_searches chronologically (oldest first)
    const sortedMonthly = [...monthly].sort((a, b) => {
        const da = a.year * 100 + a.month;
        const db = b.year * 100 + b.month;
        return da - db;
    });

    return {
        keyword: kd?.keyword ?? "",
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

// ─── UI helpers ─────────────────────────────────────────────────────────────

function competitionPill(level: string | null) {
    if (!level) return null;
    const lc = level.toUpperCase();
    if (lc === "HIGH") return "bg-red-500/20 text-red-300 border-red-500/30";
    if (lc === "MEDIUM") return "bg-amber-500/20 text-amber-300 border-amber-500/30";
    return "bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
}

function intentPill(intent: string | null) {
    if (!intent) return "bg-slate-500/20 text-slate-400 border-slate-500/30";
    const lc = intent.toLowerCase();
    if (lc.includes("commercial")) return "bg-amber-500/20 text-amber-300 border-amber-500/30";
    if (lc.includes("transactional")) return "bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
    if (lc.includes("navigational")) return "bg-violet-500/20 text-violet-300 border-violet-500/30";
    return "bg-blue-500/20 text-blue-300 border-blue-500/30";
}

function kdColor(kd: number | null) {
    if (kd === null) return "text-slate-500";
    if (kd >= 60) return "text-red-400";
    if (kd >= 30) return "text-amber-400";
    return "text-emerald-400";
}

function trendIcon(trend: number | null) {
    if (trend === null || trend === 0) return <Minus className="w-3 h-3 text-slate-500" />;
    if (trend > 0) return <TrendingUp className="w-3 h-3 text-emerald-400" />;
    return <TrendingDown className="w-3 h-3 text-red-400" />;
}

function fmtVol(v: number | null) {
    if (v === null) return "—";
    if (v === 0) return "0";
    if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
    if (v >= 1_000) return `${(v / 1_000).toFixed(1)}K`;
    return v.toString();
}

function fmtCpc(v: number | null) {
    if (v === null) return "—";
    return `$${v.toFixed(2)}`;
}

function extractKeywordValues(value: unknown, depth = 0): string[] {
    if (depth > 5 || value === null || value === undefined) return [];

    if (Array.isArray(value)) {
        return value.flatMap((item) => extractKeywordValues(item, depth + 1));
    }

    if (typeof value === "object") {
        const maybeKeyword = (value as any)?.keyword;
        if (typeof maybeKeyword === "string") {
            return extractKeywordValues(maybeKeyword, depth + 1);
        }
        return [];
    }

    if (typeof value !== "string") return [];
    const raw = value.trim();
    if (!raw) return [];

    // Heuristic: remove surrounding brackets/quotes, including malformed fragments like ["keyword
    let cleaned = raw;
    while ((cleaned.startsWith('[') && cleaned.endsWith(']')) || (cleaned.startsWith('"') && cleaned.endsWith('"')) || (cleaned.startsWith("'") && cleaned.endsWith("'"))) {
        cleaned = cleaned.slice(1, -1).trim();
        if (!cleaned) break;
    }
    cleaned = cleaned.replace(/^[\[\]"']+/, "").replace(/[\[\]"']+$/, "").trim();
    if (!cleaned) return [];

    try {
        const parsed = JSON.parse(raw);
        if (typeof parsed === "string" && parsed !== raw) {
            return extractKeywordValues(parsed, depth + 1);
        }
        if (Array.isArray(parsed)) {
            return parsed.flatMap(item => extractKeywordValues(item, depth + 1));
        }
    } catch {
        // Continue with manual parsing
    }

    const parts = cleaned
        .split(",")
        .map((part) => part.trim().replace(/^[\[\]"']+|[\[\]"']+$/g, "").trim())
        .filter(Boolean);
    return parts.length > 1 ? parts : [cleaned.replace(/^[\[\]"']+|[\[\]"']+$/g, "").trim()];
}

function normalizeKeywordKey(input: string): string {
    return String(input || "").trim().toLowerCase();
}

function collectIdeaKeywordPool(idea: ContentIdea): string[] {
    const metadata = safeJsonParse<any>((idea as any).idea_metadata, {});
    const topicKeywordResearch = metadata?.topic_keyword_research || {};
    const directCandidates = safeJsonParse<any[]>((idea as any).keyword_candidates_json, [])
        .map((row) => typeof row === 'string' ? row : String(row?.keyword || row?.term || "").trim())
        .filter(Boolean);
    const rankedCandidates = safeJsonParse<any[]>(metadata?.seo_offer_enrichment?.keyword_ranked_candidates, [])
        .map((row) => String(row?.keyword || "").trim())
        .filter(Boolean);
    const pass2Candidates = safeJsonParse<any[]>(metadata?.keyword_pass_2?.keyword_ranked_candidates, [])
        .map((row) => String(row?.keyword || "").trim())
        .filter(Boolean);
    const topicCandidates = safeJsonParse<any[]>(topicKeywordResearch?.keyword_candidates, [])
        .map((row) => String(row?.keyword || "").trim())
        .filter(Boolean);
    const qualifiedCandidates = safeJsonParse<any[]>(topicKeywordResearch?.qualified_keywords, [])
        .map((row) => String(row?.keyword || "").trim())
        .filter(Boolean);
    const merged = [
        ...extractKeywordValues((idea as any).keywords),
        ...extractKeywordValues((idea as any).primary_keywords ?? (idea as any).primary_keyword),
        ...extractKeywordValues((idea as any).secondary_keywords ?? (idea as any).secondary_keywords_json),
        ...directCandidates,
        ...extractKeywordValues(metadata?.seo_offer_enrichment?.keywords_used),
        ...extractKeywordValues(metadata?.input_keywords),
        ...extractKeywordValues(metadata?.keyword_seed_pack?.input_keywords),
        ...topicCandidates,
        ...qualifiedCandidates,
        ...rankedCandidates,
        ...pass2Candidates,
    ];
    const out: string[] = [];
    const seen = new Set<string>();
    for (const kw of merged) {
        const raw = String(kw || "").trim();
        const norm = normalizeKeywordKey(raw);
        if (!raw || !norm || seen.has(norm)) continue;
        seen.add(norm);
        out.push(raw);
    }
    return out;
}

function collectIdeaKeywordMetricMap(idea: ContentIdea): Map<string, { search_volume: number | null; keyword_difficulty: number | null; cpc: number | null }> {
    const map = new Map<string, { search_volume: number | null; keyword_difficulty: number | null; cpc: number | null }>();
    const metadata = safeJsonParse<any>((idea as any).idea_metadata, {});
    const topicKeywordResearch = metadata?.topic_keyword_research || {};
    const fromColumn = safeJsonParse<any>((idea as any).keyword_metrics, {});
    const fromSelectedMetrics = safeJsonParse<any>((idea as any).selected_keyword_metrics_json, {});
    const fromMetadata = safeJsonParse<any>(metadata?.seo_offer_enrichment?.keyword_metrics, {});
    const fromTopicCandidates = safeJsonParse<any[]>(topicKeywordResearch?.keyword_candidates, []);
    const fromQualifiedCandidates = safeJsonParse<any[]>(topicKeywordResearch?.qualified_keywords, []);
    const fromCandidates = safeJsonParse<any[]>(metadata?.seo_offer_enrichment?.keyword_ranked_candidates, []);
    const fromPass2Candidates = safeJsonParse<any[]>(metadata?.keyword_pass_2?.keyword_ranked_candidates, []);
    const directCandidates = safeJsonParse<any[]>((idea as any).keyword_candidates_json, []);

    const ingest = (keywordInput: unknown, metricInput: any) => {
        const keyword = String(keywordInput || "").trim();
        const key = normalizeKeywordKey(keyword);
        if (!keyword || !key) return;
        const volume = Number(metricInput?.search_volume);
        const kd = Number(metricInput?.keyword_difficulty);
        const cpc = Number(metricInput?.cpc);
        map.set(key, {
            search_volume: Number.isFinite(volume) ? volume : null,
            keyword_difficulty: Number.isFinite(kd) ? kd : null,
            cpc: Number.isFinite(cpc) ? cpc : null,
        });
    };

    const ingestSource = (source: any) => {
        if (Array.isArray(source)) {
            source.forEach((row) => ingest(row?.keyword || row?.term, row));
            return;
        }
        if (source && typeof source === "object") {
            Object.entries(source).forEach(([k, v]) => ingest(k, v));
        }
    };

    ingestSource(fromColumn);
    ingestSource(fromMetadata);
    ingestSource(fromTopicCandidates);
    ingestSource(fromQualifiedCandidates);
    ingestSource(fromCandidates);
    ingestSource(fromPass2Candidates);
    ingestSource(directCandidates);
    if (fromSelectedMetrics && typeof fromSelectedMetrics === 'object') {
        ingest(fromSelectedMetrics?.primary?.keyword, fromSelectedMetrics?.primary);
        const secondaries = fromSelectedMetrics?.secondary || fromSelectedMetrics?.secondaries;
        if (Array.isArray(secondaries)) {
            secondaries.forEach((row: any) => ingest(row?.keyword, row));
        }
    }
    return map;
}

function buildSyntheticParsedFromIdea(idea: ContentIdea): DFSParsedOutput | null {
    const metadata = safeJsonParse<any>((idea as any).idea_metadata, {});
    const topicKeywordResearch = metadata?.topic_keyword_research || {};
    const metricMap = collectIdeaKeywordMetricMap(idea);
    const keywords = collectIdeaKeywordPool(idea);
    if (!keywords.length && metricMap.size === 0) return null;

    const seedKeyword =
        String(topicKeywordResearch?.primary_keyword || (idea.primary_keywords || [])[0] || (idea.keywords || [])[0] || "").trim()
        || "keyword";

    const candidateRows = safeJsonParse<any[]>(topicKeywordResearch?.keyword_candidates, []);
    const candidateMap = new Map<string, any>();
    for (const row of candidateRows) {
        const keyword = String(row?.keyword || "").trim();
        const key = normalizeKeywordKey(keyword);
        if (keyword && key && !candidateMap.has(key)) {
            candidateMap.set(key, row);
        }
    }

    const rows: DFSKeywordRow[] = keywords.map((keyword, index) => {
        const key = normalizeKeywordKey(keyword);
        const metric = metricMap.get(key);
        const candidate = candidateMap.get(key) || {};
        const isSeed = key === normalizeKeywordKey(seedKeyword);
        return {
            keyword,
            type: isSeed ? "seed" : "related",
            depth: isSeed ? 0 : (index === 0 ? 1 : 2),
            search_volume: metric?.search_volume ?? candidate?.search_volume ?? null,
            competition: null,
            competition_level: candidate?.competition_level ?? null,
            cpc: metric?.cpc ?? candidate?.cpc ?? null,
            keyword_difficulty: metric?.keyword_difficulty ?? candidate?.keyword_difficulty ?? null,
            main_intent: candidate?.intent_label ?? null,
            foreign_intents: null,
            monthly_searches: [],
            search_volume_trend: null,
            low_top_of_page_bid: null,
            high_top_of_page_bid: null,
            se_results_count: null,
            related_keywords: null,
        }
    });

    return {
        seed_keyword: seedKeyword,
        total_count: rows.length,
        items_count: rows.length,
        rows,
    };
}

function mergeParsedWithIdeaKeywordPool(parsed: DFSParsedOutput | null, idea: ContentIdea): DFSParsedOutput | null {
    if (!parsed) return parsed;
    const keywordPool = collectIdeaKeywordPool(idea);
    if (!keywordPool.length) return parsed;
    const existing = new Set((parsed.rows || []).map((r) => normalizeKeywordKey(r.keyword)));
    const metricMap = collectIdeaKeywordMetricMap(idea);
    const missingRows: DFSKeywordRow[] = [];

    for (const keyword of keywordPool) {
        const norm = normalizeKeywordKey(keyword);
        if (!norm || existing.has(norm)) continue;
        existing.add(norm);
        const metric = metricMap.get(norm);
        missingRows.push({
            keyword,
            type: "related",
            depth: 1,
            search_volume: metric?.search_volume ?? null,
            competition: null,
            competition_level: null,
            cpc: metric?.cpc ?? null,
            keyword_difficulty: metric?.keyword_difficulty ?? null,
            main_intent: null,
            foreign_intents: null,
            monthly_searches: [],
            search_volume_trend: null,
            low_top_of_page_bid: null,
            high_top_of_page_bid: null,
            se_results_count: null,
            related_keywords: null,
        });
    }

    if (!missingRows.length) return parsed;
    return {
        ...parsed,
        rows: [...parsed.rows, ...missingRows],
        total_count: Number(parsed.total_count || 0) + missingRows.length,
        items_count: Number(parsed.items_count || 0) + missingRows.length,
    };
}

const MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

// ─── Trend Sparkline ─────────────────────────────────────────────────────────

function TrendChart({ data, keyword }: { data: DFSKeywordRow["monthly_searches"]; keyword: string }) {
    if (!data || data.length === 0) {
        return (
            <div className="flex items-center justify-center h-20 text-xs text-slate-500">
                No trend data available
            </div>
        );
    }

    const chartData = data.map((d) => ({
        name: `${MONTH_LABELS[d.month - 1]} ${d.year}`,
        volume: d.search_volume ?? 0,
    }));

    return (
        <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 120 }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
        >
            <div className="px-2 pt-2 pb-1">
                <p className="text-[10px] text-slate-400 mb-1 truncate">
                    Monthly trend — <span className="text-indigo-300">{keyword}</span>
                </p>
                <ResponsiveContainer width="100%" height={84}>
                    <AreaChart data={chartData} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
                        <defs>
                            <linearGradient id={`grad-${keyword.replace(/\s+/g, "-")}`} x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <XAxis
                            dataKey="name"
                            tick={{ fill: "#64748b", fontSize: 8 }}
                            tickLine={false}
                            axisLine={false}
                            interval="preserveStartEnd"
                        />
                        <YAxis hide />
                        <RechartTooltip
                            contentStyle={{
                                background: "#1e293b",
                                border: "1px solid rgba(255,255,255,0.1)",
                                borderRadius: 6,
                                fontSize: 11,
                                color: "#e2e8f0",
                            }}
                            formatter={(val: unknown) => [
                                typeof val === "number" ? val.toLocaleString() : String(val ?? ""),
                                "Searches",
                            ]}
                            labelStyle={{ color: "#94a3b8", fontSize: 10 }}
                        />
                        <Area
                            type="monotone"
                            dataKey="volume"
                            stroke="#6366f1"
                            strokeWidth={1.5}
                            fill={`url(#grad-${keyword.replace(/\s+/g, "-")})`}
                            dot={false}
                            activeDot={{ r: 3, fill: "#6366f1" }}
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </motion.div>
    );
}

// ─── Keyword Row ──────────────────────────────────────────────────────────────

interface KeywordTableRowProps {
    row: DFSKeywordRow;
    isPrimary: boolean;
    isSecondary: boolean;
    onSelectPrimary: () => void;
    onToggleSecondary: () => void;
    showChart: boolean;
    onToggleChart: () => void;
    disabled: boolean;
    isNew?: boolean;
}

function KeywordTableRow({
    row,
    isPrimary,
    isSecondary,
    onSelectPrimary,
    onToggleSecondary,
    showChart,
    onToggleChart,
    disabled,
    isNew = false,
}: KeywordTableRowProps) {
    const isSeed = row.type === "seed";
    const hasVolume = (row.search_volume ?? 0) > 0;

    const rowBg = isPrimary
        ? "bg-emerald-500/10 border-l-2 border-emerald-500"
        : isSecondary
        ? "bg-sky-500/10 border-l-2 border-sky-500"
        : isSeed
        ? "bg-indigo-500/8 border-l-2 border-indigo-400"
        : "border-l-2 border-transparent";

    return (
        <>
            <tr
                className={`border-b border-white/5 last:border-0 transition-colors ${rowBg} ${
                    !hasVolume ? "opacity-50" : ""
                }`}
            >
                {/* Primary radio */}
                <td className="px-3 py-2.5 w-10">
                    <button
                        onClick={onSelectPrimary}
                        disabled={disabled}
                        title="Set as primary keyword"
                        className={`w-4 h-4 rounded-full border-2 flex items-center justify-center transition-all ${
                            isPrimary
                                ? "border-emerald-400 bg-emerald-400"
                                : "border-slate-500 hover:border-emerald-400"
                        } disabled:opacity-40`}
                    >
                        {isPrimary && <div className="w-1.5 h-1.5 rounded-full bg-white" />}
                    </button>
                </td>

                {/* Secondary checkbox */}
                <td className="px-2 py-2.5 w-10">
                    <button
                        onClick={onToggleSecondary}
                        disabled={disabled || isPrimary}
                        title="Add as secondary keyword"
                        className={`w-4 h-4 rounded border-2 flex items-center justify-center transition-all ${
                            isSecondary
                                ? "border-sky-400 bg-sky-400"
                                : "border-slate-500 hover:border-sky-400"
                        } disabled:opacity-30`}
                    >
                        {isSecondary && (
                            <svg className="w-2.5 h-2.5 text-white" fill="none" viewBox="0 0 10 10">
                                <path d="M1.5 5L4 7.5 8.5 2.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                            </svg>
                        )}
                    </button>
                </td>

                {/* Keyword + badges */}
                <td className="px-3 py-2.5 min-w-[180px] max-w-[240px]">
                    <div className="flex flex-col gap-0.5">
                        <span
                            className={`text-xs font-medium leading-snug ${
                                isSeed ? "text-indigo-200" : "text-slate-200"
                            }`}
                        >
                            {row.keyword}
                        </span>
                        <div className="flex items-center gap-1 flex-wrap">
                            {isSeed && (
                                <span className="text-[9px] px-1 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 font-medium">
                                    SEED
                                </span>
                            )}
                            {isNew && (
                                <span className="text-[9px] px-1 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-medium">
                                    NEW
                                </span>
                            )}
                            {row.depth > 0 && (
                                <span className="text-[9px] text-slate-500">depth {row.depth}</span>
                            )}
                        </div>
                    </div>
                </td>

                {/* Volume */}
                <td className="px-3 py-2.5 text-right whitespace-nowrap">
                    <span className={`text-xs ${hasVolume ? "text-slate-200" : "text-slate-600"}`}>
                        {fmtVol(row.search_volume)}
                    </span>
                </td>

                {/* KD */}
                <td className="px-3 py-2.5 text-right whitespace-nowrap">
                    <span className={`text-xs font-medium ${kdColor(row.keyword_difficulty)}`}>
                        {row.keyword_difficulty !== null ? row.keyword_difficulty : "—"}
                    </span>
                </td>

                {/* CPC */}
                <td className="px-3 py-2.5 text-right whitespace-nowrap">
                    <span className="text-xs text-slate-300">{fmtCpc(row.cpc)}</span>
                </td>

                {/* Competition */}
                <td className="px-3 py-2.5 text-center whitespace-nowrap">
                    {row.competition_level ? (
                        <span
                            className={`text-[9px] px-1.5 py-0.5 rounded border font-medium ${competitionPill(
                                row.competition_level
                            )}`}
                        >
                            {row.competition_level}
                        </span>
                    ) : (
                        <span className="text-slate-600 text-xs">—</span>
                    )}
                </td>

                {/* Intent */}
                <td className="px-3 py-2.5 text-center whitespace-nowrap">
                    {row.main_intent ? (
                        <span
                            className={`text-[9px] px-1.5 py-0.5 rounded border font-medium ${intentPill(
                                row.main_intent
                            )}`}
                        >
                            {row.main_intent}
                        </span>
                    ) : (
                        <span className="text-slate-600 text-xs">—</span>
                    )}
                </td>

                {/* Monthly trend % */}
                <td className="px-3 py-2.5 text-right whitespace-nowrap">
                    <div className="flex items-center justify-end gap-0.5">
                        {trendIcon(row.search_volume_trend?.monthly ?? null)}
                        {row.search_volume_trend?.monthly !== undefined && row.search_volume_trend.monthly !== 0 ? (
                            <span
                                className={`text-[10px] ${
                                    row.search_volume_trend.monthly > 0 ? "text-emerald-400" : "text-red-400"
                                }`}
                            >
                                {row.search_volume_trend.monthly > 0 ? "+" : ""}
                                {row.search_volume_trend.monthly}%
                            </span>
                        ) : (
                            <span className="text-slate-600 text-[10px]">0%</span>
                        )}
                    </div>
                </td>

                {/* Chart toggle */}
                <td className="px-2 py-2.5 text-center w-10">
                    {row.monthly_searches.length > 0 ? (
                        <button
                            onClick={onToggleChart}
                            title="View monthly trend chart"
                            className={`p-1 rounded transition-colors ${
                                showChart ? "bg-indigo-500/20 text-indigo-300" : "text-slate-500 hover:text-indigo-300"
                            }`}
                        >
                            {showChart ? <ChevronUp className="w-3.5 h-3.5" /> : <BarChart2 className="w-3.5 h-3.5" />}
                        </button>
                    ) : (
                        <span className="text-slate-700 text-xs">—</span>
                    )}
                </td>
            </tr>

            {/* Trend chart row */}
            <AnimatePresence>
                {showChart && (
                    <tr>
                        <td colSpan={10} className="px-0 py-0 bg-slate-800/40">
                            <TrendChart data={row.monthly_searches} keyword={row.keyword} />
                        </td>
                    </tr>
                )}
            </AnimatePresence>
        </>
    );
}

// ─── Modal Summary Bar ────────────────────────────────────────────────────────

function SummaryBar({ data }: { data: DFSParsedOutput }) {
    const rows = data.rows;
    const volumeRows = rows.filter((r) => (r.search_volume ?? 0) > 0);
    const totalVol = volumeRows.reduce((s, r) => s + (r.search_volume ?? 0), 0);
    const kdRows = rows.filter((r) => r.keyword_difficulty !== null);
    const avgKd = kdRows.length
        ? kdRows.reduce((s, r) => s + (r.keyword_difficulty ?? 0), 0) / kdRows.length
        : null;
    const cpcRows = rows.filter((r) => r.cpc !== null && (r.cpc ?? 0) > 0);
    const avgCpc = cpcRows.length
        ? cpcRows.reduce((s, r) => s + (r.cpc ?? 0), 0) / cpcRows.length
        : null;

    const stats = [
        { label: "Keywords", value: rows.length.toString(), color: "text-indigo-300" },
        { label: "Total Volume", value: fmtVol(totalVol), color: "text-blue-300" },
        { label: "Avg KD", value: avgKd !== null ? Math.round(avgKd).toString() : "—", color: kdColor(avgKd) },
        { label: "Avg CPC", value: avgCpc !== null ? fmtCpc(avgCpc) : "—", color: "text-emerald-300" },
    ];

    return (
        <div className="grid grid-cols-4 divide-x divide-white/10 bg-white/5 border-b border-white/10">
            {stats.map((s) => (
                <div key={s.label} className="flex flex-col items-center py-3 px-4">
                    <span className={`text-lg font-bold ${s.color}`}>{s.value}</span>
                    <span className="text-[10px] text-slate-500 uppercase tracking-wide mt-0.5">{s.label}</span>
                </div>
            ))}
        </div>
    );
}

// ─── Footer ──────────────────────────────────────────────────────────────────

interface FooterProps {
    primaryKeyword: string | null;
    secondaryKeywords: string[];
    saving: boolean;
    saved: boolean;
    onSave: () => void;
    onRemoveSecondary?: (kw: string) => void;
    autoSaveActive?: boolean;
}

function Footer({ primaryKeyword, secondaryKeywords, saving, saved, onSave, onRemoveSecondary, autoSaveActive = false }: FooterProps) {
    return (
        <div className="border-t border-white/10 p-4 bg-slate-900 flex flex-col sm:flex-row items-start sm:items-center gap-3">
            <div className="flex-1 flex flex-col gap-1 min-w-0">
                <div className="flex items-center gap-2">
                    <Star className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                    <span className="text-[11px] text-slate-400 flex-shrink-0">Primary:</span>
                    {primaryKeyword ? (
                        <span className="text-xs text-emerald-300 font-medium truncate">{primaryKeyword}</span>
                    ) : (
                        <span className="text-[11px] text-slate-600 italic">None selected</span>
                    )}
                </div>
                {secondaryKeywords.length > 0 && (
                    <div className="flex items-start gap-2">
                        <Target className="w-3.5 h-3.5 text-sky-400 flex-shrink-0 mt-0.5" />
                        <span className="text-[11px] text-slate-400 flex-shrink-0">Secondary:</span>
                        <div className="flex flex-wrap gap-1">
                            {secondaryKeywords.map((kw) => (
                                <div
                                    key={kw}
                                    className="group flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded bg-sky-500/15 border border-sky-500/30 text-sky-300"
                                >
                                    <span className="truncate max-w-[120px]">{kw}</span>
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            onRemoveSecondary?.(kw);
                                        }}
                                        className="opacity-0 group-hover:opacity-100 hover:text-white transition-opacity ml-0.5"
                                        title="Remove keyword"
                                    >
                                        <X className="w-2.5 h-2.5" />
                                    </button>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>

            <Button
                onClick={onSave}
                disabled={!primaryKeyword || saving}
                className={`flex-shrink-0 h-9 px-4 text-sm font-medium transition-all ${
                    saved
                        ? "bg-emerald-600 hover:bg-emerald-600 text-white"
                        : "bg-indigo-600 hover:bg-indigo-700 text-white"
                }`}
            >
                {saving ? (
                    <>
                        <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                        {autoSaveActive ? "Auto-saving..." : "Saving…"}
                    </>
                ) : saved ? (
                    <>
                        <CheckCircle2 className="w-3.5 h-3.5 mr-1.5" />
                        {autoSaveActive ? "Saved" : "Saved!"}
                    </>
                ) : (
                    <>
                        <Save className="w-3.5 h-3.5 mr-1.5" />
                        {autoSaveActive ? "Auto-save on" : "Save Selections"}
                    </>
                )}
            </Button>
        </div>
    );
}

// ─── Main Modal ───────────────────────────────────────────────────────────────

export interface KeywordIntelligenceModalProps {
    isOpen: boolean;
    onClose: () => void;
    idea: ContentIdea;
    /** Called after a successful save so the parent can update local state */
    onSaved?: (
        primary: string,
        secondary: string[],
        metrics: { volume: number | null; difficulty: number | null; cpc: number | null },
        rawOutput?: any
    ) => void;
    /**
     * Optional override for the save persistence call.
     * Receives the same args as onSaved. Return true on success.
     * If omitted, defaults to contentIdeasService.updateKeywordSelection.
     */
    onSave?: (
        primary: string,
        secondary: string[],
        metrics: { volume: number | null; difficulty: number | null; cpc: number | null },
        rawOutput?: any
    ) => Promise<boolean>;
    /**
     * Optional loading state - true when keyword data is being fetched/enriched.
     * Shows a loading spinner instead of "No data available" when true.
     */
    isLoading?: boolean;
    /**
     * Optional label for loading state (e.g., "Fetching keyword data...")
     */
    loadingLabel?: string;
}

export function KeywordIntelligenceModal({
    isOpen,
    onClose,
    idea,
    onSaved,
    onSave,
    isLoading = false,
    loadingLabel = "Loading keyword data...",
}: KeywordIntelligenceModalProps) {
    const { user } = useAuth();

    // Parse the DataForSEO output once, then allow additions via useState
    const baseParsed = React.useMemo(() => {
        const rawField =
            (idea as any).raw_dataforseo_output ??
            (idea as any).raw_supabase_output ??
            (idea as any).idea_metadata?.seo_offer_enrichment?.raw_dataforseo_output;
        const parsed = parseDataForSEOOutput(rawField) ?? buildSyntheticParsedFromIdea(idea);
        return mergeParsedWithIdeaKeywordPool(parsed, idea);
    }, [idea]);

    // Mutable copy of parsed so we can append expanded rows
    const [parsed, setParsed] = React.useState<DFSParsedOutput | null>(baseParsed);
    // Track which keywords came from Keyword Expander (show NEW badge)
    const [expandedKeywords, setExpandedKeywords] = React.useState<Set<string>>(new Set());

    // Keyword Expander panel state
    const [expanderOpen, setExpanderOpen] = React.useState(false);
    const [expanderSeed, setExpanderSeed] = React.useState("");
    const [expanderLoading, setExpanderLoading] = React.useState(false);
    const [expanderError, setExpanderError] = React.useState<string | null>(null);
    const [expanderAdded, setExpanderAdded] = React.useState(0);
    const [expanderQualifiedOnly, setExpanderQualifiedOnly] = React.useState(true);
    const [showQualifiedOnly, setShowQualifiedOnly] = React.useState(false);

    // Resolve initial selections from stored idea data
    const initialPrimary = React.useMemo(() => {
        const stored = (idea as any).primary_keywords ?? (idea as any).primary_keyword ?? (idea as any).keywords ?? [];
        const list = extractKeywordValues(stored);
        return list[0] ?? null;
    }, [idea.id]);

    const initialSecondary = React.useMemo(() => {
        const raw = (idea as any).secondary_keywords ?? (idea as any).secondary_keywords_json;
        const parsedSecondary = extractKeywordValues(raw);
        if (parsedSecondary.length > 0) return parsedSecondary;
        // Fall back to keywords[1..] if secondary_keywords is empty
        const stored = (idea as any).keywords ?? [];
        const list = extractKeywordValues(stored);
        return list.slice(1);
    }, [idea.id, (idea as any).secondary_keywords, (idea as any).secondary_keywords_json, (idea as any).keywords]);

    const [primaryKeyword, setPrimaryKeyword] = React.useState<string | null>(initialPrimary);
    const [secondaryKeywords, setSecondaryKeywords] = React.useState<string[]>(initialSecondary);
    const [expandedChart, setExpandedChart] = React.useState<string | null>(null);
    const [saving, setSaving] = React.useState(false);
    const [saved, setSaved] = React.useState(false);
    const [saveError, setSaveError] = React.useState<string | null>(null);
    const [lastAutoSavedPrimary, setLastAutoSavedPrimary] = React.useState<string | null>(null);
    const [lastAutoSavedSecondary, setLastAutoSavedSecondary] = React.useState<string[]>([]);
    const autoSaveTimeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);

    // Internal loading state - activates when modal opens but no data yet
    const [isDataLoading, setIsDataLoading] = React.useState(false);
    const dataLoadTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);

    // Show loading indicator when modal opens but data isn't available yet
    React.useEffect(() => {
        if (isOpen && !parsed && !isLoading) {
            // Start a timer - if data doesn't appear within 1.5 seconds, show loading
            dataLoadTimerRef.current = setTimeout(() => {
                setIsDataLoading(true);
            }, 1500);

            return () => {
                if (dataLoadTimerRef.current) {
                    clearTimeout(dataLoadTimerRef.current);
                }
            };
        } else {
            setIsDataLoading(false);
        }
    }, [isOpen, parsed, isLoading]);

    // Clear loading state when data arrives
    React.useEffect(() => {
        if (parsed && isDataLoading) {
            setIsDataLoading(false);
        }
    }, [parsed, isDataLoading]);

    // Auto-save when modal closes (user switches tabs or closes without manual save)
    React.useEffect(() => {
        if (!isOpen) return;

        const handleVisibilityChange = () => {
            if (document.hidden && primaryKeyword) {
                // User switched tabs - trigger immediate save without debounce
                console.log("[KeywordIntelligenceModal] Page hidden, auto-saving expanded keywords...");
                triggerAutoSave(primaryKeyword, secondaryKeywords);
            }
        };

        document.addEventListener("visibilitychange", handleVisibilityChange);
        return () => {
            document.removeEventListener("visibilitychange", handleVisibilityChange);
        };
    }, [isOpen, primaryKeyword, secondaryKeywords]);

    // Cleanup auto-save timeout on unmount
    React.useEffect(() => {
        return () => {
            if (autoSaveTimeoutRef.current) {
                clearTimeout(autoSaveTimeoutRef.current);
            }
            if (dataLoadTimerRef.current) {
                clearTimeout(dataLoadTimerRef.current);
            }
        };
    }, []);

    const baseParsedKeywordLookup = React.useMemo(() => {
        const map = new Map<string, string>();
        for (const row of baseParsed?.rows ?? []) {
            const key = normalizeKeywordKey(row.keyword);
            if (!key || map.has(key)) continue;
            map.set(key, row.keyword);
        }
        return map;
    }, [baseParsed]);

    const parsedKeywordLookup = React.useMemo(() => {
        const map = new Map<string, string>();
        for (const row of parsed?.rows ?? []) {
            const key = normalizeKeywordKey(row.keyword);
            if (!key || map.has(key)) continue;
            map.set(key, row.keyword);
        }
        return map;
    }, [parsed]);

    // Re-sync selections when idea changes (e.g. re-opened for different idea)
    React.useEffect(() => {
        if (isOpen) {
            const nextPrimaryRaw = initialPrimary ? String(initialPrimary).trim() : null;
            const canonicalPrimary = nextPrimaryRaw
                ? baseParsedKeywordLookup.get(normalizeKeywordKey(nextPrimaryRaw)) ?? nextPrimaryRaw
                : null;
            const canonicalSecondary = Array.from(
                new Set(
                    initialSecondary
                        .map((kw) => baseParsedKeywordLookup.get(normalizeKeywordKey(kw)) ?? null)
                        .filter((kw): kw is string => Boolean(kw))
                )
            ).filter((kw) => kw !== canonicalPrimary);

            setParsed(baseParsed);
            setExpandedKeywords(new Set());
            setExpanderOpen(false);
            setExpanderSeed(canonicalPrimary || "");
            setExpanderError(null);
            setExpanderAdded(0);
            setExpanderQualifiedOnly(true);
            setShowQualifiedOnly(false);
            setPrimaryKeyword(canonicalPrimary);
            setSecondaryKeywords(canonicalSecondary);
            setExpandedChart(null);
            setSaved(false);
            setSaveError(null);
            setLastAutoSavedPrimary(canonicalPrimary);
            setLastAutoSavedSecondary(canonicalSecondary);
        }
    }, [idea.id, isOpen, baseParsedKeywordLookup]);

    const persistKeywordSelection = React.useCallback(async (
        primary: string | null,
        secondary: string[],
        options?: { showSavedState?: boolean }
    ): Promise<boolean> => {
        if (!primary || !user) return false;

        const normalizedPrimaryKey = normalizeKeywordKey(primary);
        const normalizedPrimary = parsedKeywordLookup.get(normalizedPrimaryKey) ?? primary;
        const normalizedSecondary = Array.from(
            new Set(
                secondary
                    .map((kw) => parsedKeywordLookup.get(normalizeKeywordKey(kw)) ?? kw)
                    .filter(Boolean)
            )
        ).filter((kw) => kw !== normalizedPrimary);

        setSaving(true);
        setSaveError(null);
        try {
            const primaryRow = parsed?.rows.find((r) => r.keyword === normalizedPrimary);
            const metrics = {
                volume: primaryRow?.search_volume ?? null,
                difficulty: primaryRow?.keyword_difficulty ?? null,
                cpc: primaryRow?.cpc ?? null,
            };

            const ok = onSave
                ? await onSave(normalizedPrimary, normalizedSecondary, metrics, parsed)
                : await contentIdeasService.updateKeywordSelection(
                    idea.id,
                    user.id,
                    normalizedPrimary,
                    normalizedSecondary,
                    metrics,
                    parsed
                );

            if (ok) {
                setLastAutoSavedPrimary(normalizedPrimary);
                setLastAutoSavedSecondary(normalizedSecondary);
                if (options?.showSavedState !== false) {
                    setSaved(true);
                    setTimeout(() => setSaved(false), 2000);
                }
                onSaved?.(normalizedPrimary, normalizedSecondary, metrics, parsed);
                return true;
            }

            setSaveError("Save failed. Please try again.");
            return false;
        } catch (err) {
            console.error("[KeywordIntelligenceModal] Save failed:", err);
            setSaveError("Unexpected error saving selections.");
            return false;
        } finally {
            setSaving(false);
        }
    }, [user, parsed, parsedKeywordLookup, onSave, onSaved, idea.id]);

    const handleSelectPrimary = (keyword: string) => {
        const nextSecondary = secondaryKeywords.filter((k) => k !== keyword);
        setPrimaryKeyword(keyword);
        setSecondaryKeywords(nextSecondary);
        setSaved(false);
        triggerAutoSave(keyword, nextSecondary);
    };

    const handleToggleSecondary = (keyword: string) => {
        if (keyword === primaryKeyword) return;
        const newSecondary = secondaryKeywords.includes(keyword)
            ? secondaryKeywords.filter((k) => k !== keyword)
            : [...secondaryKeywords, keyword];
        setSecondaryKeywords(newSecondary);
        setSaved(false);
        triggerAutoSave(primaryKeyword, newSecondary);
    };

    const handleRemoveSecondary = (keyword: string) => {
        const newSecondary = secondaryKeywords.filter((k) => k !== keyword);
        setSecondaryKeywords(newSecondary);
        setSaved(false);
        triggerAutoSave(primaryKeyword, newSecondary);
    };

    /** Auto-save selections with debounce to avoid excessive saves */
    const triggerAutoSave = React.useCallback(async (primary: string | null, secondary: string[]) => {
        // Clear any pending auto-save
        if (autoSaveTimeoutRef.current) {
            clearTimeout(autoSaveTimeoutRef.current);
        }

        // Debounce auto-save by 1.5 seconds to avoid saving on every keystroke
        autoSaveTimeoutRef.current = setTimeout(async () => {
            if (!primary || !user) return;

            // Skip if nothing changed since last auto-save
            if (primary === lastAutoSavedPrimary &&
                JSON.stringify(secondary.sort()) === JSON.stringify([...lastAutoSavedSecondary].sort())) {
                return;
            }
            console.log("[KeywordIntelligenceModal] Auto-saving keyword selection...", {
                primary,
                secondaryCount: secondary.length,
            });
            await persistKeywordSelection(primary, secondary);
        }, 1500);
    }, [persistKeywordSelection, lastAutoSavedPrimary, lastAutoSavedSecondary]);

    const handleClose = React.useCallback(async () => {
        if (autoSaveTimeoutRef.current) {
            clearTimeout(autoSaveTimeoutRef.current);
            autoSaveTimeoutRef.current = null;
        }

        const sortedCurrentSecondary = [...secondaryKeywords].sort();
        const sortedSavedSecondary = [...lastAutoSavedSecondary].sort();
        const hasUnsavedChanges =
            primaryKeyword !== lastAutoSavedPrimary ||
            JSON.stringify(sortedCurrentSecondary) !== JSON.stringify(sortedSavedSecondary);

        if (hasUnsavedChanges && primaryKeyword && !saving) {
            await persistKeywordSelection(primaryKeyword, secondaryKeywords, { showSavedState: false });
        }

        onClose();
    }, [
        lastAutoSavedPrimary,
        lastAutoSavedSecondary,
        onClose,
        persistKeywordSelection,
        primaryKeyword,
        saving,
        secondaryKeywords,
    ]);

    const handleToggleChart = (keyword: string) => {
        setExpandedChart((prev) => (prev === keyword ? null : keyword));
    };

    const qualifiedRows = React.useMemo(() => {
        return (parsed?.rows || []).filter((row) => {
            const volume = Number(row.search_volume || 0);
            const kd = row.keyword_difficulty;
            return volume > 100 && kd !== null && kd < 35;
        });
    }, [parsed]);

    const rowsToRender = React.useMemo(() => {
        return showQualifiedOnly ? qualifiedRows : (parsed?.rows || []);
    }, [parsed, qualifiedRows, showQualifiedOnly]);

    const handleSave = async () => {
        if (!primaryKeyword || !user) return;
        console.log("[KeywordIntelligenceModal] Initiating save...", {
            ideaId: idea.id,
            primaryKeyword,
            secondaryCount: secondaryKeywords.length,
            totalRows: parsed?.rows.length
        });
        await persistKeywordSelection(primaryKeyword, secondaryKeywords);
    };

    /** Fetch related keywords for a custom seed and merge into the table */
    const handleExpandKeywords = async () => {
        const seed = expanderSeed.trim().toLowerCase();
        if (!seed) return;
        setExpanderLoading(true);
        setExpanderError(null);
        setExpanderAdded(0);
        try {
            const existingKeywords = (parsed?.rows ?? []).map((r) => r.keyword);
            let result = await contentIdeasService.fetchRelatedKeywords(
                seed,
                existingKeywords,
                20,
                expanderQualifiedOnly
                    ? { minSearchVolume: 100, maxKeywordDifficulty: 35 }
                    : undefined,
            );
            if ((!result.success || result.keywords.length === 0) && expanderQualifiedOnly) {
                result = await contentIdeasService.fetchRelatedKeywords(
                    seed,
                    existingKeywords,
                    20,
                    undefined,
                );
                if (result.success && result.keywords.length > 0) {
                    setExpanderError("No keywords met the strict filter, so showing the best available related keywords instead.");
                }
            }
            if (!result.success || result.keywords.length === 0) {
                setExpanderError("No new keywords found. Try a different seed.");
                return;
            }
            const normalizedSeedKeyword = String(result.seed_keyword || seed).trim();
            const returnedKeywords = Array.isArray(result.keywords) ? [...result.keywords] : [];
            const hasSeedInResults = returnedKeywords.some(
                (row) => normalizeKeywordKey(String(row?.keyword || "")) === normalizeKeywordKey(normalizedSeedKeyword)
            );
            if (!hasSeedInResults && normalizedSeedKeyword) {
                returnedKeywords.unshift({
                    keyword: normalizedSeedKeyword,
                    search_volume: null as any,
                    keyword_difficulty: null as any,
                    cpc: null as any,
                    opportunity: 0,
                    intent_label: null,
                    competition_level: null,
                });
            }

            // Merge metrics for existing keywords and add new ones
            const newRows: DFSKeywordRow[] = [];
            const updatedRows = parsed ? [...parsed.rows] : [];
            let metricsUpdated = false;

            returnedKeywords.forEach((k) => {
                const existingIdx = updatedRows.findIndex(
                    (existing) => normalizeKeywordKey(existing.keyword) === normalizeKeywordKey(k.keyword)
                );

                if (existingIdx >= 0) {
                    const existingRow = updatedRows[existingIdx];
                    const nextVol = k.search_volume !== null && k.search_volume !== undefined ? k.search_volume : existingRow.search_volume;
                    const nextKd = k.keyword_difficulty !== null && k.keyword_difficulty !== undefined ? k.keyword_difficulty : existingRow.keyword_difficulty;
                    const nextCpc = k.cpc !== null && k.cpc !== undefined ? k.cpc : existingRow.cpc;
                    const nextComp = (k as any).competition_level || existingRow.competition_level;
                    const nextIntent = (k as any).intent_label || existingRow.main_intent;

                    if (
                        nextVol !== existingRow.search_volume ||
                        nextKd !== existingRow.keyword_difficulty ||
                        nextCpc !== existingRow.cpc ||
                        nextComp !== existingRow.competition_level ||
                        nextIntent !== existingRow.main_intent
                    ) {
                        metricsUpdated = true;
                        updatedRows[existingIdx] = {
                            ...existingRow,
                            search_volume: nextVol,
                            keyword_difficulty: nextKd,
                            cpc: nextCpc,
                            competition_level: nextComp,
                            main_intent: nextIntent,
                        };
                    }
                } else {
                    newRows.push({
                        keyword: k.keyword,
                        type: normalizeKeywordKey(k.keyword) === normalizeKeywordKey(normalizedSeedKeyword) ? "seed" as const : "related" as const,
                        depth: normalizeKeywordKey(k.keyword) === normalizeKeywordKey(normalizedSeedKeyword) ? 0 : 2, // mark as custom-expanded
                        search_volume: k.search_volume ?? null,
                        competition: null,
                        competition_level: (k as any).competition_level ?? null,
                        cpc: k.cpc ?? null,
                        keyword_difficulty: k.keyword_difficulty ?? null,
                        main_intent: (k as any).intent_label ?? null,
                        foreign_intents: null,
                        monthly_searches: [],
                        search_volume_trend: null,
                        low_top_of_page_bid: null,
                        high_top_of_page_bid: null,
                        se_results_count: null,
                        related_keywords: null,
                    });
                }
            });

            if (newRows.length === 0 && !metricsUpdated) {
                setExpanderError("All returned keywords are already in the list with up-to-date metrics.");
                return;
            }

            const nextParsed: DFSParsedOutput = {
                seed_keyword: normalizedSeedKeyword || seed,
                total_count: updatedRows.length + newRows.length,
                items_count: updatedRows.length + newRows.length,
                rows: [...updatedRows, ...newRows],
            };

            if (nextParsed.seed_keyword !== normalizedSeedKeyword && normalizedSeedKeyword) {
                nextParsed.seed_keyword = normalizedSeedKeyword;
            }

            setParsed(nextParsed);
            setIsDataLoading(false);
            setExpandedKeywords((prev) => {
                const next = new Set(prev);
                newRows.forEach((r) => next.add(r.keyword));
                return next;
            });
            setExpanderAdded(newRows.length);
            setExpanderSeed(""); // clear after success

            if (!primaryKeyword && user) {
                const nextPrimary = newRows[0]?.keyword || null;
                if (nextPrimary) {
                    setPrimaryKeyword(nextPrimary);

                    const primaryRow = nextParsed.rows.find((r) => r.keyword === nextPrimary);
                    const metrics = {
                        volume: primaryRow?.search_volume ?? null,
                        difficulty: primaryRow?.keyword_difficulty ?? null,
                        cpc: primaryRow?.cpc ?? null,
                    };

                    const ok = onSave
                        ? await onSave(nextPrimary, secondaryKeywords, metrics, nextParsed)
                        : await contentIdeasService.updateKeywordSelection(
                            idea.id,
                            user.id,
                            nextPrimary,
                            secondaryKeywords,
                            metrics,
                            nextParsed
                        );

                    if (ok) {
                        setLastAutoSavedPrimary(nextPrimary);
                        setLastAutoSavedSecondary(secondaryKeywords);
                        setSaved(true);
                        onSaved?.(nextPrimary, secondaryKeywords, metrics, nextParsed);
                        setTimeout(() => setSaved(false), 2000);
                    }
                }
            } else {
                // Auto-save after successfully adding new keywords from DataForSEO.
                // This ensures expanded keywords are persisted even if user switches tabs or modal closes.
                triggerAutoSave(primaryKeyword, secondaryKeywords);
            }
        } catch (err) {
            setExpanderError("Unexpected error fetching keywords.");
        } finally {
            setExpanderLoading(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div
            className="fixed inset-0 bg-black/70 backdrop-blur-sm z-[60] flex items-center justify-center p-3"
            onClick={(e) => {
                if (e.target === e.currentTarget) void handleClose();
            }}
        >
            <motion.div
                initial={{ opacity: 0, scale: 0.96, y: 12 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.96, y: 12 }}
                transition={{ duration: 0.2 }}
                className="bg-slate-900 border border-white/10 rounded-2xl shadow-2xl w-full max-w-6xl max-h-[92vh] flex flex-col overflow-hidden"
            >
                {/* ── Header ── */}
                <div className="flex items-start justify-between px-6 py-4 border-b border-white/10 flex-shrink-0">
                    <div className="flex-1 min-w-0 pr-4">
                        <div className="flex items-center gap-2 mb-0.5">
                            <BarChart2 className="w-4 h-4 text-indigo-400 flex-shrink-0" />
                            <h2 className="text-base font-bold text-white truncate">Keyword Intelligence</h2>
                        </div>
                        <p className="text-xs text-slate-400 truncate">{idea.title}</p>
                        {parsed && (
                            <p className="text-[11px] text-indigo-300 mt-0.5">
                                Seed: <span className="font-medium">{parsed.seed_keyword}</span>
                            </p>
                        )}
                    </div>
                    {/* Expander toggle + close */}
                    <div className="flex items-center gap-2 flex-shrink-0">
                        <button
                            onClick={() => { setExpanderOpen((v) => !v); setExpanderError(null); setExpanderAdded(0); }}
                            title="Expand keywords with a new seed"
                            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-medium transition-all ${
                                expanderOpen
                                    ? "bg-indigo-500/20 border-indigo-500/40 text-indigo-300"
                                    : "border-white/10 text-slate-400 hover:text-white hover:border-white/20"
                            }`}
                        >
                            <Sparkles className="w-3.5 h-3.5" />
                            Expand Keywords
                        </button>
                        <button
                            onClick={() => void handleClose()}
                            className="text-slate-500 hover:text-white transition-colors p-1 -mr-1 rounded-lg hover:bg-white/5"
                        >
                            <X className="w-5 h-5" />
                        </button>
                    </div>
                </div>

                {/* ── Keyword Expander Panel ── */}
                <AnimatePresence>
                    {expanderOpen && (
                        <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: "auto", opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.18 }}
                            className="overflow-hidden flex-shrink-0"
                        >
                            <div className="px-6 py-3 bg-indigo-950/40 border-b border-indigo-500/20 flex flex-col sm:flex-row items-start sm:items-center gap-3">
                                <div className="flex-1 flex flex-col gap-1 min-w-0">
                                    <p className="text-[11px] text-slate-400">
                                        Enter a seed keyword to fetch new related keywords from DataForSEO and merge them into the list below.
                                    </p>
                                    <div className="flex items-center gap-2">
                                        <div className="relative flex-1 max-w-xs">
                                            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500 pointer-events-none" />
                                            <input
                                                type="text"
                                                value={expanderSeed}
                                                onChange={(e) => { setExpanderSeed(e.target.value); setExpanderError(null); setExpanderAdded(0); }}
                                                onKeyDown={(e) => e.key === "Enter" && !expanderLoading && handleExpandKeywords()}
                                                placeholder="e.g. mortgage calculator free"
                                                className="w-full pl-8 pr-3 py-1.5 text-xs bg-slate-800 border border-white/10 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500/50 focus:ring-1 focus:ring-indigo-500/30 transition-all"
                                                disabled={expanderLoading}
                                            />
                                        </div>
                                        <button
                                            onClick={handleExpandKeywords}
                                            disabled={!expanderSeed.trim() || expanderLoading}
                                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs font-medium transition-all"
                                        >
                                            {expanderLoading ? (
                                                <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Fetching…</>
                                            ) : (
                                                <><Plus className="w-3.5 h-3.5" /> Fetch Keywords</>
                                            )}
                                        </button>
                                        <label className="flex items-center gap-2 text-[11px] text-slate-400">
                                            <input
                                                type="checkbox"
                                                checked={expanderQualifiedOnly}
                                                onChange={(e) => setExpanderQualifiedOnly(e.target.checked)}
                                                className="h-3.5 w-3.5 accent-emerald-500"
                                            />
                                            Only volume &gt;100 and KD &lt;35
                                        </label>
                                    </div>
                                </div>
                                <div className="flex flex-col gap-1 min-w-[130px]">
                                    {expanderError && (
                                        <p className="text-[11px] text-red-400 flex items-center gap-1">
                                            <AlertTriangle className="w-3 h-3" /> {expanderError}
                                        </p>
                                    )}
                                    {expanderAdded > 0 && (
                                        <p className="text-[11px] text-emerald-400 flex items-center gap-1">
                                            <CheckCircle2 className="w-3 h-3" /> {expanderAdded} keywords added
                                        </p>
                                    )}
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* ── Loading state ── */}
                {(isLoading || isDataLoading) && (
                    <div className="flex-1 flex flex-col items-center justify-center gap-4 p-12 text-center">
                        <Loader2 className="w-12 h-12 text-indigo-400 animate-spin" />
                        <div>
                            <p className="text-slate-300 font-medium mb-1">{loadingLabel}</p>
                            <p className="text-xs text-slate-500 max-w-sm">
                                Fetching keyword metrics from DataForSEO...
                            </p>
                        </div>
                    </div>
                )}

                {/* ── No data state ── */}
                {!isLoading && !isDataLoading && !parsed ? (
                    <div className="flex-1 flex flex-col items-center justify-center gap-4 p-12 text-center">
                        <AlertTriangle className="w-12 h-12 text-amber-500/60" />
                        <div>
                            <p className="text-slate-300 font-medium mb-1">No keyword data available</p>
                            <p className="text-xs text-slate-500 max-w-sm">
                                Raw DataForSEO output is missing for this idea. Run{" "}
                                <span className="text-sky-300">SEO/Offers enrichment</span> to populate keyword
                                data.
                            </p>
                        </div>
                        <button
                            onClick={() => void handleClose()}
                            className="flex items-center gap-1.5 px-4 py-2 rounded-xl border border-white/10 text-slate-300 hover:text-white hover:border-white/20 text-sm transition-colors"
                        >
                            <RefreshCw className="w-3.5 h-3.5" />
                            Close &amp; Re-enrich
                        </button>
                    </div>
                ) : (
                    <>
                        {/* ── Summary bar ── */}
                        {parsed && <SummaryBar data={parsed} />}

                        {parsed && (
                            <div className="border-b border-white/10 bg-slate-900/70 px-6 py-3">
                                <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                                    <div className="flex flex-wrap items-center gap-2">
                                        <button
                                            type="button"
                                            onClick={() => setShowQualifiedOnly(false)}
                                            className={`rounded-full border px-3 py-1 text-[11px] transition ${
                                                !showQualifiedOnly
                                                    ? "border-indigo-500/30 bg-indigo-500/15 text-indigo-300"
                                                    : "border-white/10 text-slate-400 hover:text-white"
                                            }`}
                                        >
                                            All Keywords ({parsed.rows.length})
                                        </button>
                                        <button
                                            type="button"
                                            onClick={() => setShowQualifiedOnly(true)}
                                            className={`rounded-full border px-3 py-1 text-[11px] transition ${
                                                showQualifiedOnly
                                                    ? "border-emerald-500/30 bg-emerald-500/15 text-emerald-300"
                                                    : "border-white/10 text-slate-400 hover:text-white"
                                            }`}
                                        >
                                            Qualified Only ({qualifiedRows.length})
                                        </button>
                                    </div>
                                    <p className="text-[11px] text-slate-500">
                                        Qualified keywords use volume &gt;100 and KD &lt;35.
                                    </p>
                                </div>
                            </div>
                        )}

                        {/* ── Column legend ── */}
                        <div className="flex-1 overflow-auto">
                            <table className="w-full text-xs border-collapse">
                                <thead className="sticky top-0 bg-slate-900 border-b border-white/10 z-10">
                                    <tr>
                                        <th className="px-3 py-2.5 w-10">
                                            <abbr title="Primary keyword">
                                                <Star className="w-3 h-3 text-emerald-400 mx-auto" />
                                            </abbr>
                                        </th>
                                        <th className="px-2 py-2.5 w-10">
                                            <abbr title="Secondary keyword">
                                                <Target className="w-3 h-3 text-sky-400 mx-auto" />
                                            </abbr>
                                        </th>
                                        <th className="px-3 py-2.5 text-left text-slate-400 font-medium">Keyword</th>
                                        <th className="px-3 py-2.5 text-right text-slate-400 font-medium whitespace-nowrap">
                                            Volume
                                        </th>
                                        <th className="px-3 py-2.5 text-right text-slate-400 font-medium">KD</th>
                                        <th className="px-3 py-2.5 text-right text-slate-400 font-medium">CPC</th>
                                        <th className="px-3 py-2.5 text-center text-slate-400 font-medium whitespace-nowrap">
                                            Competition
                                        </th>
                                        <th className="px-3 py-2.5 text-center text-slate-400 font-medium">Intent</th>
                                        <th className="px-3 py-2.5 text-right text-slate-400 font-medium whitespace-nowrap">
                                            MoM%
                                        </th>
                                        <th className="px-2 py-2.5 w-10 text-center text-slate-400 font-medium">
                                            <BarChart2 className="w-3 h-3 mx-auto" />
                                        </th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {rowsToRender.map((row) => (
                                        <KeywordTableRow
                                            key={row.keyword}
                                            row={row}
                                            isPrimary={primaryKeyword === row.keyword}
                                            isSecondary={secondaryKeywords.includes(row.keyword)}
                                            onSelectPrimary={() => handleSelectPrimary(row.keyword)}
                                            onToggleSecondary={() => handleToggleSecondary(row.keyword)}
                                            showChart={expandedChart === row.keyword}
                                            onToggleChart={() => handleToggleChart(row.keyword)}
                                            disabled={saving}
                                            isNew={expandedKeywords.has(row.keyword)}
                                        />
                                    ))}
                                </tbody>
                            </table>
                        </div>

                        {/* ── Save error ── */}
                        {saveError && (
                            <div className="mx-4 mb-2 px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/20 text-xs text-red-300 flex-shrink-0">
                                {saveError}
                            </div>
                        )}

                        {/* ── Legend ── */}
                        <div className="px-6 py-2 flex flex-wrap items-center gap-3 border-t border-white/5 flex-shrink-0 bg-slate-900/50">
                            <span className="text-[10px] text-slate-500">
                                <Star className="w-2.5 h-2.5 text-emerald-400 inline mr-0.5" />
                                Primary — radio (select one)
                            </span>
                            <span className="text-[10px] text-slate-500">
                                <Target className="w-2.5 h-2.5 text-sky-400 inline mr-0.5" />
                                Secondary — multi-select
                            </span>
                            <span className="text-[10px] text-slate-500">
                                <BarChart2 className="w-2.5 h-2.5 text-indigo-400 inline mr-0.5" />
                                12-month trend chart
                            </span>
                            <span className="text-[10px] text-slate-500">KD = Keyword Difficulty (0–100)</span>
                            <span className="text-[10px] text-slate-500">MoM% = monthly search trend</span>
                        </div>

                        {/* ── Footer ── */}
                        <Footer
                            primaryKeyword={primaryKeyword}
                            secondaryKeywords={secondaryKeywords}
                            saving={saving}
                            saved={saved}
                            onSave={handleSave}
                            autoSaveActive={Boolean(primaryKeyword)}
                            onRemoveSecondary={handleRemoveSecondary}
                        />
                    </>
                )}
            </motion.div>
        </div>
    );
}
