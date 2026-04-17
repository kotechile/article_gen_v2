import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Sparkles, Lightbulb, Loader2, Check, Save, BookOpen, Code, Info, BarChart3, ChevronDown, ChevronUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { contentIdeasService } from "@/services/content-ideas.service";
import type { ContentIdea } from "@/types/idea-burst";
import type { Subtopic } from "@/types/research";
import { useNavigate } from "react-router-dom";
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

export function IdeaBurstModal({ isOpen, onClose, subtopic, topicId, topicTitle, projectName, categoryPath }: IdeaBurstModalProps) {
    const navigate = useNavigate();
    const { user } = useAuth();
    const [loading, setLoading] = React.useState(false);
    const [error, setError] = React.useState<string | null>(null);
    const [blogIdeas, setBlogIdeas] = React.useState<ContentIdea[]>([]);
    const [softwareIdeas, setSoftwareIdeas] = React.useState<ContentIdea[]>([]);
    const [selectedBlogIdeas, setSelectedBlogIdeas] = React.useState<Set<string>>(new Set());
    const [selectedSoftwareIdeas, setSelectedSoftwareIdeas] = React.useState<Set<string>>(new Set());
    const [publishing, setPublishing] = React.useState(false);
    const [savingSoftware, setSavingSoftware] = React.useState(false);
    const [published, setPublished] = React.useState(false);
    const [saved, setSaved] = React.useState(false);
    const [expandedMetrics, setExpandedMetrics] = React.useState<string | null>(null);
    const [loadedFromCache, setLoadedFromCache] = React.useState(false);
    const lastGeneratedKeyRef = React.useRef<string | null>(null);

    const cacheKey = React.useMemo(() => {
        if (!subtopic || !user) return null;
        return `ideaBurstCache:${topicId}:${subtopic.id}:${user.id}`;
    }, [topicId, subtopic?.id, user?.id]);

    React.useEffect(() => {
        if (!isOpen || !subtopic || !user) return;
        const generationKey = `${topicId}:${subtopic.id}:${user.id}`;
        if (lastGeneratedKeyRef.current === generationKey) return;

        if (cacheKey) {
            try {
                const raw = localStorage.getItem(cacheKey);
                if (raw) {
                    const parsed = JSON.parse(raw) as CachedIdeaBurst;
                    if (Array.isArray(parsed.blogIdeas) || Array.isArray(parsed.softwareIdeas)) {
                        setBlogIdeas(Array.isArray(parsed.blogIdeas) ? parsed.blogIdeas : []);
                        setSoftwareIdeas(Array.isArray(parsed.softwareIdeas) ? parsed.softwareIdeas : []);
                        setError(null);
                        setLoadedFromCache(true);
                        lastGeneratedKeyRef.current = generationKey;
                        return;
                    }
                }
            } catch (e) {
                console.warn("Failed to read idea burst cache:", e);
            }
        }

        lastGeneratedKeyRef.current = generationKey;
        generateIdeas();
    }, [isOpen, topicId, subtopic?.id, user?.id]);

    React.useEffect(() => {
        if (!isOpen) {
            lastGeneratedKeyRef.current = null;
        }
    }, [isOpen]);

    const safeValueLayerTags = React.useMemo(() => {
        if (!subtopic) return [] as string[];
        const raw = (subtopic as any).value_layer_tags;
        return Array.isArray(raw) ? raw.filter(Boolean) : [];
    }, [subtopic]);

    const subtopicKeywordMetrics = React.useMemo(() => {
        const map = new Map<string, KeywordMetricRow>();
        if (!subtopic) return map;
        const rawKeywords = (subtopic as any).keywords;
        if (!Array.isArray(rawKeywords)) return map;

        rawKeywords.forEach((entry: any) => {
            if (!entry || typeof entry === "string") return;
            const keyword = String(entry.keyword || entry.term || "").trim();
            if (!keyword) return;
            map.set(keyword, {
                keyword,
                search_volume: typeof entry.search_volume === "number" ? entry.search_volume : null,
                keyword_difficulty:
                    typeof entry.keyword_difficulty === "number"
                        ? entry.keyword_difficulty
                        : (typeof entry.difficulty === "number" ? entry.difficulty : null),
                cpc: typeof entry.cpc === "number" ? entry.cpc : null,
            });
        });
        return map;
    }, [subtopic]);

    const generateIdeas = async () => {
        if (!subtopic || !user) return;

        setLoading(true);
        setError(null);
        setLoadedFromCache(false);
        setBlogIdeas([]);
        setSoftwareIdeas([]);
        setSelectedBlogIdeas(new Set());
        setSelectedSoftwareIdeas(new Set());
        setPublished(false);
        setSaved(false);

        try {
            const keywords = subtopic.keywords || [];
            const keywordStrings = Array.isArray(keywords)
                ? keywords.map((k: any) => typeof k === 'string' ? k : k.keyword || '').filter(Boolean)
                : [];

            const monetizationData = subtopic.monetization_data || {};
            const affiliateOffers = monetizationData.details?.affiliate_categories || [];

            const result = await contentIdeasService.generateBurst({
                topicId,
                subtopicName: subtopic.name,
                keywords: keywordStrings,
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

            if (cacheKey) {
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
            await contentIdeasService.publishContentIdeas(ideaIds, user.id);
            setPublished(true);
            setTimeout(() => {
                onClose();
                // Avoid opening Content Studio with a content_ideas id (can cause blank screen).
                // Published items are now discoverable from Content Library.
                navigate('/my-articles');
            }, 1500);
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
            await contentIdeasService.publishContentIdeas(ideaIds, user.id);
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

    const handleClearCachedIdeas = () => {
        if (!cacheKey) return;
        try {
            localStorage.removeItem(cacheKey);
        } catch (e) {
            console.warn("Failed to clear cached idea burst:", e);
        }
        setLoadedFromCache(false);
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
                        {loadedFromCache ? "Loaded previously generated candidates" : "Candidates are generated for this subtopic"}
                    </div>
                    <div className="flex items-center gap-2">
                        {cacheKey && (
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
                                            />
                                        ))}
                                    </div>

                                    {/* Blog Actions */}
                                    <div className="pt-4 border-t border-white/10">
                                        <Button
                                            onClick={handlePublishBlogs}
                                            disabled={selectedBlogIdeas.size === 0 || publishing}
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
                                            onClick={handleSaveSoftware}
                                            disabled={selectedSoftwareIdeas.size === 0 || savingSoftware}
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
}

function BlogIdeaCard({ idea, isSelected, onToggle, isExpanded, onToggleMetrics, mapContext, keywordMetricsMap }: BlogIdeaCardProps) {
    const keywords = idea.primary_keywords || idea.keywords || [];
    const rankFactors = getRankFactors(idea);

    // Calculate per-keyword metrics (distribute aggregate values)
    const keywordCount = keywords.length || 1;
    const volumePerKeyword = idea.total_search_volume > 0 ? Math.round(idea.total_search_volume / keywordCount) : 0;
    const hasAnyRealKeywordMetrics = keywords.some((kw: string) => keywordMetricsMap.has(kw));

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
                        <h4 className={`font-medium text-sm mb-1 ${isSelected ? 'text-indigo-300' : 'text-white'}`}>
                            {idea.title}
                        </h4>
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
                            {idea.total_search_volume > 0 && (
                                <span className="flex items-center gap-1">
                                    <span className="text-blue-400">Vol:</span>
                                    <span className="text-slate-300">{idea.total_search_volume.toLocaleString()}</span>
                                </span>
                            )}
                            {idea.average_difficulty > 0 && (
                                <span className="flex items-center gap-1">
                                    <span className={idea.average_difficulty > 60 ? 'text-red-400' : idea.average_difficulty > 30 ? 'text-yellow-400' : 'text-green-400'}>
                                        KD:
                                    </span>
                                    <span className="text-slate-300">{Math.round(idea.average_difficulty)}</span>
                                </span>
                            )}
                            {idea.average_cpc > 0 && (
                                <span className="flex items-center gap-1">
                                    <span className="text-emerald-400">CPC:</span>
                                    <span className="text-slate-300">${idea.average_cpc.toFixed(2)}</span>
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

            {/* Expandable Metrics Section */}
            {keywords.length > 0 && (
                <div className="border-t border-white/5">
                    <button
                        onClick={(e) => {
                            e.stopPropagation();
                            onToggleMetrics();
                        }}
                        className="w-full px-4 py-2 flex items-center justify-center gap-2 text-[11px] text-indigo-400 hover:text-indigo-300 hover:bg-indigo-500/5 transition-colors"
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
                                                        const row = keywordMetricsMap.get(kw);
                                                        const rowVolume = row?.search_volume ?? (volumePerKeyword > 0 ? volumePerKeyword : null);
                                                        const rowKD = row?.keyword_difficulty ?? (idea.average_difficulty > 0 ? Math.round(idea.average_difficulty) : null);
                                                        const rowCPC = row?.cpc ?? (idea.average_cpc > 0 ? Number(idea.average_cpc.toFixed(2)) : null);
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
                                            ? "Note: Keyword metrics use available keyword-level data, with estimated fallback where missing"
                                            : "Note: Individual keyword metrics are estimated based on aggregate data"}
                                    </p>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            )}
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
    const keywords = idea.primary_keywords || idea.keywords || [];
    const rankFactors = getRankFactors(idea);

    // Calculate per-keyword metrics (distribute aggregate values)
    const keywordCount = keywords.length || 1;
    const volumePerKeyword = idea.total_search_volume > 0 ? Math.round(idea.total_search_volume / keywordCount) : 0;
    const hasAnyRealKeywordMetrics = keywords.some((kw: string) => keywordMetricsMap.has(kw));

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
                        <h4 className={`font-medium text-sm mb-1 ${isSelected ? 'text-amber-300' : 'text-white'}`}>
                            {idea.title}
                        </h4>
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
                            {idea.total_search_volume > 0 && (
                                <span className="flex items-center gap-1">
                                    <span className="text-blue-400">Demand:</span>
                                    <span className="text-slate-300">{idea.total_search_volume.toLocaleString()}/mo</span>
                                </span>
                            )}
                            {idea.average_difficulty > 0 && (
                                <span className="flex items-center gap-1">
                                    <span className={idea.average_difficulty > 60 ? 'text-red-400' : idea.average_difficulty > 30 ? 'text-yellow-400' : 'text-green-400'}>
                                        KD:
                                    </span>
                                    <span className="text-slate-300">{Math.round(idea.average_difficulty)}</span>
                                </span>
                            )}
                            {idea.average_cpc > 0 && (
                                <span className="flex items-center gap-1">
                                    <span className="text-emerald-400">CPC:</span>
                                    <span className="text-slate-300">${idea.average_cpc.toFixed(2)}</span>
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
                                                        const row = keywordMetricsMap.get(kw);
                                                        const rowVolume = row?.search_volume ?? (volumePerKeyword > 0 ? volumePerKeyword : null);
                                                        const rowKD = row?.keyword_difficulty ?? (idea.average_difficulty > 0 ? Math.round(idea.average_difficulty) : null);
                                                        const rowCPC = row?.cpc ?? (idea.average_cpc > 0 ? Number(idea.average_cpc.toFixed(2)) : null);
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
                                            ? "Note: Keyword metrics use available keyword-level data, with estimated fallback where missing"
                                            : "Note: Individual keyword metrics are estimated based on aggregate data"}
                                    </p>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            )}
        </motion.div>
    );
}
