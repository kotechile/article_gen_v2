import * as React from "react";
import { motion } from "framer-motion";
import { X, Sparkles, Lightbulb, Loader2, Check, Save, BookOpen, Code, Info } from "lucide-react";
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
}

export function IdeaBurstModal({ isOpen, onClose, subtopic, topicId, topicTitle: _topicTitle }: IdeaBurstModalProps) {
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

    React.useEffect(() => {
        if (isOpen && subtopic && user) {
            generateIdeas();
        }
    }, [isOpen, subtopic, user]);

    const generateIdeas = async () => {
        if (!subtopic || !user) return;

        setLoading(true);
        setError(null);
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
                userId: user.id
            });

            setBlogIdeas(result.blog_ideas || []);
            setSoftwareIdeas(result.software_ideas || []);
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
                const firstIdea = blogIdeas.find(i => selectedBlogIdeas.has(i.id));
                if (firstIdea) {
                    navigate(`/content-studio?id=${firstIdea.id}`);
                }
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

                {/* Content */}
                <div className="flex-1 overflow-auto p-6">
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
}

function BlogIdeaCard({ idea, isSelected, onToggle }: BlogIdeaCardProps) {
    const keywords = idea.primary_keywords || idea.keywords || [];

    return (
        <motion.div
            layout
            onClick={onToggle}
            className={`p-4 rounded-xl border cursor-pointer transition-all duration-200 ${
                isSelected
                    ? 'bg-indigo-500/10 border-indigo-500/50'
                    : 'bg-white/5 border-white/10 hover:border-white/20 hover:bg-white/8'
            }`}
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
                    </div>

                    {/* Affiliate Hook - Full width */}
                    {idea.monetization_hook && (
                        <div className="mt-2 pt-2 border-t border-white/5">
                            <p className="text-[11px] text-amber-400/80">
                                <span className="text-amber-500 font-medium">💰 Monetization:</span> {idea.monetization_hook}
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </motion.div>
    );
}

// Software Idea Card Component
interface SoftwareIdeaCardProps {
    idea: ContentIdea;
    isSelected: boolean;
    onToggle: () => void;
}

function SoftwareIdeaCard({ idea, isSelected, onToggle }: SoftwareIdeaCardProps) {
    const keywords = idea.primary_keywords || idea.keywords || [];

    return (
        <motion.div
            layout
            onClick={onToggle}
            className={`p-4 rounded-xl border cursor-pointer transition-all duration-200 ${
                isSelected
                    ? 'bg-amber-500/10 border-amber-500/50'
                    : 'bg-white/5 border-white/10 hover:border-white/20 hover:bg-white/8'
            }`}
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
                    </div>

                    {/* Monetization Strategy */}
                    {idea.monetization_hook && (
                        <div className="mt-2 pt-2 border-t border-white/5">
                            <p className="text-[11px] text-emerald-400/80">
                                <span className="text-emerald-500 font-medium">💡 Revenue Model:</span> {idea.monetization_hook}
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </motion.div>
    );
}
