import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Sparkles, FileText, Lightbulb, Loader2, Check, ArrowRight, Save } from "lucide-react";
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

export function IdeaBurstModal({ isOpen, onClose, subtopic, topicId, topicTitle }: IdeaBurstModalProps) {
    const navigate = useNavigate();
    const { user } = useAuth();
    const [loading, setLoading] = React.useState(false);
    const [error, setError] = React.useState<string | null>(null);
    const [blogIdeas, setBlogIdeas] = React.useState<ContentIdea[]>([]);
    const [softwareIdeas, setSoftwareIdeas] = React.useState<ContentIdea[]>([]);
    const [selectedIdeas, setSelectedIdeas] = React.useState<Set<string>>(new Set());
    const [publishing, setPublishing] = React.useState(false);
    const [published, setPublished] = React.useState(false);

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
        setSelectedIdeas(new Set());
        setPublished(false);

        try {
            // Get keywords from the subtopic
            const keywords = subtopic.keywords || [];
            const keywordStrings = Array.isArray(keywords)
                ? keywords.map((k: any) => typeof k === 'string' ? k : k.keyword || '').filter(Boolean)
                : [];

            // Get monetization data for affiliate offers
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

    const toggleIdeaSelection = (ideaId: string) => {
        setSelectedIdeas(prev => {
            const newSet = new Set(prev);
            if (newSet.has(ideaId)) {
                newSet.delete(ideaId);
            } else {
                newSet.add(ideaId);
            }
            return newSet;
        });
    };

    const handlePublish = async () => {
        if (!user || selectedIdeas.size === 0) return;

        setPublishing(true);
        try {
            const ideaIds = Array.from(selectedIdeas);
            await contentIdeasService.publishContentIdeas(ideaIds, user.id);
            setPublished(true);
            setTimeout(() => {
                onClose();
                // Navigate to content studio with the first selected idea
                const firstIdea = blogIdeas.find(i => selectedIdeas.has(i.id)) ||
                                  softwareIdeas.find(i => selectedIdeas.has(i.id));
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

    const selectAll = () => {
        const allIds = [...blogIdeas, ...softwareIdeas].map(i => i.id);
        setSelectedIdeas(new Set(allIds));
    };

    const deselectAll = () => {
        setSelectedIdeas(new Set());
    };

    if (!isOpen || !subtopic) return null;

    const totalIdeas = blogIdeas.length + softwareIdeas.length;

    return (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="bg-slate-900 border border-white/10 rounded-2xl shadow-2xl max-w-4xl w-full max-h-[90vh] flex flex-col"
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

                    {!loading && !error && totalIdeas === 0 && (
                        <div className="py-16 text-center">
                            <Lightbulb className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                            <p className="text-slate-400">No ideas generated.</p>
                            <Button onClick={generateIdeas} variant="outline" className="mt-4 border-white/10">
                                Regenerate
                            </Button>
                        </div>
                    )}

                    {!loading && !error && totalIdeas > 0 && (
                        <div className="space-y-6">
                            {/* Selection Controls */}
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <span className="text-sm text-slate-400">
                                        {selectedIdeas.size} of {totalIdeas} selected
                                    </span>
                                </div>
                                <div className="flex gap-2">
                                    <Button
                                        onClick={selectAll}
                                        variant="ghost"
                                        size="sm"
                                        className="text-xs text-slate-400 hover:text-white"
                                    >
                                        Select All
                                    </Button>
                                    <Button
                                        onClick={deselectAll}
                                        variant="ghost"
                                        size="sm"
                                        className="text-xs text-slate-400 hover:text-white"
                                    >
                                        Deselect All
                                    </Button>
                                </div>
                            </div>

                            {/* Blog Ideas Section */}
                            {blogIdeas.length > 0 && (
                                <div>
                                    <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                                        <FileText className="w-4 h-4 text-blue-400" />
                                        Blog Articles ({blogIdeas.length})
                                    </h3>
                                    <div className="space-y-3">
                                        {blogIdeas.map((idea) => (
                                            <IdeaCard
                                                key={idea.id}
                                                idea={idea}
                                                isSelected={selectedIdeas.has(idea.id)}
                                                onToggle={() => toggleIdeaSelection(idea.id)}
                                            />
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Software/Commercial Ideas Section */}
                            {softwareIdeas.length > 0 && (
                                <div>
                                    <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                                        <Lightbulb className="w-4 h-4 text-amber-400" />
                                        Software & Commercial ({softwareIdeas.length})
                                    </h3>
                                    <div className="space-y-3">
                                        {softwareIdeas.map((idea) => (
                                            <IdeaCard
                                                key={idea.id}
                                                idea={idea}
                                                isSelected={selectedIdeas.has(idea.id)}
                                                onToggle={() => toggleIdeaSelection(idea.id)}
                                            />
                                        ))}
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
                            <p className="text-green-400 font-medium">Ideas published successfully!</p>
                            <p className="text-xs text-slate-500 mt-1">Redirecting to Content Studio...</p>
                        </motion.div>
                    )}
                </div>

                {/* Footer */}
                {!loading && !error && totalIdeas > 0 && (
                    <div className="border-t border-white/10 p-6 flex items-center justify-between">
                        <div className="text-xs text-slate-500">
                            {subtopic.search_volume?.toLocaleString() || 0} monthly searches
                            {subtopic.cpc && ` • $${subtopic.cpc.toFixed(2)} CPC`}
                        </div>
                        <div className="flex gap-3">
                            <Button
                                onClick={onClose}
                                variant="ghost"
                                className="text-slate-400 hover:text-white"
                            >
                                Close
                            </Button>
                            <Button
                                onClick={handlePublish}
                                disabled={selectedIdeas.size === 0 || publishing}
                                className="bg-indigo-600 hover:bg-indigo-700 text-white"
                            >
                                {publishing ? (
                                    <>
                                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                        Publishing...
                                    </>
                                ) : (
                                    <>
                                        <Save className="w-4 h-4 mr-2" />
                                        Publish {selectedIdeas.size > 0 && `(${selectedIdeas.size})`}
                                    </>
                                )}
                            </Button>
                        </div>
                    </div>
                )}
            </motion.div>
        </div>
    );
}

// Idea Card Component
interface IdeaCardProps {
    idea: ContentIdea;
    isSelected: boolean;
    onToggle: () => void;
}

function IdeaCard({ idea, isSelected, onToggle }: IdeaCardProps) {
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

                    {/* Metrics */}
                    <div className="flex items-center gap-4 text-[11px] text-slate-500">
                        {idea.total_search_volume > 0 && (
                            <span className="flex items-center gap-1">
                                <span className="text-blue-400">Vol:</span>
                                {idea.total_search_volume.toLocaleString()}
                            </span>
                        )}
                        {idea.average_difficulty > 0 && (
                            <span className="flex items-center gap-1">
                                <span className={idea.average_difficulty > 60 ? 'text-red-400' : idea.average_difficulty > 30 ? 'text-yellow-400' : 'text-green-400'}>
                                    KD:
                                </span>
                                {Math.round(idea.average_difficulty)}
                            </span>
                        )}
                        {idea.viability_score > 0 && (
                            <span className="flex items-center gap-1">
                                <span className="text-indigo-400">Viability:</span>
                                {Math.round(idea.viability_score)}%
                            </span>
                        )}
                        {idea.monetization_hook && (
                            <span className="text-amber-400 truncate max-w-[150px]">
                                {idea.monetization_hook}
                            </span>
                        )}
                    </div>
                </div>
                <ArrowRight className={`w-4 h-4 flex-shrink-0 transition-colors ${
                    isSelected ? 'text-indigo-400' : 'text-slate-700'
                }`} />
            </div>
        </motion.div>
    );
}
