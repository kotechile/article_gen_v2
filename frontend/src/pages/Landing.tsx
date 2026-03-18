import { useRef, useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import {
    Sparkles, Newspaper, PenLine, BookOpen,
    Search, ArrowRight, Loader2, Zap, Lock, Clock, FlaskConical
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useProject } from '@/context/project-context';
import { useAuth } from '@/context/auth-context';
import { ProjectSwitcher } from '@/components/layout/ProjectSwitcher';
import { TrendReportModal } from '@/components/TrendReportModal';
import { researchTopicsService } from '@/services/research-topics.service';
import type { ResearchTopic } from '@/types/research';
import { apiClient } from '@/api-client';

// ─── Propose-Topics Modal ─────────────────────────────────────────────────────
interface ProposedTopic {
    title: string;
    rationale: string;
}

function ProposeTopicsModal({ isOpen, onClose, topics, loading, error }: {
    isOpen: boolean;
    onClose: () => void;
    topics: ProposedTopic[];
    loading: boolean;
    error: string | null;
}) {
    const navigate = useNavigate();
    const { activeProject } = useProject();

    const handleTopicClick = async (title: string) => {
        try {
            const newTopic = await researchTopicsService.createResearchTopic({
                title,
                description: `AI-proposed topic for ${activeProject?.domain || activeProject?.app_name}: ${title}`
            });
            onClose();
            navigate(`/research/${newTopic.id}`);
        } catch (e) {
            console.error(e);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="bg-slate-900 border border-white/10 rounded-2xl shadow-2xl max-w-lg w-full p-6"
            >
                <div className="flex items-center gap-3 mb-6">
                    <div className="p-2 bg-indigo-500/20 rounded-xl">
                        <Sparkles className="w-5 h-5 text-indigo-400" />
                    </div>
                    <div>
                        <h2 className="text-lg font-bold text-white">AI-Proposed Topics</h2>
                        <p className="text-xs text-slate-500">Based on your active niche</p>
                    </div>
                    <button onClick={onClose} className="ml-auto text-slate-500 hover:text-white transition-colors text-xl leading-none">×</button>
                </div>

                {loading && (
                    <div className="py-12 flex flex-col items-center gap-4">
                        <Loader2 className="w-10 h-10 text-indigo-400 animate-spin" />
                        <p className="text-slate-400">Generating niche-specific topics...</p>
                    </div>
                )}
                {error && <p className="text-red-400 text-center py-8">{error}</p>}
                {!loading && !error && topics.length > 0 && (
                    <div className="space-y-3">
                        {topics.map((t, i) => (
                            <button
                                key={i}
                                onClick={() => handleTopicClick(t.title)}
                                className="w-full text-left p-4 rounded-xl bg-white/5 border border-white/8 hover:border-indigo-500/50 hover:bg-indigo-500/10 transition-all group"
                            >
                                <div className="flex items-start gap-3">
                                    <span className="flex-shrink-0 w-6 h-6 rounded-lg bg-indigo-500/20 text-indigo-400 text-xs font-bold flex items-center justify-center mt-0.5">
                                        {i + 1}
                                    </span>
                                    <div>
                                        <p className="font-semibold text-white group-hover:text-indigo-300 transition-colors">{t.title}</p>
                                        {t.rationale && <p className="text-xs text-slate-500 mt-1 leading-relaxed">{t.rationale}</p>}
                                    </div>
                                    <ArrowRight className="w-4 h-4 text-slate-600 group-hover:text-indigo-400 ml-auto flex-shrink-0 transition-colors mt-1" />
                                </div>
                            </button>
                        ))}
                        <p className="text-[11px] text-slate-600 text-center pt-2">Click a topic to start a full research session</p>
                    </div>
                )}

                {/* Cancel button — always visible */}
                <div className="mt-5 pt-4 border-t border-white/8">
                    <button
                        id="propose-topics-cancel"
                        onClick={onClose}
                        className="w-full py-2.5 rounded-xl text-sm font-medium text-slate-400 hover:text-white bg-white/5 hover:bg-white/10 border border-white/8 hover:border-white/15 transition-all"
                    >
                        Cancel
                    </button>
                </div>
            </motion.div>
        </div>
    );
}

// ─── Main Landing ─────────────────────────────────────────────────────────────
export function Landing() {
    const navigate = useNavigate();
    const { activeProject, isLoading: projectLoading } = useProject();
    const searchRef = useRef<HTMLInputElement>(null);
    const [searchTerm, setSearchTerm] = useState('');
    const [isExploding, setIsExploding] = useState(false);

    // Propose Topics state
    const [proposeOpen, setProposeOpen] = useState(false);
    const [proposedTopics, setProposedTopics] = useState<ProposedTopic[]>([]);
    const [proposeLoading, setProposeLoading] = useState(false);
    const [proposeError, setProposeError] = useState<string | null>(null);

    // News Pulse state
    const [trendOpen, setTrendOpen] = useState(false);

    // Recent Topics state
    const { user } = useAuth();
    const [recentTopics, setRecentTopics] = useState<ResearchTopic[]>([]);
    const [topicsLoading, setTopicsLoading] = useState(false);

    useEffect(() => {
        if (!user) return;
        setTopicsLoading(true);
        researchTopicsService.listResearchTopics({ order_by: 'created_at', order_direction: 'desc', size: 6 })
            .then(r => setRecentTopics(r.items || []))
            .catch(() => {})
            .finally(() => setTopicsLoading(false));
    }, [user]);


    const handleTopicExplosion = async () => {
        if (!searchTerm.trim() || !activeProject) return;
        setIsExploding(true);
        try {
            const newTopic = await researchTopicsService.createResearchTopic({
                title: searchTerm,
                description: `Topic explosion for ${activeProject.domain || activeProject.app_name}: ${searchTerm}`
            });
            navigate(`/research/${newTopic.id}`);
        } catch (e) {
            console.error(e);
            setIsExploding(false);
        }
    };

    // ── Propose Topics ───────────────────────────────────────────────────────
    const handleProposeTopics = async () => {
        if (!activeProject) return;
        setProposeOpen(true);
        setProposeLoading(true);
        setProposeError(null);
        setProposedTopics([]);
        try {
            const niche = activeProject.site_description || activeProject.websiteDescription || activeProject.domain || activeProject.app_name;
            const res = await apiClient.post<any>('/ai/propose-topics', {
                niche_description: niche,
                count: 5
            });
            if (res.topics) {
                setProposedTopics(res.topics);
            } else if (Array.isArray(res)) {
                setProposedTopics(res);
            } else {
                setProposeError('No topics returned. Try again.');
            }
        } catch (e: any) {
            setProposeError(e?.response?.data?.detail || e?.message || 'Failed to propose topics');
        } finally {
            setProposeLoading(false);
        }
    };

    const hasProject = !!activeProject;

    return (
        <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-start relative overflow-hidden">
            {/* Background */}
            <div className="absolute inset-0 bg-gradient-to-b from-indigo-900/25 via-slate-950 to-slate-950 pointer-events-none" />
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[500px] bg-indigo-600/8 rounded-full blur-3xl pointer-events-none" />

            <div className="relative z-10 w-full max-w-4xl mx-auto px-6 pt-12 pb-20">

                {/* ── Brand Header ─────────────────────────────────────────────── */}
                <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4 }}
                    className="flex items-center justify-between mb-12"
                >
                    <div>
                        <h1 className="text-2xl font-extrabold tracking-tight bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
                            Zenith Creator
                        </h1>
                        <p className="text-xs text-slate-600 uppercase tracking-widest font-medium">Command Center</p>
                    </div>

                    {/* Project Switcher — top right */}
                    <ProjectSwitcher />
                </motion.div>

                {/* ── Hero Headline ─────────────────────────────────────────────── */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.05 }}
                    className="text-center mb-10"
                >
                    <h2 className="text-4xl md:text-5xl font-extrabold tracking-tight text-white leading-tight mb-3">
                        {hasProject
                            ? <>Working in <span className="bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">{activeProject.domain || activeProject.app_name}</span></>
                            : 'Select a Project to Begin'
                        }
                    </h2>
                    <p className="text-slate-400 text-base md:text-lg max-w-xl mx-auto">
                        {hasProject && (activeProject.site_description || activeProject.websiteDescription)
                            ? (activeProject.site_description || activeProject.websiteDescription)
                            : hasProject
                                ? 'No niche description set — add one in Settings.'
                                : 'Choose a project from the dropdown above to activate your Command Center.'
                        }
                    </p>
                </motion.div>

                {/* ── Action Hero Section ───────────────────────────────────────── */}
                <motion.div
                    initial={{ opacity: 0, y: 24 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.1 }}
                    className="relative mb-8"
                >
                    {/* Blur overlay when no project is selected */}
                    <AnimatePresence>
                        {!hasProject && !projectLoading && (
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                className="absolute inset-0 z-10 rounded-3xl backdrop-blur-md bg-slate-950/70 flex flex-col items-center justify-center gap-3 border border-amber-500/20"
                            >
                                <Lock className="w-8 h-8 text-amber-400" />
                                <p className="text-white font-semibold text-lg">Select a Project to Unlock</p>
                                <p className="text-slate-400 text-sm">Use the workspace switcher above to get started</p>
                            </motion.div>
                        )}
                    </AnimatePresence>

                    {/* Search Bar */}
                    <div className={`transition-all duration-300 ${!hasProject ? 'pointer-events-none' : ''}`}>
                        <div className="relative group">
                            {/* Animated glow */}
                            <div className="absolute -inset-1 bg-gradient-to-r from-indigo-500/30 to-purple-500/30 rounded-3xl blur-xl opacity-0 group-focus-within:opacity-100 transition-opacity duration-500" />

                            <div className="relative flex items-center bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl overflow-hidden focus-within:border-indigo-500/50 transition-all duration-300 shadow-xl">
                                <Search className="absolute left-6 h-5 w-5 text-slate-500 pointer-events-none" />
                                <input
                                    ref={searchRef}
                                    id="topic-search"
                                    type="text"
                                    placeholder="Enter a broad topic to explode into SEO ideas..."
                                    className="flex-1 pl-14 pr-4 h-16 text-base bg-transparent border-0 text-white placeholder:text-slate-600 focus:outline-none"
                                    value={searchTerm}
                                    onChange={e => setSearchTerm(e.target.value)}
                                    onKeyDown={e => e.key === 'Enter' && handleTopicExplosion()}
                                />
                                <button
                                    id="topic-explode-btn"
                                    onClick={handleTopicExplosion}
                                    disabled={!searchTerm.trim() || isExploding}
                                    className="m-2 h-12 px-6 bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-500 hover:to-indigo-600 text-white rounded-xl transition-all flex items-center gap-2 font-semibold disabled:opacity-40 disabled:cursor-not-allowed shadow-lg shadow-indigo-600/30"
                                >
                                    {isExploding
                                        ? <Loader2 className="h-4 w-4 animate-spin" />
                                        : <Zap className="h-4 w-4" />
                                    }
                                    Explode
                                </button>
                            </div>
                        </div>

                        {/* ── Quick-Start Action Row ─────────────────────────────── */}
                        <div className="grid grid-cols-3 gap-3 mt-4">
                            {/* Button A — Propose Topics */}
                            <button
                                id="btn-propose-topics"
                                onClick={handleProposeTopics}
                                disabled={!hasProject}
                                className="group flex flex-col items-center gap-2 p-4 rounded-2xl bg-white/5 border border-white/8 hover:border-indigo-500/40 hover:bg-indigo-500/8 transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
                            >
                                <div className="w-10 h-10 rounded-xl bg-indigo-500/15 group-hover:bg-indigo-500/25 flex items-center justify-center transition-colors">
                                    <Sparkles className="w-5 h-5 text-indigo-400" />
                                </div>
                                <div className="text-center">
                                    <p className="text-sm font-semibold text-white">Propose Topics</p>
                                    <p className="text-[11px] text-slate-500 leading-tight mt-0.5">AI picks 5 topics for your niche</p>
                                </div>
                            </button>

                            {/* Button B — News Pulse */}
                            <button
                                id="btn-news-pulse"
                                onClick={() => setTrendOpen(true)}
                                disabled={!hasProject}
                                className="group flex flex-col items-center gap-2 p-4 rounded-2xl bg-white/5 border border-white/8 hover:border-purple-500/40 hover:bg-purple-500/8 transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
                            >
                                <div className="w-10 h-10 rounded-xl bg-purple-500/15 group-hover:bg-purple-500/25 flex items-center justify-center transition-colors">
                                    <Newspaper className="w-5 h-5 text-purple-400" />
                                </div>
                                <div className="text-center">
                                    <p className="text-sm font-semibold text-white">News Pulse</p>
                                    <p className="text-[11px] text-slate-500 leading-tight mt-0.5">Latest trends for your niche</p>
                                </div>
                            </button>

                            {/* Button C — Manual Entry */}
                            <button
                                id="btn-manual-entry"
                                onClick={() => searchRef.current?.focus()}
                                disabled={!hasProject}
                                className="group flex flex-col items-center gap-2 p-4 rounded-2xl bg-white/5 border border-white/8 hover:border-slate-400/40 hover:bg-slate-500/8 transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
                            >
                                <div className="w-10 h-10 rounded-xl bg-slate-500/15 group-hover:bg-slate-500/25 flex items-center justify-center transition-colors">
                                    <PenLine className="w-5 h-5 text-slate-400" />
                                </div>
                                <div className="text-center">
                                    <p className="text-sm font-semibold text-white">Manual Entry</p>
                                    <p className="text-[11px] text-slate-500 leading-tight mt-0.5">Type your own topic above</p>
                                </div>
                            </button>
                        </div>
                    </div>
                </motion.div>

                {/* ── Secondary: Navigation Links ──────────────────────────── */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.5, delay: 0.25 }}
                    className="border-t border-white/5 pt-8"
                >
                    <p className="text-xs text-slate-600 uppercase tracking-widest font-medium text-center mb-5">Continue Your Work</p>
                    <div className="flex flex-col sm:flex-row justify-center gap-4">
                        {/* Research Projects Button */}
                        <Link
                            to="/research"
                            className="group flex items-center gap-4 px-6 py-4 rounded-2xl bg-white/4 border border-white/8 hover:border-indigo-500/30 hover:bg-indigo-500/6 transition-all duration-200"
                        >
                            <div className="w-10 h-10 rounded-xl bg-indigo-500/15 flex items-center justify-center">
                                <Search className="w-5 h-5 text-indigo-400" />
                            </div>
                            <div>
                                <p className="font-semibold text-white group-hover:text-indigo-300 transition-colors">All Research Topics</p>
                                <p className="text-xs text-slate-500">View and manage all your research projects</p>
                            </div>
                            <ArrowRight className="w-4 h-4 text-slate-600 group-hover:text-indigo-400 group-hover:translate-x-1 transition-all ml-2" />
                        </Link>

                        {/* Content Library Button */}
                        <Link
                            to="/my-articles"
                            className="group flex items-center gap-4 px-6 py-4 rounded-2xl bg-white/4 border border-white/8 hover:border-purple-500/30 hover:bg-purple-500/6 transition-all duration-200"
                        >
                            <div className="w-10 h-10 rounded-xl bg-purple-500/15 flex items-center justify-center">
                                <BookOpen className="w-5 h-5 text-purple-400" />
                            </div>
                            <div>
                                <p className="font-semibold text-white group-hover:text-purple-300 transition-colors">Content Library</p>
                                <p className="text-xs text-slate-500">View saved roadmaps, drafts &amp; published articles</p>
                            </div>
                            <ArrowRight className="w-4 h-4 text-slate-600 group-hover:text-purple-400 group-hover:translate-x-1 transition-all ml-2" />
                        </Link>
                    </div>
                </motion.div>

                {/* ── Recent Research Topics ───────────────────────────────── */}

                {(recentTopics.length > 0 || topicsLoading) && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.5, delay: 0.35 }}
                        className="border-t border-white/5 pt-8 mt-2"
                    >
                        <div className="flex items-center justify-between mb-4">
                            <p className="text-xs text-slate-600 uppercase tracking-widest font-medium">Recent Research Topics</p>
                            <Link to="/research" className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors">View all →</Link>
                        </div>

                        {topicsLoading ? (
                            <div className="flex gap-3">
                                {[1,2,3].map(i => (
                                    <div key={i} className="flex-1 h-16 rounded-xl bg-white/4 animate-pulse" />
                                ))}
                            </div>
                        ) : (
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                                {recentTopics.map(topic => (
                                    <Link
                                        key={topic.id}
                                        to={`/research/${topic.id}`}
                                        className="group flex items-center gap-3 px-4 py-3 rounded-xl bg-white/4 border border-white/6 hover:border-indigo-500/30 hover:bg-indigo-500/6 transition-all duration-200"
                                    >
                                        <div className="w-8 h-8 rounded-lg bg-indigo-500/15 flex items-center justify-center flex-shrink-0">
                                            <FlaskConical className="w-4 h-4 text-indigo-400" />
                                        </div>
                                        <div className="min-w-0 flex-1">
                                            <p className="text-sm font-medium text-white group-hover:text-indigo-300 transition-colors truncate">{topic.title}</p>
                                            <p className="text-[11px] text-slate-600 flex items-center gap-1 mt-0.5">
                                                <Clock className="w-3 h-3" />
                                                {new Date(topic.created_at).toLocaleDateString()}
                                            </p>
                                        </div>
                                        <ArrowRight className="w-4 h-4 text-slate-700 group-hover:text-indigo-400 flex-shrink-0 transition-colors" />
                                    </Link>
                                ))}
                            </div>
                        )}
                    </motion.div>
                )}

            </div>

            {/* ── Modals ──────────────────────────────────────────────────────── */}
            <ProposeTopicsModal
                isOpen={proposeOpen}
                onClose={() => setProposeOpen(false)}
                topics={proposedTopics}
                loading={proposeLoading}
                error={proposeError}
            />

            {trendOpen && activeProject && (
                <TrendReportModal
                    siteId={activeProject.id}
                    siteDomain={activeProject.domain || activeProject.app_name}
                    isOpen={trendOpen}
                    onClose={() => setTrendOpen(false)}
                />
            )}
        </div>
    );
}
