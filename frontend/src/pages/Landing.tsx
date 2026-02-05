import { Link } from 'react-router-dom';
import { Zap, BookOpen, ArrowRight } from 'lucide-react';
import { motion } from 'framer-motion';

export function Landing() {
    return (
        <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center p-8 relative overflow-hidden">
            {/* Radial gradient background */}
            <div className="absolute inset-0 bg-gradient-to-b from-indigo-900/20 to-transparent pointer-events-none" />

            <div className="relative z-10 w-full max-w-6xl mx-auto">
                {/* Hero Section */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                    className="text-center mb-16"
                >
                    {/* Logo/Brand */}
                    <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight mb-6 bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
                        Zenith Creator
                    </h1>

                    {/* Headline */}
                    <h2 className="text-4xl md:text-5xl font-bold tracking-tight text-white mb-6">
                        Where every spark becomes a roadmap.
                    </h2>

                    {/* Subheadline */}
                    <p className="text-lg md:text-xl text-slate-400 max-w-3xl mx-auto leading-relaxed">
                        The minimalist workspace for strategic creators to brainstorm profitable niches and manage their content empire.
                    </p>
                </motion.div>

                {/* Decision Portal - Two Cards */}
                <div className="grid md:grid-cols-2 gap-8 mb-16">
                    {/* Card A: The Explorer */}
                    <motion.div
                        initial={{ opacity: 0, y: 40 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6, delay: 0.1 }}
                    >
                        <Link to="/research">
                            <div className="group h-full bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-8 hover:-translate-y-2 hover:border-indigo-500/50 transition-all duration-300 cursor-pointer">
                                {/* Icon */}
                                <div className="w-16 h-16 rounded-full bg-gradient-to-br from-indigo-500/20 to-indigo-600/20 flex items-center justify-center mb-6 group-hover:from-indigo-500/30 group-hover:to-indigo-600/30 transition-colors">
                                    <Zap className="w-8 h-8 text-indigo-400" />
                                </div>

                                {/* Title */}
                                <h3 className="text-2xl font-bold text-white mb-4 tracking-tight">
                                    Explore New Topic
                                </h3>

                                {/* Description */}
                                <p className="text-slate-400 mb-8 leading-relaxed">
                                    Analyze a broad subject, uncover high-profit keywords, and automatically generate dozens of article and software concepts.
                                </p>

                                {/* CTA */}
                                <div className="flex items-center gap-2 text-indigo-400 font-medium group-hover:gap-4 transition-all">
                                    <span>Start Brainstorming</span>
                                    <ArrowRight className="w-5 h-5" />
                                </div>
                            </div>
                        </Link>
                    </motion.div>

                    {/* Card B: The Architect */}
                    <motion.div
                        initial={{ opacity: 0, y: 40 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6, delay: 0.2 }}
                    >
                        <Link to="/my-articles">
                            <div className="group h-full bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-8 hover:-translate-y-2 hover:border-purple-500/50 transition-all duration-300 cursor-pointer">
                                {/* Icon */}
                                <div className="w-16 h-16 rounded-full bg-gradient-to-br from-purple-500/20 to-purple-600/20 flex items-center justify-center mb-6 group-hover:from-purple-500/30 group-hover:to-purple-600/30 transition-colors">
                                    <BookOpen className="w-8 h-8 text-purple-400" />
                                </div>

                                {/* Title */}
                                <h3 className="text-2xl font-bold text-white mb-4 tracking-tight">
                                    Content Library
                                </h3>

                                {/* Description */}
                                <p className="text-slate-400 mb-8 leading-relaxed">
                                    Access your saved roadmap. Pick up where you left off, track SEO scores, and move ideas into full-scale production.
                                </p>

                                {/* CTA */}
                                <div className="flex items-center gap-2 text-purple-400 font-medium group-hover:gap-4 transition-all">
                                    <span>View My Articles</span>
                                    <ArrowRight className="w-5 h-5" />
                                </div>
                            </div>
                        </Link>
                    </motion.div>
                </div>

                {/* Footer Attribution */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.6, delay: 0.4 }}
                    className="text-center"
                >
                    <p className="text-xs text-slate-600 uppercase tracking-widest font-medium">
                        Powered by Advanced Intelligence
                    </p>
                </motion.div>
            </div>
        </div>
    );
}
