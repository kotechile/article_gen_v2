import * as React from 'react';
import { ChevronDown, Globe, FolderOpen, PlusCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useProject } from '@/context/project-context';
import { Link } from 'react-router-dom';
import type { Project } from '@/types';

export function ProjectSwitcher() {
    const { projects, activeProject, setActiveProject, isLoading } = useProject();
    const [open, setOpen] = React.useState(false);
    const ref = React.useRef<HTMLDivElement>(null);

    // Close dropdown when clicking outside
    React.useEffect(() => {
        const handler = (e: MouseEvent) => {
            if (ref.current && !ref.current.contains(e.target as Node)) {
                setOpen(false);
            }
        };
        document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, []);

    const getDisplayName = (p: Project | null) => {
        if (!p) return 'Select a Project';
        return p.domain || p.app_name || 'Unnamed Project';
    };

    const getNicheLabel = (p: Project | null) => {
        if (!p) return null;
        const desc = p.site_description || p.websiteDescription;
        if (!desc) return null;
        // Trim to a short teaser
        return desc.length > 55 ? desc.slice(0, 55) + '…' : desc;
    };

    return (
        <div ref={ref} className="relative">
            <button
                id="project-switcher-btn"
                onClick={() => setOpen(o => !o)}
                className={`
                    group flex items-center gap-3 px-4 py-2.5 rounded-2xl border transition-all duration-200
                    ${activeProject
                        ? 'bg-white/8 border-indigo-500/30 hover:border-indigo-400/60'
                        : 'bg-amber-500/10 border-amber-500/40 hover:border-amber-400/70 animate-pulse-subtle'
                    }
                    backdrop-blur-sm cursor-pointer min-w-[260px] max-w-[420px]
                `}
                aria-label="Switch active project"
            >
                {/* Icon */}
                <div className={`flex-shrink-0 w-8 h-8 rounded-xl flex items-center justify-center
                    ${activeProject ? 'bg-indigo-500/20' : 'bg-amber-500/20'}`}>
                    {activeProject
                        ? <Globe className="w-4 h-4 text-indigo-400" />
                        : <FolderOpen className="w-4 h-4 text-amber-400" />
                    }
                </div>

                {/* Text */}
                <div className="flex-1 text-left overflow-hidden">
                    <p className="text-[10px] uppercase tracking-widest font-semibold text-slate-500 leading-none mb-0.5">
                        Active Project
                    </p>
                    {isLoading ? (
                        <div className="h-4 w-32 bg-white/10 rounded animate-pulse" />
                    ) : (
                        <p className={`text-sm font-semibold truncate leading-tight
                            ${activeProject ? 'text-white' : 'text-amber-400'}`}>
                            {getDisplayName(activeProject)}
                        </p>
                    )}
                    {activeProject && getNicheLabel(activeProject) && (
                        <p className="text-[11px] text-slate-500 truncate leading-tight mt-0.5">
                            {getNicheLabel(activeProject)}
                        </p>
                    )}
                </div>

                <ChevronDown className={`w-4 h-4 text-slate-500 flex-shrink-0 transition-transform duration-200 ${open ? 'rotate-180' : ''}`} />
            </button>

            {/* Dropdown */}
            <AnimatePresence>
                {open && (
                    <motion.div
                        initial={{ opacity: 0, y: -8, scale: 0.97 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -8, scale: 0.97 }}
                        transition={{ duration: 0.15 }}
                        className="absolute top-full mt-2 left-0 w-full min-w-[300px] z-50 bg-slate-900/95 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl overflow-hidden"
                    >
                        {/* Header */}
                        <div className="px-4 py-3 border-b border-white/5">
                            <p className="text-xs text-slate-500 font-medium uppercase tracking-widest">
                                Your Projects ({projects.length})
                            </p>
                        </div>

                        {/* Projects list */}
                        <div className="max-h-64 overflow-y-auto py-1">
                            {projects.length === 0 ? (
                                <div className="px-4 py-6 text-center text-slate-500 text-sm">
                                    No projects yet. Create one in Settings.
                                </div>
                            ) : (
                                projects.map((p) => (
                                    <button
                                        key={p.id}
                                        onClick={() => { setActiveProject(p); setOpen(false); }}
                                        className={`w-full flex items-center gap-3 px-4 py-3 text-left transition-colors
                                            ${activeProject?.id === p.id
                                                ? 'bg-indigo-500/15 text-indigo-300'
                                                : 'hover:bg-white/5 text-slate-300'
                                            }`}
                                    >
                                        <div className={`w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0
                                            ${activeProject?.id === p.id ? 'bg-indigo-500/25' : 'bg-white/8'}`}>
                                            <Globe className="w-3.5 h-3.5" />
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <p className="text-sm font-semibold truncate">
                                                {p.domain || p.app_name || 'Unnamed Project'}
                                            </p>
                                            {(p.site_description || p.websiteDescription) && (
                                                <p className="text-[11px] text-slate-500 truncate">
                                                    {(p.site_description || p.websiteDescription)?.slice(0, 60)}…
                                                </p>
                                            )}
                                        </div>
                                        {activeProject?.id === p.id && (
                                            <div className="w-1.5 h-1.5 rounded-full bg-indigo-400 flex-shrink-0" />
                                        )}
                                    </button>
                                ))
                            )}
                        </div>

                        {/* Footer — link to settings */}
                        <div className="border-t border-white/5 p-2">
                            <Link
                                to="/settings?tab=niches"
                                onClick={() => setOpen(false)}
                                className="flex items-center gap-2 px-3 py-2.5 rounded-xl hover:bg-white/5 text-slate-400 hover:text-slate-200 transition-colors text-sm"
                            >
                                <PlusCircle className="w-4 h-4" />
                                Manage Niches / Websites
                            </Link>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
