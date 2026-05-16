import * as React from 'react';
import { ChevronDown, Globe, FolderOpen, PlusCircle } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useProject } from '@/context/project-context';
import { Link } from 'react-router-dom';
import type { Project } from '@/types';

type ProjectSwitcherProps = {
    collapsed?: boolean
}

export function ProjectSwitcher({ collapsed = false }: ProjectSwitcherProps) {
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
                    group flex items-center gap-3 rounded-2xl border transition-all duration-200
                    ${activeProject
                        ? 'bg-background/80 border-border hover:border-ring/50'
                        : 'bg-accent/40 border-border hover:border-ring/50 animate-pulse-subtle'
                    }
                    ${collapsed
                        ? 'h-11 w-full justify-center px-0'
                        : 'min-w-0 w-full max-w-full px-4 py-2.5'
                    }
                    backdrop-blur-sm cursor-pointer
                `}
                aria-label="Switch active project"
                title={collapsed ? getDisplayName(activeProject) : undefined}
            >
                {/* Icon */}
                <div className={`flex-shrink-0 w-8 h-8 rounded-xl flex items-center justify-center
                    ${activeProject ? 'bg-primary/10' : 'bg-accent/60'}`}>
                    {activeProject
                        ? <Globe className="w-4 h-4 text-primary" />
                        : <FolderOpen className="w-4 h-4 text-accent-foreground" />
                    }
                </div>

                {/* Text */}
                {!collapsed && (
                    <div className="flex-1 text-left overflow-hidden">
                        <p className="text-[10px] uppercase tracking-widest font-semibold text-muted-foreground leading-none mb-0.5">
                            Active Project
                        </p>
                        {isLoading ? (
                            <div className="h-4 w-32 bg-muted/50 rounded animate-pulse" />
                        ) : (
                            <p className={`text-sm font-semibold truncate leading-tight
                                text-foreground`}>
                                {getDisplayName(activeProject)}
                            </p>
                        )}
                        {activeProject && getNicheLabel(activeProject) && (
                            <p className="text-[11px] text-muted-foreground truncate leading-tight mt-0.5">
                                {getNicheLabel(activeProject)}
                            </p>
                        )}
                    </div>
                )}

                {!collapsed && <ChevronDown className={`w-4 h-4 text-muted-foreground flex-shrink-0 transition-transform duration-200 ${open ? 'rotate-180' : ''}`} />}
            </button>

            {/* Dropdown */}
            <AnimatePresence>
                {open && (
                    <motion.div
                        initial={{ opacity: 0, y: -8, scale: 0.97 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -8, scale: 0.97 }}
                        transition={{ duration: 0.15 }}
                        className={`absolute top-full mt-2 z-50 bg-popover/95 backdrop-blur-xl border border-border rounded-2xl shadow-2xl overflow-hidden ${
                            collapsed ? 'left-full ml-3 w-[320px]' : 'left-0 w-full min-w-[300px]'
                        }`}
                    >
                        {/* Header */}
                        <div className="px-4 py-3 border-b border-border">
                            <p className="text-xs text-muted-foreground font-medium uppercase tracking-widest">
                                Your Projects ({projects.length})
                            </p>
                        </div>

                        {/* Projects list */}
                        <div className="max-h-64 overflow-y-auto py-1">
                            {projects.length === 0 ? (
                                <div className="px-4 py-6 text-center text-muted-foreground text-sm">
                                    No projects yet. Create one in Settings.
                                </div>
                            ) : (
                                projects.map((p) => (
                                    <button
                                        key={p.id}
                                        onClick={() => { setActiveProject(p); setOpen(false); }}
                                        className={`w-full flex items-center gap-3 px-4 py-3 text-left transition-colors
                                            ${activeProject?.id === p.id
                                                ? 'bg-accent text-accent-foreground'
                                                : 'hover:bg-muted text-foreground'
                                            }`}
                                    >
                                        <div className={`w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0
                                            ${activeProject?.id === p.id ? 'bg-primary/10 text-primary' : 'bg-muted/50'}`}>
                                            <Globe className="w-3.5 h-3.5" />
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <p className="text-sm font-semibold truncate">
                                                {p.domain || p.app_name || 'Unnamed Project'}
                                            </p>
                                            {(p.site_description || p.websiteDescription) && (
                                                <p className="text-[11px] text-muted-foreground truncate">
                                                    {(p.site_description || p.websiteDescription)?.slice(0, 60)}…
                                                </p>
                                            )}
                                        </div>
                                        {activeProject?.id === p.id && (
                                            <div className="w-1.5 h-1.5 rounded-full bg-primary flex-shrink-0" />
                                        )}
                                    </button>
                                ))
                            )}
                        </div>

                        {/* Footer — link to settings */}
                        <div className="border-t border-border p-2">
                            <Link
                                to="/settings?tab=niches"
                                onClick={() => setOpen(false)}
                                className="flex items-center gap-2 px-3 py-2.5 rounded-xl hover:bg-muted text-muted-foreground hover:text-foreground transition-colors text-sm"
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
