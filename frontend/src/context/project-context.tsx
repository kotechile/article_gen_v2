import * as React from 'react';
import { supabase } from '@/lib/supabase';
import { useAuth } from '@/context/auth-context';
import type { Project } from '@/types';

const STORAGE_KEY = 'zenith_active_project_id';

interface ProjectContextType {
    projects: Project[];
    activeProject: Project | null;
    isLoading: boolean;
    setActiveProject: (project: Project | null) => void;
    refreshProjects: () => Promise<void>;
}

const ProjectContext = React.createContext<ProjectContextType | undefined>(undefined);

export function ProjectProvider({ children }: { children: React.ReactNode }) {
    const { user } = useAuth();
    const [projects, setProjects] = React.useState<Project[]>([]);
    const [activeProject, setActiveProjectState] = React.useState<Project | null>(null);
    const [isLoading, setIsLoading] = React.useState(true);

    const fetchProjects = React.useCallback(async () => {
        if (!user) {
            setProjects([]);
            setActiveProjectState(null);
            setIsLoading(false);
            return;
        }
        setIsLoading(true);
        try {
            const { data, error } = await supabase
                .from('projects')
                .select('*')
                .eq('user_id', user.id)
                .order('created_at', { ascending: false });

            if (error) {
                console.error('ProjectContext: Failed to fetch projects', error);
                setProjects([]);
                return;
            }

            let fetched = (data as Project[]) || [];
            try {
                const { data: wpDetails } = await supabase
                    .from('wordPress_details')
                    .select('*')
                    .eq('user_id', user.id);

                if (wpDetails && wpDetails.length > 0) {
                    fetched = fetched.map(p => {
                        const normP = (p.domain || '').trim().toLowerCase().replace(/^https?:\/\//, '').replace(/\/$/, '');
                        const matchedWp = wpDetails.find(w => {
                            const normW = (w.domain || '').trim().toLowerCase().replace(/^https?:\/\//, '').replace(/\/$/, '');
                            return (normW && normW === normP) || (w.app_name && p.app_name && w.app_name.toLowerCase() === p.app_name.toLowerCase());
                        });
                        if (matchedWp) {
                            return {
                                ...p,
                                wpUserName: p.wpUserName || (p as any).wpusername || matchedWp.wpUserName || matchedWp.wpusername || '',
                                wpusername: (p as any).wpusername || p.wpUserName || matchedWp.wpusername || matchedWp.wpUserName || '',
                                wordpress_key: p.wordpress_key || matchedWp.wordpress_key || '',
                                seo_plugin: p.seo_plugin || matchedWp.seo_plugin || 'unknown',
                                cms_url: (p as any).cms_url || matchedWp.cms_url || matchedWp.cms || '',
                            };
                        }
                        return {
                            ...p,
                            wpUserName: p.wpUserName || (p as any).wpusername || '',
                            wpusername: (p as any).wpusername || p.wpUserName || '',
                        };
                    });
                }
            } catch (wpErr) {
                console.warn('ProjectContext: Could not enrich with wordPress_details', wpErr);
            }

            setProjects(fetched);

            // Restore previously selected project from localStorage
            const savedId = localStorage.getItem(STORAGE_KEY);
            if (savedId) {
                const restored = fetched.find(p => p.id === savedId) ?? null;
                setActiveProjectState(restored);
            } else if (fetched.length > 0 && !activeProject) {
                // Auto-select the first project if nothing is saved
                setActiveProjectState(fetched[0]);
                localStorage.setItem(STORAGE_KEY, fetched[0].id);
            }
        } finally {
            setIsLoading(false);
        }
    }, [user]);

    React.useEffect(() => {
        fetchProjects();
    }, [fetchProjects]);

    const setActiveProject = (project: Project | null) => {
        setActiveProjectState(project);
        if (project) {
            localStorage.setItem(STORAGE_KEY, project.id);
        } else {
            localStorage.removeItem(STORAGE_KEY);
        }
    };

    return (
        <ProjectContext.Provider value={{
            projects,
            activeProject,
            isLoading,
            setActiveProject,
            refreshProjects: fetchProjects,
        }}>
            {children}
        </ProjectContext.Provider>
    );
}

export function useProject() {
    const ctx = React.useContext(ProjectContext);
    if (!ctx) throw new Error('useProject must be used within a ProjectProvider');
    return ctx;
}
