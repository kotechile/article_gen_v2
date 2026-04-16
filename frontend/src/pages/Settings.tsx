
import React, { useState, useEffect } from 'react';
import {
    Settings as SettingsIcon,
    Search,
    Wand2,
    Globe,
    FileText,
    Save,
    Loader2,
    Plus,
    Trash2,
    Edit2,
    RefreshCw,
    CheckCircle2,
    AlertCircle,
    TrendingUp,
    Layout,
    FolderTree,
    ChevronRight
} from 'lucide-react';
import { TrendReportModal } from '../components/TrendReportModal';
import { supabase } from '../lib/supabase';
import { useAuth } from '../context/auth-context';
import { useProject } from '../context/project-context';
import { apiClient } from '../api-client';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Switch } from '../components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import type { Project } from '../types';

// Read tab from URL query (?tab=niches)
function getInitialTab() {
    const params = new URLSearchParams(window.location.search);
    return params.get('tab') === 'niches' ? 'niches' : 'research';
}

interface ResearchSettings {
    min_volume: number;
    max_difficulty: number;
    min_cpc: number;
    strict_mode: boolean;
}

interface ApplicationSettings {
    id: number;
    chatGptKey?: string;
    openAIKey?: string;
    geminiKey?: string;
    perplexityAI_key?: string;
    claudeKey?: string;
    stabilityAiKey?: string;
    youTubeKey?: string;
    unsplashKey?: string;
    geminiModel?: string;
    perplexityModel?: string;
    openAIModel?: string;
    [key: string]: any;
}

export const Settings: React.FC = () => {
    const { user } = useAuth();
    const { refreshProjects } = useProject();
    const [activeTab, setActiveTab] = useState(getInitialTab);
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);

    // Research Settings State
    const [researchSettings, setResearchSettings] = useState<ResearchSettings>({
        min_volume: 50,
        max_difficulty: 50,
        min_cpc: 0.5,
        strict_mode: true,
    });

    // Content Settings State (LLM Keys etc)
    const [appSettings, setAppSettings] = useState<Partial<ApplicationSettings>>({});

    // Projects (niches/websites) State
    const [projects, setProjects] = useState<Project[]>([]);
    const [editingId, setEditingId] = useState<string | null>(null);
    const [formData, setFormData] = useState<Partial<Project>>({});
    const [isSaving, setIsSaving] = useState(false);
    const [showWpFields, setShowWpFields] = useState(false);

    // Posts State
    const [importedPosts, setImportedPosts] = useState<any[]>([]);
    const [postsLoading, setPostsLoading] = useState(false);
    const [isSyncing, setIsSyncing] = useState(false);

    // Trend Report State
    const [trendProject, setTrendProject] = useState<Project | null>(null);

    // Categories State
    const [selectedProjectForCategories, setSelectedProjectForCategories] = useState<Project | null>(null);
    const [categories, setCategories] = useState<any[]>([]);
    const [categoriesLoading, setCategoriesLoading] = useState(false);
    const [editingCategory, setEditingCategory] = useState<{ id?: string; name: string; level: 1 | 2; parent_category_id?: string } | null>(null);
    const [isSavingCategory, setIsSavingCategory] = useState(false);

    useEffect(() => {
        if (user) {
            fetchAllData();
        }
    }, [user]);

    const fetchAllData = async () => {
        setLoading(true);
        try {
            await Promise.all([
                fetchResearchSettings(),
                fetchAppSettings(),
                fetchProjects(),
                fetchImportedPosts()
            ]);
        } catch (err) {
            console.error("Error fetching settings:", err);
            setError("Failed to load some settings.");
        } finally {
            setLoading(false);
        }
    };

    const fetchResearchSettings = async () => {
        try {
            const response = await apiClient.get<any>('/settings/research');
            if (response.success && response.data) {
                setResearchSettings(response.data);
            }
        } catch (err) {
            console.error("Research settings fetch failed:", err);
        }
    };

    const fetchAppSettings = async () => {
        const { data, error } = await supabase
            .from('application_settings')
            .select('*')
            .eq('id', 1)
            .single();

        if (!error && data) {
            setAppSettings(data);
        }
    };

    const fetchProjects = async () => {
        const { data, error } = await supabase
            .from('projects')
            .select('*')
            .eq('user_id', user!.id)
            .order('created_at', { ascending: false });

        if (!error) {
            setProjects((data as Project[]) || []);
        }
    };

    const fetchImportedPosts = async () => {
        setPostsLoading(true);
        const { data, error } = await supabase
            .from('wordpress_imported_posts')
            .select('*')
            .eq('user_id', user!.id)
            .order('created_at', { ascending: false });

        if (!error) {
            setImportedPosts(data || []);
        }
        setPostsLoading(false);
    };

    const fetchCategories = async (projectId: string) => {
        setCategoriesLoading(true);
        const { data, error } = await supabase
            .from('project_categories')
            .select('*')
            .eq('project_id', projectId)
            .eq('user_id', user!.id)
            .order('level', { ascending: true })
            .order('sort_order', { ascending: true })
            .order('name', { ascending: true });

        if (!error) {
            setCategories(data || []);
        }
        setCategoriesLoading(false);
    };

    const handleSaveCategory = async () => {
        if (!selectedProjectForCategories || !editingCategory || !user) return;
        if (!editingCategory.name.trim()) {
            alert('Category name is required');
            return;
        }

        setIsSavingCategory(true);
        try {
            const slug = editingCategory.name
                .toLowerCase()
                .replace(/[^a-z0-9]+/g, '-')
                .replace(/(^-|-$)/g, '');

            const payload = {
                project_id: selectedProjectForCategories.id,
                user_id: user.id,
                name: editingCategory.name,
                slug,
                level: editingCategory.level,
                parent_category_id: editingCategory.level === 2 ? editingCategory.parent_category_id : null,
                sort_order: categories.filter(c => c.level === editingCategory?.level).length,
            };

            let error;
            if (editingCategory.id) {
                // Update existing
                const result = await supabase
                    .from('project_categories')
                    .update({ ...payload, updated_at: new Date().toISOString() })
                    .eq('id', editingCategory.id);
                error = result.error;
            } else {
                // Insert new
                const result = await supabase
                    .from('project_categories')
                    .insert([payload]);
                error = result.error;
            }

            if (error) throw error;

            setEditingCategory(null);
            await fetchCategories(selectedProjectForCategories.id);
        } catch (err: any) {
            alert(err.message || 'Failed to save category');
        } finally {
            setIsSavingCategory(false);
        }
    };

    const handleDeleteCategory = async (categoryId: string) => {
        if (!confirm('Delete this category? Child categories will also be deleted.')) return;

        try {
            const { error } = await supabase
                .from('project_categories')
                .delete()
                .eq('id', categoryId);

            if (error) throw error;

            if (selectedProjectForCategories) {
                await fetchCategories(selectedProjectForCategories.id);
            }
        } catch (err: any) {
            alert(err.message || 'Failed to delete category');
        }
    };

    const handleSelectProjectForCategories = (project: Project) => {
        setSelectedProjectForCategories(project);
        fetchCategories(project.id);
    };

    const handleSaveResearch = async () => {
        setSaving(true);
        setError(null);
        setSuccess(null);
        try {
            const result = await apiClient.post<any>('/settings/research', researchSettings);
            if (result.success) {
                setSuccess("Research settings saved!");
            } else {
                setError(result.message || "Failed to save settings");
            }
        } catch (err) {
            setError("Connection error");
        } finally {
            setSaving(false);
        }
    };

    const handleSaveAppSettings = async () => {
        setSaving(true);
        setError(null);
        setSuccess(null);
        try {
            const { error } = await supabase
                .from('application_settings')
                .update(appSettings)
                .eq('id', 1);

            if (error) throw error;
            setSuccess("Content Generation settings saved!");
        } catch (err: any) {
            setError(err.message || "Failed to save app settings");
        } finally {
            setSaving(false);
        }
    };

    const handleSaveProject = async () => {
        if (!user) return;
        // Require at minimum an app_name or domain
        if (!formData.app_name && !formData.domain) {
            alert('Please enter at least a Project Name or Domain URL.');
            return;
        }
        setIsSaving(true);
        try {
            const payload = { ...formData, user_id: user.id };
            let saveError;
            if (editingId && editingId !== 'new') {
                const { error: err } = await supabase.from('projects').update(payload).eq('id', editingId);
                saveError = err;
            } else {
                const { error: err } = await supabase.from('projects').insert([payload]);
                saveError = err;
            }
            if (saveError) throw saveError;
            setEditingId(null);
            setFormData({});
            setShowWpFields(false);
            await fetchProjects();
            await refreshProjects(); // update global context
        } catch (err: any) {
            alert(err.message || "Failed to save project");
        } finally {
            setIsSaving(false);
        }
    };

    const handleDeleteProject = async (id: string) => {
        if (!confirm('Delete this project? This cannot be undone.')) return;
        await supabase.from('projects').delete().eq('id', id);
        await fetchProjects();
        await refreshProjects();
    };

    const handleSync = async () => {
        if (!user) return;
        setIsSyncing(true);
        try {
            const response = await fetch('/api/wordpress/sync-posts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: user.id })
            });
            if (!response.ok) throw new Error('Sync failed');
            fetchImportedPosts();
        } catch (err) {
            console.error("Sync error:", err);
        } finally {
            setIsSyncing(false);
        }
    };

    if (loading) return (
        <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
            <Loader2 className="h-10 w-10 animate-spin text-indigo-600" />
            <p className="text-gray-500 animate-pulse">Loading settings...</p>
        </div>
    );

    return (
        <div className="container mx-auto py-8 max-w-5xl space-y-8">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b pb-6">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white flex items-center gap-3">
                        <SettingsIcon className="w-8 h-8 text-indigo-600" />
                        Settings
                    </h1>
                    <p className="text-muted-foreground mt-1">Manage your analysis thresholds and integrations</p>
                </div>
                {success && (
                    <div className="flex items-center gap-2 px-4 py-2 bg-green-50 text-green-700 rounded-lg border border-green-200 animate-in fade-in slide-in-from-top-2">
                        <CheckCircle2 className="w-4 h-4" />
                        <span className="text-sm font-medium">{success}</span>
                    </div>
                )}
                {error && (
                    <div className="flex items-center gap-2 px-4 py-2 bg-red-50 text-red-700 rounded-lg border border-red-200 animate-in fade-in slide-in-from-top-2">
                        <AlertCircle className="w-4 h-4" />
                        <span className="text-sm font-medium">{error}</span>
                    </div>
                )}
            </div>

            <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
                <TabsList className="bg-gray-100/50 dark:bg-gray-800/50 p-1 rounded-xl w-full md:w-auto h-auto grid grid-cols-2 md:grid-cols-4 gap-1">
                    <TabsTrigger value="research" className="rounded-lg data-[state=active]:bg-white dark:data-[state=active]:bg-gray-700 data-[state=active]:text-indigo-600 dark:data-[state=active]:text-indigo-400 data-[state=active]:shadow-sm py-2.5">
                        <Search className="w-4 h-4 mr-2" />
                        Topic Research
                    </TabsTrigger>
                    <TabsTrigger value="content" className="rounded-lg data-[state=active]:bg-white dark:data-[state=active]:bg-gray-700 data-[state=active]:text-indigo-600 dark:data-[state=active]:text-indigo-400 data-[state=active]:shadow-sm py-2.5">
                        <Wand2 className="w-4 h-4 mr-2" />
                        Content Generation
                    </TabsTrigger>
                    <TabsTrigger value="niches" className="rounded-lg data-[state=active]:bg-white dark:data-[state=active]:bg-gray-700 data-[state=active]:text-indigo-600 dark:data-[state=active]:text-indigo-400 data-[state=active]:shadow-sm py-2.5">
                        <Layout className="w-4 h-4 mr-2" />
                        Projects: Niches/Websites
                    </TabsTrigger>
                    <TabsTrigger value="posts" className="rounded-lg data-[state=active]:bg-white dark:data-[state=active]:bg-gray-700 data-[state=active]:text-indigo-600 dark:data-[state=active]:text-indigo-400 data-[state=active]:shadow-sm py-2.5">
                        <FileText className="w-4 h-4 mr-2" />
                        External Posts
                    </TabsTrigger>
                </TabsList>

                {/* Research Settings */}
                <TabsContent value="research" className="animate-in fade-in-50 duration-500">
                    <Card className="border-gray-200 dark:border-gray-800 shadow-sm overflow-hidden">
                        <CardHeader className="bg-gray-50/50 dark:bg-gray-900/50 border-b">
                            <CardTitle>ProfitPath Filtering Thresholds</CardTitle>
                            <CardDescription>
                                Configure how aggressively the system filters keywords during analysis.
                            </CardDescription>
                        </CardHeader>
                        <CardContent className="p-6 space-y-8">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                <div className="space-y-3">
                                    <Label htmlFor="min_volume" className="text-sm font-semibold">Minimum Search Volume</Label>
                                    <Input
                                        id="min_volume"
                                        type="number"
                                        value={researchSettings.min_volume}
                                        onChange={(e) => setResearchSettings({ ...researchSettings, min_volume: parseInt(e.target.value) || 0 })}
                                        className="h-11 rounded-xl"
                                    />
                                    <p className="text-[12px] text-muted-foreground leading-relaxed">
                                        Keywords with search volume below this threshold will be flagged or removed (Recommended: 50).
                                    </p>
                                </div>

                                <div className="space-y-3">
                                    <Label htmlFor="max_difficulty" className="text-sm font-semibold">Maximum Keyword Difficulty (0-100)</Label>
                                    <Input
                                        id="max_difficulty"
                                        type="number"
                                        value={researchSettings.max_difficulty}
                                        onChange={(e) => setResearchSettings({ ...researchSettings, max_difficulty: parseInt(e.target.value) || 0 })}
                                        className="h-11 rounded-xl"
                                    />
                                    <p className="text-[12px] text-muted-foreground leading-relaxed">
                                        Keywords with difficulty above this threshold will be considered high-competition (Recommended: 50).
                                    </p>
                                </div>

                                <div className="space-y-3">
                                    <Label htmlFor="min_cpc" className="text-sm font-semibold">Minimum CPC ($)</Label>
                                    <Input
                                        id="min_cpc"
                                        type="number"
                                        step="0.1"
                                        value={researchSettings.min_cpc}
                                        onChange={(e) => setResearchSettings({ ...researchSettings, min_cpc: parseFloat(e.target.value) || 0 })}
                                        className="h-11 rounded-xl"
                                    />
                                    <p className="text-[12px] text-muted-foreground leading-relaxed">
                                        Filters keywords by commercial intent based on cost-per-click values (Recommended: 0.5).
                                    </p>
                                </div>

                                <div className="flex items-center justify-between p-4 bg-indigo-50/50 dark:bg-indigo-900/10 rounded-2xl border border-indigo-100/50 dark:border-indigo-900/20">
                                    <div className="space-y-1">
                                        <Label htmlFor="strict_mode" className="text-sm font-semibold">Strict Mode</Label>
                                        <p className="text-xs text-muted-foreground">Automatically delete keywords failing thresholds.</p>
                                    </div>
                                    <Switch
                                        id="strict_mode"
                                        checked={researchSettings.strict_mode}
                                        onCheckedChange={(checked: boolean) => setResearchSettings({ ...researchSettings, strict_mode: checked })}
                                    />
                                </div>
                            </div>

                            <div className="flex justify-end pt-4 border-t">
                                <Button onClick={handleSaveResearch} disabled={saving} className="h-11 px-8 rounded-xl bg-indigo-600 hover:bg-indigo-700 shadow-lg shadow-indigo-600/20">
                                    {saving ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : <Save className="w-4 h-4 mr-2" />}
                                    Save Research Settings
                                </Button>
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>

                {/* Content Generation Settings */}
                <TabsContent value="content" className="animate-in fade-in-50 duration-500">
                    <Card className="border-gray-200 dark:border-gray-800 shadow-sm">
                        <CardHeader className="bg-gray-50/50 dark:bg-gray-900/50 border-b">
                            <CardTitle>AI &amp; Media API Keys</CardTitle>
                            <CardDescription>
                                Configure the credentials used for content generation and media sourcing.
                            </CardDescription>
                        </CardHeader>
                        <CardContent className="p-6 space-y-8">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-8">
                                {/* LLM Keys */}
                                <div className="space-y-6">
                                    <h3 className="text-sm font-bold uppercase tracking-wider text-gray-500 flex items-center gap-2">
                                        <div className="w-1.5 h-1.5 rounded-full bg-indigo-500"></div>
                                        Core LLM Providers
                                    </h3>
                                    <div className="space-y-4">
                                        <div className="space-y-2">
                                            <Label className="text-xs font-semibold text-gray-700 dark:text-gray-300">OpenAI API Key</Label>
                                            <Input type="password" value={appSettings.openAIKey || ''} onChange={e => setAppSettings({ ...appSettings, openAIKey: e.target.value })} className="h-10 rounded-lg font-mono text-sm" placeholder="sk-..." />
                                        </div>
                                        <div className="space-y-2">
                                            <Label className="text-xs font-semibold text-gray-700 dark:text-gray-300">Google Gemini Key</Label>
                                            <Input type="password" value={appSettings.geminiKey || ''} onChange={e => setAppSettings({ ...appSettings, geminiKey: e.target.value })} className="h-10 rounded-lg font-mono text-sm" placeholder="AIza..." />
                                        </div>
                                        <div className="space-y-2">
                                            <Label className="text-xs font-semibold text-gray-700 dark:text-gray-300">Claude (Anthropic) Key</Label>
                                            <Input type="password" value={appSettings.claudeKey || ''} onChange={e => setAppSettings({ ...appSettings, claudeKey: e.target.value })} className="h-10 rounded-lg font-mono text-sm" placeholder="sk-ant-..." />
                                        </div>
                                        <div className="space-y-2">
                                            <Label className="text-xs font-semibold text-gray-700 dark:text-gray-300">Perplexity AI Key</Label>
                                            <Input type="password" value={appSettings.perplexityAI_key || ''} onChange={e => setAppSettings({ ...appSettings, perplexityAI_key: e.target.value })} className="h-10 rounded-lg font-mono text-sm" placeholder="pplx-..." />
                                        </div>
                                    </div>
                                </div>

                                {/* Media & Utils */}
                                <div className="space-y-6">
                                    <h3 className="text-sm font-bold uppercase tracking-wider text-gray-500 flex items-center gap-2">
                                        <div className="w-1.5 h-1.5 rounded-full bg-orange-500"></div>
                                        Media &amp; Utils
                                    </h3>
                                    <div className="space-y-4">
                                        <div className="space-y-2">
                                            <Label className="text-xs font-semibold text-gray-700 dark:text-gray-300">YouTube API Key</Label>
                                            <Input type="password" value={appSettings.youTubeKey || ''} onChange={e => setAppSettings({ ...appSettings, youTubeKey: e.target.value })} className="h-10 rounded-lg font-mono text-sm" />
                                        </div>
                                        <div className="space-y-2">
                                            <Label className="text-xs font-semibold text-gray-700 dark:text-gray-300">Unsplash Access Key</Label>
                                            <Input type="password" value={appSettings.unsplashKey || ''} onChange={e => setAppSettings({ ...appSettings, unsplashKey: e.target.value })} className="h-10 rounded-lg font-mono text-sm" />
                                        </div>
                                        <div className="space-y-2">
                                            <Label className="text-xs font-semibold text-gray-700 dark:text-gray-300">Stability AI Key</Label>
                                            <Input type="password" value={appSettings.stabilityAiKey || ''} onChange={e => setAppSettings({ ...appSettings, stabilityAiKey: e.target.value })} className="h-10 rounded-lg font-mono text-sm" />
                                        </div>
                                        <div className="space-y-2">
                                            <Label className="text-xs font-semibold text-gray-700 dark:text-gray-300">Flux Key</Label>
                                            <Input type="password" value={appSettings.fluxKey || ''} onChange={e => setAppSettings({ ...appSettings, fluxKey: e.target.value })} className="h-10 rounded-lg font-mono text-sm" />
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="flex justify-end pt-4 border-t">
                                <Button onClick={handleSaveAppSettings} disabled={saving} className="h-11 px-8 rounded-xl bg-indigo-600 hover:bg-indigo-700 shadow-lg shadow-indigo-600/20">
                                    {saving ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : <Save className="w-4 h-4 mr-2" />}
                                    Update API Credentials
                                </Button>
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>

                {/* ── Niches / Websites Tab ──────────────────────────────────── */}
                <TabsContent value="niches" className="animate-in fade-in-50 duration-500">
                    <div className="space-y-6">
                        {/* Info Banner */}
                        <div className="flex items-start gap-3 p-4 rounded-xl bg-indigo-50/50 dark:bg-indigo-900/10 border border-indigo-100/50 dark:border-indigo-900/20">
                            <Globe className="w-5 h-5 text-indigo-500 flex-shrink-0 mt-0.5" />
                            <div>
                                <p className="text-sm font-semibold text-gray-800 dark:text-gray-200">Projects:Niches &amp; Websites</p>
                                <p className="text-xs text-muted-foreground mt-0.5">
                                    Add any project here — a WordPress site with full sync, or just a niche description for AI-driven content without a website. The active project on your Command Center is selected from this list.
                                </p>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                            {projects.map(project => (
                                <Card key={project.id} className="relative group overflow-hidden border-gray-200 dark:border-gray-800 hover:border-indigo-500/50 transition-all">
                                    <div className="absolute top-0 left-0 w-1 h-full bg-indigo-500 group-hover:w-2 transition-all"></div>
                                    <CardHeader className="pb-2">
                                        <div className="flex justify-between items-start">
                                            <div>
                                                <CardTitle className="text-lg truncate max-w-[180px]">
                                                    {project.domain || project.app_name || 'Unnamed Project'}
                                                </CardTitle>
                                                <CardDescription className="flex items-center gap-1.5 mt-1">
                                                    {project.domain
                                                        ? <><span className="w-1.5 h-1.5 rounded-full bg-green-500"></span> WordPress</>
                                                        : <><span className="w-1.5 h-1.5 rounded-full bg-indigo-500"></span> Niche Only</>
                                                    }
                                                </CardDescription>
                                            </div>
                                            <div className="flex gap-1">
                                                <Button variant="ghost" size="icon" className="h-8 w-8 text-gray-400 hover:text-indigo-600" onClick={() => { setEditingId(project.id); setFormData(project); setShowWpFields(!!project.wordpress_key || !!project.wpUserName); }}>
                                                    <Edit2 className="h-4 w-4" />
                                                </Button>
                                                <Button variant="ghost" size="icon" className="h-8 w-8 text-gray-400 hover:text-red-600" onClick={() => handleDeleteProject(project.id)}>
                                                    <Trash2 className="h-4 w-4" />
                                                </Button>
                                            </div>
                                        </div>
                                    </CardHeader>
                                    <CardContent>
                                        {(project.site_description || project.websiteDescription) && (
                                            <p className="text-xs text-muted-foreground line-clamp-2 mb-3">
                                                {project.site_description || project.websiteDescription}
                                            </p>
                                        )}
                                        <div className="flex items-center justify-between mt-2">
                                            <Button
                                                size="sm"
                                                variant="outline"
                                                className="h-7 text-xs gap-1.5 border-indigo-200 text-indigo-600 hover:bg-indigo-50 hover:text-indigo-700"
                                                onClick={() => setTrendProject(project)}
                                            >
                                                <TrendingUp className="w-3 h-3" />
                                                What's Trending
                                            </Button>
                                            <Globe className="w-4 h-4 text-gray-300" />
                                        </div>
                                    </CardContent>
                                </Card>
                            ))}

                            <button
                                onClick={() => { setEditingId('new'); setFormData({}); setShowWpFields(false); }}
                                className="h-[160px] border-2 border-dashed border-gray-200 dark:border-gray-800 rounded-2xl flex flex-col items-center justify-center gap-2 text-gray-400 hover:border-indigo-500 hover:text-indigo-500 hover:bg-indigo-50/30 transition-all group"
                            >
                                <Plus className="w-8 h-8 group-hover:scale-110 transition-transform" />
                                <span className="text-sm font-medium">Add Niche / Website</span>
                            </button>
                        </div>

                        {/* Edit / Create Form */}
                        {editingId && (
                            <Card className="border-indigo-100 dark:border-indigo-900 bg-indigo-50/20 dark:bg-indigo-900/5 overflow-hidden ring-1 ring-indigo-500/20">
                                <CardHeader className="border-b border-indigo-100/50 dark:border-indigo-900/50">
                                    <CardTitle className="text-lg">
                                        {editingId === 'new' ? 'New Niche / Website' : 'Edit Project'}
                                    </CardTitle>
                                    <CardDescription>
                                        Fill in a project name and niche description. Add WordPress credentials only if you want to sync content.
                                    </CardDescription>
                                </CardHeader>
                                <CardContent className="p-6 space-y-6">
                                    {/* Core fields — always shown */}
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                        <div className="space-y-2">
                                            <Label className="text-xs font-semibold">Project / Niche Name <span className="text-indigo-500">*</span></Label>
                                            <Input
                                                placeholder="e.g. Home & DIY Blog"
                                                value={formData.app_name || ''}
                                                onChange={e => setFormData({ ...formData, app_name: e.target.value })}
                                                className="h-10 rounded-lg"
                                            />
                                        </div>
                                        <div className="space-y-2">
                                            <Label className="text-xs font-semibold">Website URL (optional)</Label>
                                            <Input
                                                placeholder="wellroost.com"
                                                value={formData.domain || ''}
                                                onChange={e => setFormData({ ...formData, domain: e.target.value })}
                                                className="h-10 rounded-lg"
                                            />
                                        </div>
                                        <div className="space-y-2 md:col-span-2">
                                            <Label className="text-xs font-semibold">Niche Description (used by AI) <span className="text-indigo-500">*</span></Label>
                                            <textarea
                                                placeholder="E.g. A home improvement blog focusing on DIY renovations, sustainable materials, and budget-friendly interior design for first-time homeowners."
                                                value={formData.site_description || ''}
                                                onChange={e => setFormData({ ...formData, site_description: e.target.value })}
                                                rows={3}
                                                className="w-full px-3 py-2 text-sm rounded-lg border border-input bg-background resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500/40"
                                            />
                                        </div>
                                        <div className="space-y-2 md:col-span-2">
                                            <Label className="text-xs font-semibold">Target Audience Description (optional)</Label>
                                            <Input
                                                placeholder="E.g. First-time homeowners aged 25-45 looking for budget DIY tips"
                                                value={formData.targetAudienceDescription || ''}
                                                onChange={e => setFormData({ ...formData, targetAudienceDescription: e.target.value })}
                                                className="h-10 rounded-lg"
                                            />
                                        </div>
                                    </div>

                                    {/* WordPress toggle */}
                                    <div className="flex items-center gap-3">
                                        <Switch
                                            id="wp-toggle"
                                            checked={showWpFields}
                                            onCheckedChange={setShowWpFields}
                                        />
                                        <Label htmlFor="wp-toggle" className="text-sm font-medium cursor-pointer">
                                            This project has a WordPress website (add sync credentials)
                                        </Label>
                                    </div>

                                    {showWpFields && (
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2 border-t border-indigo-100/50 dark:border-indigo-900/30">
                                            <div className="space-y-2">
                                                <Label className="text-xs font-semibold">WP Username</Label>
                                                <Input
                                                    placeholder="admin"
                                                    value={formData.wpUserName || ''}
                                                    onChange={e => setFormData({ ...formData, wpUserName: e.target.value })}
                                                    className="h-10 rounded-lg"
                                                />
                                            </div>
                                            <div className="space-y-2">
                                                <Label className="text-xs font-semibold">Application Password</Label>
                                                <Input
                                                    type="password"
                                                    placeholder="xxxx xxxx xxxx xxxx"
                                                    value={formData.wordpress_key || ''}
                                                    onChange={e => setFormData({ ...formData, wordpress_key: e.target.value })}
                                                    className="h-10 rounded-lg font-mono"
                                                />
                                            </div>
                                        </div>
                                    )}

                                    <div className="flex justify-end gap-3 pt-2 border-t border-indigo-100/50 dark:border-indigo-900/30">
                                        <Button variant="ghost" onClick={() => { setEditingId(null); setFormData({}); setShowWpFields(false); }}>Cancel</Button>
                                        <Button onClick={handleSaveProject} disabled={isSaving} className="bg-indigo-600 hover:bg-indigo-700 rounded-xl px-8 h-10">
                                            {isSaving ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : <Save className="w-4 h-4 mr-2" />}
                                            {editingId === 'new' ? 'Add Project' : 'Update Project'}
                                        </Button>
                                    </div>
                                </CardContent>
                            </Card>
                        )}

                        {/* Categories Management Section */}
                        <div className="mt-10">
                            <div className="flex items-center justify-between mb-4">
                                <div>
                                    <h2 className="text-lg font-semibold text-gray-900 dark:text-white flex items-center gap-2">
                                        <FolderTree className="w-5 h-5 text-indigo-600" />
                                        Project Categories
                                    </h2>
                                    <p className="text-sm text-muted-foreground mt-1">
                                        Manage primary (Level 1) and secondary (Level 2) categories for your projects.
                                    </p>
                                </div>
                            </div>

                            {/* Project Selector */}
                            <div className="mb-6">
                                <Label className="text-xs font-semibold mb-2 block">Select Project</Label>
                                <select
                                    value={selectedProjectForCategories?.id || ''}
                                    onChange={(e) => {
                                        const project = projects.find(p => p.id === e.target.value);
                                        if (project) handleSelectProjectForCategories(project);
                                    }}
                                    className="h-10 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/40 w-full md:w-96"
                                >
                                    <option value="">Choose a project...</option>
                                    {projects.map(project => (
                                        <option key={project.id} value={project.id}>
                                            {project.domain || project.app_name || 'Unnamed Project'}
                                        </option>
                                    ))}
                                </select>
                            </div>

                            {/* Categories Display & Management */}
                            {selectedProjectForCategories && (
                                <Card className="border-gray-200 dark:border-gray-800">
                                    <CardHeader className="bg-gray-50/50 dark:bg-gray-900/50 border-b">
                                        <div className="flex items-center justify-between">
                                            <div>
                                                <CardTitle className="text-base">
                                                    Categories for {selectedProjectForCategories.domain || selectedProjectForCategories.app_name}
                                                </CardTitle>
                                                <CardDescription>
                                                    Level 1 categories are primary groups. Level 2 categories are subcategories.
                                                </CardDescription>
                                            </div>
                                            <Button
                                                onClick={() => setEditingCategory({ name: '', level: 1 })}
                                                className="h-9 rounded-lg bg-indigo-600 hover:bg-indigo-700"
                                            >
                                                <Plus className="w-4 h-4 mr-2" />
                                                Add Category
                                            </Button>
                                        </div>
                                    </CardHeader>
                                    <CardContent className="p-0">
                                        {categoriesLoading ? (
                                            <div className="p-12 flex justify-center">
                                                <Loader2 className="w-8 h-8 animate-spin text-gray-300" />
                                            </div>
                                        ) : categories.length === 0 ? (
                                            <div className="p-12 text-center">
                                                <FolderTree className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                                                <p className="text-sm text-gray-500">No categories yet. Add your first category to get started.</p>
                                            </div>
                                        ) : (
                                            <div className="divide-y divide-gray-100 dark:divide-gray-800">
                                                {/* Level 1 Categories */}
                                                {categories.filter(c => c.level === 1).map(category => {
                                                    const childCategories = categories.filter(c => c.parent_category_id === category.id);
                                                    return (
                                                        <div key={category.id} className="p-4">
                                                            <div className="flex items-center justify-between">
                                                                <div className="flex items-center gap-3">
                                                                    <div className="w-8 h-8 rounded-lg bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center">
                                                                        <span className="text-xs font-bold text-indigo-600 dark:text-indigo-400">L1</span>
                                                                    </div>
                                                                    <div>
                                                                        <p className="text-sm font-semibold text-gray-900 dark:text-white">{category.name}</p>
                                                                        <p className="text-xs text-gray-500">{childCategories.length} subcategories</p>
                                                                    </div>
                                                                </div>
                                                                <div className="flex items-center gap-2">
                                                                    <Button
                                                                        variant="ghost"
                                                                        size="icon"
                                                                        className="h-8 w-8 text-gray-400 hover:text-indigo-600"
                                                                        onClick={() => setEditingCategory({ id: category.id, name: category.name, level: 1 })}
                                                                    >
                                                                        <Edit2 className="h-4 w-4" />
                                                                    </Button>
                                                                    <Button
                                                                        variant="ghost"
                                                                        size="icon"
                                                                        className="h-8 w-8 text-gray-400 hover:text-red-600"
                                                                        onClick={() => handleDeleteCategory(category.id)}
                                                                    >
                                                                        <Trash2 className="h-4 w-4" />
                                                                    </Button>
                                                                </div>
                                                            </div>

                                                            {/* Child Categories (Level 2) */}
                                                            {childCategories.length > 0 && (
                                                                <div className="mt-3 ml-11 space-y-2">
                                                                    {childCategories.map(child => (
                                                                        <div key={child.id} className="flex items-center justify-between p-2.5 rounded-lg bg-gray-50 dark:bg-gray-800/50">
                                                                            <div className="flex items-center gap-2">
                                                                                <ChevronRight className="w-4 h-4 text-gray-400" />
                                                                                <div className="w-6 h-6 rounded bg-gray-200 dark:bg-gray-700 flex items-center justify-center">
                                                                                    <span className="text-[10px] font-bold text-gray-600 dark:text-gray-400">L2</span>
                                                                                </div>
                                                                                <span className="text-sm text-gray-700 dark:text-gray-300">{child.name}</span>
                                                                            </div>
                                                                            <div className="flex items-center gap-1">
                                                                                <Button
                                                                                    variant="ghost"
                                                                                    size="icon"
                                                                                    className="h-7 w-7 text-gray-400 hover:text-indigo-600"
                                                                                    onClick={() => setEditingCategory({ id: child.id, name: child.name, level: 2, parent_category_id: category.id })}
                                                                                >
                                                                                    <Edit2 className="h-3.5 w-3.5" />
                                                                                </Button>
                                                                                <Button
                                                                                    variant="ghost"
                                                                                    size="icon"
                                                                                    className="h-7 w-7 text-gray-400 hover:text-red-600"
                                                                                    onClick={() => handleDeleteCategory(child.id)}
                                                                                >
                                                                                    <Trash2 className="h-3.5 w-3.5" />
                                                                                </Button>
                                                                            </div>
                                                                        </div>
                                                                    ))}
                                                                </div>
                                                            )}
                                                        </div>
                                                    );
                                                })}
                                            </div>
                                        )}
                                    </CardContent>
                                </Card>
                            )}

                            {/* Edit/Add Category Form */}
                            {editingCategory && selectedProjectForCategories && (
                                <Card className="mt-6 border-indigo-100 dark:border-indigo-900 bg-indigo-50/20 dark:bg-indigo-900/5 overflow-hidden ring-1 ring-indigo-500/20">
                                    <CardHeader className="border-b border-indigo-100/50 dark:border-indigo-900/50">
                                        <CardTitle className="text-lg">
                                            {editingCategory.id ? 'Edit Category' : 'Add New Category'}
                                        </CardTitle>
                                        <CardDescription>
                                            {editingCategory.level === 1
                                                ? 'Level 1 categories are primary groups (e.g., "Audience Growth", "Commercial Intent")'
                                                : 'Level 2 categories are subcategories under a Level 1 parent'}
                                        </CardDescription>
                                    </CardHeader>
                                    <CardContent className="p-6 space-y-6">
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                            <div className="space-y-2">
                                                <Label className="text-xs font-semibold">Category Name</Label>
                                                <Input
                                                    placeholder="e.g., Audience Growth"
                                                    value={editingCategory.name}
                                                    onChange={(e) => setEditingCategory({ ...editingCategory, name: e.target.value })}
                                                    className="h-10 rounded-lg"
                                                />
                                            </div>
                                            <div className="space-y-2">
                                                <Label className="text-xs font-semibold">Category Level</Label>
                                                <select
                                                    value={editingCategory.level}
                                                    onChange={(e) => setEditingCategory({
                                                        ...editingCategory,
                                                        level: parseInt(e.target.value) as 1 | 2,
                                                        parent_category_id: parseInt(e.target.value) === 1 ? undefined : editingCategory.parent_category_id
                                                    })}
                                                    className="h-10 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/40"
                                                >
                                                    <option value={1}>Level 1 (Primary Category)</option>
                                                    <option value={2}>Level 2 (Subcategory)</option>
                                                </select>
                                            </div>
                                            {editingCategory.level === 2 && (
                                                <div className="space-y-2 md:col-span-2">
                                                    <Label className="text-xs font-semibold">Parent Category (Level 1)</Label>
                                                    <select
                                                        value={editingCategory.parent_category_id || ''}
                                                        onChange={(e) => setEditingCategory({ ...editingCategory, parent_category_id: e.target.value })}
                                                        className="h-10 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 px-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/40"
                                                    >
                                                        <option value="">Select a parent category...</option>
                                                        {categories.filter(c => c.level === 1).map(cat => (
                                                            <option key={cat.id} value={cat.id}>{cat.name}</option>
                                                        ))}
                                                    </select>
                                                </div>
                                            )}
                                        </div>

                                        <div className="flex justify-end gap-3 pt-2 border-t border-indigo-100/50 dark:border-indigo-900/30">
                                            <Button
                                                variant="ghost"
                                                onClick={() => setEditingCategory(null)}
                                            >
                                                Cancel
                                            </Button>
                                            <Button
                                                onClick={handleSaveCategory}
                                                disabled={isSavingCategory}
                                                className="bg-indigo-600 hover:bg-indigo-700 rounded-xl px-8 h-10"
                                            >
                                                {isSavingCategory ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : <Save className="w-4 h-4 mr-2" />}
                                                {editingCategory.id ? 'Update Category' : 'Add Category'}
                                            </Button>
                                        </div>
                                    </CardContent>
                                </Card>
                            )}
                        </div>
                    </div>
                </TabsContent>

                {/* Imported Posts */}
                <TabsContent value="posts" className="animate-in fade-in-50 duration-500">
                    <Card className="border-gray-200 dark:border-gray-800 shadow-sm">
                        <CardHeader className="bg-gray-50/50 dark:bg-gray-900/50 border-b flex flex-row items-center justify-between h-auto py-5">
                            <div>
                                <CardTitle>External Articles History</CardTitle>
                                <CardDescription>Articles imported from your WordPress sites for internal linking.</CardDescription>
                            </div>
                            <Button onClick={handleSync} disabled={isSyncing || projects.filter(p => p.wordpress_key).length === 0} className="rounded-xl border-gray-200" variant="outline">
                                <RefreshCw className={cn("w-4 h-4 mr-2", isSyncing && "animate-spin")} />
                                {isSyncing ? "Syncing..." : "Sync Posts"}
                            </Button>
                        </CardHeader>
                        <CardContent className="p-0">
                            {postsLoading ? (
                                <div className="p-12 flex justify-center"><Loader2 className="w-8 h-8 animate-spin text-gray-300" /></div>
                            ) : importedPosts.length === 0 ? (
                                <div className="p-20 text-center space-y-4">
                                    <div className="w-16 h-16 bg-gray-50 dark:bg-gray-900 rounded-full flex items-center justify-center mx-auto">
                                        <FileText className="w-8 h-8 text-gray-300" />
                                    </div>
                                    <p className="text-gray-500">No posts imported yet. Run a sync to fetch articles.</p>
                                </div>
                            ) : (
                                <div className="divide-y divide-gray-100 dark:divide-gray-800">
                                    {importedPosts.map(post => (
                                        <div key={post.id} className="p-5 flex items-center justify-between hover:bg-gray-50/50 dark:hover:bg-gray-900/50 transition-colors group">
                                            <div className="space-y-1 pr-6 flex-1 overflow-hidden">
                                                <h4 className="text-sm font-semibold text-gray-900 dark:text-white truncate group-hover:text-indigo-600 transition-colors">{post.title}</h4>
                                                <div className="flex items-center gap-3">
                                                    <span className="text-[10px] font-bold uppercase px-2 py-0.5 bg-gray-100 dark:bg-gray-800 rounded-md text-gray-500">
                                                        {post.source_site || 'WordPress'}
                                                    </span>
                                                    <a href={post.link} target="_blank" rel="noopener noreferrer" className="text-xs text-indigo-500 hover:underline truncate">
                                                        {post.link}
                                                    </a>
                                                </div>
                                            </div>
                                            <div className="text-xs text-gray-400 font-medium whitespace-nowrap">
                                                {new Date(post.created_at).toLocaleDateString()}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </CardContent>
                    </Card>
                </TabsContent>
            </Tabs>

            {trendProject && (
                <TrendReportModal
                    siteId={trendProject.id}
                    siteDomain={trendProject.domain || trendProject.app_name}
                    isOpen={!!trendProject}
                    onClose={() => setTrendProject(null)}
                />
            )}
        </div>
    );
};

function cn(...classes: any[]) {
    return classes.filter(Boolean).join(' ');
}
