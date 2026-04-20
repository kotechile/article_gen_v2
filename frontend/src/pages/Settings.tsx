
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

    // Categories State (managed per project being edited)
    const [categories, setCategories] = useState<any[]>([]);
    const [categoriesLoading, setCategoriesLoading] = useState(false);
    const [editingCategory, setEditingCategory] = useState<{ id?: string; name: string; level: 1 | 2; parent_category_id?: string } | null>(null);
    const [isSavingCategory, setIsSavingCategory] = useState(false);
    const [isSyncingCategories, setIsSyncingCategories] = useState(false);

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
        if (!editingId || editingId === 'new' || !editingCategory || !user) return;
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
                project_id: editingId,
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

            await fetchCategories(editingId);

            // UX: keep the "Add New Category" panel open after adding,
            // so the user can rapidly add multiple categories (especially Level 2).
            if (editingCategory.id) {
                setEditingCategory(null);
            } else {
                setEditingCategory(prev => prev ? { ...prev, name: '' } : prev);
            }
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

            if (editingId && editingId !== 'new') {
                await fetchCategories(editingId);
            }
        } catch (err: any) {
            alert(err.message || 'Failed to delete category');
        }
    };

    // Fetch categories when editing an existing project
    useEffect(() => {
        if (editingId && editingId !== 'new') {
            fetchCategories(editingId);
        } else {
            setCategories([]);
            setEditingCategory(null);
        }
    }, [editingId]);

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

    const handleSyncProjectCategories = async () => {
        if (!user || !editingId || editingId === 'new') return;
        setIsSyncingCategories(true);
        setError(null);
        setSuccess(null);
        try {
            const response = await fetch('/api/wordpress/sync-project-categories', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: user.id,
                    project_id: editingId
                })
            });

            const payload = await response.json();
            if (!response.ok) {
                throw new Error(payload?.error || 'Category sync failed');
            }

            await fetchCategories(editingId);
            if (payload?.errors_count > 0) {
                setError(payload?.details || 'Category sync completed with errors. Check WordPress credentials and category slugs.');
            } else {
                setSuccess(payload?.details || 'Categories synced with WordPress.');
            }
        } catch (err: any) {
            setError(err?.message || 'Failed to sync categories with WordPress');
        } finally {
            setIsSyncingCategories(false);
        }
    };

    if (loading) return (
        <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
            <Loader2 className="h-10 w-10 animate-spin text-primary" />
            <p className="text-muted-foreground animate-pulse">Loading settings...</p>
        </div>
    );

    return (
        <div className="container mx-auto py-8 max-w-5xl space-y-8">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-border pb-6">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight text-foreground flex items-center gap-3">
                        <SettingsIcon className="w-8 h-8 text-primary" />
                        Settings
                    </h1>
                    <p className="text-muted-foreground mt-1">Manage your analysis thresholds and integrations</p>
                </div>
                {success && (
                    <div className="flex items-center gap-2 px-4 py-2 bg-emerald-500/10 text-emerald-500 dark:text-emerald-400 rounded-lg border border-emerald-500/20 animate-in fade-in slide-in-from-top-2">
                        <CheckCircle2 className="w-4 h-4" />
                        <span className="text-sm font-medium">{success}</span>
                    </div>
                )}
                {error && (
                    <div className="flex items-center gap-2 px-4 py-2 bg-destructive/10 text-destructive rounded-lg border border-destructive/20 animate-in fade-in slide-in-from-top-2">
                        <AlertCircle className="w-4 h-4" />
                        <span className="text-sm font-medium">{error}</span>
                    </div>
                )}
            </div>

                <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
                    <TabsList className="bg-muted/50 p-1 rounded-xl w-full md:w-auto h-auto grid grid-cols-2 md:grid-cols-4 gap-1">
                    <TabsTrigger value="research" className="rounded-lg data-[state=active]:bg-background data-[state=active]:text-primary data-[state=active]:shadow-sm py-2.5">
                        <Search className="w-4 h-4 mr-2" />
                        Topic Research
                    </TabsTrigger>
                    <TabsTrigger value="content" className="rounded-lg data-[state=active]:bg-background data-[state=active]:text-primary data-[state=active]:shadow-sm py-2.5">
                        <Wand2 className="w-4 h-4 mr-2" />
                        Content Generation
                    </TabsTrigger>
                    <TabsTrigger value="niches" className="rounded-lg data-[state=active]:bg-background data-[state=active]:text-primary data-[state=active]:shadow-sm py-2.5">
                        <Layout className="w-4 h-4 mr-2" />
                        Projects: Niches/Websites
                    </TabsTrigger>
                    <TabsTrigger value="posts" className="rounded-lg data-[state=active]:bg-background data-[state=active]:text-primary data-[state=active]:shadow-sm py-2.5">
                        <FileText className="w-4 h-4 mr-2" />
                        External Posts
                    </TabsTrigger>
                </TabsList>

                {/* Research Settings */}
                <TabsContent value="research" className="animate-in fade-in-50 duration-500">
                    <Card className="border-border shadow-sm overflow-hidden">
                        <CardHeader className="bg-muted/30 border-b border-border">
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

                                <div className="flex items-center justify-between p-4 bg-primary/10 rounded-2xl border border-primary/20">
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

                            <div className="flex justify-end pt-4 border-t border-border">
                                <Button onClick={handleSaveResearch} disabled={saving} className="h-11 px-8 rounded-xl bg-primary hover:bg-primary/90 shadow-lg">
                                    {saving ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : <Save className="w-4 h-4 mr-2" />}
                                    Save Research Settings
                                </Button>
                            </div>
                        </CardContent>
                    </Card>
                </TabsContent>

                {/* Content Generation Settings */}
                <TabsContent value="content" className="animate-in fade-in-50 duration-500">
                    <Card className="border-border shadow-sm">
                        <CardHeader className="bg-muted/30 border-b border-border">
                            <CardTitle>AI &amp; Media API Keys</CardTitle>
                            <CardDescription>
                                Configure the credentials used for content generation and media sourcing.
                            </CardDescription>
                        </CardHeader>
                        <CardContent className="p-6 space-y-8">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-8">
                                {/* LLM Keys */}
                                <div className="space-y-6">
                                    <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground flex items-center gap-2">
                                        <div className="w-1.5 h-1.5 rounded-full bg-primary"></div>
                                        Core LLM Providers
                                    </h3>
                                    <div className="space-y-4">
                                        <div className="space-y-2">
                                            <Label className="text-xs font-semibold">OpenAI API Key</Label>
                                            <Input type="password" value={appSettings.openAIKey || ''} onChange={e => setAppSettings({ ...appSettings, openAIKey: e.target.value })} className="h-10 rounded-lg font-mono text-sm" placeholder="sk-..." />
                                        </div>
                                        <div className="space-y-2">
                                            <Label className="text-xs font-semibold">Google Gemini Key</Label>
                                            <Input type="password" value={appSettings.geminiKey || ''} onChange={e => setAppSettings({ ...appSettings, geminiKey: e.target.value })} className="h-10 rounded-lg font-mono text-sm" placeholder="AIza..." />
                                        </div>
                                        <div className="space-y-2">
                                            <Label className="text-xs font-semibold">Claude (Anthropic) Key</Label>
                                            <Input type="password" value={appSettings.claudeKey || ''} onChange={e => setAppSettings({ ...appSettings, claudeKey: e.target.value })} className="h-10 rounded-lg font-mono text-sm" placeholder="sk-ant-..." />
                                        </div>
                                        <div className="space-y-2">
                                            <Label className="text-xs font-semibold">Perplexity AI Key</Label>
                                            <Input type="password" value={appSettings.perplexityAI_key || ''} onChange={e => setAppSettings({ ...appSettings, perplexityAI_key: e.target.value })} className="h-10 rounded-lg font-mono text-sm" placeholder="pplx-..." />
                                        </div>
                                    </div>
                                </div>

                                {/* Media & Utils */}
                                <div className="space-y-6">
                                    <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground flex items-center gap-2">
                                        <div className="w-1.5 h-1.5 rounded-full bg-orange-500"></div>
                                        Media &amp; Utils
                                    </h3>
                                    <div className="space-y-4">
                                        <div className="space-y-2">
                                            <Label className="text-xs font-semibold">YouTube API Key</Label>
                                            <Input type="password" value={appSettings.youTubeKey || ''} onChange={e => setAppSettings({ ...appSettings, youTubeKey: e.target.value })} className="h-10 rounded-lg font-mono text-sm" />
                                        </div>
                                        <div className="space-y-2">
                                            <Label className="text-xs font-semibold">Unsplash Access Key</Label>
                                            <Input type="password" value={appSettings.unsplashKey || ''} onChange={e => setAppSettings({ ...appSettings, unsplashKey: e.target.value })} className="h-10 rounded-lg font-mono text-sm" />
                                        </div>
                                        <div className="space-y-2">
                                            <Label className="text-xs font-semibold">Stability AI Key</Label>
                                            <Input type="password" value={appSettings.stabilityAiKey || ''} onChange={e => setAppSettings({ ...appSettings, stabilityAiKey: e.target.value })} className="h-10 rounded-lg font-mono text-sm" />
                                        </div>
                                        <div className="space-y-2">
                                            <Label className="text-xs font-semibold">Flux Key</Label>
                                            <Input type="password" value={appSettings.fluxKey || ''} onChange={e => setAppSettings({ ...appSettings, fluxKey: e.target.value })} className="h-10 rounded-lg font-mono text-sm" />
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="flex justify-end pt-4 border-t border-border">
                                <Button onClick={handleSaveAppSettings} disabled={saving} className="h-11 px-8 rounded-xl bg-primary hover:bg-primary/90 shadow-lg">
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
                        <div className="flex items-start gap-3 p-4 rounded-xl bg-primary/10 border border-primary/20">
                            <Globe className="w-5 h-5 text-primary flex-shrink-0 mt-0.5" />
                            <div>
                                <p className="text-sm font-semibold text-foreground">Projects:Niches &amp; Websites</p>
                                <p className="text-xs text-muted-foreground mt-0.5">
                                    Add any project here — a WordPress site with full sync, or just a niche description for AI-driven content without a website. The active project on your Command Center is selected from this list.
                                </p>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                            {projects.map(project => (
                                <Card key={project.id} className="relative group overflow-hidden border-border hover:border-ring/50 transition-all">
                                    <div className="absolute top-0 left-0 w-1 h-full bg-primary group-hover:w-2 transition-all"></div>
                                    <CardHeader className="pb-2">
                                        <div className="flex justify-between items-start">
                                            <div>
                                                <CardTitle className="text-lg truncate max-w-[180px]">
                                                    {project.domain || project.app_name || 'Unnamed Project'}
                                                </CardTitle>
                                                <CardDescription className="flex items-center gap-1.5 mt-1">
                                                    {project.domain
                                                        ? <><span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span> WordPress</>
                                                        : <><span className="w-1.5 h-1.5 rounded-full bg-primary"></span> Niche Only</>
                                                    }
                                                </CardDescription>
                                            </div>
                                            <div className="flex gap-1">
                                                <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-primary" onClick={() => { setEditingId(project.id); setFormData(project); setShowWpFields(!!project.wordpress_key || !!project.wpUserName); }}>
                                                    <Edit2 className="h-4 w-4" />
                                                </Button>
                                                <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-red-500 dark:hover:text-red-400" onClick={() => handleDeleteProject(project.id)}>
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
                                                className="h-7 text-xs gap-1.5 border-primary/20 text-primary hover:bg-primary/10"
                                                onClick={() => setTrendProject(project)}
                                            >
                                                <TrendingUp className="w-3 h-3" />
                                                What's Trending
                                            </Button>
                                            <Globe className="w-4 h-4 text-muted-foreground" />
                                        </div>
                                    </CardContent>
                                </Card>
                            ))}

                            <button
                                onClick={() => { setEditingId('new'); setFormData({}); setShowWpFields(false); }}
                                className="h-[160px] border-2 border-dashed border-border rounded-2xl flex flex-col items-center justify-center gap-2 text-muted-foreground hover:border-ring hover:text-primary hover:bg-primary/10 transition-all group"
                            >
                                <Plus className="w-8 h-8 group-hover:scale-110 transition-transform" />
                                <span className="text-sm font-medium">Add Niche / Website</span>
                            </button>
                        </div>

                        {/* Edit / Create Form */}
                        {editingId && (
                            <Card className="border-primary/20 bg-primary/5 overflow-hidden ring-1 ring-primary/20">
                                <CardHeader className="border-b border-primary/20">
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
                                            <Label className="text-xs font-semibold">Project / Niche Name <span className="text-primary">*</span></Label>
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
                                            <Label className="text-xs font-semibold">Niche Description (used by AI) <span className="text-primary">*</span></Label>
                                            <textarea
                                                placeholder="E.g. A home improvement blog focusing on DIY renovations, sustainable materials, and budget-friendly interior design for first-time homeowners."
                                                value={formData.site_description || ''}
                                                onChange={e => setFormData({ ...formData, site_description: e.target.value })}
                                                rows={6}
                                                className="w-full px-3 py-2 text-sm rounded-lg border border-input bg-background resize-none focus:outline-none focus:ring-2 focus:ring-ring/40"
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
                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-2 border-t border-primary/20">
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

                                    {/* Categories Management - Only show when editing existing project */}
                                    {editingId && editingId !== 'new' && (
                                        <div className="pt-6 border-t border-primary/20">
                                            <div className="flex items-center justify-between mb-4">
                                                <div>
                                                    <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
                                                        <FolderTree className="w-4 h-4 text-primary" />
                                                        Categories
                                                    </h3>
                                                    <p className="text-xs text-muted-foreground mt-1">
                                                        Manage categories for {formData.domain || formData.app_name}
                                                    </p>
                                                </div>
                                                <div className="flex items-center gap-2">
                                                    <Button
                                                        onClick={handleSyncProjectCategories}
                                                        size="sm"
                                                        variant="outline"
                                                        disabled={isSyncingCategories}
                                                        className="h-8 rounded-lg border-primary/30 text-primary hover:bg-primary/10"
                                                    >
                                                        <RefreshCw className={cn("w-3.5 h-3.5 mr-1", isSyncingCategories && "animate-spin")} />
                                                        {isSyncingCategories ? 'Syncing...' : 'Sync Categories with WordPress'}
                                                    </Button>
                                                    <Button
                                                        onClick={() => {
                                                            setEditingCategory({ name: '', level: 1 });
                                                            fetchCategories(editingId);
                                                        }}
                                                        size="sm"
                                                        className="h-8 rounded-lg bg-primary hover:bg-primary/90"
                                                    >
                                                        <Plus className="w-3.5 h-3.5 mr-1" />
                                                        Add Category
                                                    </Button>
                                                </div>
                                            </div>

                                            {/* Categories List */}
                                            {categoriesLoading ? (
                                                <div className="py-8 flex justify-center">
                                                    <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
                                                </div>
                                            ) : categories.length === 0 ? (
                                                <div className="py-8 text-center">
                                                    <FolderTree className="w-8 h-8 text-muted-foreground mx-auto mb-2" />
                                                    <p className="text-xs text-muted-foreground">No categories yet. Add your first category to get started.</p>
                                                </div>
                                            ) : (
                                                <div className="space-y-3">
                                                    {/* Level 1 Categories */}
                                                    {categories.filter(c => c.level === 1).map(category => {
                                                        const childCategories = categories.filter(c => c.parent_category_id === category.id);
                                                        return (
                                                            <div key={category.id} className="rounded-lg border border-border overflow-hidden">
                                                                <div className="flex items-center justify-between p-3 bg-muted/30">
                                                                    <div className="flex items-center gap-2">
                                                                        <div className="w-6 h-6 rounded bg-primary/10 flex items-center justify-center">
                                                                            <span className="text-[10px] font-bold text-primary">L1</span>
                                                                        </div>
                                                                        <div>
                                                                            <p className="text-sm font-medium text-foreground">{category.name}</p>
                                                                            <p className="text-[10px] text-muted-foreground">{childCategories.length} subcategories</p>
                                                                        </div>
                                                                    </div>
                                                                    <div className="flex items-center gap-1">
                                                                        <Button
                                                                            variant="ghost"
                                                                            size="icon"
                                                                            className="h-7 w-7 text-muted-foreground hover:text-primary"
                                                                            onClick={() => {
                                                                                setEditingCategory({ id: category.id, name: category.name, level: 1 });
                                                                                fetchCategories(editingId);
                                                                            }}
                                                                        >
                                                                            <Edit2 className="h-3.5 w-3.5" />
                                                                        </Button>
                                                                        <Button
                                                                            variant="ghost"
                                                                            size="icon"
                                                                            className="h-7 w-7 text-muted-foreground hover:text-red-500 dark:hover:text-red-400"
                                                                            onClick={() => handleDeleteCategory(category.id)}
                                                                        >
                                                                            <Trash2 className="h-3.5 w-3.5" />
                                                                        </Button>
                                                                    </div>
                                                                </div>

                                                                {/* Child Categories (Level 2) */}
                                                                {childCategories.length > 0 && (
                                                                    <div className="bg-background divide-y divide-border">
                                                                        {childCategories.map(child => (
                                                                            <div key={child.id} className="flex items-center justify-between p-2.5 pl-8">
                                                                                <div className="flex items-center gap-2">
                                                                                    <ChevronRight className="w-3.5 h-3.5 text-muted-foreground" />
                                                                                    <div className="w-5 h-5 rounded bg-muted/50 flex items-center justify-center">
                                                                                        <span className="text-[9px] font-bold text-muted-foreground">L2</span>
                                                                                    </div>
                                                                                    <span className="text-sm text-foreground">{child.name}</span>
                                                                                </div>
                                                                                <div className="flex items-center gap-1">
                                                                                    <Button
                                                                                        variant="ghost"
                                                                                        size="icon"
                                                                                        className="h-6 w-6 text-muted-foreground hover:text-primary"
                                                                                        onClick={() => {
                                                                                            setEditingCategory({ id: child.id, name: child.name, level: 2, parent_category_id: category.id });
                                                                                            fetchCategories(editingId);
                                                                                        }}
                                                                                    >
                                                                                        <Edit2 className="h-3 w-3" />
                                                                                    </Button>
                                                                                    <Button
                                                                                        variant="ghost"
                                                                                        size="icon"
                                                                                        className="h-6 w-6 text-muted-foreground hover:text-red-500 dark:hover:text-red-400"
                                                                                        onClick={() => handleDeleteCategory(child.id)}
                                                                                    >
                                                                                        <Trash2 className="h-3 w-3" />
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

                                            {/* Edit/Add Category Form */}
                                            {editingCategory && (
                                                <div className="mt-4 p-4 rounded-lg border border-primary/20 bg-primary/5">
                                                    <div className="space-y-4">
                                                        <div>
                                                            <p className="text-sm font-medium text-foreground mb-3">
                                                                {editingCategory.id ? 'Edit Category' : 'Add New Category'}
                                                            </p>
                                                            <div className="grid grid-cols-1 gap-3">
                                                                <div>
                                                                    <Label className="text-xs font-semibold mb-1 block">Category Name</Label>
                                                                    <Input
                                                                        placeholder="e.g., Audience Growth"
                                                                        value={editingCategory.name}
                                                                        onChange={(e) => setEditingCategory({ ...editingCategory, name: e.target.value })}
                                                                        className="h-9 rounded-lg"
                                                                    />
                                                                </div>
                                                                <div>
                                                                    <Label className="text-xs font-semibold mb-1 block">Category Level</Label>
                                                                    <select
                                                                        value={editingCategory.level}
                                                                        onChange={(e) => setEditingCategory({
                                                                            ...editingCategory,
                                                                            level: parseInt(e.target.value) as 1 | 2,
                                                                            parent_category_id: parseInt(e.target.value) === 1 ? undefined : editingCategory.parent_category_id
                                                                        })}
                                                                        className="h-9 rounded-lg border border-border bg-background px-3 text-sm"
                                                                    >
                                                                        <option value={1}>Level 1 (Primary)</option>
                                                                        <option value={2}>Level 2 (Subcategory)</option>
                                                                    </select>
                                                                </div>
                                                                {editingCategory.level === 2 && (
                                                                    <div>
                                                                        <Label className="text-xs font-semibold mb-1 block">Parent Category</Label>
                                                                        <select
                                                                            value={editingCategory.parent_category_id || ''}
                                                                            onChange={(e) => setEditingCategory({ ...editingCategory, parent_category_id: e.target.value })}
                                                                            className="h-9 rounded-lg border border-border bg-background px-3 text-sm"
                                                                        >
                                                                            <option value="">Select parent...</option>
                                                                            {categories.filter(c => c.level === 1).map(cat => (
                                                                                <option key={cat.id} value={cat.id}>{cat.name}</option>
                                                                            ))}
                                                                        </select>
                                                                    </div>
                                                                )}
                                                            </div>
                                                        </div>
                                                        <div className="flex items-center gap-2 pt-2 border-t border-primary/20">
                                                            <Button
                                                                size="sm"
                                                                variant="ghost"
                                                                onClick={() => setEditingCategory(null)}
                                                            >
                                                                Cancel
                                                            </Button>
                                                            <Button
                                                                size="sm"
                                                                onClick={handleSaveCategory}
                                                                disabled={isSavingCategory}
                                                                className="bg-primary hover:bg-primary/90 rounded-lg"
                                                            >
                                                                {isSavingCategory ? <Loader2 className="w-3.5 h-3.5 animate-spin mr-2" /> : <Save className="w-3.5 h-3.5 mr-2" />}
                                                                {editingCategory.id ? 'Update' : 'Add Category'}
                                                            </Button>
                                                        </div>
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    )}

                                    <div className="flex justify-end gap-3 pt-2 border-t border-primary/20">
                                        <Button variant="ghost" onClick={() => { setEditingId(null); setFormData({}); setShowWpFields(false); setEditingCategory(null); }}>Cancel</Button>
                                        <Button onClick={handleSaveProject} disabled={isSaving} className="bg-primary hover:bg-primary/90 rounded-xl px-8 h-10">
                                            {isSaving ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : <Save className="w-4 h-4 mr-2" />}
                                            {editingId === 'new' ? 'Add Project' : 'Update Project'}
                                        </Button>
                                    </div>
                                </CardContent>
                            </Card>
                        )}
                    </div>
                </TabsContent>

                {/* Imported Posts */}
                <TabsContent value="posts" className="animate-in fade-in-50 duration-500">
                    <Card className="border-border shadow-sm">
                        <CardHeader className="bg-muted/30 border-b border-border flex flex-row items-center justify-between h-auto py-5">
                            <div>
                                <CardTitle>External Articles History</CardTitle>
                                <CardDescription>Articles imported from your WordPress sites for internal linking.</CardDescription>
                            </div>
                            <Button onClick={handleSync} disabled={isSyncing || projects.filter(p => p.wordpress_key).length === 0} className="rounded-xl border-border" variant="outline">
                                <RefreshCw className={cn("w-4 h-4 mr-2", isSyncing && "animate-spin")} />
                                {isSyncing ? "Syncing..." : "Sync Posts"}
                            </Button>
                        </CardHeader>
                        <CardContent className="p-0">
                            {postsLoading ? (
                                <div className="p-12 flex justify-center"><Loader2 className="w-8 h-8 animate-spin text-muted-foreground" /></div>
                            ) : importedPosts.length === 0 ? (
                                <div className="p-20 text-center space-y-4">
                                    <div className="w-16 h-16 bg-muted/50 rounded-full flex items-center justify-center mx-auto">
                                        <FileText className="w-8 h-8 text-muted-foreground" />
                                    </div>
                                    <p className="text-muted-foreground">No posts imported yet. Run a sync to fetch articles.</p>
                                </div>
                            ) : (
                                <div className="divide-y divide-border">
                                    {importedPosts.map(post => (
                                        <div key={post.id} className="p-5 flex items-center justify-between hover:bg-muted/30 transition-colors group">
                                            <div className="space-y-1 pr-6 flex-1 overflow-hidden">
                                                <h4 className="text-sm font-semibold text-foreground truncate group-hover:text-primary transition-colors">{post.title}</h4>
                                                <div className="flex items-center gap-3">
                                                    <span className="text-[10px] font-bold uppercase px-2 py-0.5 bg-muted/50 rounded-md text-muted-foreground">
                                                        {post.source_site || 'WordPress'}
                                                    </span>
                                                    <a href={post.link} target="_blank" rel="noopener noreferrer" className="text-xs text-primary hover:underline truncate">
                                                        {post.link}
                                                    </a>
                                                </div>
                                            </div>
                                            <div className="text-xs text-muted-foreground font-medium whitespace-nowrap">
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
