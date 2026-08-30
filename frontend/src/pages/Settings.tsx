
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
    ChevronRight,
    Share2
} from 'lucide-react';
import { TrendReportModal } from '../components/TrendReportModal';
import { supabase } from '../lib/supabase';
import { useAuth } from '../context/auth-context';
import { useProject } from '../context/project-context';
import { apiClient } from '../api-client';
import {
    getLinkedInAccount,
    getLinkedInAuthUrl,
    disconnectLinkedInAccount,
    type LinkedInAccountStatus
} from '../services/linkedinService';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Switch } from '../components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import type { Project } from '../types';

// Read tab from URL query (?tab=niches or ?tab=integrations)
function getInitialTab() {
    const params = new URLSearchParams(window.location.search);
    const tab = params.get('tab');
    if (tab === 'niches' || tab === 'posts' || tab === 'content' || tab === 'integrations' || tab === 'linkedin') {
        return tab === 'linkedin' ? 'integrations' : tab;
    }
    return 'research';
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

type SvgPromptVersion = 'prompt1' | 'prompt2';

const normalizeOptionalHexColor = (value?: string) => {
    const trimmed = String(value || '').trim();
    if (!trimmed) return '';
    return /^#?[0-9a-fA-F]{6}$/.test(trimmed)
        ? (trimmed.startsWith('#') ? trimmed : `#${trimmed}`)
        : trimmed;
};

const normalizeProjectDomain = (value?: string) =>
    String(value || '')
        .trim()
        .toLowerCase()
        .replace(/^https?:\/\//, '')
        .replace(/\/$/, '');

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
    const [svgPromptVersion, setSvgPromptVersion] = useState<SvgPromptVersion>('prompt1');

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

    // Categories State (managed per project being editing)
    const [categories, setCategories] = useState<any[]>([]);
    const [categoriesLoading, setCategoriesLoading] = useState(false);
    const [editingCategory, setEditingCategory] = useState<{
        id?: string;
        name: string;
        description?: string;
        level: 1 | 2;
        parent_category_id?: string;
    } | null>(null);
    const [isSavingCategory, setIsSavingCategory] = useState(false);
    const [isSyncingCategories, setIsSyncingCategories] = useState(false);

    // LinkedIn Integration State
    const [linkedInStatus, setLinkedInStatus] = useState<LinkedInAccountStatus | null>(null);
    const [linkedInLoading, setLinkedInLoading] = useState(false);
    const [disconnectingLinkedIn, setDisconnectingLinkedIn] = useState(false);

    useEffect(() => {
        // Check for LinkedIn OAuth redirect callback params
        const params = new URLSearchParams(window.location.search);
        if (params.get('linkedin_connected') === 'true') {
            setSuccess('LinkedIn personal account successfully connected!');
            // Clean URL
            window.history.replaceState({}, document.title, window.location.pathname + '?tab=integrations');
        } else if (params.get('linkedin_error')) {
            setError(`LinkedIn connection failed: ${params.get('linkedin_error')}`);
            window.history.replaceState({}, document.title, window.location.pathname + '?tab=integrations');
        }
    }, []);

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
                fetchInfographicSvgSettings(),
                fetchProjects(),
                fetchImportedPosts(),
                fetchLinkedInAccountStatus()
            ]);
        } catch (err) {
            console.error("Error fetching settings:", err);
            setError("Failed to load some settings.");
        } finally {
            setLoading(false);
        }
    };

    const fetchLinkedInAccountStatus = async () => {
        try {
            setLinkedInLoading(true);
            const status = await getLinkedInAccount();
            setLinkedInStatus(status);
        } catch (err) {
            console.warn("Could not load LinkedIn account status:", err);
        } finally {
            setLinkedInLoading(false);
        }
    };

    const handleConnectLinkedIn = async () => {
        try {
            const authUrl = await getLinkedInAuthUrl();
            window.location.href = authUrl;
        } catch (err: any) {
            setError(err?.message || 'Failed to initiate LinkedIn connection');
        }
    };

    const handleDisconnectLinkedIn = async () => {
        if (!confirm('Are you sure you want to disconnect your LinkedIn account?')) return;
        try {
            setDisconnectingLinkedIn(true);
            await disconnectLinkedInAccount();
            await fetchLinkedInAccountStatus();
            setSuccess('LinkedIn account disconnected.');
        } catch (err: any) {
            setError(err?.message || 'Failed to disconnect LinkedIn account.');
        } finally {
            setDisconnectingLinkedIn(false);
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

    const fetchInfographicSvgSettings = async () => {
        try {
            const response = await apiClient.get<any>('/settings/infographic-svg');
            if (response?.success && response?.data?.svg_prompt_version) {
                const value = String(response.data.svg_prompt_version).trim().toLowerCase();
                setSvgPromptVersion(value === 'prompt2' ? 'prompt2' : 'prompt1');
            }
        } catch (err) {
            console.error("Infographic SVG settings fetch failed:", err);
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
                description: (editingCategory.description || '').trim() || null,
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
                setEditingCategory(prev => prev ? { ...prev, name: '', description: '' } : prev);
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
            const svgSettingsResult = await apiClient.post<any>('/settings/infographic-svg', {
                svg_prompt_version: svgPromptVersion,
            });
            if (!svgSettingsResult?.success) {
                throw new Error(svgSettingsResult?.message || 'Failed to save SVG prompt settings');
            }
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
            const normalizedDomain = normalizeProjectDomain(formData.domain) || null;
            const projectsPayload = {
                user_id: user.id,
                app_name: formData.app_name || null,
                domain: normalizedDomain,
                site_description: formData.site_description || null,
                websitedescription: formData.websitedescription || null,
                targetaudiencedescription: formData.targetaudiencedescription || formData.targetAudienceDescription || null,
                wordpress_key: formData.wordpress_key || null,
                wpusername: formData.wpusername || formData.wpUserName || null,
                brand_primary_color: normalizeOptionalHexColor(formData.brand_primary_color) || null,
                brand_text_color: normalizeOptionalHexColor(formData.brand_text_color) || null,
                brand_secondary_color: normalizeOptionalHexColor(formData.brand_secondary_color) || null,
                brand_neutral_color: normalizeOptionalHexColor(formData.brand_neutral_color) || null,
                site_url_override: formData.site_url_override?.trim() || null,
                social_default_image_url: formData.social_default_image_url?.trim() || null,
                branding_updated_at: new Date().toISOString(),
            };
            let saveError;
            if (editingId && editingId !== 'new') {
                const { error: err } = await supabase.from('projects').update(projectsPayload).eq('id', editingId);
                saveError = err;
            } else {
                const { error: err } = await supabase.from('projects').insert([projectsPayload]);
                saveError = err;
            }
            if (saveError) throw saveError;

            if (normalizedDomain) {
                const wpBrandPayload = {
                    user_id: user.id,
                    domain: normalizedDomain,
                    app_name: projectsPayload.app_name || null,
                    wpUserName: formData.wpUserName || formData.wpusername || null,
                    wordpress_key: formData.wordpress_key || null,
                    seo_plugin: formData.seo_plugin || 'unknown',
                    cms: formData.cms_url ? normalizeProjectDomain(formData.cms_url) : null,
                    cms_url: formData.cms_url?.trim() || null,
                    site_url_override: projectsPayload.site_url_override,
                    social_default_image_url: projectsPayload.social_default_image_url,
                    brand_primary_color: projectsPayload.brand_primary_color,
                    brand_text_color: projectsPayload.brand_text_color,
                    brand_secondary_color: projectsPayload.brand_secondary_color,
                    brand_neutral_color: projectsPayload.brand_neutral_color,
                    branding_updated_at: projectsPayload.branding_updated_at,
                };

                const { data: existingWpSite } = await supabase
                    .from('wordPress_details')
                    .select('id')
                    .eq('user_id', user.id)
                    .eq('domain', normalizedDomain)
                    .maybeSingle();

                if (existingWpSite?.id) {
                    await supabase
                        .from('wordPress_details')
                        .update(wpBrandPayload)
                        .eq('id', existingWpSite.id);
                } else if (showWpFields || projectsPayload.brand_primary_color || projectsPayload.brand_text_color || projectsPayload.brand_secondary_color || projectsPayload.brand_neutral_color) {
                    await supabase
                        .from('wordPress_details')
                        .insert([wpBrandPayload]);
                }
            }

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
            await apiClient.post<any>('/wordpress/sync-posts', { user_id: user.id });
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
            const payload = await apiClient.post<any>('/wordpress/sync-project-categories', {
                user_id: user.id,
                project_id: editingId
            });

            await fetchCategories(editingId);
            if (payload?.errors_count > 0) {
                setError(payload?.details || 'Category sync completed with errors. Check WordPress credentials and category slugs.');
            } else {
                setSuccess(payload?.details || 'Categories synced with WordPress.');
            }
        } catch (err: any) {
            const serverMessage =
                err?.response?.data?.details ||
                err?.response?.data?.message ||
                err?.response?.data?.error;
            setError(serverMessage || err?.message || 'Failed to sync categories with WordPress');
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
                    <TabsList className="bg-muted/50 p-1 rounded-xl w-full md:w-auto h-auto grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-1">
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
                    <TabsTrigger value="integrations" className="rounded-lg data-[state=active]:bg-background data-[state=active]:text-primary data-[state=active]:shadow-sm py-2.5">
                        <Share2 className="w-4 h-4 mr-2 text-[#0A66C2]" />
                        LinkedIn
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

                            <div className="rounded-2xl border border-border bg-muted/20 p-5 space-y-4">
                                <div>
                                    <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground flex items-center gap-2">
                                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500"></div>
                                        SVG Infographic Prompt
                                    </h3>
                                    <p className="text-sm text-muted-foreground mt-2">
                                        Choose which backend SVG prompt should be used for inline infographic generation from selected article text.
                                    </p>
                                </div>

                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <button
                                        type="button"
                                        onClick={() => setSvgPromptVersion('prompt1')}
                                        className={`rounded-xl border p-4 text-left transition-all ${
                                            svgPromptVersion === 'prompt1'
                                                ? 'border-primary bg-primary/10 ring-1 ring-primary/30'
                                                : 'border-border bg-background hover:border-primary/40'
                                        }`}
                                    >
                                        <div className="flex items-center justify-between gap-3">
                                            <div>
                                                <p className="text-sm font-semibold">SVG Prompt 1</p>
                                                <p className="text-xs text-muted-foreground mt-1">
                                                    Minimal Bento-box layout with precision geometry and restrained color.
                                                </p>
                                            </div>
                                            <div className={`h-4 w-4 rounded-full border ${svgPromptVersion === 'prompt1' ? 'border-primary bg-primary' : 'border-muted-foreground/40 bg-transparent'}`} />
                                        </div>
                                    </button>

                                    <button
                                        type="button"
                                        onClick={() => setSvgPromptVersion('prompt2')}
                                        className={`rounded-xl border p-4 text-left transition-all ${
                                            svgPromptVersion === 'prompt2'
                                                ? 'border-primary bg-primary/10 ring-1 ring-primary/30'
                                                : 'border-border bg-background hover:border-primary/40'
                                        }`}
                                    >
                                        <div className="flex items-center justify-between gap-3">
                                            <div>
                                                <p className="text-sm font-semibold">SVG Prompt 2</p>
                                                <p className="text-xs text-muted-foreground mt-1">
                                                    Editorial infographic style with richer illustration, asymmetry, and custom spatial storytelling.
                                                </p>
                                            </div>
                                            <div className={`h-4 w-4 rounded-full border ${svgPromptVersion === 'prompt2' ? 'border-primary bg-primary' : 'border-muted-foreground/40 bg-transparent'}`} />
                                        </div>
                                    </button>
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
                                        <div className="space-y-3 md:col-span-2 rounded-xl border border-border bg-background p-4">
                                            <div>
                                                <Label className="text-xs font-semibold">Brand Colors</Label>
                                                <p className="text-xs text-muted-foreground mt-1">
                                                    Used for infographic generation and site-aware visual styling. These act as the project default.
                                                </p>
                                            </div>
                                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                                <div className="space-y-2">
                                                    <Label className="text-xs font-semibold">Primary Accent</Label>
                                                    <div className="flex items-center gap-3">
                                                        <Input
                                                            type="color"
                                                            value={normalizeOptionalHexColor(formData.brand_primary_color) || '#3b82f6'}
                                                            onChange={e => setFormData({ ...formData, brand_primary_color: e.target.value })}
                                                            className="h-10 w-16 rounded-lg p-1"
                                                        />
                                                        <Input
                                                            placeholder="#3b82f6"
                                                            value={formData.brand_primary_color || ''}
                                                            onChange={e => setFormData({ ...formData, brand_primary_color: e.target.value })}
                                                            className="h-10 rounded-lg font-mono"
                                                        />
                                                    </div>
                                                </div>
                                                <div className="space-y-2">
                                                    <Label className="text-xs font-semibold">Primary Text</Label>
                                                    <div className="flex items-center gap-3">
                                                        <Input
                                                            type="color"
                                                            value={normalizeOptionalHexColor(formData.brand_text_color) || '#1e293b'}
                                                            onChange={e => setFormData({ ...formData, brand_text_color: e.target.value })}
                                                            className="h-10 w-16 rounded-lg p-1"
                                                        />
                                                        <Input
                                                            placeholder="#1e293b"
                                                            value={formData.brand_text_color || ''}
                                                            onChange={e => setFormData({ ...formData, brand_text_color: e.target.value })}
                                                            className="h-10 rounded-lg font-mono"
                                                        />
                                                    </div>
                                                </div>
                                                <div className="space-y-2">
                                                    <Label className="text-xs font-semibold">Secondary Accent (optional)</Label>
                                                    <div className="flex items-center gap-3">
                                                        <Input
                                                            type="color"
                                                            value={normalizeOptionalHexColor(formData.brand_secondary_color) || '#60a5fa'}
                                                            onChange={e => setFormData({ ...formData, brand_secondary_color: e.target.value })}
                                                            className="h-10 w-16 rounded-lg p-1"
                                                        />
                                                        <Input
                                                            placeholder="#60a5fa"
                                                            value={formData.brand_secondary_color || ''}
                                                            onChange={e => setFormData({ ...formData, brand_secondary_color: e.target.value })}
                                                            className="h-10 rounded-lg font-mono"
                                                        />
                                                    </div>
                                                </div>
                                                <div className="space-y-2">
                                                    <Label className="text-xs font-semibold">Neutral (optional)</Label>
                                                    <div className="flex items-center gap-3">
                                                        <Input
                                                            type="color"
                                                            value={normalizeOptionalHexColor(formData.brand_neutral_color) || '#94a3b8'}
                                                            onChange={e => setFormData({ ...formData, brand_neutral_color: e.target.value })}
                                                            className="h-10 w-16 rounded-lg p-1"
                                                        />
                                                        <Input
                                                            placeholder="#94a3b8"
                                                            value={formData.brand_neutral_color || ''}
                                                            onChange={e => setFormData({ ...formData, brand_neutral_color: e.target.value })}
                                                            className="h-10 rounded-lg font-mono"
                                                        />
                                                    </div>
                                                </div>
                                            </div>
                                            <div className="flex items-center gap-2 pt-1">
                                                {[formData.brand_primary_color, formData.brand_text_color, formData.brand_secondary_color, formData.brand_neutral_color]
                                                    .map((color) => normalizeOptionalHexColor(color))
                                                    .filter(Boolean)
                                                    .map((color) => (
                                                        <span
                                                            key={color}
                                                            className="h-8 w-8 rounded-full border border-border shadow-sm"
                                                            style={{ backgroundColor: color }}
                                                            title={color}
                                                        />
                                                    ))}
                                            </div>
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
                                            <div className="space-y-2">
                                                <Label className="text-xs font-semibold">SEO Plugin</Label>
                                                <select
                                                    value={formData.seo_plugin || 'unknown'}
                                                    onChange={e => setFormData({ ...formData, seo_plugin: e.target.value as Project['seo_plugin'] })}
                                                    className="h-10 w-full rounded-lg border border-input bg-background px-3 text-sm"
                                                >
                                                    <option value="unknown">Unknown / auto fallback</option>
                                                    <option value="yoast">Yoast SEO</option>
                                                    <option value="rankmath">Rank Math</option>
                                                    <option value="custom">Custom theme/plugin</option>
                                                    <option value="none">No SEO plugin</option>
                                                </select>
                                            </div>
                                            <div className="space-y-2">
                                                <Label className="text-xs font-semibold">CMS / WordPress API Base URL</Label>
                                                <Input
                                                    placeholder="https://cms.example.com"
                                                    value={formData.cms_url || ''}
                                                    onChange={e => setFormData({ ...formData, cms_url: e.target.value })}
                                                    className="h-10 rounded-lg"
                                                />
                                                <p className="text-xs text-muted-foreground">
                                                    Used for WordPress REST API calls like categories, media uploads, and post publishing.
                                                </p>
                                            </div>
                                            <div className="space-y-2">
                                                <Label className="text-xs font-semibold">Canonical Base URL Override</Label>
                                                <Input
                                                    placeholder="https://www.example.com"
                                                    value={formData.site_url_override || ''}
                                                    onChange={e => setFormData({ ...formData, site_url_override: e.target.value })}
                                                    className="h-10 rounded-lg"
                                                />
                                            </div>
                                            <div className="space-y-2 md:col-span-2">
                                                <Label className="text-xs font-semibold">Default Social Image URL</Label>
                                                <Input
                                                    placeholder="https://www.example.com/default-social.jpg"
                                                    value={formData.social_default_image_url || ''}
                                                    onChange={e => setFormData({ ...formData, social_default_image_url: e.target.value })}
                                                    className="h-10 rounded-lg"
                                                />
                                                <p className="text-xs text-muted-foreground">
                                                    Used as a fallback for Open Graph and Twitter Card images when an article has no featured image.
                                                </p>
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
                                                            setEditingCategory({ name: '', description: '', level: 1 });
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
                                                                            {category.description && (
                                                                                <p className="text-[10px] text-muted-foreground mt-0.5 line-clamp-2 max-w-[680px]">
                                                                                    {category.description}
                                                                                </p>
                                                                            )}
                                                                        </div>
                                                                    </div>
                                                                    <div className="flex items-center gap-1">
                                                                        <Button
                                                                            variant="ghost"
                                                                            size="icon"
                                                                            className="h-7 w-7 text-muted-foreground hover:text-primary"
                                                                            onClick={() => {
                                                                                setEditingCategory({
                                                                                    id: category.id,
                                                                                    name: category.name,
                                                                                    description: category.description || '',
                                                                                    level: 1
                                                                                });
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
                                                                                            setEditingCategory({
                                                                                                id: child.id,
                                                                                                name: child.name,
                                                                                                description: child.description || '',
                                                                                                level: 2,
                                                                                                parent_category_id: category.id
                                                                                            });
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
                                                                    <Label className="text-xs font-semibold mb-1 block">Category Description (for WordPress SEO context)</Label>
                                                                    <textarea
                                                                        placeholder="Describe what this category covers. Used for WordPress category descriptions."
                                                                        value={editingCategory.description || ''}
                                                                        onChange={(e) => setEditingCategory({ ...editingCategory, description: e.target.value })}
                                                                        rows={3}
                                                                        className="w-full px-3 py-2 text-sm rounded-lg border border-input bg-background resize-none focus:outline-none focus:ring-2 focus:ring-ring/40"
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

                {/* LinkedIn Integration */}
                <TabsContent value="integrations" className="animate-in fade-in-50 duration-500 space-y-6">
                    <Card className="border-border shadow-sm overflow-hidden">
                        <CardHeader className="bg-muted/30 border-b border-border">
                            <div className="flex items-center gap-3">
                                <div className="w-10 h-10 rounded-xl bg-[#0A66C2] text-white flex items-center justify-center font-bold text-xl shadow-sm">
                                    in
                                </div>
                                <div>
                                    <CardTitle>LinkedIn Integration</CardTitle>
                                    <CardDescription>
                                        Connect your personal LinkedIn account to generate and publish thought leadership posts and articles.
                                    </CardDescription>
                                </div>
                            </div>
                        </CardHeader>
                        <CardContent className="p-6 space-y-6">
                            {/* Connection Status Panel */}
                            <div className="rounded-2xl border border-border bg-card p-5 shadow-sm">
                                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                                    <div className="flex items-center gap-4">
                                        {linkedInLoading ? (
                                            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                                <Loader2 className="w-5 h-5 animate-spin text-[#0A66C2]" />
                                                Checking LinkedIn account status...
                                            </div>
                                        ) : linkedInStatus?.connected && linkedInStatus?.account ? (
                                            <>
                                                {linkedInStatus.account.profile_picture_url ? (
                                                    <img
                                                        src={linkedInStatus.account.profile_picture_url}
                                                        alt={linkedInStatus.account.account_name}
                                                        className="w-14 h-14 rounded-full object-cover border-2 border-[#0A66C2]/30 shadow-sm"
                                                    />
                                                ) : (
                                                    <div className="w-14 h-14 rounded-full bg-[#0A66C2] text-white font-bold text-xl flex items-center justify-center">
                                                        {linkedInStatus.account.account_name.charAt(0)}
                                                    </div>
                                                )}
                                                <div>
                                                    <div className="flex items-center gap-2">
                                                        <h4 className="text-base font-bold text-foreground">
                                                            {linkedInStatus.account.account_name}
                                                        </h4>
                                                        <span className="px-2.5 py-0.5 text-xs bg-emerald-500/10 text-emerald-500 dark:text-emerald-400 border border-emerald-500/20 rounded-full font-semibold">
                                                            Connected
                                                        </span>
                                                    </div>
                                                    <p className="text-xs text-muted-foreground mt-0.5">
                                                        Author URN: <code className="text-[11px] bg-muted px-1.5 py-0.5 rounded">{linkedInStatus.account.linkedin_urn}</code>
                                                    </p>
                                                    {linkedInStatus.account.expires_at && (
                                                        <p className="text-[11px] text-muted-foreground mt-1">
                                                            Token valid until: {new Date(linkedInStatus.account.expires_at).toLocaleDateString()}
                                                        </p>
                                                    )}
                                                </div>
                                            </>
                                        ) : (
                                            <div className="flex items-center gap-3.5">
                                                <div className="w-12 h-12 rounded-xl bg-muted flex items-center justify-center text-muted-foreground">
                                                    <Share2 className="w-6 h-6" />
                                                </div>
                                                <div>
                                                    <h4 className="text-sm font-semibold text-foreground">No LinkedIn Account Connected</h4>
                                                    <p className="text-xs text-muted-foreground mt-0.5">
                                                        Authenticate with LinkedIn to publish directly from the Article Editor.
                                                    </p>
                                                </div>
                                            </div>
                                        )}
                                    </div>

                                    <div>
                                        {linkedInStatus?.connected ? (
                                            <Button
                                                variant="outline"
                                                onClick={handleDisconnectLinkedIn}
                                                disabled={disconnectingLinkedIn}
                                                className="text-destructive hover:text-destructive hover:bg-destructive/10 border-destructive/30 text-xs"
                                            >
                                                {disconnectingLinkedIn ? (
                                                    <>
                                                        <Loader2 className="w-3.5 h-3.5 mr-1.5 animate-spin" />
                                                        Disconnecting...
                                                    </>
                                                ) : (
                                                    <>
                                                        <Trash2 className="w-3.5 h-3.5 mr-1.5" />
                                                        Disconnect Account
                                                    </>
                                                )}
                                            </Button>
                                        ) : (
                                            <Button
                                                onClick={handleConnectLinkedIn}
                                                className="bg-[#0A66C2] hover:bg-[#084e96] text-white shadow-md text-sm font-medium"
                                            >
                                                <Share2 className="w-4 h-4 mr-2" />
                                                Connect Personal LinkedIn
                                            </Button>
                                        )}
                                    </div>
                                </div>
                            </div>

                            {/* Features Overview */}
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
                                <div className="p-4 rounded-xl border border-border bg-muted/20 space-y-1.5">
                                    <h5 className="text-xs font-bold uppercase tracking-wider text-foreground">⚡ Direct Feed Publishing</h5>
                                    <p className="text-xs text-muted-foreground leading-relaxed">
                                        Publish posts with up to 3,000 characters, custom line breaks, emojis, and hashtags directly into your LinkedIn feed.
                                    </p>
                                </div>
                                <div className="p-4 rounded-xl border border-border bg-muted/20 space-y-1.5">
                                    <h5 className="text-xs font-bold uppercase tracking-wider text-foreground">🖼️ Image & Media Attachments</h5>
                                    <p className="text-xs text-muted-foreground leading-relaxed">
                                        Upload and attach featured images or AI-generated infographics directly with your post.
                                    </p>
                                </div>
                                <div className="p-4 rounded-xl border border-border bg-muted/20 space-y-1.5">
                                    <h5 className="text-xs font-bold uppercase tracking-wider text-foreground">🤖 1-Click AI Repurposing</h5>
                                    <p className="text-xs text-muted-foreground leading-relaxed">
                                        Automatically distill any long-form blog article into a scroll-stopping LinkedIn post with hook, takeaways, and call to action.
                                    </p>
                                </div>
                            </div>
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
