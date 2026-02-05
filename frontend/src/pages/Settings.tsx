
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
    TrendingUp
} from 'lucide-react';
import { TrendReportModal } from '../components/TrendReportModal';
import { supabase } from '../lib/supabase';
import { useAuth } from '../context/auth-context';
import { apiClient } from '../api-client'; // Use the same client as Research.tsx
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Switch } from '../components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import type { WordPressDetail } from '../types';

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
    const [activeTab, setActiveTab] = useState('research');
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

    // WordPress State
    const [sites, setSites] = useState<WordPressDetail[]>([]);
    const [editingSiteId, setEditingSiteId] = useState<string | null>(null);
    const [siteFormData, setSiteFormData] = useState<Partial<WordPressDetail>>({});
    const [isSavingSite, setIsSavingSite] = useState(false);

    // Posts State
    const [importedPosts, setImportedPosts] = useState<any[]>([]);
    const [postsLoading, setPostsLoading] = useState(false);
    const [isSyncing, setIsSyncing] = useState(false);

    // Trend Report State
    const [trendSite, setTrendSite] = useState<WordPressDetail | null>(null);

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
                fetchWordPressSites(),
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

    const fetchWordPressSites = async () => {
        const { data, error } = await supabase
            .from('wordPress_details')
            .select('*')
            .eq('user_id', user!.id);

        if (!error) {
            setSites(data || []);
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

    const handleSaveSite = async () => {
        if (!user || !siteFormData.domain || !siteFormData.wpUserName || !siteFormData.wordpress_key) return;
        setIsSavingSite(true);
        try {
            const payload = { ...siteFormData, user_id: user.id };
            let error;
            if (editingSiteId && editingSiteId !== 'new') {
                const { error: err } = await supabase.from('wordPress_details').update(payload).eq('id', editingSiteId);
                error = err;
            } else {
                const { error: err } = await supabase.from('wordPress_details').insert([payload]);
                error = err;
            }
            if (error) throw error;
            setEditingSiteId(null);
            setSiteFormData({});
            fetchWordPressSites();
        } catch (err: any) {
            alert(err.message || "Failed to save site");
        } finally {
            setIsSavingSite(false);
        }
    };

    const handleDeleteSite = async (id: string) => {
        if (!confirm('Are you sure?')) return;
        await supabase.from('wordPress_details').delete().eq('id', id);
        fetchWordPressSites();
    };

    const handleSync = async () => {
        if (!user) return;
        setIsSyncing(true);
        try {
            // Unified backend is on port 8000 (proxy handled by /api)
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
                    <TabsTrigger value="research" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm py-2.5">
                        <Search className="w-4 h-4 mr-2" />
                        Topic Research
                    </TabsTrigger>
                    <TabsTrigger value="content" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm py-2.5">
                        <Wand2 className="w-4 h-4 mr-2" />
                        Content Generation
                    </TabsTrigger>
                    <TabsTrigger value="wordpress" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm py-2.5">
                        <Globe className="w-4 h-4 mr-2" />
                        WordPress Sync
                    </TabsTrigger>
                    <TabsTrigger value="posts" className="rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm py-2.5">
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
                            <CardTitle>AI & Media API Keys</CardTitle>
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
                                        Media & Utils
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

                {/* WordPress Sites */}
                <TabsContent value="wordpress" className="animate-in fade-in-50 duration-500">
                    <div className="space-y-6">
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                            {sites.map(site => (
                                <Card key={site.id} className="relative group overflow-hidden border-gray-200 dark:border-gray-800 hover:border-indigo-500/50 transition-all">
                                    <div className="absolute top-0 left-0 w-1 h-full bg-indigo-500 group-hover:w-2 transition-all"></div>
                                    <CardHeader className="pb-2">
                                        <div className="flex justify-between items-start">
                                            <div>
                                                <CardTitle className="text-lg truncate max-w-[180px]">{site.domain}</CardTitle>
                                                <CardDescription className="flex items-center gap-1.5 mt-1">
                                                    <span className="w-1.5 h-1.5 rounded-full bg-green-500"></span>
                                                    {site.wpUserName}
                                                </CardDescription>
                                            </div>
                                            <div className="flex gap-1">
                                                <Button variant="ghost" size="icon" className="h-8 w-8 text-gray-400 hover:text-indigo-600" onClick={() => { setEditingSiteId(site.id); setSiteFormData(site); }}>
                                                    <Edit2 className="h-4 w-4" />
                                                </Button>
                                                <Button variant="ghost" size="icon" className="h-8 w-8 text-gray-400 hover:text-red-600" onClick={() => handleDeleteSite(site.id)}>
                                                    <Trash2 className="h-4 w-4" />
                                                </Button>
                                            </div>
                                        </div>
                                    </CardHeader>
                                    <CardContent>
                                        <div className="flex items-center justify-between mt-4">
                                            <div className="flex gap-2">
                                                <Button
                                                    size="sm"
                                                    variant="outline"
                                                    className="h-7 text-xs gap-1.5 border-indigo-200 text-indigo-600 hover:bg-indigo-50 hover:text-indigo-700"
                                                    onClick={() => setTrendSite(site)}
                                                >
                                                    <TrendingUp className="w-3 h-3" />
                                                    What's Trending
                                                </Button>
                                            </div>
                                            <Globe className="w-4 h-4 text-gray-300" />
                                        </div>
                                    </CardContent>
                                </Card>
                            ))}

                            <button
                                onClick={() => { setEditingSiteId('new'); setSiteFormData({}); }}
                                className="h-[140px] border-2 border-dashed border-gray-200 dark:border-gray-800 rounded-2xl flex flex-col items-center justify-center gap-2 text-gray-400 hover:border-indigo-500 hover:text-indigo-500 hover:bg-indigo-50/30 transition-all group"
                            >
                                <Plus className="w-8 h-8 group-hover:scale-110 transition-transform" />
                                <span className="text-sm font-medium">Add WordPress Site</span>
                            </button>
                        </div>

                        {editingSiteId && (
                            <Card className="border-indigo-100 dark:border-indigo-900 bg-indigo-50/20 dark:bg-indigo-900/5 overflow-hidden ring-1 ring-indigo-500/20">
                                <CardHeader className="border-b border-indigo-100/50 dark:border-indigo-900/50">
                                    <CardTitle className="text-lg">{editingSiteId === 'new' ? 'New WordPress Configuration' : 'Edit Site Configuration'}</CardTitle>
                                </CardHeader>
                                <CardContent className="p-6">
                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                        <div className="space-y-2">
                                            <Label className="text-xs font-semibold">Domain URL</Label>
                                            <Input placeholder="example.com" value={siteFormData.domain || ''} onChange={e => setSiteFormData({ ...siteFormData, domain: e.target.value })} className="h-10 rounded-lg" />
                                        </div>
                                        <div className="space-y-2">
                                            <Label className="text-xs font-semibold">WP Username</Label>
                                            <Input placeholder="admin" value={siteFormData.wpUserName || ''} onChange={e => setSiteFormData({ ...siteFormData, wpUserName: e.target.value })} className="h-10 rounded-lg" />
                                        </div>
                                        <div className="space-y-2">
                                            <Label className="text-xs font-semibold">Application Password</Label>
                                            <Input type="password" placeholder="xxxx xxxx xxxx xxxx" value={siteFormData.wordpress_key || ''} onChange={e => setSiteFormData({ ...siteFormData, wordpress_key: e.target.value })} className="h-10 rounded-lg font-mono" />
                                        </div>
                                        <div className="space-y-2 md:col-span-3">
                                            <Label className="text-xs font-semibold">Site Description (for AI Trend Analysis)</Label>
                                            <Input
                                                placeholder="E.g. A home improvement blog focusing on DIY renovations and sustainable materials..."
                                                value={siteFormData.site_description || ''}
                                                onChange={e => setSiteFormData({ ...siteFormData, site_description: e.target.value })}
                                                className="h-10 rounded-lg"
                                            />
                                        </div>
                                    </div>
                                    <div className="flex justify-end gap-3 mt-8">
                                        <Button variant="ghost" onClick={() => { setEditingSiteId(null); setSiteFormData({}); }}>Cancel</Button>
                                        <Button onClick={handleSaveSite} disabled={isSavingSite} className="bg-indigo-600 hover:bg-indigo-700 rounded-xl px-8 h-10">
                                            {isSavingSite ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : <Save className="w-4 h-4 mr-2" />}
                                            {editingSiteId === 'new' ? 'Add Site' : 'Update Site'}
                                        </Button>
                                    </div>
                                </CardContent>
                            </Card>
                        )}
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
                            <Button onClick={handleSync} disabled={isSyncing || sites.length === 0} className="rounded-xl border-gray-200" variant="outline">
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

            {trendSite && (
                <TrendReportModal
                    siteId={trendSite.id}
                    siteDomain={trendSite.domain}
                    isOpen={!!trendSite}
                    onClose={() => setTrendSite(null)}
                />
            )}
        </div>
    );
};

// Helper for cn
function cn(...classes: any[]) {
    return classes.filter(Boolean).join(' ');
}
