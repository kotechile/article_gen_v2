
import React, { useEffect, useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { supabase } from '../lib/supabase';
import { useAuth } from '../context/auth-context';
import { Loader2, Wand2, Save, BarChart3, BrainCircuit } from 'lucide-react';
import axios from 'axios';
import { Gauge } from '../components/Gauge';
import { METRIC_EXPLANATIONS } from '../types/metrics';
import { MetricTooltip } from '../components/Tooltip';
import { GenerationModal } from '../components/GenerationModal';


// Types
interface ArticleData {
    id: string;
    Title: string;
    userDescription: string;
    Keywords: string;
    articleLength: string;
    tone: string;
    LLM: string;
    status: string;
    rag_collection_name?: string;
    rag_query_type?: string;
    rag_balance_emphasis?: string;
    // Metrics
    seo_optimization_score?: number;
    readability_score?: number;
    viral_potential_score?: number;
    // New Metrics
    difficulty_level?: string;
    estimated_reading_time?: string;
    target_audience?: string;
    overall_quality_score?: number;
    audience_alignment_score?: number;
    content_feasibility_score?: number;
    business_impact_score?: number;
    total_search_volume?: number;
    avg_keyword_difficulty?: number;
    traffic_potential_score?: number;
    competition_score?: number;
    // Affiliate
    affiliate_opportunities?: any;
}

interface RagCollection {
    id: string;
    name: string;
}

interface AppSettings {
    // Add specific keys if needed
    [key: string]: any;
}

const TONE_OPTIONS = [
    { label: "Academic", value: "academic" },
    { label: "Journalistic", value: "journalistic" },
    { label: "Professional", value: "professional" },
    { label: "Casual", value: "casual" },
    { label: "Technical", value: "technical" },
    { label: "Persuasive", value: "persuasive" },
    { label: "Friendly", value: "friendly" },
];

const RAG_QUERY_TYPES = [
    { label: "Basic RAG", value: "/query_simple" },
    { label: "Hybrid Enhanced", value: "/query_hybrid_enhanced" },
    { label: "Agentic Iterative", value: "/query_agentic_iterative" },
    { label: "Agentic Focused", value: "/query_truly_agentic" },
    { label: "Truly Agentic", value: "/query_agentic_fixed" },
];

const EMPHASIS_OPTIONS = [
    { label: "Balanced", value: "balanced" },
    { label: "Comprehensive", value: "comprehensive" },
    { label: "News Focused", value: "news_focused" },
    { label: "Auto", value: "auto" },
];

export const ContentStudio: React.FC = () => {
    const { user } = useAuth();
    const navigate = useNavigate();
    const [searchParams] = useSearchParams();
    const articleId = searchParams.get('id');

    const [loading, setLoading] = useState(true);
    const [generating, setGenerating] = useState(false);
    const [saving, setSaving] = useState(false);
    const [showProgress, setShowProgress] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [taskId, setTaskId] = useState<string | null>(null);

    // Data State
    const [article, setArticle] = useState<ArticleData | null>(null);
    const [ragCollections, setRagCollections] = useState<RagCollection[]>([]);
    const [appSettings, setAppSettings] = useState<AppSettings | null>(null);
    const [llmModels, setLlmModels] = useState<any[]>([]);

    // Form State
    const [formData, setFormData] = useState({
        title: '',
        description: '',
        keywords: '',
        articleLength: '2500',
        tone: 'journalistic',
        ragCollection: '',
        ragQueryType: '/query_hybrid_enhanced',
        emphasis: 'balanced',
        claimsValidation: true,
        // Add Metrics Explanations to UI <!-- id: 3 -->
        llmModel: '',
    });

    // Fetch Data
    useEffect(() => {
        if (!user || !articleId) return;

        const fetchData = async () => {
            setLoading(true);
            try {
                // 1. Fetch Article
                const { data: artData, error: artError } = await supabase
                    .from('Titles')
                    .select('*')
                    .eq('id', articleId)
                    .single();

                if (artError) throw artError;
                setArticle(artData);

                // Restore progress modal if generating or still new/processing
                if (artData.status === 'Generating' || artData.status === 'Researching' || artData.status === 'Writing') {
                    setShowProgress(true);
                }

                // 2. Fetch Settings
                const { data: settingsData, error: settingsError } = await supabase
                    .from('application_settings')
                    .select('*')
                    .single(); // Assuming global settings for now

                if (!settingsError && settingsData) {
                    setAppSettings(settingsData);
                }

                // 3. Fetch RAG Collections
                const { data: ragData, error: ragError } = await supabase
                    .from('lindex_collections')
                    .select('id, name')
                    .eq('user_id', user.id);

                if (!ragError) {
                    setRagCollections(ragData || []);
                }

                // 4. Fetch LLM Providers
                const { data: llmData, error: llmError } = await supabase
                    .from('llm_providers')
                    .select('id, name, model_name, provider')
                    .eq('is_active', true);

                if (llmError) {
                    console.error("Error fetching LLM models:", llmError);
                } else {
                    setLlmModels(llmData || []);
                }

                // Initialize Form
                setFormData({
                    title: artData.Title || '',
                    description: artData.userDescription || '',
                    keywords: artData.Keywords || '',
                    articleLength: artData.articleLength || '2500',
                    tone: artData.Tone || 'journalistic',
                    ragCollection: artData.rag_collection_name || '', // Assuming stored as name or we need to map
                    ragQueryType: artData.rag_query_type || '/query_hybrid_enhanced',
                    emphasis: artData.rag_balance_emphasis || 'balanced',
                    claimsValidation: true, // Default
                    llmModel: artData.LLM || '',
                });

            } catch (error) {
                console.error("Error fetching data:", error);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [user, articleId]);

    // Handle Input Changes
    const handleChange = (field: string, value: any) => {
        setFormData(prev => ({ ...prev, [field]: value }));
    };

    // Save Changes
    const handleSave = async () => {
        if (!articleId) return;
        setSaving(true);
        try {
            // Calculate estimated reading time
            const wpm = 250;
            const words = parseInt(formData.articleLength, 10) || 0;
            const minutes = Math.ceil(words / wpm);
            const readingTime = `${minutes} min`;

            const updates = {
                Title: formData.title,
                userDescription: formData.description,
                Keywords: formData.keywords,
                articleLength: formData.articleLength,
                Tone: formData.tone,
                rag_collection_name: formData.ragCollection,
                rag_query_type: formData.ragQueryType,
                rag_balance_emphasis: formData.emphasis,
                LLM: formData.llmModel,
                estimated_reading_time: readingTime,
            };

            const { error } = await supabase
                .from('Titles')
                .update(updates)
                .eq('id', articleId);

            if (error) throw error;

            // Update local state to reflect changes immediately
            if (article) {
                setArticle({
                    ...article,
                    ...updates,
                    // Ensure type compatibility if needed, though spreading updates should work 
                    // if keys match partial ArticleData. 
                    // However, updates uses 'Tone' while interface sends 'tone' (Title vs title)
                    // We just need to ensure estimated_reading_time is set.
                    estimated_reading_time: readingTime,
                });
            }

            // Optional: show toast
        } catch (error) {
            console.error("Error saving:", error);
        } finally {
            setSaving(false);
        }
    };

    // Generate Article
    const handleGenerate = async () => {
        if (!articleId || !appSettings) return;
        setGenerating(true);
        setError(null);

        try {
            // Save first
            await handleSave();

            // Clear any old progress from local storage
            localStorage.removeItem(`gen_progress_${articleId}`);

            // Prepare Payload
            // Note: Logic allows checking if user has specific overwrites, but for now using basic logic

            if (formData.description.length < 10) {
                alert("Please provide a longer description (at least 10 characters).");
                setGenerating(false);
                return;
            }

            // Find selected model details
            const selectedModel = llmModels.find(m => m.model_name === formData.llmModel);

            if (!selectedModel) {
                setError("Please select a valid LLM model from the dropdown.");
                setGenerating(false);
                return;
            }

            const provider = selectedModel.provider;
            const modelName = selectedModel.model_name;

            const payload = {
                brief: formData.description,
                keywords: formData.keywords,
                draft_title: formData.title,
                provider: provider,
                model: modelName,
                // llm_model: `${provider}/${modelName}`, // Optional legacy field
                // llm_key is now resolved by backend from DB
                depth: formData.emphasis === 'balanced' ? 'standard' : 'comprehensive', // Mapping logic
                tone: formData.tone,
                target_word_count: parseInt(formData.articleLength, 10),
                claims_research_enabled: formData.claimsValidation,
                rag_enabled: !!formData.ragCollection,
                rag_collection_name: formData.ragCollection,
                rag_endpoint: appSettings.rag_url + formData.ragQueryType, // "query_hybrid_enhanced" etc.
                rag_balance_emphasis: formData.emphasis,
                article_id: articleId,
            };

            // Call Backend
            // Use axios with session token from supabase
            const { data: { session } } = await supabase.auth.getSession();
            const token = session?.access_token;

            if (!token) throw new Error("No session token found");

            // Set taskId to null before new request
            setTaskId(null);

            const response = await axios.post('http://localhost:5001/api/v1/research', payload, {
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'X-API-Key': 'development'
                }
            });

            if (response.data && response.data.research_id) {
                setTaskId(response.data.research_id);
            }

            // Show Progress Modal instead of navigating immediately
            setShowProgress(true);
            setGenerating(false); // Stop loading button, start modal

        } catch (error: any) {
            console.error("Generation error:", error);
            const serverMessage = error.response?.data?.error?.message || error.response?.data?.message;
            const validationErrors = error.response?.data?.validation_errors;

            let errorMessage = 'Failed to start generation.';
            if (validationErrors) {
                errorMessage += ` Validation: ${validationErrors[0].message}`;
            } else if (serverMessage) {
                errorMessage += ` Server: ${serverMessage}`;
            }

            setError(errorMessage);
            setGenerating(false);
        }
    };

    if (loading) return <div className="flex justify-center p-12"><Loader2 className="animate-spin text-indigo-600" /></div>;
    if (!article) return <div className="p-8">Article not found</div>;

    const getCompetitionColor = (score: number) => {
        if (score < 40) return 'text-green-500';
        if (score < 70) return 'text-yellow-500';
        return 'text-red-500';
    };

    return (
        <div className="max-w-5xl mx-auto space-y-6">
            {error && (
                <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-xl flex items-center gap-2">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-red-500" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                    <span>{error}</span>
                </div>
            )}
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Content Studio</h1>
                    <p className="text-gray-500 dark:text-gray-400">Configure and generate your article</p>
                </div>
                <div className="flex gap-3">
                    <button
                        onClick={handleSave}
                        disabled={saving}
                        className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl hover:bg-gray-50 dark:hover:bg-gray-700 transition"
                    >
                        {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
                        Save
                    </button>
                    <button
                        onClick={handleGenerate}
                        disabled={generating}
                        className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl hover:opacity-90 transition shadow-lg shadow-indigo-500/25"
                    >
                        {generating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Wand2 className="w-4 h-4" />}
                        Generate Article
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Column: Form */}
                <div className="lg:col-span-2 space-y-6">
                    <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl border border-gray-100 dark:border-gray-700 shadow-sm space-y-4">
                        <h3 className="font-semibold text-gray-900 dark:text-white">Article Details</h3>

                        <div>
                            <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Article Title</label>
                            <input
                                className="w-full px-4 py-2 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:ring-2 focus:ring-indigo-500 outline-none"
                                value={formData.title}
                                onChange={(e) => handleChange('title', e.target.value)}
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Brief Description</label>
                            <textarea
                                className="w-full px-4 py-2 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:ring-2 focus:ring-indigo-500 outline-none h-32"
                                value={formData.description}
                                onChange={(e) => handleChange('description', e.target.value)}
                                placeholder="Describe what you want to write about..."
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Keywords</label>
                            <input
                                className="w-full px-4 py-2 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:ring-2 focus:ring-indigo-500 outline-none"
                                value={formData.keywords}
                                onChange={(e) => handleChange('keywords', e.target.value)}
                                placeholder="Comma separated keywords"
                            />
                        </div>
                    </div>

                    <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl border border-gray-100 dark:border-gray-700 shadow-sm space-y-4">
                        <h3 className="font-semibold text-gray-900 dark:text-white">Configuration</h3>

                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Tone</label>
                                <select
                                    className="w-full px-4 py-2 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:ring-2 focus:ring-indigo-500 outline-none"
                                    value={formData.tone}
                                    onChange={(e) => handleChange('tone', e.target.value)}
                                >
                                    {TONE_OPTIONS.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Target Word Count</label>
                                <input
                                    type="number"
                                    className="w-full px-4 py-2 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:ring-2 focus:ring-indigo-500 outline-none"
                                    value={formData.articleLength}
                                    onChange={(e) => handleChange('articleLength', e.target.value)}
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">LAI Model</label>
                                <div className="relative">
                                    <BrainCircuit className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                                    <select
                                        className="w-full pl-9 pr-4 py-2 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:ring-2 focus:ring-indigo-500 outline-none appearance-none"
                                        value={formData.llmModel}
                                        onChange={(e) => handleChange('llmModel', e.target.value)}
                                    >
                                        <option value="">Select a Model</option>
                                        {llmModels.map(model => (
                                            <option key={model.id} value={model.model_name}>
                                                {model.name}
                                            </option>
                                        ))}
                                    </select>
                                </div>
                            </div>
                        </div>

                        <div className="pt-4 border-t border-gray-100 dark:border-gray-700">
                            <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-3">RAG & Research</h4>
                            <div className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">RAG Collection</label>
                                    <select
                                        className="w-full px-4 py-2 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:ring-2 focus:ring-indigo-500 outline-none"
                                        value={formData.ragCollection}
                                        onChange={(e) => handleChange('ragCollection', e.target.value)}
                                    >
                                        <option value="">None (Disabled)</option>
                                        {ragCollections.map(col => <option key={col.id} value={col.name}>{col.name}</option>)}
                                    </select>
                                </div>

                                {formData.ragCollection && (
                                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                        <div>
                                            <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Query Type</label>
                                            <select
                                                className="w-full px-4 py-2 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:ring-2 focus:ring-indigo-500 outline-none"
                                                value={formData.ragQueryType}
                                                onChange={(e) => handleChange('ragQueryType', e.target.value)}
                                            >
                                                {RAG_QUERY_TYPES.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
                                            </select>
                                        </div>
                                        <div>
                                            <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Emphasis</label>
                                            <select
                                                className="w-full px-4 py-2 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:ring-2 focus:ring-indigo-500 outline-none"
                                                value={formData.emphasis}
                                                onChange={(e) => handleChange('emphasis', e.target.value)}
                                            >
                                                {EMPHASIS_OPTIONS.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
                                            </select>
                                        </div>
                                    </div>
                                )}

                                <div className="flex items-center gap-2 mt-2">
                                    <input
                                        type="checkbox"
                                        id="claims"
                                        className="w-4 h-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                                        checked={formData.claimsValidation}
                                        onChange={(e) => handleChange('claimsValidation', e.target.checked)}
                                    />
                                    <label htmlFor="claims" className="text-sm text-gray-700 dark:text-gray-300">Enable Claims Validation (Web Search)</label>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Right Column: Metrics & Insights */}
                <div className="space-y-6">
                    <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl border border-gray-100 dark:border-gray-700 shadow-sm">
                        <div className="flex items-center gap-2 mb-4">
                            <BarChart3 className="w-5 h-5 text-indigo-500" />
                            <h3 className="font-semibold text-gray-900 dark:text-white">Content Metrics</h3>
                        </div>

                        <div className="space-y-6">
                            {/* Top Section: Main Gauges */}
                            <div className="grid grid-cols-2 gap-4 justify-items-center">
                                <Gauge value={article.seo_optimization_score || 0} label="SEO Score" color="text-indigo-600" explanation={METRIC_EXPLANATIONS.seo_score} />

                                <Gauge
                                    value={article.competition_score || 0}
                                    label="Key. Difficulty"
                                    color={getCompetitionColor(article.competition_score || 0)}
                                    explanation={METRIC_EXPLANATIONS.difficulty}
                                />
                                <Gauge
                                    value={article.total_search_volume ? Math.min((article.total_search_volume / 1000), 100) : 0}
                                    displayValue={article.total_search_volume || 0}
                                    label="Search Volume"
                                    color="text-blue-500"
                                    explanation={METRIC_EXPLANATIONS.search_volume}
                                    unit=""
                                />
                                <Gauge value={article.readability_score || 0} label="Readability" color="text-green-500" />
                            </div>

                            <div className="border-t border-gray-100 dark:border-gray-700 my-6"></div>

                            {/* Secondary Scores Grid */}
                            <div className="grid grid-cols-2 gap-3">
                                <div className="p-2 bg-gray-50 dark:bg-gray-900/50 rounded-lg text-center">
                                    <div className="text-lg font-bold text-gray-900 dark:text-white">{article.overall_quality_score || 0}%</div>
                                    <div className="text-[10px] text-gray-500 uppercase">Quality</div>
                                </div>
                                <div className="p-2 bg-gray-50 dark:bg-gray-900/50 rounded-lg text-center relative group">
                                    <div className="text-lg font-bold text-gray-900 dark:text-white">{article.traffic_potential_score || 0}%</div>
                                    <div className="flex justify-center items-center gap-1">
                                        <div className="text-[10px] text-gray-500 uppercase">Traffic</div>
                                        <MetricTooltip explanation={METRIC_EXPLANATIONS.traffic_potential} />
                                    </div>
                                </div>
                            </div>

                            <div className="border-t border-gray-100 dark:border-gray-700 my-6"></div>

                            {/* Bottom Section: Text Metrics */}
                            <div className="space-y-3">
                                <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900/50 rounded-xl">
                                    <span className="text-sm text-gray-500">Difficulty Level</span>
                                    <span className="font-medium text-gray-900 dark:text-white">{article.difficulty_level || '-'}</span>
                                </div>
                                <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900/50 rounded-xl">
                                    <div className="flex items-center gap-1">
                                        <span className="text-sm text-gray-500">Est. Reading Time</span>
                                        <MetricTooltip explanation={METRIC_EXPLANATIONS.reading_time} />
                                    </div>
                                    <span className="font-medium text-gray-900 dark:text-white">{article.estimated_reading_time || '-'}</span>
                                </div>
                                <div className="p-3 bg-gray-50 dark:bg-gray-900/50 rounded-xl">
                                    <div className="flex items-center gap-1 mb-1">
                                        <span className="text-sm text-gray-500 block">Target Audience</span>
                                        <MetricTooltip explanation={METRIC_EXPLANATIONS.audience_align} />
                                    </div>
                                    <span className="font-medium text-gray-900 dark:text-white text-sm">{article.target_audience || '-'}</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="bg-white dark:bg-gray-800 p-6 rounded-2xl border border-gray-100 dark:border-gray-700 shadow-sm">
                        <h3 className="font-semibold text-gray-900 dark:text-white mb-4">Affiliate Opportunities</h3>
                        {article.affiliate_opportunities?.programs?.length > 0 ? (
                            <div className="space-y-3">
                                {article.affiliate_opportunities.programs.slice(0, 3).map((prog: any, i: number) => (
                                    <div key={i} className="p-3 border border-gray-100 dark:border-gray-700 rounded-xl hover:bg-gray-50 dark:hover:bg-gray-700/50 transition">
                                        <div className="font-medium text-sm truncate">{prog.name}</div>
                                        <div className="text-xs text-green-600 dark:text-green-400 mt-1">{prog.commission_rate}% Commission</div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="text-sm text-gray-500 text-center py-4">No opportunities found</div>
                        )}
                    </div>
                </div>
            </div>

            <GenerationModal
                articleId={articleId || ''}
                taskId={taskId}
                isOpen={showProgress}
                onClose={() => setShowProgress(false)}
                onComplete={() => {
                    setShowProgress(false);
                    navigate('/');
                }}
            />
        </div>
    );
};
