
import React, { useEffect, useState } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { supabase } from '../lib/supabase';
import { useAuth } from '../context/auth-context';
import { Loader2, Wand2, Save, BarChart3, BrainCircuit, ShieldCheck, AlertTriangle, Globe2, Tag, KeyRound } from 'lucide-react';
import axios from 'axios';
import { Gauge } from '../components/Gauge';
import { METRIC_EXPLANATIONS } from '../types/metrics';
import { MetricTooltip } from '../components/Tooltip';
import { GenerationModal } from '../components/GenerationModal';
import { KeywordIntelligenceModal } from '../components/KeywordIntelligenceModal';
import { computeGEOContext } from '../utils/seoUtils';
import { contentIdeasService } from '../services/content-ideas.service';
import type { ContentIdea } from '../types/idea-burst';


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
    estimated_reading_time?: number | string;
    target_audience?: string;
    overall_quality_score?: number;
    audience_alignment_score?: number;
    content_feasibility_score?: number;
    business_impact_score?: number;
    total_search_volume?: number;
    avg_keyword_difficulty?: number;
    traffic_potential_score?: number;
    competition_score?: number;
    // Legacy single keyword field
    primary_keyword?: string;
    secondary_keywords_json?: string[] | string;
    keyword_research_source?: string;
    keyword_research_confidence?: number;
    selected_keyword_intent?: string;
    selected_keyword_search_volume?: number;
    selected_keyword_difficulty?: number;
    selected_keyword_metrics_json?: any;
    keyword_selection_source?: string;
    supporting_entities_json?: string[] | string;
    priority_questions_json?: string[] | string;
    // SEO-First fields (from Keyword Intelligence)
    primary_keywords?: string[];
    secondary_keywords?: string[];
    search_phrase?: string;
    // Site / WP context
    domain?: string;
    wordpress_category_id?: number | null;
    wordpress_parent_category_id?: number | null;
    source_idea_id?: string;
    // Affiliate
    affiliate_opportunities?: any;
    // DataForSEO traces for Keyword Intelligence modal
    raw_dataforseo_output?: any;
    raw_supabase_output?: any;
    idea_metadata?: any;
}

function normalizeKeywordList(value: unknown): string[] {
    if (Array.isArray(value)) {
        return value
            .map((item) => String(item ?? '').trim())
            .filter(Boolean);
    }

    if (typeof value === 'string') {
        const raw = value.trim();
        if (!raw) return [];
        try {
            const parsed = JSON.parse(raw);
            if (Array.isArray(parsed)) {
                return parsed
                    .map((item) => String(item ?? '').trim())
                    .filter(Boolean);
            }
        } catch {
            // Fall back to comma-separated parsing.
        }

        return raw
            .split(',')
            .map((item) => item.trim())
            .filter(Boolean);
    }

    return [];
}

// ─── SEO Validation ───────────────────────────────────────────────────────────

interface SEOValidation {
    isValid: boolean;
    warnings: string[];
    errors: string[];
}

function validateSEOReadiness(article: ArticleData): SEOValidation {
    const errors: string[] = [];
    const warnings: string[] = [];

    const hasPrimary =
        (article.primary_keywords?.length ?? 0) > 0 ||
        !!article.search_phrase ||
        !!article.primary_keyword;

    if (!hasPrimary) {
        errors.push('Missing primary keyword — open Keyword Intelligence and save a selection first.');
    }

    const hasSecondary = (article.secondary_keywords?.length ?? 0) > 0;
    if (!hasSecondary) {
        errors.push('Missing secondary keywords — save secondary keyword strategy before generation.');
    }

    if (!article.domain || !String(article.domain).trim()) {
        errors.push('Missing domain context — link a project/domain before generation.');
    }

    if (!article.wordpress_category_id) {
        errors.push('Missing WordPress category mapping — assign wordpress_category_id before generation.');
    }

    return { isValid: errors.length === 0, warnings, errors };
}

interface RagCollection {
    id: string;
    name: string;
}

interface AppSettings {
    // Add specific keys if needed
    [key: string]: any;
}

interface LlmProviderRow {
    id: string;
    name: string;
    model_name: string;
    provider: string;
    is_default?: boolean | null;
    is_active?: boolean | null;
    api_keys_id?: string | null;
    api_key_id?: string | null;
    llm_key_id?: string | null;
}

async function fetchLlmProviderRows(): Promise<LlmProviderRow[]> {
    const queryAttempts: Array<{
        label: string;
        select: string;
        activeOnly: boolean;
    }> = [
        {
            label: 'active-with-flags',
            select: 'id, name, model_name, provider, api_keys_id, is_default, is_active',
            activeOnly: true,
        },
        {
            label: 'all-with-flags',
            select: 'id, name, model_name, provider, api_keys_id, is_default, is_active',
            activeOnly: false,
        },
        {
            label: 'all-with-default',
            select: 'id, name, model_name, provider, api_keys_id, is_default',
            activeOnly: false,
        },
        {
            label: 'all-core-fields',
            select: 'id, name, model_name, provider, api_keys_id',
            activeOnly: false,
        },
    ];

    for (const attempt of queryAttempts) {
        let query = supabase.from('llm_providers').select(attempt.select);
        if (attempt.activeOnly) {
            query = query.eq('is_active', true);
        }

        const { data, error } = await query;
        if (error) {
            console.warn(`[ContentStudio] LLM model query "${attempt.label}" failed:`, error);
            continue;
        }

        const rows = (Array.isArray(data) ? data : [])
            .filter((row: any) => row && row.model_name)
            .map((row: any) => ({
                id: String(row.id ?? ''),
                name: String(row.name ?? row.model_name ?? ''),
                model_name: String(row.model_name ?? ''),
                provider: String(row.provider ?? ''),
                api_keys_id: row.api_keys_id ? String(row.api_keys_id) : null,
                is_default: typeof row.is_default === 'boolean' ? row.is_default : null,
                is_active: typeof row.is_active === 'boolean' ? row.is_active : null,
            }));

        if (rows.length > 0) {
            return rows;
        }
    }

    return [];
}

function sortLlmModels(models: LlmProviderRow[]): LlmProviderRow[] {
    return [...models].sort((a, b) => {
        if (!!a.is_default !== !!b.is_default) return a.is_default ? -1 : 1;
        return String(a.name || a.model_name).localeCompare(String(b.name || b.model_name));
    });
}

function resolveInitialLlmModel(
    storedValue: string | undefined | null,
    models: LlmProviderRow[]
): string {
    const rows = sortLlmModels(models);
    if (rows.length === 0) return '';

    const defaultModel = rows.find((row) => row.is_default)?.model_name || rows[0].model_name;
    const raw = String(storedValue || '').trim();
    if (!raw) return defaultModel;

    // Stored value can be model_name, provider/model_name, or display name.
    const exactModelMatch = rows.find((row) => row.model_name === raw);
    if (exactModelMatch) return exactModelMatch.model_name;

    const providerModelMatch = rows.find((row) => `${row.provider}/${row.model_name}` === raw);
    if (providerModelMatch) return providerModelMatch.model_name;

    const displayNameMatch = rows.find((row) => row.name === raw);
    if (displayNameMatch) return displayNameMatch.model_name;

    return defaultModel;
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
    const [seoShiftEnabled, setSeoShiftEnabled] = useState(true);
    const [seoValidation, setSeoValidation] = useState<SEOValidation | null>(null);
    const [showKeywordModal, setShowKeywordModal] = useState(false);

    // Data State
    const [article, setArticle] = useState<ArticleData | null>(null);
    const [ragCollections, setRagCollections] = useState<RagCollection[]>([]);
    const [appSettings, setAppSettings] = useState<AppSettings | null>(null);
    const [llmModels, setLlmModels] = useState<LlmProviderRow[]>([]);

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
                    .select('*, primary_keywords, secondary_keywords, search_phrase, domain, wordpress_category_id, wordpress_parent_category_id, source_idea_id')
                    .eq('id', articleId)
                    .single();

                if (artError) throw artError;
                const normalizedArticle: ArticleData = {
                    ...(artData as ArticleData),
                    primary_keywords: normalizeKeywordList((artData as any)?.primary_keywords),
                    secondary_keywords: normalizeKeywordList((artData as any)?.secondary_keywords),
                };
                setArticle(normalizedArticle);
                setSeoValidation(validateSEOReadiness(normalizedArticle));

                // Do not auto-open progress modal from DB status alone.
                // It creates stale 90% hangs when previous runs crashed and left status as Generating.
                setShowProgress(false);

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

                // 4. Fetch LLM Providers with schema-safe fallbacks.
                // Some environments don't expose optional columns like is_active/is_default;
                // this chain keeps the dropdown populated from core columns.
                const llmRows = await fetchLlmProviderRows();

                const normalizedLlmRows = sortLlmModels(llmRows);
                setLlmModels(normalizedLlmRows);
                if (normalizedLlmRows.length === 0) {
                    setError('No active LLM models were found. Check llm_providers visibility/policies and model rows.');
                }

                // Initialize Form
                setFormData({
                    title: normalizedArticle.Title || '',
                    description: normalizedArticle.userDescription || '',
                    keywords: (
                        // Prefer primary_keywords array, fall back to legacy Keywords field
                        normalizedArticle.primary_keywords && normalizedArticle.primary_keywords.length > 0
                            ? normalizedArticle.primary_keywords.join(', ')
                            : normalizedArticle.Keywords || ''
                    ),
                    articleLength: normalizedArticle.articleLength || '2500',
                    tone: (artData as any).Tone || 'journalistic',
                    ragCollection: normalizedArticle.rag_collection_name || '',
                    ragQueryType: normalizedArticle.rag_query_type || '/query_hybrid_enhanced',
                    emphasis: normalizedArticle.rag_balance_emphasis || 'balanced',
                    claimsValidation: true,
                    llmModel: resolveInitialLlmModel(normalizedArticle.LLM, normalizedLlmRows),
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

    const formatReadingTime = (value?: number | string | null) => {
        if (typeof value === 'number' && Number.isFinite(value)) {
            return `${value} min`;
        }
        if (typeof value === 'string' && value.trim()) {
            return value.includes('min') ? value : `${value} min`;
        }
        return '-';
    };

    const formatConfidence = (value?: number | null) => {
        if (typeof value !== 'number' || !Number.isFinite(value)) return '-';
        const pct = value <= 1 ? value * 100 : value;
        return `${Math.round(pct)}%`;
    };

    const keywordModalIdea: ContentIdea | null = article
        ? {
            id: article.id,
            title: article.Title || formData.title || 'Untitled Article',
            content_type: 'blog',
            primary_keywords: article.primary_keywords ?? [],
            secondary_keywords: article.secondary_keywords ?? [],
            seo_optimization_score: article.seo_optimization_score ?? 0,
            traffic_potential_score: article.traffic_potential_score ?? 0,
            total_search_volume: article.total_search_volume ?? null,
            average_difficulty: article.avg_keyword_difficulty ?? null,
            average_cpc: null,
            created_at: '',
            user_id: user?.id || '',
            topic_id: article.source_idea_id || '',
            description: article.userDescription,
            search_phrase: article.search_phrase,
            keyword_metrics: article.selected_keyword_metrics_json || undefined,
            raw_dataforseo_output: article.raw_dataforseo_output ?? null,
            raw_supabase_output: article.raw_supabase_output ?? null,
            idea_metadata: article.idea_metadata || undefined,
        }
        : null;

    // Save Changes
    const handleSave = async (): Promise<boolean> => {
        if (!articleId) return false;
        setSaving(true);
        try {
            // Calculate estimated reading time
            const wpm = 250;
            const words = parseInt(formData.articleLength, 10) || 0;
            const minutes = Math.ceil(words / wpm);
            const readingTime = Math.max(1, minutes);

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
            return true;
        } catch (error) {
            console.error("Error saving:", error);
            setError("Failed to save article settings. Please verify DB schema/settings and try again.");
            return false;
        } finally {
            setSaving(false);
        }
    };

    const resolveSelectedModelKey = async (selectedModel: LlmProviderRow): Promise<string | null> => {
        const candidateKeyIds = [
            selectedModel.api_keys_id,
            selectedModel.api_key_id,
            selectedModel.llm_key_id,
        ]
            .map((value) => String(value || '').trim())
            .filter(Boolean);

        if (candidateKeyIds.length > 0) {
            const { data: apiKeyRows, error: apiKeyError } = await supabase
                .from('api_keys')
                .select('id, key_value')
                .in('id', candidateKeyIds);

            if (!apiKeyError && apiKeyRows && apiKeyRows.length > 0) {
                const byId = new Map<string, string>();
                for (const row of apiKeyRows as Array<{ id: string; key_value?: string | null }>) {
                    byId.set(String(row.id), String(row.key_value || '').trim());
                }
                for (const id of candidateKeyIds) {
                    const key = byId.get(id);
                    if (key) return key;
                }
            }
        }

        // Backward-compatible fallback: if llm_keys table exists in this deployment, use it.
        const llmKeyId = String(selectedModel.llm_key_id || '').trim();
        if (llmKeyId) {
            try {
                const { data: llmKeyRow, error: llmKeyError } = await supabase
                    .from('llm_keys')
                    .select('id, key_value')
                    .eq('id', llmKeyId)
                    .maybeSingle();

                if (!llmKeyError) {
                    const key = String((llmKeyRow as any)?.key_value || '').trim();
                    if (key) return key;
                }
            } catch {
                // Ignore missing table/permissions and continue to provider fallback.
            }
        }

        // Last fallback: provider-level active key lookup.
        const provider = String(selectedModel.provider || '').trim();
        if (provider) {
            const { data: providerKeyRows, error: providerKeyError } = await supabase
                .from('api_keys')
                .select('key_value')
                .eq('provider', provider)
                .eq('is_active', true)
                .limit(1);

            if (!providerKeyError && providerKeyRows && providerKeyRows.length > 0) {
                const key = String((providerKeyRows[0] as any)?.key_value || '').trim();
                if (key) return key;
            }
        }

        return null;
    };

    // Generate Article
    const handleGenerate = async () => {
        if (!articleId || !appSettings) return;
        setGenerating(true);
        setError(null);

        try {
            // Save first
            const saveOk = await handleSave();
            if (!saveOk) {
                setGenerating(false);
                return;
            }

            // Clear any old progress from local storage
            localStorage.removeItem(`gen_progress_${articleId}`);

            if (formData.description.length < 10) {
                alert('Please provide a longer description (at least 10 characters).');
                setGenerating(false);
                return;
            }

            // ── Phase 2: Validate SEO readiness ──────────────────────────────────
            if (article) {
                const validation = validateSEOReadiness(article);
                setSeoValidation(validation);
                if (!validation.isValid) {
                    setGenerating(false);
                    setError(validation.errors.join(' '));
                    return;
                }
            }

            // Find selected model details
            const selectedModel = llmModels.find(m => m.model_name === formData.llmModel);

            if (!selectedModel) {
                setError('Please select a valid LLM model from the dropdown.');
                setGenerating(false);
                return;
            }

            const provider = selectedModel.provider;
            const modelName = selectedModel.model_name;
            const llmModel = `${provider}/${modelName}`;
            const llmKey = await resolveSelectedModelKey(selectedModel);

            if (!llmKey) {
                setError('No API key found for the selected LLM model. Link the model to an active key in llm_providers/api_keys.');
                setGenerating(false);
                return;
            }

            // ── Phase 3: Build SEO-enriched brief (Directional Shift) ─────────────
            const primaryKw =
                article?.primary_keywords?.[0] ??
                article?.search_phrase ??
                article?.primary_keyword ??
                formData.keywords.split(',')[0]?.trim() ??
                '';

            const secondaryKwList = (
                article?.secondary_keywords?.slice(0, 4) ??
                []
            ).join(', ');

            let generationBrief = formData.description;
            let seodirective = '';

            if (seoShiftEnabled && primaryKw) {
                const geoCtx = computeGEOContext(primaryKw, article?.domain);

                const seoParts = [
                    '[SEO + GENERATIVE ENGINE OPTIMIZATION (GEO) DIRECTIVE]',
                    `PRIMARY KEYWORD: "${primaryKw}"`,
                    `- Integrate naturally in the H1, opening paragraph, and 2–3 subheadings.`,
                    `- Do NOT keyword-stuff; density should be ~1–2%.`,
                    secondaryKwList
                        ? `SECONDARY KEYWORDS (weave in naturally across body sections): ${secondaryKwList}.`
                        : '',
                    '',
                    `GENERATIVE ENGINE OPTIMIZATION (GEO):`,
                    `- This article must be optimized so that AI search engines (Perplexity, ChatGPT Search, Google AI Overview, Gemini) surface it as a high-confidence citation.`,
                    `- Write at least one "Direct Answer" paragraph that clearly and concisely answers the core query intent of "${primaryKw}".`,
                    `- Use structured, scannable sections with clear H2/H3 headings — Generative AI engines prefer content they can excerpt.`,
                    `- Include specific, authoritative data points, statistics, or expert-framed assertions (entity density matters for AI citation ranking).`,
                    `- Where relevant, use definition-style openers (e.g., "X is defined as...") as they are favored for AI answer extraction.`,
                    geoCtx.hasGEOSignal ? `- GEO focus area detected: ${geoCtx.optimizationFocus}. Apply accordingly.` : '',
                    '',
                    `TITLE & DESCRIPTION REWRITE AUTHORIZATION:`,
                    `- You ARE authorized to rewrite the Title and Description if the current versions do not integrate the primary keyword or are not GEO-ready.`,
                    `- Ensure the rewritten title is under 60 characters and includes "${primaryKw}".`,
                    '',
                    `ORIGINAL CREATIVE INTENT (preserve direction, adapt execution):`,
                    formData.description,
                ].filter(Boolean).join('\n');

                generationBrief = seoParts;
                seodirective = 'seo_geo_llm_enriched';
            }

            const payload = {
                brief: generationBrief,
                keywords: formData.keywords,
                draft_title: formData.title,
                llm_model: llmModel,
                llm_key: llmKey,
                domain: article?.domain || undefined,
                wordpress_category_id: article?.wordpress_category_id ?? undefined,
                seo_primary_keyword: primaryKw || undefined,
                seo_secondary_keywords: article?.secondary_keywords?.length ? article.secondary_keywords : undefined,
                seo_directive: seodirective || undefined,
                // Keep normalized fields for compatibility with both research API variants.
                provider: provider,
                model: modelName,
                depth: formData.emphasis === 'balanced' ? 'standard' : 'comprehensive',
                tone: formData.tone,
                target_word_count: parseInt(formData.articleLength, 10),
                claims_research_enabled: formData.claimsValidation,
                rag_enabled: !!formData.ragCollection,
                rag_collection_name: formData.ragCollection,
                rag_endpoint: appSettings.rag_url + formData.ragQueryType,
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

            const response = await axios.post(`${import.meta.env.VITE_API_URL || 'http://localhost:5001'}/api/v1/research`, payload, {
                headers: {
                    'Authorization': `Bearer ${token}`,
                    'X-API-Key': 'development'
                }
            });

            if (response.data && response.data.research_id) {
                setTaskId(response.data.research_id);
                // Show Progress Modal only when we have a valid task id
                setShowProgress(true);
                setGenerating(false);
            } else {
                throw new Error("Generation started but no task id was returned by the API.");
            }

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

    if (loading) return <div className="flex justify-center p-12"><Loader2 className="animate-spin text-primary" /></div>;
    if (!article) return <div className="p-8 text-muted-foreground">Article not found</div>;

    const getCompetitionColor = (score: number) => {
        if (score < 40) return 'text-chart-2';
        if (score < 70) return 'text-chart-4';
        return 'text-destructive';
    };

    return (
        <div className="max-w-5xl mx-auto space-y-6">
            {error && (
                <div className="bg-destructive/10 border border-destructive/20 text-destructive px-4 py-3 rounded-xl flex items-center gap-2">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-destructive" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                    <span>{error}</span>
                </div>
            )}
            {/* SEO Pre-Flight Panel */}
            {seoValidation && (seoValidation.errors.length > 0 || seoValidation.warnings.length > 0) && (
                <div className="rounded-xl border overflow-hidden">
                    {seoValidation.errors.length > 0 && (
                        <div className="bg-destructive/10 border-b border-destructive/20 px-4 py-3">
                            <div className="flex items-center gap-2 mb-1">
                                <AlertTriangle className="w-4 h-4 text-destructive flex-shrink-0" />
                                <span className="text-sm font-semibold text-destructive">SEO Validation — Generation Blocked</span>
                            </div>
                            <ul className="space-y-1 pl-6 list-disc">
                                {seoValidation.errors.map((e, i) => (
                                    <li key={i} className="text-xs text-destructive">{e}</li>
                                ))}
                            </ul>
                        </div>
                    )}
                    {seoValidation.warnings.length > 0 && (
                        <div className="bg-amber-500/10 px-4 py-3">
                            <div className="flex items-center gap-2 mb-1">
                                <AlertTriangle className="w-4 h-4 text-amber-500 flex-shrink-0" />
                                <span className="text-sm font-semibold text-amber-600 dark:text-amber-400">SEO Warnings</span>
                            </div>
                            <ul className="space-y-1 pl-6 list-disc">
                                {seoValidation.warnings.map((w, i) => (
                                    <li key={i} className="text-xs text-amber-700 dark:text-amber-300">{w}</li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            )}
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-foreground">Content Studio</h1>
                    <p className="text-muted-foreground">Configure and generate your article</p>
                </div>
                <div className="flex gap-3">
                    <button
                        onClick={() => setShowKeywordModal(true)}
                        disabled={!keywordModalIdea}
                        className="flex items-center gap-2 px-4 py-2 bg-background border border-border rounded-xl hover:bg-muted transition disabled:opacity-50 disabled:cursor-not-allowed"
                        title="Maintain primary and secondary keywords"
                    >
                        <KeyRound className="w-4 h-4" />
                        Keywords
                    </button>
                    <button
                        onClick={handleSave}
                        disabled={saving}
                        className="flex items-center gap-2 px-4 py-2 bg-background border border-border rounded-xl hover:bg-muted transition"
                    >
                        {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
                        Save
                    </button>
                    <button
                        onClick={handleGenerate}
                        disabled={generating || (seoValidation !== null && !seoValidation.isValid)}
                        className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-xl hover:bg-primary/90 transition shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
                        title={seoValidation && !seoValidation.isValid ? seoValidation.errors[0] : undefined}
                    >
                        {generating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Wand2 className="w-4 h-4" />}
                        Generate Article
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Column: Form */}
                <div className="lg:col-span-2 space-y-6">
                    <div className="bg-background p-6 rounded-2xl border border-border shadow-sm space-y-4">
                        <h3 className="font-semibold text-foreground">Article Details</h3>

                        <div>
                            <label className="block text-sm font-medium mb-1">Article Title</label>
                            <input
                                className="w-full px-4 py-2 rounded-xl border border-border bg-muted/50 focus:ring-2 focus:ring-ring outline-none"
                                value={formData.title}
                                onChange={(e) => handleChange('title', e.target.value)}
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium mb-1">Brief Description</label>
                            <textarea
                                className="w-full px-4 py-2 rounded-xl border border-border bg-muted/50 focus:ring-2 focus:ring-ring outline-none h-32"
                                value={formData.description}
                                onChange={(e) => handleChange('description', e.target.value)}
                                placeholder="Describe what you want to write about..."
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium mb-1">Keywords</label>
                            <input
                                className="w-full px-4 py-2 rounded-xl border border-border bg-muted/50 focus:ring-2 focus:ring-ring outline-none"
                                value={formData.keywords}
                                onChange={(e) => handleChange('keywords', e.target.value)}
                                placeholder="Comma separated keywords"
                            />
                        </div>

                        {/* SEO-First Directional Shift Toggle */}
                        <div className={`flex items-start gap-3 p-3 rounded-xl border transition ${
                            seoShiftEnabled
                                ? 'border-primary/30 bg-primary/5'
                                : 'border-border bg-muted/30'
                        }`}>
                            <button
                                id="seo-shift-toggle"
                                onClick={() => setSeoShiftEnabled(v => !v)}
                                className={`mt-0.5 relative w-9 h-5 rounded-full transition-colors flex-shrink-0 ${
                                    seoShiftEnabled ? 'bg-primary' : 'bg-muted-foreground/30'
                                }`}
                                aria-checked={seoShiftEnabled}
                                role="switch"
                            >
                                <span className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${
                                    seoShiftEnabled ? 'translate-x-4' : 'translate-x-0'
                                }`} />
                            </button>
                            <div className="flex-1">
                                <label htmlFor="seo-shift-toggle" className="text-sm font-medium text-foreground cursor-pointer">
                                    GEO/SEO First & Generative App Mode
                                </label>
                                <p className="text-xs text-muted-foreground mt-0.5">
                                    Enrich the generation with a strict GEO-SEO directive. Optimizes for Google search rankings and Generative AI engines (ChatGPT, Gemini, Perplexity) by surfacing citations, entities, and direct answers.
                                </p>
                            </div>
                        </div>
                    </div>

                    <div className="bg-background p-6 rounded-2xl border border-border shadow-sm space-y-4">
                        <h3 className="font-semibold text-foreground">Configuration</h3>

                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm font-medium mb-1">Tone</label>
                                <select
                                    className="w-full px-4 py-2 rounded-xl border border-border bg-muted/50 focus:ring-2 focus:ring-ring outline-none"
                                    value={formData.tone}
                                    onChange={(e) => handleChange('tone', e.target.value)}
                                >
                                    {TONE_OPTIONS.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm font-medium mb-1">Target Word Count</label>
                                <input
                                    type="number"
                                    className="w-full px-4 py-2 rounded-xl border border-border bg-muted/50 focus:ring-2 focus:ring-ring outline-none"
                                    value={formData.articleLength}
                                    onChange={(e) => handleChange('articleLength', e.target.value)}
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium mb-1">LAI Model</label>
                                <div className="relative">
                                    <BrainCircuit className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                                    <select
                                        className="w-full pl-9 pr-4 py-2 rounded-xl border border-border bg-muted/50 focus:ring-2 focus:ring-ring outline-none appearance-none"
                                        value={formData.llmModel}
                                        onChange={(e) => handleChange('llmModel', e.target.value)}
                                    >
                                        <option value="">
                                            {llmModels.length > 0 ? 'Select a Model' : 'No models available'}
                                        </option>
                                        {llmModels.map(model => (
                                            <option key={model.id} value={model.model_name}>
                                                {model.name || model.model_name}
                                            </option>
                                        ))}
                                    </select>
                                </div>
                            </div>
                        </div>

                        <div className="pt-4 border-t border-border">
                            <h4 className="text-sm font-medium text-foreground mb-3">RAG & Research</h4>
                            <div className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium mb-1">RAG Collection</label>
                                    <select
                                        className="w-full px-4 py-2 rounded-xl border border-border bg-muted/50 focus:ring-2 focus:ring-ring outline-none"
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
                                            <label className="block text-sm font-medium mb-1">Query Type</label>
                                            <select
                                                className="w-full px-4 py-2 rounded-xl border border-border bg-muted/50 focus:ring-2 focus:ring-ring outline-none"
                                                value={formData.ragQueryType}
                                                onChange={(e) => handleChange('ragQueryType', e.target.value)}
                                            >
                                                {RAG_QUERY_TYPES.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
                                            </select>
                                        </div>
                                        <div>
                                            <label className="block text-sm font-medium mb-1">Emphasis</label>
                                            <select
                                                className="w-full px-4 py-2 rounded-xl border border-border bg-muted/50 focus:ring-2 focus:ring-ring outline-none"
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
                                        className="w-4 h-4 rounded border-border text-primary focus:ring-ring"
                                        checked={formData.claimsValidation}
                                        onChange={(e) => handleChange('claimsValidation', e.target.checked)}
                                    />
                                    <label htmlFor="claims" className="text-sm">Enable Claims Validation (Web Search)</label>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Right Column: Metrics & Insights */}
                <div className="space-y-6">
                    <div className="bg-background p-6 rounded-2xl border border-border shadow-sm">
                        <div className="flex items-center gap-2 mb-4">
                            <BarChart3 className="w-5 h-5 text-primary" />
                            <h3 className="font-semibold text-foreground">Content Metrics</h3>
                        </div>

                        <div className="space-y-6">
                            {/* Top Section: Main Gauges */}
                            <div className="grid grid-cols-2 gap-4 justify-items-center">
                                <Gauge value={article.seo_optimization_score || 0} label="SEO Score" color="text-primary" explanation={METRIC_EXPLANATIONS.seo_score} />

                                <Gauge
                                    value={article.selected_keyword_difficulty ?? article.avg_keyword_difficulty ?? article.competition_score ?? 0}
                                    label="Key. Difficulty"
                                    color={getCompetitionColor(article.selected_keyword_difficulty ?? article.avg_keyword_difficulty ?? article.competition_score ?? 0)}
                                    explanation={METRIC_EXPLANATIONS.difficulty}
                                />
                                <Gauge
                                    value={(article.selected_keyword_search_volume ?? article.total_search_volume) ? Math.min(((article.selected_keyword_search_volume ?? article.total_search_volume ?? 0) / 1000), 100) : 0}
                                    displayValue={article.selected_keyword_search_volume ?? article.total_search_volume ?? 0}
                                    label="Search Volume"
                                    color="text-chart-1"
                                    explanation={METRIC_EXPLANATIONS.search_volume}
                                    unit=""
                                />
                                <Gauge value={article.readability_score || 0} label="Readability" color="text-chart-2" />
                            </div>

                            <div className="border-t border-border my-6"></div>

                            {/* Secondary Scores Grid */}
                            <div className="grid grid-cols-2 gap-3">
                                <div className="p-2 bg-muted/50 rounded-lg text-center">
                                    <div className="text-lg font-bold text-foreground">{article.overall_quality_score || 0}%</div>
                                    <div className="text-[10px] text-muted-foreground uppercase">Quality</div>
                                </div>
                                <div className="p-2 bg-muted/50 rounded-lg text-center relative group">
                                    <div className="text-lg font-bold text-foreground">{article.traffic_potential_score || 0}%</div>
                                    <div className="flex justify-center items-center gap-1">
                                        <div className="text-[10px] text-muted-foreground uppercase">Traffic</div>
                                        <MetricTooltip explanation={METRIC_EXPLANATIONS.traffic_potential} />
                                    </div>
                                </div>
                            </div>

                            <div className="border-t border-border my-6"></div>

                            {/* Bottom Section: Text Metrics */}
                            <div className="space-y-3">
                                <div className="flex items-center justify-between p-3 bg-muted/50 rounded-xl">
                                    <span className="text-sm text-muted-foreground">Difficulty Level</span>
                                    <span className="font-medium text-foreground">{article.difficulty_level || '-'}</span>
                                </div>
                                <div className="flex items-center justify-between p-3 bg-muted/50 rounded-xl">
                                    <div className="flex items-center gap-1">
                                        <span className="text-sm text-muted-foreground">Est. Reading Time</span>
                                        <MetricTooltip explanation={METRIC_EXPLANATIONS.reading_time} />
                                    </div>
                                    <span className="font-medium text-foreground">{formatReadingTime(article.estimated_reading_time)}</span>
                                </div>
                                <div className="p-3 bg-muted/50 rounded-xl">
                                    <div className="flex items-center gap-1 mb-1">
                                        <span className="text-sm text-muted-foreground block">Target Audience</span>
                                        <MetricTooltip explanation={METRIC_EXPLANATIONS.audience_align} />
                                    </div>
                                    <span className="font-medium text-foreground text-sm">{article.target_audience || '-'}</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="bg-background p-6 rounded-2xl border border-border shadow-sm">
                        <h3 className="font-semibold text-foreground mb-4">Affiliate Opportunities</h3>
                        {article.affiliate_opportunities?.programs?.length > 0 ? (
                            <div className="space-y-3">
                                {article.affiliate_opportunities.programs.slice(0, 3).map((prog: any, i: number) => (
                                    <div key={i} className="p-3 border border-border rounded-xl hover:bg-muted/30 transition">
                                        <div className="font-medium text-sm truncate">{prog.name}</div>
                                        <div className="text-xs text-emerald-500 dark:text-emerald-400 mt-1">{prog.commission_rate}% Commission</div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="text-sm text-muted-foreground text-center py-4">No opportunities found</div>
                        )}
                    </div>

                    <div className="bg-background p-6 rounded-2xl border border-border shadow-sm">
                        <div className="flex items-center gap-2 mb-4">
                            <BrainCircuit className="w-5 h-5 text-primary" />
                            <h3 className="font-semibold text-foreground">Keyword Strategy</h3>
                        </div>
                        <div className="space-y-3">
                            <div className="flex items-center justify-between p-3 bg-muted/50 rounded-xl">
                                <span className="text-sm text-muted-foreground">Primary Keyword</span>
                                <span className="font-medium text-foreground text-right max-w-[60%] truncate">
                                    {article.primary_keywords?.[0] ?? article.search_phrase ?? article.primary_keyword ?? '-'}
                                </span>
                            </div>
                            <div className="grid grid-cols-2 gap-3">
                                <div className="p-3 bg-muted/50 rounded-xl">
                                    <div className="text-[10px] text-muted-foreground uppercase mb-1">Source</div>
                                    <div className="font-medium text-sm text-foreground truncate">
                                        {article.keyword_selection_source || article.keyword_research_source || '-'}
                                    </div>
                                </div>
                                <div className="p-3 bg-muted/50 rounded-xl">
                                    <div className="text-[10px] text-muted-foreground uppercase mb-1">Confidence</div>
                                    <div className="font-medium text-sm text-foreground">
                                        {formatConfidence(article.keyword_research_confidence)}
                                    </div>
                                </div>
                            </div>
                                <div className="grid grid-cols-2 gap-3">
                                    <div className="p-3 bg-muted/50 rounded-xl">
                                        <div className="text-[10px] text-muted-foreground uppercase mb-1">Intent</div>
                                        <div className="font-medium text-sm text-foreground">{article.selected_keyword_intent || '-'}</div>
                                    </div>
                                    <div className="p-3 bg-muted/50 rounded-xl">
                                        <div className="text-[10px] text-muted-foreground uppercase mb-1">Volume</div>
                                        <div className="font-medium text-sm text-foreground">{article.selected_keyword_search_volume ?? article.total_search_volume ?? '-'}</div>
                                    </div>
                                </div>
                                <div className="p-3 bg-muted/50 rounded-xl">
                                    <div className="text-[10px] text-muted-foreground uppercase mb-1">Difficulty</div>
                                        <div className="font-medium text-sm text-foreground">{article.selected_keyword_difficulty ?? article.avg_keyword_difficulty ?? '-'}</div>
                                </div>
                                <div className="p-3 bg-muted/50 rounded-xl">
                                    <div className="text-[10px] text-muted-foreground uppercase mb-1">Secondary Keywords</div>
                                    <div className="font-medium text-sm text-foreground">
                                        {(() => {
                                            const list = article.secondary_keywords ?? [];
                                            if (list.length > 0) return list.slice(0, 5).join(', ');
                                            const raw = article.secondary_keywords_json;
                                            const fallback = Array.isArray(raw)
                                                ? raw
                                                : (typeof raw === 'string'
                                                    ? raw.split(',').map((x) => x.trim()).filter(Boolean)
                                                    : []);
                                            return fallback.length > 0 ? fallback.slice(0, 5).join(', ') : '-';
                                        })()}
                                    </div>
                                </div>
                            </div>
                        </div>

                    {/* SEO Data Panel */}
                    <div className="bg-background p-6 rounded-2xl border border-border shadow-sm">
                        <div className="flex items-center gap-2 mb-4">
                            <ShieldCheck className="w-5 h-5 text-primary" />
                            <h3 className="font-semibold text-foreground">SEO Data</h3>
                            {seoValidation && (
                                <span className={`ml-auto text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wide ${
                                    seoValidation.isValid
                                        ? 'bg-emerald-500/15 text-emerald-400'
                                        : 'bg-destructive/15 text-destructive'
                                }`}>
                                    {seoValidation.isValid ? 'Ready' : 'Incomplete'}
                                </span>
                            )}
                        </div>
                        <div className="space-y-3">
                            <div className="p-3 bg-muted/50 rounded-xl">
                                <div className="flex items-center gap-1.5 mb-1">
                                    <Tag className="w-3 h-3 text-muted-foreground" />
                                    <span className="text-[10px] text-muted-foreground uppercase">Primary Keyword</span>
                                </div>
                                <div className="font-medium text-sm text-foreground">
                                    {article.primary_keywords?.[0] ?? article.search_phrase ?? (
                                        <span className="text-muted-foreground italic">Not set — use Keyword Intelligence</span>
                                    )}
                                </div>
                            </div>
                            {(article.secondary_keywords?.length ?? 0) > 0 && (
                                <div className="p-3 bg-muted/50 rounded-xl">
                                    <div className="flex items-center gap-1.5 mb-1.5">
                                        <Tag className="w-3 h-3 text-muted-foreground" />
                                        <span className="text-[10px] text-muted-foreground uppercase">Secondary Keywords ({article.secondary_keywords!.length})</span>
                                    </div>
                                    <div className="flex flex-wrap gap-1">
                                        {article.secondary_keywords!.slice(0, 6).map((kw) => (
                                            <span key={kw} className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 border border-primary/20 text-primary">
                                                {kw}
                                            </span>
                                        ))}
                                        {(article.secondary_keywords!.length > 6) && (
                                            <span className="text-[10px] text-muted-foreground">+{article.secondary_keywords!.length - 6} more</span>
                                        )}
                                    </div>
                                </div>
                            )}
                            <div className="grid grid-cols-2 gap-3">
                                <div className="p-3 bg-muted/50 rounded-xl">
                                    <div className="flex items-center gap-1.5 mb-1">
                                        <Globe2 className="w-3 h-3 text-muted-foreground" />
                                        <span className="text-[10px] text-muted-foreground uppercase">Target Domain</span>
                                    </div>
                                    <div className="font-medium text-xs text-foreground truncate">
                                        {article.domain ?? <span className="text-muted-foreground italic">Not linked</span>}
                                    </div>
                                </div>
                                <div className="p-3 bg-muted/50 rounded-xl">
                                    <div className="text-[10px] text-muted-foreground uppercase mb-1">WP Category ID</div>
                                    <div className={`font-medium text-xs ${article.wordpress_category_id ? 'text-foreground' : 'text-amber-500'}`}>
                                        {article.wordpress_category_id ?? 'Not mapped'}
                                    </div>
                                </div>
                            </div>
                            {(() => {
                                const primaryKw = article.primary_keywords?.[0] ?? article.search_phrase ?? '';
                                if (!primaryKw) return null;
                                const geo = computeGEOContext(primaryKw, article.domain);
                                if (!geo.hasGEOSignal) return null;
                                return (
                                    <div className="p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-xl">
                                        <div className="flex items-center gap-1.5 mb-1">
                                            <BrainCircuit className="w-3 h-3 text-emerald-400" />
                                            <span className="text-[10px] text-emerald-400 uppercase font-semibold">Generative Engine Optimization</span>
                                        </div>
                                        <p className="text-xs text-emerald-300">Ready for AI-Answer Engines: {geo.optimizationFocus}</p>
                                    </div>
                                );
                            })()}
                        </div>
                    </div>
                </div>
            </div>

            <GenerationModal
                articleId={articleId || ''}
                taskId={taskId}
                isOpen={showProgress}
                onClose={() => {
                    setShowProgress(false);
                    setTaskId(null);
                }}
                onComplete={() => {
                    setShowProgress(false);
                    setTaskId(null);
                    navigate('/');
                }}
            />

            {showKeywordModal && keywordModalIdea && user?.id && (
                <KeywordIntelligenceModal
                    isOpen={showKeywordModal}
                    onClose={() => setShowKeywordModal(false)}
                    idea={keywordModalIdea}
                    onSave={async (primary, secondary, metrics, rawOutput) => {
                        return contentIdeasService.updateTitleKeywordSelection(
                            article.id,
                            user.id,
                            primary,
                            secondary,
                            metrics,
                            rawOutput
                        );
                    }}
                    onSaved={(primary, secondary, metrics, rawOutput) => {
                        const nextPrimary = primary ? [primary] : [];
                        setArticle((prev) => {
                            if (!prev) return prev;
                            const updated: ArticleData = {
                                ...prev,
                                primary_keywords: nextPrimary,
                                secondary_keywords: secondary,
                                search_phrase: primary || prev.search_phrase,
                                primary_keyword: primary || prev.primary_keyword,
                                Keywords: [primary, ...secondary].filter(Boolean).join(', '),
                                selected_keyword_search_volume: metrics.volume ?? prev.selected_keyword_search_volume,
                                selected_keyword_difficulty: metrics.difficulty ?? prev.selected_keyword_difficulty,
                                raw_dataforseo_output: rawOutput ?? prev.raw_dataforseo_output,
                            };
                            setSeoValidation(validateSEOReadiness(updated));
                            return updated;
                        });
                        setFormData((prev) => ({
                            ...prev,
                            keywords: [primary, ...secondary].filter(Boolean).join(', '),
                        }));
                    }}
                />
            )}
        </div>
    );
};
