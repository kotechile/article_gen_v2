
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

function sanitizeErrorForLog(error: unknown): Record<string, unknown> {
    if (axios.isAxiosError(error)) {
        return {
            type: 'axios_error',
            message: error.message,
            status: error.response?.status,
            statusText: error.response?.statusText,
            method: error.config?.method,
            url: error.config?.url,
            serverMessage: error.response?.data?.message || error.response?.data?.error?.message,
        };
    }

    if (error instanceof Error) {
        return {
            type: 'error',
            name: error.name,
            message: error.message,
        };
    }

    return {
        type: typeof error,
        message: String(error),
    };
}

function logSafeError(prefix: string, error: unknown, level: 'error' | 'warn' = 'error') {
    const payload = sanitizeErrorForLog(error);
    if (level === 'warn') {
        console.warn(prefix, payload);
        return;
    }
    console.error(prefix, payload);
}

function getContentStudioGenerationStorageKey(articleId: string): string {
    return `content_studio_generation_${articleId}`;
}


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
    source_strategy?: string;
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
    topic_id?: string | null;
    // Affiliate
    affiliate_opportunities?: any;
    // DataForSEO traces for Keyword Intelligence modal
    raw_dataforseo_output?: any;
    raw_supabase_output?: any;
    idea_metadata?: any;
}

function normalizeKeywordList(value: unknown): string[] {
    return extractKeywordValues(value);
}

function extractKeywordValues(value: unknown, depth = 0): string[] {
    if (depth > 5 || value === null || value === undefined) return [];

    if (Array.isArray(value)) {
        return value.flatMap((item) => extractKeywordValues(item, depth + 1));
    }

    if (typeof value === 'object') {
        const maybeKeyword = (value as any)?.keyword;
        if (typeof maybeKeyword === 'string') {
            return extractKeywordValues(maybeKeyword, depth + 1);
        }
        return [];
    }

    if (typeof value !== 'string') return [];
    const raw = value.trim();
    if (!raw) return [];

    let cleaned = raw;
    while (
        (cleaned.startsWith('[') && cleaned.endsWith(']')) ||
        (cleaned.startsWith('"') && cleaned.endsWith('"')) ||
        (cleaned.startsWith("'") && cleaned.endsWith("'"))
    ) {
        cleaned = cleaned.slice(1, -1).trim();
        if (!cleaned) break;
    }
    if (!cleaned) return [];

    try {
        const parsed = JSON.parse(raw);
        if (typeof parsed === 'string' && parsed !== raw) {
            return extractKeywordValues(parsed, depth + 1);
        }
        if (Array.isArray(parsed)) {
            return parsed.flatMap((item) => extractKeywordValues(item, depth + 1));
        }
    } catch {
        // Fall back to comma-separated parsing.
    }

    const parts = cleaned
        .split(',')
        .map((item) => item.trim().replace(/^["']|["']$/g, '').trim())
        .filter(Boolean);
    return parts.length > 1 ? parts : [cleaned.replace(/^["']|["']$/g, '').trim()];
}

function getPrimaryKeyword(article: ArticleData | null, fallbackKeywords = ''): string {
    const candidates = [
        ...(article?.primary_keywords ?? []),
        ...extractKeywordValues(article?.search_phrase),
        ...extractKeywordValues(article?.primary_keyword),
        ...extractKeywordValues(fallbackKeywords),
    ];

    for (const candidate of candidates) {
        const keyword = String(candidate || '').trim();
        if (keyword) return keyword;
    }

    return '';
}

function pickFirstNonEmptyString(...values: unknown[]): string {
    for (const value of values) {
        if (typeof value === 'string') {
            const trimmed = value.trim();
            if (trimmed) return trimmed;
        }
    }
    return '';
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
    const primaryKeyword = getPrimaryKeyword(article);

    const hasPrimary = Boolean(primaryKeyword);

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

function deriveSEOValidation(
    article: ArticleData | null,
    seoShiftEnabled: boolean
): SEOValidation | null {
    if (!seoShiftEnabled || !article) {
        return null;
    }
    return validateSEOReadiness(article);
}

function shouldEnableSEOShiftByDefault(article: ArticleData): boolean {
    const hasDomain = Boolean(String(article.domain || '').trim());
    const hasTopicLink = Boolean(String((article as any).topic_id || '').trim());
    const hasIdeaLink = Boolean(String(article.source_idea_id || '').trim());
    const hasWordPressMapping = Boolean(article.wordpress_category_id);

    return hasDomain || hasTopicLink || hasIdeaLink || hasWordPressMapping;
}

interface RagCollection {
    id: string;
    name: string;
}

interface AppSettings {
    // Add specific keys if needed
    [key: string]: any;
}

interface MetadataRefinementPreview {
    refined_title: string;
    refined_description: string;
    rationale?: string;
    changed: boolean;
    fallback_used?: boolean;
    options?: Array<{
        refined_title: string;
        refined_description: string;
        rationale?: string;
    }>;
}

function buildRefinementSignature(
    title: string,
    description: string,
    primaryKeyword: string
): string {
    return [
        String(title || '').trim(),
        String(description || '').trim(),
        String(primaryKeyword || '').trim(),
    ].join('||');
}

function titleHasKeywordAndWithinLengthLimit(
    title: string,
    primaryKeyword: string
): boolean {
    if (!primaryKeyword) return false;
    const trimmed = title.trim();
    return trimmed.length <= 60 && trimmed.toLowerCase().includes(primaryKeyword.toLowerCase());
}

function inferSourceMode(params: {
    sourceStrategy?: string | null;
    ragCollection?: string | null;
    claimsValidation?: boolean;
}): string {
    const explicit = String(params.sourceStrategy || '').trim();
    if (explicit && SOURCE_MODE_OPTIONS.some((opt) => opt.value === explicit)) {
        return explicit;
    }

    const hasRag = Boolean(String(params.ragCollection || '').trim());
    const claimsValidation = Boolean(params.claimsValidation);
    if (!hasRag) return 'dossier_only';
    return claimsValidation ? 'dossier_plus_rag_plus_live_web' : 'dossier_plus_rag';
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

const SOURCE_MODE_OPTIONS = [
    { label: "Dossier only", value: "dossier_only" },
    { label: "Dossier + RAG", value: "dossier_plus_rag" },
    { label: "Dossier + RAG + Live Web Refresh", value: "dossier_plus_rag_plus_live_web" },
    { label: "RAG only", value: "rag_only" },
];

const SOURCE_STRATEGY_REFACTOR_ENABLED =
    String(import.meta.env.VITE_SOURCE_STRATEGY_REFACTOR_ENABLED || 'false').toLowerCase() === 'true';

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
    const [showRefinementGate, setShowRefinementGate] = useState(false);
    const [requestingRefinement, setRequestingRefinement] = useState(false);
    const [refinementPreview, setRefinementPreview] = useState<MetadataRefinementPreview | null>(null);
    const [refinementDraft, setRefinementDraft] = useState({ title: '', description: '' });
    const [approvedRefinementSignature, setApprovedRefinementSignature] = useState('');

    const clearStoredGenerationSession = React.useCallback((targetArticleId?: string | null) => {
        if (!targetArticleId) return;
        localStorage.removeItem(getContentStudioGenerationStorageKey(targetArticleId));
        localStorage.removeItem(`gen_progress_${targetArticleId}`);
    }, []);

    // Data State
    const [article, setArticle] = useState<ArticleData | null>(null);
    const [categoryPath, setCategoryPath] = useState<string | null>(null);
    const [ragCollections, setRagCollections] = useState<RagCollection[]>([]);
    const [appSettings, setAppSettings] = useState<AppSettings | null>(null);
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
        sourceMode: 'dossier_only',
        claimsValidation: true,
    });

    useEffect(() => {
        setSeoValidation(deriveSEOValidation(article, seoShiftEnabled));
    }, [article, seoShiftEnabled]);

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

                let resolvedCategoryPath: string | null =
                    typeof (artData as any)?.idea_metadata?.category_context?.category_path === 'string'
                        ? String((artData as any).idea_metadata.category_context.category_path).trim() || null
                        : null;

                if (!resolvedCategoryPath) {
                    let linkedTopicId = (artData as any)?.topic_id ? String((artData as any).topic_id) : null;

                    if (!linkedTopicId && (artData as any)?.source_idea_id) {
                        const { data: linkedIdea } = await supabase
                            .from('content_ideas')
                            .select('topic_id, idea_metadata')
                            .eq('id', (artData as any).source_idea_id)
                            .maybeSingle();

                        linkedTopicId = linkedIdea?.topic_id ? String(linkedIdea.topic_id) : null;
                        const ideaCategoryPath = linkedIdea?.idea_metadata?.category_context?.category_path;
                        if (typeof ideaCategoryPath === 'string' && ideaCategoryPath.trim()) {
                            resolvedCategoryPath = ideaCategoryPath.trim();
                        }
                    }

                    if (linkedTopicId && !resolvedCategoryPath) {
                        const { data: topicRow } = await supabase
                            .from('research_topics')
                            .select('primary_category_id, secondary_category_id')
                            .eq('id', linkedTopicId)
                            .maybeSingle();

                        const categoryIds = [topicRow?.primary_category_id, topicRow?.secondary_category_id].filter(Boolean);
                        if (categoryIds.length > 0) {
                            const { data: categoryRows } = await supabase
                                .from('project_categories')
                                .select('id, name')
                                .in('id', categoryIds as string[]);

                            const categoryById = new Map<string, string>();
                            for (const row of categoryRows || []) {
                                categoryById.set(String((row as any).id), String((row as any).name || '').trim());
                            }

                            resolvedCategoryPath = [
                                categoryById.get(String(topicRow?.primary_category_id || '')),
                                categoryById.get(String(topicRow?.secondary_category_id || '')),
                            ].filter(Boolean).join(' / ') || null;
                        }
                    }
                }

                setArticle(normalizedArticle);
                setSeoShiftEnabled(shouldEnableSEOShiftByDefault(normalizedArticle));
                setCategoryPath(resolvedCategoryPath);
                const savedGenerationSession = localStorage.getItem(getContentStudioGenerationStorageKey(articleId));
                if (savedGenerationSession) {
                    try {
                        const parsed = JSON.parse(savedGenerationSession);
                        const savedTaskId = String(parsed?.taskId || '').trim();
                        const savedArticleId = String(parsed?.articleId || '').trim();
                        if (
                            savedTaskId &&
                            savedArticleId === articleId &&
                            !['Created', 'Generated', 'Editing', 'WP Published', 'Scheduled', 'Error', 'Failed'].includes(String(normalizedArticle.status || ''))
                        ) {
                            setTaskId(savedTaskId);
                            setShowProgress(true);
                        } else {
                            clearStoredGenerationSession(articleId);
                            setTaskId(null);
                            setShowProgress(false);
                        }
                    } catch {
                        clearStoredGenerationSession(articleId);
                        setTaskId(null);
                        setShowProgress(false);
                    }
                } else {
                    setTaskId(null);
                    setShowProgress(false);
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
                    sourceMode: inferSourceMode({
                        sourceStrategy: normalizedArticle.source_strategy,
                        ragCollection: normalizedArticle.rag_collection_name || '',
                        claimsValidation: false,
                    }),
                    claimsValidation: true,
                });

            } catch (error) {
                logSafeError("Error fetching data:", error);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [user, articleId, clearStoredGenerationSession]);

    // Handle Input Changes
    const handleChange = (field: string, value: any) => {
        setFormData((prev) => {
            const next = { ...prev, [field]: value };
            if (SOURCE_STRATEGY_REFACTOR_ENABLED && field === 'ragCollection') {
                const hasRag = Boolean(String(value || '').trim());
                if (!hasRag) {
                    next.sourceMode = 'dossier_only';
                } else if (prev.sourceMode === 'dossier_only') {
                    next.sourceMode = 'dossier_plus_rag';
                }
            }
            return next;
        });
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
            topic_id: article.topic_id || article.source_idea_id || '',
            description: article.userDescription,
            search_phrase: article.search_phrase,
            keyword_metrics: article.selected_keyword_metrics_json || undefined,
            raw_dataforseo_output: article.raw_dataforseo_output ?? null,
            raw_supabase_output: article.raw_supabase_output ?? null,
            idea_metadata: article.idea_metadata || undefined,
        }
        : null;

    // Save Changes
    const handleSave = async (
        overrides?: Partial<typeof formData>
    ): Promise<boolean> => {
        if (!articleId) return false;
        setSaving(true);
        try {
            const effectiveFormData = {
                ...formData,
                ...(overrides || {}),
            };

            // Calculate estimated reading time
            const wpm = 250;
            const words = parseInt(effectiveFormData.articleLength, 10) || 0;
            const minutes = Math.ceil(words / wpm);
            const readingTime = Math.max(1, minutes);

            const updates = {
                Title: effectiveFormData.title,
                userDescription: effectiveFormData.description,
                Keywords: effectiveFormData.keywords,
                articleLength: effectiveFormData.articleLength,
                Tone: effectiveFormData.tone,
                rag_collection_name: effectiveFormData.ragCollection,
                rag_query_type: effectiveFormData.ragQueryType,
                rag_balance_emphasis: effectiveFormData.emphasis,
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

            if (overrides && Object.keys(overrides).length > 0) {
                setFormData((prev) => ({ ...prev, ...overrides }));
            }

            // Optional: show toast
            return true;
        } catch (error) {
            logSafeError("Error saving:", error);
            setError("Failed to save article settings. Please verify DB schema/settings and try again.");
            return false;
        } finally {
            setSaving(false);
        }
    };

    const requestMetadataRefinementPreview = async (params: {
        title: string;
        description: string;
        primaryKw: string;
        secondaryKeywords: string[];
    }): Promise<MetadataRefinementPreview> => {
        const { data: { session } } = await supabase.auth.getSession();
        const token = session?.access_token;
        if (!token) throw new Error("No session token found");

        const ideaMetadata = (article?.idea_metadata && typeof article.idea_metadata === 'object')
            ? article.idea_metadata
            : {};
        const rawOutput = (article?.raw_supabase_output && typeof article.raw_supabase_output === 'object')
            ? article.raw_supabase_output
            : {};
        const affiliatePrograms = Array.isArray((article?.affiliate_opportunities as any)?.programs)
            ? (article?.affiliate_opportunities as any).programs
            : [];
        const supportingEntities = normalizeKeywordList(article?.supporting_entities_json).slice(0, 8);
        const priorityQuestions = normalizeKeywordList(article?.priority_questions_json).slice(0, 6);
        const affiliateOfferNames = affiliatePrograms
            .map((program: any) => pickFirstNonEmptyString(program?.program_name, program?.name, program?.merchant, program?.title))
            .filter(Boolean)
            .slice(0, 6);

        const response = await axios.post(
            `${import.meta.env.VITE_API_URL || 'http://localhost:5001'}/api/v1/research/refine-metadata`,
            {
                title: params.title,
                description: params.description,
                primary_keyword: params.primaryKw || undefined,
                secondary_keywords: params.secondaryKeywords,
                domain: article?.domain || undefined,
                context: {
                    target_audience: article?.target_audience || undefined,
                    article_length: article?.articleLength || undefined,
                    tone: formData.tone || article?.tone || undefined,
                    keyword_intent: article?.selected_keyword_intent || undefined,
                    keyword_search_volume: article?.selected_keyword_search_volume || undefined,
                    keyword_difficulty: article?.selected_keyword_difficulty || undefined,
                    supporting_entities: supportingEntities,
                    priority_questions: priorityQuestions,
                    decision_focus: pickFirstNonEmptyString(
                        (ideaMetadata as any)?.decision_focus,
                        (rawOutput as any)?.decision_focus,
                    ) || undefined,
                    angle_question: pickFirstNonEmptyString(
                        (ideaMetadata as any)?.angle_question,
                        (rawOutput as any)?.angle_question,
                    ) || undefined,
                    primary_user_outcome: pickFirstNonEmptyString(
                        (ideaMetadata as any)?.primary_user_outcome,
                        (rawOutput as any)?.primary_user_outcome,
                    ) || undefined,
                    internal_link_hook: pickFirstNonEmptyString(
                        (ideaMetadata as any)?.internal_link_hook,
                        (rawOutput as any)?.internal_link_hook,
                    ) || undefined,
                    affiliate_offer_names: affiliateOfferNames,
                },
            },
            {
                headers: {
                    Authorization: `Bearer ${token}`,
                    'X-API-Key': 'development',
                }
            }
        );

        const payload = response.data?.data || {};
        const rawOptions = Array.isArray(payload.options) ? payload.options : [];
        const options = rawOptions
            .map((option: any) => ({
                refined_title: String(option?.refined_title || '').trim(),
                refined_description: String(option?.refined_description || '').trim(),
                rationale: String(option?.rationale || '').trim(),
            }))
            .filter((option: any) => option.refined_title && option.refined_description);
        return {
            refined_title: String(payload.refined_title || params.title),
            refined_description: String(payload.refined_description || params.description),
            rationale: String(payload.rationale || ''),
            changed: Boolean(payload.changed),
            fallback_used: Boolean(payload.fallback_used),
            options,
        };
    };

    const startGeneration = async (options?: {
        skipApprovalGate?: boolean;
        approvedTitle?: string;
        approvedDescription?: string;
    }) => {
        if (!articleId || !appSettings) return;
        setGenerating(true);
        setError(null);

        const effectiveTitle = String(options?.approvedTitle ?? formData.title ?? '').trim();
        const effectiveDescription = String(options?.approvedDescription ?? formData.description ?? '').trim();

        try {
            // Save first (including approved metadata overrides when provided)
            const saveOk = await handleSave({
                title: effectiveTitle,
                description: effectiveDescription,
            });
            if (!saveOk) {
                setGenerating(false);
                return;
            }

            // Clear any old progress from local storage
            clearStoredGenerationSession(articleId);

            if (effectiveDescription.length < 10) {
                alert('Please provide a longer description (at least 10 characters).');
                setGenerating(false);
                return;
            }

            // ── Phase 2: Validate SEO readiness ──────────────────────────────────
            if (article && seoShiftEnabled) {
                const validation = validateSEOReadiness(article);
                setSeoValidation(validation);
                if (!validation.isValid) {
                    setGenerating(false);
                    setError(validation.errors.join(' '));
                    return;
                }
            } else {
                setSeoValidation(null);
            }

            // ── Phase 3: Build SEO-enriched brief (Directional Shift) ─────────────
            const primaryKw = getPrimaryKeyword(article, formData.keywords);

            const secondaryKeywords = article?.secondary_keywords ?? [];
            const secondaryKwList = secondaryKeywords.slice(0, 4).join(', ');
            const titleKeywordReady = titleHasKeywordAndWithinLengthLimit(effectiveTitle, primaryKw);

            if (
                seoShiftEnabled &&
                primaryKw &&
                !options?.skipApprovalGate &&
                approvedRefinementSignature !== buildRefinementSignature(
                    effectiveTitle,
                    effectiveDescription,
                    primaryKw
                ) &&
                !titleKeywordReady
            ) {
                setRequestingRefinement(true);
                try {
                    let preview: MetadataRefinementPreview;
                    try {
                        preview = await requestMetadataRefinementPreview({
                            title: effectiveTitle,
                            description: effectiveDescription,
                            primaryKw,
                            secondaryKeywords,
                        });
                    } catch (previewError) {
                        logSafeError('Metadata refinement preview failed; opening manual approval with original metadata.', previewError, 'warn');
                        preview = {
                            refined_title: effectiveTitle,
                            refined_description: effectiveDescription,
                            changed: false,
                            fallback_used: true,
                            rationale: 'Automatic refinement is unavailable right now. Review and approve to continue.',
                            options: [{
                                refined_title: effectiveTitle,
                                refined_description: effectiveDescription,
                                rationale: 'Original metadata',
                            }],
                        };
                    }
                    setRefinementPreview(preview);
                    const firstOption = preview.options?.[0];
                    setRefinementDraft({
                        title: firstOption?.refined_title || preview.refined_title || effectiveTitle,
                        description: firstOption?.refined_description || preview.refined_description || effectiveDescription,
                    });
                    setShowRefinementGate(true);
                } finally {
                    setRequestingRefinement(false);
                }
                setGenerating(false);
                return;
            }

            let generationBrief = effectiveDescription;
            let seodirective = '';
            const selectedSourceMode = SOURCE_STRATEGY_REFACTOR_ENABLED
                ? inferSourceMode({
                    sourceStrategy: formData.sourceMode,
                    ragCollection: formData.ragCollection,
                    claimsValidation: formData.claimsValidation,
                })
                : inferSourceMode({
                    ragCollection: formData.ragCollection,
                    claimsValidation: formData.claimsValidation,
                });
            const strategyUsesRag = ['dossier_plus_rag', 'dossier_plus_rag_plus_live_web', 'rag_only'].includes(selectedSourceMode);
            const strategyUsesLiveWeb = selectedSourceMode === 'dossier_plus_rag_plus_live_web';

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
                    effectiveDescription,
                ].filter(Boolean).join('\n');

                generationBrief = seoParts;
                seodirective = 'seo_geo_llm_enriched';
            }

            const payload = {
                brief: generationBrief,
                keywords: formData.keywords,
                draft_title: effectiveTitle,
                domain: article?.domain || undefined,
                wordpress_category_id: article?.wordpress_category_id ?? undefined,
                seo_primary_keyword: primaryKw || undefined,
                seo_secondary_keywords: article?.secondary_keywords?.length ? article.secondary_keywords : undefined,
                seo_directive: seodirective || undefined,
                // Keep normalized fields for compatibility with both research API variants.
                depth: formData.emphasis === 'balanced' ? 'standard' : 'comprehensive',
                tone: formData.tone,
                target_word_count: parseInt(formData.articleLength, 10),
                source_strategy: selectedSourceMode,
                claims_research_enabled: strategyUsesLiveWeb,
                rag_enabled: strategyUsesRag && !!formData.ragCollection,
                rag_collection_name: formData.ragCollection,
                rag_endpoint: (strategyUsesRag && formData.ragCollection)
                    ? appSettings.rag_url + formData.ragQueryType
                    : undefined,
                rag_balance_emphasis: formData.emphasis,
                article_id: articleId,
            };

            // Call Backend
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
                const nextTaskId = String(response.data.research_id);
                setTaskId(nextTaskId);
                localStorage.setItem(
                    getContentStudioGenerationStorageKey(articleId),
                    JSON.stringify({
                        articleId,
                        taskId: nextTaskId,
                        startedAt: new Date().toISOString(),
                    }),
                );
                // Show Progress Modal only when we have a valid task id
                setShowProgress(true);
                setGenerating(false);
            } else {
                throw new Error("Generation started but no task id was returned by the API.");
            }

        } catch (error: any) {
            logSafeError("Generation error:", error);
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

    // Generate Article
    const handleGenerate = async () => {
        await startGeneration();
    };

    const handleApproveRefinement = async () => {
        const primaryKw = getPrimaryKeyword(article, formData.keywords);
        const signature = buildRefinementSignature(
            refinementDraft.title,
            refinementDraft.description,
            primaryKw
        );
        setApprovedRefinementSignature(signature);
        setShowRefinementGate(false);
        await startGeneration({
            skipApprovalGate: true,
            approvedTitle: refinementDraft.title,
            approvedDescription: refinementDraft.description,
        });
    };

    if (loading) return <div className="flex justify-center p-12"><Loader2 className="animate-spin text-primary" /></div>;
    if (!article) return <div className="p-8 text-muted-foreground">Article not found</div>;

    const getCompetitionColor = (score: number) => {
        if (score < 40) return 'text-chart-2';
        if (score < 70) return 'text-chart-4';
        return 'text-destructive';
    };

    const selectedSourceMode = inferSourceMode({
        sourceStrategy: formData.sourceMode,
        ragCollection: formData.ragCollection,
        claimsValidation: formData.claimsValidation,
    });
    const sourceModeUsesRag = ['dossier_plus_rag', 'dossier_plus_rag_plus_live_web', 'rag_only'].includes(selectedSourceMode);

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
            {seoShiftEnabled && seoValidation && (seoValidation.errors.length > 0 || seoValidation.warnings.length > 0) && (
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
                        onClick={() => {
                            void handleSave();
                        }}
                        disabled={saving}
                        className="flex items-center gap-2 px-4 py-2 bg-background border border-border rounded-xl hover:bg-muted transition"
                    >
                        {saving ? <Loader2 className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
                        Save
                    </button>
                    <button
                        onClick={handleGenerate}
                        disabled={generating || requestingRefinement || (seoShiftEnabled && seoValidation !== null && !seoValidation.isValid)}
                        className="flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-xl hover:bg-primary/90 transition shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
                        title={seoShiftEnabled && seoValidation && !seoValidation.isValid ? seoValidation.errors[0] : undefined}
                    >
                        {(generating || requestingRefinement) ? <Loader2 className="w-4 h-4 animate-spin" /> : <Wand2 className="w-4 h-4" />}
                        {requestingRefinement ? 'Preparing Approval Gate...' : 'Generate Article'}
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
                        </div>

                        <div className="rounded-xl border border-border bg-muted/30 px-4 py-3 text-sm text-muted-foreground">
                            LLM selection is now managed automatically in the backend by task type. Article generation, SVG infographics, final review, and future ToC/deep research flows can each use a different model profile.
                        </div>

                        <div className="pt-4 border-t border-border">
                            <h4 className="text-sm font-medium text-foreground mb-3">
                                {SOURCE_STRATEGY_REFACTOR_ENABLED ? 'Sources' : 'RAG & Research'}
                            </h4>
                            <div className="space-y-4">
                                {SOURCE_STRATEGY_REFACTOR_ENABLED && (
                                    <div>
                                        <label className="block text-sm font-medium mb-1">Source Mode</label>
                                        <select
                                            className="w-full px-4 py-2 rounded-xl border border-border bg-muted/50 focus:ring-2 focus:ring-ring outline-none"
                                            value={selectedSourceMode}
                                            onChange={(e) => handleChange('sourceMode', e.target.value)}
                                        >
                                            {SOURCE_MODE_OPTIONS.map((opt) => (
                                                <option key={opt.value} value={opt.value}>{opt.label}</option>
                                            ))}
                                        </select>
                                        <p className="text-xs text-muted-foreground mt-1">
                                            Use Deep Research dossier as baseline; optionally add private RAG knowledge and live web refresh.
                                        </p>
                                    </div>
                                )}

                                {(!SOURCE_STRATEGY_REFACTOR_ENABLED || sourceModeUsesRag) && (
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
                                )}

                                {formData.ragCollection && (
                                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                                        <div>
                                            <label className="block text-sm font-medium mb-1">
                                                {SOURCE_STRATEGY_REFACTOR_ENABLED ? 'Advanced RAG: Query Type' : 'Query Type'}
                                            </label>
                                            <select
                                                className="w-full px-4 py-2 rounded-xl border border-border bg-muted/50 focus:ring-2 focus:ring-ring outline-none"
                                                value={formData.ragQueryType}
                                                onChange={(e) => handleChange('ragQueryType', e.target.value)}
                                            >
                                                {RAG_QUERY_TYPES.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
                                            </select>
                                        </div>
                                        <div>
                                            <label className="block text-sm font-medium mb-1">
                                                {SOURCE_STRATEGY_REFACTOR_ENABLED ? 'Advanced RAG: Emphasis' : 'Emphasis'}
                                            </label>
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

                                {!SOURCE_STRATEGY_REFACTOR_ENABLED && (
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
                                )}
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
                                    {getPrimaryKeyword(article) || '-'}
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
                            {seoShiftEnabled && seoValidation && (
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
                                    {getPrimaryKeyword(article) || (
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
                                    {categoryPath && (
                                        <div className="mt-1 text-[10px] text-muted-foreground">
                                            {categoryPath}
                                        </div>
                                    )}
                                </div>
                            </div>
                            {(() => {
                                const primaryKw = getPrimaryKeyword(article);
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

            {showRefinementGate && (
                <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4">
                    <div className="w-full max-w-2xl bg-background border border-border rounded-2xl shadow-xl p-6 space-y-4">
                        <div>
                            <h3 className="text-lg font-semibold text-foreground">Approve GEO Metadata Refinement</h3>
                            <p className="text-sm text-muted-foreground">
                                The title does not include the primary keyword or exceeds 60 characters. Refine the metadata before generation.
                            </p>
                        </div>

                        {refinementPreview?.rationale && (
                            <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-300">
                                {refinementPreview.rationale}
                            </div>
                        )}

                        <div className="space-y-5">
                            {Array.isArray(refinementPreview?.options) && refinementPreview!.options!.length > 1 && (
                                <section className="rounded-2xl border border-sky-400/35 bg-slate-950/80 p-4 shadow-[inset_0_0_0_1px_rgba(56,189,248,0.08)]">
                                    <div className="flex items-start justify-between gap-3 mb-3">
                                        <div>
                                            <div className="inline-flex items-center gap-2 rounded-full border border-sky-400/35 bg-sky-400/10 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-sky-200 mb-2">
                                                Step 1
                                            </div>
                                            <label className="block text-sm font-semibold text-foreground">Choose a Proposed Option</label>
                                            <p className="text-xs text-muted-foreground mt-1">
                                                Pick the closest keyword-aligned draft. The selected option will populate the editable fields below.
                                            </p>
                                        </div>
                                        <div className="shrink-0 rounded-full border border-sky-400/30 bg-sky-400/10 px-2.5 py-1 text-[11px] font-medium text-sky-200">
                                            {refinementPreview!.options!.length} options
                                        </div>
                                    </div>
                                    <div className="grid grid-cols-1 gap-3 max-h-72 overflow-auto pr-1 pt-1">
                                        {refinementPreview!.options!.map((option, idx) => {
                                            const selected =
                                                refinementDraft.title.trim() === option.refined_title.trim() &&
                                                refinementDraft.description.trim() === option.refined_description.trim();
                                            return (
                                                <button
                                                    key={`refine-option-${idx}`}
                                                    type="button"
                                                    onClick={() => setRefinementDraft({
                                                        title: option.refined_title,
                                                        description: option.refined_description,
                                                    })}
                                                    className={`text-left rounded-2xl border px-4 py-3 transition focus:outline-none ${
                                                        selected
                                                            ? 'border-emerald-300/90 bg-emerald-500/10 shadow-[0_0_0_1px_rgba(16,185,129,0.35)]'
                                                            : 'border-slate-700 bg-slate-900/70 hover:border-sky-400/45 hover:bg-slate-900'
                                                    }`}
                                                >
                                                    <div className="flex items-center justify-between gap-3 mb-2">
                                                        <div className="text-xs font-semibold text-foreground">Option {idx + 1}</div>
                                                        <div
                                                            className={`rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${
                                                                selected
                                                                    ? 'bg-emerald-300 text-emerald-950'
                                                                    : 'bg-slate-800 text-slate-300'
                                                            }`}
                                                        >
                                                            {selected ? 'Selected' : 'Preview'}
                                                        </div>
                                                    </div>
                                                    <div className="text-sm font-medium text-foreground line-clamp-1">{option.refined_title}</div>
                                                    <div className="text-xs text-muted-foreground line-clamp-3 mt-1.5 leading-relaxed">{option.refined_description}</div>
                                                </button>
                                            );
                                        })}
                                    </div>
                                </section>
                            )}
                            <div className="border-t border-white/10" />
                            <section className="rounded-2xl border border-emerald-400/35 bg-emerald-500/[0.06] p-4 shadow-[inset_0_0_0_1px_rgba(16,185,129,0.08)]">
                                <div className="flex items-start justify-between gap-3 mb-3">
                                    <div>
                                        <div className="inline-flex items-center gap-2 rounded-full border border-emerald-400/35 bg-emerald-400/10 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-emerald-200 mb-2">
                                            Step 2
                                        </div>
                                        <h4 className="text-sm font-semibold text-foreground">Selected Draft</h4>
                                        <p className="text-xs text-muted-foreground mt-1">
                                            This is the version that will be used for article generation. You can still edit it before continuing.
                                        </p>
                                    </div>
                                    <div className="shrink-0 rounded-full border border-emerald-400/35 bg-emerald-400/10 px-2.5 py-1 text-[11px] font-medium text-emerald-200">
                                        Active version
                                    </div>
                                </div>
                                <div>
                                    <label className="block text-sm font-medium mb-1">Refined Title <span className="text-muted-foreground text-xs">(must include primary keyword, ≤60 chars)</span></label>
                                    <input
                                        className="w-full px-4 py-2 rounded-xl border border-emerald-400/20 bg-background/80 focus:ring-2 focus:ring-emerald-400/40 outline-none"
                                        value={refinementDraft.title}
                                        onChange={(e) => setRefinementDraft((prev) => ({ ...prev, title: e.target.value }))}
                                    />
                                    <p className="text-xs text-muted-foreground mt-1">{refinementDraft.title.length}/60 characters</p>
                                </div>
                                <div className="mt-3">
                                    <label className="block text-sm font-medium mb-1">Refined Description <span className="text-muted-foreground text-xs">(directionally aligned with original)</span></label>
                                    <textarea
                                        className="w-full px-4 py-2 rounded-xl border border-emerald-400/20 bg-background/80 focus:ring-2 focus:ring-emerald-400/40 outline-none min-h-[130px]"
                                        value={refinementDraft.description}
                                        onChange={(e) => setRefinementDraft((prev) => ({ ...prev, description: e.target.value }))}
                                    />
                                </div>
                            </section>
                        </div>

                        <div className="flex items-center justify-end gap-2 pt-1">
                            <button
                                onClick={() => setShowRefinementGate(false)}
                                className="px-4 py-2 rounded-xl border border-border bg-background hover:bg-muted transition"
                            >
                                Cancel
                            </button>
                            <button
                                onClick={handleApproveRefinement}
                                disabled={!refinementDraft.title.trim() || refinementDraft.title.trim().length > 60 || refinementDraft.description.trim().length < 10}
                                className="px-4 py-2 rounded-xl bg-primary text-primary-foreground hover:bg-primary/90 transition disabled:opacity-50 disabled:cursor-not-allowed"
                                title={refinementDraft.title.trim().length > 60 ? 'Title exceeds 60 characters' : undefined}
                            >
                                Approve & Continue
                            </button>
                        </div>
                    </div>
                </div>
            )}

            <GenerationModal
                articleId={articleId || ''}
                taskId={taskId}
                isOpen={showProgress}
                onClose={() => {
                    setShowProgress(false);
                    setTaskId(null);
                    clearStoredGenerationSession(articleId);
                }}
                onComplete={(completedArticleId) => {
                    setShowProgress(false);
                    setTaskId(null);
                    clearStoredGenerationSession(completedArticleId || articleId || article?.id);
                    const targetId = completedArticleId || articleId || article?.id;
                    if (targetId) {
                        navigate(`/article-editor/${targetId}`);
                    }
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
