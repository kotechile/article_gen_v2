import React, { useState, useEffect } from 'react';
import { Loader2, Sparkles, Search, CheckCircle, RefreshCw, Wand2 } from 'lucide-react';
import {
    analyzeContextImage,
    generateContextImage,
    getImageProviderModels,
    getImageApplicationConfig
} from '../../services/imageService';
import type {
    ImageMetadata,
    ImageProviderModel,
    ContextAnalyzeResult,
    ContextReferenceImage
} from '../../types/image';

interface SmartContextImageGenerationProps {
    userId: string;
    selectedText?: string;
    onImageGenerated: (imageUrl: string, metadata: Partial<ImageMetadata>) => void;
}

export const SmartContextImageGeneration: React.FC<SmartContextImageGenerationProps> = ({
    userId,
    selectedText = '',
    onImageGenerated,
}) => {
    const [text, setText] = useState(selectedText);
    const [userInstructions, setUserInstructions] = useState('');
    const [models, setModels] = useState<ImageProviderModel[]>([]);
    const [selectedModel, setSelectedModel] = useState('');
    const [aspectRatio, setAspectRatio] = useState('16:9');
    const [resolution, setResolution] = useState('1K');
    const [isolateBackground, setIsolateBackground] = useState(false);

    // Pipeline states
    const [analyzing, setAnalyzing] = useState(false);
    const [generating, setGenerating] = useState(false);
    const [analysis, setAnalysis] = useState<ContextAnalyzeResult | null>(null);
    const [selectedRefUrl, setSelectedRefUrl] = useState<string>('');
    const [editablePrompt, setEditablePrompt] = useState('');
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        loadModels();
    }, []);

    useEffect(() => {
        if (selectedText && selectedText !== text) {
            setText(selectedText);
        }
    }, [selectedText]);

    const loadModels = async () => {
        try {
            const [modelsRes, appConfigRes] = await Promise.allSettled([
                getImageProviderModels(),
                getImageApplicationConfig()
            ]);

            const data = modelsRes.status === 'fulfilled' ? modelsRes.value : [];
            setModels(data);

            const appConfig = appConfigRes.status === 'fulfilled' ? appConfigRes.value : null;
            const configuredModel = appConfig?.applications?.article_image?.model_name;

            if (configuredModel && data.some(m => m.model_technical_name === configuredModel)) {
                setSelectedModel(configuredModel);
            } else if (data.length > 0) {
                setSelectedModel(data[0].model_technical_name);
            }
        } catch (err) {
            console.error('Error loading image models:', err);
        }
    };

    const handleAnalyze = async () => {
        if (!text.trim()) {
            setError('Please provide or highlight some article text to analyze.');
            return;
        }

        setAnalyzing(true);
        setError(null);

        try {
            const res = await analyzeContextImage({
                text: text.trim(),
                user_instructions: userInstructions.trim() || undefined,
                max_reference_images: 6
            });

            setAnalysis(res.data);
            setEditablePrompt(res.data.generation_prompt);

            // Default to first reference image if available
            if (res.data.candidate_references && res.data.candidate_references.length > 0) {
                setSelectedRefUrl(res.data.candidate_references[0].url);
            } else {
                setSelectedRefUrl('');
            }
        } catch (err: any) {
            console.error('Context analysis error:', err);
            setError(err.message || 'Failed to analyze text for reference imagery.');
        } finally {
            setAnalyzing(false);
        }
    };

    const handleGenerate = async (autoOneClick = false) => {
        if (!text.trim() && !editablePrompt.trim()) {
            setError('Please provide text or a prompt to generate the image.');
            return;
        }

        setGenerating(true);
        setError(null);

        try {
            const res = await generateContextImage({
                text: text.trim(),
                prompt: editablePrompt.trim() || undefined,
                reference_image_url: selectedRefUrl || undefined,
                model: selectedModel,
                aspectRatio,
                resolution,
                user_id: userId,
                isolate_background: isolateBackground,
                application: 'article_image'
            });

            onImageGenerated(res.imageUrl, res.metadata);
        } catch (err: any) {
            console.error('Context generation error:', err);
            setError(err.message || 'Failed to generate contextualized image.');
        } finally {
            setGenerating(false);
        }
    };

    const selectedModelInfo = models.find(m => m.model_technical_name === selectedModel);
    const aspectRatios = selectedModelInfo?.supported_aspect_ratios || ['16:9', '1:1', '4:3', '3:2', '9:16'];
    const rawResolutions = selectedModelInfo?.supported_resolutions || ['1K', '2K', '4K'];
    const resolutions = rawResolutions.includes('1K')
        ? ['1K', ...rawResolutions.filter(r => r !== '1K')]
        : ['1K', ...rawResolutions];

    return (
        <div className="space-y-6">
            {/* Header Description */}
            <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-950/40 dark:to-purple-950/40 p-4 rounded-xl border border-indigo-100 dark:border-indigo-900/50">
                <div className="flex items-start gap-3">
                    <Wand2 className="w-5 h-5 text-indigo-600 dark:text-indigo-400 mt-0.5 flex-shrink-0" />
                    <div>
                        <h3 className="text-sm font-semibold text-indigo-950 dark:text-indigo-200">
                            Smart Context Image Generation
                        </h3>
                        <p className="text-xs text-indigo-700 dark:text-indigo-300 mt-0.5">
                            Automatically identifies the primary physical entity (gadget, car, hardware, product) from your article, retrieves online reference photography via Linkup & Tavily, and generates a contextual scene.
                        </p>
                    </div>
                </div>
            </div>

            {error && (
                <div className="p-4 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-xl text-red-700 dark:text-red-300 text-sm">
                    {error}
                </div>
            )}

            {/* Context Text Input */}
            <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Article Section / Context Text
                </label>
                <textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Highlight or paste an article section describing a specific product, vehicle, hardware, or object..."
                    rows={3}
                    className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white placeholder-gray-400"
                />
            </div>

            {/* Optional User Creative Direction */}
            <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Creative Direction / Action (Optional)
                </label>
                <input
                    type="text"
                    value={userInstructions}
                    onChange={(e) => setUserInstructions(e.target.value)}
                    placeholder="e.g. Person using it while swimming in the ocean, or minimalist studio desk"
                    className="w-full px-4 py-2.5 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white placeholder-gray-400 text-sm"
                />
            </div>

            {/* Action Bar: Analyze vs 1-Click */}
            <div className="flex flex-wrap items-center gap-3">
                <button
                    type="button"
                    onClick={handleAnalyze}
                    disabled={analyzing || generating || !text.trim()}
                    className="flex-1 min-w-[200px] flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white font-medium shadow-sm transition-colors"
                >
                    {analyzing ? (
                        <>
                            <Loader2 className="w-4 h-4 animate-spin" />
                            <span>Analyzing & Finding Reference...</span>
                        </>
                    ) : (
                        <>
                            <Search className="w-4 h-4" />
                            <span>Find Reference Imagery & Synthesize Prompt</span>
                        </>
                    )}
                </button>

                <button
                    type="button"
                    onClick={() => handleGenerate(true)}
                    disabled={analyzing || generating || !text.trim()}
                    className="flex items-center justify-center gap-2 px-5 py-3 rounded-xl bg-purple-600 hover:bg-purple-700 disabled:opacity-50 text-white font-medium shadow-sm transition-colors"
                    title="Automatically extract entity, pick top reference image, and generate final image in one step"
                >
                    {generating ? (
                        <>
                            <Loader2 className="w-4 h-4 animate-spin" />
                            <span>Generating...</span>
                        </>
                    ) : (
                        <>
                            <Sparkles className="w-4 h-4" />
                            <span>1-Click Auto Generate</span>
                        </>
                    )}
                </button>
            </div>

            {/* Analysis & Reference Selection Card */}
            {analysis && (
                <div className="space-y-6 pt-4 border-t border-gray-200 dark:border-gray-700">
                    {/* Entity & Query Badges */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 bg-gray-50 dark:bg-gray-900/50 p-4 rounded-xl border border-gray-200 dark:border-gray-700">
                        <div>
                            <div className="flex items-center gap-2">
                                <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                                    {analysis.is_metaphorical || analysis.entity_type === 'metaphorical'
                                        ? 'Metaphorical Subject'
                                        : 'Target Entity Identified'}
                                </span>
                                {analysis.is_metaphorical || analysis.entity_type === 'metaphorical' ? (
                                    <span className="px-2 py-0.5 text-[10px] font-semibold bg-amber-100 dark:bg-amber-950/60 text-amber-800 dark:text-amber-300 rounded-full border border-amber-200 dark:border-amber-800">
                                        Metaphorical Concept
                                    </span>
                                ) : (
                                    <span className="px-2 py-0.5 text-[10px] font-semibold bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300 rounded-full border border-emerald-200 dark:border-emerald-800">
                                        Physical Object
                                    </span>
                                )}
                            </div>
                            <div className="text-sm font-bold text-indigo-600 dark:text-indigo-400 mt-0.5">
                                {analysis.main_object || 'General Subject'}
                            </div>
                        </div>
                        <div>
                            <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                                Reference Search Query
                            </span>
                            <div className="text-sm text-gray-700 dark:text-gray-300 mt-0.5 truncate" title={analysis.search_query}>
                                {analysis.search_query}
                            </div>
                        </div>
                    </div>

                    {/* Candidate Reference Images Found */}
                    <div>
                        <div className="flex items-center justify-between mb-2">
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                                Select Reference Image ({analysis.candidate_references.length} found online)
                            </label>
                            {selectedRefUrl && (
                                <button
                                    type="button"
                                    onClick={() => setSelectedRefUrl('')}
                                    className="text-xs text-red-500 hover:underline"
                                >
                                    Clear Reference Selection
                                </button>
                            )}
                        </div>

                        {analysis.candidate_references.length === 0 ? (
                            <div className="p-4 rounded-xl border border-dashed border-gray-300 dark:border-gray-700 text-center text-sm text-gray-500">
                                No online reference photos returned. The model will generate directly from the synthesized prompt.
                            </div>
                        ) : (
                            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
                                {analysis.candidate_references.map((item, idx) => {
                                    const isSelected = selectedRefUrl === item.url;
                                    return (
                                        <div
                                            key={idx}
                                            onClick={() => setSelectedRefUrl(item.url)}
                                            className={`relative group cursor-pointer rounded-xl overflow-hidden border-2 transition-all aspect-square bg-gray-100 dark:bg-gray-800 ${
                                                isSelected
                                                    ? 'border-indigo-600 ring-2 ring-indigo-500/50 scale-[1.02]'
                                                    : 'border-transparent hover:border-gray-300 dark:hover:border-gray-600'
                                            }`}
                                        >
                                            <img
                                                src={item.thumbnail_url || item.url}
                                                alt={item.title || 'Reference candidate'}
                                                className="w-full h-full object-cover"
                                                loading="lazy"
                                                onError={(e) => {
                                                    // Fallback for broken web links
                                                    (e.target as HTMLElement).style.display = 'none';
                                                }}
                                            />
                                            {isSelected && (
                                                <div className="absolute top-1.5 right-1.5 bg-indigo-600 text-white rounded-full p-0.5 shadow">
                                                    <CheckCircle className="w-4 h-4" />
                                                </div>
                                            )}
                                            <div className="absolute inset-x-0 bottom-0 bg-black/60 p-1 text-[10px] text-white truncate text-center opacity-0 group-hover:opacity-100 transition-opacity">
                                                {item.source_domain || 'Web'}
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                    </div>

                    {/* Synthesized Diffusion Prompt */}
                    <div>
                        <div className="flex items-center justify-between mb-1">
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                                Synthesized Scene Prompt
                            </label>
                            <span className="text-xs text-gray-400">Editable</span>
                        </div>
                        <textarea
                            value={editablePrompt}
                            onChange={(e) => setEditablePrompt(e.target.value)}
                            rows={3}
                            className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white text-sm"
                        />
                    </div>

                    {/* Model, Aspect Ratio, Resolution Controls */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                                AI Model
                            </label>
                            <select
                                value={selectedModel}
                                onChange={(e) => setSelectedModel(e.target.value)}
                                className="w-full px-3 py-2.5 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white"
                            >
                                {models.map((m) => (
                                    <option key={m.id} value={m.model_technical_name}>
                                        {m.model_name}
                                    </option>
                                ))}
                            </select>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                                Aspect Ratio
                            </label>
                            <select
                                value={aspectRatio}
                                onChange={(e) => setAspectRatio(e.target.value)}
                                className="w-full px-3 py-2.5 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white"
                            >
                                {aspectRatios.map((ratio) => (
                                    <option key={ratio} value={ratio}>
                                        {ratio}
                                    </option>
                                ))}
                            </select>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                                Resolution
                            </label>
                            <select
                                value={resolution}
                                onChange={(e) => setResolution(e.target.value)}
                                className="w-full px-3 py-2.5 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white"
                            >
                                {resolutions.map((res) => (
                                    <option key={res} value={res}>
                                        {res === '1K' ? '1K (Default)' : res}
                                    </option>
                                ))}
                            </select>
                        </div>
                    </div>

                    {/* Background Isolation Toggle */}
                    {selectedRefUrl && (
                        <div className="flex items-center gap-2 pt-1">
                            <input
                                id="isolate-bg-toggle"
                                type="checkbox"
                                checked={isolateBackground}
                                onChange={(e) => setIsolateBackground(e.target.checked)}
                                className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded cursor-pointer"
                            />
                            <label htmlFor="isolate-bg-toggle" className="text-xs text-gray-700 dark:text-gray-300 cursor-pointer select-none">
                                Isolate subject / remove background from reference photo before generation
                            </label>
                        </div>
                    )}

                    {/* Generate Button */}
                    <div className="pt-2">
                        <button
                            type="button"
                            onClick={() => handleGenerate(false)}
                            disabled={generating || !editablePrompt.trim()}
                            className="w-full flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white font-medium shadow-md transition-colors"
                        >
                            {generating ? (
                                <>
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                    <span>Generating Contextualized Scene...</span>
                                </>
                            ) : (
                                <>
                                    <Sparkles className="w-5 h-5" />
                                    <span>
                                        Generate Scene {selectedRefUrl ? 'Conditioned on Reference' : 'from Prompt'}
                                    </span>
                                </>
                            )}
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};
