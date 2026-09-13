import React, { useState, useEffect } from 'react';
import { Loader2, Sparkles, Search, CheckCircle, Wand2, Upload, X, Image as ImageIcon, Trash2 } from 'lucide-react';
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

    // Custom uploaded reference state
    const [customRefFile, setCustomRefFile] = useState<File | null>(null);
    const [customRefPreview, setCustomRefPreview] = useState<string | null>(null);

    // Pipeline states
    const [analyzing, setAnalyzing] = useState(false);
    const [generating, setGenerating] = useState(false);
    const [analysis, setAnalysis] = useState<ContextAnalyzeResult | null>(null);
    const [selectedRefUrl, setSelectedRefUrl] = useState<string>('');
    const [selectedRefType, setSelectedRefType] = useState<'online' | 'upload' | 'none'>('none');
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

    const handleCustomRefChange = (file: File | null) => {
        if (customRefPreview) {
            URL.revokeObjectURL(customRefPreview);
        }
        setCustomRefFile(file);
        if (file) {
            const previewUrl = URL.createObjectURL(file);
            setCustomRefPreview(previewUrl);
            setSelectedRefType('upload');
            setSelectedRefUrl('');
        } else {
            setCustomRefPreview(null);
            if (analysis?.candidate_references && analysis.candidate_references.length > 0) {
                setSelectedRefType('online');
                setSelectedRefUrl(analysis.candidate_references[0].url);
            } else {
                setSelectedRefType('none');
                setSelectedRefUrl('');
            }
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

            // If user already uploaded a custom reference, keep it selected
            if (customRefFile) {
                setSelectedRefType('upload');
                setSelectedRefUrl('');
            } else if (res.data.candidate_references && res.data.candidate_references.length > 0) {
                // Otherwise default to first online reference image
                setSelectedRefType('online');
                setSelectedRefUrl(res.data.candidate_references[0].url);
            } else {
                setSelectedRefType('none');
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
            let refBase64: string | undefined = undefined;
            let refUrl: string | undefined = undefined;

            if (selectedRefType === 'upload' && customRefFile) {
                const reader = new FileReader();
                refBase64 = await new Promise<string>((resolve, reject) => {
                    reader.onload = () => {
                        const res = reader.result as string;
                        const cleanB64 = res.includes(',') ? res.split(',')[1] : res;
                        resolve(cleanB64);
                    };
                    reader.onerror = reject;
                    reader.readAsDataURL(customRefFile);
                });
            } else if (selectedRefType === 'online' && selectedRefUrl) {
                refUrl = selectedRefUrl;
            }

            const res = await generateContextImage({
                text: text.trim(),
                prompt: editablePrompt.trim() || undefined,
                reference_image_url: refUrl,
                reference_image_base64: refBase64,
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

    const hasActiveReference = selectedRefType === 'upload' ? Boolean(customRefPreview) : Boolean(selectedRefUrl);

    return (
        <div className="space-y-6">
            {/* Header Description */}
            <div className="bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-950/40 dark:to-purple-950/40 p-4 rounded-xl border border-indigo-100 dark:border-indigo-900/50">
                <div className="flex items-start gap-3">
                    <Wand2 className="w-5 h-5 text-indigo-600 dark:text-indigo-400 mt-0.5 flex-shrink-0" />
                    <div>
                        <h3 className="text-sm font-semibold text-indigo-950 dark:text-indigo-200">
                            Smart Context AI Image Generation
                        </h3>
                        <p className="text-xs text-indigo-700 dark:text-indigo-300 mt-0.5">
                            Identifies the primary physical entity (gadget, car, product) from your article, retrieves online reference photography or uses your own uploaded photo, and generates a contextual scene conditioned on the reference.
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

            {/* Optional Pre-analysis Custom Reference Upload */}
            {!analysis && (
                <div>
                    <label className="block text-xs font-semibold text-gray-600 dark:text-gray-400 uppercase tracking-wider mb-2">
                        Upload Reference Photo (Optional)
                    </label>
                    {customRefPreview ? (
                        <div className="flex items-center justify-between p-3.5 bg-indigo-50/70 dark:bg-indigo-950/30 border-2 border-indigo-500 rounded-xl">
                            <div className="flex items-center gap-3 min-w-0">
                                <img
                                    src={customRefPreview}
                                    alt="Uploaded reference"
                                    className="w-14 h-14 rounded-lg object-cover border border-indigo-200 dark:border-indigo-800 flex-shrink-0"
                                />
                                <div className="min-w-0">
                                    <div className="flex items-center gap-2">
                                        <span className="text-xs font-semibold text-gray-900 dark:text-white truncate">
                                            {customRefFile?.name}
                                        </span>
                                        <span className="px-2 py-0.5 text-[10px] font-semibold bg-indigo-600 text-white rounded-full">
                                            Custom Reference Attached
                                        </span>
                                    </div>
                                    <p className="text-[11px] text-gray-500 dark:text-gray-400 mt-0.5">
                                        {((customRefFile?.size || 0) / 1024).toFixed(1)} KB • Will be used as direct visual reference
                                    </p>
                                </div>
                            </div>
                            <button
                                type="button"
                                onClick={() => handleCustomRefChange(null)}
                                className="p-2 text-gray-400 hover:text-red-500 rounded-lg hover:bg-red-50 dark:hover:bg-red-950/30 transition-colors ml-2 flex-shrink-0"
                                title="Remove uploaded reference"
                            >
                                <X className="w-4 h-4" />
                            </button>
                        </div>
                    ) : (
                        <label className="flex items-center justify-center gap-3 p-4 border-2 border-dashed border-gray-300 dark:border-gray-700 hover:border-indigo-400 dark:hover:border-indigo-500 rounded-xl cursor-pointer bg-gray-50/50 dark:bg-gray-900/30 hover:bg-indigo-50/20 transition-colors group">
                            <div className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-500 group-hover:text-indigo-600 group-hover:bg-indigo-100 dark:group-hover:bg-indigo-950/60 transition-colors">
                                <Upload className="w-4 h-4" />
                            </div>
                            <div className="text-left">
                                <p className="text-xs font-medium text-gray-700 dark:text-gray-300">
                                    <span className="text-indigo-600 dark:text-indigo-400 underline">Upload your own photo</span> to use as the visual reference
                                </p>
                                <p className="text-[11px] text-gray-400 dark:text-gray-500">
                                    Supports PNG, JPG, or WEBP
                                </p>
                            </div>
                            <input
                                type="file"
                                accept="image/*"
                                onChange={(e) => handleCustomRefChange(e.target.files?.[0] || null)}
                                className="hidden"
                            />
                        </label>
                    )}
                </div>
            )}

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
                                    {analysis.has_physical_entity
                                        ? 'Target Physical Entity'
                                        : 'Metaphorical Subject'}
                                </span>
                                {analysis.has_physical_entity ? (
                                    <span className="px-2 py-0.5 text-[10px] font-semibold bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300 rounded-full border border-emerald-200 dark:border-emerald-800">
                                        Physical Object
                                    </span>
                                ) : (
                                    <span className="px-2 py-0.5 text-[10px] font-semibold bg-purple-100 dark:bg-purple-950/60 text-purple-800 dark:text-purple-300 rounded-full border border-purple-200 dark:border-purple-800">
                                        Direct Diffusion
                                    </span>
                                )}
                            </div>
                            <div className="text-sm font-bold text-indigo-600 dark:text-indigo-400 mt-0.5">
                                {analysis.main_object || 'General Subject'}
                            </div>
                        </div>
                        <div>
                            <span className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                                {analysis.has_physical_entity ? 'Reference Search Query' : 'Generation Mode'}
                            </span>
                            <div className="text-sm text-gray-700 dark:text-gray-300 mt-0.5 truncate" title={analysis.search_query || 'Direct text-to-image prompt'}>
                                {analysis.has_physical_entity
                                    ? (analysis.search_query || 'N/A')
                                    : 'Direct text-to-image (or use uploaded reference photo)'}
                            </div>
                        </div>
                    </div>

                    {/* Reference Selection Section */}
                    <div>
                        <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                                    Reference Image Selection
                                </label>
                                {selectedRefType === 'upload' && (
                                    <span className="px-2 py-0.5 text-[10px] font-semibold bg-indigo-100 dark:bg-indigo-950/60 text-indigo-700 dark:text-indigo-300 rounded-full border border-indigo-200 dark:border-indigo-800">
                                        Using Uploaded Photo
                                    </span>
                                )}
                                {selectedRefType === 'online' && (
                                    <span className="px-2 py-0.5 text-[10px] font-semibold bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300 rounded-full border border-emerald-200 dark:border-emerald-800">
                                        Using Online Candidate
                                    </span>
                                )}
                                {selectedRefType === 'none' && (
                                    <span className="px-2 py-0.5 text-[10px] font-medium bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 rounded-full">
                                        No Reference (Direct Prompt)
                                    </span>
                                )}
                            </div>
                            {hasActiveReference && (
                                <button
                                    type="button"
                                    onClick={() => {
                                        setSelectedRefType('none');
                                        setSelectedRefUrl('');
                                    }}
                                    className="text-xs text-red-500 hover:underline"
                                >
                                    Clear Reference Selection
                                </button>
                            )}
                        </div>

                        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
                            {/* Option 1: Custom Upload Card */}
                            {customRefPreview ? (
                                <div
                                    onClick={() => {
                                        setSelectedRefType('upload');
                                        setSelectedRefUrl('');
                                    }}
                                    className={`relative group cursor-pointer rounded-xl overflow-hidden border-2 transition-all aspect-square bg-gray-100 dark:bg-gray-800 ${
                                        selectedRefType === 'upload'
                                            ? 'border-indigo-600 ring-2 ring-indigo-500/50 scale-[1.02]'
                                            : 'border-gray-200 dark:border-gray-700 hover:border-indigo-300'
                                    }`}
                                    title="Click to use your uploaded photo as reference"
                                >
                                    <img
                                        src={customRefPreview}
                                        alt="Custom reference"
                                        className="w-full h-full object-cover"
                                    />
                                    {selectedRefType === 'upload' && (
                                        <div className="absolute top-1.5 right-1.5 p-1 bg-indigo-600 text-white rounded-full shadow">
                                            <CheckCircle className="w-3 h-3" />
                                        </div>
                                    )}
                                    <button
                                        type="button"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            handleCustomRefChange(null);
                                        }}
                                        className="absolute top-1.5 left-1.5 p-1 bg-black/70 hover:bg-red-600 text-white rounded-md opacity-0 group-hover:opacity-100 transition-opacity"
                                        title="Delete uploaded photo"
                                    >
                                        <Trash2 className="w-3 h-3" />
                                    </button>
                                    <div className="absolute inset-x-0 bottom-0 p-1 bg-indigo-900/90 text-[10px] text-white font-medium text-center truncate">
                                        Custom Upload
                                    </div>
                                </div>
                            ) : (
                                <label className="flex flex-col items-center justify-center p-2 rounded-xl border-2 border-dashed border-indigo-300 dark:border-indigo-700 hover:border-indigo-500 hover:bg-indigo-50/30 dark:hover:bg-indigo-950/30 cursor-pointer transition-all aspect-square text-center group">
                                    <div className="p-2 rounded-full bg-indigo-50 dark:bg-indigo-900/40 text-indigo-600 dark:text-indigo-400 group-hover:scale-110 transition-transform mb-1">
                                        <Upload className="w-4 h-4" />
                                    </div>
                                    <span className="text-[11px] font-semibold text-indigo-700 dark:text-indigo-300">
                                        Upload Mine
                                    </span>
                                    <span className="text-[9px] text-gray-400 dark:text-gray-500">
                                        PNG, JPG, WEBP
                                    </span>
                                    <input
                                        type="file"
                                        accept="image/*"
                                        onChange={(e) => handleCustomRefChange(e.target.files?.[0] || null)}
                                        className="hidden"
                                    />
                                </label>
                            )}

                            {/* Option 2: Online Candidates Grid */}
                            {analysis.candidate_references.map((item, idx) => {
                                const isSelected = selectedRefType === 'online' && selectedRefUrl === item.url;
                                return (
                                    <div
                                        key={idx}
                                        onClick={() => {
                                            setSelectedRefType('online');
                                            setSelectedRefUrl(item.url);
                                        }}
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
                                                (e.target as HTMLElement).style.display = 'none';
                                            }}
                                        />
                                        {isSelected && (
                                            <div className="absolute top-1.5 right-1.5 p-1 bg-indigo-600 text-white rounded-full shadow">
                                                <CheckCircle className="w-3 h-3" />
                                            </div>
                                        )}
                                        <div className="absolute inset-x-0 bottom-0 p-1.5 bg-gradient-to-t from-black/80 via-black/40 to-transparent text-[10px] text-white truncate opacity-0 group-hover:opacity-100 transition-opacity">
                                            {item.source_domain || item.provider}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>

                        {analysis.candidate_references.length === 0 && !customRefPreview && (
                            <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                                No online reference photos found. You can upload your own photo above or generate directly from the prompt.
                            </p>
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
                    {hasActiveReference && (
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
                                        {selectedRefType === 'upload'
                                            ? 'Generate Scene Conditioned on Uploaded Photo'
                                            : selectedRefType === 'online'
                                            ? 'Generate Scene Conditioned on Reference'
                                            : 'Generate Scene from Prompt'}
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
