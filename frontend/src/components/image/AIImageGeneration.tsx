import React, { useState, useEffect } from 'react';
import { Loader2, Sparkles, RefreshCw, Upload, X, Wand2, Palette, Check } from 'lucide-react';
import {
    generateAIImage,
    getImageProviderModels,
    getImageApplicationConfig,
    synthesizeImagePrompt
} from '../../services/imageService';
import type { ImageProviderModel, ImageMetadata } from '../../types/image';

export interface StylePreset {
    id: string;
    name: string;
    description: string;
    promptModifier: string;
}

export const STYLE_PRESETS: StylePreset[] = [
    {
        id: 'cinematic',
        name: 'Cinematic',
        description: 'Dramatic lighting, film still composition',
        promptModifier: 'cinematic lighting, dramatic atmosphere, depth of field, 35mm film still, 8k resolution, photorealistic masterpiece',
    },
    {
        id: 'whiteboard',
        name: 'Whiteboard',
        description: 'Clean markers, sketch notes, clear outlines',
        promptModifier: 'clean whiteboard drawing, markers, clear outlines, hand-drawn sketch illustration, whiteboard notes on crisp white background',
    },
    {
        id: 'studio',
        name: 'Studio Lighting',
        description: 'High-end commercial photo, softbox, sharp focus',
        promptModifier: 'professional studio lighting, clean softbox reflections, commercial product photography, razor sharp focus, pristine background, 8k',
    },
    {
        id: 'watercolor',
        name: 'Watercolor',
        description: 'Artistic brush strokes, soft fluid wash',
        promptModifier: 'vibrant watercolor painting, expressive brushstrokes, soft fluid wash, textured watercolor paper, artistic illustration',
    },
    {
        id: 'oil_painting',
        name: 'Oil Painting',
        description: 'Rich impasto textures, classic fine art',
        promptModifier: 'classical oil painting, visible canvas texture, rich impasto brushwork, fine art masterpiece, museum quality',
    },
    {
        id: 'line_art',
        name: 'Line Art',
        description: 'Crisp vector line art, clean outlines',
        promptModifier: 'crisp vector line art, clean black and white outlines, minimalist illustration, modern elegant contour drawing',
    },
    {
        id: 'macro',
        name: 'Macro Photography',
        description: 'Ultra close-up details, smooth bokeh',
        promptModifier: 'extreme macro photography, ultra close-up fine texture details, shallow depth of field, delicate lighting, smooth bokeh',
    },
    {
        id: 'cyberpunk',
        name: 'Cyberpunk',
        description: 'Neon lighting, futuristic high-tech glow',
        promptModifier: 'cyberpunk style, vibrant neon glow, futuristic night atmosphere, teal and magenta palette, sci-fi aesthetic',
    },
    {
        id: 'flat_design',
        name: 'Flat Design',
        description: 'Bold geometric shapes, modern 2D vector',
        promptModifier: 'modern flat design illustration, bold geometric shapes, clean vector graphics, vibrant harmonious color palette, 2D minimalist vector',
    },
    {
        id: '3d_render',
        name: '3D Render',
        description: 'Isometric 3D, octane render, smooth materials',
        promptModifier: '3D isometric render, octane render, smooth stylized 3D materials, soft volumetric lighting, vibrant modern 3D artwork',
    },
    {
        id: 'minimalist',
        name: 'Minimalist',
        description: 'Generous negative space, clean elegance',
        promptModifier: 'minimalist aesthetic, spacious negative space, clean elegant composition, subdued sophisticated colors, sleek design',
    },
    {
        id: 'vintage',
        name: 'Vintage Photography',
        description: 'Analog film grain, 70s Kodachrome tones',
        promptModifier: 'vintage analog 35mm film photography, authentic film grain, 1970s Kodachrome color grading, subtle retro light leak',
    },
];

interface AIImageGenerationProps {
    userId: string;
    selectedText?: string;
    onImageGenerated: (imageUrl: string, metadata: Partial<ImageMetadata>) => void;
}

export const AIImageGeneration: React.FC<AIImageGenerationProps> = ({
    userId,
    selectedText = '',
    onImageGenerated
}) => {
    const [contextText, setContextText] = useState(selectedText);
    const [selectedStyleId, setSelectedStyleId] = useState<string>('cinematic');
    const [prompt, setPrompt] = useState('');
    const [synthesizingPrompt, setSynthesizingPrompt] = useState(false);

    const [models, setModels] = useState<ImageProviderModel[]>([]);
    const [selectedModel, setSelectedModel] = useState('');
    const [aspectRatio, setAspectRatio] = useState('16:9');
    const [resolution, setResolution] = useState('1K');
    const [referenceImage, setReferenceImage] = useState<File | null>(null);
    const [referenceImagePreview, setReferenceImagePreview] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [loadingModels, setLoadingModels] = useState(true);
    const [generatedImage, setGeneratedImage] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        loadModels();
    }, []);

    useEffect(() => {
        if (selectedText && selectedText !== contextText) {
            setContextText(selectedText);
        }
    }, [selectedText]);

    // When component mounts with selectedText, synthesize an initial prompt if prompt is empty
    useEffect(() => {
        if (selectedText && !prompt) {
            handleSynthesizePrompt(selectedText, selectedStyleId);
        }
    }, []);

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
            setError('Failed to load AI models');
            console.error(err);
        } finally {
            setLoadingModels(false);
        }
    };

    const handleSynthesizePrompt = async (textToUse?: string, styleIdToUse?: string) => {
        const text = (textToUse !== undefined ? textToUse : contextText).trim();
        const styleId = styleIdToUse !== undefined ? styleIdToUse : selectedStyleId;
        const style = STYLE_PRESETS.find(s => s.id === styleId);

        if (!text) {
            if (style) {
                setPrompt(style.promptModifier);
            }
            return;
        }

        setSynthesizingPrompt(true);
        setError(null);

        try {
            const res = await synthesizeImagePrompt({
                text,
                style: style?.name,
                style_prompt_modifier: style?.promptModifier,
            });

            if (res.prompt) {
                setPrompt(res.prompt);
            } else {
                // Fallback prompt generation
                const fallback = style
                    ? `${text.slice(0, 150)}, ${style.promptModifier}`
                    : text;
                setPrompt(fallback);
            }
        } catch (err) {
            console.warn('Synthesize prompt failed; using direct style formulation:', err);
            const fallback = style
                ? `${text.slice(0, 150)}, ${style.promptModifier}`
                : text;
            setPrompt(fallback);
        } finally {
            setSynthesizingPrompt(false);
        }
    };

    const handleSelectStyle = (styleId: string) => {
        const newStyleId = selectedStyleId === styleId ? '' : styleId;
        setSelectedStyleId(newStyleId);

        // If we have contextText, update/synthesize the prompt
        if (contextText.trim()) {
            handleSynthesizePrompt(contextText, newStyleId);
        } else if (newStyleId) {
            const style = STYLE_PRESETS.find(s => s.id === newStyleId);
            if (style) {
                setPrompt(style.promptModifier);
            }
        }
    };

    const handleReferenceImageChange = (file: File | null) => {
        setReferenceImage(file);
        if (file) {
            const previewUrl = URL.createObjectURL(file);
            setReferenceImagePreview(previewUrl);
        } else {
            setReferenceImagePreview(null);
        }
    };

    const handleGenerate = async () => {
        let activePrompt = prompt.trim();

        // If prompt is empty but context exists, synthesize on the fly
        if (!activePrompt && contextText.trim()) {
            const style = STYLE_PRESETS.find(s => s.id === selectedStyleId);
            activePrompt = style
                ? `${contextText.trim().slice(0, 150)}, ${style.promptModifier}`
                : contextText.trim();
            setPrompt(activePrompt);
        }

        if (!activePrompt) {
            setError('Please enter a description or select text from your article to generate an image.');
            return;
        }

        setLoading(true);
        setError(null);
        setGeneratedImage(null);

        try {
            let referenceImageBase64: string | undefined;
            if (referenceImage) {
                const reader = new FileReader();
                referenceImageBase64 = await new Promise((resolve) => {
                    reader.onload = () => resolve(reader.result as string);
                    reader.readAsDataURL(referenceImage);
                }).then((result) => (result as string).split(',')[1]);
            }

            const response = await generateAIImage({
                prompt: activePrompt,
                model: selectedModel,
                application: 'article_image',
                aspectRatio,
                resolution,
                referenceImage: referenceImageBase64,
                user_id: userId
            });

            setGeneratedImage(response.imageUrl);
        } catch (err: any) {
            setError(err.message || 'Failed to generate image');
        } finally {
            setLoading(false);
        }
    };

    const handleAccept = () => {
        if (generatedImage) {
            const modelInfo = models.find(m => m.model_technical_name === selectedModel);
            const styleInfo = STYLE_PRESETS.find(s => s.id === selectedStyleId);
            const styleTag = styleInfo ? ` [Style: ${styleInfo.name}]` : '';

            onImageGenerated(generatedImage, {
                ImageUrl: generatedImage,
                ImageAuthor: `AI - ${modelInfo?.model_name || selectedModel}${styleTag}`,
                MediaAltText: prompt.substring(0, 200),
                mediaTitle: prompt.substring(0, 100)
            });
        }
    };

    const selectedModelInfo = models.find(m => m.model_technical_name === selectedModel);
    const aspectRatios = selectedModelInfo?.supported_aspect_ratios || ['16:9', '1:1', '4:3', '3:2', '9:16'];
    const rawResolutions = selectedModelInfo?.supported_resolutions || ['1K', '2K', '4K'];
    const resolutions = rawResolutions.includes('1K')
        ? ['1K', ...rawResolutions.filter(r => r !== '1K')]
        : ['1K', ...rawResolutions];

    if (loadingModels) {
        return (
            <div className="flex items-center justify-center py-12">
                <Loader2 className="w-8 h-8 animate-spin text-indigo-600" />
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Header / Intro Banner */}
            <div className="bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-950/40 dark:to-indigo-950/40 p-4 rounded-xl border border-purple-100 dark:border-purple-900/50">
                <div className="flex items-start gap-3">
                    <Sparkles className="w-5 h-5 text-purple-600 dark:text-purple-400 mt-0.5 flex-shrink-0" />
                    <div>
                        <h3 className="text-sm font-semibold text-purple-950 dark:text-purple-200">
                            AI Image Generation
                        </h3>
                        <p className="text-xs text-purple-700 dark:text-purple-300 mt-0.5">
                            Select or paste text from your article, pick a visual style preset, synthesize an image prompt, and generate high-fidelity AI imagery.
                        </p>
                    </div>
                </div>
            </div>

            {/* Step 1: Context / Article Selection */}
            <div>
                <div className="flex items-center justify-between mb-1.5">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                        Article Text / Section
                    </label>
                    {contextText && (
                        <button
                            type="button"
                            onClick={() => handleSynthesizePrompt(contextText, selectedStyleId)}
                            disabled={synthesizingPrompt}
                            className="text-xs font-semibold text-indigo-600 dark:text-indigo-400 hover:text-indigo-800 dark:hover:text-indigo-300 flex items-center gap-1.5 transition-colors disabled:opacity-50"
                        >
                            {synthesizingPrompt ? (
                                <>
                                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                    Synthesizing Prompt...
                                </>
                            ) : (
                                <>
                                    <Wand2 className="w-3.5 h-3.5" />
                                    Synthesize Prompt from Text
                                </>
                            )}
                        </button>
                    )}
                </div>
                <textarea
                    value={contextText}
                    onChange={(e) => setContextText(e.target.value)}
                    rows={3}
                    className="w-full px-4 py-2.5 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white text-sm placeholder-gray-400"
                    placeholder="Selected article text will appear here, or paste any paragraph/idea..."
                />
                {selectedText && (
                    <p className="mt-1 text-xs text-indigo-600 dark:text-indigo-400 flex items-center gap-1">
                        <Sparkles className="w-3.5 h-3.5" />
                        Loaded from article selection
                    </p>
                )}
            </div>

            {/* Step 2: Visual Style Presets */}
            <div>
                <div className="flex items-center justify-between mb-2">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 flex items-center gap-1.5">
                        <Palette className="w-4 h-4 text-indigo-500" />
                        Visual Style Preset
                    </label>
                    {selectedStyleId && (
                        <button
                            type="button"
                            onClick={() => setSelectedStyleId('')}
                            className="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                        >
                            Clear style
                        </button>
                    )}
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2.5">
                    {STYLE_PRESETS.map((style) => {
                        const isSelected = selectedStyleId === style.id;
                        return (
                            <button
                                key={style.id}
                                type="button"
                                onClick={() => handleSelectStyle(style.id)}
                                className={`text-left p-2.5 rounded-xl border transition-all flex flex-col justify-between ${isSelected
                                    ? 'border-indigo-600 bg-indigo-50/80 dark:bg-indigo-950/60 ring-2 ring-indigo-500/20 shadow-sm'
                                    : 'border-gray-200 dark:border-gray-700/80 bg-white dark:bg-gray-800/60 hover:border-gray-300 dark:hover:border-gray-600'
                                    }`}
                            >
                                <div className="flex items-center justify-between w-full mb-1">
                                    <span className={`text-xs font-semibold ${isSelected ? 'text-indigo-950 dark:text-indigo-200' : 'text-gray-900 dark:text-white'}`}>
                                        {style.name}
                                    </span>
                                    {isSelected && (
                                        <div className="w-4 h-4 rounded-full bg-indigo-600 text-white flex items-center justify-center flex-shrink-0">
                                            <Check className="w-2.5 h-2.5" />
                                        </div>
                                    )}
                                </div>
                                <p className="text-[10px] text-gray-500 dark:text-gray-400 line-clamp-2 leading-tight">
                                    {style.description}
                                </p>
                            </button>
                        );
                    })}
                </div>
            </div>

            {/* Step 3: Editable Synthesized Prompt */}
            <div>
                <div className="flex items-center justify-between mb-1.5">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                        Generated Image Prompt (Editable)
                    </label>
                    {prompt && (
                        <button
                            type="button"
                            onClick={() => setPrompt('')}
                            className="text-xs text-gray-400 hover:text-red-500"
                        >
                            Clear
                        </button>
                    )}
                </div>
                <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    rows={3}
                    className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white text-sm"
                    placeholder="Synthesized or custom prompt for image generation..."
                />
            </div>

            {/* Step 4: Model, Aspect Ratio, Resolution */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
                        AI Model
                    </label>
                    <select
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                        className="w-full px-3.5 py-2.5 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white text-sm"
                    >
                        {models.map((model) => (
                            <option key={model.id} value={model.model_technical_name}>
                                {model.model_name}
                            </option>
                        ))}
                    </select>
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
                        Aspect Ratio
                    </label>
                    <select
                        value={aspectRatio}
                        onChange={(e) => setAspectRatio(e.target.value)}
                        className="w-full px-3.5 py-2.5 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white text-sm"
                    >
                        {aspectRatios.map((ratio) => (
                            <option key={ratio} value={ratio}>
                                {ratio}
                            </option>
                        ))}
                    </select>
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1.5">
                        Resolution
                    </label>
                    <select
                        value={resolution}
                        onChange={(e) => setResolution(e.target.value)}
                        className="w-full px-3.5 py-2.5 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white text-sm"
                    >
                        {resolutions.map((res) => (
                            <option key={res} value={res}>
                                {res === '1K' ? '1K (Default)' : res}
                            </option>
                        ))}
                    </select>
                </div>
            </div>

            {/* Step 5: Optional Reference Image */}
            <div>
                <div className="flex items-center justify-between mb-1.5">
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                        Reference Image (Optional)
                    </label>
                    {referenceImage && (
                        <button
                            type="button"
                            onClick={() => handleReferenceImageChange(null)}
                            className="text-xs font-medium text-red-500 hover:text-red-700 dark:text-red-400 flex items-center gap-1"
                        >
                            <X className="w-3.5 h-3.5" /> Remove reference
                        </button>
                    )}
                </div>
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">
                    Upload an image to guide style or object transfer, or leave empty to generate purely from text and style preset.
                </p>

                {referenceImage && referenceImagePreview ? (
                    <div className="flex items-center gap-4 p-3 bg-gray-50 dark:bg-gray-900/60 rounded-xl border border-indigo-200 dark:border-indigo-800/60">
                        <div className="relative w-14 h-14 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 flex-shrink-0 bg-black/5">
                            <img
                                src={referenceImagePreview}
                                alt="Reference preview"
                                className="w-full h-full object-cover"
                            />
                        </div>
                        <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                                <span className="text-xs font-semibold text-gray-900 dark:text-white truncate">
                                    {referenceImage.name}
                                </span>
                                <span className="px-2 py-0.5 text-[10px] font-medium bg-indigo-100 dark:bg-indigo-950/60 text-indigo-700 dark:text-indigo-300 rounded-full border border-indigo-200 dark:border-indigo-800">
                                    Reference Attached
                                </span>
                            </div>
                            <p className="text-[11px] text-gray-500 dark:text-gray-400 mt-0.5">
                                {(referenceImage.size / 1024).toFixed(1)} KB • Guiding {selectedModelInfo?.model_name || 'AI model'}
                            </p>
                        </div>
                        <button
                            type="button"
                            onClick={() => handleReferenceImageChange(null)}
                            className="p-1.5 rounded-lg text-gray-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-950/30 transition-colors"
                            title="Remove reference image"
                        >
                            <X className="w-4 h-4" />
                        </button>
                    </div>
                ) : (
                    <label className="flex flex-col items-center justify-center p-4 border-2 border-dashed border-gray-300 dark:border-gray-700 hover:border-indigo-400 dark:hover:border-indigo-500 rounded-xl cursor-pointer bg-gray-50/50 dark:bg-gray-900/30 hover:bg-indigo-50/20 transition-colors group">
                        <div className="flex items-center justify-center gap-2 text-center">
                            <div className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 group-hover:bg-indigo-100 dark:group-hover:bg-indigo-950/60 text-gray-500 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
                                <Upload className="w-4 h-4" />
                            </div>
                            <div className="text-left">
                                <p className="text-xs font-medium text-gray-700 dark:text-gray-300">
                                    <span className="text-indigo-600 dark:text-indigo-400 underline">Upload reference image</span> (Optional)
                                </p>
                                <p className="text-[10px] text-gray-400 dark:text-gray-500">
                                    Supports PNG, JPG, or WEBP
                                </p>
                            </div>
                        </div>
                        <input
                            type="file"
                            accept="image/*"
                            onChange={(e) => handleReferenceImageChange(e.target.files?.[0] || null)}
                            className="hidden"
                        />
                    </label>
                )}
            </div>

            {/* Error Message */}
            {error && (
                <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-4">
                    <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
                </div>
            )}

            {/* Generated Image Preview */}
            {generatedImage && (
                <div className="space-y-4">
                    <div className="rounded-xl overflow-hidden border border-gray-200 dark:border-gray-700 shadow-sm bg-black/5">
                        <img
                            src={generatedImage}
                            alt="Generated preview"
                            className="w-full h-auto max-h-[480px] object-contain mx-auto"
                        />
                    </div>
                    <div className="flex gap-3">
                        <button
                            onClick={handleGenerate}
                            className="flex items-center gap-2 px-6 py-3 rounded-xl border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                        >
                            <RefreshCw className="w-4 h-4" />
                            Regenerate
                        </button>
                        <button
                            onClick={handleAccept}
                            className="flex-1 bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-3 rounded-xl font-medium transition-colors shadow-sm"
                        >
                            Accept & Insert Image
                        </button>
                    </div>
                </div>
            )}

            {/* Generate Button */}
            {!generatedImage && (
                <button
                    onClick={handleGenerate}
                    disabled={loading || (!prompt.trim() && !contextText.trim())}
                    className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white px-6 py-3 rounded-xl font-medium transition-colors shadow-sm"
                >
                    {loading ? (
                        <>
                            <Loader2 className="w-5 h-5 animate-spin" />
                            Generating Image...
                        </>
                    ) : (
                        <>
                            <Sparkles className="w-5 h-5" />
                            Generate Image
                        </>
                    )}
                </button>
            )}
        </div>
    );
};
