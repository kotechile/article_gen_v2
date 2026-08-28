import React, { useState, useEffect } from 'react';
import { Loader2, Sparkles, RefreshCw, Upload, X, Image as ImageIcon } from 'lucide-react';
import { generateAIImage, getImageProviderModels, getImageApplicationConfig } from '../../services/imageService';
import type { ImageProviderModel, ImageMetadata } from '../../types/image';

interface AIImageGenerationProps {
    userId: string;
    selectedText?: string;
    onImageGenerated: (imageUrl: string, metadata: Partial<ImageMetadata>) => void;
}

export const AIImageGeneration: React.FC<AIImageGenerationProps> = ({
    userId,
    selectedText,
    onImageGenerated
}) => {
    const [prompt, setPrompt] = useState(selectedText || '');
    const [models, setModels] = useState<ImageProviderModel[]>([]);
    const [selectedModel, setSelectedModel] = useState('');
    const [aspectRatio, setAspectRatio] = useState('1:1');
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
        if (!prompt.trim()) {
            setError('Please enter a prompt');
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
                prompt,
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
            onImageGenerated(generatedImage, {
                ImageUrl: generatedImage,
                ImageAuthor: `AI - ${modelInfo?.model_name || selectedModel}`,
                MediaAltText: prompt.substring(0, 200),
                mediaTitle: prompt.substring(0, 100)
            });
        }
    };

    const selectedModelInfo = models.find(m => m.model_technical_name === selectedModel);
    const aspectRatios = selectedModelInfo?.supported_aspect_ratios || ['1:1', '16:9', '4:3', '3:2', '9:16'];
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
            {/* Prompt Input */}
            <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Image Description
                </label>
                <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    rows={4}
                    className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white"
                    placeholder="Describe the image you want to generate..."
                />
                {selectedText && (
                    <p className="mt-2 text-sm text-indigo-600 dark:text-indigo-400 flex items-center gap-1">
                        <Sparkles className="w-4 h-4" />
                        Context loaded from selected text
                    </p>
                )}
            </div>

            {/* Model, Aspect Ratio, and Resolution Selection */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        AI Model
                    </label>
                    <select
                        value={selectedModel}
                        onChange={(e) => setSelectedModel(e.target.value)}
                        className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white"
                    >
                        {models.map((model) => (
                            <option key={model.id} value={model.model_technical_name}>
                                {model.model_name}
                            </option>
                        ))}
                    </select>
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Aspect Ratio
                    </label>
                    <select
                        value={aspectRatio}
                        onChange={(e) => setAspectRatio(e.target.value)}
                        className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white"
                    >
                        {aspectRatios.map((ratio) => (
                            <option key={ratio} value={ratio}>
                                {ratio}
                            </option>
                        ))}
                    </select>
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Resolution
                    </label>
                    <select
                        value={resolution}
                        onChange={(e) => setResolution(e.target.value)}
                        className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white"
                    >
                        {resolutions.map((res) => (
                            <option key={res} value={res}>
                                {res === '1K' ? '1K (Default)' : res}
                            </option>
                        ))}
                    </select>
                </div>
            </div>

            {/* Reference Image (Optional - Can be present or absent for article_image) */}
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
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-2.5">
                    Upload an image to guide generation (e.g. style or object transfer), or leave empty to generate purely from prompt.
                </p>

                {referenceImage && referenceImagePreview ? (
                    <div className="flex items-center gap-4 p-3.5 bg-gray-50 dark:bg-gray-900/60 rounded-xl border border-indigo-200 dark:border-indigo-800/60">
                        <div className="relative w-16 h-16 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 flex-shrink-0 bg-black/5">
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
                                {(referenceImage.size / 1024).toFixed(1)} KB • Will be used as reference for {selectedModelInfo?.model_name || 'AI model'}
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
                    <label className="flex flex-col items-center justify-center p-5 border-2 border-dashed border-gray-300 dark:border-gray-700 hover:border-indigo-400 dark:hover:border-indigo-500 rounded-xl cursor-pointer bg-gray-50/50 dark:bg-gray-900/30 hover:bg-indigo-50/20 transition-colors group">
                        <div className="flex flex-col items-center justify-center text-center">
                            <div className="p-2.5 rounded-full bg-gray-100 dark:bg-gray-800 group-hover:bg-indigo-100 dark:group-hover:bg-indigo-950/60 text-gray-500 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors mb-2">
                                <Upload className="w-5 h-5" />
                            </div>
                            <p className="text-xs font-medium text-gray-700 dark:text-gray-300">
                                <span className="text-indigo-600 dark:text-indigo-400 underline">Click to upload</span> or drag and drop a reference image
                            </p>
                            <p className="text-[11px] text-gray-400 dark:text-gray-500 mt-0.5">
                                Supports PNG, JPG, or WEBP (compatible with Nano Banana Pro & Flux 2)
                            </p>
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
                    <div className="rounded-xl overflow-hidden border border-gray-200 dark:border-gray-700">
                        <img
                            src={generatedImage}
                            alt="Generated preview"
                            className="w-full h-auto"
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
                            className="flex-1 bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-3 rounded-xl font-medium transition-colors"
                        >
                            Accept & Continue
                        </button>
                    </div>
                </div>
            )}

            {/* Generate Button */}
            {!generatedImage && (
                <button
                    onClick={handleGenerate}
                    disabled={loading || !prompt.trim()}
                    className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white px-6 py-3 rounded-xl font-medium transition-colors"
                >
                    {loading ? (
                        <>
                            <Loader2 className="w-5 h-5 animate-spin" />
                            Generating...
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
