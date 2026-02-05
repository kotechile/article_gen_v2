import React, { useState, useEffect } from 'react';
import { Loader2, Sparkles, RefreshCw } from 'lucide-react';
import { generateAIImage, getImageProviderModels } from '../../services/imageService';
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
    const [referenceImage, setReferenceImage] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);
    const [loadingModels, setLoadingModels] = useState(true);
    const [generatedImage, setGeneratedImage] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        loadModels();
    }, []);

    const loadModels = async () => {
        try {
            const data = await getImageProviderModels();
            setModels(data);
            if (data.length > 0) {
                setSelectedModel(data[0].model_technical_name);
            }
        } catch (err) {
            setError('Failed to load AI models');
            console.error(err);
        } finally {
            setLoadingModels(false);
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
                aspectRatio,
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

            {/* Model Selection */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
            </div>

            {/* Reference Image (if supported) */}
            {selectedModelInfo?.supports_reference_image && (
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Reference Image (Optional)
                    </label>
                    <input
                        type="file"
                        accept="image/*"
                        onChange={(e) => setReferenceImage(e.target.files?.[0] || null)}
                        className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white"
                    />
                </div>
            )}

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
