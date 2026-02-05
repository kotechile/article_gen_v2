import React, { useState, useEffect } from 'react';
import { Loader2, BarChart3, Sparkles } from 'lucide-react';
import { getInfographicTemplates, generateInfographic } from '../../services/imageService';
import type { InfographicTemplate, ImageMetadata } from '../../types/image';

interface InfographicGeneratorProps {
    userId: string;
    selectedText?: string;
    onInfographicGenerated: (imageUrl: string, metadata: Partial<ImageMetadata>) => void;
}

export const InfographicGenerator: React.FC<InfographicGeneratorProps> = ({
    userId,
    selectedText,
    onInfographicGenerated
}) => {
    const [templates, setTemplates] = useState<InfographicTemplate[]>([]);
    const [selectedTemplate, setSelectedTemplate] = useState<number | null>(null);
    const [storyText, setStoryText] = useState(selectedText || '');
    const [loading, setLoading] = useState(false);
    const [loadingTemplates, setLoadingTemplates] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        loadTemplates();
    }, []);

    const loadTemplates = async () => {
        try {
            const data = await getInfographicTemplates();
            setTemplates(data);
            if (data.length > 0) {
                setSelectedTemplate(data[0].id);
            }
        } catch (err: any) {
            setError('Failed to load infographic templates');
            console.error(err);
        } finally {
            setLoadingTemplates(false);
        }
    };

    const handleGenerate = async () => {
        if (!selectedTemplate || !storyText.trim()) {
            setError('Please select a template and enter content');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const response = await generateInfographic({
                templateId: selectedTemplate,
                storyText,
                user_id: userId
            });

            const template = templates.find(t => t.id === selectedTemplate);
            onInfographicGenerated(response.imageUrl, {
                ...response.metadata,
                ImageAuthor: `Infographic - ${template?.Label || 'Custom'}`,
            });
        } catch (err: any) {
            setError(err.message || 'Failed to generate infographic');
        } finally {
            setLoading(false);
        }
    };

    const selectedTemplateData = templates.find(t => t.id === selectedTemplate);

    if (loadingTemplates) {
        return (
            <div className="flex items-center justify-center py-12">
                <Loader2 className="w-8 h-8 animate-spin text-indigo-600" />
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Template Selection */}
            <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Template
                </label>
                <select
                    value={selectedTemplate || ''}
                    onChange={(e) => setSelectedTemplate(Number(e.target.value))}
                    className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white"
                >
                    {templates.map((template) => (
                        <option key={template.id} value={template.id}>
                            {template.Label} ({template.numberOfItems} items)
                        </option>
                    ))}
                </select>
            </div>

            {/* Sample Preview */}
            {selectedTemplateData?.sampleImage && (
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Template Preview
                    </label>
                    <div className="rounded-xl overflow-hidden border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
                        <img
                            src={selectedTemplateData.sampleImage}
                            alt={selectedTemplateData.Label}
                            className="w-full h-auto max-h-64 object-contain"
                        />
                    </div>
                </div>
            )}

            {/* Story Text */}
            <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Content
                </label>
                <textarea
                    value={storyText}
                    onChange={(e) => setStoryText(e.target.value)}
                    rows={8}
                    className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white"
                    placeholder="Enter the content or story for your infographic. The AI will automatically structure it based on the selected template..."
                />
                {selectedText && (
                    <p className="mt-2 text-sm text-indigo-600 dark:text-indigo-400 flex items-center gap-1">
                        <Sparkles className="w-4 h-4" />
                        Content loaded from selected text
                    </p>
                )}
            </div>

            {/* Error Message */}
            {error && (
                <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-4">
                    <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
                </div>
            )}

            {/* Generate Button */}
            <button
                onClick={handleGenerate}
                disabled={loading || !storyText.trim() || !selectedTemplate}
                className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white px-6 py-3 rounded-xl font-medium transition-colors"
            >
                {loading ? (
                    <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        Generating Infographic...
                    </>
                ) : (
                    <>
                        <BarChart3 className="w-5 h-5" />
                        Generate Infographic
                    </>
                )}
            </button>

            {/* Info Text */}
            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-4">
                <p className="text-sm text-blue-600 dark:text-blue-400">
                    <strong>How it works:</strong> Our AI will analyze your content and automatically
                    map it to the template structure, creating a professional infographic with appropriate
                    icons and formatting.
                </p>
            </div>
        </div>
    );
};
