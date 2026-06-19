import React, { useState } from 'react';
import { Loader2, ArrowLeft, Save } from 'lucide-react';
import { saveImageMetadata } from '../../services/imageService';
import type { ImageMetadata } from '../../types/image';

interface ImageMetadataEditorProps {
    imageUrl: string;
    initialMetadata: Partial<ImageMetadata>;
    userId: string;
    onSave: (metadata: ImageMetadata) => void;
    onBack: () => void;
}

export const ImageMetadataEditor: React.FC<ImageMetadataEditorProps> = ({
    imageUrl,
    initialMetadata,
    userId,
    onSave,
    onBack
}) => {
    const [metadata, setMetadata] = useState<Partial<ImageMetadata>>({
        ImageUrl: imageUrl,
        ImageAuthor: initialMetadata.ImageAuthor || '',
        MediaAltText: initialMetadata.MediaAltText || '',
        mediaTitle: initialMetadata.mediaTitle || '',
        mediaCaption: initialMetadata.mediaCaption || '',
        width: initialMetadata.width || '100%',
        alignment: initialMetadata.alignment || 'center',
    });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleSave = async () => {
        setLoading(true);
        setError(null);

        try {
            const fullMetadata: ImageMetadata = {
                ...metadata,
                user_id: userId,
                ImageUrl: imageUrl,
            } as ImageMetadata;

            const saved = await saveImageMetadata(fullMetadata);
            onSave({
                ...saved,
                width: metadata.width,
                alignment: metadata.alignment
            });
        } catch (err: any) {
            setError(err.message || 'Failed to save image metadata');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {/* Image Preview */}
            <div className="rounded-xl overflow-hidden border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
                <img
                    src={imageUrl}
                    alt="Preview"
                    className="w-full h-auto max-h-96 object-contain"
                />
            </div>

            {/* Metadata Fields */}
            <div className="space-y-4">
                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Title
                    </label>
                    <input
                        type="text"
                        value={metadata.mediaTitle || ''}
                        onChange={(e) => setMetadata({ ...metadata, mediaTitle: e.target.value })}
                        className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white"
                        placeholder="Image title..."
                    />
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Author / Credit
                    </label>
                    <input
                        type="text"
                        value={metadata.ImageAuthor || ''}
                        onChange={(e) => setMetadata({ ...metadata, ImageAuthor: e.target.value })}
                        className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white"
                        placeholder="Image author or source..."
                    />
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Alt Text
                        <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">
                            (For SEO and accessibility)
                        </span>
                    </label>
                    <textarea
                        value={metadata.MediaAltText || ''}
                        onChange={(e) => setMetadata({ ...metadata, MediaAltText: e.target.value })}
                        rows={3}
                        className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white"
                        placeholder="Describe the image for screen readers and search engines..."
                    />
                </div>

                <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                        Caption
                        <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">
                            (Displayed below image in article)
                        </span>
                    </label>
                    <textarea
                        value={metadata.mediaCaption || ''}
                        onChange={(e) => setMetadata({ ...metadata, mediaCaption: e.target.value })}
                        rows={2}
                        className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white"
                        placeholder="Optional caption to display below the image..."
                    />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                            Image Size (Width)
                        </label>
                        <select
                            value={metadata.width || '100%'}
                            onChange={(e) => setMetadata({ ...metadata, width: e.target.value })}
                            className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white"
                        >
                            <option value="100%">Full Width (100%)</option>
                            <option value="75%">Large (75%)</option>
                            <option value="50%">Medium (50%)</option>
                            <option value="25%">Small (25%)</option>
                        </select>
                    </div>

                    <div>
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                            Horizontal Alignment
                        </label>
                        <select
                            value={metadata.alignment || 'center'}
                            onChange={(e) => setMetadata({ ...metadata, alignment: e.target.value as any })}
                            className="w-full px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white"
                        >
                            <option value="center">Center (No Wrap)</option>
                            <option value="left">Left (Wrap Text)</option>
                            <option value="right">Right (Wrap Text)</option>
                        </select>
                    </div>
                </div>
            </div>

            {/* Error Message */}
            {error && (
                <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-4">
                    <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
                </div>
            )}

            {/* Actions */}
            <div className="flex gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
                <button
                    onClick={onBack}
                    className="flex items-center gap-2 px-6 py-3 rounded-xl border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                >
                    <ArrowLeft className="w-4 h-4" />
                    Back
                </button>
                <button
                    onClick={handleSave}
                    disabled={loading}
                    className="flex-1 flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white px-6 py-3 rounded-xl font-medium transition-colors"
                >
                    {loading ? (
                        <>
                            <Loader2 className="w-5 h-5 animate-spin" />
                            Saving...
                        </>
                    ) : (
                        <>
                            <Save className="w-5 h-5" />
                            Save & Insert into Article
                        </>
                    )}
                </button>
            </div>
        </div>
    );
};
