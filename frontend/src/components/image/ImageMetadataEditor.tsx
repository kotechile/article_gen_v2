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
            onSave(saved);
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
