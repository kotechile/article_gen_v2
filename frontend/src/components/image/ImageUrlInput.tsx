import React, { useState } from 'react';
import { Link as LinkIcon, Loader2 } from 'lucide-react';
import type { ImageMetadata } from '../../types/image';

interface ImageUrlInputProps {
    onImageSelected: (imageUrl: string, metadata: Partial<ImageMetadata>) => void;
}

export const ImageUrlInput: React.FC<ImageUrlInputProps> = ({ onImageSelected }) => {
    const [url, setUrl] = useState('');
    const [preview, setPreview] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const isValidUrl = (urlString: string): boolean => {
        try {
            const urlObj = new URL(urlString);
            return urlObj.protocol === 'http:' || urlObj.protocol === 'https:';
        } catch {
            return false;
        }
    };

    const handleUrlChange = (value: string) => {
        setUrl(value);
        setError(null);
        setPreview(null);
    };

    const handlePreview = () => {
        if (!url.trim()) {
            setError('Please enter a URL');
            return;
        }

        if (!isValidUrl(url)) {
            setError('Please enter a valid HTTP or HTTPS URL');
            return;
        }

        setLoading(true);
        setError(null);

        // Test if image loads
        const img = new Image();
        img.onload = () => {
            setPreview(url);
            setLoading(false);
        };
        img.onerror = () => {
            setError('Failed to load image from URL. Please check the URL and try again.');
            setLoading(false);
        };
        img.src = url;
    };

    const handleContinue = () => {
        if (preview) {
            onImageSelected(preview, {
                ImageUrl: preview,
                ImageAuthor: 'External URL',
                mediaTitle: 'Image from URL'
            });
        }
    };

    return (
        <div className="space-y-6">
            {/* URL Input */}
            <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Image URL
                </label>
                <div className="flex gap-2">
                    <input
                        type="url"
                        value={url}
                        onChange={(e) => handleUrlChange(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && handlePreview()}
                        placeholder="https://example.com/image.jpg"
                        className="flex-1 px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white"
                    />
                    <button
                        onClick={handlePreview}
                        disabled={loading || !url.trim()}
                        className="px-6 py-3 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white rounded-xl font-medium transition-colors flex items-center gap-2"
                    >
                        {loading ? (
                            <Loader2 className="w-5 h-5 animate-spin" />
                        ) : (
                            <>
                                <LinkIcon className="w-5 h-5" />
                                Preview
                            </>
                        )}
                    </button>
                </div>
            </div>

            {/* Error Message */}
            {error && (
                <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-4">
                    <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
                </div>
            )}

            {/* Image Preview */}
            {preview && (
                <div className="space-y-4">
                    <div className="rounded-xl overflow-hidden border border-gray-200 dark:border-gray-700">
                        <img
                            src={preview}
                            alt="Preview"
                            className="w-full h-auto max-h-96 object-contain bg-gray-50 dark:bg-gray-900"
                        />
                    </div>

                    <button
                        onClick={handleContinue}
                        className="w-full bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-3 rounded-xl font-medium transition-colors"
                    >
                        Continue with this Image
                    </button>
                </div>
            )}

            {/* Help Text */}
            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-4">
                <p className="text-sm text-blue-600 dark:text-blue-400">
                    <strong>Tip:</strong> Enter a direct link to an image (e.g., ending in .jpg, .png, .gif).
                    The image will be referenced by URL and not uploaded to storage.
                </p>
            </div>
        </div>
    );
};
