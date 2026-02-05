import React, { useState } from 'react';
import { Search, Loader2, ChevronLeft, ChevronRight } from 'lucide-react';
import { searchStockImages, downloadAndUploadStockImage } from '../../services/imageService';
import type { StockImageResult, ImageMetadata } from '../../types/image';

interface StockImageSearchProps {
    userId: string;
    onImageSelected: (imageUrl: string, metadata: Partial<ImageMetadata>) => void;
}

export const StockImageSearch: React.FC<StockImageSearchProps> = ({
    userId,
    onImageSelected
}) => {
    const [provider, setProvider] = useState<'pexels' | 'unsplash'>('unsplash');
    const [query, setQuery] = useState('');
    const [images, setImages] = useState<StockImageResult[]>([]);
    const [currentPage, setCurrentPage] = useState(1);
    const [totalPages, setTotalPages] = useState(0);
    const [loading, setLoading] = useState(false);
    const [uploadingId, setUploadingId] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);

    const handleSearch = async (page: number = 1) => {
        if (!query.trim()) return;

        setLoading(true);
        setError(null);

        try {
            const response = await searchStockImages(provider, query, page, 10);
            setImages(response.images);
            setTotalPages(response.totalPages);
            setCurrentPage(page);
        } catch (err: any) {
            setError(err.message || 'Failed to search images');
        } finally {
            setLoading(false);
        }
    };

    const handleSelectImage = async (image: StockImageResult) => {
        setUploadingId(image.id);
        setError(null);

        try {
            // Download and upload to Supabase
            const { imageUrl } = await downloadAndUploadStockImage(
                image.downloadUrl || image.url,
                userId
            );

            onImageSelected(imageUrl, {
                ImageUrl: imageUrl,
                ImageAuthor: `${provider.charAt(0).toUpperCase() + provider.slice(1)} Image - ${image.author}`,
                MediaAltText: image.description,
                mediaTitle: query
            });
        } catch (err: any) {
            setError(err.message || 'Failed to select image');
        } finally {
            setUploadingId(null);
        }
    };

    return (
        <div className="space-y-6">
            {/* Provider Toggle */}
            <div className="flex gap-2 p-1 bg-gray-100 dark:bg-gray-900 rounded-xl">
                <button
                    onClick={() => setProvider('unsplash')}
                    className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${provider === 'unsplash'
                        ? 'bg-white dark:bg-gray-800 text-indigo-600 dark:text-indigo-400 shadow'
                        : 'text-gray-600 dark:text-gray-400'
                        }`}
                >
                    Unsplash
                </button>
                <button
                    onClick={() => setProvider('pexels')}
                    className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${provider === 'pexels'
                        ? 'bg-white dark:bg-gray-800 text-indigo-600 dark:text-indigo-400 shadow'
                        : 'text-gray-600 dark:text-gray-400'
                        }`}
                >
                    Pexels
                </button>
            </div>

            {/* Search Bar */}
            <div className="flex gap-2">
                <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                    placeholder="Search for images..."
                    className="flex-1 px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-900 dark:text-white"
                />
                <button
                    onClick={() => handleSearch()}
                    disabled={loading || !query.trim()}
                    className="px-6 py-3 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white rounded-xl font-medium transition-colors flex items-center gap-2"
                >
                    <Search className="w-5 h-5" />
                    Search
                </button>
            </div>

            {/* Error Message */}
            {error && (
                <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-4">
                    <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
                </div>
            )}

            {/* Loading State */}
            {loading && (
                <div className="flex items-center justify-center py-12">
                    <Loader2 className="w-8 h-8 animate-spin text-indigo-600" />
                </div>
            )}

            {/* Image Grid */}
            {!loading && images.length > 0 && (
                <>
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                        {images.map((image) => (
                            <div
                                key={image.id}
                                className="group relative bg-gray-100 dark:bg-gray-900 rounded-xl overflow-hidden cursor-pointer"
                                onClick={() => uploadingId !== image.id && handleSelectImage(image)}
                            >
                                <img
                                    src={image.thumbnail}
                                    alt={image.description}
                                    className="w-full h-48 object-cover"
                                />
                                <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                    {uploadingId === image.id ? (
                                        <Loader2 className="w-8 h-8 text-white animate-spin" />
                                    ) : (
                                        <span className="text-white font-medium">Select</span>
                                    )}
                                </div>
                                <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent p-2">
                                    <p className="text-xs text-white truncate">by {image.author}</p>
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Pagination */}
                    {totalPages > 1 && (
                        <div className="flex items-center justify-center gap-2">
                            <button
                                onClick={() => handleSearch(currentPage - 1)}
                                disabled={currentPage === 1 || loading}
                                className="p-2 rounded-lg border border-gray-300 dark:border-gray-600 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                            >
                                <ChevronLeft className="w-5 h-5" />
                            </button>
                            <span className="text-sm text-gray-600 dark:text-gray-400">
                                Page {currentPage} of {totalPages}
                            </span>
                            <button
                                onClick={() => handleSearch(currentPage + 1)}
                                disabled={currentPage === totalPages || loading}
                                className="p-2 rounded-lg border border-gray-300 dark:border-gray-600 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                            >
                                <ChevronRight className="w-5 h-5" />
                            </button>
                        </div>
                    )}
                </>
            )}

            {/* No Results */}
            {!loading && query && images.length === 0 && (
                <div className="text-center py-12 text-gray-500 dark:text-gray-400">
                    No images found. Try a different search term.
                </div>
            )}
        </div>
    );
};
