import React, { useState, useRef } from 'react';
import { Upload, Loader2, X } from 'lucide-react';
import { uploadImageToSupabase } from '../../services/imageService';
import type { ImageMetadata } from '../../types/image';

interface LocalImageUploadProps {
    userId: string;
    onImageUploaded: (imageUrl: string, metadata: Partial<ImageMetadata>) => void;
}

export const LocalImageUpload: React.FC<LocalImageUploadProps> = ({
    userId,
    onImageUploaded
}) => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [dragActive, setDragActive] = useState(false);
    const [shortDescription, setShortDescription] = useState('');
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileSelect = (file: File) => {
        if (!file.type.startsWith('image/')) {
            setError('Please select an image file');
            return;
        }

        setSelectedFile(file);
        setError(null);

        // Create preview
        const reader = new FileReader();
        reader.onload = () => setPreview(reader.result as string);
        reader.readAsDataURL(file);
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    };

    const handleDrag = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    };

    const handleUpload = async () => {
        if (!selectedFile) return;

        setLoading(true);
        setError(null);

        try {
            const { imageUrl } = await uploadImageToSupabase(selectedFile, userId);

            onImageUploaded(imageUrl, {
                ImageUrl: imageUrl,
                ImageAuthor: 'User Upload',
                mediaTitle: selectedFile.name.replace(/\.[^/.]+$/, ''),
                MediaAltText: shortDescription.trim() || selectedFile.name.replace(/\.[^/.]+$/, ''),
                mediaCaption: shortDescription.trim() || ''
            });
        } catch (err: any) {
            setError(err.message || 'Failed to upload image');
        } finally {
            setLoading(false);
        }
    };

    const handleClear = () => {
        setSelectedFile(null);
        setPreview(null);
        setError(null);
        setShortDescription('');
    };

    return (
        <div className="space-y-6">
            {/* Upload Zone */}
            <div
                onDrop={handleDrop}
                onDragOver={handleDrag}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onClick={() => fileInputRef.current?.click()}
                className={`relative border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors ${dragActive
                    ? 'border-indigo-500 bg-indigo-50 dark:bg-indigo-900/20'
                    : 'border-gray-300 dark:border-gray-600 hover:border-indigo-400 dark:hover:border-indigo-500'
                    }`}
            >
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                    className="hidden"
                />

                <div className="flex flex-col items-center gap-4">
                    <div className="w-16 h-16 bg-indigo-100 dark:bg-indigo-900/30 rounded-full flex items-center justify-center">
                        <Upload className="w-8 h-8 text-indigo-600 dark:text-indigo-400" />
                    </div>
                    <div>
                        <p className="text-lg font-medium text-gray-900 dark:text-white">
                            Drop your image here, or click to browse
                        </p>
                        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                            Supports JPG, PNG, GIF, WebP
                        </p>
                    </div>
                </div>
            </div>

            {/* Error Message */}
            {error && (
                <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-4">
                    <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
                </div>
            )}

            {/* Preview */}
            {preview && selectedFile && (
                <div className="space-y-4">
                    <div className="relative rounded-xl overflow-hidden border border-gray-200 dark:border-gray-700">
                        <img
                            src={preview}
                            alt="Preview"
                            className="w-full h-auto max-h-96 object-contain bg-gray-50 dark:bg-gray-900"
                        />
                        <button
                            onClick={(e) => {
                                e.stopPropagation();
                                handleClear();
                            }}
                            className="absolute top-2 right-2 p-2 bg-black/50 hover:bg-black/70 rounded-lg transition-colors"
                        >
                            <X className="w-4 h-4 text-white" />
                        </button>
                    </div>

                    <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-900 rounded-xl">
                        <div className="flex-1 space-y-3">
                            <p className="text-sm font-medium text-gray-900 dark:text-white">
                                {selectedFile.name}
                            </p>
                            <p className="text-xs text-gray-500 dark:text-gray-400">
                                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                            </p>
                            <div>
                                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                                    Short Description (used for WordPress image text)
                                </label>
                                <textarea
                                    value={shortDescription}
                                    onChange={(e) => setShortDescription(e.target.value)}
                                    rows={2}
                                    maxLength={240}
                                    className="w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-white text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                    placeholder="Describe this uploaded image in 1-2 short sentences..."
                                />
                            </div>
                        </div>
                        <button
                            onClick={handleUpload}
                            disabled={loading}
                            className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white px-6 py-3 rounded-xl font-medium transition-colors"
                        >
                            {loading ? (
                                <>
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                    Uploading...
                                </>
                            ) : (
                                <>
                                    <Upload className="w-5 h-5" />
                                    Upload & Continue
                                </>
                            )}
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};
