import React, { useState } from 'react';
import { X, Sparkles, Images, Upload, Link as LinkIcon, BarChart3, Wand2 } from 'lucide-react';
import type { ImageMetadata, ImageSourceTab } from '../types/image';
import { AIImageGeneration } from './image/AIImageGeneration';
import { SmartContextImageGeneration } from './image/SmartContextImageGeneration';
import { StockImageSearch } from './image/StockImageSearch';
import { LocalImageUpload } from './image/LocalImageUpload';
import { ImageUrlInput } from './image/ImageUrlInput';
import { InfographicGenerator } from './image/InfographicGenerator';
import { ImageMetadataEditor } from './image/ImageMetadataEditor';

interface AddImageModalProps {
    onClose: () => void;
    onImageSelected: (imageUrl: string, metadata: ImageMetadata) => void;
    selectedText?: string;
    userId: string;
    initialTab?: ImageSourceTab;
}

export const AddImageModal: React.FC<AddImageModalProps> = ({
    onClose,
    onImageSelected,
    selectedText,
    userId,
    initialTab = selectedText && selectedText.trim() ? 'smart' : 'ai'
}) => {
    const [activeTab, setActiveTab] = useState<ImageSourceTab>(initialTab);
    const [selectedImage, setSelectedImage] = useState<{ url: string; metadata: Partial<ImageMetadata> } | null>(null);
    const [showMetadataEditor, setShowMetadataEditor] = useState(false);

    const handleImageGenerated = (imageUrl: string, metadata: Partial<ImageMetadata>) => {
        setSelectedImage({ url: imageUrl, metadata });
        setShowMetadataEditor(true);
    };

    const handleMetadataSaved = (metadata: ImageMetadata) => {
        onImageSelected(metadata.ImageUrl || '', metadata);
        onClose();
    };

    const tabs = [
        { id: 'smart' as ImageSourceTab, label: 'Smart Context', icon: Wand2 },
        { id: 'ai' as ImageSourceTab, label: 'AI Generation', icon: Sparkles },
        { id: 'stock' as ImageSourceTab, label: 'Stock Images', icon: Images },
        { id: 'upload' as ImageSourceTab, label: 'Upload', icon: Upload },
        { id: 'url' as ImageSourceTab, label: 'Image URL', icon: LinkIcon },
        { id: 'infographic' as ImageSourceTab, label: 'Infographic', icon: BarChart3 },
    ];

    return (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl max-w-5xl w-full max-h-[90vh] overflow-hidden flex flex-col">
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                        {showMetadataEditor ? 'Edit Image Details' : 'Add Image'}
                    </h2>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                    >
                        <X className="w-5 h-5 text-gray-500 dark:text-gray-400" />
                    </button>
                </div>

                {showMetadataEditor && selectedImage ? (
                    <ImageMetadataEditor
                        imageUrl={selectedImage.url}
                        initialMetadata={selectedImage.metadata}
                        userId={userId}
                        onSave={handleMetadataSaved}
                        onBack={() => setShowMetadataEditor(false)}
                    />
                ) : (
                    <>
                        {/* Tabs */}
                        <div className="flex border-b border-gray-200 dark:border-gray-700 overflow-x-auto">
                            {tabs.map((tab) => {
                                const Icon = tab.icon;
                                return (
                                    <button
                                        key={tab.id}
                                        onClick={() => setActiveTab(tab.id)}
                                        className={`flex items-center gap-2 px-6 py-4 font-medium transition-colors whitespace-nowrap ${activeTab === tab.id
                                            ? 'text-indigo-600 dark:text-indigo-400 border-b-2 border-indigo-600 dark:border-indigo-400'
                                            : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
                                            }`}
                                    >
                                        <Icon className="w-4 h-4" />
                                        {tab.label}
                                    </button>
                                );
                            })}
                        </div>

                        {/* Content */}
                        <div className="flex-1 overflow-y-auto p-6">
                            {activeTab === 'smart' && (
                                <SmartContextImageGeneration
                                    userId={userId}
                                    selectedText={selectedText}
                                    onImageGenerated={handleImageGenerated}
                                />
                            )}
                            {activeTab === 'ai' && (
                                <AIImageGeneration
                                    userId={userId}
                                    selectedText={selectedText}
                                    onImageGenerated={handleImageGenerated}
                                />
                            )}
                            {activeTab === 'stock' && (
                                <StockImageSearch
                                    userId={userId}
                                    onImageSelected={handleImageGenerated}
                                />
                            )}
                            {activeTab === 'upload' && (
                                <LocalImageUpload
                                    userId={userId}
                                    onImageUploaded={handleImageGenerated}
                                />
                            )}
                            {activeTab === 'url' && (
                                <ImageUrlInput
                                    onImageSelected={handleImageGenerated}
                                />
                            )}
                            {activeTab === 'infographic' && (
                                <InfographicGenerator
                                    userId={userId}
                                    selectedText={selectedText}
                                    onInfographicGenerated={handleImageGenerated}
                                />
                            )}
                        </div>
                    </>
                )}
            </div>
        </div>
    );
};
