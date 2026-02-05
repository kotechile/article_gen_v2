import React, { useState } from 'react';
import { X, Loader2 } from 'lucide-react';
import type { Collection } from '../../types/knowledge';

interface ManualEntryModalProps {
    isOpen: boolean;
    onClose: () => void;
    currentCollection: Collection | null;
    onSubmit: (title: string, content: string) => Promise<void>;
}

export const ManualEntryModal: React.FC<ManualEntryModalProps> = ({ isOpen, onClose, currentCollection, onSubmit }) => {
    const [title, setTitle] = useState('');
    const [content, setContent] = useState('');
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [error, setError] = useState<string | null>(null);

    if (!isOpen) return null;

    const handleSubmit = async () => {
        if (!title.trim() || !content.trim()) {
            setError("Title and content are required.");
            return;
        }
        if (!currentCollection) {
            setError("No collection selected.");
            return;
        }

        setIsSubmitting(true);
        setError(null);

        try {
            await onSubmit(title, content);
            onClose();
            setTitle(''); // Reset
            setContent('');
        } catch (err: any) {
            setError(err.message || "Failed to save entry");
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
            <div className="bg-white dark:bg-gray-800 rounded-2xl w-full max-w-2xl shadow-xl border border-gray-200 dark:border-gray-700 overflow-hidden flex flex-col h-[660px]"> {/* Fixed height matching figma approx */}
                <div className="p-6 border-b border-gray-100 dark:border-gray-700 flex justify-between items-center shrink-0">
                    <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
                        RAG Text Input
                    </h3>
                    <button onClick={onClose} className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300">
                        <X className="w-5 h-5" />
                    </button>
                </div>

                <div className="p-6 flex flex-col flex-1 overflow-hidden space-y-6">

                    <div className="space-y-2 shrink-0">
                        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                            Title
                        </label>
                        <input
                            type="text"
                            value={title}
                            onChange={(e) => setTitle(e.target.value)}
                            placeholder="Enter document title..."
                            className="w-full px-4 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all dark:text-white"
                        />
                    </div>

                    <div className="space-y-2 flex-1 flex flex-col min-h-0">
                        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                            Content
                        </label>
                        <textarea
                            value={content}
                            onChange={(e) => setContent(e.target.value)}
                            placeholder="Paste your text here..."
                            className="w-full flex-1 p-4 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-700 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all resize-none dark:text-white"
                        />
                    </div>

                    {error && (
                        <div className="p-3 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 text-sm rounded-lg shrink-0">
                            {error}
                        </div>
                    )}

                    <div className="flex justify-end gap-3 pt-2 shrink-0">
                        <button
                            onClick={onClose}
                            className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            onClick={handleSubmit}
                            disabled={isSubmitting}
                            className="flex items-center gap-2 px-6 py-2 text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded-lg transition-colors disabled:opacity-50"
                        >
                            {isSubmitting && <Loader2 className="w-4 h-4 animate-spin" />}
                            Save
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};
