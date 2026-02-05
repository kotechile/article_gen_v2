import React, { useState } from 'react';
import { X, Check } from 'lucide-react';

interface Citation {
    id?: string;
    title: string;
    url?: string;
    author?: string;
    source_type?: string;
    publication_date?: string;
    publisher?: string;
}

interface ReferenceSelectorProps {
    citations: Citation[];
    selectedCitations: Set<number>;
    showInTextCitations: boolean;
    onClose: () => void;
    onApply: (selectedIndices: Set<number>, showInText: boolean) => void;
}

export const ReferenceSelector: React.FC<ReferenceSelectorProps> = ({
    citations,
    selectedCitations: initialSelected,
    showInTextCitations: initialShowInText,
    onClose,
    onApply
}) => {
    const [selectedIndices, setSelectedIndices] = useState<Set<number>>(new Set(initialSelected));
    const [showInText, setShowInText] = useState(initialShowInText);

    const toggleCitation = (index: number) => {
        const newSelected = new Set(selectedIndices);
        if (newSelected.has(index)) {
            newSelected.delete(index);
        } else {
            newSelected.add(index);
        }
        setSelectedIndices(newSelected);
    };

    const handleSelectAll = () => {
        if (selectedIndices.size === citations.length) {
            setSelectedIndices(new Set());
        } else {
            setSelectedIndices(new Set(citations.map((_, i) => i)));
        }
    };

    const handleApply = () => {
        onApply(selectedIndices, showInText);
        onClose();
    };

    return (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl max-w-3xl w-full max-h-[80vh] flex flex-col">
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
                    <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                        Manage References
                    </h2>
                    <button
                        onClick={onClose}
                        className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Settings */}
                <div className="p-6 border-b border-gray-200 dark:border-gray-700 space-y-4">
                    <div className="flex items-center justify-between">
                        <label className="flex items-center gap-3 cursor-pointer">
                            <input
                                type="checkbox"
                                checked={showInText}
                                onChange={(e) => setShowInText(e.target.checked)}
                                className="w-5 h-5 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 cursor-pointer"
                            />
                            <span className="text-sm font-medium text-gray-900 dark:text-white">
                                Show in-text citation numbers
                            </span>
                        </label>
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                            {showInText ? 'e.g., [1]' : 'Hidden'}
                        </span>
                    </div>

                    <div className="flex items-center justify-between">
                        <button
                            onClick={handleSelectAll}
                            className="text-sm font-medium text-indigo-600 hover:text-indigo-700 dark:text-indigo-400 dark:hover:text-indigo-300"
                        >
                            {selectedIndices.size === citations.length ? 'Deselect All' : 'Select All'}
                        </button>
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                            {selectedIndices.size} of {citations.length} selected
                        </span>
                    </div>
                </div>

                {/* Citations List */}
                <div className="flex-1 overflow-y-auto p-6">
                    {citations.length === 0 ? (
                        <div className="text-center py-12">
                            <p className="text-gray-500 dark:text-gray-400">
                                No references found for this article.
                            </p>
                        </div>
                    ) : (
                        <div className="space-y-3">
                            {citations.map((citation, index) => (
                                <label
                                    key={index}
                                    className={`flex items-start gap-4 p-4 rounded-xl border-2 transition-all cursor-pointer ${selectedIndices.has(index)
                                        ? 'border-indigo-500 bg-indigo-50 dark:bg-indigo-900/20'
                                        : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                                        }`}
                                >
                                    <input
                                        type="checkbox"
                                        checked={selectedIndices.has(index)}
                                        onChange={() => toggleCitation(index)}
                                        className="mt-1 w-5 h-5 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 cursor-pointer flex-shrink-0"
                                    />
                                    <div className="flex-1 min-w-0">
                                        <div className="flex items-start justify-between gap-2 mb-1">
                                            <span className="font-semibold text-sm text-gray-900 dark:text-white">
                                                [{index + 1}]
                                            </span>
                                            {citation.source_type && (
                                                <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 uppercase flex-shrink-0">
                                                    {citation.source_type}
                                                </span>
                                            )}
                                        </div>
                                        <p className="text-sm font-medium text-gray-900 dark:text-white mb-1 line-clamp-2">
                                            {citation.title || 'Unknown Source'}
                                        </p>
                                        {citation.author && citation.author !== 'Unknown Author' && (
                                            <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">
                                                {citation.author}
                                                {citation.publication_date && ` (${citation.publication_date})`}
                                            </p>
                                        )}
                                        {citation.url && citation.url !== '#' && (
                                            <a
                                                href={citation.url}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                onClick={(e) => e.stopPropagation()}
                                                className="text-xs text-indigo-600 dark:text-indigo-400 hover:underline truncate block"
                                            >
                                                {citation.url}
                                            </a>
                                        )}
                                    </div>
                                </label>
                            ))}
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="p-6 border-t border-gray-200 dark:border-gray-700 flex items-center justify-end gap-3">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-xl transition"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleApply}
                        className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl font-medium text-sm transition shadow-lg shadow-indigo-500/20"
                    >
                        <Check className="w-4 h-4" />
                        Apply Changes
                    </button>
                </div>
            </div>
        </div>
    );
};
