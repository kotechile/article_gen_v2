import React, { useState } from 'react';
import { Plus, ChevronDown } from 'lucide-react';
import type { Collection } from '../../types/knowledge';

interface CollectionSelectorProps {
    collections: Collection[];
    selectedCollection: Collection | null;
    onSelect: (collection: Collection) => void;
    onCreateCollection: (name: string) => Promise<void>;
}

export const CollectionSelector: React.FC<CollectionSelectorProps> = ({ collections, selectedCollection, onSelect, onCreateCollection }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [isCreating, setIsCreating] = useState(false);
    const [newCollectionName, setNewCollectionName] = useState('');

    const handleCreate = async () => {
        if (!newCollectionName.trim()) return;
        await onCreateCollection(newCollectionName);
        setNewCollectionName('');
        setIsCreating(false);
        setIsOpen(false);
    };

    return (
        <div className="relative">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="flex items-center justify-between w-64 px-4 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
            >
                <span className="text-sm font-medium text-gray-700 dark:text-gray-200 truncate">
                    {selectedCollection ? selectedCollection.name : "Select Collection"}
                </span>
                <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
            </button>

            {isOpen && (
                <div className="absolute top-full left-0 mt-2 w-72 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg z-40 p-2">
                    {/* List */}
                    <div className="max-h-60 overflow-y-auto mb-2 space-y-1">
                        {collections.map(col => (
                            <button
                                key={col.id}
                                onClick={() => {
                                    onSelect(col);
                                    setIsOpen(false);
                                }}
                                className={`w-full text-left px-3 py-2 rounded-md text-sm ${selectedCollection?.id === col.id ? 'bg-indigo-50 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400' : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'}`}
                            >
                                {col.name}
                            </button>
                        ))}
                    </div>

                    {/* Create New Interface */}
                    <div className="pt-2 border-t border-gray-100 dark:border-gray-700">
                        {isCreating ? (
                            <div className="space-y-2">
                                <input
                                    autoFocus
                                    type="text"
                                    value={newCollectionName}
                                    onChange={(e) => setNewCollectionName(e.target.value)}
                                    placeholder="Collection Name"
                                    className="w-full px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-transparent dark:text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                />
                                <div className="flex justify-end gap-2">
                                    <button onClick={() => setIsCreating(false)} className="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300">Cancel</button>
                                    <button onClick={handleCreate} className="text-xs bg-indigo-600 text-white px-2 py-1 rounded hover:bg-indigo-700">Save</button>
                                </div>
                            </div>
                        ) : (
                            <button
                                onClick={() => setIsCreating(true)}
                                className="flex items-center gap-2 w-full px-3 py-2 text-sm text-indigo-600 dark:text-indigo-400 hover:bg-indigo-50 dark:hover:bg-indigo-900/20 rounded-md transition-colors"
                            >
                                <Plus className="w-4 h-4" />
                                Add New Collection
                            </button>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};
