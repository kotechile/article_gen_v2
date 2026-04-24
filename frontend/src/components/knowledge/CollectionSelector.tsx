import React, { useState } from 'react';
import { Plus, ChevronDown } from 'lucide-react';
import type { Collection } from '../../types/knowledge';

interface CollectionSelectorProps {
    collections: Collection[];
    selectedCollection: Collection | null;
    onSelect: (collection: Collection) => void;
    onCreateCollection: (name: string) => Promise<void>;
    disabled?: boolean;
}

export const CollectionSelector: React.FC<CollectionSelectorProps> = ({
    collections,
    selectedCollection,
    onSelect,
    onCreateCollection,
    disabled = false,
}) => {
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
                disabled={disabled}
                className="flex w-full max-w-full items-center justify-between gap-3 rounded-lg border border-gray-200 bg-white px-4 py-2 shadow-sm transition-colors hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-60 dark:border-gray-700 dark:bg-gray-800 dark:hover:bg-gray-700 md:w-72"
            >
                <span className="min-w-0 flex-1 truncate text-sm font-medium text-gray-700 dark:text-gray-200">
                    {selectedCollection ? selectedCollection.name : "Select Collection"}
                </span>
                <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
            </button>

            {isOpen && (
                <div className="absolute right-0 top-full z-40 mt-2 w-[min(100vw-2rem,20rem)] max-w-full rounded-lg border border-gray-200 bg-white p-2 shadow-lg dark:border-gray-700 dark:bg-gray-800">
                    {/* List */}
                    <div className="max-h-60 overflow-y-auto mb-2 space-y-1">
                        {collections.map(col => (
                            <button
                                key={col.id}
                                onClick={() => {
                                    onSelect(col);
                                    setIsOpen(false);
                                }}
                                className={`w-full truncate rounded-md px-3 py-2 text-left text-sm ${selectedCollection?.id === col.id ? 'bg-indigo-50 text-indigo-600 dark:bg-indigo-900/30 dark:text-indigo-400' : 'text-gray-700 hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-gray-700'}`}
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
