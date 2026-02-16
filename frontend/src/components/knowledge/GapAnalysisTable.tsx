import React, { useState } from 'react';
import type { Title } from '../../types/knowledge';
import { Sparkles, CheckSquare, Search } from 'lucide-react';

interface GapAnalysisTableProps {
    titles: Title[];
    isLoading: boolean;
    onStartGapFill: (selectedTitles: Title[]) => Promise<void>;
    onEnhancePlus: (selectedTitles: Title[]) => Promise<void>;
    researchMethod: 'standard' | 'deep';
    onResearchMethodChange: (method: 'standard' | 'deep') => void;
}

export const GapAnalysisTable: React.FC<GapAnalysisTableProps> = ({ titles, isLoading, onStartGapFill, onEnhancePlus, researchMethod, onResearchMethodChange }) => {
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
    const [isProcessing, setIsProcessing] = useState(false);

    const toggleSelect = (id: string) => {
        const newSet = new Set(selectedIds);
        if (newSet.has(id)) newSet.delete(id);
        else newSet.add(id);
        setSelectedIds(newSet);
    };

    const toggleSelectAll = () => {
        if (selectedIds.size === titles.length) setSelectedIds(new Set());
        else setSelectedIds(new Set(titles.map(t => t.id)));
    };

    const handleStart = async () => {
        if (selectedIds.size === 0) return;
        setIsProcessing(true);
        try {
            const selectedTitles = titles.filter(t => selectedIds.has(t.id));
            await onStartGapFill(selectedTitles);
            setSelectedIds(new Set());
        } finally {
            setIsProcessing(false);
        }
    };

    const handleEnhance = async () => {
        if (selectedIds.size === 0) return;
        setIsProcessing(true);
        try {
            const selectedTitles = titles.filter(t => selectedIds.has(t.id));
            await onEnhancePlus(selectedTitles);
            setSelectedIds(new Set());
        } finally {
            setIsProcessing(false);
        }
    };

    if (isLoading) return <div className="p-8 text-center text-gray-500">Loading new titles...</div>;

    if (titles.length === 0) {
        return (
            <div className="p-12 border-2 border-dashed border-gray-200 dark:border-gray-700 rounded-xl flex flex-col items-center justify-center text-gray-400">
                <Search className="w-12 h-12 mb-3 opacity-50" />
                <p>No new titles found to analyze.</p>
            </div>
        );
    }

    return (
        <div className="flex flex-col h-full bg-white dark:bg-gray-900 rounded-xl overflow-hidden border border-gray-200 dark:border-gray-800">
            {/* Header */}
            {/* Header */}
            <div className="p-6 border-b border-gray-100 dark:border-gray-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                    <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Identify Gaps for New Ideas</h3>
                    <p className="text-gray-500 text-sm mt-1">Select article ideas to research and fill knowledge gaps</p>
                </div>
                <div className="flex items-center gap-4">
                    {selectedIds.size > 0 && (
                        <span className="text-sm text-gray-500 animate-fadeIn">
                            {selectedIds.size} selected
                        </span>
                    )}
                    <div className="flex gap-3 items-center">
                        <div className="flex bg-gray-100 dark:bg-gray-800 p-1 rounded-lg mr-2">
                            <button
                                onClick={() => onResearchMethodChange('standard')}
                                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${researchMethod === 'standard'
                                        ? 'bg-white dark:bg-gray-700 shadow text-gray-900 dark:text-white'
                                        : 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
                                    }`}
                            >
                                Standard
                            </button>
                            <button
                                onClick={() => onResearchMethodChange('deep')}
                                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${researchMethod === 'deep'
                                        ? 'bg-white dark:bg-gray-700 shadow text-indigo-600 dark:text-indigo-400'
                                        : 'text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
                                    }`}
                            >
                                Deep Research
                            </button>
                        </div>

                        <button
                            onClick={handleEnhance}
                            disabled={selectedIds.size === 0 || isProcessing}
                            className="px-4 py-2 bg-purple-100 hover:bg-purple-200 text-purple-700 font-medium rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 text-sm"
                        >
                            <Sparkles className="w-4 h-4" />
                            Enhance+
                        </button>
                        <button
                            onClick={handleStart}
                            disabled={selectedIds.size === 0 || isProcessing}
                            className="px-6 py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-sm text-sm"
                        >
                            {isProcessing ? 'Processing...' : 'Start'}
                        </button>
                    </div>
                </div>
            </div>

            <div className="flex-1 overflow-auto">
                <table className="w-full text-left text-sm">
                    <thead>
                        <tr className="bg-white dark:bg-gray-900 border-b border-gray-100 dark:border-gray-800 sticky top-0 z-10">
                            <th className="w-16 px-6 py-4">
                                <button onClick={toggleSelectAll} className="flex items-center text-gray-400 hover:text-gray-600 transition-colors">
                                    {selectedIds.size === titles.length && titles.length > 0 ? (
                                        <CheckSquare className="w-5 h-5 text-indigo-600" />
                                    ) : (
                                        <div className="w-5 h-5 border-2 border-gray-300 rounded hover:border-gray-400" />
                                    )}
                                </button>
                            </th>
                            <th className="px-6 py-4 font-medium text-gray-500 dark:text-gray-400">Title</th>
                            <th className="px-6 py-4 font-medium text-gray-500 dark:text-gray-400 text-right">Traffic Score</th>
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
                        {titles.map((title) => (
                            <tr key={title.id} onClick={() => toggleSelect(title.id)} className="hover:bg-gray-50 dark:hover:bg-gray-800/50 cursor-pointer transition-colors">
                                <td className="px-6 py-4">
                                    {selectedIds.has(title.id) ? (
                                        <CheckSquare className="w-5 h-5 text-indigo-600" />
                                    ) : (
                                        <div className="w-5 h-5 border-2 border-gray-300 rounded" />
                                    )}
                                </td>
                                <td className="px-6 py-4 font-medium text-gray-900 dark:text-gray-200">
                                    {title.Title}
                                </td>
                                <td className="px-6 py-4 text-right font-mono text-gray-600 dark:text-gray-400">
                                    {title.traffic_potential_score ? title.traffic_potential_score.toLocaleString() : '-'}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>


        </div>
    );
};
