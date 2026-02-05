import React from 'react';
import type { ManualAction } from '../../types/knowledge';
import { ClipboardList, Trash2, Clock, DollarSign, BarChart3 } from 'lucide-react';

interface ActionSuggestionsListProps {
    actions: ManualAction[];
    isLoading: boolean;
    onComplete: (id: string | number, currentStatus: string) => void;
    onDelete: (id: string | number) => void;
}

// Helper for Priority Colors
const getPriorityColor = (level?: string) => {
    switch (level?.toLowerCase()) {
        case 'critical': return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400';
        case 'high': return 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400';
        case 'medium': return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400';
        case 'low': return 'bg-gray-100 text-gray-800 dark:bg-gray-800/50 dark:text-gray-400';
        default: return 'bg-gray-100 text-gray-800 dark:bg-gray-800/50 dark:text-gray-400';
    }
};

export const ActionSuggestionsList: React.FC<ActionSuggestionsListProps> = ({ actions, isLoading, onComplete, onDelete }) => {

    if (isLoading) return <div className="p-8 text-center text-gray-500">Loading suggestions...</div>;

    if (actions.length === 0) {
        return (
            <div className="p-12 border-2 border-dashed border-gray-200 dark:border-gray-700 rounded-xl flex flex-col items-center justify-center text-gray-400">
                <ClipboardList className="w-12 h-12 mb-3 opacity-50" />
                <p>No manual actions required at this time.</p>
            </div>
        );
    }

    return (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {actions.map(action => (
                <div key={action.id} className="group relative flex flex-col bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 shadow-sm hover:shadow-md transition-all duration-300">

                    {/* Header Badges */}
                    <div className="p-5 pb-0 flex justify-between items-start">
                        <div className="flex flex-wrap gap-2">
                            {/* Priority Badge */}
                            {action.priority_level && (
                                <span className={`inline-flex items-center px-2 py-1 rounded-md text-xs font-semibold uppercase tracking-wider ${getPriorityColor(action.priority_level)}`}>
                                    {action.priority_level}
                                </span>
                            )}
                            {/* Difficulty Badge */}
                            {action.difficulty_level && (
                                <span className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300 border border-gray-200 dark:border-gray-700 uppercase tracking-wider">
                                    {action.difficulty_level}
                                </span>
                            )}
                        </div>

                        {/* Delete Action (Top Right) */}
                        <button
                            onClick={() => onDelete(action.id)}
                            className="text-gray-300 hover:text-red-500 transition-colors opacity-0 group-hover:opacity-100"
                            title="Delete Action"
                        >
                            <Trash2 className="w-4 h-4" />
                        </button>
                    </div>

                    <div className="p-5 flex-1">
                        <div className="flex justify-between items-start mb-2">
                            <h4 className="font-bold text-lg text-gray-900 dark:text-gray-100 leading-tight">
                                {action.title}
                            </h4>
                            {/* Status Indicator (if needed explicitly besides checkbox) */}
                            {action.status === 'Completed' && (
                                <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400 border border-green-200 dark:border-green-800">
                                    Done
                                </span>
                            )}
                        </div>

                        <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 leading-relaxed">
                            {action.description}
                        </p>

                        {/* Metrics Grid */}
                        <div className="grid grid-cols-2 gap-3 mb-4">
                            {action.estimated_effort_hours && (
                                <div className="flex items-center gap-2 text-xs font-medium text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-800/50 p-2 rounded-lg border border-gray-100 dark:border-gray-800">
                                    <Clock className="w-3.5 h-3.5 text-indigo-500" />
                                    <span>{action.estimated_effort_hours}h Effort</span>
                                </div>
                            )}
                            {action.cost_estimate && (
                                <div className="flex items-center gap-2 text-xs font-medium text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-800/50 p-2 rounded-lg border border-gray-100 dark:border-gray-800">
                                    <DollarSign className="w-3.5 h-3.5 text-green-500" />
                                    <span>{action.cost_estimate}</span>
                                </div>
                            )}
                            {action.impact_score && (
                                <div className="flex items-center gap-2 text-xs font-medium text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-800/50 p-2 rounded-lg border border-gray-100 dark:border-gray-800 col-span-2">
                                    <BarChart3 className="w-3.5 h-3.5 text-purple-500" />
                                    <div className="flex-1 flex items-center gap-2">
                                        <span>Impact Score</span>
                                        <div className="flex-1 h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                                            <div
                                                className="h-full bg-purple-500 rounded-full"
                                                style={{ width: `${Math.min(100, Math.max(0, action.impact_score))}%` }}
                                            />
                                        </div>
                                        <span className="text-gray-700 dark:text-gray-300">{action.impact_score}</span>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Extra Details (Conditional) */}
                        {action.expected_benefit && (
                            <div className="text-xs text-gray-500 dark:text-gray-400 mb-2">
                                <span className="font-semibold text-gray-700 dark:text-gray-300">Benefit: </span>
                                {action.expected_benefit}
                            </div>
                        )}

                    </div>

                    {/* Footer Checkbox */}
                    <div className="p-4 border-t border-gray-100 dark:border-gray-800 bg-gray-50/50 dark:bg-gray-800/30 rounded-b-xl">
                        <label className="flex items-center gap-3 cursor-pointer group/check">
                            <div className="relative flex items-center">
                                <input
                                    type="checkbox"
                                    checked={action.status === 'Completed' || action.status === 'completed'}
                                    onChange={() => onComplete(action.id, action.status)}
                                    className="peer w-5 h-5 border-2 border-gray-300 dark:border-gray-600 rounded text-indigo-600 focus:ring-indigo-500 dark:focus:ring-indigo-400 transition-colors cursor-pointer"
                                />
                            </div>
                            <span className={`text-sm font-medium transition-colors ${action.status === 'Completed' || action.status === 'completed'
                                    ? 'text-gray-400 line-through decoration-gray-400'
                                    : 'text-gray-700 dark:text-gray-200 group-hover/check:text-indigo-600 dark:group-hover/check:text-indigo-400'
                                }`}>
                                Mark as Completed
                            </span>
                        </label>
                    </div>
                </div>
            ))}
        </div>
    );
};
