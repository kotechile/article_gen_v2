import React, { useState } from 'react';
import { Info } from 'lucide-react';
import type { MetricExplanation } from '../types/metrics';

interface TooltipProps {
    explanation: MetricExplanation;
    className?: string;
    children?: React.ReactNode;
}

export const MetricTooltip: React.FC<TooltipProps> = ({ explanation, className = '', children }) => {
    const [isVisible, setIsVisible] = useState(false);

    return (
        <div
            className={`relative inline-flex items-center gap-1 cursor-help group ${className}`}
            onMouseEnter={() => setIsVisible(true)}
            onMouseLeave={() => setIsVisible(false)}
        >
            {children}
            <Info className="w-3.5 h-3.5 text-gray-400 hover:text-indigo-500 transition-colors" />

            {isVisible && (
                <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-72 bg-white dark:bg-gray-800 rounded-xl shadow-xl border border-gray-100 dark:border-gray-700 p-4 z-50 animate-in fade-in zoom-in-95 duration-200">
                    <div className="space-y-3 text-left">
                        <div className="flex items-center justify-between border-b border-gray-100 dark:border-gray-700 pb-2">
                            <h4 className="font-semibold text-gray-900 dark:text-white text-sm">{explanation.title}</h4>
                            <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${explanation.origin_type === 'Real Data'
                                ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                                : explanation.origin_type === 'AI Estimate'
                                    ? 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400'
                                    : 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
                                }`}>
                                {explanation.origin_type}
                            </span>
                        </div>

                        <div>
                            <span className="text-xs font-medium text-gray-500 uppercase block mb-1">Meaning</span>
                            <p className="text-sm text-gray-700 dark:text-gray-300 leading-snug">
                                {explanation.meaning}
                            </p>
                        </div>

                        <div>
                            <span className="text-xs font-medium text-gray-500 uppercase block mb-1">Calculation</span>
                            <div className="text-sm text-gray-600 dark:text-gray-400 leading-snug bg-gray-50 dark:bg-gray-900/50 p-2 rounded-lg text-xs">
                                {explanation.calculation}
                            </div>
                        </div>

                        <div className="flex gap-4 border-t border-gray-100 dark:border-gray-700 pt-2">
                            <div>
                                <span className="text-[10px] mobile:text-xs font-medium text-gray-500 uppercase block mb-0.5">Unit</span>
                                <span className="text-xs text-gray-900 dark:text-white font-medium">{explanation.unit}</span>
                            </div>
                            <div>
                                <span className="text-[10px] mobile:text-xs font-medium text-gray-500 uppercase block mb-0.5">Range</span>
                                <span className="text-xs text-gray-900 dark:text-white font-medium">{explanation.range}</span>
                            </div>
                        </div>
                    </div>

                    {/* Arrow */}
                    <div className="absolute left-1/2 -translate-x-1/2 top-full w-0 h-0 border-l-[8px] border-l-transparent border-r-[8px] border-r-transparent border-t-[8px] border-t-white dark:border-t-gray-800 drop-shadow-sm"></div>
                </div>
            )}
        </div>
    );
};
