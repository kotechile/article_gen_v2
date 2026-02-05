import React from 'react';

import { MetricTooltip } from './Tooltip';
import type { MetricExplanation } from '../types/metrics';

interface GaugeProps {
    value: number; // 0 to 100
    label: string;
    size?: number;
    color?: string;
    explanation?: MetricExplanation;
    unit?: string;
    displayValue?: string | number;
}

export const Gauge: React.FC<GaugeProps> = ({ value, label, size = 120, color = 'text-indigo-600', explanation, unit = '%', displayValue }) => {
    // Ensure value is between 0 and 100
    const cleanValue = Math.min(Math.max(value, 0), 100);

    // SVG Geometry
    const strokeWidth = 10;
    const radius = size / 2 - strokeWidth;
    const circumference = Math.PI * radius; // Semi-circle
    const dashOffset = probs_to_dash(cleanValue, circumference);

    // Color mapping based on value if no color provided
    const getColor = (val: number) => {
        if (color !== 'text-indigo-600') return color;
        if (val >= 80) return 'text-green-500';
        if (val >= 50) return 'text-yellow-500';
        return 'text-red-500';
    };

    const finalColor = getColor(cleanValue);

    return (
        <div className="flex flex-col items-center justify-center p-4 bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-100 dark:border-gray-700">
            <div className="relative flex items-center justify-center" style={{ width: size, height: size / 1.6 }}>
                <svg className="transform" width={size} height={size / 2 + strokeWidth} viewBox={`0 0 ${size} ${size / 2 + strokeWidth}`}>
                    {/* Background Track */}
                    <path
                        className="text-gray-200 dark:text-gray-700 stroke-current"
                        strokeWidth={strokeWidth}
                        strokeLinecap="round"
                        fill="none"
                        d={`M${strokeWidth} ${size / 2} A${radius} ${radius} 0 0 1 ${size - strokeWidth} ${size / 2}`}
                    />
                    {/* Value Arc */}
                    <path
                        className={`${finalColor} stroke-current transition-all duration-1000 ease-out`}
                        strokeWidth={strokeWidth}
                        strokeLinecap="round"
                        fill="none"
                        strokeDasharray={circumference}
                        strokeDashoffset={dashOffset}
                        d={`M${strokeWidth} ${size / 2} A${radius} ${radius} 0 0 1 ${size - strokeWidth} ${size / 2}`}
                    />
                </svg>
                <div className="absolute bottom-0 text-center transform translate-y-1">
                    <span className={`text-2xl font-bold ${finalColor}`}>{displayValue !== undefined ? displayValue : cleanValue}{unit}</span>
                </div>
            </div>

            <div className="flex items-center gap-1 mt-2">
                <span className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">{label}</span>
                {explanation && (
                    <MetricTooltip explanation={explanation} />
                )}
            </div>
        </div>
    );
};

function probs_to_dash(val: number, circumference: number) {
    // val is 0-100
    // 0% -> full offset (hidden)
    // 100% -> 0 offset (fully visible)
    // But since it's a dasharray of circumference, we want:
    return circumference * (1 - val / 100);
}
