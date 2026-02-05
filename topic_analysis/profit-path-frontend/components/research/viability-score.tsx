"use client"

import * as React from "react"

interface ViabilityScoreProps {
    score: number;
    size?: number;
}

export function ViabilityScore({ score, size = 60 }: ViabilityScoreProps) {
    const radius = (size - 8) / 2
    const circumference = 2 * Math.PI * radius
    const offset = circumference - (score / 100) * circumference

    // Determine color based on score
    const getColor = () => {
        if (score >= 70) return 'text-green-500'
        if (score >= 50) return 'text-blue-500'
        return 'text-orange-500'
    }

    const getStrokeColor = () => {
        if (score >= 70) return 'stroke-green-500'
        if (score >= 50) return 'stroke-blue-500'
        return 'stroke-orange-500'
    }

    return (
        <div className="relative inline-flex items-center justify-center">
            <svg width={size} height={size} className="transform -rotate-90">
                {/* Background circle */}
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    className="stroke-muted"
                    strokeWidth="4"
                    fill="none"
                />
                {/* Progress circle */}
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    className={getStrokeColor()}
                    strokeWidth="4"
                    fill="none"
                    strokeDasharray={circumference}
                    strokeDashoffset={offset}
                    strokeLinecap="round"
                />
            </svg>
            <div className={`absolute inset-0 flex items-center justify-center text-sm font-bold ${getColor()}`}>
                {score}
            </div>
        </div>
    )
}
