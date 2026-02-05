"use client"

import * as React from "react"

interface SparklineChartProps {
    data: Array<{ date: string; value: number }>;
    direction?: 'up' | 'down' | 'neutral';
    width?: number;
    height?: number;
}

export function SparklineChart({
    data,
    direction = 'neutral',
    width = 100,
    height = 30
}: SparklineChartProps) {
    if (!data || data.length === 0) {
        return <div className="w-[100px] h-[30px] bg-muted/20 rounded" />
    }

    // Calculate min and max for scaling
    const values = data.map(d => d.value)
    const min = Math.min(...values)
    const max = Math.max(...values)
    const range = max - min || 1

    // Generate SVG path
    const points = data.map((d, i) => {
        const x = (i / (data.length - 1)) * width
        const y = height - ((d.value - min) / range) * height
        return `${x},${y}`
    })

    const pathData = `M ${points.join(' L ')}`

    // Determine color based on direction
    const color = direction === 'up'
        ? 'stroke-green-500'
        : direction === 'down'
            ? 'stroke-red-500'
            : 'stroke-blue-500'

    return (
        <svg
            width={width}
            height={height}
            className="inline-block"
            viewBox={`0 0 ${width} ${height}`}
        >
            <path
                d={pathData}
                fill="none"
                className={`${color} stroke-2`}
                strokeLinecap="round"
                strokeLinejoin="round"
            />
        </svg>
    )
}
