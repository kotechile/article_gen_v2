"use client"

import * as React from "react"
import { cn } from "@/lib/utils"

interface TrendSparklineProps {
    data: number[] // Expect exactly 12 points for 12 months
    width?: number
    height?: number
    className?: string
}

export function TrendSparkline({ data, width = 120, height = 40, className }: TrendSparklineProps) {
    // Determine trend direction
    const start = data[0]
    const end = data[data.length - 1]
    const isPositive = end >= start

    // Normalize data to fit within height with padding
    const padding = 4
    const min = Math.min(...data)
    const max = Math.max(...data)
    const range = max - min || 1 // Avoid divide by zero

    const points = data.map((val, i) => {
        const x = (i / (data.length - 1)) * width
        const y = height - padding - ((val - min) / range) * (height - 2 * padding)
        return `${x},${y}`
    }).join(" ")

    return (
        <svg
            width={width}
            height={height}
            className={cn("overflow-visible", className)}
        >
            <polyline
                points={points}
                fill="none"
                stroke={isPositive ? "#10b981" : "#ef4444"} // green-500 : red-500
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
            />
        </svg>
    )
}
