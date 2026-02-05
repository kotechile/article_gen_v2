"use client"

import * as React from "react"
import { cn } from "@/lib/utils"

interface ViabilityScoreProps {
    score: number // 0 to 100
    className?: string
}

export function ViabilityScore({ score, className }: ViabilityScoreProps) {
    // Color logic based on score
    const getColor = (val: number) => {
        if (val >= 80) return "#10b981" // green-500
        if (val >= 50) return "#f59e0b" // amber-500
        return "#ef4444" // red-500
    }

    const color = getColor(score)

    return (
        <div className={cn("relative flex items-center justify-center w-8 h-8", className)}>
            <div
                className="absolute inset-0 rounded-full"
                style={{
                    background: `conic-gradient(${color} ${score}%, #27272a ${score}%)`, // #27272a is zinc-800
                    maskImage: "radial-gradient(transparent 55%, black 56%)",
                    WebkitMaskImage: "radial-gradient(transparent 55%, black 56%)"
                }}
            />
            <span className="text-[10px] font-medium text-muted-foreground">{score}</span>
        </div>
    )
}
