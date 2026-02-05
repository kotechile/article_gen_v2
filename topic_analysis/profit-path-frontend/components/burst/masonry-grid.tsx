import * as React from "react"
import { cn } from "@/lib/utils"

interface MasonryGridProps {
    children: React.ReactNode
    className?: string
}

export function MasonryGrid({ children, className }: MasonryGridProps) {
    return (
        <div className={cn("columns-1 md:columns-2 lg:columns-3 gap-4 space-y-4", className)}>
            {children}
        </div>
    )
}
