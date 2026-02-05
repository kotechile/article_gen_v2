"use client"

import * as React from "react"
import { Check } from "lucide-react"
import { cn } from "@/lib/utils"

const Checkbox = React.forwardRef<
    HTMLInputElement,
    React.InputHTMLAttributes<HTMLInputElement>
>(({ className, ...props }, ref) => (
    <div className="relative flex items-center">
        <input
            type="checkbox"
            className="peer h-4 w-4 shrink-0 opacity-0 absolute inset-0 z-10 cursor-pointer"
            ref={ref}
            {...props}
        />
        <div className={cn(
            "flex h-4 w-4 shrink-0 items-center justify-center rounded-sm border border-primary ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground peer-checked:bg-primary peer-checked:text-primary-foreground",
            className
        )}>
            <Check className="h-3 w-3 text-white hidden peer-checked:block" strokeWidth={3} />
        </div>
    </div>
))
Checkbox.displayName = "Checkbox"

export { Checkbox }
