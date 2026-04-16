import { cn } from "@/lib/utils"

interface ZenithLogoProps {
  compact?: boolean
  className?: string
}

export function ZenithLogo({ compact = false, className }: ZenithLogoProps) {
  if (compact) {
    return (
      <div className={cn("rounded-xl bg-[#111633] px-2.5 py-2 text-center", className)}>
        <div className="text-base font-semibold tracking-tight text-[#9489ff]">Zenith</div>
      </div>
    )
  }

  return (
    <div className={cn("flex items-baseline gap-2", className)}>
      <span className="text-lg font-semibold tracking-tight text-[#9489ff]">Zenith</span>
      <span className="text-[11px] uppercase tracking-[0.2em] text-slate-500">Research</span>
    </div>
  )
}