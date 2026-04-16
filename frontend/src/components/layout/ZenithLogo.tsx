import { cn } from "@/lib/utils"

interface ZenithLogoProps {
  compact?: boolean
  className?: string
  subtitle?: string
}

export function ZenithLogo({ compact = false, className, subtitle }: ZenithLogoProps) {
  if (compact) {
    return (
      <div className={cn("rounded-xl bg-[#111633] px-2.5 py-2 text-center", className)}>
        <div className="text-base font-semibold tracking-tight text-[#9489ff]">Zenith</div>
      </div>
    )
  }

  return (
    <div className={cn("flex flex-col", className)}>
      <span className="text-lg font-semibold tracking-tight text-[#9489ff]">Zenith Creator</span>
      {subtitle && (
        <span className="text-[11px] uppercase tracking-[0.2em] text-slate-500">{subtitle}</span>
      )}
    </div>
  )
}