import { cn } from "@/lib/utils"

interface ZenithLogoProps {
  compact?: boolean
  className?: string
}

export function ZenithLogo({ compact = false, className }: ZenithLogoProps) {
  if (compact) {
    return (
      <div className={cn("rounded-2xl bg-[#111633] px-3 py-2 text-center", className)}>
        <div className="text-lg font-semibold tracking-tight text-[#9489ff]">Zenith</div>
        <div className="text-[9px] uppercase tracking-[0.28em] text-slate-500">Center</div>
      </div>
    )
  }

  return (
    <div className={cn("rounded-[28px] bg-[#111633] px-6 py-5", className)}>
      <p className="text-[42px] font-semibold leading-none tracking-tight text-[#9489ff] sm:text-[56px]">
        Zenith Creator
      </p>
      <p className="mt-2 text-[18px] uppercase tracking-[0.34em] text-slate-500 sm:text-[22px]">
        Command Center
      </p>
    </div>
  )
}
