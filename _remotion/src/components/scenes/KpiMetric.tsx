import React from "react";
import {
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

interface KpiMetricProps {
  heading: string;
  kpiData?: {
    value: string;
    label: string;
  };
  brandColors: { primary: string; secondary: string; background: string };
  format: "landscape" | "vertical";
}

/**
 * Parses a string like "+250.5% MoM" or "1.2M" into a prefix, number, decimal precision, and suffix
 */
function parseKpiValue(valStr: string) {
  const cleanVal = valStr.trim();
  const match = cleanVal.match(/^([^\d\.]*)(\d+\.?\d*)(.*)$/);
  if (!match) {
    return { prefix: "", numericValue: 0, suffix: cleanVal, decimals: 0 };
  }
  const prefix = match[1] || "";
  const numericValue = parseFloat(match[2]);
  const suffix = match[3] || "";
  const dotIndex = match[2].indexOf(".");
  const decimals = dotIndex === -1 ? 0 : match[2].length - dotIndex - 1;
  return { prefix, numericValue, suffix, decimals };
}

export const KpiMetric: React.FC<KpiMetricProps> = ({
  heading,
  kpiData,
  brandColors,
  format,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const valueStr = kpiData?.value || "0";
  const label = kpiData?.label || "";

  const { prefix, numericValue, suffix, decimals } = parseKpiValue(valueStr);

  // Entrance animation for the card container
  const cardEntrance = spring({
    frame,
    fps,
    config: {
      damping: 15,
      mass: 0.9,
      stiffness: 100,
    },
  });

  // Ticker animation from 0 -> numericValue
  const tickerSpring = spring({
    frame,
    fps,
    config: {
      damping: 20,
      mass: 1.2,
      stiffness: 80,
    },
  });

  const animatedNumeric = interpolate(tickerSpring, [0, 1], [0, numericValue]);
  const formattedValue = `${prefix}${animatedNumeric.toFixed(decimals)}${suffix}`;

  const scale = interpolate(cardEntrance, [0, 1], [0.8, 1]);
  const opacity = interpolate(cardEntrance, [0, 1], [0, 1]);

  const valueSize = format === "vertical" ? "text-8xl" : "text-9xl";
  const headingSize = format === "vertical" ? "text-3xl" : "text-4xl";
  const labelSize = format === "vertical" ? "text-xl" : "text-2xl";

  return (
    <div
      className="absolute inset-0 flex flex-col justify-center items-center px-8"
      style={{
        backgroundColor: brandColors.background,
        color: "#ffffff",
      }}
    >
      {/* Background ambient glow */}
      <div
        className="absolute rounded-full opacity-20 blur-[120px] w-96 h-96"
        style={{
          background: `radial-gradient(circle, ${brandColors.primary} 0%, ${brandColors.secondary} 100%)`,
          transform: "translate(-10%, -10%)",
        }}
      />

      <div
        className="w-full max-w-2xl px-8 py-12 rounded-3xl border border-white/10 backdrop-blur-md bg-white/5 flex flex-col items-center justify-center text-center shadow-[0_20px_50px_rgba(0,0,0,0.5)]"
        style={{
          transform: `scale(${scale})`,
          opacity,
          borderColor: `${brandColors.primary}33`,
        }}
      >
        <h2 className={`${headingSize} font-bold font-display text-gray-400 mb-6`}>
          {heading}
        </h2>

        {/* Large count-up number */}
        <span
          className={`${valueSize} font-extrabold tracking-tighter font-display drop-shadow-[0_0_15px_rgba(0,255,255,0.4)] mb-4`}
          style={{
            backgroundImage: `linear-gradient(to bottom, #ffffff, #dcdcdc)`,
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            color: "#ffffff",
          }}
        >
          {formattedValue}
        </span>

        {label && (
          <p
            className={`${labelSize} font-semibold`}
            style={{
              color: brandColors.secondary,
            }}
          >
            {label}
          </p>
        )}

        {/* Highlight line */}
        <div
          className="mt-8 h-1.5 w-32 rounded-full"
          style={{
            backgroundImage: `linear-gradient(to right, ${brandColors.primary}, ${brandColors.secondary})`,
          }}
        />
      </div>
    </div>
  );
};
