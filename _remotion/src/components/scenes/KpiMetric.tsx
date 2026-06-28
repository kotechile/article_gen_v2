import React from "react";
import {
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
  AbsoluteFill,
} from "remotion";
import { useResilientAsset } from "../../utils/fetchWithRetry";

interface KpiMetricProps {
  heading: string;
  kpiData?: {
    value: string;
    label: string;
  };
  brandColors: { primary: string; secondary: string; background: string };
  format: "landscape" | "vertical";
  visualAssetUrl?: string;
}

/**
 * Parses a string like "+250.5% MoM" or "1.2M" into a prefix, number, decimal precision, and suffix
 */
function parseNumericKpi(kpiString: string) {
  const clean = kpiString.trim();
  const match = clean.match(/^([^\d-]*)(-?\d+(?:\.\d+)?)([^\d]*)$/);
  if (!match) return { prefix: "", numericValue: 100, decimals: 0, suffix: "" };
  
  const prefix = match[1];
  const valueStr = match[2];
  const suffix = match[3];
  
  const numericValue = parseFloat(valueStr);
  const dotIndex = valueStr.indexOf(".");
  const decimals = dotIndex === -1 ? 0 : valueStr.length - dotIndex - 1;
  
  return { prefix, numericValue, decimals, suffix };
}

export const KpiMetric: React.FC<KpiMetricProps> = ({
  heading,
  kpiData,
  brandColors,
  format,
  visualAssetUrl,
}) => {
  const frame = useCurrentFrame();
  const { localUrl } = useResilientAsset(visualAssetUrl);
  const { fps } = useVideoConfig();

  const labelText = kpiData?.label || "";
  const valueText = kpiData?.value || "100";
  const { prefix, numericValue, decimals, suffix } = parseNumericKpi(valueText);

  // Entrance spring for the metric card
  const cardEntrance = spring({
    frame,
    fps,
    config: {
      damping: 14,
      mass: 0.8,
      stiffness: 90,
    },
  });

  // Numbers count-up animation
  const tickerSpring = spring({
    frame,
    fps,
    config: {
      damping: 18,
      mass: 1.2,
      stiffness: 70,
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
    <AbsoluteFill
      className="flex flex-col justify-center items-center px-8"
      style={{
        backgroundColor: brandColors.background,
        color: "#ffffff",
      }}
    >
      {/* Background Image with overlay */}
      {localUrl && (
        <AbsoluteFill className="z-0">
          <img
            src={localUrl}
            alt="background"
            className="w-full h-full object-cover"
          />
          {/* Overlay to guarantee contrast */}
          <div className="absolute inset-0 bg-black/70 backdrop-blur-[1px]" />
        </AbsoluteFill>
      )}

      {/* Background ambient glow */}
      <div
        className="absolute rounded-full opacity-20 blur-[120px] w-96 h-96 z-0"
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
        {/* Metric Label */}
        <span
          className={`${labelSize} font-bold font-display uppercase tracking-widest text-gray-400 mb-6`}
        >
          {labelText}
        </span>

        {/* Big Animated Metric Value */}
        <span
          className={`${valueSize} font-extrabold tracking-tighter font-display drop-shadow-[0_0_15px_rgba(0,255,255,0.4)] mb-4`}
          style={{
            color: brandColors.secondary,
          }}
        >
          {formattedValue}
        </span>

        {/* Supporting Heading */}
        <h3 className={`${headingSize} font-bold leading-snug max-w-lg mt-2`}>
          {heading}
        </h3>

        {/* Highlight line */}
        <div
          className="mt-8 h-1.5 w-32 rounded-full"
          style={{
            backgroundImage: `linear-gradient(to right, ${brandColors.primary}, ${brandColors.secondary})`,
          }}
        />
      </div>
    </AbsoluteFill>
  );
};
