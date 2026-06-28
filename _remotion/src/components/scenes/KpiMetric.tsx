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
  imageSizePercent?: number;
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
  imageSizePercent,
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


  return (
    <AbsoluteFill
      style={{
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        padding: "32px",
        backgroundColor: brandColors.background,
        color: "#ffffff",
      }}
    >
      {/* Background Image with overlay */}
      {localUrl && (
        <AbsoluteFill style={{ zIndex: 0, display: "flex", justifyContent: "center", alignItems: "center" }}>
          <img
            src={localUrl}
            alt="background"
            style={{
              width: `${imageSizePercent || 100}%`,
              height: `${imageSizePercent || 100}%`,
              objectFit: (imageSizePercent && imageSizePercent < 100) ? "contain" : "cover",
            }}
          />
          {/* Overlay to guarantee contrast */}
          <div
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundColor: "rgba(0,0,0,0.7)",
              backdropFilter: "blur(1px)",
              zIndex: 1,
            }}
          />
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
        style={{
          width: "100%",
          maxWidth: "672px",
          padding: "48px 32px",
          borderRadius: "24px",
          border: "1px solid rgba(255, 255, 255, 0.1)",
          backgroundColor: "rgba(255, 255, 255, 0.05)",
          backdropFilter: "blur(12px)",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          textAlign: "center",
          boxShadow: "0 20px 50px rgba(0,0,0,0.5)",
          transform: `scale(${scale})`,
          opacity,
          borderColor: `${brandColors.primary}33`,
          zIndex: 10,
        }}
      >
        {/* Metric Label */}
        <span
          style={{
            fontSize: format === "vertical" ? "20px" : "24px",
            fontWeight: "bold",
            fontFamily: "system-ui, -apple-system, sans-serif",
            textTransform: "uppercase",
            letterSpacing: "0.1em",
            color: "#9ca3af",
            marginBottom: "24px",
            margin: 0,
          }}
        >
          {labelText}
        </span>

        {/* Big Animated Metric Value */}
        <span
          style={{
            fontSize: format === "vertical" ? "72px" : "96px",
            fontWeight: 800,
            fontFamily: "system-ui, -apple-system, sans-serif",
            letterSpacing: "-0.04em",
            color: brandColors.secondary,
            textShadow: "0 0 15px rgba(0,255,255,0.4)",
            marginBottom: "16px",
            margin: 0,
          }}
        >
          {formattedValue}
        </span>

        {/* Supporting Heading */}
        <h3
          style={{
            fontSize: format === "vertical" ? "28px" : "36px",
            fontWeight: "bold",
            fontFamily: "system-ui, -apple-system, sans-serif",
            lineHeight: 1.2,
            maxWidth: "512px",
            margin: 0,
            marginTop: "8px",
          }}
        >
          {heading}
        </h3>

        {/* Highlight line */}
        <div
          style={{
            marginTop: "32px",
            height: "6px",
            width: "128px",
            borderRadius: "9999px",
            backgroundImage: `linear-gradient(to right, ${brandColors.primary}, ${brandColors.secondary})`,
          }}
        />
      </div>
    </AbsoluteFill>
  );
};
