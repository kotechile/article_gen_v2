import React from "react";
import {
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
  AbsoluteFill,
} from "remotion";
import { useResilientAsset } from "../../utils/fetchWithRetry";

interface BRollImageProps {
  heading: string;
  subheading?: string;
  visualAssetUrl?: string;
  brandColors: { primary: string; secondary: string; background: string };
  format: "landscape" | "vertical";
  durationInFrames: number;
  imageSizePercent?: number;
}

export const BRollImage: React.FC<BRollImageProps> = ({
  heading,
  subheading,
  visualAssetUrl,
  brandColors,
  format,
  durationInFrames,
  imageSizePercent,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Resilience: Preload the B-Roll image asset
  const { localUrl } = useResilientAsset(visualAssetUrl);

  // Ken Burns zoom effect
  const scale = interpolate(frame, [0, durationInFrames], [1.0, 1.12], {
    extrapolateRight: "clamp",
  });

  // Ken Burns slight pan effect
  const translateX = interpolate(frame, [0, durationInFrames], [0, -15], {
    extrapolateRight: "clamp",
  });
  const translateY = interpolate(frame, [0, durationInFrames], [0, -10], {
    extrapolateRight: "clamp",
  });

  // Title card entrance spring
  const overlayEntrance = spring({
    frame,
    fps,
    config: {
      damping: 15,
      mass: 0.8,
      stiffness: 90,
    },
  });

  const overlayY = interpolate(overlayEntrance, [0, 1], [30, 0]);
  const overlayOpacity = interpolate(overlayEntrance, [0, 1], [0, 1]);

  const cardStyle: React.CSSProperties = {
    position: "absolute",
    zIndex: 20,
    padding: "24px",
    borderRadius: "16px",
    border: "1px solid rgba(255, 255, 255, 0.1)",
    backgroundColor: "rgba(0, 0, 0, 0.4)",
    backdropFilter: "blur(12px)",
    boxShadow: "0 10px 15px -3px rgba(0,0,0,0.3)",
    transform: `translateY(${overlayY}px)`,
    opacity: overlayOpacity,
    borderColor: `${brandColors.secondary}22`,
    bottom: format === "vertical" ? "96px" : "48px",
    left: format === "vertical" ? "24px" : "48px",
    right: format === "vertical" ? "24px" : "auto",
    maxWidth: format === "vertical" ? "auto" : "576px",
  };

  return (
    <AbsoluteFill
      style={{
        backgroundColor: brandColors.background,
        overflow: "hidden",
      }}
    >
      {/* Resilient B-Roll Image */}
      {localUrl ? (
        <div style={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0, display: "flex", justifyContent: "center", alignItems: "center", zIndex: 0 }}>
          <img
            src={localUrl}
            alt="B-Roll"
            style={{
              width: `${imageSizePercent || 100}%`,
              height: `${imageSizePercent || 100}%`,
              objectFit: (imageSizePercent && imageSizePercent < 100) ? "contain" : "cover",
              transform: `scale(${scale}) translate(${translateX}px, ${translateY}px)`,
            }}
          />
        </div>
      ) : (
        // Fallback gradient if no asset provided
        <div
          style={{
            width: "100%",
            height: "100%",
            opacity: 0.6,
            background: `linear-gradient(135deg, ${brandColors.background} 0%, ${brandColors.primary} 100%)`,
          }}
        />
      )}

      {/* Dark overlay gradient from bottom */}
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: "linear-gradient(to top, rgba(0,0,0,0.8) 0%, rgba(0,0,0,0.2) 50%, transparent 100%)",
          zIndex: 10,
        }}
      />

      {/* Overlay Caption Card */}
      <div style={cardStyle}>
        <h3
          style={{
            fontSize: "24px",
            fontWeight: "bold",
            fontFamily: "system-ui, -apple-system, sans-serif",
            lineHeight: 1.2,
            marginBottom: "8px",
            color: brandColors.secondary,
            margin: 0,
          }}
        >
          {heading}
        </h3>
        {subheading && (
          <p
            style={{
              fontSize: "14px",
              fontWeight: 500,
              color: "#d1d5db",
              margin: 0,
              fontFamily: "system-ui, -apple-system, sans-serif",
              marginTop: "4px",
            }}
          >
            {subheading}
          </p>
        )}
      </div>
    </AbsoluteFill>
  );
};
