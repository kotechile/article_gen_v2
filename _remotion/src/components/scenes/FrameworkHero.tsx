import React from "react";
import {
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
  AbsoluteFill,
} from "remotion";
import { useResilientAsset } from "../../utils/fetchWithRetry";

interface FrameworkHeroProps {
  heading: string;
  subheading?: string;
  visualAssetUrl?: string;
  brandColors: { primary: string; secondary: string; background: string };
  format: "landscape" | "vertical";
  durationInFrames: number;
}

export const FrameworkHero: React.FC<FrameworkHeroProps> = ({
  heading,
  subheading,
  visualAssetUrl,
  brandColors,
  format,
  durationInFrames,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Resilience: Preload the background image
  const { localUrl } = useResilientAsset(visualAssetUrl);

  // Smooth entrance using spring
  const entranceSpring = spring({
    frame,
    fps,
    config: {
      damping: 14,
      mass: 0.8,
      stiffness: 90,
    },
  });

  // Slide up and fade in for heading
  const textTranslateY = interpolate(entranceSpring, [0, 1], [40, 0]);
  const textOpacity = interpolate(entranceSpring, [0, 1], [0, 1]);

  // Subtle zoom/pan effect for background image
  const bgScale = interpolate(frame, [0, durationInFrames], [1.05, 1.15], {
    extrapolateRight: "clamp",
  });


  return (
    <AbsoluteFill
      style={{
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        overflow: "hidden",
        padding: "32px",
        textAlign: "center",
        backgroundColor: brandColors.background,
        color: "#ffffff",
      }}
    >
      {/* Background Image with overlay */}
      {localUrl && (
        <div style={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0, zIndex: 0 }}>
          <img
            src={localUrl}
            alt="background"
            style={{
              width: "100%",
              height: "100%",
              objectFit: "cover",
              transform: `scale(${bgScale})`,
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
              backgroundColor: "rgba(0,0,0,0.6)",
              backdropFilter: "blur(2px)",
            }}
          />
        </div>
      )}

      {/* Hero Content */}
      <div
        style={{
          position: "relative",
          zIndex: 10,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          maxWidth: "896px",
          transform: `translateY(${textTranslateY}px)`,
          opacity: textOpacity,
        }}
      >
        <h1
          style={{
            fontSize: format === "vertical" ? "48px" : "64px",
            fontWeight: 800,
            fontFamily: "system-ui, -apple-system, sans-serif",
            letterSpacing: "-0.02em",
            marginBottom: "16px",
            color: brandColors.secondary,
            textShadow: "0 4px 20px rgba(0,0,0,0.6)",
            margin: 0,
          }}
        >
          {heading}
        </h1>

        {subheading && (
          <p
            style={{
              fontSize: format === "vertical" ? "20px" : "28px",
              color: "#d1d5db",
              fontWeight: 500,
              maxWidth: "672px",
              fontFamily: "system-ui, -apple-system, sans-serif",
              margin: 0,
              marginTop: "8px",
            }}
          >
            {subheading}
          </p>
        )}

        {/* Decorative dynamic bar */}
        <div
          style={{
            marginTop: "24px",
            height: "4px",
            width: "96px",
            borderRadius: "9999px",
            backgroundImage: `linear-gradient(to right, ${brandColors.primary}, ${brandColors.secondary})`,
            transform: `scaleX(${entranceSpring})`,
          }}
        />
      </div>
    </AbsoluteFill>
  );
};
