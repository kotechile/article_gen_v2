import React from "react";
import {
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
  AbsoluteFill,
} from "remotion";
import { useResilientAsset } from "../../utils/fetchWithRetry";

interface CallToActionProps {
  heading: string;
  subheading?: string;
  visualAssetUrl?: string;
  brandColors: { primary: string; secondary: string; background: string };
  format: "landscape" | "vertical";
  durationInFrames: number;
}

export const CallToAction: React.FC<CallToActionProps> = ({
  heading,
  subheading,
  visualAssetUrl,
  brandColors,
  format,
  durationInFrames,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Resilience: Preload background image
  const { localUrl } = useResilientAsset(visualAssetUrl);

  // Entrance spring for CTA elements
  const entranceSpring = spring({
    frame,
    fps,
    config: {
      damping: 14,
      mass: 0.8,
      stiffness: 95,
    },
  });

  const contentTranslateY = interpolate(entranceSpring, [0, 1], [60, 0]);
  const contentOpacity = interpolate(entranceSpring, [0, 1], [0, 1]);

  // Pulse animation for the button
  const pulseFactor = interpolate(
    Math.sin((frame / 30) * Math.PI * 2),
    [-1, 1],
    [1.0, 1.05]
  );


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
            }}
          />
          {/* Dark overlay to focus on call to action */}
          <div
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundColor: "rgba(0,0,0,0.8)",
              backdropFilter: "blur(2px)",
            }}
          />
        </div>
      )}

      {/* Content wrapper */}
      <div
        style={{
          position: "relative",
          zIndex: 10,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          maxWidth: "768px",
          transform: `translateY(${contentTranslateY}px)`,
          opacity: contentOpacity,
        }}
      >
        {/* Glow behind headline */}
        <div
          style={{
            position: "absolute",
            top: "-40px",
            borderRadius: "9999px",
            opacity: 0.2,
            filter: "blur(80px)",
            width: "256px",
            height: "256px",
            background: `radial-gradient(circle, ${brandColors.primary} 0%, ${brandColors.secondary} 100%)`,
            zIndex: 0,
          }}
        />

        <h1
          style={{
            fontSize: format === "vertical" ? "36px" : "56px",
            fontWeight: 800,
            fontFamily: "system-ui, -apple-system, sans-serif",
            letterSpacing: "-0.02em",
            marginBottom: "16px",
            color: brandColors.secondary,
            textShadow: "0 4px 20px rgba(0,0,0,0.6)",
            zIndex: 10,
            margin: 0,
          }}
        >
          {heading}
        </h1>

        {subheading && (
          <p
            style={{
              fontSize: format === "vertical" ? "18px" : "24px",
              color: "#d1d5db",
              fontWeight: 500,
              maxWidth: "576px",
              fontFamily: "system-ui, -apple-system, sans-serif",
              marginBottom: "32px",
              zIndex: 10,
              margin: 0,
              marginTop: "8px",
            }}
          >
            {subheading}
          </p>
        )}

        {/* Dynamic manual Call to Action button */}
        <div
          style={{
            zIndex: 10,
            borderRadius: "9999px",
            padding: "2px",
            boxShadow: `0 0 25px ${brandColors.primary}44`,
            backgroundImage: `linear-gradient(135deg, ${brandColors.primary} 0%, ${brandColors.secondary} 100%)`,
            transform: `scale(${pulseFactor})`,
            transition: "transform 0.3s",
          }}
        >
          <div
            style={{
              padding: format === "vertical" ? "16px 32px" : "20px 48px",
              fontSize: format === "vertical" ? "16px" : "20px",
              borderRadius: "9999px",
              backgroundColor: "rgba(0, 0, 0, 0.9)",
              fontWeight: "bold",
              textTransform: "uppercase",
              letterSpacing: "0.05em",
              color: "#ffffff",
              border: "1px solid transparent",
              textShadow: "0 2px 4px rgba(0,0,0,0.5)",
            }}
          >
            Learn More
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
export default CallToAction;
