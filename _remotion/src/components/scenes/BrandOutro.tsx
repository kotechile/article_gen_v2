import React from "react";
import {
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
  AbsoluteFill,
  staticFile,
} from "remotion";

interface BrandOutroProps {
  brandColors: { primary: string; secondary: string; background: string };
  format: "landscape" | "vertical";
  durationInFrames: number;
}

export const BrandOutro: React.FC<BrandOutroProps> = ({
  brandColors,
  format,
  durationInFrames,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Entrance spring for logo
  const entranceSpring = spring({
    frame,
    fps,
    config: {
      damping: 15,
      mass: 0.7,
      stiffness: 100,
    },
  });

  const logoScale = interpolate(entranceSpring, [0, 1], [0.3, 1]);
  const logoOpacity = interpolate(entranceSpring, [0, 1], [0, 1]);

  // Gentle pulse after entrance completes
  const pulseFactor = interpolate(
    Math.sin((Math.max(0, frame - 15) / 45) * Math.PI * 2),
    [-1, 1],
    [0.98, 1.02]
  );

  return (
    <AbsoluteFill
      style={{
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        backgroundColor: brandColors.background,
        overflow: "hidden",
      }}
    >
      {/* Glowing background radial flare */}
      <div
        style={{
          position: "absolute",
          borderRadius: "50%",
          width: format === "vertical" ? "350px" : "550px",
          height: format === "vertical" ? "350px" : "550px",
          background: `radial-gradient(circle, ${brandColors.primary}22 0%, transparent 70%)`,
          zIndex: 0,
        }}
      />

      {/* Pulsing logo asset */}
      <div
        style={{
          position: "relative",
          zIndex: 10,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          opacity: logoOpacity,
          transform: `scale(${logoScale * pulseFactor})`,
        }}
      >
        <img
          src={staticFile("gini_loh_logo.jpg")}
          alt="Gini Loh Logo"
          style={{
            width: format === "vertical" ? "180px" : "240px",
            height: format === "vertical" ? "180px" : "240px",
            borderRadius: "40px",
            boxShadow: `0 15px 50px rgba(0,0,0,0.6), 0 0 30px ${brandColors.secondary}22`,
            border: `3px solid ${brandColors.secondary}55`,
          }}
        />
      </div>
    </AbsoluteFill>
  );
};
export default BrandOutro;
