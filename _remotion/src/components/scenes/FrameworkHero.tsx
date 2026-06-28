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

  const headingSize = format === "vertical" ? "text-5xl" : "text-7xl";
  const subheadingSize = format === "vertical" ? "text-xl" : "text-3xl";

  return (
    <AbsoluteFill
      className="flex flex-col justify-center items-center overflow-hidden px-8 text-center"
      style={{
        backgroundColor: brandColors.background,
        color: "#ffffff",
      }}
    >
      {/* Background Image with overlay */}
      {localUrl && (
        <div className="absolute inset-0 z-0">
          <img
            src={localUrl}
            alt="background"
            className="w-full h-full object-cover"
            style={{
              transform: `scale(${bgScale})`,
            }}
          />
          {/* Overlay to guarantee contrast */}
          <div className="absolute inset-0 bg-black/60 backdrop-blur-[2px]" />
        </div>
      )}

      {/* Hero Content */}
      <div
        className="relative z-10 flex flex-col items-center max-w-4xl"
        style={{
          transform: `translateY(${textTranslateY}px)`,
          opacity: textOpacity,
        }}
      >
        <h1
          className={`${headingSize} font-extrabold tracking-tight font-display mb-4`}
          style={{
            color: brandColors.secondary,
            textShadow: "0 4px 20px rgba(0,0,0,0.6)",
          }}
        >
          {heading}
        </h1>

        {subheading && (
          <p className={`${subheadingSize} text-gray-300 font-medium max-w-2xl`}>
            {subheading}
          </p>
        )}

        {/* Decorative dynamic bar */}
        <div
          className="mt-6 h-1 w-24 rounded-full"
          style={{
            backgroundImage: `linear-gradient(to right, ${brandColors.primary}, ${brandColors.secondary})`,
            transform: `scaleX(${entranceSpring})`,
          }}
        />
      </div>
    </AbsoluteFill>
  );
};
