import React from "react";
import {
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { useResilientAsset } from "../../utils/fetchWithRetry";

interface BRollImageProps {
  heading: string;
  subheading?: string;
  visualAssetUrl?: string;
  brandColors: { primary: string; secondary: string; background: string };
  format: "landscape" | "vertical";
  durationInFrames: number;
}

export const BRollImage: React.FC<BRollImageProps> = ({
  heading,
  subheading,
  visualAssetUrl,
  brandColors,
  format,
  durationInFrames,
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

  const cardPositionClass =
    format === "vertical" ? "bottom-24 left-6 right-6" : "bottom-12 left-12 max-w-xl";

  return (
    <div
      className="absolute inset-0 overflow-hidden"
      style={{
        backgroundColor: brandColors.background,
      }}
    >
      {/* Resilient B-Roll Image */}
      {localUrl ? (
        <img
          src={localUrl}
          alt="B-Roll"
          className="w-full h-full object-cover"
          style={{
            transform: `scale(${scale}) translate(${translateX}px, ${translateY}px)`,
          }}
        />
      ) : (
        // Fallback gradient if no asset provided
        <div
          className="w-full h-full opacity-60"
          style={{
            background: `linear-gradient(135deg, ${brandColors.background} 0%, ${brandColors.primary} 100%)`,
          }}
        />
      )}

      {/* Dark overlay gradient from bottom */}
      <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent z-10" />

      {/* Overlay Caption Card */}
      <div
        className={`absolute z-20 p-6 rounded-2xl border border-white/10 bg-black/40 backdrop-blur-md shadow-lg ${cardPositionClass}`}
        style={{
          transform: `translateY(${overlayY}px)`,
          opacity: overlayOpacity,
          borderColor: `${brandColors.secondary}22`,
        }}
      >
        <h3
          className="text-2xl font-bold font-display leading-tight mb-2"
          style={{
            color: brandColors.secondary,
          }}
        >
          {heading}
        </h3>
        {subheading && (
          <p className="text-sm font-medium text-gray-300">
            {subheading}
          </p>
        )}
      </div>
    </div>
  );
};
