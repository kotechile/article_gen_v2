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

  const headingSize = format === "vertical" ? "text-4xl" : "text-6xl";
  const subheadingSize = format === "vertical" ? "text-lg" : "text-2xl";
  const buttonPadding = format === "vertical" ? "px-8 py-4 text-base" : "px-12 py-5 text-xl";

  return (
    <AbsoluteFill
      className="flex flex-col justify-center items-center px-8 text-center overflow-hidden"
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
          />
          {/* Dark overlay to focus on call to action */}
          <div className="absolute inset-0 bg-black/80 backdrop-blur-[2px]" />
        </div>
      )}

      {/* Content wrapper */}
      <div
        className="relative z-10 flex flex-col items-center max-w-3xl"
        style={{
          transform: `translateY(${contentTranslateY}px)`,
          opacity: contentOpacity,
        }}
      >
        {/* Glow behind headline */}
        <div
          className="absolute -top-10 rounded-full opacity-20 blur-[80px] w-64 h-64 z-0"
          style={{
            background: `radial-gradient(circle, ${brandColors.primary} 0%, ${brandColors.secondary} 100%)`,
          }}
        />

        <h1
          className={`${headingSize} font-extrabold tracking-tight font-display mb-4 z-10`}
          style={{
            color: brandColors.secondary,
            textShadow: "0 4px 20px rgba(0,0,0,0.6)",
          }}
        >
          {heading}
        </h1>

        {subheading && (
          <p className={`${subheadingSize} text-gray-300 font-medium max-w-xl mb-8 z-10`}>
            {subheading}
          </p>
        )}

        {/* Dynamic manual Call to Action button */}
        <div
          className="z-10 rounded-full p-[2px] shadow-2xl transition-transform"
          style={{
            backgroundImage: `linear-gradient(135deg, ${brandColors.primary} 0%, ${brandColors.secondary} 100%)`,
            transform: `scale(${pulseFactor})`,
            boxShadow: `0 0 25px ${brandColors.primary}44`,
          }}
        >
          <div
            className={`${buttonPadding} rounded-full bg-black/90 font-bold uppercase tracking-wider text-white border border-transparent`}
            style={{
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
