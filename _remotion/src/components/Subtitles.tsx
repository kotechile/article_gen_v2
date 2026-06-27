import React from "react";
import {
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { RemotionVideoPayload } from "../types/schema";

interface SubtitlesProps {
  subtitles: RemotionVideoPayload["subtitles"];
  brandColors: { primary: string; secondary: string; background: string };
  format: "landscape" | "vertical";
  captionPosition?: "center" | "bottom" | "top";
}

export const Subtitles: React.FC<SubtitlesProps> = ({
  subtitles,
  brandColors,
  format,
  captionPosition = "center",
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Find the active subtitle item
  const activeSubtitleIndex = subtitles.findIndex(
    (sub) => frame >= sub.startFrame && frame <= sub.endFrame
  );

  const activeSubtitle = subtitles[activeSubtitleIndex];

  // If there's no subtitle active at this frame, render nothing
  if (!activeSubtitle) {
    return null;
  }

  // Calculate frames elapsed since the subtitle word started
  const elapsedFrames = frame - activeSubtitle.startFrame;

  // Kinetic Pop spring animation
  const popSpring = spring({
    frame: elapsedFrames,
    fps,
    config: {
      damping: 10,
      stiffness: 150,
      mass: 0.6,
    },
  });

  // Animate scale (pop out then settle) and rotation for energetic typography
  const scale = interpolate(popSpring, [0, 1], [0.85, 1.15]);
  const rotation = interpolate(popSpring, [0, 1], [-3, 0]); // slight tilt
  const opacity = interpolate(popSpring, [0, 0.5, 1], [0, 0.9, 1]);

  // Determine vertical offsets based on captionPosition option
  const getVerticalOffsets = () => {
    if (format === "vertical") {
      if (captionPosition === "top") return { top: "12%", bottom: "58%" };
      if (captionPosition === "bottom") return { top: "58%", bottom: "12%" };
      return { top: "35%", bottom: "35%" }; // default center 30% safe zone
    } else {
      if (captionPosition === "top") return { top: "10%" };
      if (captionPosition === "center") return { top: "45%" };
      return { bottom: "10%" }; // default bottom
    }
  };

  const containerStyle: React.CSSProperties = {
    position: "absolute",
    left: format === "vertical" ? "5%" : "10%",
    right: format === "vertical" ? "5%" : "10%",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    pointerEvents: "none",
    zIndex: 50,
    ...getVerticalOffsets(),
  };

  const textClass = format === "vertical" ? "text-5xl" : "text-4xl";

  return (
    <div style={containerStyle}>
      <div
        className="flex flex-col items-center"
        style={{
          transform: `scale(${scale}) rotate(${rotation}deg)`,
          opacity,
        }}
      >
        <span
          className={`${textClass} font-black uppercase font-display tracking-tight text-center px-6 py-3 rounded-2xl shadow-xl`}
          style={{
            backgroundColor: "rgba(0, 0, 0, 0.85)",
            border: `2px solid ${brandColors.secondary}`,
            color: "#ffffff",
            textShadow: `0 0 10px ${brandColors.secondary}66`,
          }}
        >
          {activeSubtitle.text}
        </span>
      </div>
    </div>
  );
};
