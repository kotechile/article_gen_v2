import React from "react";
import { Audio, interpolate, useCurrentFrame } from "remotion";
import { useResilientAsset } from "../utils/fetchWithRetry";
import { RemotionVideoPayload } from "../types/schema";

interface AudioTracksProps {
  voiceoverUrl: string;
  subtitles: RemotionVideoPayload["subtitles"];
  backgroundMusicUrl?: string;
}

// Default royalty-free background track url
const DEFAULT_BG_MUSIC_URL =
  "background.mp3";

export const AudioTracks: React.FC<AudioTracksProps> = ({
  voiceoverUrl,
  subtitles,
  backgroundMusicUrl,
}) => {
  const frame = useCurrentFrame();

  // Resilience: Preload both audio assets to guarantee synchronization
  const { localUrl: voiceoverLocalUrl } = useResilientAsset(voiceoverUrl);
  const isNone = !backgroundMusicUrl || backgroundMusicUrl === "none" || backgroundMusicUrl.endsWith("/none");
  const { localUrl: bgLocalUrl } = useResilientAsset(isNone ? undefined : (backgroundMusicUrl || DEFAULT_BG_MUSIC_URL));

  // Lookahead and Release buffers (in frames) for audio ducking
  // 6 frames lookahead (0.2s), 12 frames release (0.4s)
  const lookahead = 6;
  const release = 12;

  // Function to determine volume for the current frame
  const getDuckedVolume = (): number => {
    // Check if speaking in the neighborhood of the current frame
    const isSpeaking = subtitles.some(
      (sub) => frame >= sub.startFrame - lookahead && frame <= sub.endFrame + release
    );

    // If speaking, background music volume is 10% (0.1). Otherwise, 50% (0.5).
    const targetVolume = isSpeaking ? 0.1 : 0.5;

    // Smoothly transition volume using simple interpolation over 5 frames
    // We check the transition boundary
    let volume = targetVolume;
    
    // Find closest boundary to interpolate volume transitions
    let minDistance = 999;

    for (const sub of subtitles) {
      // Transitioning down (before speech)
      const startBoundary = sub.startFrame - lookahead;
      if (frame >= startBoundary - 5 && frame < startBoundary) {
        const dist = frame - (startBoundary - 5); // 0 to 5
        const tVolume = interpolate(dist, [0, 5], [0.5, 0.1]);
        if (dist < minDistance) {
          volume = tVolume;
          minDistance = dist;
        }
      }
      
      // Transitioning up (after speech)
      const endBoundary = sub.endFrame + release;
      if (frame >= endBoundary && frame < endBoundary + 10) {
        const dist = frame - endBoundary; // 0 to 10
        const tVolume = interpolate(dist, [0, 10], [0.1, 0.5]);
        if (dist < minDistance) {
          volume = tVolume;
          minDistance = dist;
        }
      }
    }

    return volume;
  };

  const bgVolume = getDuckedVolume();

  return (
    <>
      {/* Main Voiceover Track */}
      {voiceoverLocalUrl && <Audio src={voiceoverLocalUrl} volume={1.0} />}

      {/* Background Soundtrack Ducked Concurrently */}
      {bgLocalUrl && <Audio src={bgLocalUrl} volume={bgVolume} loop />}
    </>
  );
};
