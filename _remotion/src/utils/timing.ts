import { RemotionVideoPayload } from "../types/schema";

/**
 * Adjusts the duration of the last scene in the payload so that the total sum of
 * scenes.durationInFrames matches the total composition duration (calculated from audio).
 *
 * If the adjusted duration of the last scene would be <= 0, we log a warning and clamp
 * it to a minimum of 30 frames (1 second), adjusting other scenes if absolutely necessary.
 */
function isVideoFile(url?: string): boolean {
  if (!url) return false;
  const lower = url.toLowerCase();
  return lower.endsWith('.mp4') || lower.endsWith('.mov') || lower.endsWith('.webm') || lower.endsWith('.m4v') || lower.endsWith('.qt');
}

export function getAdjustedScenes(
  scenes: RemotionVideoPayload["scenes"],
  totalDurationInFrames: number
): RemotionVideoPayload["scenes"] {
  // Sanitization: Convert any video_clip scene that does not point to a video file into broll_image
  const sanitizedScenes = scenes.map((scene) => {
    if (scene.type === "video_clip" && !isVideoFile(scene.visualAssetUrl)) {
      console.warn(`⚠️ Corrected type of scene ${scene.sceneId} from video_clip to broll_image because asset is an image: ${scene.visualAssetUrl}`);
      return {
        ...scene,
        type: "broll_image" as const
      };
    }
    return scene;
  });

  if (sanitizedScenes.length === 0) return [];

  // If there's only one scene, it takes the whole duration
  if (sanitizedScenes.length === 1) {
    return [
      {
        ...sanitizedScenes[0],
        durationInFrames: totalDurationInFrames,
      },
    ];
  }

  const otherScenesSum = sanitizedScenes
    .slice(0, -1)
    .reduce((sum, s) => sum + s.durationInFrames, 0);

  const adjustedLastSceneDuration = totalDurationInFrames - otherScenesSum;

  const newScenes = [...sanitizedScenes];
  const lastIndex = newScenes.length - 1;

  if (adjustedLastSceneDuration > 0) {
    newScenes[lastIndex] = {
      ...newScenes[lastIndex],
      durationInFrames: adjustedLastSceneDuration,
    };
  } else {
    console.warn(
      `Audio duration (${totalDurationInFrames}f) is shorter than scenes prefix sum (${otherScenesSum}f).` +
        ` Clamping last scene to 30 frames (1s) to avoid rendering issues.`
    );
    newScenes[lastIndex] = {
      ...newScenes[lastIndex],
      durationInFrames: 30,
    };
  }

  return newScenes;
}

/**
 * Calculates start frames for each scene in the sequence.
 */
export function getSceneStartFrames(scenes: RemotionVideoPayload["scenes"]): number[] {
  const starts: number[] = [];
  let current = 0;
  for (const scene of scenes) {
    starts.push(current);
    current += scene.durationInFrames;
  }
  return starts;
}
