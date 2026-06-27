import React from "react";
import { Series } from "remotion";
import { RemotionVideoPayload } from "./types/schema";
import { getAdjustedScenes } from "./utils/timing";
import { FrameworkHero } from "./components/scenes/FrameworkHero";
import { KpiMetric } from "./components/scenes/KpiMetric";
import { ComparisonTable } from "./components/scenes/ComparisonTable";
import { BRollImage } from "./components/scenes/BRollImage";
import { Subtitles } from "./components/Subtitles";
import { AudioTracks } from "./components/AudioTracks";

export const VideoMain: React.FC<RemotionVideoPayload> = ({
  metadata,
  audioTrackUrl,
  subtitles,
  scenes,
}) => {
  const { format, brandColors, totalDurationInSeconds, captionPosition } = metadata;
  const fps = 30;
  const totalDurationInFrames = Math.round(totalDurationInSeconds * fps);

  // Mismatch Enforcement: adjust scene lengths to align exactly with total voiceover audio duration
  const adjustedScenes = getAdjustedScenes(scenes, totalDurationInFrames);

  return (
    <div
      className="w-full h-full relative font-sans overflow-hidden"
      style={{
        backgroundColor: brandColors.background,
      }}
    >
      {/* Audio Layer with ducking */}
      <AudioTracks voiceoverUrl={audioTrackUrl} subtitles={subtitles} />

      {/* Video Scenes Layer */}
      <Series>
        {adjustedScenes.map((scene) => {
          const { sceneId, type, durationInFrames, heading, subheading, visualAssetUrl, tableData, kpiData } = scene;

          return (
            <Series.Sequence
              key={sceneId}
              durationInFrames={durationInFrames}
              layout="absolute-fill"
            >
              {type === "framework_hero" && (
                <FrameworkHero
                  heading={heading}
                  subheading={subheading}
                  visualAssetUrl={visualAssetUrl}
                  brandColors={brandColors}
                  format={format}
                  durationInFrames={durationInFrames}
                />
              )}
              {type === "kpi_metric" && (
                <KpiMetric
                  heading={heading}
                  kpiData={kpiData}
                  brandColors={brandColors}
                  format={format}
                />
              )}
              {type === "comparison_table" && (
                <ComparisonTable
                  heading={heading}
                  tableData={tableData}
                  brandColors={brandColors}
                  format={format}
                  durationInFrames={durationInFrames}
                />
              )}
              {type === "broll_image" && (
                <BRollImage
                  heading={heading}
                  subheading={subheading}
                  visualAssetUrl={visualAssetUrl}
                  brandColors={brandColors}
                  format={format}
                  durationInFrames={durationInFrames}
                />
              )}
            </Series.Sequence>
          );
        })}
      </Series>

      {/* Global Kinetic Typography / Subtitles Overlay */}
      <Subtitles
        subtitles={subtitles}
        brandColors={brandColors}
        format={format}
        captionPosition={captionPosition}
      />
    </div>
  );
};
