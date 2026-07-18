import React from "react";
import { Series, AbsoluteFill } from "remotion";
import { RemotionVideoPayload } from "./types/schema";
import { getAdjustedScenes } from "./utils/timing";
import { FrameworkHero } from "./components/scenes/FrameworkHero";
import { KpiMetric } from "./components/scenes/KpiMetric";
import { ComparisonTable } from "./components/scenes/ComparisonTable";
import { BRollImage } from "./components/scenes/BRollImage";
import { CallToAction } from "./components/scenes/CallToAction";
import { VideoClip } from "./components/scenes/VideoClip";
import { BrandOutro } from "./components/scenes/BrandOutro";
import { Subtitles } from "./components/Subtitles";
import { AudioTracks } from "./components/AudioTracks";

export const VideoMain: React.FC<RemotionVideoPayload> = ({
  metadata,
  audioTrackUrl,
  subtitles,
  scenes,
}) => {
  const { format, brandColors, totalDurationInSeconds, captionPosition, brandLogoUrl } = metadata;
  const fps = 30;
  const totalDurationInFrames = Math.round(totalDurationInSeconds * fps);

  // Mismatch Enforcement: adjust scene lengths to align exactly with total voiceover audio duration
  const adjustedScenes = getAdjustedScenes(scenes, totalDurationInFrames);

  return (
    <AbsoluteFill
      className="font-sans overflow-hidden"
      style={{
        backgroundColor: brandColors.background,
      }}
    >
      {/* Audio Layer with ducking */}
      <AudioTracks 
        voiceoverUrl={audioTrackUrl} 
        subtitles={subtitles} 
        backgroundMusicUrl={metadata.backgroundMusicUrl} 
      />

      {/* Video Scenes Layer */}
      <Series>
        {adjustedScenes.map((scene) => {
          const { sceneId, type, durationInFrames, heading, subheading, visualAssetUrl, tableData, kpiData, imageSizePercent, imageValign, imageHalign } = scene;

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
                  imageSizePercent={imageSizePercent}
                  imageValign={imageValign}
                  imageHalign={imageHalign}
                />
              )}
              {type === "kpi_metric" && (
                <KpiMetric
                  heading={heading}
                  kpiData={kpiData}
                  brandColors={brandColors}
                  format={format}
                  visualAssetUrl={visualAssetUrl}
                  imageSizePercent={imageSizePercent}
                  imageValign={imageValign}
                  imageHalign={imageHalign}
                />
              )}
              {type === "comparison_table" && (
                <ComparisonTable
                  heading={heading}
                  tableData={tableData}
                  brandColors={brandColors}
                  format={format}
                  durationInFrames={durationInFrames}
                  visualAssetUrl={visualAssetUrl}
                  imageSizePercent={imageSizePercent}
                  imageValign={imageValign}
                  imageHalign={imageHalign}
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
                  imageSizePercent={imageSizePercent}
                  imageValign={imageValign}
                  imageHalign={imageHalign}
                />
              )}
              {type === "call_to_action" && (
                <CallToAction
                  heading={heading}
                  subheading={subheading}
                  visualAssetUrl={visualAssetUrl}
                  brandColors={brandColors}
                  format={format}
                  durationInFrames={durationInFrames}
                  imageSizePercent={imageSizePercent}
                  imageValign={imageValign}
                  imageHalign={imageHalign}
                  brandLogoUrl={brandLogoUrl}
                />
              )}
              {type === "video_clip" && (
                <VideoClip
                  heading={heading}
                  subheading={subheading}
                  visualAssetUrl={visualAssetUrl}
                  brandColors={brandColors}
                  format={format}
                  durationInFrames={durationInFrames}
                  imageSizePercent={imageSizePercent}
                  imageValign={imageValign}
                  imageHalign={imageHalign}
                  hasBackgroundMusic={Boolean(metadata.backgroundMusicUrl && metadata.backgroundMusicUrl.trim())}
                />
              )}
              {type === "outro_logo" && (
                <BrandOutro
                   brandColors={brandColors}
                   format={format}
                   durationInFrames={durationInFrames}
                   brandLogoUrl={brandLogoUrl}
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
    </AbsoluteFill>
  );
};
