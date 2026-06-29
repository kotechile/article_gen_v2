export interface RemotionVideoPayload {
  metadata: {
    title: string;
    format: "landscape" | "vertical";
    totalDurationInSeconds: number;
    brandColors: {
      primary: string;
      secondary: string;
      background: string;
    };
    captionPosition?: "center" | "bottom" | "top" | "none";
    backgroundMusicUrl?: string;
  };
  audioTrackUrl: string; // URL to compiled ElevenLabs voiceover asset
  subtitles: Array<{
    text: string;
    startFrame: number;
    endFrame: number;
  }>;
  scenes: Array<{
    sceneId: string;
    type: "framework_hero" | "comparison_table" | "kpi_metric" | "broll_image" | "call_to_action";
    durationInFrames: number;
    heading: string;
    subheading?: string;
    visualAssetUrl?: string; // Direct link to AI generated b-roll scene
    imageSizePercent?: number;
    imageValign?: "top" | "center" | "bottom";
    imageHalign?: "left" | "center" | "right";
    tableData?: {
      headers: string[];
      rows: string[][];
    };
    kpiData?: {
      value: string;
      label: string;
    };
  }>;
}
