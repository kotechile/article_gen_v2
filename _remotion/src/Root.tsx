import "./index.css";
import React from "react";
import { Composition } from "remotion";
import { VideoMain } from "./Video";
import mockPayload from "./mockPayload.json";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      {/* Landscape composition (16:9) */}
      <Composition
        id="landscape"
        component={VideoMain as any}
        fps={30}
        width={1920}
        height={1080}
        defaultProps={mockPayload as any}
        calculateMetadata={({ props }) => {
          const payload = (props || {}) as any;
          const totalDurationInSeconds =
            payload.metadata?.totalDurationInSeconds ?? 15;
          return {
            durationInFrames: Math.round(totalDurationInSeconds * 30),
            props: payload,
          };
        }}
      />

      {/* Vertical composition (9:16) */}
      <Composition
        id="vertical"
        component={VideoMain as any}
        fps={30}
        width={1080}
        height={1920}
        defaultProps={mockPayload as any}
        calculateMetadata={({ props }) => {
          const payload = (props || {}) as any;
          const totalDurationInSeconds =
            payload.metadata?.totalDurationInSeconds ?? 15;
          return {
            durationInFrames: Math.round(totalDurationInSeconds * 30),
            props: payload,
          };
        }}
      />
    </>
  );
};
export default RemotionRoot;
