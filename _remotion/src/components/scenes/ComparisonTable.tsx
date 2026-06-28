import React from "react";
import {
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
  AbsoluteFill,
} from "remotion";
import { useResilientAsset } from "../../utils/fetchWithRetry";

interface ComparisonTableProps {
  heading: string;
  tableData?: {
    headers: string[];
    rows: string[][];
  };
  brandColors: { primary: string; secondary: string; background: string };
  format: "landscape" | "vertical";
  durationInFrames: number;
  visualAssetUrl?: string;
}

export const ComparisonTable: React.FC<ComparisonTableProps> = ({
  heading,
  tableData,
  brandColors,
  format,
  durationInFrames,
  visualAssetUrl,
}) => {
  const frame = useCurrentFrame();
  const { localUrl } = useResilientAsset(visualAssetUrl);
  const { fps } = useVideoConfig();

  const headers = tableData?.headers || [];
  const rows = tableData?.rows || [];
  const numRows = rows.length;

  // Entrance animation for table container
  const tableEntrance = spring({
    frame,
    fps,
    config: {
      damping: 16,
      mass: 0.9,
      stiffness: 90,
    },
  });

  const translateY = interpolate(tableEntrance, [0, 1], [50, 0]);
  const opacity = interpolate(tableEntrance, [0, 1], [0, 1]);

  // Determine active row based on current frame progress
  // First 15 frames are buffer for entrance, last 15 frames are exit buffer
  const activeDuration = durationInFrames - 30;
  const segmentDuration = Math.max(10, Math.floor(activeDuration / Math.max(1, numRows)));

  const getActiveRowIndex = () => {
    if (frame < 15) return 0;
    if (frame >= durationInFrames - 15) return numRows - 1;
    const index = Math.floor((frame - 15) / segmentDuration);
    return Math.min(numRows - 1, Math.max(0, index));
  };

  const activeRowIndex = getActiveRowIndex();

  const headingSize = format === "vertical" ? "text-2xl" : "text-4xl";
  const tableTextSize = format === "vertical" ? "text-xs" : "text-base";
  const paddingCell = format === "vertical" ? "px-2 py-3" : "px-6 py-4";

  return (
    <AbsoluteFill
      className="flex flex-col justify-center items-center px-4 md:px-8 overflow-hidden"
      style={{
        backgroundColor: brandColors.background,
        color: "#ffffff",
      }}
    >
      {/* Background Image with overlay */}
      {localUrl && (
        <AbsoluteFill className="z-0">
          <img
            src={localUrl}
            alt="background"
            className="w-full h-full object-cover"
          />
          {/* Overlay to guarantee contrast */}
          <div className="absolute inset-0 bg-black/75 backdrop-blur-[1px]" />
        </AbsoluteFill>
      )}
      {/* Table Title */}
      <h2
        className={`${headingSize} font-bold font-display text-center mb-8`}
        style={{
          color: brandColors.secondary,
        }}
      >
        {heading}
      </h2>

      {/* Table Container */}
      <div
        className="w-full max-w-4xl overflow-hidden rounded-2xl border border-white/10 bg-white/5 backdrop-blur-md shadow-2xl"
        style={{
          transform: `translateY(${translateY}px)`,
          opacity,
          borderColor: `${brandColors.primary}22`,
        }}
      >
        <table className={`w-full table-fixed border-collapse ${tableTextSize}`}>
          <thead>
            <tr
              style={{
                backgroundColor: `${brandColors.primary}20`,
                borderBottom: "1px solid rgba(255,255,255,0.15)",
              }}
            >
              {headers.map((header, idx) => (
                <th
                  key={idx}
                  className={`${paddingCell} text-left font-display font-bold text-gray-300 uppercase tracking-wider`}
                >
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIdx) => {
              const isActive = rowIdx === activeRowIndex;

              // Apply smooth opacity and scaling to the active row
              const rowBg = isActive
                ? `rgba(255, 255, 255, 0.08)`
                : "transparent";
              const borderLeftColor = isActive
                ? brandColors.secondary
                : "transparent";
              const textWeight = isActive ? "font-bold text-white" : "text-gray-400";
              const scale = isActive ? 1.01 : 1.0;

              return (
                <tr
                  key={rowIdx}
                  className="transition-all duration-300 border-b border-white/5"
                  style={{
                    backgroundColor: rowBg,
                    borderLeft: `4px solid ${borderLeftColor}`,
                    transform: `scale(${scale})`,
                  }}
                >
                  {row.map((cell, cellIdx) => (
                    <td
                      key={cellIdx}
                      className={`${paddingCell} ${textWeight} transition-colors duration-300`}
                    >
                      {cell}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </AbsoluteFill>
  );
};
