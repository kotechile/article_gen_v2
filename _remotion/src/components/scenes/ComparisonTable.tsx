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


  return (
    <AbsoluteFill
      style={{
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        padding: format === "vertical" ? "16px" : "32px",
        overflow: "hidden",
        backgroundColor: brandColors.background,
        color: "#ffffff",
      }}
    >
      {/* Background Image with overlay */}
      {localUrl && (
        <AbsoluteFill style={{ zIndex: 0 }}>
          <img
            src={localUrl}
            alt="background"
            style={{
              width: "100%",
              height: "100%",
              objectFit: "cover",
            }}
          />
          {/* Overlay to guarantee contrast */}
          <div
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundColor: "rgba(0,0,0,0.75)",
              backdropFilter: "blur(1px)",
            }}
          />
        </AbsoluteFill>
      )}

      {/* Table Title */}
      <h2
        style={{
          fontSize: format === "vertical" ? "24px" : "36px",
          fontWeight: "bold",
          fontFamily: "system-ui, -apple-system, sans-serif",
          textAlign: "center",
          marginBottom: "32px",
          color: brandColors.secondary,
          zIndex: 10,
          margin: 0,
        }}
      >
        {heading}
      </h2>

      {/* Table Container */}
      <div
        style={{
          width: "100%",
          maxWidth: "896px",
          overflow: "hidden",
          borderRadius: "16px",
          border: "1px solid rgba(255, 255, 255, 0.1)",
          backgroundColor: "rgba(255, 255, 255, 0.05)",
          backdropFilter: "blur(12px)",
          boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.5)",
          transform: `translateY(${translateY}px)`,
          opacity,
          borderColor: `${brandColors.primary}22`,
          zIndex: 10,
        }}
      >
        <table
          style={{
            width: "100%",
            tableLayout: "fixed",
            borderCollapse: "collapse",
            fontSize: format === "vertical" ? "12px" : "16px",
            fontFamily: "system-ui, -apple-system, sans-serif",
          }}
        >
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
                  style={{
                    padding: format === "vertical" ? "12px 8px" : "16px 24px",
                    textAlign: "left",
                    fontWeight: "bold",
                    color: "#d1d5db",
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                  }}
                >
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIdx) => {
              const isActive = rowIdx === activeRowIndex;

              const rowBg = isActive
                ? `rgba(255, 255, 255, 0.08)`
                : "transparent";
              const borderLeftColor = isActive
                ? brandColors.secondary
                : "transparent";
              const textColor = isActive ? "#ffffff" : "#9ca3af";
              const textWeight = isActive ? "bold" : "normal";
              const scale = isActive ? 1.01 : 1.0;

              return (
                <tr
                  key={rowIdx}
                  style={{
                    backgroundColor: rowBg,
                    borderLeft: `4px solid ${borderLeftColor}`,
                    transform: `scale(${scale})`,
                    borderBottom: "1px solid rgba(255,255,255,0.05)",
                    transition: "all 0.3s",
                  }}
                >
                  {row.map((cell, cellIdx) => (
                    <td
                      key={cellIdx}
                      style={{
                        padding: format === "vertical" ? "12px 8px" : "16px 24px",
                        color: textColor,
                        fontWeight: textWeight,
                        transition: "colors 0.3s",
                      }}
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
