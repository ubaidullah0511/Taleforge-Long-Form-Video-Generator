import React from "react";
import { Composition } from "remotion";
import { Assembly } from "./Assembly";
import type { AssemblyProps } from "./types";

const defaultProps: AssemblyProps = {
  segments: [],
  words: [],
  style: { pacing: "medium", transitionStyle: "fade", emphasis: [] },
  fps: 30,
  width: 1920,
  height: 1080,
  totalDurationInFrames: 30,
  includeCaptions: false,
};

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="Assembly"
      component={Assembly}
      durationInFrames={defaultProps.totalDurationInFrames}
      fps={defaultProps.fps}
      width={defaultProps.width}
      height={defaultProps.height}
      defaultProps={defaultProps}
      calculateMetadata={async ({ props }) => {
        const p = props as AssemblyProps;
        return {
          durationInFrames: p.totalDurationInFrames,
          fps: p.fps,
          width: p.width,
          height: p.height,
        };
      }}
    />
  );
};
