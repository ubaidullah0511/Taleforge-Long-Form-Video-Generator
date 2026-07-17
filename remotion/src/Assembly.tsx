import React from "react";
import { AbsoluteFill, OffthreadVideo, Sequence, interpolate, staticFile, useCurrentFrame } from "remotion";
import { Captions } from "./Captions";
import type { AssemblyProps, Pacing, TransitionStyle } from "./types";

// In/out effect length in frames, keyed by pacing. Kept deliberately short and
// ALWAYS fully inside a clip's own durationInFrames (never more than half of
// it) — this is a stylistic in/out effect, not an overlap-based crossfade, so
// no clip ever borrows frames from its neighbor and total timeline length is
// exactly the sum of the upstream table's row durations, unchanged.
const TRANSITION_FRAMES_BY_PACING: Record<Pacing, number> = {
  fast: 6,
  medium: 12,
  slow: 20,
};

const ClipLayer: React.FC<{
  path: string;
  durationInFrames: number;
  transitionStyle: TransitionStyle;
  pacing: Pacing;
  isFirst: boolean;
  isLast: boolean;
}> = ({ path, durationInFrames, transitionStyle, pacing, isFirst, isLast }) => {
  const frame = useCurrentFrame();
  const fadeFrames = Math.min(TRANSITION_FRAMES_BY_PACING[pacing] ?? 12, Math.floor(durationInFrames / 2));

  let opacity = 1;
  let transform = "none";

  if (transitionStyle !== "hard_cut" && fadeFrames > 0) {
    const fadeIn = isFirst ? 1 : interpolate(frame, [0, fadeFrames], [0, 1], { extrapolateRight: "clamp" });
    const fadeOut = isLast
      ? 1
      : interpolate(frame, [durationInFrames - fadeFrames, durationInFrames], [1, 0], { extrapolateLeft: "clamp" });
    opacity = Math.min(fadeIn, fadeOut);

    if (transitionStyle === "slide") {
      const slideIn = isFirst ? 0 : interpolate(frame, [0, fadeFrames], [80, 0], { extrapolateRight: "clamp" });
      transform = `translateX(${slideIn}px)`;
    } else if (transitionStyle === "zoom") {
      const zoomIn = isFirst ? 1 : interpolate(frame, [0, fadeFrames], [1.1, 1], { extrapolateRight: "clamp" });
      transform = `scale(${zoomIn})`;
    }
  }

  return (
    <AbsoluteFill style={{ opacity, transform }}>
      <OffthreadVideo src={staticFile(path)} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
    </AbsoluteFill>
  );
};

export const Assembly: React.FC<AssemblyProps> = ({ segments, words, style, width, height, includeCaptions }) => {
  return (
    <AbsoluteFill style={{ backgroundColor: "black", width, height }}>
      {segments.map((segment, index) => (
        <Sequence
          key={segment.clipNumber}
          from={segment.startFrame}
          durationInFrames={segment.durationInFrames}
          layout="none"
        >
          <ClipLayer
            path={segment.path}
            durationInFrames={segment.durationInFrames}
            transitionStyle={style.transitionStyle}
            pacing={style.pacing}
            isFirst={index === 0}
            isLast={index === segments.length - 1}
          />
        </Sequence>
      ))}
      {includeCaptions && <Captions words={words} emphasis={style.emphasis} pacing={style.pacing} />}
    </AbsoluteFill>
  );
};
