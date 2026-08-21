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

// Ken Burns "slow zoom" effect: scale growth per second, not a fixed total
// per clip — so a 3s and a 5s clip both zoom at the same visual speed
// instead of the same total zoom amount looking faster when compressed
// into a shorter clip. Tune this value visually.
const ZOOM_RATE_PER_SECOND = 0.04;

const ClipLayer: React.FC<{
  path: string;
  durationInFrames: number;
  transition: "hard_cut" | "dissolve";
  transitionStyle: TransitionStyle;
  pacing: Pacing;
  effect: string;
  mediaType: "video" | "image";
  fps: number;
  isFirst: boolean;
  isLast: boolean;
}> = ({ path, durationInFrames, transition, transitionStyle, pacing, effect, mediaType, fps, isFirst, isLast }) => {
  const frame = useCurrentFrame();
  const fadeFrames = Math.min(TRANSITION_FRAMES_BY_PACING[pacing] ?? 12, Math.floor(durationInFrames / 2));

  let opacity = 1;
  let transform = "none";

  // This clip's own transition (from VISUAL_TYPE_EDIT_MAP, e.g. Sports/
  // Exercise Demo's intentional "hard cut" for pacing) decides whether it
  // fades at all — the global transitionStyle only picks the fade's visual
  // flavor (fade/slide/zoom) once we already know a fade is happening, it
  // no longer forces every clip in the video to hard-cut just because the
  // script's overall style came back "hard_cut". A segment that IS meant to
  // dissolve but the global style has no flavor to offer (transitionStyle
  // itself is "hard_cut") still gets a plain fade rather than no fade.
  const effectiveStyle = transitionStyle === "hard_cut" ? "fade" : transitionStyle;

  if (transition !== "hard_cut" && fadeFrames > 0) {
    const fadeIn = isFirst ? 1 : interpolate(frame, [0, fadeFrames], [0, 1], { extrapolateRight: "clamp" });
    const fadeOut = isLast
      ? 1
      : interpolate(frame, [durationInFrames - fadeFrames, durationInFrames], [1, 0], { extrapolateLeft: "clamp" });
    opacity = Math.min(fadeIn, fadeOut);

    if (effectiveStyle === "slide") {
      const slideIn = isFirst ? 0 : interpolate(frame, [0, fadeFrames], [80, 0], { extrapolateRight: "clamp" });
      transform = `translateX(${slideIn}px)`;
    } else if (effectiveStyle === "zoom") {
      const zoomIn = isFirst ? 1 : interpolate(frame, [0, fadeFrames], [1.1, 1], { extrapolateRight: "clamp" });
      transform = `scale(${zoomIn})`;
    }
  }

  let zoomScale = 1;
  if (mediaType === "image" && effect.includes("slow zoom")) {
    const endScale = 1 + ZOOM_RATE_PER_SECOND * (durationInFrames / fps);
    zoomScale = interpolate(frame, [0, durationInFrames], [1, endScale], { extrapolateRight: "clamp" });
  }

  return (
    <AbsoluteFill style={{ opacity, transform }}>
      <AbsoluteFill style={{ transform: `scale(${zoomScale})` }}>
        <OffthreadVideo src={staticFile(path)} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

export const Assembly: React.FC<AssemblyProps> = ({ segments, words, style, width, height, includeCaptions, fps }) => {
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
            transition={segment.transition}
            transitionStyle={style.transitionStyle}
            pacing={style.pacing}
            effect={segment.effect}
            mediaType={segment.mediaType}
            fps={fps}
            isFirst={index === 0}
            isLast={index === segments.length - 1}
          />
        </Sequence>
      ))}
      {includeCaptions && <Captions words={words} emphasis={style.emphasis} pacing={style.pacing} />}
    </AbsoluteFill>
  );
};
