import React from "react";
import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import type { CaptionWord, Pacing } from "./types";

const WORDS_PER_LINE = 7;

// Faster pacing = snappier pop-in, less lingering; slower pacing = softer,
// more deliberate motion. This is the only thing "pacing" affects — it never
// touches word/clip start-end frames, those are fixed inputs.
const POP_SPRING_BY_PACING: Record<Pacing, { damping: number; stiffness: number }> = {
  fast: { damping: 200, stiffness: 400 },
  medium: { damping: 200, stiffness: 260 },
  slow: { damping: 200, stiffness: 140 },
};

function normalizeWord(word: string): string {
  return word.replace(/[^\w']/g, "").toLowerCase();
}

function isEmphasized(word: string, emphasis: string[]): boolean {
  const clean = normalizeWord(word);
  return clean.length > 0 && emphasis.some((e) => e.toLowerCase() === clean);
}

export const Captions: React.FC<{ words: CaptionWord[]; emphasis: string[]; pacing: Pacing }> = ({
  words,
  emphasis,
  pacing,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const springConfig = POP_SPRING_BY_PACING[pacing] ?? POP_SPRING_BY_PACING.medium;

  if (words.length === 0) return null;

  const lines: CaptionWord[][] = [];
  for (let i = 0; i < words.length; i += WORDS_PER_LINE) {
    lines.push(words.slice(i, i + WORDS_PER_LINE));
  }

  const activeLine = lines.find((line) => {
    const start = line[0].startFrame;
    const last = line[line.length - 1];
    const end = last.startFrame + last.durationInFrames;
    return frame >= start && frame < end;
  });

  if (!activeLine) return null;

  const lineStart = activeLine[0].startFrame;
  const popIn = spring({ frame: frame - lineStart, fps, config: springConfig });
  const opacity = interpolate(popIn, [0, 1], [0, 1]);
  const translateY = interpolate(popIn, [0, 1], [24, 0]);

  return (
    <AbsoluteFill style={{ justifyContent: "flex-end", alignItems: "center", paddingBottom: 120 }}>
      <div
        style={{
          opacity,
          transform: `translateY(${translateY}px)`,
          display: "flex",
          flexWrap: "wrap",
          justifyContent: "center",
          maxWidth: "85%",
          gap: "0.4em",
          fontFamily: "Arial, sans-serif",
          fontSize: 64,
          textAlign: "center",
          textShadow: "0 4px 12px rgba(0,0,0,0.85)",
        }}
      >
        {activeLine.map((word, i) => {
          const isActive = frame >= word.startFrame && frame < word.startFrame + word.durationInFrames;
          const emphasized = isEmphasized(word.text, emphasis);
          const wordPop = spring({ frame: frame - word.startFrame, fps, config: springConfig });
          const scale = isActive ? interpolate(wordPop, [0, 1], [1, 1.12]) : 1;
          return (
            <span
              key={`${word.startFrame}-${i}`}
              style={{
                color: isActive ? "#FFD400" : "#FFFFFF",
                fontWeight: emphasized ? 900 : 700,
                fontSize: emphasized ? 76 : 64,
                transform: `scale(${scale})`,
                display: "inline-block",
              }}
            >
              {word.text}
            </span>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
