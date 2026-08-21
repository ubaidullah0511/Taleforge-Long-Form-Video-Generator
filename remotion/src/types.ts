export type TransitionStyle = "fade" | "slide" | "zoom" | "hard_cut";
export type Pacing = "fast" | "medium" | "slow";

export type StyleProps = {
  pacing: Pacing;
  transitionStyle: TransitionStyle;
  emphasis: string[];
};

// A single row's already fully-decided clip segment. startFrame/durationInFrames
// come straight from the upstream table/assembly stage (converted from its
// start/end timestamps at a fixed fps) — Remotion only sequences and styles
// these, it never computes or adjusts them. `path` is relative to the
// --public-dir the render was launched with (see render.mjs), consumed via
// staticFile() — Remotion's renderer can only fetch assets its own dev
// server serves over http(s), not arbitrary absolute paths.
export type ClipSegment = {
  clipNumber: number;
  path: string;
  startFrame: number;
  durationInFrames: number;
  // Free-text label from app.documentary_pipeline.edit_recommendation, e.g.
  // "slow zoom" or "film grain, dust overlay, slow zoom" — ClipLayer matches
  // on substring, not an exact enum, since niches combine multiple cues here.
  effect: string;
  // "video" plays its own real motion and must never get the Ken Burns zoom
  // on top of it; "image" (real photo, AI-generated, or a placeholder's
  // black frame) is static and is the only mediaType the zoom applies to.
  mediaType: "video" | "image";
  // Per-clip fade-or-not, derived from app.documentary_pipeline.edit_recommendation's
  // VISUAL_TYPE_EDIT_MAP (e.g. "hard cut" for Sports/Exercise Demo pacing,
  // "dissolve" for everything else including "cross dissolve"/"fade to
  // black"). This is the actual authority on whether THIS clip boundary
  // fades at all — style.transitionStyle below only picks the visual flavor
  // (fade/slide/zoom) for boundaries that do fade, it no longer blankets
  // every clip in the video as a hard cut just because the LLM picked
  // "hard_cut" as the overall script tone (see ClipLayer).
  transition: "hard_cut" | "dissolve";
};

// A single caption word's frame range. When a manually-supplied voiceover
// recording is provided, this is real Whisper ASR timing (no TTS — the user
// supplies the audio; see app/transcription.py); otherwise it falls back to
// a word-count pacing estimate (app/subtitles.py::build_word_timings). Either
// way, it's already computed in Python before this ever reaches Remotion.
export type CaptionWord = {
  text: string;
  startFrame: number;
  durationInFrames: number;
};

export type AssemblyProps = {
  segments: ClipSegment[];
  words: CaptionWord[];
  style: StyleProps;
  fps: number;
  width: number;
  height: number;
  totalDurationInFrames: number;
  includeCaptions: boolean;
};
