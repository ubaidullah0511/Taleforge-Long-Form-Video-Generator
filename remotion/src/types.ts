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
