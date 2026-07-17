import logging
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

from app.models import TimelineEntry
from app.timecode import parse_timestamp

logger = logging.getLogger(__name__)

_MAX_CHARS_PER_LINE = 42
_MAX_LINES_PER_CUE = 2


class SubtitleError(Exception):
    """Raised when there's no script text to caption, or the ffmpeg burn-in
    pass fails — never silently skipped, never a half-written output file."""


@dataclass
class SubtitleCue:
    start: float
    end: float
    text: str


def _chunk_text_for_captions(text: str, max_chars_per_line: int = _MAX_CHARS_PER_LINE,
                              max_lines: int = _MAX_LINES_PER_CUE) -> list[str]:
    lines = textwrap.wrap(text, width=max_chars_per_line) or [text]
    return ["\n".join(lines[i:i + max_lines]) for i in range(0, len(lines), max_lines)]


# ponytail: this pipeline has no TTS narration audio (see README) — cue timing
# is the same word-count/pace estimate already driving the silent video, split
# proportionally by word count across sub-cues, but never leaving the row's
# own already-validated start/end window. Real audio timestamps (e.g. via
# Whisper on generated narration) would replace this function's body only.
def build_cues(timeline: list[TimelineEntry]) -> list[SubtitleCue]:
    cues = []
    for entry in sorted(timeline, key=lambda e: e.clip_number):
        text = entry.script.strip()
        if not text:
            continue
        row_start = float(parse_timestamp(entry.start))
        row_end = float(parse_timestamp(entry.end))
        row_duration = row_end - row_start
        if row_duration <= 0:
            continue

        chunks = _chunk_text_for_captions(text)
        word_counts = [max(1, len(chunk.split())) for chunk in chunks]
        total_words = sum(word_counts)

        cursor = row_start
        for chunk, words in zip(chunks, word_counts):
            chunk_duration = row_duration * (words / total_words)
            cue_end = min(row_end, cursor + chunk_duration)
            cues.append(SubtitleCue(start=cursor, end=cue_end, text=chunk))
            cursor = cue_end
    return cues


@dataclass
class WordTiming:
    text: str
    start: float
    end: float


# ponytail: whisper_words (real ASR timestamps — see app/transcription.py)
# takes priority when a caller supplies them, returned verbatim since
# Whisper's own timing IS the caption timing. The word-count pacing estimate
# below is now only the FALLBACK for when no manually-supplied narration
# audio exists (see documentary_pipeline.run's audio_path param).
def build_word_timings(
    timeline: list[TimelineEntry], whisper_words: list[WordTiming] | None = None
) -> list[WordTiming]:
    if whisper_words is not None:
        return whisper_words

    words: list[WordTiming] = []
    for entry in sorted(timeline, key=lambda e: e.clip_number):
        text = entry.script.strip()
        if not text:
            continue
        row_start = float(parse_timestamp(entry.start))
        row_end = float(parse_timestamp(entry.end))
        row_duration = row_end - row_start
        if row_duration <= 0:
            continue

        tokens = text.split()
        if not tokens:
            continue
        per_word = row_duration / len(tokens)

        cursor = row_start
        for token in tokens:
            word_end = min(row_end, cursor + per_word)
            words.append(WordTiming(text=token, start=cursor, end=word_end))
            cursor = word_end
    return words


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def render_srt(cues: list[SubtitleCue]) -> str:
    blocks = [
        f"{index}\n{_format_srt_timestamp(cue.start)} --> {_format_srt_timestamp(cue.end)}\n{cue.text}\n"
        for index, cue in enumerate(cues, start=1)
    ]
    return "\n".join(blocks)


def _escape_ffmpeg_filter_path(path: str) -> str:
    # ffmpeg's filtergraph parser treats ':' as a key=value separator and '\'
    # as an escape char — both appear in a Windows absolute path (drive-letter
    # colon, backslash separators), so both need escaping.
    return path.replace("\\", "/").replace(":", "\\:")


def burn_subtitles(video_path: Path, srt_path: Path, output_path: Path) -> bool:
    escaped_srt = _escape_ffmpeg_filter_path(str(srt_path.resolve()))
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vf", f"subtitles='{escaped_srt}'",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_path)],
        capture_output=True,
    )
    if result.returncode != 0:
        logger.warning("subtitles: ffmpeg burn-in failed: %s", result.stderr.decode(errors="replace")[-800:])
    return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0


def generate_subtitled_video(
    project_dir: Path, timeline: list[TimelineEntry], final_video_path: Path
) -> tuple[Path, Path]:
    """Builds captions.srt from the table's own row timing and burns it onto
    the already-assembled final_video_path — a pure final pass, no
    re-assembly. Raises SubtitleError rather than producing a partial file."""
    cues = build_cues(timeline)
    if not cues:
        raise SubtitleError("no non-empty script text found to caption")

    srt_path = project_dir / "captions.srt"
    srt_path.write_text(render_srt(cues), encoding="utf-8")

    subtitled_path = project_dir / "final_video_subtitled.mp4"
    if not burn_subtitles(final_video_path, srt_path, subtitled_path):
        raise SubtitleError("ffmpeg failed to burn captions.srt onto the final video")

    return srt_path, subtitled_path
