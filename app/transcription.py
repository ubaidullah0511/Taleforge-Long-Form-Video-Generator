import difflib
import logging
import re
from functools import lru_cache
from pathlib import Path

import whisper

from app.config import settings
from app.ffmpeg_utils import get_ffmpeg_path
from app.subtitles import WordTiming

logger = logging.getLogger(__name__)

# Below this word-level similarity ratio (difflib.SequenceMatcher.ratio()),
# the manually-provided script and Whisper's actual transcription are
# considered to have deviated enough to flag for manual review.
_ALIGNMENT_WARNING_THRESHOLD = 0.85
_MAX_MISMATCHES_LOGGED = 20

_PUNCTUATION_RE = re.compile(r"[.,!?;:\"'()\[\]]")


class TranscriptionError(Exception):
    """Raised when the manually-supplied audio file is missing, or the
    transcription call fails or returns no word-level timestamps. Audio is an
    explicit opt-in to real timing — silently falling back to the word-count
    estimate here would hide a real problem (bad path, corrupt file, missing
    model weights) behind output that looks fine but isn't what was asked for."""


@lru_cache(maxsize=1)
def _local_whisper_model():
    """Loads the local Whisper model once per process (model load is the
    expensive part — multiple seconds to tens of seconds depending on size)
    and reuses it for every subsequent transcription call, same caching
    pattern as app.llm_client._client()."""
    logger.info("transcription: loading local Whisper model %r (first call only, cached)", settings.whisper_model_size)
    return whisper.load_model(settings.whisper_model_size)


def _transcribe_audio_local(audio_path: str) -> dict:
    # word_timestamps=True gets real per-word alignment (cross-attention/DTW)
    # from local Whisper, not just per-segment timing — same granularity the
    # old Groq API call provided, which downstream code already depends on.
    return _local_whisper_model().transcribe(audio_path, word_timestamps=True)


def get_narration_timing(audio_path: str) -> list[WordTiming]:
    """Transcribes a manually-supplied voiceover recording locally (offline,
    no API key, no rate limits) and returns real per-word timestamps. This is
    the source of truth for both caption text/timing
    (app/subtitles.py::build_word_timings) and per-clip narration duration
    (app/documentary_table.py::assign_timings) whenever audio is supplied —
    see documentary_pipeline.run()."""
    if not Path(audio_path).exists():
        raise TranscriptionError(f"audio file not found: {audio_path}")

    try:
        # whisper.load_audio() shells out to a bare "ffmpeg" on PATH to decode
        # the file — it can't be pointed at an explicit path, so this project's
        # bundled bin/ffmpeg.exe only reaches it via app.ffmpeg_utils prepending
        # bin/ to PATH on import (see that module's docstring). If neither the
        # bundled copy nor a PATH ffmpeg exists, subprocess would otherwise
        # raise a bare `FileNotFoundError: [WinError 2] The system cannot find
        # the file specified` with no mention of which file — easily misread
        # as the *audio* file having gone missing. Fail fast here instead with
        # an unambiguous message.
        get_ffmpeg_path()
    except FileNotFoundError as exc:
        raise TranscriptionError(f"cannot transcribe (required by Whisper to decode audio): {exc}") from exc

    try:
        result = _transcribe_audio_local(audio_path)
    except Exception as exc:
        raise TranscriptionError(f"transcription failed for {audio_path}: {exc}") from exc

    raw_words = [w for segment in result.get("segments", []) for w in segment.get("words", [])]
    if not raw_words:
        raise TranscriptionError(
            f"transcription returned no word-level timestamps for {audio_path} "
            "(response had no usable 'words' field)"
        )

    return [WordTiming(text=w["word"].strip(), start=float(w["start"]), end=float(w["end"])) for w in raw_words]


def _normalize_token(token: str) -> str:
    return _PUNCTUATION_RE.sub("", token).lower()


def check_alignment(whisper_words: list[WordTiming], script: str) -> str | None:
    """Sanity check only — compares Whisper's actual transcription against
    the manually-provided script. Never raises, never blocks the pipeline,
    never changes what gets used (Whisper's own words/timing remain the
    source of truth regardless of the result here). Returns a short warning
    message (and logs it) if they deviate significantly, else None."""
    transcript_tokens = [_normalize_token(w.text) for w in whisper_words if w.text.strip()]
    script_tokens = [_normalize_token(t) for t in script.split() if t.strip()]
    transcript_tokens = [t for t in transcript_tokens if t]
    script_tokens = [t for t in script_tokens if t]

    matcher = difflib.SequenceMatcher(a=script_tokens, b=transcript_tokens, autojunk=False)
    ratio = matcher.ratio()

    if ratio >= _ALIGNMENT_WARNING_THRESHOLD:
        logger.info(
            "transcription: script/transcript alignment ratio %.2f — looks consistent (%d script words, %d transcript words)",
            ratio, len(script_tokens), len(transcript_tokens),
        )
        return None

    mismatches = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        script_span = " ".join(script_tokens[i1:i2]) or "∅"
        transcript_span = " ".join(transcript_tokens[j1:j2]) or "∅"
        mismatches.append(f"  {tag}: script{[i1, i2]}={script_span!r} vs transcript{[j1, j2]}={transcript_span!r}")
        if len(mismatches) >= _MAX_MISMATCHES_LOGGED:
            mismatches.append(f"  ... additional mismatches truncated (showing first {_MAX_MISMATCHES_LOGGED})")
            break

    warning = (
        f"transcription: script/transcript alignment ratio {ratio:.2f} is below "
        f"{_ALIGNMENT_WARNING_THRESHOLD} threshold ({len(script_tokens)} script words, "
        f"{len(transcript_tokens)} transcript words) — review manually:\n" + "\n".join(mismatches)
    )
    logger.warning(warning)
    return warning
