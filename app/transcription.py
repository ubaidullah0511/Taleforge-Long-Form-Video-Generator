import difflib
import logging
import re
from pathlib import Path

from app.config import settings
from app.llm_client import transcribe_audio as _transcribe_audio_raw
from app.subtitles import WordTiming

logger = logging.getLogger(__name__)

# Below this word-level similarity ratio (difflib.SequenceMatcher.ratio()),
# the manually-provided script and Whisper's actual transcription are
# considered to have deviated enough to flag for manual review.
_ALIGNMENT_WARNING_THRESHOLD = 0.85
_MAX_MISMATCHES_LOGGED = 20

_PUNCTUATION_RE = re.compile(r"[.,!?;:\"'()\[\]]")


class TranscriptionError(Exception):
    """Raised when the manually-supplied audio file is missing, or Groq's
    Whisper call fails or returns no word-level timestamps. Audio is an
    explicit opt-in to real timing — silently falling back to the word-count
    estimate here would hide a real problem (bad path, expired key, network
    error) behind output that looks fine but isn't what was asked for."""


def get_narration_timing(audio_path: str) -> list[WordTiming]:
    """Transcribes a manually-supplied voiceover recording via Groq's Whisper
    endpoint and returns real per-word timestamps. This is the source of
    truth for both caption text/timing (app/subtitles.py::build_word_timings)
    and per-clip narration duration (app/documentary_table.py::assign_timings)
    whenever audio is supplied — see documentary_pipeline.run()."""
    if not Path(audio_path).exists():
        raise TranscriptionError(f"audio file not found: {audio_path}")

    try:
        data = _transcribe_audio_raw(audio_path, settings.llm_whisper_model)
    except Exception as exc:
        raise TranscriptionError(f"Groq Whisper transcription failed for {audio_path}: {exc}") from exc

    raw_words = data.get("words") or []
    if not raw_words:
        raise TranscriptionError(
            f"Groq Whisper returned no word-level timestamps for {audio_path} "
            "(response had no usable 'words' field)"
        )

    return [WordTiming(text=w["word"], start=float(w["start"]), end=float(w["end"])) for w in raw_words]


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
