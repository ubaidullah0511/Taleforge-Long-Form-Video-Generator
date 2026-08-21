"""Offline tests for app.transcription — the local Whisper model call is
mocked throughout (no real audio file, no model weights needed).
Run with: pytest tests/test_transcription.py
"""
import tempfile
from pathlib import Path
from unittest.mock import patch

from app.subtitles import WordTiming
from app.transcription import TranscriptionError, check_alignment, get_narration_timing


def _mock_whisper_response(words):
    return {
        "text": " ".join(w["word"] for w in words),
        "segments": [{"words": words}],
    }


def test_get_narration_timing_parses_word_level_timestamps():
    words = [
        {"word": " hello", "start": 0.0, "end": 0.4},
        {"word": " world", "start": 0.4, "end": 0.9},
    ]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"fake audio bytes")
        audio_path = f.name

    try:
        with patch("app.transcription._transcribe_audio_local", return_value=_mock_whisper_response(words)):
            timings = get_narration_timing(audio_path)
    finally:
        Path(audio_path).unlink(missing_ok=True)

    assert timings == [
        WordTiming(text="hello", start=0.0, end=0.4),
        WordTiming(text="world", start=0.4, end=0.9),
    ]


def test_get_narration_timing_raises_on_missing_file():
    try:
        get_narration_timing("D:/does/not/exist.wav")
        raise AssertionError("expected TranscriptionError")
    except TranscriptionError as exc:
        assert "not found" in str(exc)


def test_get_narration_timing_raises_when_local_whisper_call_fails():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"fake audio bytes")
        audio_path = f.name

    try:
        with patch("app.transcription._transcribe_audio_local", side_effect=RuntimeError("corrupt audio")):
            try:
                get_narration_timing(audio_path)
                raise AssertionError("expected TranscriptionError")
            except TranscriptionError as exc:
                assert "corrupt audio" in str(exc)
    finally:
        Path(audio_path).unlink(missing_ok=True)


def test_get_narration_timing_raises_when_response_has_no_words():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"fake audio bytes")
        audio_path = f.name

    try:
        with patch("app.transcription._transcribe_audio_local", return_value={"text": "hello", "segments": []}):
            try:
                get_narration_timing(audio_path)
                raise AssertionError("expected TranscriptionError")
            except TranscriptionError as exc:
                assert "no usable" in str(exc)
    finally:
        Path(audio_path).unlink(missing_ok=True)


def test_check_alignment_returns_none_for_matching_script_and_transcript():
    script = "The quick brown fox jumps over the lazy dog"
    whisper_words = [WordTiming(text=w, start=i * 0.3, end=(i + 1) * 0.3) for i, w in enumerate(script.split())]
    assert check_alignment(whisper_words, script) is None


def test_check_alignment_flags_significant_deviation_and_lists_mismatches():
    script = "The quick brown fox jumps over the lazy dog in the forest today"
    # Transcript is missing a large chunk and has extra unrelated words —
    # deliberately well below the 0.85 similarity threshold.
    transcript_text = "Something completely unrelated was said instead here"
    whisper_words = [
        WordTiming(text=w, start=i * 0.3, end=(i + 1) * 0.3) for i, w in enumerate(transcript_text.split())
    ]

    warning = check_alignment(whisper_words, script)

    assert warning is not None
    assert "alignment ratio" in warning
    assert "replace" in warning or "delete" in warning or "insert" in warning  # difflib opcode tags present


if __name__ == "__main__":
    test_get_narration_timing_parses_word_level_timestamps()
    test_get_narration_timing_raises_on_missing_file()
    test_get_narration_timing_raises_when_local_whisper_call_fails()
    test_get_narration_timing_raises_when_response_has_no_words()
    test_check_alignment_returns_none_for_matching_script_and_transcript()
    test_check_alignment_flags_significant_deviation_and_lists_mismatches()
    print("OK")
