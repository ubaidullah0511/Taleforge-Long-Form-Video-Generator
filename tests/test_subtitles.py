"""Tests for app.subtitles. Cue/SRT tests are pure offline logic; the burn-in
test uses real ffmpeg (synthetic lavfi source, no network).
Run with: pytest tests/test_subtitles.py
"""
import subprocess
import tempfile
from pathlib import Path

from app.clip_ingest import probe
from app.models import AssetInfo, TimelineEntry
from app.subtitles import (
    SubtitleError,
    WordTiming,
    _chunk_text_for_captions,
    build_cues,
    build_word_timings,
    generate_subtitled_video,
    render_srt,
)


def _entry(clip_number, start, end, duration, script):
    return TimelineEntry(
        clip_number=clip_number, start=start, end=end, duration=duration, script=script,
        asset_path="x", asset_metadata=AssetInfo(
            path="x", source="t", source_id=str(clip_number), media_type="video", duration=duration, score=90.0,
        ),
        recommended_effect="slow zoom", transition="cross dissolve",
    )


def test_chunk_text_wraps_at_max_chars_and_groups_two_lines_per_cue():
    text = "This is a longer sentence that should wrap across more than two lines when chunked for captions"
    chunks = _chunk_text_for_captions(text, max_chars_per_line=20, max_lines=2)
    for chunk in chunks:
        lines = chunk.split("\n")
        assert len(lines) <= 2
        assert all(len(line) <= 20 for line in lines)
    assert " ".join(chunks).replace("\n", " ") == " ".join(text.split())  # no words dropped


def test_build_cues_stays_within_row_time_window():
    entries = [_entry(1, "00:00:00", "00:00:04", 4.0, "One two three four five six seven eight nine ten eleven twelve")]
    cues = build_cues(entries)
    assert cues[0].start == 0.0
    assert cues[-1].end == 4.0
    for cue in cues:
        assert 0.0 <= cue.start <= cue.end <= 4.0


def test_build_cues_splits_proportionally_by_word_count_across_multiple_rows():
    entries = [
        _entry(1, "00:00:00", "00:00:03", 3.0, "First beat"),
        _entry(2, "00:00:03", "00:00:07", 4.0, "Second beat here"),
    ]
    cues = build_cues(entries)
    assert len(cues) == 2
    assert cues[0].start == 0.0 and cues[0].end == 3.0
    assert cues[1].start == 3.0 and cues[1].end == 7.0


def test_build_cues_skips_empty_script_rows():
    entries = [
        _entry(1, "00:00:00", "00:00:03", 3.0, "   "),
        _entry(2, "00:00:03", "00:00:06", 3.0, "Real text"),
    ]
    cues = build_cues(entries)
    assert len(cues) == 1
    assert cues[0].text == "Real text"


def test_render_srt_produces_valid_format():
    entries = [_entry(1, "00:00:01", "00:00:03", 2.0, "Hello world")]
    cues = build_cues(entries)
    srt = render_srt(cues)
    assert srt.startswith("1\n00:00:01,000 --> 00:00:03,000\nHello world")


def test_build_word_timings_splits_evenly_within_row_and_stays_in_window():
    entries = [_entry(1, "00:00:00", "00:00:04", 4.0, "one two three four")]
    words = build_word_timings(entries)
    assert [w.text for w in words] == ["one", "two", "three", "four"]
    assert words[0].start == 0.0
    assert words[-1].end == 4.0
    for word in words:
        assert 0.0 <= word.start <= word.end <= 4.0
    # four equal-length words across a 4s row -> 1s each
    assert all(abs((w.end - w.start) - 1.0) < 1e-9 for w in words)


def test_build_word_timings_continues_across_rows_without_gap():
    entries = [
        _entry(1, "00:00:00", "00:00:02", 2.0, "first row"),
        _entry(2, "00:00:02", "00:00:05", 3.0, "second row here"),
    ]
    words = build_word_timings(entries)
    assert len(words) == 5
    assert words[1].end == 2.0  # last word of row 1 ends exactly at row 1's boundary
    assert words[2].start == 2.0  # first word of row 2 starts exactly there, no gap/overlap
    assert words[-1].end == 5.0


def test_build_word_timings_skips_empty_script_rows():
    entries = [
        _entry(1, "00:00:00", "00:00:03", 3.0, "   "),
        _entry(2, "00:00:03", "00:00:06", 3.0, "Real words here"),
    ]
    words = build_word_timings(entries)
    assert [w.text for w in words] == ["Real", "words", "here"]


def test_build_word_timings_returns_whisper_words_verbatim_ignoring_timeline():
    entries = [_entry(1, "00:00:00", "00:00:04", 4.0, "one two three four")]
    whisper_words = [
        WordTiming(text="totally", start=0.1, end=0.5),
        WordTiming(text="different", start=0.5, end=1.2),
    ]
    words = build_word_timings(entries, whisper_words=whisper_words)
    assert words is whisper_words  # returned verbatim, no re-derivation from timeline at all


def test_generate_subtitled_video_raises_on_empty_timeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            generate_subtitled_video(Path(tmpdir), [], Path(tmpdir) / "final_video.mp4")
            raise AssertionError("expected SubtitleError")
        except SubtitleError as exc:
            assert "no non-empty script text" in str(exc)


def test_generate_subtitled_video_burns_captions_without_changing_duration():
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        video = project_dir / "src.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=320x240:r=25:d=4",
             "-pix_fmt", "yuv420p", str(video)],
            capture_output=True, check=True,
        )

        entries = [
            _entry(1, "00:00:00", "00:00:02", 2.0, "This is the first caption line to display on screen"),
            _entry(2, "00:00:02", "00:00:04", 2.0, "Second beat of narration text here"),
        ]

        srt_path, subtitled_path = generate_subtitled_video(project_dir, entries, video)

        assert srt_path.exists() and srt_path.read_text(encoding="utf-8").startswith("1\n")
        assert subtitled_path.exists()
        duration, _res = probe(str(subtitled_path))
        assert abs(duration - 4.0) < 0.1  # burning captions must not change video length


if __name__ == "__main__":
    test_chunk_text_wraps_at_max_chars_and_groups_two_lines_per_cue()
    test_build_cues_stays_within_row_time_window()
    test_build_cues_splits_proportionally_by_word_count_across_multiple_rows()
    test_build_cues_skips_empty_script_rows()
    test_render_srt_produces_valid_format()
    test_build_word_timings_splits_evenly_within_row_and_stays_in_window()
    test_build_word_timings_continues_across_rows_without_gap()
    test_build_word_timings_skips_empty_script_rows()
    test_build_word_timings_returns_whisper_words_verbatim_ignoring_timeline()
    test_generate_subtitled_video_raises_on_empty_timeline()
    test_generate_subtitled_video_burns_captions_without_changing_duration()
    print("OK")
