"""Integration test for app.remotion_render — exercises the real render.mjs
script end to end (real Node, real headless Chromium, no mocks). Slow (~15-30s:
bundling + a real short render each run). Run with: pytest tests/test_remotion_render.py -s
"""
import subprocess
import tempfile
from pathlib import Path

from app.clip_ingest import probe
from app.documentary_assembly import _FPS, _HEIGHT, _WIDTH
from app.models import AssetInfo, TimelineEntry
from app.remotion_render import RemotionRenderError, _build_props, _segment_transition, render_final_video
from app.style_decision import StyleDecision
from app.subtitles import WordTiming


def _make_clip(path: Path, color: str, duration: float) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s={_WIDTH}x{_HEIGHT}:r={_FPS}:d={duration}",
         "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, check=True,
    )


def _entry(clip_number, start, end, duration, script, rendered_path, media_type="video"):
    return TimelineEntry(
        clip_number=clip_number, start=start, end=end, duration=duration, script=script,
        asset_path=rendered_path,
        asset_metadata=AssetInfo(path=rendered_path, source="test", source_id=str(clip_number),
                                  media_type=media_type, width=_WIDTH, height=_HEIGHT, duration=duration, score=90.0),
        recommended_effect="slow zoom", transition="cross dissolve",
        rendered_clip_path=rendered_path, status="ok",
    )


def test_build_props_uses_row_boundaries_for_frame_numbers_not_accumulation():
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        seg1 = project_dir / "clips" / "clip_001" / "clip001_rendered.mp4"
        seg2 = project_dir / "clips" / "clip_002" / "clip002_rendered.mp4"
        seg1.parent.mkdir(parents=True)
        seg2.parent.mkdir(parents=True)
        seg1.write_bytes(b"x")
        seg2.write_bytes(b"x")

        timeline = [
            _entry(1, "00:00:00", "00:00:02", 2.0, "hello world", str(seg1)),
            _entry(2, "00:00:02", "00:00:05", 3.0, "second clip here", str(seg2)),
        ]
        props = _build_props(project_dir, timeline, StyleDecision())

        assert props["segments"][0] == {"clipNumber": 1, "path": "clips/clip_001/clip001_rendered.mp4",
                                         "startFrame": 0, "durationInFrames": 2 * _FPS, "effect": "slow zoom",
                                         "mediaType": "video", "transition": "dissolve"}
        assert props["segments"][1]["startFrame"] == 2 * _FPS  # exactly where row 1 ends, no drift
        assert props["totalDurationInFrames"] == 5 * _FPS


def test_segment_transition_only_hard_cut_string_skips_the_fade():
    """Only VISUAL_TYPE_EDIT_MAP's literal "hard cut" (Sports/Exercise Demo's
    intentional pacing choice) maps to "hard_cut" — every other value,
    including an unrecognized future one, dissolves rather than silently
    hard-cutting a clip that was never meant to (see _segment_transition)."""
    assert _segment_transition("hard cut") == "hard_cut"
    assert _segment_transition("Hard Cut") == "hard_cut"
    assert _segment_transition("cross dissolve") == "dissolve"
    assert _segment_transition("fade to black") == "dissolve"
    assert _segment_transition("something unrecognized") == "dissolve"


def test_build_props_forwards_per_clip_hard_cut_transition_distinct_from_dissolve():
    """Regression test: entry.transition ("cross dissolve"/"hard cut"/"fade
    to black" from edit_recommendation) must reach Remotion as each
    segment's own transition field — previously computed but never
    forwarded, so VISUAL_TYPE_EDIT_MAP's per-clip-type distinction (e.g.
    Sports' intentional hard cut for pacing) had zero effect on the actual
    rendered video; only the single global style.transitionStyle governed
    every clip boundary uniformly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        seg1 = project_dir / "clips" / "clip_001" / "clip001_rendered.mp4"
        seg2 = project_dir / "clips" / "clip_002" / "clip002_rendered.mp4"
        seg1.parent.mkdir(parents=True)
        seg2.parent.mkdir(parents=True)
        seg1.write_bytes(b"x")
        seg2.write_bytes(b"x")

        entry1 = _entry(1, "00:00:00", "00:00:02", 2.0, "a", str(seg1))
        entry1 = entry1.model_copy(update={"transition": "hard cut"})
        entry2 = _entry(2, "00:00:02", "00:00:04", 2.0, "b", str(seg2))
        entry2 = entry2.model_copy(update={"transition": "cross dissolve"})

        props = _build_props(project_dir, [entry1, entry2], StyleDecision())

        assert props["segments"][0]["transition"] == "hard_cut"
        assert props["segments"][1]["transition"] == "dissolve"


def test_build_props_forwards_media_type_so_zoom_never_applies_to_real_video():
    """Regression test: the Ken Burns "slow zoom" effect must only ever
    apply to static sources (real/AI-generated images), never to a real
    video clip's own motion (see ClipLayer's mediaType gate in
    Assembly.tsx) — this covers the Python-side half of that fix, that
    asset_metadata.media_type is actually forwarded into the props Remotion
    consumes, for both media types and for a placeholder (no asset_metadata
    at all, defaults to "image" since a placeholder is a static black
    frame, not real motion)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        seg1 = project_dir / "clips" / "clip_001" / "clip001_rendered.mp4"
        seg2 = project_dir / "clips" / "clip_002" / "clip002_rendered.mp4"
        seg3 = project_dir / "clips" / "clip_003" / "clip003_rendered.mp4"
        for seg in (seg1, seg2, seg3):
            seg.parent.mkdir(parents=True)
            seg.write_bytes(b"x")

        placeholder_entry = TimelineEntry(
            clip_number=3, start="00:00:04", end="00:00:06", duration=2.0, script="c",
            asset_path=None, asset_metadata=None, recommended_effect="slow zoom", transition="t",
            placeholder=True, rendered_clip_path=str(seg3), status="ok",
        )
        timeline = [
            _entry(1, "00:00:00", "00:00:02", 2.0, "a", str(seg1), media_type="video"),
            _entry(2, "00:00:02", "00:00:04", 2.0, "b", str(seg2), media_type="image"),
            placeholder_entry,
        ]
        props = _build_props(project_dir, timeline, StyleDecision())

        assert props["segments"][0]["mediaType"] == "video"
        assert props["segments"][1]["mediaType"] == "image"
        assert props["segments"][2]["mediaType"] == "image"  # placeholder defaults to image (static black frame)


def test_build_props_raises_when_a_row_has_no_rendered_clip_path():
    entry = TimelineEntry(
        clip_number=1, start="00:00:00", end="00:00:02", duration=2.0, script="x",
        asset_path=None, asset_metadata=None, recommended_effect="e", transition="t",
        placeholder=True, rendered_clip_path=None,
    )
    try:
        _build_props(Path("."), [entry], StyleDecision())
        raise AssertionError("expected RemotionRenderError")
    except RemotionRenderError as exc:
        assert "clip 1" in str(exc)


def test_build_props_uses_whisper_words_for_captions_bypassing_estimate():
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        seg1 = project_dir / "clips" / "clip_001" / "clip001_rendered.mp4"
        seg1.parent.mkdir(parents=True)
        seg1.write_bytes(b"x")

        # timeline's own script text would estimate very different words/timing —
        # whisper_words must win outright, proving no re-derivation happens.
        timeline = [_entry(1, "00:00:00", "00:00:02", 2.0, "unrelated timeline text", str(seg1))]
        whisper_words = [
            WordTiming(text="real", start=0.0, end=0.5),
            WordTiming(text="speech", start=0.5, end=1.1),
        ]

        props = _build_props(project_dir, timeline, StyleDecision(), whisper_words=whisper_words)

        assert [w["text"] for w in props["words"]] == ["real", "speech"]
        assert props["words"][0]["startFrame"] == 0
        assert props["words"][1]["startFrame"] == round(0.5 * _FPS)


def test_build_props_defaults_include_captions_to_false_but_still_computes_words():
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        seg1 = project_dir / "clips" / "clip_001" / "clip001_rendered.mp4"
        seg1.parent.mkdir(parents=True)
        seg1.write_bytes(b"x")

        timeline = [_entry(1, "00:00:00", "00:00:02", 2.0, "hello world", str(seg1))]

        props = _build_props(project_dir, timeline, StyleDecision())
        assert props["includeCaptions"] is False
        assert len(props["words"]) > 0  # caption words still computed, just not flagged for render

        props = _build_props(project_dir, timeline, StyleDecision(), include_captions=True)
        assert props["includeCaptions"] is True


def test_render_final_video_produces_correct_duration_and_resolution():
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        seg1 = project_dir / "clips" / "clip_001" / "clip001_rendered.mp4"
        seg2 = project_dir / "clips" / "clip_002" / "clip002_rendered.mp4"
        seg1.parent.mkdir(parents=True)
        seg2.parent.mkdir(parents=True)
        _make_clip(seg1, "red", 1.0)
        _make_clip(seg2, "blue", 1.0)

        timeline = [
            _entry(1, "00:00:00", "00:00:01", 1.0, "hello world", str(seg1)),
            _entry(2, "00:00:01", "00:00:02", 1.0, "second clip", str(seg2)),
        ]
        style = StyleDecision(pacing="fast", transition_style="fade", caption_emphasis=["second"])

        output_path = render_final_video(project_dir, timeline, style)

        assert output_path.exists()
        assert output_path.stat().st_size > 0
        duration, resolution = probe(str(output_path))
        assert abs(duration - 2.0) < 0.3
        assert resolution == f"{_WIDTH}x{_HEIGHT}"


if __name__ == "__main__":
    test_build_props_uses_row_boundaries_for_frame_numbers_not_accumulation()
    test_segment_transition_only_hard_cut_string_skips_the_fade()
    test_build_props_forwards_per_clip_hard_cut_transition_distinct_from_dissolve()
    test_build_props_forwards_media_type_so_zoom_never_applies_to_real_video()
    test_build_props_raises_when_a_row_has_no_rendered_clip_path()
    test_build_props_uses_whisper_words_for_captions_bypassing_estimate()
    test_build_props_defaults_include_captions_to_false_but_still_computes_words()
    test_render_final_video_produces_correct_duration_and_resolution()
    print("OK")
