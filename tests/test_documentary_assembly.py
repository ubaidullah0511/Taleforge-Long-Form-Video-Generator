"""Integration test for app.documentary_assembly — exercises real ffmpeg (no
network, no mocks) to prove exact-duration trim/loop/image/concat behavior.
Run with: pytest tests/test_documentary_assembly.py
"""
import subprocess
import tempfile
from pathlib import Path

from app.clip_ingest import probe
from app.documentary_assembly import (
    AssemblyValidationError,
    _validate_contiguity,
    assemble_video,
    render_all_segments,
    rerender_single_clip_segment,
)
from app.models import AssetInfo, TimelineEntry


def _make_source_video(path: Path, color: str, duration: float, size: str = "640x360") -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s={size}:r=25:d={duration}",
         "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, check=True,
    )


def _make_source_video_with_short_video_stream(
    path: Path, video_duration: float, container_duration: float, size: str = "640x360",
) -> None:
    """Reproduces the actual shape of the real file that triggered the
    project_1785734747 bug: a video stream shorter than the container/audio
    duration app.clip_ingest.probe() reads (format=duration, i.e. driven by
    whichever stream — here audio — is longer). container_duration must be
    > video_duration."""
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-t", str(video_duration), "-i", f"color=c=red:s={size}:r=30",
         "-f", "lavfi", "-t", str(container_duration), "-i", "anullsrc=r=44100:cl=stereo",
         "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
        capture_output=True, check=True,
    )


def _make_source_image(path: Path, color: str, size: str = "640x360") -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s={size}:d=1", "-frames:v", "1", str(path)],
        capture_output=True, check=True,
    )


def _entry(
    clip_number, start, end, duration, asset_path=None, media_type="video", placeholder=False,
    width=640, height=360, fill_asset_path=None,
):
    asset_metadata = None
    if asset_path is not None:
        asset_metadata = AssetInfo(
            path=asset_path, source="test", source_id=str(clip_number), media_type=media_type,
            width=width, height=height, duration=duration, score=90.0,
        )
    fill_asset_metadata = None
    if fill_asset_path is not None:
        fill_asset_metadata = AssetInfo(
            path=fill_asset_path, source="test", source_id=f"{clip_number}_fill", media_type="video",
            width=width, height=height, duration=duration, score=80.0,
        )
    return TimelineEntry(
        clip_number=clip_number, start=start, end=end, duration=duration,
        script="beat", asset_path=asset_path, asset_metadata=asset_metadata,
        fill_asset_path=fill_asset_path, fill_asset_metadata=fill_asset_metadata,
        recommended_effect="slow zoom", transition="cross dissolve",
        placeholder=placeholder, note="no asset found" if placeholder else "",
    )


def test_validate_contiguity_passes_for_well_formed_timeline():
    entries = [
        _entry(1, "00:00:00", "00:00:03", 3.0, asset_path="x", media_type="video"),
        _entry(2, "00:00:03", "00:00:06", 3.0, asset_path="x", media_type="video"),
    ]
    _validate_contiguity(entries)  # must not raise


def test_validate_contiguity_raises_with_exact_row_numbers_on_gap():
    entries = [
        _entry(1, "00:00:00", "00:00:03", 3.0, asset_path="x", media_type="video"),
        _entry(2, "00:00:04", "00:00:07", 3.0, asset_path="x", media_type="video"),  # gap: should start at 00:00:03
    ]
    try:
        _validate_contiguity(entries)
        raise AssertionError("expected AssemblyValidationError")
    except AssemblyValidationError as exc:
        assert "clip 1" in str(exc) and "clip 2" in str(exc)
        assert "00:00:03" in str(exc) and "00:00:04" in str(exc)


def test_assemble_video_end_to_end_with_trim_forced_speed_match_image_and_missing():
    """Regression note: clip 2's 2.0s source in a 5.0s row (ratio 0.4, outside
    the natural speed-match range) used to loop the same short clip to fill
    the gap ("looped-to-fill"). Looping was removed as a fallback entirely —
    with no second candidate available (entry.fill_asset_path unset here),
    it now gets a forced (out-of-natural-range but never repeated) speed
    adjustment instead, still reported as "speed-matched". See
    test_assemble_video_uses_a_distinct_fill_clip_instead_of_looping below
    for the split-fill case where a second candidate IS available."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        clips_dir = project_dir / "clips"
        clips_dir.mkdir()

        long_source = clips_dir / "long_red.mp4"
        short_source = clips_dir / "short_blue.mp4"
        image_source = clips_dir / "green.jpg"
        _make_source_video(long_source, "red", 6.0)   # longer than its row -> trimmed
        _make_source_video(short_source, "blue", 2.0)  # shorter than its row, no fill candidate -> forced speed-match
        _make_source_image(image_source, "green")

        timeline = [
            _entry(1, "00:00:00", "00:00:03", 3.0, asset_path=str(long_source), media_type="video"),
            _entry(2, "00:00:03", "00:00:08", 5.0, asset_path=str(short_source), media_type="video"),
            _entry(3, "00:00:08", "00:00:10", 2.0, asset_path=str(image_source), media_type="image"),
            _entry(4, "00:00:10", "00:00:12", 2.0, placeholder=True),
        ]

        final_path, updated = assemble_video(project_dir, timeline)

        assert final_path.exists()
        statuses = {e.clip_number: e.status for e in updated}
        assert statuses == {1: "trimmed", 2: "speed-matched", 3: "ok", 4: "missing"}
        assert "looped-to-fill" not in statuses.values()
        assert all(e.rendered_clip_path and Path(e.rendered_clip_path).exists() for e in updated)

        # every rendered per-row segment must be exactly its row's duration
        for entry in updated:
            actual, _res = probe(entry.rendered_clip_path)
            assert abs(actual - entry.duration) < 0.1, f"clip {entry.clip_number}: {actual} != {entry.duration}"

        total_duration, _res = probe(str(final_path))
        assert abs(total_duration - 12.0) < 0.25  # last row ends at 00:00:12


def test_assemble_video_uses_a_distinct_fill_clip_instead_of_looping():
    """The actual fix under test: when documentary_pipeline._resolve_clip has
    already resolved a second, distinct candidate for a too-short primary
    clip (entry.fill_asset_path), assembly must use it to cover the
    remainder — never loop the primary clip — and report status
    "split-fill". A 1.0s primary source in a 5.0s row would previously have
    looped 5x; here it plays once (its own real 1.0s) and a different 4.0s
    clip covers the remaining 4.0s."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        clips_dir = project_dir / "clips"
        clips_dir.mkdir()

        primary_source = clips_dir / "primary_blue.mp4"
        fill_source = clips_dir / "fill_yellow.mp4"
        _make_source_video(primary_source, "blue", 1.0)
        _make_source_video(fill_source, "yellow", 4.0)

        timeline = [_entry(
            1, "00:00:00", "00:00:05", 5.0,
            asset_path=str(primary_source), fill_asset_path=str(fill_source), media_type="video",
        )]

        final_path, updated = assemble_video(project_dir, timeline)

        assert final_path.exists()
        assert updated[0].status == "split-fill"
        actual, _res = probe(updated[0].rendered_clip_path)
        assert abs(actual - 5.0) < 0.1


def test_assemble_video_forces_speed_match_when_short_clip_has_no_fill_candidate():
    """Same too-short primary as above but with no fill_asset_path resolved
    (mirrors _resolve_clip finding no distinct second candidate) — must fall
    back to a forced speed adjustment (out of the normally natural-looking
    range, but still just this one clip played once, never repeated) rather
    than loop, and report "speed-matched"."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        clips_dir = project_dir / "clips"
        clips_dir.mkdir()

        primary_source = clips_dir / "primary_blue.mp4"
        _make_source_video(primary_source, "blue", 1.0)

        timeline = [_entry(1, "00:00:00", "00:00:05", 5.0, asset_path=str(primary_source), media_type="video")]

        final_path, updated = assemble_video(project_dir, timeline)

        assert final_path.exists()
        assert updated[0].status == "speed-matched"
        actual, _res = probe(updated[0].rendered_clip_path)
        assert abs(actual - 5.0) < 0.1


def test_render_all_segments_produces_per_clip_files_without_final_concat():
    """Mode A ("Generate & Edit" — see documentary_pipeline.generate_and_edit)
    calls this instead of assemble_video: every clip must still get its own
    real rendered_clip_path (what editor preview/single-clip-re-render both
    need), but no final_video.mp4 concat/re-encode pass should run at all —
    that stays exclusive to assemble_video / "Generate Full Video"."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        clips_dir = project_dir / "clips"
        clips_dir.mkdir()

        source = clips_dir / "red.mp4"
        _make_source_video(source, "red", 3.0)

        timeline = [
            _entry(1, "00:00:00", "00:00:03", 3.0, asset_path=str(source), media_type="video"),
            _entry(2, "00:00:03", "00:00:05", 2.0, placeholder=True),
        ]

        updated = render_all_segments(project_dir, timeline)

        assert all(e.rendered_clip_path and Path(e.rendered_clip_path).exists() for e in updated)
        assert not (project_dir / "final_video.mp4").exists()


def test_assemble_video_succeeds_when_first_row_does_not_start_at_zero():
    """Real bug: real Whisper-timed narration commonly produces a table
    whose first row starts a bit after 00:00:00 (the first detected word
    isn't at the very start of the audio) — contiguity between rows still
    holds, so render_all_segments succeeds, but the OLD final-duration check
    compared the concatenated file's real length against
    parse_timestamp(last_row.end) (an ABSOLUTE clock value, here 6.0s)
    instead of the sum of row durations (here 3.0 + 2.0 = 5.0s) — a
    guaranteed spurious 1.0s mismatch on every such project, discarding
    final_video_path and silently skipping Remotion render + narration
    audio mux downstream (see documentary_pipeline._finalize_and_render's
    `if final_video_path is not None:` gate). Every one of a handful of
    real projects inspected during this investigation had exactly this
    shape (first row starting at 00:00:01, not 00:00:00)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        clips_dir = project_dir / "clips"
        clips_dir.mkdir()
        source = clips_dir / "red.mp4"
        _make_source_video(source, "red", 3.0)

        timeline = [
            _entry(1, "00:00:01", "00:00:04", 3.0, asset_path=str(source), media_type="video"),
            _entry(2, "00:00:04", "00:00:06", 2.0, placeholder=True),
        ]

        final_path, updated = assemble_video(project_dir, timeline)

        assert final_path.exists()
        actual_total, _res = probe(str(final_path))
        assert abs(actual_total - 5.0) < 0.25  # sum of row durations, NOT 6.0s (last row's absolute end)


def test_render_all_segments_still_validates_contiguity_and_orientation():
    """The per-row validations that matter for editing (contiguous table,
    16:9 real assets) still apply — only the final concat/duration/
    resolution checks (which only matter for the finished video) are
    skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        (project_dir / "clips").mkdir()
        gapped_timeline = [
            _entry(1, "00:00:00", "00:00:03", 3.0, placeholder=True),
            _entry(2, "00:00:04", "00:00:06", 2.0, placeholder=True),  # gap: 03 != 04
        ]
        try:
            render_all_segments(project_dir, gapped_timeline)
            raise AssertionError("expected AssemblyValidationError")
        except AssemblyValidationError as exc:
            assert "gap/overlap" in str(exc)


def test_assemble_video_speed_matches_a_clip_within_the_natural_range_instead_of_trim_loop():
    """A source needing a 1.2x speed-up (within the 0.7x-1.4x natural-looking
    range) should get its own playback speed retimed via setpts to exactly
    fill the row's duration, not trimmed or looped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        (project_dir / "clips").mkdir()
        source = project_dir / "clips" / "needs_speedup.mp4"
        _make_source_video(source, "red", 3.6)  # 3.6s source vs 3.0s row -> speed 1.2x, in range

        timeline = [_entry(1, "00:00:00", "00:00:03", 3.0, asset_path=str(source), media_type="video")]

        final_path, updated = assemble_video(project_dir, timeline)

        assert final_path.exists()
        assert updated[0].status == "speed-matched"
        actual, _res = probe(updated[0].rendered_clip_path)
        assert abs(actual - 3.0) < 0.1  # retimed to exactly the row's duration


def test_assemble_video_falls_back_to_trim_when_speed_factor_is_too_extreme():
    """A source needing a 3x speed-up (well outside the 0.7x-1.4x range)
    would look unnaturally sped-up, so it must fall back to the existing
    trim/loop behavior instead of setpts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        (project_dir / "clips").mkdir()
        source = project_dir / "clips" / "needs_extreme_speedup.mp4"
        _make_source_video(source, "red", 9.0)  # 9.0s source vs 3.0s row -> speed 3.0x, out of range

        timeline = [_entry(1, "00:00:00", "00:00:03", 3.0, asset_path=str(source), media_type="video")]

        final_path, updated = assemble_video(project_dir, timeline)

        assert final_path.exists()
        assert updated[0].status == "trimmed"  # fell back, not speed-matched
        actual, _res = probe(updated[0].rendered_clip_path)
        assert abs(actual - 3.0) < 0.1


def _ffprobe_frame_count(path: str) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip())


def test_assemble_video_speed_matched_clip_is_frame_exact_not_one_short():
    """Regression for project_1785734747: clip 36's real source file had a
    145-frame/4.8333s video stream but a 4.852993s audio stream — probe()
    reads format=duration (the longer, audio-driven value), so the computed
    speed/pts_factor under-corrects and the retimed video content itself
    lands short of the row's duration before frame-count cutoff even applies.
    Unlike every other render path (infinite -stream_loop, or -loop 1 for
    images), _render_speed_matched_video_segment's input plays once with no
    "free" extra material for -frames:v to draw from once the source runs
    out — this reproduces that exact video/container duration mismatch
    (source 4.852993s per probe() vs row 5.0s, speed 0.9706x) that measured
    149/150 frames (4.967s instead of 5.000s) on 11 of 25 speed-matched
    clips in the real run, before tpad clone-padding was added. A loose
    (< 0.1s) tolerance would not catch a single dropped frame at 30fps
    (0.033s) — this asserts the exact frame count instead."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        (project_dir / "clips").mkdir()
        source = project_dir / "clips" / "needs_slowdown.mp4"
        # video stream 145 frames @ 30fps = 4.833333s, container/audio 4.852993s
        # (matches clip036_video1.mp4 exactly) -> probe() reports 4.852993s.
        _make_source_video_with_short_video_stream(source, video_duration=4.833333, container_duration=4.852993)

        timeline = [_entry(1, "00:00:00", "00:00:05", 5.0, asset_path=str(source), media_type="video")]

        final_path, updated = assemble_video(project_dir, timeline)

        assert updated[0].status == "speed-matched"
        frame_count = _ffprobe_frame_count(updated[0].rendered_clip_path)
        assert frame_count == 150, f"expected exactly 150 frames (30fps * 5.0s), got {frame_count}"
        actual, _res = probe(updated[0].rendered_clip_path)
        assert abs(actual - 5.0) < 0.01, f"expected ~5.000s, got {actual:.4f}s"


def test_assemble_video_rejects_non_16_9_asset_with_clip_id_and_dimensions():
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        (project_dir / "clips").mkdir()
        bad_source = project_dir / "clips" / "four_three.mp4"
        _make_source_video(bad_source, "red", 3.0, size="320x240")  # 4:3 — not 16:9

        timeline = [_entry(1, "00:00:00", "00:00:03", 3.0, asset_path=str(bad_source),
                            media_type="video", width=320, height=240)]

        try:
            assemble_video(project_dir, timeline)
            raise AssertionError("expected AssemblyValidationError")
        except AssemblyValidationError as exc:
            assert "clip 1" in str(exc)
            assert "320x240" in str(exc)


def test_rerender_single_clip_segment_only_rerenders_target_and_reuses_others():
    """The editor's fast single-clip edit path: only clip_number's own
    segment gets re-rendered (from its NEW asset_path) and re-concatenated —
    every other clip's already-rendered file must be reused byte-for-byte,
    not regenerated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        (project_dir / "clips").mkdir()
        source1 = project_dir / "clips" / "red.mp4"
        source2 = project_dir / "clips" / "blue.mp4"
        _make_source_video(source1, "red", 3.0)
        _make_source_video(source2, "blue", 3.0)

        timeline = [
            _entry(1, "00:00:00", "00:00:03", 3.0, asset_path=str(source1), media_type="video"),
            _entry(2, "00:00:03", "00:00:06", 3.0, asset_path=str(source2), media_type="video"),
        ]
        final_path, updated = assemble_video(project_dir, timeline)
        original_clip2_rendered = updated[1].rendered_clip_path
        original_clip2_mtime = Path(original_clip2_rendered).stat().st_mtime

        # Simulate the editor swapping clip 1's asset to a new source.
        new_source = project_dir / "clips" / "green.mp4"
        _make_source_video(new_source, "green", 3.0)
        updated[0] = updated[0].model_copy(update={"asset_path": str(new_source)})

        final_path2, updated2 = rerender_single_clip_segment(project_dir, updated, clip_number=1)

        assert final_path2 == final_path
        assert final_path2.exists()
        # clip 2's rendered file is untouched (same path, same mtime) — never re-rendered.
        assert updated2[1].rendered_clip_path == original_clip2_rendered
        assert Path(original_clip2_rendered).stat().st_mtime == original_clip2_mtime
        # clip 1's rendered segment reflects the new source, still exactly 3.0s.
        actual, _res = probe(updated2[0].rendered_clip_path)
        assert abs(actual - 3.0) < 0.1
        total_duration, _res = probe(str(final_path2))
        assert abs(total_duration - 6.0) < 0.25


def test_rerender_single_clip_segment_raises_if_another_clip_was_never_rendered():
    """A clip that has no rendered_clip_path yet (e.g. timeline.json from a
    project that never finished a full assemble_video pass) must fail loudly
    instead of silently concatenating a missing/stale file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        (project_dir / "clips").mkdir()
        source1 = project_dir / "clips" / "red.mp4"
        _make_source_video(source1, "red", 3.0)

        timeline = [
            _entry(1, "00:00:00", "00:00:03", 3.0, asset_path=str(source1), media_type="video"),
            _entry(2, "00:00:03", "00:00:06", 3.0, asset_path=str(source1), media_type="video"),
        ]
        # clip 2 never went through assemble_video -> no rendered_clip_path.
        try:
            rerender_single_clip_segment(project_dir, timeline, clip_number=1)
            raise AssertionError("expected AssemblyValidationError")
        except AssemblyValidationError as exc:
            assert "clip 2" in str(exc)


if __name__ == "__main__":
    test_validate_contiguity_passes_for_well_formed_timeline()
    test_validate_contiguity_raises_with_exact_row_numbers_on_gap()
    test_assemble_video_end_to_end_with_trim_forced_speed_match_image_and_missing()
    test_assemble_video_uses_a_distinct_fill_clip_instead_of_looping()
    test_assemble_video_forces_speed_match_when_short_clip_has_no_fill_candidate()
    test_render_all_segments_produces_per_clip_files_without_final_concat()
    test_assemble_video_succeeds_when_first_row_does_not_start_at_zero()
    test_render_all_segments_still_validates_contiguity_and_orientation()
    test_assemble_video_speed_matches_a_clip_within_the_natural_range_instead_of_trim_loop()
    test_assemble_video_falls_back_to_trim_when_speed_factor_is_too_extreme()
    test_assemble_video_speed_matched_clip_is_frame_exact_not_one_short()
    test_assemble_video_rejects_non_16_9_asset_with_clip_id_and_dimensions()
    test_rerender_single_clip_segment_only_rerenders_target_and_reuses_others()
    test_rerender_single_clip_segment_raises_if_another_clip_was_never_rendered()
    print("OK")
