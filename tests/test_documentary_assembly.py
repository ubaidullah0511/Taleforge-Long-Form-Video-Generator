"""Integration test for app.documentary_assembly — exercises real ffmpeg (no
network, no mocks) to prove exact-duration trim/loop/image/concat behavior.
Run with: pytest tests/test_documentary_assembly.py
"""
import subprocess
import tempfile
from pathlib import Path

from app.clip_ingest import probe
from app.documentary_assembly import AssemblyValidationError, _validate_contiguity, assemble_video
from app.models import AssetInfo, TimelineEntry


def _make_source_video(path: Path, color: str, duration: float, size: str = "640x360") -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s={size}:r=25:d={duration}",
         "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, check=True,
    )


def _make_source_image(path: Path, color: str, size: str = "640x360") -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s={size}:d=1", "-frames:v", "1", str(path)],
        capture_output=True, check=True,
    )


def _entry(clip_number, start, end, duration, asset_path=None, media_type="video", placeholder=False, width=640, height=360):
    asset_metadata = None
    if asset_path is not None:
        asset_metadata = AssetInfo(
            path=asset_path, source="test", source_id=str(clip_number), media_type=media_type,
            width=width, height=height, duration=duration, score=90.0,
        )
    return TimelineEntry(
        clip_number=clip_number, start=start, end=end, duration=duration,
        script="beat", asset_path=asset_path, asset_metadata=asset_metadata,
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


def test_assemble_video_end_to_end_with_trim_loop_image_and_missing():
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        clips_dir = project_dir / "clips"
        clips_dir.mkdir()

        long_source = clips_dir / "long_red.mp4"
        short_source = clips_dir / "short_blue.mp4"
        image_source = clips_dir / "green.jpg"
        _make_source_video(long_source, "red", 6.0)   # longer than its row -> trimmed
        _make_source_video(short_source, "blue", 2.0)  # shorter than its row -> looped-to-fill
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
        assert statuses == {1: "trimmed", 2: "looped-to-fill", 3: "ok", 4: "missing"}
        assert all(e.rendered_clip_path and Path(e.rendered_clip_path).exists() for e in updated)

        # every rendered per-row segment must be exactly its row's duration
        for entry in updated:
            actual, _res = probe(entry.rendered_clip_path)
            assert abs(actual - entry.duration) < 0.1, f"clip {entry.clip_number}: {actual} != {entry.duration}"

        total_duration, _res = probe(str(final_path))
        assert abs(total_duration - 12.0) < 0.25  # last row ends at 00:00:12


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


if __name__ == "__main__":
    test_validate_contiguity_passes_for_well_formed_timeline()
    test_validate_contiguity_raises_with_exact_row_numbers_on_gap()
    test_assemble_video_end_to_end_with_trim_loop_image_and_missing()
    test_assemble_video_speed_matches_a_clip_within_the_natural_range_instead_of_trim_loop()
    test_assemble_video_falls_back_to_trim_when_speed_factor_is_too_extreme()
    test_assemble_video_rejects_non_16_9_asset_with_clip_id_and_dimensions()
    print("OK")
