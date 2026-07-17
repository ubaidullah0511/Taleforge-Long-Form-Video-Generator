import logging
import subprocess
import tempfile
from pathlib import Path

from app.clip_ingest import probe
from app.models import TimelineEntry
from app.timecode import parse_timestamp

logger = logging.getLogger(__name__)

_WIDTH, _HEIGHT, _FPS = 1920, 1080, 30
# Below this gap, a video source counts as an exact match ("ok"), not
# trimmed/looped — ffmpeg's own frame quantization can be off by a hair.
_DURATION_EPSILON = 0.05
# Slack allowed between the assembled video's total length and the last row's
# end time — frame-rate rounding across many concatenated segments.
_FINAL_DURATION_TOLERANCE = 0.25
_TARGET_RATIO = 16 / 9
# Same tolerance and rationale as app.asset_selection: absorbs mod-16 encode
# padding, not a genuinely different aspect ratio.
_RATIO_TOLERANCE = 0.05


def _parse_resolution(resolution: str) -> tuple[int, int]:
    try:
        width_str, height_str = resolution.split("x")
        return int(width_str), int(height_str)
    except ValueError:
        return 0, 0

_SCALE_PAD_FILTER = (
    f"scale={_WIDTH}:{_HEIGHT}:force_original_aspect_ratio=decrease,"
    f"pad={_WIDTH}:{_HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1"
)
# Outside this range, retiming playback speed to fill the row would look
# unnaturally fast/slow (sped-up or slow-motion footage is obviously off) —
# fall back to trim/loop for those instead.
_SPEED_MATCH_MIN, _SPEED_MATCH_MAX = 0.7, 1.4


class AssemblyValidationError(Exception):
    """Raised when the table isn't contiguous, a row's segment can't be
    rendered, or the final concat doesn't match the expected total duration —
    i.e. every case where we refuse to render a misaligned video."""


def _validate_contiguity(timeline: list[TimelineEntry]) -> None:
    ordered = sorted(timeline, key=lambda e: e.clip_number)
    for current, nxt in zip(ordered, ordered[1:]):
        if current.end != nxt.start:
            raise AssemblyValidationError(
                f"timeline gap/overlap between clip {current.clip_number} (ends {current.end}) "
                f"and clip {nxt.clip_number} (starts {nxt.start}) — table rows must be contiguous"
            )


def _validate_orientation(entry: TimelineEntry) -> None:
    """Hard gate before render: refuses any row whose source asset isn't 16:9
    horizontal. app.asset_selection already filters/crops stock candidates
    for the simple pipeline before they're ever selected; the documentary
    pipeline's own scoring (app/scoring.py) does not filter by orientation,
    so this is the safety net that catches it here instead of silently
    letterboxing a vertical/square source into a mostly-black frame."""
    if entry.placeholder or not entry.asset_path or entry.asset_metadata is None:
        return  # placeholder rows render a black _WIDTHx_HEIGHT frame directly

    _duration, resolution = probe(entry.asset_path)
    width, height = _parse_resolution(resolution)
    if width <= 0 or height <= 0 or width <= height or abs((width / height) - _TARGET_RATIO) > _RATIO_TOLERANCE:
        ratio = (width / height) if height else 0.0
        raise AssemblyValidationError(
            f"clip {entry.clip_number}: asset {entry.asset_path} is {width}x{height} "
            f"(ratio {ratio:.3f}) — not 16:9 horizontal (need {_TARGET_RATIO:.3f} ± {_RATIO_TOLERANCE})"
        )


def _run_ffmpeg(args: list[str]) -> bool:
    result = subprocess.run(["ffmpeg", "-y", *args], capture_output=True)
    if result.returncode != 0:
        logger.warning("documentary_assembly: ffmpeg failed: %s", result.stderr.decode(errors="replace")[-800:])
    return result.returncode == 0


def _render_video_segment(src: str, dest: Path, duration: float) -> bool:
    # -stream_loop -1 before -i makes the input read repeat indefinitely; -t
    # on the output then cuts at exactly `duration` regardless of whether a
    # loop actually happened — one command handles both trim and loop-to-fill.
    return _run_ffmpeg([
        "-stream_loop", "-1", "-i", src, "-t", str(duration),
        "-vf", _SCALE_PAD_FILTER, "-r", str(_FPS), "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-an", str(dest),
    ])


def _render_speed_matched_video_segment(src: str, dest: Path, duration: float, speed: float) -> bool:
    """Retimes the source's own playback speed (setpts) so its full content
    plays out across exactly `duration` — nothing trimmed off the end, nothing
    looped/repeated — instead of the trim/loop behavior in
    _render_video_segment. Audio is dropped here same as every other render
    path (-an); the row's real narration audio and captions are driven by the
    table's own Whisper timestamps and are muxed in separately afterward —
    they must never be retimed to match a video clip's speed adjustment."""
    pts_factor = 1.0 / speed  # e.g. speed 1.2x (needs to play faster) -> setpts factor 0.833 (compresses timestamps)
    return _run_ffmpeg([
        "-i", src, "-vf", f"{_SCALE_PAD_FILTER},setpts={pts_factor:.6f}*PTS",
        "-t", str(duration), "-r", str(_FPS), "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-an", str(dest),
    ])


def _render_image_segment(src: str, dest: Path, duration: float) -> bool:
    return _run_ffmpeg([
        "-loop", "1", "-i", src, "-t", str(duration),
        "-vf", _SCALE_PAD_FILTER, "-r", str(_FPS), "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-an", str(dest),
    ])


def _render_black_segment(dest: Path, duration: float) -> bool:
    return _run_ffmpeg([
        "-f", "lavfi", "-i", f"color=c=black:s={_WIDTH}x{_HEIGHT}:r={_FPS}:d={duration}",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-an", str(dest),
    ])


def _render_row_segment(entry: TimelineEntry, clip_dir: Path) -> tuple[Path, str]:
    dest = clip_dir / f"clip{entry.clip_number:03d}_rendered.mp4"

    if entry.placeholder or not entry.asset_path or entry.asset_metadata is None:
        if not _render_black_segment(dest, entry.duration):
            raise AssemblyValidationError(f"clip {entry.clip_number}: failed to render placeholder segment")
        return dest, "missing"

    if entry.asset_metadata.media_type == "image":
        if not _render_image_segment(entry.asset_path, dest, entry.duration):
            raise AssemblyValidationError(f"clip {entry.clip_number}: failed to render image segment")
        return dest, "ok"

    source_duration, _resolution = probe(entry.asset_path)

    if source_duration > _DURATION_EPSILON and entry.duration > _DURATION_EPSILON and \
            abs(source_duration - entry.duration) > _DURATION_EPSILON:
        speed = source_duration / entry.duration
        if _SPEED_MATCH_MIN <= speed <= _SPEED_MATCH_MAX:
            if not _render_speed_matched_video_segment(entry.asset_path, dest, entry.duration, speed):
                raise AssemblyValidationError(
                    f"clip {entry.clip_number}: failed to render speed-matched video segment"
                )
            return dest, "speed-matched"
        logger.info(
            "documentary_assembly: clip %d — required speed factor %.2fx is outside the "
            "natural-looking range %.1fx-%.1fx (source %.2fs vs row %.2fs), falling back to "
            "trim/loop instead of retiming playback speed",
            entry.clip_number, speed, _SPEED_MATCH_MIN, _SPEED_MATCH_MAX, source_duration, entry.duration,
        )

    if not _render_video_segment(entry.asset_path, dest, entry.duration):
        raise AssemblyValidationError(f"clip {entry.clip_number}: failed to render video segment")

    if source_duration < entry.duration - _DURATION_EPSILON:
        status = "looped-to-fill"
    elif source_duration > entry.duration + _DURATION_EPSILON:
        status = "trimmed"
    else:
        status = "ok"
    return dest, status


def _concat_segments(segment_paths: list[Path], dest: Path) -> bool:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for path in segment_paths:
            f.write(f"file '{path.resolve().as_posix()}'\n")
        filelist_path = f.name
    try:
        return _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", filelist_path, "-c", "copy", str(dest)])
    finally:
        Path(filelist_path).unlink(missing_ok=True)


def assemble_video(project_dir: Path, timeline: list[TimelineEntry]) -> tuple[Path, list[TimelineEntry]]:
    """Renders every table row to an exact-duration, uniformly-encoded segment
    (video trimmed/looped, image extended, missing rows filled with black) and
    concatenates them in clip order into <project_dir>/final_video.mp4.
    Raises AssemblyValidationError — never silently renders a misaligned
    video — if the table isn't contiguous, a row fails to render, or the
    final duration doesn't match the table's total."""
    if not timeline:
        raise AssemblyValidationError("cannot assemble an empty timeline")

    _validate_contiguity(timeline)

    clips_root = project_dir / "clips"
    ordered = sorted(timeline, key=lambda e: e.clip_number)

    updated_entries = []
    segment_paths = []
    for entry in ordered:
        _validate_orientation(entry)
        clip_dir = clips_root / f"clip_{entry.clip_number:03d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        segment_path, status = _render_row_segment(entry, clip_dir)
        segment_paths.append(segment_path)
        updated_entries.append(entry.model_copy(update={"status": status, "rendered_clip_path": str(segment_path)}))

    final_path = project_dir / "final_video.mp4"
    if not _concat_segments(segment_paths, final_path):
        raise AssemblyValidationError("ffmpeg concat failed while assembling final_video.mp4")

    expected_total = float(parse_timestamp(ordered[-1].end))
    actual_total, final_resolution = probe(str(final_path))
    final_width, final_height = _parse_resolution(final_resolution)
    if (final_width, final_height) != (_WIDTH, _HEIGHT):
        raise AssemblyValidationError(
            f"assembled video resolution {final_width}x{final_height} does not match the "
            f"required standard 16:9 frame size {_WIDTH}x{_HEIGHT}"
        )
    if abs(actual_total - expected_total) > _FINAL_DURATION_TOLERANCE:
        raise AssemblyValidationError(
            f"assembled video duration {actual_total:.2f}s does not match expected "
            f"total {expected_total:.2f}s (last row ends at {ordered[-1].end})"
        )

    return final_path, updated_entries
