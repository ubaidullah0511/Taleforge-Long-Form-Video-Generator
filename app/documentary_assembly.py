import logging
import subprocess
import tempfile
from pathlib import Path

from app.clip_ingest import probe
from app.ffmpeg_utils import get_ffmpeg_path
from app.models import TimelineEntry

logger = logging.getLogger(__name__)

_WIDTH, _HEIGHT, _FPS = 1920, 1080, 30
# Below this gap, a video source counts as an exact match ("ok"), not
# trimmed/looped — ffmpeg's own frame quantization can be off by a hair.
_DURATION_EPSILON = 0.05
# Slack allowed between the assembled video's total length and the last row's
# end time — frame-rate rounding across many concatenated segments.
_FINAL_DURATION_TOLERANCE = 0.25
# Padding added to every -t/lavfi "d=" cutoff below so -frames:v is what
# actually stops the encode, not the time-based cutoff — ffmpeg races -t
# against -frames:v and whichever hits first wins, and -t alone was clipping
# ~1 frame short on ~15% of segments in some real runs, undershooting the
# frame-exact contract this two-pass design depends on. All sources here are
# either infinite-looped or freshly generated, so extra padding costs nothing.
_FRAME_SAFETY_PADDING_SECONDS = 1.0
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


def _duration_to_frames(duration: float) -> int:
    return round(duration * _FPS)


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
    result = subprocess.run([get_ffmpeg_path(), "-y", *args], capture_output=True)
    if result.returncode != 0:
        logger.warning("documentary_assembly: ffmpeg failed: %s", result.stderr.decode(errors="replace")[-800:])
    return result.returncode == 0


def _render_video_segment(src: str, dest: Path, duration: float) -> bool:
    # -stream_loop -1 before -i makes the input read repeat indefinitely; -t
    # is a coarse safety cutoff but -frames:v is the authoritative one — it
    # pins the segment to an exact frame count so per-row float rounding
    # can't accumulate across the final concat (see assemble_video).
    return _run_ffmpeg([
        "-stream_loop", "-1", "-i", src, "-t", str(duration + _FRAME_SAFETY_PADDING_SECONDS),
        "-vf", _SCALE_PAD_FILTER, "-r", str(_FPS), "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-an", "-frames:v", str(_duration_to_frames(duration)), str(dest),
    ])


def _render_speed_matched_video_segment(src: str, dest: Path, duration: float, speed: float) -> bool:
    """Retimes the source's own playback speed (setpts) so its full content
    plays out across exactly `duration` — nothing trimmed off the end, nothing
    looped/repeated — instead of the trim/loop behavior in
    _render_video_segment. Audio is dropped here same as every other render
    path (-an); the row's real narration audio and captions are driven by the
    table's own Whisper timestamps and are muxed in separately afterward —
    they must never be retimed to match a video clip's speed adjustment.

    Unlike every other _render_*_segment (infinite -stream_loop, or -loop 1
    for images), this path's input is finite and played exactly once — the
    "-t is a coarse cutoff, -frames:v is authoritative" premise at the top of
    this module only holds when the source can supply extra frames on demand
    for free. Here it can't: once the decoder exhausts the (sped-up/down)
    source, -r's fps conversion can land exactly one frame short of
    -frames:v's target with no more input to draw from — confirmed via
    ffprobe on a real run (project_1785734747) losing exactly one 30fps frame
    (0.033s) on 11 of 25 speed-matched clips, which alone accounted for the
    entire 0.37s final-duration mismatch (every trim/loop and image segment
    matched exactly). tpad clones the last decoded frame for
    _FRAME_SAFETY_PADDING_SECONDS after the source ends, giving -frames:v the
    same "free" padding to draw from that the looped paths already have."""
    pts_factor = 1.0 / speed  # e.g. speed 1.2x (needs to play faster) -> setpts factor 0.833 (compresses timestamps)
    return _run_ffmpeg([
        "-i", src, "-vf",
        f"{_SCALE_PAD_FILTER},setpts={pts_factor:.6f}*PTS,"
        f"tpad=stop_mode=clone:stop_duration={_FRAME_SAFETY_PADDING_SECONDS}",
        "-t", str(duration + _FRAME_SAFETY_PADDING_SECONDS), "-r", str(_FPS), "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-an", "-frames:v", str(_duration_to_frames(duration)), str(dest),
    ])


def _render_fill_segment(src: str, dest: Path, duration: float) -> bool:
    """Renders `src` to cover exactly `duration` seconds without ever looping
    it: trims if it's already long enough, otherwise retimes its playback
    speed (never repeats/loops it) — even outside _SPEED_MATCH_MIN/MAX's
    normally "natural-looking" range. Used both for a row's fill clip (the
    second, distinct asset covering the remainder after the primary clip's
    own real duration — see _render_row_segment's "split-fill" branch) and,
    when no distinct second candidate was available at all, for the primary
    clip alone as the final non-looping fallback."""
    src_duration, _resolution = probe(src)
    if src_duration >= duration - _DURATION_EPSILON:
        return _render_video_segment(src, dest, duration)
    speed = max(src_duration / duration, 0.01) if duration > 0 else 1.0  # guards a zero/near-zero probe
    return _render_speed_matched_video_segment(src, dest, duration, speed)


def _render_image_segment(src: str, dest: Path, duration: float) -> bool:
    return _run_ffmpeg([
        "-loop", "1", "-i", src, "-t", str(duration + _FRAME_SAFETY_PADDING_SECONDS),
        "-vf", _SCALE_PAD_FILTER, "-r", str(_FPS), "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-an", "-frames:v", str(_duration_to_frames(duration)), str(dest),
    ])


def _render_black_segment(dest: Path, duration: float) -> bool:
    return _run_ffmpeg([
        "-f", "lavfi",
        "-i", f"color=c=black:s={_WIDTH}x{_HEIGHT}:r={_FPS}:d={duration + _FRAME_SAFETY_PADDING_SECONDS}",
        "-pix_fmt", "yuv420p", "-c:v", "libx264", "-an",
        "-frames:v", str(_duration_to_frames(duration)), str(dest),
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
            "natural-looking range %.1fx-%.1fx (source %.2fs vs row %.2fs)",
            entry.clip_number, speed, _SPEED_MATCH_MIN, _SPEED_MATCH_MAX, source_duration, entry.duration,
        )

    if source_duration < entry.duration - _DURATION_EPSILON:
        # Too short for a natural-looking speed adjustment — never loop the
        # same clip to fill the gap. Use the second, distinct candidate
        # documentary_pipeline._resolve_clip already resolved for this row
        # (if any) to cover the remainder; otherwise fall back to a forced
        # speed adjustment (still just this one clip, never repeated).
        if entry.fill_asset_path:
            primary_part = clip_dir / f"clip{entry.clip_number:03d}_part1.mp4"
            fill_part = clip_dir / f"clip{entry.clip_number:03d}_part2.mp4"
            if not _render_video_segment(entry.asset_path, primary_part, source_duration):
                raise AssemblyValidationError(f"clip {entry.clip_number}: failed to render primary split segment")
            remaining = entry.duration - source_duration
            if not _render_fill_segment(entry.fill_asset_path, fill_part, remaining):
                raise AssemblyValidationError(f"clip {entry.clip_number}: failed to render fill split segment")
            if not _concat_segments([primary_part, fill_part], _duration_to_frames(entry.duration), dest):
                raise AssemblyValidationError(f"clip {entry.clip_number}: failed to concat split-fill segments")
            return dest, "split-fill"

        if not _render_fill_segment(entry.asset_path, dest, entry.duration):
            raise AssemblyValidationError(f"clip {entry.clip_number}: failed to render video segment")
        return dest, "speed-matched"

    if not _render_video_segment(entry.asset_path, dest, entry.duration):
        raise AssemblyValidationError(f"clip {entry.clip_number}: failed to render video segment")
    status = "trimmed" if source_duration > entry.duration + _DURATION_EPSILON else "ok"
    return dest, status


def _concat_segments(segment_paths: list[Path], total_frames: int, dest: Path) -> bool:
    """Concatenate segments with the ffmpeg concat demuxer, then re-encode
    with a hard -frames:v cutoff at the exact sum of each row's own frame
    count. Each segment was itself rendered to an exact frame count (see
    _duration_to_frames), so total_frames is authoritative — trimming to a
    re-measured (ffprobe) duration here would reintroduce the float rounding
    this two-pass approach exists to eliminate."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for path in segment_paths:
            f.write(f"file '{path.resolve().as_posix()}'\n")
        filelist_path = f.name

    try:
        # First pass: raw concat
        temp_concat = dest.with_suffix(".concat.mp4")
        if not _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", filelist_path,
                            "-c", "copy", str(temp_concat)]):
            return False

        # Second pass: re-encode with a hard frame-count cutoff to eliminate
        # keyframe drift from the -c copy concat. No -c:a here — every segment
        # upstream is rendered with -an, so temp_concat has no audio stream to
        # begin with; real narration audio is muxed in later (app/audio_mux.py).
        if not _run_ffmpeg(["-i", str(temp_concat), "-frames:v", str(total_frames),
                            "-c:v", "libx264", "-preset", "fast", "-crf", "23", str(dest)]):
            temp_concat.unlink(missing_ok=True)
            return False

        temp_concat.unlink(missing_ok=True)
        return True
    finally:
        Path(filelist_path).unlink(missing_ok=True)


def rerender_single_clip_segment(
    project_dir: Path, timeline: list[TimelineEntry], clip_number: int,
) -> tuple[Path, list[TimelineEntry]]:
    """Fast-path counterpart to assemble_video: re-renders ONLY clip_number's
    own segment (its asset_path/asset_metadata are assumed already updated by
    the caller) and re-concats final_video.mp4 from the full segment list,
    reusing every other clip's already-rendered file untouched. Every other
    entry must already carry a real rendered_clip_path from a prior full
    assemble_video run — this function never re-renders them."""
    ordered = sorted(timeline, key=lambda e: e.clip_number)
    target_index = next((i for i, e in enumerate(ordered) if e.clip_number == clip_number), None)
    if target_index is None:
        raise AssemblyValidationError(f"clip {clip_number} not found in timeline")

    for entry in ordered:
        if entry.clip_number != clip_number and (
            not entry.rendered_clip_path or not Path(entry.rendered_clip_path).exists()
        ):
            raise AssemblyValidationError(
                f"clip {entry.clip_number} has no existing rendered segment on disk — "
                "run the full pipeline (assemble_video) at least once before a single-clip re-render"
            )

    target = ordered[target_index]
    _validate_orientation(target)
    clip_dir = project_dir / "clips" / f"clip_{clip_number:03d}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    segment_path, status = _render_row_segment(target, clip_dir)
    ordered[target_index] = target.model_copy(update={"status": status, "rendered_clip_path": str(segment_path)})

    segment_paths = [Path(e.rendered_clip_path) for e in ordered]
    total_frames = sum(_duration_to_frames(e.duration) for e in ordered)
    final_path = project_dir / "final_video.mp4"
    if not _concat_segments(segment_paths, total_frames, final_path):
        raise AssemblyValidationError(
            f"ffmpeg concat failed while re-assembling final_video.mp4 after re-rendering clip {clip_number}"
        )
    return final_path, ordered


def render_all_segments(project_dir: Path, timeline: list[TimelineEntry]) -> list[TimelineEntry]:
    """Renders every table row to its own exact-duration, uniformly-encoded
    segment file (video trimmed/looped, image extended, missing rows filled
    with black) and returns the timeline with status/rendered_clip_path
    populated — no concat, no final-video validation. This is the cheap,
    per-clip half of assemble_video (below), factored out so Mode A
    ("Generate & Edit" — see documentary_pipeline.generate_and_edit) can get
    every clip individually previewable/editable without also paying for
    (and being all-or-nothing gated on) the final ffmpeg concat + re-encode
    pass, which only matters once the user asks for the real final video.
    Still raises AssemblyValidationError for a genuinely broken table
    (non-contiguous rows, a non-16:9 real asset, a row that fails to
    render) — those would break per-clip editing too, not just the concat."""
    if not timeline:
        raise AssemblyValidationError("cannot render segments for an empty timeline")

    _validate_contiguity(timeline)

    clips_root = project_dir / "clips"
    ordered = sorted(timeline, key=lambda e: e.clip_number)

    updated_entries = []
    for entry in ordered:
        _validate_orientation(entry)
        clip_dir = clips_root / f"clip_{entry.clip_number:03d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        segment_path, status = _render_row_segment(entry, clip_dir)
        updated_entries.append(entry.model_copy(update={"status": status, "rendered_clip_path": str(segment_path)}))
    return updated_entries


def assemble_video(project_dir: Path, timeline: list[TimelineEntry]) -> tuple[Path, list[TimelineEntry]]:
    """render_all_segments() above, plus concatenating those segments in clip
    order into <project_dir>/final_video.mp4. Raises AssemblyValidationError
    — never silently renders a misaligned video — if the table isn't
    contiguous, a row fails to render, the ffmpeg concat fails, or the final
    duration/resolution doesn't match the table's total."""
    updated_entries = render_all_segments(project_dir, timeline)
    segment_paths = [Path(e.rendered_clip_path) for e in updated_entries]
    total_frames = sum(_duration_to_frames(e.duration) for e in updated_entries)

    final_path = project_dir / "final_video.mp4"
    if not _concat_segments(segment_paths, total_frames, final_path):
        raise AssemblyValidationError("ffmpeg concat failed while assembling final_video.mp4")

    # Sum of each row's own duration, NOT parse_timestamp(last_row.end) — the
    # table's start/end timestamps are absolute clock values, and the first
    # row does not always start at 00:00:00 (e.g. real Whisper-timed
    # narration commonly has the first detected word land a bit after t=0,
    # see app.documentary_table's timing assignment). Comparing the
    # concatenated file's real duration against the last row's absolute end
    # timestamp spuriously failed by exactly that leading offset on every
    # such project — the file was written correctly (see _concat_segments
    # above, which already sums row durations the same way) but this check
    # raised anyway, discarding final_video_path and silently skipping
    # Remotion render + narration audio mux for every one of them.
    expected_total = sum(entry.duration for entry in updated_entries)
    actual_total, final_resolution = probe(str(final_path))
    final_width, final_height = _parse_resolution(final_resolution)
    if (final_width, final_height) != (_WIDTH, _HEIGHT):
        raise AssemblyValidationError(
            f"assembled video resolution {final_width}x{final_height} does not match the "
            f"required standard 16:9 frame size {_WIDTH}x{_HEIGHT}"
        )
    if abs(actual_total - expected_total) > _FINAL_DURATION_TOLERANCE:
        raise AssemblyValidationError(
            f"assembled video duration {actual_total:.2f}s does not match the "
            f"{expected_total:.2f}s expected total (sum of all {len(updated_entries)} row durations)"
        )

    return final_path, updated_entries
