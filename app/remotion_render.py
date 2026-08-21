import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

from app.config import settings
from app.documentary_assembly import _FPS, _HEIGHT, _WIDTH
from app.models import TimelineEntry
from app.style_decision import StyleDecision
from app.subtitles import WordTiming, build_word_timings
from app.timecode import parse_timestamp

logger = logging.getLogger(__name__)

# ponytail: on Windows, @remotion/bundler always does a real recursive file
# copy of --public-dir into the bundle (no symlink support), and a transient
# 404/timeout has been observed mid-render — a bare retry succeeded against
# the exact same files. Cheap insurance for a flaky-but-rare failure mode
# rather than a real logic bug (see chat: reproduced, retry succeeded).
_MAX_RENDER_ATTEMPTS = 2
_RETRY_DELAY_SECONDS = 3.0


class RemotionRenderError(Exception):
    """Raised when the Remotion render script fails, produces no output, or
    a row is missing its rendered_clip_path — never silently returns a
    partial/missing final video, same philosophy as AssemblyValidationError."""


def _to_frame(seconds: float) -> int:
    return round(seconds * _FPS)


def _segment_transition(transition: str) -> str:
    """Maps app.documentary_pipeline.edit_recommendation's free-text
    per-clip transition ("cross dissolve", "fade to black", "hard cut") down
    to the two-way choice Assembly.tsx's ClipLayer actually acts on: whether
    THIS clip boundary fades at all. Only "hard cut" (Sports/Exercise Demo's
    intentional pacing choice in VISUAL_TYPE_EDIT_MAP) skips the fade —
    every other value, including a future/unrecognized one, dissolves rather
    than risk silently hard-cutting a clip that was never meant to."""
    return "hard_cut" if transition.strip().lower() == "hard cut" else "dissolve"


def _build_props(
    project_dir: Path, timeline: list[TimelineEntry], style: StyleDecision,
    whisper_words: list[WordTiming] | None = None, include_captions: bool = False,
) -> dict:
    ordered = sorted(timeline, key=lambda e: e.clip_number)
    segments = []
    for entry in ordered:
        if not entry.rendered_clip_path:
            raise RemotionRenderError(
                f"clip {entry.clip_number} has no rendered_clip_path — assemble_video must run first"
            )
        # Frame boundaries come from rounding each row's own absolute start/end
        # (not accumulating per-row durations), so they land exactly on the
        # same contiguous boundaries assemble_video already validated in
        # second-space — Remotion never recomputes timing, only converts units.
        start_frame = _to_frame(float(parse_timestamp(entry.start)))
        end_frame = _to_frame(float(parse_timestamp(entry.end)))
        rel_path = Path(entry.rendered_clip_path).resolve().relative_to(project_dir.resolve()).as_posix()
        segments.append({
            "clipNumber": entry.clip_number,
            "path": rel_path,
            "startFrame": start_frame,
            "durationInFrames": max(1, end_frame - start_frame),
            "effect": entry.recommended_effect,
            # Real video already has its own motion — the Ken Burns "slow
            # zoom" effect must only apply to static sources (real photos,
            # AI-generated images, or a placeholder's black frame), never on
            # top of playing video (see ClipLayer in Assembly.tsx).
            "mediaType": entry.asset_metadata.media_type if entry.asset_metadata else "image",
            "transition": _segment_transition(entry.transition),
        })

    total_duration_frames = (segments[-1]["startFrame"] + segments[-1]["durationInFrames"]) if segments else 0

    words = build_word_timings(timeline, whisper_words=whisper_words)
    word_props = []
    for word in words:
        start_frame = _to_frame(word.start)
        end_frame = _to_frame(word.end)
        word_props.append({
            "text": word.text,
            "startFrame": start_frame,
            "durationInFrames": max(1, end_frame - start_frame),
        })

    return {
        "segments": segments,
        "words": word_props,
        "style": {
            "pacing": style.pacing,
            "transitionStyle": style.transition_style,
            "emphasis": style.caption_emphasis,
        },
        "fps": _FPS,
        "width": _WIDTH,
        "height": _HEIGHT,
        "totalDurationInFrames": total_duration_frames,
        "includeCaptions": include_captions,
    }


def _stage_render_assets(project_dir: Path, segments: list[dict]) -> Path:
    """Copies only the exact-duration rendered segments Remotion actually
    needs into a small scratch dir, instead of pointing --public-dir at the
    whole project directory. The project dir also holds full raw/un-trimmed
    downloads, prior final_video.mp4 outputs, etc. — on Windows @remotion/
    bundler copies (never symlinks) everything under --public-dir into the
    bundle on every render, so this cuts what gets copied roughly in half
    and shrinks the window for the transient copy/AV-scan race above."""
    stage_dir = project_dir / "_remotion_stage"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    for segment in segments:
        src = project_dir / segment["path"]
        dest = stage_dir / segment["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    return stage_dir


def _run_render_script(props_path: Path, output_path: Path, public_dir: Path) -> subprocess.CompletedProcess:
    remotion_dir = settings.remotion_dir.resolve()
    render_script = remotion_dir / "render.mjs"
    return subprocess.run(
        [
            "node", str(render_script),
            f"--props={props_path.resolve()}",
            f"--output={output_path.resolve()}",
            f"--public-dir={public_dir.resolve()}",
        ],
        # cwd MUST be the remotion project dir. Without it, node inherits
        # whatever cwd this Python process happens to be running from (e.g.
        # the repo root under uvicorn) and @remotion/bundler resolves its
        # project root from THAT instead — causing an intermittent 404 for a
        # file that genuinely exists (empirically isolated: identical staged
        # inputs succeed with this cwd set and fail without it, reliably,
        # every time — not GPU, not a file-write race).
        capture_output=True, cwd=str(remotion_dir), timeout=settings.remotion_render_timeout_seconds,
    )


def render_final_video(
    project_dir: Path, timeline: list[TimelineEntry], style: StyleDecision,
    whisper_words: list[WordTiming] | None = None, include_captions: bool = False,
) -> Path:
    """Renders the final downloadable video via Remotion: composites the
    already-exact-duration per-row segments assemble_video produced, with
    transitions in one pass. Consumes clip timing (start/end/duration) and
    word timing as fixed inputs only — never recomputes them, only converts
    seconds to frame numbers. Passing whisper_words (real ASR — see
    app/transcription.py) makes captions use real transcript timing instead
    of the word-count estimate, if include_captions is enabled. Caption
    words are always computed (cheap, and kept available for a future
    re-enable) but only rendered onto the video when include_captions=True."""
    props = _build_props(project_dir, timeline, style, whisper_words=whisper_words, include_captions=include_captions)

    props_path = project_dir / "remotion_props.json"
    props_path.write_text(json.dumps(props, indent=2), encoding="utf-8")

    output_path = project_dir / "final_video_remotion.mp4"
    stage_dir = _stage_render_assets(project_dir, props["segments"])

    try:
        result = None
        for attempt in range(1, _MAX_RENDER_ATTEMPTS + 1):
            result = _run_render_script(props_path, output_path, stage_dir)
            if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                break
            if attempt < _MAX_RENDER_ATTEMPTS:
                logger.warning(
                    "remotion_render: attempt %d/%d failed (exit %s), retrying in %.0fs: %s",
                    attempt, _MAX_RENDER_ATTEMPTS, result.returncode,
                    _RETRY_DELAY_SECONDS, result.stderr.decode(errors="replace")[-500:],
                )
                time.sleep(_RETRY_DELAY_SECONDS)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)

    if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        raise RemotionRenderError(
            f"remotion render failed after {_MAX_RENDER_ATTEMPTS} attempt(s) (exit {result.returncode}): "
            f"{result.stderr.decode(errors='replace')[-2000:]}"
        )

    logger.info("remotion_render: wrote %s (%d clips, %d words, style=%s)",
                output_path, len(props["segments"]), len(props["words"]), props["style"])
    return output_path
