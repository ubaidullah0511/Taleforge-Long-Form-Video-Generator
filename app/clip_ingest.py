import json
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Literal

from app.config import settings
from app.ffmpeg_utils import get_ffmpeg_path, get_ffprobe_path
from app.llm_client import generate_json_from_video
from app.models import ClipRecord
from app.vector_store import upsert_clip

ANALYSIS_PROMPT = """Analyze this video clip for use as B-roll footage. Return \
a JSON object with keys: "description" (detailed visual description), \
"keywords" (list of 5-8 search terms), "objects" (list of visible objects), \
"camera" (camera angle/movement), "motion" (description of movement in the \
shot), "scene_type" (e.g. "nature", "urban", "office"), "mood" (one or two \
words)."""


def probe(video_path: str) -> tuple[float, str]:
    """Returns (duration_seconds, resolution) via ffprobe."""
    result = subprocess.run(
        [
            get_ffprobe_path(), "-v", "error", "-print_format", "json",
            "-show_entries", "format=duration:stream=width,height",
            video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    duration = float(data["format"].get("duration", 0) or 0)  # images have no duration entry
    stream = next((s for s in data.get("streams", []) if "width" in s), {})
    resolution = f"{stream.get('width', 0)}x{stream.get('height', 0)}"
    return duration, resolution


def _extract_segment(video_path: str, start: float, end: float) -> str:
    tmp_path = str(Path(tempfile.gettempdir()) / f"seg_{uuid.uuid4().hex}.mp4")
    subprocess.run(
        [
            get_ffmpeg_path(), "-y", "-ss", str(start), "-to", str(end),
            "-i", video_path, "-c", "copy", tmp_path,
        ],
        capture_output=True, check=True,
    )
    return tmp_path


def analyze_and_store(
    video_path: str,
    start: float,
    end: float,
    source: str,
    source_id: str = "",
    fingerprint: str = "",
    media_type: Literal["video", "image"] = "video",
    precomputed_analysis: dict | None = None,
) -> ClipRecord:
    """precomputed_analysis lets a caller (asset_selection, during candidate
    scoring) hand over an already-generated description so this doesn't
    re-call the LLM a second time on the same winning asset. Required for
    images, since there's no video-analysis fallback that applies to a still."""
    probed_duration, resolution = probe(video_path)

    if media_type == "image":
        if precomputed_analysis is None:
            raise ValueError("precomputed_analysis is required for image assets")
        analysis = precomputed_analysis
        start, end, duration = 0.0, 0.0, 0.0
    elif precomputed_analysis is not None:
        analysis = precomputed_analysis
        duration = end - start
    elif start != 0 or end != probed_duration:
        segment_path = _extract_segment(video_path, start, end)
        try:
            analysis = generate_json_from_video(ANALYSIS_PROMPT, segment_path, settings.llm_vision_model)
        finally:
            Path(segment_path).unlink(missing_ok=True)
        duration = end - start
    else:
        analysis = generate_json_from_video(ANALYSIS_PROMPT, video_path, settings.llm_vision_model)
        duration = end - start

    clip = ClipRecord(
        id=uuid.uuid4().hex,
        video_path=video_path,
        start=start,
        end=end,
        duration=duration,
        description=analysis["description"],
        keywords=analysis.get("keywords", []),
        resolution=resolution,
        source=source,
        source_id=source_id,
        fingerprint=fingerprint,
        media_type=media_type,
    )
    upsert_clip(clip)
    return clip
