import base64
import json
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from functools import lru_cache
from pathlib import Path

from groq import Groq

from app.config import settings

_RATE_LIMIT_WINDOW = 60.0  # seconds
_call_times: deque[float] = deque()
_rate_lock = threading.Lock()


# ponytail: sliding-window rate limiter (not a token-bucket library) — this is
# the single chokepoint every LLM call goes through (scene segmentation, scene
# understanding, clip analysis, documentary table generation, semantic keyword
# fallback), so throttling here covers all of them without touching each caller.
def _throttle() -> None:
    limit = settings.llm_max_requests_per_minute
    with _rate_lock:
        now = time.monotonic()
        while _call_times and now - _call_times[0] >= _RATE_LIMIT_WINDOW:
            _call_times.popleft()
        if len(_call_times) >= limit:
            time.sleep(max(0.0, _RATE_LIMIT_WINDOW - (now - _call_times[0])))
            now = time.monotonic()
            while _call_times and now - _call_times[0] >= _RATE_LIMIT_WINDOW:
                _call_times.popleft()
        _call_times.append(time.monotonic())


@lru_cache(maxsize=1)
def _client() -> Groq:
    return Groq(api_key=settings.groq_api_key)


def generate_json(prompt: str, model: str) -> dict:
    """Returns a JSON *object* — Groq's json_object mode enforces an object at
    the root, so any prompt that logically wants a list asks for it wrapped
    (e.g. {"scenes": [...]}) and the caller unwraps it."""
    _throttle()
    response = _client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def generate_json_from_images(prompt: str, images: list[bytes], model: str, mime_type: str = "image/jpeg") -> dict:
    """Inline-image vision call — used to describe a still image or a handful
    of sampled video frames. Groq caps requests at 5 images."""
    _throttle()
    content = [{"type": "text", "text": prompt}]
    for img in images[:5]:
        b64 = base64.b64encode(img).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}})
    response = _client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def transcribe_audio(audio_path: str, model: str) -> dict:
    """Raw Groq Whisper call — verbose_json + word-level timestamps. Requests
    both "word" and "segment" granularities (not just "word") since some
    client wrappers have reported empty results when requesting word-only;
    the caller only ever reads the "words" field. Returns the full response
    as a plain dict via model_dump() rather than the SDK's Transcription
    object, so callers don't depend on SDK-internal attribute access."""
    _throttle()
    with open(audio_path, "rb") as f:
        response = _client().audio.transcriptions.create(
            model=model,
            file=(Path(audio_path).name, f.read()),
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )
    return response.model_dump()


def _probe_duration(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_entries", "format=duration", video_path],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(result.stdout)
    return float(data["format"].get("duration", 0) or 0)


# Public: also used by app.asset_selection to sample fully-downloaded stock
# candidates (see that module's docstring for why frames are sampled locally
# instead of via remote ffmpeg seeking).
def sample_local_frames(video_path: str, max_frames: int = 3) -> list[bytes]:
    duration = _probe_duration(video_path)
    fractions = (0.0, 0.5, 0.9) if duration > 0 else (0.0,)
    timestamps = sorted({max(0.0, min(duration - 0.1, duration * f)) for f in fractions})

    frames = []
    for ts in timestamps[:max_frames]:
        tmp_path = Path(tempfile.gettempdir()) / f"frame_{uuid.uuid4().hex}.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(ts), "-i", video_path, "-frames:v", "1", "-q:v", "2", str(tmp_path)],
            capture_output=True,
        )
        if tmp_path.exists() and tmp_path.stat().st_size > 0:
            frames.append(tmp_path.read_bytes())
        tmp_path.unlink(missing_ok=True)
    return frames


# ponytail: Groq has no video-file ingestion API (unlike Gemini's files.upload),
# so a full clip is described from a few locally-sampled frames (start/middle/
# end) through the vision endpoint instead — same technique app.asset_selection
# uses for remote candidates, just sampling a local file with ffmpeg.
def generate_json_from_video(prompt: str, video_path: str, model: str) -> dict:
    frames = sample_local_frames(video_path)
    if not frames:
        raise RuntimeError(f"could not sample any frames from {video_path}")
    return generate_json_from_images(prompt, frames, model)
