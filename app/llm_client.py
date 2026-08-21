import base64
import json
import logging
import re
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Callable, TypeVar

from openai import OpenAI, RateLimitError

from app.config import settings
from app.ffmpeg_utils import get_ffmpeg_path, get_ffprobe_path

logger = logging.getLogger(__name__)

_RATE_LIMIT_WINDOW = 60.0  # seconds
_call_times: deque[float] = deque()
_rate_lock = threading.Lock()

_T = TypeVar("_T")
# On a 429, the API reports a short suggested wait (retry-after header or
# "try again in Xs" in the message) rather than requiring a full reset —
# worth sleeping and retrying instead of failing the whole pipeline run.
# Capped so a real outage still fails fast-ish rather than hanging forever.
_RATE_LIMIT_MAX_RETRIES = 5
_RATE_LIMIT_MAX_WAIT = 300.0  # seconds
_RETRY_AFTER_RE = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s")
# The API's own retry-after (header or message) is calibrated for short
# per-request throttling and is often ~1s — nowhere near long enough to clear
# a tokens-per-minute cap exhausted by a burst of multi-image vision calls.
# This floor, combined with the exponential multiplier below, makes repeated
# hits back off meaningfully instead of retrying every ~1s until the retry
# budget is exhausted.
_RATE_LIMIT_BASE_WAIT = 5.0
_RATE_LIMIT_BACKOFF_MULTIPLIER = 2.0


def _retry_after_seconds(exc: RateLimitError) -> float:
    header = exc.response.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    match = _RETRY_AFTER_RE.search(str(exc))
    if match:
        minutes, seconds = match.groups()
        return (float(minutes) if minutes else 0.0) * 60 + float(seconds)
    return 30.0


def _call_with_rate_limit_retry(fn: Callable[[], _T]) -> _T:
    for attempt in range(_RATE_LIMIT_MAX_RETRIES):
        try:
            return fn()
        except RateLimitError as exc:
            # insufficient_quota means no billing/credit on the account —
            # waiting never helps, so don't burn retries on it.
            if getattr(exc, "code", None) == "insufficient_quota":
                raise
            wait = min(
                max(_retry_after_seconds(exc), _RATE_LIMIT_BASE_WAIT) * (_RATE_LIMIT_BACKOFF_MULTIPLIER ** attempt),
                _RATE_LIMIT_MAX_WAIT,
            )
            if attempt == _RATE_LIMIT_MAX_RETRIES - 1:
                raise
            logger.warning(
                "llm_client: rate limit hit, waiting %.0fs (attempt %d/%d)",
                wait, attempt + 1, _RATE_LIMIT_MAX_RETRIES,
            )
            time.sleep(wait)
    raise AssertionError("unreachable")


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
def _client() -> OpenAI:
    # max_retries=0: the SDK's own built-in auto-retry otherwise fires (with
    # its own short internal backoff) BEFORE a RateLimitError ever reaches
    # _call_with_rate_limit_retry's except clause below — two independent
    # retry loops stacking meant requests were firing faster than our own
    # throttle/backoff intended, worsening exactly the 429s it exists to
    # avoid. This makes _call_with_rate_limit_retry the sole retry authority.
    return OpenAI(api_key=settings.openai_api_key, max_retries=0)


def generate_json(prompt: str, model: str) -> dict:
    """Returns a JSON *object* — json_object mode enforces an object at
    the root, so any prompt that logically wants a list asks for it wrapped
    (e.g. {"scenes": [...]}) and the caller unwraps it."""
    _throttle()
    response = _call_with_rate_limit_retry(lambda: _client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    ))
    return json.loads(response.choices[0].message.content)


def generate_json_from_images(prompt: str, images: list[bytes], model: str, mime_type: str = "image/jpeg") -> dict:
    """Inline-image vision call — used to describe a still image or a handful
    of sampled video frames. Capped at 5 images per request (our own
    conservative cap, not an API-imposed one)."""
    _throttle()
    content = [{"type": "text", "text": prompt}]
    for img in images[:5]:
        b64 = base64.b64encode(img).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}", "detail": "low"}})
    response = _call_with_rate_limit_retry(lambda: _client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        response_format={"type": "json_object"},
    ))
    return json.loads(response.choices[0].message.content)


def _probe_duration(video_path: str) -> float:
    result = subprocess.run(
        [get_ffprobe_path(), "-v", "error", "-print_format", "json", "-show_entries", "format=duration", video_path],
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
            [get_ffmpeg_path(), "-y", "-ss", str(ts), "-i", video_path, "-frames:v", "1", "-q:v", "2", str(tmp_path)],
            capture_output=True,
        )
        if tmp_path.exists() and tmp_path.stat().st_size > 0:
            frames.append(tmp_path.read_bytes())
        tmp_path.unlink(missing_ok=True)
    return frames


# ponytail: OpenAI's chat completions API has no video-file ingestion (unlike
# Gemini's files.upload), so a full clip is described from a few locally-
# sampled frames (start/middle/end) through the vision endpoint instead — same
# technique app.asset_selection uses for remote candidates, just sampling a
# local file with ffmpeg.
def generate_json_from_video(prompt: str, video_path: str, model: str) -> dict:
    frames = sample_local_frames(video_path)
    if not frames:
        raise RuntimeError(f"could not sample any frames from {video_path}")
    return generate_json_from_images(prompt, frames, model)
