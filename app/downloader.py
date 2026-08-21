import asyncio
import hashlib
import logging
import re
import subprocess
from pathlib import Path

import httpx

from app.config import settings
from app.ffmpeg_utils import get_ffmpeg_path

logger = logging.getLogger(__name__)

_URL_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


async def download(url: str, dest_filename: str, dest_dir: Path | None = None) -> tuple[str, str]:
    """Download url to dest_dir/<dest_filename> (default clips/downloaded/). Returns (path, sha256 fingerprint)."""
    dest_dir = dest_dir or settings.downloaded_clips_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / dest_filename
    async with httpx.AsyncClient(follow_redirects=True) as client:
        async with client.stream("GET", url, timeout=60) as resp:
            resp.raise_for_status()
            digest = hashlib.sha256()
            with open(dest_path, "wb") as f:
                async for chunk in resp.aiter_bytes():
                    f.write(chunk)
                    digest.update(chunk)
    return str(dest_path), digest.hexdigest()


def _ffmpeg_trim_sync(input_arg: str, dest_path: Path, max_seconds: float) -> bool:
    # -t before -i limits how much of the INPUT is read, not just the output —
    # for a network URL that means ffmpeg stops fetching once it has enough.
    result = subprocess.run(
        [get_ffmpeg_path(), "-y", "-t", str(max_seconds), "-i", input_arg, "-c", "copy", str(dest_path)],
        capture_output=True,
    )
    ok = result.returncode == 0 and dest_path.exists() and dest_path.stat().st_size > 0
    if not ok:
        logger.warning(
            "downloader: ffmpeg trim failed for %r (returncode=%s): %s",
            input_arg, result.returncode, result.stderr.decode(errors="replace")[-500:],
        )
    return ok


# ponytail: subprocess.run in a thread, not asyncio.create_subprocess_exec —
# uvicorn's worker can end up on Windows' Selector event loop, which raises
# NotImplementedError for any asyncio-native subprocess. Running the blocking
# call via asyncio.to_thread sidesteps the loop's subprocess transport entirely.
async def _ffmpeg_trim(input_arg: str, dest_path: Path, max_seconds: float) -> bool:
    return await asyncio.to_thread(_ffmpeg_trim_sync, input_arg, dest_path, max_seconds)


async def download_video_trimmed(url: str, dest_filename: str, dest_dir: Path, max_seconds: float) -> str:
    """Fetch url and cap it to max_seconds via ffmpeg, reading straight off the
    network so a long source video isn't fully downloaded just to keep a few
    seconds of it. Falls back to a full download + local trim if the source
    doesn't support ffmpeg reading/cutting it directly over HTTP."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / dest_filename

    if await _ffmpeg_trim(url, dest_path, max_seconds):
        return str(dest_path)

    if not _URL_SCHEME_RE.match(url):
        # Local sources (e.g. local_library) hand us a plain filesystem path,
        # not a fetchable URL — there's nothing to download over the network,
        # so a failed direct trim here is a real, final failure (see the
        # ffmpeg stderr logged in _ffmpeg_trim_sync above), not a case for
        # the network-download fallback below.
        raise RuntimeError(f"ffmpeg trim failed for local file {url!r}")

    # ponytail: some sources (odd redirects, no progressive playback) don't let
    # ffmpeg trim straight off the network — fall back to a full download first.
    raw_path, _fingerprint = await download(url, f"raw_{dest_filename}", dest_dir=dest_dir)
    try:
        if not await _ffmpeg_trim(raw_path, dest_path, max_seconds):
            raise RuntimeError(f"ffmpeg trim failed for {url}")
    finally:
        Path(raw_path).unlink(missing_ok=True)
    return str(dest_path)
