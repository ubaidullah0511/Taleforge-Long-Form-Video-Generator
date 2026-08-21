"""Resolves the ffmpeg/ffprobe binaries every ffmpeg-shelling module in this
project uses. The static build bundled at bin/ (see bin/FFMPEG-README.txt for
its exact version and bin/FFMPEG-LICENSE.txt for its GPL v3 terms) takes
priority, so an installed end user never depends on ffmpeg being separately
installed or present on PATH. Falls back to PATH's `ffmpeg`/`ffprobe` so a
developer machine that already has one globally installed doesn't need a
second copy.

Importing this module also prepends bin/ to the process's PATH (idempotent,
safe to repeat). That's what makes the bundled copy reachable to code this
project doesn't control -- namely `whisper.audio.load_audio()`, which shells
out to a bare "ffmpeg" on PATH internally and can't be pointed at an explicit
path without patching the whisper package itself.
"""
import os
import shutil
from functools import lru_cache
from pathlib import Path

_BIN_DIR = Path(__file__).resolve().parent.parent / "bin"


def _ensure_bundled_dir_on_path() -> None:
    if not _BIN_DIR.is_dir():
        return
    bin_str = str(_BIN_DIR)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if bin_str not in path_entries:
        os.environ["PATH"] = os.pathsep.join([bin_str, *path_entries])


_ensure_bundled_dir_on_path()


def _resolve(name: str) -> str:
    bundled = _BIN_DIR / f"{name}.exe"
    if bundled.is_file():
        return str(bundled)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"{name} not found: no bundled copy at {bundled} and no {name!r} on PATH. "
        "Reinstall, or run setup.bat, to restore the bundled binary."
    )


@lru_cache(maxsize=1)
def get_ffmpeg_path() -> str:
    """Absolute path to the bundled ffmpeg.exe, or the PATH-resolved one if
    the bundled copy isn't present (dev machines with ffmpeg already
    installed globally)."""
    return _resolve("ffmpeg")


@lru_cache(maxsize=1)
def get_ffprobe_path() -> str:
    """ffprobe counterpart to get_ffmpeg_path()."""
    return _resolve("ffprobe")
