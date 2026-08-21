import logging
import subprocess
from pathlib import Path

from app.clip_ingest import probe
from app.ffmpeg_utils import get_ffmpeg_path

logger = logging.getLogger(__name__)

_DURATION_EPSILON = 0.05


class AudioMuxError(Exception):
    """Raised when muxing the uploaded narration audio into the rendered
    video fails — never silently ships a "final" video missing the
    soundtrack it was explicitly supposed to have."""


def mux_narration_audio(video_path: Path, audio_path: str, output_path: Path) -> Path:
    """Combines the already-rendered (silent) video with the manually
    supplied narration audio, in full and unmodified — no trimming, no
    re-timing. Clip durations and captions were already timed against this
    exact audio's real Whisper timestamps (see app/transcription.py), so
    muxing it back in starting at t=0 lines up by construction.

    If the video is shorter than the audio (e.g. trailing narration beyond
    the table's last clip — see documentary_table.py's "trailing whisper
    words" warning), the last video frame is held (frozen) to cover the
    remainder via ffmpeg's tpad filter, rather than cutting the audio off
    early. If the video is the same length or longer, the audio is muxed in
    as-is with no changes to the video at all."""
    try:
        video_duration, _ = probe(str(video_path))
        audio_duration, _ = probe(audio_path)
    except Exception as exc:
        raise AudioMuxError(f"could not probe video/audio duration before muxing: {exc}") from exc

    if audio_duration > video_duration + _DURATION_EPSILON:
        pad_seconds = audio_duration - video_duration
        logger.info(
            "audio_mux: video (%.2fs) is shorter than the narration audio (%.2fs) — "
            "holding the last frame for %.2fs so no audio gets cut off",
            video_duration, audio_duration, pad_seconds,
        )
        args = [
            get_ffmpeg_path(), "-y", "-i", str(video_path), "-i", str(audio_path),
            "-filter_complex", f"[0:v]tpad=stop_mode=clone:stop_duration={pad_seconds}[v]",
            "-map", "[v]", "-map", "1:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-shortest", str(output_path),
        ]
    else:
        # Video already covers the full audio — no video changes needed at
        # all. Deliberately no -shortest here: the video must keep its full
        # length exactly as the table decided, even past where narration ends.
        args = [
            get_ffmpeg_path(), "-y", "-i", str(video_path), "-i", str(audio_path),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            str(output_path),
        ]

    result = subprocess.run(args, capture_output=True)
    if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        raise AudioMuxError(
            f"ffmpeg failed to mux narration audio into the final video: "
            f"{result.stderr.decode(errors='replace')[-2000:]}"
        )

    logger.info("audio_mux: wrote %s (video %.2fs, audio %.2fs)", output_path, video_duration, audio_duration)
    return output_path
