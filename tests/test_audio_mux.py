"""Integration test for app.audio_mux — exercises real ffmpeg (synthetic
lavfi video + sine-wave audio, no network). Run with: pytest tests/test_audio_mux.py
"""
import subprocess
import tempfile
from pathlib import Path

from app.audio_mux import AudioMuxError, mux_narration_audio
from app.clip_ingest import probe


def _make_silent_video(path: Path, duration: float) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=blue:s=640x360:r=30:d={duration}",
         "-pix_fmt", "yuv420p", "-an", str(path)],
        capture_output=True, check=True,
    )


def _make_tone_audio(path: Path, duration: float) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}", str(path)],
        capture_output=True, check=True,
    )


def _has_audio_stream(path: Path) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return "audio" in result.stdout


def test_mux_keeps_full_video_length_when_video_covers_the_whole_audio():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        video = tmp / "video.mp4"
        audio = tmp / "audio.wav"
        _make_silent_video(video, 5.0)
        _make_tone_audio(audio, 3.0)

        output = mux_narration_audio(video, str(audio), tmp / "out.mp4")

        assert output.exists()
        assert _has_audio_stream(output)
        duration, _res = probe(str(output))
        assert abs(duration - 5.0) < 0.2  # video's own full length preserved, not cut to audio's 3s


def test_mux_extends_video_by_freezing_last_frame_when_audio_is_longer():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        video = tmp / "video.mp4"
        audio = tmp / "audio.wav"
        _make_silent_video(video, 3.0)
        _make_tone_audio(audio, 6.0)

        output = mux_narration_audio(video, str(audio), tmp / "out.mp4")

        assert output.exists()
        assert _has_audio_stream(output)
        duration, _res = probe(str(output))
        assert abs(duration - 6.0) < 0.3  # video extended (frozen last frame) to cover the full audio


def test_mux_raises_audio_mux_error_on_ffmpeg_failure():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        video = tmp / "video.mp4"
        _make_silent_video(video, 2.0)
        bad_audio = tmp / "not_real_audio.wav"
        bad_audio.write_bytes(b"this is not a real audio file")

        try:
            mux_narration_audio(video, str(bad_audio), tmp / "out.mp4")
            raise AssertionError("expected AudioMuxError")
        except AudioMuxError:
            pass


if __name__ == "__main__":
    test_mux_keeps_full_video_length_when_video_covers_the_whole_audio()
    test_mux_extends_video_by_freezing_last_frame_when_audio_is_longer()
    test_mux_raises_audio_mux_error_on_ffmpeg_failure()
    print("OK")
