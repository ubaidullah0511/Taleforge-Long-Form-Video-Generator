"""Offline test for app.downloader.download_video_trimmed. No ffmpeg/network needed
(the sync ffmpeg helper itself is mocked). Run with: pytest tests/test_downloader.py
"""
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app import downloader


def test_direct_url_trim_succeeds_without_falling_back():
    with tempfile.TemporaryDirectory() as tmpdir:
        dest_dir = Path(tmpdir)

        def fake_trim_sync(input_arg, dest_path, max_seconds):
            dest_path.write_bytes(b"trimmed video")
            return True

        with patch("app.downloader._ffmpeg_trim_sync", side_effect=fake_trim_sync), \
             patch("app.downloader.download", new=AsyncMock()) as mock_full_download:
            path = asyncio.run(
                downloader.download_video_trimmed("http://x/v.mp4", "clip001_video1.mp4", dest_dir, 5.0)
            )

        assert Path(path).read_bytes() == b"trimmed video"
        mock_full_download.assert_not_called()  # direct-from-URL trim worked, no fallback needed


def test_falls_back_to_full_download_when_direct_trim_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        dest_dir = Path(tmpdir)
        raw_path = dest_dir / "raw_clip001_video1.mp4"
        raw_path.write_bytes(b"full raw video")
        attempts = {"n": 0}

        def fake_trim_sync(input_arg, dest_path, max_seconds):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return False  # direct-URL attempt fails
            dest_path.write_bytes(b"trimmed from local file")
            return True  # local-file fallback attempt succeeds

        with patch("app.downloader._ffmpeg_trim_sync", side_effect=fake_trim_sync), \
             patch("app.downloader.download", new=AsyncMock(return_value=(str(raw_path), "fp"))):
            path = asyncio.run(
                downloader.download_video_trimmed("http://x/v.mp4", "clip001_video1.mp4", dest_dir, 5.0)
            )

        assert Path(path).read_bytes() == b"trimmed from local file"
        assert not raw_path.exists()  # raw intermediate file cleaned up
        assert attempts["n"] == 2


def test_local_file_trim_failure_raises_instead_of_hitting_network():
    # local_library hands us a plain filesystem path (no http:// scheme). If
    # ffmpeg fails to trim it directly, there's nothing to "download" — a
    # network fallback would previously crash with httpx.UnsupportedProtocol
    # instead of surfacing a clear, catchable error.
    with tempfile.TemporaryDirectory() as tmpdir:
        dest_dir = Path(tmpdir)
        local_path = str(dest_dir / "source.mp4")

        with patch("app.downloader._ffmpeg_trim_sync", return_value=False), \
             patch("app.downloader.download", new=AsyncMock()) as mock_full_download:
            with pytest.raises(RuntimeError, match="ffmpeg trim failed"):
                asyncio.run(
                    downloader.download_video_trimmed(local_path, "clip001_video1.mp4", dest_dir, 5.0)
                )

        mock_full_download.assert_not_called()  # never attempted an httpx GET on a local path


if __name__ == "__main__":
    test_direct_url_trim_succeeds_without_falling_back()
    test_falls_back_to_full_download_when_direct_trim_fails()
    test_local_file_trim_failure_raises_instead_of_hitting_network()
    print("OK")
