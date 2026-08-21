"""Offline tests for app.ffmpeg_utils's bundled-first/PATH-fallback resolution.
No real ffmpeg needed (filesystem/PATH lookups are mocked). Run with:
pytest tests/test_ffmpeg_utils.py
"""
from pathlib import Path
from unittest.mock import patch

import pytest

from app import ffmpeg_utils


def _clear_caches():
    ffmpeg_utils.get_ffmpeg_path.cache_clear()
    ffmpeg_utils.get_ffprobe_path.cache_clear()


def test_prefers_bundled_binary_over_path():
    _clear_caches()
    try:
        with patch("app.ffmpeg_utils.Path.is_file", return_value=True), \
             patch("app.ffmpeg_utils.shutil.which") as mock_which:
            resolved = ffmpeg_utils.get_ffmpeg_path()
        assert resolved == str(ffmpeg_utils._BIN_DIR / "ffmpeg.exe")
        mock_which.assert_not_called()  # bundled copy found first, PATH never consulted
    finally:
        _clear_caches()


def test_falls_back_to_path_when_bundled_binary_missing():
    _clear_caches()
    try:
        with patch("app.ffmpeg_utils.Path.is_file", return_value=False), \
             patch("app.ffmpeg_utils.shutil.which", return_value=r"C:\Tools\ffmpeg.exe"):
            resolved = ffmpeg_utils.get_ffmpeg_path()
        assert resolved == r"C:\Tools\ffmpeg.exe"
    finally:
        _clear_caches()


def test_raises_clear_error_when_neither_bundled_nor_path_available():
    _clear_caches()
    try:
        with patch("app.ffmpeg_utils.Path.is_file", return_value=False), \
             patch("app.ffmpeg_utils.shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="ffprobe"):
                ffmpeg_utils.get_ffprobe_path()
    finally:
        _clear_caches()


def test_bundled_dir_prepended_to_path():
    import os
    bin_str = str(ffmpeg_utils._BIN_DIR)
    with patch.dict("os.environ", {"PATH": r"C:\Windows;C:\Windows\System32"}), \
         patch("app.ffmpeg_utils.Path.is_dir", return_value=True):
        ffmpeg_utils._ensure_bundled_dir_on_path()
        assert os.environ["PATH"].startswith(bin_str)


def test_ensure_bundled_dir_on_path_is_idempotent():
    import os
    with patch.dict("os.environ", {"PATH": r"C:\Windows"}), \
         patch("app.ffmpeg_utils.Path.is_dir", return_value=True):
        ffmpeg_utils._ensure_bundled_dir_on_path()
        ffmpeg_utils._ensure_bundled_dir_on_path()
        assert os.environ["PATH"].count(str(ffmpeg_utils._BIN_DIR)) == 1
