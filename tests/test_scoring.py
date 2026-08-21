"""Offline tests for app.scoring — no network/API keys needed
(embed() is mocked, same pattern as tests/test_asset_selection.py).
Run with: pytest tests/test_scoring.py
"""
from unittest.mock import patch

from app.models import TimelineClip
from app.scoring import _duration_adequacy, score_asset
from app.stock.base import StockHit


def _clip() -> TimelineClip:
    return TimelineClip(
        clip_number=1, section="Intro", script_beat="A tennis player serves",
        canva_keyword="tennis player serving", fallback_keyword="tennis",
        visual_type="Sports", edit_note="",
    )


def _hit(duration: float, media_type: str = "video", source: str = "pexels", text: str = "a tennis player serving on a court") -> StockHit:
    return StockHit(
        source=source, source_id="x", download_url="http://x/v.mp4",
        width=1920, height=1080, duration=duration, media_type=media_type,
        text=text,
    )


def test_duration_adequacy_penalizes_sub_floor_video_clips():
    assert _duration_adequacy(_hit(0.14)) < _duration_adequacy(_hit(2.0))
    assert _duration_adequacy(_hit(2.0)) == 1.0
    assert _duration_adequacy(_hit(10.0)) == 1.0  # capped at 1.0, no bonus for being longer


def test_duration_adequacy_ignores_images():
    """Images are extended (-loop 1) at render time, not source-limited like
    video, so a missing/zero duration on an image candidate must not be
    penalized the way it would be for a video."""
    assert _duration_adequacy(_hit(0, media_type="image")) == 1.0


def test_score_asset_prefers_longer_clip_when_otherwise_equal():
    """Bug report: a real 0.14s local_library clip won as the top candidate
    for a 3s beat and had to loop ~20x in the final render ("repeats a
    1-second segment instead of playing full content"). Duration data itself
    was verified correct (index.json/ffprobe matched exactly across 1298
    indexed files) — the actual gap was score_asset having no opinion on
    clip length at all. With two otherwise-identical candidates, the one
    that needs far fewer loop cycles to fill a realistic beat must now win."""
    clip = _clip()
    with patch("app.scoring.embed", return_value=[1.0, 0.0]):
        score_short = score_asset(_hit(0.14), clip, [1.0, 0.0])
        score_long = score_asset(_hit(3.0), clip, [1.0, 0.0])
    assert score_long > score_short


def test_score_asset_still_gives_short_clip_a_nonzero_score():
    """Not a hard cutoff — a very short clip must still score > 0 so it can
    win when it's genuinely the only usable candidate."""
    clip = _clip()
    with patch("app.scoring.embed", return_value=[1.0, 0.0]):
        score = score_asset(_hit(0.05), clip, [1.0, 0.0])
    assert score > 0


if __name__ == "__main__":
    test_duration_adequacy_penalizes_sub_floor_video_clips()
    test_duration_adequacy_ignores_images()
    test_score_asset_prefers_longer_clip_when_otherwise_equal()
    test_score_asset_still_gives_short_clip_a_nonzero_score()
    print("OK")
