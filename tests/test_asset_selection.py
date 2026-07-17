"""Offline test for app.asset_selection.select_best_asset's ranking logic.
No LLM/network/ffmpeg needed — candidate embeddings are supplied directly
and the query embedding is mocked. Run with: pytest tests/test_asset_selection.py
"""
import tempfile
from pathlib import Path
from unittest.mock import patch

from app.asset_selection import Candidate, _is_horizontal_16_9, cleanup_rejected, local_candidates, select_best_asset
from app.config import settings
from app.models import ClipRecord
from app.stock.base import StockHit


def _candidate(source, media_type, embedding, width=1920, height=1080, duration=10.0, source_id="x"):
    hit = StockHit(
        source=source, source_id=source_id, download_url="http://x", width=width, height=height,
        duration=duration, media_type=media_type,
    )
    return Candidate(
        source=source, media_type=media_type, description="d", keywords=["k"], analysis={},
        width=width, height=height, duration=duration, embedding=embedding, hit=hit,
    )


def test_returns_none_for_empty_candidates():
    assert select_best_asset([], "scene text", "video") is None


def test_picks_highest_semantic_score_when_media_type_matches():
    video = _candidate("pexels", "video", [1.0, 0.0])
    image = _candidate("pexels_images", "image", [0.5, 0.5])
    with patch("app.asset_selection.embed", return_value=[1.0, 0.0]):
        selection = select_best_asset([video, image], "scene text", "video")
    assert selection.candidate is video


def test_media_type_override_when_other_type_scores_significantly_higher():
    video = _candidate("pexels", "video", [1.0, 0.0])          # cosine 1.0
    image = _candidate("pexels_images", "image", [0.5, 0.0])   # cosine 0.5
    with patch("app.asset_selection.embed", return_value=[1.0, 0.0]), \
         patch.object(settings, "asset_selection_score_margin", 0.05):
        selection = select_best_asset([video, image], "scene text", "image")
    assert selection.candidate is video  # image is far below margin, overall best wins instead
    assert "no 'image' candidate within" in selection.reason


def test_media_type_preference_kept_when_within_margin():
    video = _candidate("pexels", "video", [0.90, 0.0])
    image = _candidate("pexels_images", "image", [0.87, 0.0])
    with patch("app.asset_selection.embed", return_value=[1.0, 0.0]), \
         patch.object(settings, "asset_selection_score_margin", 0.05):
        selection = select_best_asset([video, image], "scene text", "image")
    assert selection.candidate is image  # within margin of the top score, preference wins


def test_resolution_used_as_tiebreaker_on_near_equal_scores():
    low_res = _candidate("pexels", "video", [1.0, 0.0], width=640, height=360, source_id="low")
    high_res = _candidate("pixabay", "video", [1.0, 0.0], width=1920, height=1080, source_id="high")
    with patch("app.asset_selection.embed", return_value=[1.0, 0.0]):
        selection = select_best_asset([low_res, high_res], "scene text", "video")
    assert selection.candidate is high_res
    assert "tiebreaker" in selection.reason


def test_target_duration_filters_out_short_videos():
    short = _candidate("pexels", "video", [1.0, 0.0], duration=3.0, source_id="short")
    long = _candidate("pixabay", "video", [0.6, 0.0], duration=10.0, source_id="long")
    with patch("app.asset_selection.embed", return_value=[1.0, 0.0]):
        selection = select_best_asset([short, long], "scene text", "video", target_duration=5.0)
    assert selection.candidate is long  # short one filtered out despite a higher raw score


def test_target_duration_does_not_discard_all_candidates_if_none_qualify():
    short = _candidate("pexels", "video", [1.0, 0.0], duration=2.0, source_id="short")
    with patch("app.asset_selection.embed", return_value=[1.0, 0.0]):
        selection = select_best_asset([short], "scene text", "video", target_duration=5.0)
    assert selection.candidate is short  # nothing else available, keep it rather than return None


def test_cleanup_rejected_deletes_losers_but_keeps_winner():
    with tempfile.TemporaryDirectory() as tmpdir:
        winner_file = Path(tmpdir) / "winner.mp4"
        loser_file = Path(tmpdir) / "loser.mp4"
        winner_file.write_bytes(b"w")
        loser_file.write_bytes(b"l")

        winner = _candidate("pexels", "video", [1.0, 0.0], source_id="winner")
        winner.local_path = winner_file
        loser = _candidate("pixabay", "video", [0.5, 0.0], source_id="loser")
        loser.local_path = loser_file

        cleanup_rejected([winner, loser], winner)

        assert winner_file.exists()
        assert not loser_file.exists()


def test_is_horizontal_16_9_accepts_exact_and_near_matches():
    assert _is_horizontal_16_9(1920, 1080) is True  # exact
    assert _is_horizontal_16_9(1920, 1088) is True   # mod-16 encode padding, within tolerance
    assert _is_horizontal_16_9(1280, 720) is True


def test_is_horizontal_16_9_rejects_non_horizontal_aspects():
    assert _is_horizontal_16_9(1080, 1920) is False  # vertical
    assert _is_horizontal_16_9(1080, 1080) is False  # square
    assert _is_horizontal_16_9(1440, 1080) is False  # 4:3
    assert _is_horizontal_16_9(0, 0) is False


def _local_clip(resolution, clip_id):
    return ClipRecord(
        id=clip_id, video_path="clips/local/x.mp4", start=0, end=5, duration=5,
        description="d", keywords=["k"], resolution=resolution, source="local", media_type="video",
    )


def test_local_candidates_filters_out_non_16_9_clips():
    horizontal = _local_clip("1920x1080", "h")
    vertical = _local_clip("1080x1920", "v")
    with patch("app.asset_selection.embed", return_value=[1.0, 0.0]):
        candidates = local_candidates([(horizontal, 0.9), (vertical, 0.9)])
    assert len(candidates) == 1
    assert candidates[0].clip.id == "h"


if __name__ == "__main__":
    test_returns_none_for_empty_candidates()
    test_picks_highest_semantic_score_when_media_type_matches()
    test_media_type_override_when_other_type_scores_significantly_higher()
    test_media_type_preference_kept_when_within_margin()
    test_resolution_used_as_tiebreaker_on_near_equal_scores()
    test_target_duration_filters_out_short_videos()
    test_target_duration_does_not_discard_all_candidates_if_none_qualify()
    test_cleanup_rejected_deletes_losers_but_keeps_winner()
    test_is_horizontal_16_9_accepts_exact_and_near_matches()
    test_is_horizontal_16_9_rejects_non_horizontal_aspects()
    test_local_candidates_filters_out_non_16_9_clips()
    print("OK")
