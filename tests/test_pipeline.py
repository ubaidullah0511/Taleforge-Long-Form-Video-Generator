"""Smoke test for the branchy local-hit / stock-fallback logic in app.pipeline.
Mocks the LLM, embeddings, vector store and stock providers so it runs offline
with no API keys. Run with: pytest tests/test_pipeline.py
"""
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.asset_selection import Candidate, Selection
from app.config import settings
from app.models import Scene, SceneAnalysis, ClipRecord
from app.stock.base import StockHit
from app import pipeline


def _clip(source="local", media_type="video"):
    return ClipRecord(
        id="c1", video_path="clips/local/forest.mp4", start=12, end=18,
        duration=6, description="forest", keywords=["forest"], source=source,
        media_type=media_type, resolution="1920x1080",
    )


def test_local_match_returned_when_similarity_high():
    scene = Scene(scene=1, text="The Amazon rainforest is home to millions of species.")
    with patch("app.pipeline.analyze", return_value=SceneAnalysis(description="rainforest", keywords=["forest"])), \
         patch("app.pipeline.decide_media_type", return_value="video"), \
         patch("app.pipeline.query_similar", return_value=[(_clip(), 0.9)]), \
         patch("app.asset_selection.embed", return_value=[1.0, 0.0]):
        result = asyncio.run(pipeline._resolve_scene(scene, scene.text, "testrun"))
    assert result.source == "local"
    assert result.selected_clip == "clips/local/forest.mp4"
    assert result.downloaded is False
    assert result.media_type == "video"
    assert result.selection_reason


def test_low_similarity_falls_back_to_stock_and_downloads():
    # Candidate.local_path stands in for what describe_candidates() would have
    # produced: a real scratch file already fully downloaded before scoring.
    scene = Scene(scene=2, text="A businessman walking in a modern city.")
    hit = StockHit(source="pexels", source_id="999", download_url="http://example/video.mp4", width=1920, height=1080, duration=8)
    downloaded_clip = _clip(source="pexels")

    with tempfile.TemporaryDirectory() as tmpdir:
        scratch_file = Path(tmpdir) / "scratch_pexels_999.mp4"
        scratch_file.write_bytes(b"fake video bytes")
        downloaded_dir = Path(tmpdir) / "downloaded"  # settings.downloaded_clips_dir = clips_dir / "downloaded"

        candidate = Candidate(
            source="pexels", media_type="video", description="city walk", keywords=["city"],
            analysis={"description": "city walk", "keywords": ["city"]}, width=1920, height=1080,
            duration=8, embedding=[1.0, 0.0], hit=hit, local_path=scratch_file, fingerprint="abc123",
        )
        selection = Selection(candidate=candidate, score=0.9, reason="preferred media type 'video'; semantic score 0.90")

        with patch("app.pipeline.analyze", return_value=SceneAnalysis(description="city walk", keywords=["city"])), \
             patch("app.pipeline.decide_media_type", return_value="video"), \
             patch("app.pipeline.query_similar", return_value=[]), \
             patch("app.pipeline._search_stock", new=AsyncMock(return_value=[hit])), \
             patch("app.pipeline.describe_candidates", new=AsyncMock(return_value=[candidate])), \
             patch("app.pipeline.select_best_asset", return_value=selection), \
             patch("app.pipeline.get_by_source_id", return_value=None), \
             patch("app.pipeline.get_by_fingerprint", return_value=None), \
             patch.object(settings, "clips_dir", Path(tmpdir)), \
             patch("app.pipeline.analyze_and_store", return_value=downloaded_clip):
            result = asyncio.run(pipeline._resolve_scene(scene, scene.text, "testrun"))

        assert result.source == "pexels"
        assert result.downloaded is True
        assert result.selection_reason == selection.reason
        assert not scratch_file.exists()  # moved out of scratch, not left behind
        assert (downloaded_dir / "pexels_999.mp4").exists()  # winner landed in permanent store


if __name__ == "__main__":
    test_local_match_returned_when_similarity_high()
    test_low_similarity_falls_back_to_stock_and_downloads()
    print("OK")
