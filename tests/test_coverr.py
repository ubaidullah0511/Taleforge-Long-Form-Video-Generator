"""Offline tests for app.stock.coverr — no real network access (httpx
MockTransport stands in for api.coverr.co). Run with:
pytest tests/test_coverr.py
"""
import asyncio
from unittest.mock import patch

import httpx

from app.config import settings
from app.stock import coverr
from app.stock.coverr import _text_for

_RealAsyncClient = httpx.AsyncClient


def test_text_for_combines_title_description_tags():
    hit = {"title": "Gym Workout", "description": "A man lifting weights", "tags": ["gym", "fitness"]}
    text = _text_for(hit)
    assert "Gym Workout" in text
    assert "A man lifting weights" in text
    assert "gym" in text and "fitness" in text


def test_text_for_handles_missing_fields():
    assert _text_for({}) == ""


def _run_search(query, hits_json, per_page=5, capture_request=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if capture_request is not None:
            capture_request.append(request)
        return httpx.Response(200, json={"hits": hits_json})

    transport = httpx.MockTransport(handler)
    with patch.object(coverr.httpx, "AsyncClient", lambda **kw: _RealAsyncClient(transport=transport, **kw)), \
         patch.object(settings, "coverr_api_key", "test-key-123"):
        return asyncio.run(coverr.search(query, per_page=per_page))


def test_search_sends_correct_query_params_and_auth_header():
    captured = []
    _run_search("bodybuilder deadlift", [], per_page=7, capture_request=captured)

    assert len(captured) == 1
    request = captured[0]
    assert request.url.params["query"] == "bodybuilder deadlift"
    assert request.url.params["page"] == "0"
    assert request.url.params["page_size"] == "7"
    assert request.url.params["urls"] == "true"
    assert request.headers["Authorization"] == "Bearer test-key-123"
    assert str(request.url).startswith("https://api.coverr.co/videos")


def test_search_maps_response_fields_to_stockhit():
    hits_json = [{
        "id": "abc123",
        "title": "Gym Workout",
        "description": "A bodybuilder performing a deadlift",
        "tags": ["gym", "deadlift", "fitness"],
        "max_width": 1920,
        "max_height": 1080,
        "duration": 12.5,
        "urls": {"mp4": "https://coverr.co/videos/abc123/preview.mp4",
                 "mp4_download": "https://coverr.co/videos/abc123/download.mp4",
                 "mp4_preview": "https://coverr.co/videos/abc123/small.mp4"},
    }]

    hits = _run_search("deadlift", hits_json)

    assert len(hits) == 1
    hit = hits[0]
    assert hit.source == "coverr"
    assert hit.source_id == "abc123"
    assert hit.download_url == "https://coverr.co/videos/abc123/download.mp4"
    assert hit.width == 1920 and hit.height == 1080
    assert hit.duration == 12.5
    assert "Gym Workout" in hit.text
    assert "deadlift" in hit.text


def test_search_falls_back_to_mp4_url_when_no_download_url():
    hits_json = [{
        "id": "xyz", "title": "t", "description": "d", "tags": [],
        "urls": {"mp4": "https://coverr.co/videos/xyz/stream.mp4"},
    }]
    hits = _run_search("q", hits_json)
    assert hits[0].download_url == "https://coverr.co/videos/xyz/stream.mp4"


def test_search_skips_hits_with_no_usable_url():
    hits_json = [
        {"id": "no-urls-field", "title": "t", "description": "d", "tags": []},
        {"id": "empty-urls", "title": "t", "description": "d", "tags": [], "urls": {}},
        {"id": "good", "title": "t", "description": "d", "tags": [],
         "urls": {"mp4_download": "https://coverr.co/videos/good/download.mp4"}},
    ]
    hits = _run_search("q", hits_json)
    assert [h.source_id for h in hits] == ["good"]


if __name__ == "__main__":
    test_text_for_combines_title_description_tags()
    test_text_for_handles_missing_fields()
    test_search_sends_correct_query_params_and_auth_header()
    test_search_maps_response_fields_to_stockhit()
    test_search_falls_back_to_mp4_url_when_no_download_url()
    test_search_skips_hits_with_no_usable_url()
    print("OK")
