"""Offline tests for app.stock.internet_archive's license filtering, metadata
text extraction, and file-format resolution — no real network access (httpx
MockTransport stands in for archive.org). Run with:
pytest tests/test_internet_archive.py
"""
import asyncio
from unittest.mock import patch

import httpx

from app.stock import internet_archive
from app.stock.internet_archive import _has_open_license, _text_for

_RealAsyncClient = httpx.AsyncClient


def test_has_open_license_accepts_public_domain():
    assert _has_open_license("https://creativecommons.org/publicdomain/mark/1.0/")


def test_has_open_license_accepts_cc_by():
    assert _has_open_license("https://creativecommons.org/licenses/by/4.0/")


def test_has_open_license_rejects_missing():
    assert not _has_open_license("")


def test_has_open_license_rejects_non_commercial():
    # The exact ambiguous case flagged in Archive.org's own forum discussion:
    # a restrictive/non-commercial license must never be treated as usable.
    assert not _has_open_license("https://creativecommons.org/licenses/by-nc/4.0/")


def test_text_for_combines_title_description_subject():
    doc = {
        "title": "Highway Film",
        "description": "1950s road construction",
        "subject": ["highway", "construction"],
    }
    text = _text_for(doc)
    assert "Highway Film" in text
    assert "1950s road construction" in text
    assert "highway" in text and "construction" in text


def _run_search(query, docs, metadata_by_id):
    metadata_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "advancedsearch.php" in url:
            return httpx.Response(200, json={"response": {"docs": docs}})
        identifier = url.rsplit("/", 1)[-1]
        metadata_calls.append(identifier)
        return httpx.Response(200, json={"files": metadata_by_id.get(identifier, [])})

    transport = httpx.MockTransport(handler)
    internet_archive._files_cache.clear()
    with patch.object(internet_archive.httpx, "AsyncClient", lambda **kw: _RealAsyncClient(transport=transport, **kw)):
        hits = asyncio.run(internet_archive.search(query))
    return hits, metadata_calls


def test_search_excludes_items_without_allowed_license():
    docs = [
        {"identifier": "open1", "title": "Open Item",
         "licenseurl": "https://creativecommons.org/publicdomain/mark/1.0/"},
        {"identifier": "restricted1", "title": "Restricted Item",
         "licenseurl": "https://creativecommons.org/licenses/by-nc/4.0/"},
        {"identifier": "missing1", "title": "No License Item"},
    ]
    metadata_by_id = {"open1": [{"name": "open1.mp4", "width": "1920", "height": "1080", "length": "30"}]}

    hits, metadata_calls = _run_search("highway construction", docs, metadata_by_id)

    assert {h.source_id for h in hits} == {"open1"}
    # restricted/missing-license items must be excluded before ever costing a
    # metadata lookup, not just filtered out of the final result list.
    assert metadata_calls == ["open1"]


def test_search_includes_items_with_allowed_license_as_valid_stock_hit():
    docs = [{"identifier": "open2", "title": "Public Domain Reel", "description": "archival footage",
             "licenseurl": "https://creativecommons.org/publicdomain/zero/1.0/"}]
    metadata_by_id = {"open2": [{"name": "open2.mp4", "width": "1280", "height": "720", "length": "45"}]}

    hits, _ = _run_search("military convoy", docs, metadata_by_id)

    assert len(hits) == 1
    hit = hits[0]
    assert hit.source == "internet_archive"
    assert hit.source_id == "open2"
    assert hit.download_url == "https://archive.org/download/open2/open2.mp4"
    assert hit.media_type == "video"
    assert hit.width == 1280 and hit.height == 720


def test_search_picks_video_file_and_skips_non_video_files_in_mixed_listing():
    docs = [{"identifier": "mixed1", "title": "Mixed Files Item",
             "licenseurl": "https://creativecommons.org/publicdomain/mark/1.0/"}]
    metadata_by_id = {"mixed1": [
        {"name": "mixed1_meta.xml"},
        {"name": "mixed1_thumbs.png", "width": "320", "height": "240"},
        {"name": "mixed1.mp4", "width": "1920", "height": "1080", "length": "60"},
    ]}

    hits, _ = _run_search("1950s highway construction", docs, metadata_by_id)

    assert len(hits) == 1
    assert hits[0].download_url == "https://archive.org/download/mixed1/mixed1.mp4"
    assert hits[0].media_type == "video"


if __name__ == "__main__":
    test_has_open_license_accepts_public_domain()
    test_has_open_license_accepts_cc_by()
    test_has_open_license_rejects_missing()
    test_has_open_license_rejects_non_commercial()
    test_text_for_combines_title_description_subject()
    test_search_excludes_items_without_allowed_license()
    test_search_includes_items_with_allowed_license_as_valid_stock_hit()
    test_search_picks_video_file_and_skips_non_video_files_in_mixed_listing()
    print("OK")
