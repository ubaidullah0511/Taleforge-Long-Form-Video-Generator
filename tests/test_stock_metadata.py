"""Offline test for app.stock.pexels's URL-slug title extraction — no
network/API keys needed. Run with: pytest tests/test_stock_metadata.py
"""
from app.stock.pexels import _title_from_url


def test_title_from_url_extracts_readable_slug():
    url = "https://www.pexels.com/video/semi-truck-climbing-a-mountain-18749833/"
    assert _title_from_url(url) == "semi truck climbing a mountain"


def test_title_from_url_handles_no_trailing_slash():
    url = "https://www.pexels.com/video/cars-in-the-road-7190575"
    assert _title_from_url(url) == "cars in the road"


def test_title_from_url_handles_empty_string():
    assert _title_from_url("") == ""


if __name__ == "__main__":
    test_title_from_url_extracts_readable_slug()
    test_title_from_url_handles_no_trailing_slash()
    test_title_from_url_handles_empty_string()
    print("OK")
