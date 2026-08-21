import httpx

from app.config import settings
from app.stock.base import StockHit

name = "coverr"


def _text_for(hit: dict) -> str:
    """Coverr returns real title/description/tags fields directly on every
    hit — unlike Pexels' video-search API, which returns no usable tags at
    all and forces a URL-slug workaround (see pexels._title_from_url). No
    derivation needed here."""
    tags = hit.get("tags") or []
    tags_text = " ".join(tags) if isinstance(tags, list) else str(tags)
    return " ".join(filter(None, [hit.get("title", ""), hit.get("description", ""), tags_text]))


async def search(query: str, per_page: int = 5) -> list[StockHit]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.coverr.co/videos",
            headers={"Authorization": f"Bearer {settings.coverr_api_key}"},
            # urls=true is required — the list response omits the urls object
            # (mp4/mp4_preview/mp4_download) entirely without it.
            params={"query": query, "page": 0, "page_size": per_page, "urls": "true"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

    hits = []
    for hit in data.get("hits", []):
        urls = hit.get("urls") or {}
        download_url = urls.get("mp4_download") or urls.get("mp4")
        if not download_url:
            continue
        hits.append(
            StockHit(
                source=name,
                source_id=str(hit["id"]),
                download_url=download_url,
                width=hit.get("max_width", 0),
                height=hit.get("max_height", 0),
                duration=hit.get("duration", 0),
                text=_text_for(hit),
            )
        )
    return hits
