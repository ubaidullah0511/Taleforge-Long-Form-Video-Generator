import httpx

from app.config import settings
from app.stock.base import StockHit

name = "pixabay"

_QUALITY_ORDER = ["large", "medium", "small", "tiny"]


async def search(query: str, per_page: int = 5) -> list[StockHit]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://pixabay.com/api/videos/",
            params={"key": settings.pixabay_api_key, "q": query[:100], "per_page": per_page},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

    hits = []
    for video in data.get("hits", []):
        videos = video.get("videos", {})
        best = next((videos[q] for q in _QUALITY_ORDER if q in videos), None)
        if not best:
            continue
        hits.append(
            StockHit(
                source=name,
                source_id=str(video["id"]),
                download_url=best["url"],
                width=best.get("width", 0),
                height=best.get("height", 0),
                duration=video.get("duration", 0),
            )
        )
    return hits
