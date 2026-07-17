import httpx

from app.config import settings
from app.stock.base import StockHit

name = "pexels"


async def search(query: str, per_page: int = 5) -> list[StockHit]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.pexels.com/videos/search",
            headers={"Authorization": settings.pexels_api_key},
            params={"query": query, "per_page": per_page},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

    hits = []
    for video in data.get("videos", []):
        files = sorted(video.get("video_files", []), key=lambda f: f.get("width", 0), reverse=True)
        if not files:
            continue
        best = files[0]
        hits.append(
            StockHit(
                source=name,
                source_id=str(video["id"]),
                download_url=best["link"],
                width=best.get("width", 0),
                height=best.get("height", 0),
                duration=video.get("duration", 0),
            )
        )
    return hits
