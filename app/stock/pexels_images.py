import httpx

from app.config import settings
from app.stock.base import StockHit

name = "pexels_images"


async def search(query: str, per_page: int = 5) -> list[StockHit]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": settings.pexels_api_key},
            params={"query": query, "per_page": per_page},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

    hits = []
    for photo in data.get("photos", []):
        src = photo.get("src", {})
        url = src.get("large2x") or src.get("original")
        if not url:
            continue
        hits.append(
            StockHit(
                source=name,
                source_id=str(photo["id"]),
                download_url=url,
                width=photo.get("width", 0),
                height=photo.get("height", 0),
                media_type="image",
                text=photo.get("alt", "") or "",
            )
        )
    return hits
