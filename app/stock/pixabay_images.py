import httpx

from app.config import settings
from app.stock.base import StockHit

name = "pixabay_images"


async def search(query: str, per_page: int = 5) -> list[StockHit]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://pixabay.com/api/",
            params={"key": settings.pixabay_api_key, "q": query[:100], "per_page": per_page},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

    hits = []
    for hit in data.get("hits", []):
        url = hit.get("largeImageURL")
        if not url:
            continue
        hits.append(
            StockHit(
                source=name,
                source_id=str(hit["id"]),
                download_url=url,
                width=hit.get("imageWidth", 0),
                height=hit.get("imageHeight", 0),
                media_type="image",
                text=hit.get("tags", ""),
            )
        )
    return hits
