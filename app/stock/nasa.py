import asyncio

import httpx

from app.stock._image_dimensions import probe_remote_image_size
from app.stock.base import StockHit

name = "nasa"


async def search(query: str, per_page: int = 5) -> list[StockHit]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://images-api.nasa.gov/search",
            params={"q": query, "media_type": "image"},
            timeout=15,
        )
        resp.raise_for_status()
        items = resp.json().get("collection", {}).get("items", [])[:per_page]

        hits = []
        for item in items:
            data = (item.get("data") or [{}])[0]
            links = item.get("links") or []
            href = next((link["href"] for link in links if link.get("rel") == "preview"), None)
            if not href:
                continue
            hits.append(
                StockHit(
                    source=name,
                    source_id=data.get("nasa_id", ""),
                    download_url=href,
                    media_type="image",
                    text=data.get("title", ""),
                )
            )

        # NASA's search API never reports width/height (unlike pexels/pixabay),
        # so the pre-download 16:9 filter in documentary_pipeline.py would
        # never catch a non-horizontal NASA image without this — every
        # candidate would be fully downloaded before its aspect ratio is known.
        sizes = await asyncio.gather(*(probe_remote_image_size(client, hit.download_url) for hit in hits))
        for hit, (width, height) in zip(hits, sizes):
            hit.width, hit.height = width, height
    return hits
