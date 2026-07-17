import httpx

from app.stock.base import StockHit

name = "wikimedia"


# ponytail: image-only, Commons video/ogg support not handled — add if a clip
# specifically needs archival video from Commons.
async def search(query: str, per_page: int = 5) -> list[StockHit]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": per_page,
                "gsrnamespace": 6,
                "prop": "imageinfo",
                "iiprop": "url|size",
                "format": "json",
            },
            timeout=15,
        )
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})

    hits = []
    for page in pages.values():
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        url = info.get("url", "")
        if not url.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        hits.append(
            StockHit(
                source=name,
                source_id=str(page.get("pageid", page.get("title", ""))),
                download_url=url,
                width=info.get("width", 0),
                height=info.get("height", 0),
                media_type="image",
                text=page.get("title", ""),
            )
        )
    return hits
