import asyncio

import httpx

from app.stock._image_dimensions import probe_remote_image_size
from app.stock.base import StockHit

name = "internet_archive"

_VIDEO_EXTS = (".mp4", ".ogv", ".webm")
_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0


async def _fetch_files(client: httpx.AsyncClient, identifier: str) -> list[dict]:
    try:
        resp = await client.get(f"https://archive.org/metadata/{identifier}", timeout=10)
        resp.raise_for_status()
        return resp.json().get("files", [])
    except httpx.HTTPError:
        return []


async def search(query: str, per_page: int = 5) -> list[StockHit]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://archive.org/advancedsearch.php",
            params={
                "q": f"{query} AND mediatype:(movies OR image)",
                "fl[]": ["identifier", "title"],
                "rows": per_page,
                "page": 1,
                "output": "json",
            },
            timeout=15,
        )
        resp.raise_for_status()
        docs = resp.json().get("response", {}).get("docs", [])

        files_per_doc = await asyncio.gather(*(_fetch_files(client, doc["identifier"]) for doc in docs))

        hits = []
        for doc, files in zip(docs, files_per_doc):
            video_file = next((f for f in files if f.get("name", "").lower().endswith(_VIDEO_EXTS)), None)
            image_file = next((f for f in files if f.get("name", "").lower().endswith(_IMAGE_EXTS)), None)
            best, media_type = (video_file, "video") if video_file else (image_file, "image")
            if not best:
                continue
            hits.append(
                StockHit(
                    source=name,
                    source_id=doc["identifier"],
                    download_url=f"https://archive.org/download/{doc['identifier']}/{best['name']}",
                    width=int(_num(best.get("width"))),
                    height=int(_num(best.get("height"))),
                    duration=_num(best.get("length")),
                    media_type=media_type,
                    text=doc.get("title", ""),
                )
            )

        # Internet Archive's file metadata frequently omits width/height
        # (confirmed empirically — plain image files often have neither) even
        # though the pre-download 16:9 filter is ready to use it. Only images
        # are probed — video header parsing is a materially different, more
        # complex format and is out of scope here; non-16:9 video candidates
        # still get caught by the existing post-download safety net.
        needs_probe = [h for h in hits if h.media_type == "image" and (h.width <= 0 or h.height <= 0)]
        sizes = await asyncio.gather(*(probe_remote_image_size(client, h.download_url) for h in needs_probe))
        for hit, (width, height) in zip(needs_probe, sizes):
            hit.width, hit.height = width, height
    return hits
