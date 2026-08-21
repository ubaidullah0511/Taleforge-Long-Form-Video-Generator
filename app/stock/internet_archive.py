import asyncio
import logging

import httpx

from app.stock._image_dimensions import probe_remote_image_size
from app.stock.base import StockHit

logger = logging.getLogger(__name__)

name = "internet_archive"

_VIDEO_EXTS = (".mp4", ".ogv", ".webm")
_IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# Archive.org hosts both genuinely public-domain material AND third-party
# uploads under restricted/non-commercial-only licenses (e.g. some Periscope
# Film LLC items — confirmed via Archive.org's own forum discussion on this
# exact ambiguity) that look identical to PD footage in search results. Only
# an explicit, recognized open-license URL is trusted; a missing or
# unrecognized licenseurl is excluded rather than assumed safe.
_ALLOWED_LICENSE_PREFIXES = (
    "https://creativecommons.org/publicdomain",
    "http://creativecommons.org/publicdomain",
    "https://creativecommons.org/licenses/by/",
    "http://creativecommons.org/licenses/by/",
    # Legacy (pre-CC0) public domain dedication URL, still used by some older
    # Archive.org uploads — confirmed live against a real item (NewNeigh1953)
    # that would otherwise be wrongly excluded despite being genuinely PD.
    "https://creativecommons.org/licenses/publicdomain/",
    "http://creativecommons.org/licenses/publicdomain/",
)

_HEADERS = {"User-Agent": "clip-generation-documentary-pipeline/1.0 (automated stock-footage search)"}

# Per-identifier file-listing cache: the same item can surface in more than
# one clip's search results within a single pipeline run, and its file
# listing doesn't change mid-run, so a repeat lookup is pure waste against a
# free, unauthenticated public API.
_files_cache: dict[str, list[dict]] = {}


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0


def _has_open_license(license_url: str) -> bool:
    return bool(license_url) and license_url.startswith(_ALLOWED_LICENSE_PREFIXES)


def _text_for(doc: dict) -> str:
    """Title alone (the old behavior) is thin for niche-compliance filtering
    and CLIP verification; description/subject are real human-written fields
    Archive.org items generally have, unlike Pexels/Pixabay's often-empty tags."""
    subject = doc.get("subject", "")
    subject_text = " ".join(subject) if isinstance(subject, list) else str(subject)
    return " ".join(filter(None, [doc.get("title", ""), doc.get("description", ""), subject_text]))


async def _fetch_files(client: httpx.AsyncClient, identifier: str) -> list[dict]:
    if identifier in _files_cache:
        return _files_cache[identifier]
    try:
        resp = await client.get(f"https://archive.org/metadata/{identifier}", timeout=10)
        resp.raise_for_status()
        files = resp.json().get("files", [])
    except httpx.HTTPError:
        files = []
    _files_cache[identifier] = files
    return files


async def search(query: str, per_page: int = 5) -> list[StockHit]:
    async with httpx.AsyncClient(headers=_HEADERS) as client:
        resp = await client.get(
            "https://archive.org/advancedsearch.php",
            params={
                "q": f"{query} AND mediatype:(movies OR image)",
                "fl[]": ["identifier", "title", "description", "subject", "licenseurl"],
                "rows": per_page,
                "page": 1,
                "output": "json",
            },
            timeout=15,
        )
        resp.raise_for_status()
        docs = resp.json().get("response", {}).get("docs", [])

        open_docs = []
        for doc in docs:
            license_url = doc.get("licenseurl", "")
            if not _has_open_license(license_url):
                logger.debug(
                    "internet_archive: excluding %s — no confirmed open license (licenseurl=%r)",
                    doc.get("identifier"), license_url,
                )
                continue
            open_docs.append(doc)

        files_per_doc = await asyncio.gather(*(_fetch_files(client, doc["identifier"]) for doc in open_docs))

        hits = []
        for doc, files in zip(open_docs, files_per_doc):
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
                    text=_text_for(doc),
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
