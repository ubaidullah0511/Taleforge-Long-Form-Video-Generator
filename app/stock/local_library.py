import json
import logging

from app.config import settings
from app.scoring import keyword_overlap_ratio
from app.stock.base import StockHit

logger = logging.getLogger(__name__)

name = "local_library"


async def search(query: str, niche: str, per_page: int = 5) -> list[StockHit]:
    """Loads local_clips/<niche>/index.json (built by
    app.stock.local_library_index) and ranks entries by keyword_overlap_ratio
    against each clip's vision-generated caption — same StockHit shape stock
    providers return, so downstream scoring/niche-filter/CLIP/dedup logic
    needs no special-casing. download_url is a plain local file path rather
    than a remote URL: every download path in app.downloader shells out to
    ffmpeg, which accepts a local path via -i exactly like a URL, so no
    download-layer changes are needed either."""
    clips_dir = settings.local_clips_dir / niche
    index_path = clips_dir / "index.json"
    if not index_path.exists():
        # A missing index at a niche's own configured local_library_path is
        # not "this niche has no local library" (that case is already
        # filtered out upstream — see documentary_pipeline's
        # _filter_providers_by_source_toggles, which drops this provider
        # entirely when local_library_path is empty). Reaching here means a
        # PATH WAS CONFIGURED but nothing is actually there — e.g. the
        # folder was moved/renamed in custom_niches.json without the actual
        # files following (or a long-running server never restarted after
        # such an edit, so it's still resolving the OLD path). That silently
        # returning [] here previously left local_library search
        # indistinguishable from "no candidates found" in every log — no
        # trace of the actual misconfiguration ever surfaced. Logging the
        # resolved path makes that class of bug traceable instead of
        # requiring a manual code walk to rediscover it every time.
        logger.warning(
            "local_library: no index.json at %s (niche path %r) — returning no candidates for "
            "this query; confirm the niche's local_library_path in custom_niches.json matches "
            "where its clips actually are on disk",
            index_path, niche,
        )
        return []
    index: dict = json.loads(index_path.read_text())

    ranked = sorted(
        index.items(),
        key=lambda item: keyword_overlap_ratio(query, item[1].get("caption", "")),
        reverse=True,
    )
    hits = []
    for filename, meta in ranked:
        if len(hits) >= per_page:
            break
        # index.json can drift from disk (a file deleted after indexing,
        # e.g. by the near-duplicate cleanup pass) — see
        # app.stock.local_library_index's --verify reconciliation. Skipping
        # here (not just cleaning the index offline) is the root-cause fix:
        # every caller of search() — primary candidates AND the editor's
        # alternates list — is protected without needing its own check.
        if not (clips_dir / filename).exists():
            continue
        hits.append(StockHit(
            source=name,
            source_id=filename,
            download_url=str((clips_dir / filename).resolve()),
            width=meta.get("width", 0),
            height=meta.get("height", 0),
            duration=meta.get("duration", 0),
            text=meta.get("caption", ""),
        ))
    return hits
