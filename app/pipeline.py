import asyncio
import logging
import shutil
import uuid

from app.asset_selection import cleanup_rejected, describe_candidates, local_candidates, select_best_asset
from app.clip_ingest import analyze_and_store
from app.config import settings
from app.models import Scene, SceneResult
from app.scene_segmentation import split
from app.scene_understanding import analyze, decide_media_type
from app.stock import pexels, pexels_images, pixabay, pixabay_images
from app.vector_store import get_by_fingerprint, get_by_source_id, query_similar

logger = logging.getLogger(__name__)

# ponytail: flat provider list, not a registry/plugin system - all 4 are
# queried on every stock search (no conditional dispatch needed), just split
# evenly to fill the configurable candidate pool. Add a registry if providers
# start needing per-query conditional selection.
STOCK_PROVIDERS = [pexels, pixabay, pexels_images, pixabay_images]


async def _search_stock(query: str, pool_size: int | None = None):
    pool_size = pool_size or settings.candidate_pool_size
    per_provider = max(1, pool_size // len(STOCK_PROVIDERS))
    results = await asyncio.gather(*(p.search(query, per_page=per_provider) for p in STOCK_PROVIDERS))
    hits = [hit for provider_hits in results for hit in provider_hits]
    return hits[:pool_size]


async def _resolve_scene(scene: Scene, script: str, run_id: str) -> SceneResult:
    scene_analysis = await asyncio.to_thread(analyze, scene.text)
    query_text = f"{scene_analysis.description} {' '.join(scene_analysis.keywords)}"
    preferred_media_type = await asyncio.to_thread(decide_media_type, scene.text)
    logger.info("pipeline: scene %d preferred media type = %s", scene.scene, preferred_media_type)

    local_matches = await asyncio.to_thread(query_similar, query_text, settings.candidate_pool_size)
    acceptable_local = [(clip, sim) for clip, sim in local_matches if sim >= settings.similarity_threshold]

    if acceptable_local:
        selection = select_best_asset(local_candidates(acceptable_local), scene.text, preferred_media_type)
        clip = selection.candidate.clip
        return SceneResult(
            scene=scene.scene, script=scene.text, selected_clip=clip.video_path,
            similarity=selection.score, source="local", start=clip.start, end=clip.end,
            media_type=clip.media_type, selection_reason=selection.reason,
        )

    hits = await _search_stock(query_text)
    if not hits:
        return SceneResult(scene=scene.scene, script=scene.text, selected_clip=None, source="none")

    logger.info("pipeline: scene %d fetched %d stock candidates", scene.scene, len(hits))
    scratch_dir = settings.candidate_scratch_dir / f"{run_id}_scene{scene.scene}"
    try:
        described = await describe_candidates(hits, scene.text, scratch_dir)
        if not described:
            logger.warning(
                "pipeline: scene %d — no 16:9 horizontal candidate survived filtering "
                "out of %d fetched (query=%r); not falling back to a vertical/square clip",
                scene.scene, len(hits), query_text,
            )
            return SceneResult(scene=scene.scene, script=scene.text, selected_clip=None, source="none")

        selection = select_best_asset(described, scene.text, preferred_media_type)
        best = selection.candidate.hit
        winner_path = selection.candidate.local_path
        fingerprint = selection.candidate.fingerprint
        cleanup_rejected(described, selection.candidate)

        cached = get_by_source_id(best.source, best.source_id)
        if cached:
            winner_path.unlink(missing_ok=True)
            logger.info("pipeline: scene %d winner %s:%s already in library, deleting scratch copy",
                        scene.scene, best.source, best.source_id)
            return SceneResult(
                scene=scene.scene, script=scene.text, selected_clip=cached.video_path,
                source=cached.source, start=cached.start, end=cached.end, cached=True,
                media_type=cached.media_type, selection_reason=selection.reason,
            )

        dup = get_by_fingerprint(fingerprint)
        if dup:
            winner_path.unlink(missing_ok=True)
            logger.info("pipeline: scene %d winner %s:%s duplicates an existing asset, deleting scratch copy",
                        scene.scene, best.source, best.source_id)
            return SceneResult(
                scene=scene.scene, script=scene.text, selected_clip=dup.video_path,
                source=dup.source, start=dup.start, end=dup.end, cached=True,
                media_type=dup.media_type, selection_reason=selection.reason,
            )

        settings.downloaded_clips_dir.mkdir(parents=True, exist_ok=True)
        asset_path = settings.downloaded_clips_dir / f"{best.source}_{best.source_id}{winner_path.suffix}"
        shutil.move(str(winner_path), str(asset_path))
        logger.info("pipeline: scene %d kept winning candidate %s:%s -> %s",
                    scene.scene, best.source, best.source_id, asset_path)

        clip = await asyncio.to_thread(
            analyze_and_store, str(asset_path), 0, best.duration or 0, best.source, best.source_id, fingerprint,
            best.media_type, selection.candidate.analysis,
        )
        return SceneResult(
            scene=scene.scene, script=scene.text, selected_clip=clip.video_path,
            source=clip.source, start=clip.start, end=clip.end, downloaded=True,
            media_type=clip.media_type, selection_reason=selection.reason,
        )
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


async def run(script: str) -> list[SceneResult]:
    scenes = await asyncio.to_thread(split, script)
    run_id = uuid.uuid4().hex[:8]
    return list(await asyncio.gather(*(_resolve_scene(scene, script, run_id) for scene in scenes)))
