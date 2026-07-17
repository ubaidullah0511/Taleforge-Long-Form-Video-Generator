import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from app.asset_selection import (
    _is_exact_16_9,
    _is_horizontal_16_9,
    _is_too_low_res,
    _parse_resolution,
    normalize_to_16_9,
)
from app.audio_mux import AudioMuxError, mux_narration_audio
from app.clip_ingest import probe
from app.config import settings
from app.documentary_assembly import _HEIGHT, _WIDTH, AssemblyValidationError, assemble_video
from app.documentary_table import generate_table, render_table_markdown
from app.downloader import download, download_video_trimmed
from app.llm_client import generate_json
from app.models import AssetInfo, DocumentaryResult, TimelineClip, TimelineEntry
from app.remotion_render import RemotionRenderError, render_final_video
from app.scoring import _semantic_similarity, query_embedding_for, score_asset
from app.stock import internet_archive, nasa, pexels_images, pixabay_images, wikimedia
from app.stock import pexels as pexels_video
from app.stock import pixabay as pixabay_video
from app.stock.base import StockHit
from app.style_decision import StyleDecision, decide_style
from app.subtitles import WordTiming, build_cues, render_srt
from app.timecode import duration_seconds as _duration_seconds
from app.transcription import check_alignment, get_narration_timing

logger = logging.getLogger(__name__)

Niche = Literal["historical", "modern", "general"]

_HISTORICAL_TYPES = {"Archive", "Historical", "Military", "Documents"}
_MODERN_TYPES = {"Industry", "Technology", "Sports", "Exercise Demo"}

# Spec's default 7-provider priority order (Pexels/Pixabay first).
PROVIDER_ORDER_DEFAULT = [
    pexels_video, pixabay_video, pexels_images, pixabay_images,
    internet_archive, wikimedia, nasa,
]
# Historical scripts need real archival sources first — stock sites rarely have them.
PROVIDER_ORDER_HISTORICAL = [
    internet_archive, wikimedia, nasa,
    pexels_video, pixabay_video, pexels_images, pixabay_images,
]

VISUAL_TYPE_EDIT_MAP: dict[str, tuple[str, str]] = {
    "Archive": ("film grain, dust overlay, slow zoom", "cross dissolve"),
    "Documents": ("Ken Burns, parallax", "cross dissolve"),
    "Sports": ("speed ramp, punch-in zoom, motion blur", "hard cut"),
    "Exercise Demo": ("speed ramp, punch-in zoom, motion blur", "hard cut"),
    "Laboratory": ("light leaks, slow push-in", "cross dissolve"),
    "Emotion": ("slow motion, fade to black", "fade to black"),
}
_DEFAULT_EFFECT: tuple[str, str] = ("slow zoom", "cross dissolve")

_SEMANTIC_KEYWORD_PROMPT = """Generate 3 short (3-5 word) stock-footage search \
phrases that could visually represent this documentary beat, without relying \
on specific historical footage that may not exist on stock sites.

CRITICAL: each phrase must describe a LITERAL, FILMABLE visual — a real \
object, place, person, or action that could actually appear in a stock photo \
or stock video. Do NOT use metaphors, idioms, or abstract concepts as the \
search query itself — stock sites index literal visual content, not figures \
of speech, so a metaphorical query returns nothing usable.

Example — WRONG vs RIGHT for a beat about a company's quiet, invisible \
market dominance:
  Beat: "That invisibility was not an accident. It was the foundational \
architecture of Microsoft's entire market dominance."
  WRONG (abstract/metaphorical — will not match real stock footage):
    "invisible hand rising", "silent giant emerging", "shadow market control"
  RIGHT (literal, filmable):
    "corporate office building exterior", "tech company headquarters", \
"business executives in boardroom meeting"

Beat: \"\"\"{beat}\"\"\"
Keywords already tried with no results: {tried}
Return a JSON object with a single key "keywords" whose value is an array of \
exactly 3 strings, each a literal, filmable visual description.
"""

_CONCRETENESS_CHECK_PROMPT = """For each of these proposed stock-footage \
search phrases, answer whether it describes a concrete, filmable, literal \
visual (a real object, place, person, or action that could be photographed \
or filmed) rather than an abstract concept, metaphor, or idiom.

Phrases: {phrases}

Return a JSON object with a single key "concrete" whose value is an array of \
exactly {n} booleans (true if concrete/filmable, false if abstract/\
metaphorical), in the same order as the phrases given.
"""


@dataclass
class ScoredAsset:
    hit: StockHit
    score: float


def niche_for_clip(clip: TimelineClip) -> Niche:
    if clip.visual_type in _HISTORICAL_TYPES:
        return "historical"
    if clip.visual_type in _MODERN_TYPES:
        return "modern"
    return "general"


def provider_order_for_niche(niche: Niche) -> list:
    return PROVIDER_ORDER_HISTORICAL if niche == "historical" else PROVIDER_ORDER_DEFAULT


def edit_recommendation(visual_type: str) -> tuple[str, str]:
    return VISUAL_TYPE_EDIT_MAP.get(visual_type, _DEFAULT_EFFECT)


async def _safe_search(provider, query: str, per_page: int = 5, timeout: float = 10.0) -> list[StockHit] | None:
    """Returns None (distinct from a genuinely empty []) when the provider
    timed out or failed to connect at all — callers use that signal to stop
    retrying further keywords against this same provider for this clip
    instead of repeating the same doomed call (e.g. a provider that's
    unreachable in this environment, which otherwise burns the full timeout
    once per keyword tried)."""
    try:
        return await asyncio.wait_for(provider.search(query, per_page=per_page), timeout=timeout)
    except (asyncio.TimeoutError, httpx.HTTPError):
        return None


def _keywords_are_concrete(keywords: list[str]) -> bool:
    """Self-check: asks the model whether its own proposed keywords are
    literal/filmable rather than abstract/metaphorical (see
    _SEMANTIC_KEYWORD_PROMPT's "invisible hand rising" example — this is
    the guard that catches it if the model does it anyway). Never blocks
    the pipeline over a flaky check — any failure/malformed response is
    treated as "assume OK" rather than retrying forever."""
    try:
        data = generate_json(
            _CONCRETENESS_CHECK_PROMPT.format(phrases=json.dumps(keywords), n=len(keywords)),
            settings.llm_text_model,
        )
        flags = data.get("concrete", [])
        if len(flags) != len(keywords):
            return True
        return all(bool(f) for f in flags)
    except Exception as exc:
        logger.warning("documentary_pipeline: keyword concreteness self-check failed (%s), assuming OK", exc)
        return True


def _generate_semantic_keywords(clip: TimelineClip, tried: list[str]) -> list[str]:
    data = generate_json(
        _SEMANTIC_KEYWORD_PROMPT.format(beat=clip.script_beat, tried=", ".join(tried)),
        settings.llm_text_model,
    )
    keywords = [str(k) for k in data["keywords"]][:3]

    if not _keywords_are_concrete(keywords):
        logger.warning(
            "documentary_pipeline: clip %d semantic keywords %r failed the concreteness "
            "self-check (abstract/metaphorical) — regenerating once",
            clip.clip_number, keywords,
        )
        retry_tried = tried + keywords
        data = generate_json(
            _SEMANTIC_KEYWORD_PROMPT.format(beat=clip.script_beat, tried=", ".join(retry_tried)),
            settings.llm_text_model,
        )
        keywords = [str(k) for k in data["keywords"]][:3]

    return keywords


def _satisfied(hits: list[ScoredAsset], required: int) -> bool:
    if any(a.score >= settings.documentary_high_quality_score for a in hits):
        return True
    acceptable = [a for a in hits if a.score >= settings.documentary_min_score]
    return len(acceptable) >= required


def _known_and_not_horizontal(hit: StockHit) -> bool:
    """Pure predicate, kept for its own sake (and its existing test) — True
    only when the provider reported dimensions AND they're confirmed
    non-16:9. No longer used to reject candidates pre-download: non-16:9 is
    now normalized (cropped/blur-padded) after download instead of skipped
    (see _known_unusably_low_res, normalize_to_16_9)."""
    if hit.width <= 0 or hit.height <= 0:
        return False
    return not _is_horizontal_16_9(hit.width, hit.height)


def _known_unusably_low_res(hit: StockHit) -> bool:
    """True only when the provider reported dimensions AND they're already
    confirmed too low-res to be worth downloading at all. Replaces
    _known_and_not_horizontal as the pre-download rejection gate: aspect
    ratio alone is no longer a reason to skip a candidate, since it gets
    normalized to 16:9 after download instead (see normalize_to_16_9 in
    _resolve_clip). Unknown dimensions still pass through here, same
    rationale as before — verified for real after download either way."""
    if hit.width <= 0 or hit.height <= 0:
        return False
    return _is_too_low_res(hit.width, hit.height)


async def search_clip(
    clip: TimelineClip,
    used_assets: set[tuple[str, str]],
    allow_duplicates: bool = False,
) -> tuple[list[ScoredAsset], list[ScoredAsset]]:
    niche = niche_for_clip(clip)
    providers = provider_order_for_niche(niche)
    required = settings.documentary_niche_min_assets.get(niche, settings.documentary_niche_min_assets["general"])

    query_embedding = query_embedding_for(clip)
    video_hits: list[ScoredAsset] = []
    image_hits: list[ScoredAsset] = []
    semantic_keywords: list[str] | None = None

    def _has_unused_hit(hits: list[StockHit]) -> bool:
        """A tier's raw hits only count as 'results found' if at least one
        isn't already claimed by an earlier row in this same run — a tier
        that returns only already-used clip(s) must be treated the same as
        an empty tier so the search escalates to the next keyword tier
        instead of silently contributing zero candidates and letting the
        row fall through to a placeholder."""
        return allow_duplicates or any((h.source, h.source_id) not in used_assets for h in hits)

    for provider in providers:
        if _satisfied(video_hits, required["video"]) and _satisfied(image_hits, required["image"]):
            break

        tried = [clip.canva_keyword, clip.fallback_keyword]
        tiers = ["canva_keyword", "fallback_keyword"]
        raw_hits: list[StockHit] = []
        provider_timed_out = False
        for tier, query in zip(tiers, tried):
            logger.info(
                "documentary_pipeline: clip %d [%s] -> %s search using %s: %r",
                clip.clip_number, clip.section, provider.name, tier, query,
            )
            result = await _safe_search(provider, query)
            if result is None:
                logger.warning(
                    "documentary_pipeline: clip %d [%s] -> %s timed out/failed to connect on %s — "
                    "skipping remaining keywords for this provider on this clip",
                    clip.clip_number, clip.section, provider.name, tier,
                )
                provider_timed_out = True
                raw_hits = []
                break
            if _has_unused_hit(result):
                raw_hits = result
                break
            if result:
                logger.info(
                    "documentary_pipeline: clip %d [%s] -> %s %s returned only already-used "
                    "candidate(s) — trying next keyword tier instead of repeating a clip",
                    clip.clip_number, clip.section, provider.name, tier,
                )
            raw_hits = []

        if not raw_hits and not provider_timed_out:
            if semantic_keywords is None:
                semantic_keywords = await asyncio.to_thread(_generate_semantic_keywords, clip, tried)
            for query in semantic_keywords:
                logger.info(
                    "documentary_pipeline: clip %d [%s] -> %s search using semantic_fallback "
                    "(canva/fallback keywords returned nothing usable): %r",
                    clip.clip_number, clip.section, provider.name, query,
                )
                result = await _safe_search(provider, query)
                if result is None:
                    logger.warning(
                        "documentary_pipeline: clip %d [%s] -> %s timed out/failed to connect on "
                        "semantic_fallback — skipping remaining keywords for this provider on this clip",
                        clip.clip_number, clip.section, provider.name,
                    )
                    break
                if _has_unused_hit(result):
                    raw_hits = result
                    break
                if result:
                    logger.info(
                        "documentary_pipeline: clip %d [%s] -> %s semantic_fallback %r returned only "
                        "already-used candidate(s) — trying next semantic keyword",
                        clip.clip_number, clip.section, provider.name, query,
                    )

        candidates_before = len(video_hits) + len(image_hits)
        for hit in raw_hits:
            key = (hit.source, hit.source_id)
            if not allow_duplicates and key in used_assets:
                continue
            if _known_unusably_low_res(hit):
                logger.info(
                    "documentary_pipeline: clip %d rejecting %s:%s — %dx%d is below the low-res floor",
                    clip.clip_number, hit.source, hit.source_id, hit.width, hit.height,
                )
                continue
            scored = ScoredAsset(hit=hit, score=score_asset(hit, clip, query_embedding))
            (video_hits if hit.media_type == "video" else image_hits).append(scored)
        candidates_after = len(video_hits) + len(image_hits)
        if raw_hits:
            logger.info(
                "documentary_pipeline: clip %d [%s] -> %s contributed %d/%d raw hits as scored "
                "candidates (running total %d -> %d)",
                clip.clip_number, clip.section, provider.name,
                candidates_after - candidates_before, len(raw_hits), candidates_before, candidates_after,
            )

    _log_top_scored_candidates(clip, video_hits, image_hits, query_embedding)
    return video_hits, image_hits


def _log_top_scored_candidates(
    clip: TimelineClip,
    video_hits: list[ScoredAsset],
    image_hits: list[ScoredAsset],
    query_embedding: list[float],
) -> None:
    """Diagnostic-only (no effect on ranking/selection): logs the top 3
    scored candidates for this clip with score_asset's total score AND the
    semantic-similarity component specifically, so a relevance mismatch is
    debuggable directly from the logs — e.g. a low-similarity hit winning on
    the strength of score_asset's other components (historical accuracy,
    resolution/quality, cinematic value, motion) becomes visible here instead
    of only being guessed at."""
    all_scored = sorted(video_hits + image_hits, key=lambda a: a.score, reverse=True)
    if not all_scored:
        logger.info("documentary_pipeline: clip %d [%s] — no scored candidates", clip.clip_number, clip.section)
        return
    winner = all_scored[0]
    for rank, scored in enumerate(all_scored[:3], start=1):
        similarity = _semantic_similarity(scored.hit, query_embedding)
        logger.info(
            "documentary_pipeline: clip %d [%s] top-%d: %s:%s (%s) total_score=%.2f "
            "semantic_similarity=%.3f text=%r%s",
            clip.clip_number, clip.section, rank, scored.hit.source, scored.hit.source_id,
            scored.hit.media_type, scored.score, similarity, scored.hit.text[:80],
            " <- WINNER" if scored is winner else "",
        )


def rank_acceptable_assets(video_hits: list[ScoredAsset], image_hits: list[ScoredAsset]) -> list[ScoredAsset]:
    """Every acceptable hit (score >= documentary_min_score) across both media
    types, ranked purely by score, best first — no hard preference for video
    over image. score_asset's own weighting (motion + cinematic value) tends
    to score video higher, but a stronger-scoring image still wins here; there
    is no separate rule privileging one media type. _resolve_clip walks this
    list in order so a top pick that fails the 16:9 check falls through to
    the next-best rather than giving up straight to a placeholder."""
    acceptable = [a for a in video_hits + image_hits if a.score >= settings.documentary_min_score]
    return sorted(acceptable, key=lambda a: a.score, reverse=True)


def pick_best_asset(video_hits: list[ScoredAsset], image_hits: list[ScoredAsset]) -> ScoredAsset | None:
    """Single winner per clip: highest-scoring acceptable hit across both media
    types (see rank_acceptable_assets for the no-video-preference rationale)."""
    ranked = rank_acceptable_assets(video_hits, image_hits)
    return ranked[0] if ranked else None


async def _download_asset(asset: ScoredAsset, dest_dir: Path, prefix: str, index: int, target_duration: float) -> AssetInfo:
    """target_duration is the table row's own timeline length, not a flat
    cap. download_video_trimmed's -t is a ceiling, not a mandate — if the
    source is shorter than target_duration, ffmpeg just fetches the whole
    (shorter) source, and the assembly stage loops it to fill the row."""
    hit = asset.hit
    ext = Path(hit.download_url.split("?")[0]).suffix or (".mp4" if hit.media_type == "video" else ".jpg")
    filename = f"{prefix}_{hit.media_type}{index}{ext}"

    if hit.media_type == "video":
        path = await download_video_trimmed(hit.download_url, filename, dest_dir, target_duration)
        duration = min(hit.duration, target_duration) if hit.duration else target_duration
    else:
        path, _fingerprint = await download(hit.download_url, filename, dest_dir=dest_dir)
        duration = hit.duration

    return AssetInfo(
        path=path, source=hit.source, source_id=hit.source_id, media_type=hit.media_type,
        width=hit.width, height=hit.height, duration=duration, score=asset.score,
    )


def _placeholder_entry(clip: TimelineClip, note: str) -> TimelineEntry:
    effect, transition = edit_recommendation(clip.visual_type)
    return TimelineEntry(
        clip_number=clip.clip_number, start=clip.start, end=clip.end,
        duration=_duration_seconds(clip.start, clip.end), script=clip.script_beat,
        asset_path=None, asset_metadata=None, alternates=[],
        recommended_effect=effect, transition=transition, placeholder=True, note=note,
    )


async def _resolve_clip(
    clip: TimelineClip,
    clips_root: Path,
    used_assets: set[tuple[str, str]],
    allow_duplicates: bool,
    downloads_so_far: int,
    limit: int | None,
) -> tuple[TimelineEntry, int]:
    if limit is not None and downloads_so_far >= limit:
        return _placeholder_entry(clip, "project download limit reached"), downloads_so_far

    video_hits, image_hits = await search_clip(clip, used_assets, allow_duplicates)
    candidates = rank_acceptable_assets(video_hits, image_hits)

    if not candidates:
        return _placeholder_entry(clip, "no acceptable asset found across all providers"), downloads_so_far

    clip_dir = clips_root / f"clip_{clip.clip_number:03d}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"clip{clip.clip_number:03d}"
    target_duration = _duration_seconds(clip.start, clip.end)

    # Walk candidates best-score-first. A candidate that turns out too
    # low-res to normalize (bad/missing provider metadata — see
    # _known_unusably_low_res) falls through to the next-best already-scored
    # candidate instead of giving up straight to a placeholder; the clip only
    # becomes a placeholder once every acceptable candidate has been tried
    # and rejected. Non-16:9 candidates are no longer rejected here at all —
    # they're normalized (cropped or blur-padded) to fit, same as
    # asset_selection.normalize_to_16_9.
    last_rejection_reason: str | None = None
    for index, candidate in enumerate(candidates, start=1):
        if limit is not None and downloads_so_far >= limit:
            return _placeholder_entry(clip, "project download limit reached"), downloads_so_far

        asset = await _download_asset(candidate, clip_dir, prefix, index, target_duration)
        used_assets.add((asset.source, asset.source_id))
        downloads_so_far += 1

        # Verify the actual downloaded file instead of trusting provider
        # metadata (which can be missing or wrong), so a vertical/square
        # asset never reaches assembly's hard validation still un-normalized.
        _actual_duration, resolution = await asyncio.to_thread(probe, asset.path)
        width, height = _parse_resolution(resolution)

        if not _is_exact_16_9(width, height):
            if _is_too_low_res(width, height):
                logger.warning(
                    "documentary_pipeline: clip %d candidate %d/%d (%s:%s) is %dx%d — below the "
                    "low-res floor, not worth normalizing, trying next-best candidate",
                    clip.clip_number, index, len(candidates), asset.source, asset.source_id, width, height,
                )
                last_rejection_reason = f"{asset.source}:{asset.source_id} was {width}x{height}, too low-res"
                Path(asset.path).unlink(missing_ok=True)
                continue

            normalized = await asyncio.to_thread(normalize_to_16_9, Path(asset.path), asset.media_type, width, height)
            if not normalized:
                logger.warning(
                    "documentary_pipeline: clip %d candidate %d/%d (%s:%s, %dx%d) failed to normalize "
                    "to 16:9, trying next-best candidate",
                    clip.clip_number, index, len(candidates), asset.source, asset.source_id, width, height,
                )
                last_rejection_reason = f"failed to normalize {asset.source}:{asset.source_id} ({width}x{height}) to 16:9"
                Path(asset.path).unlink(missing_ok=True)
                continue
            logger.info(
                "documentary_pipeline: clip %d normalized %s:%s from %dx%d to %dx%d (16:9, %s)",
                clip.clip_number, asset.source, asset.source_id, width, height, _WIDTH, _HEIGHT,
                "cropped" if width >= height else "blur-padded",
            )
            asset = asset.model_copy(update={"width": _WIDTH, "height": _HEIGHT})

        logger.info(
            "documentary_pipeline: clip %d accepted candidate %d/%d (%s:%s) — %d rejected before it",
            clip.clip_number, index, len(candidates), asset.source, asset.source_id, index - 1,
        )
        effect, transition = edit_recommendation(clip.visual_type)
        entry = TimelineEntry(
            clip_number=clip.clip_number, start=clip.start, end=clip.end,
            duration=target_duration, script=clip.script_beat,
            asset_path=asset.path, asset_metadata=asset, alternates=[],
            recommended_effect=effect, transition=transition,
        )
        return entry, downloads_so_far

    note = f"all {len(candidates)} acceptable candidate(s) were rejected for being too low-resolution to normalize"
    if last_rejection_reason:
        note += f" (last: {last_rejection_reason})"
    return _placeholder_entry(clip, note), downloads_so_far


async def run(
    script: str,
    project_name: str | None = None,
    allow_duplicates: bool = False,
    max_downloads: int | None = None,
    script_path: str | None = None,
    audio_path: str | None = None,
) -> DocumentaryResult:
    if script_path:
        script = Path(script_path).read_text(encoding="utf-8")
    if not script or not script.strip():
        raise ValueError("documentary_pipeline.run requires non-empty script text (via `script` or `script_path`)")

    project_name = project_name or f"project_{int(time.time())}"
    project_dir = settings.documentary_projects_dir / project_name
    clips_root = project_dir / "clips"
    clips_root.mkdir(parents=True, exist_ok=True)

    # Manually-supplied narration audio (no TTS integration — the user
    # provides this file themselves): real Whisper word timestamps replace
    # the word-count pacing estimate as the source of truth for both caption
    # text/timing and per-clip narration duration. check_alignment is a
    # sanity check only — it never changes what gets used, just flags
    # significant script/transcript deviation for manual review.
    whisper_words: list[WordTiming] | None = None
    transcript_alignment_warning: str | None = None
    if audio_path:
        whisper_words = await asyncio.to_thread(get_narration_timing, audio_path)
        transcript_alignment_warning = await asyncio.to_thread(check_alignment, whisper_words, script)

    table = await asyncio.to_thread(generate_table, script, whisper_words)

    limit = max_downloads if max_downloads is not None else settings.documentary_max_downloads_per_project
    used_assets: set[tuple[str, str]] = set()
    downloads_so_far = 0

    # Sequential per clip: dedup bookkeeping (used_assets) and the download-limit
    # counter are shared mutable state, so clips can't safely search concurrently.
    # ponytail: revisit with a lock-protected shared set if project size makes
    # sequential search a bottleneck.
    timeline: list[TimelineEntry] = []
    for clip in table:
        entry, downloads_so_far = await _resolve_clip(
            clip, clips_root, used_assets, allow_duplicates, downloads_so_far, limit
        )
        timeline.append(entry)

    final_video_path: str | None = None
    assembly_error: str | None = None
    try:
        final_path, timeline = await asyncio.to_thread(assemble_video, project_dir, timeline)
        final_video_path = str(final_path)
    except AssemblyValidationError as exc:
        logger.error("documentary_pipeline: assembly failed: %s", exc)
        assembly_error = str(exc)

    # Final render stage, only once assembly produced a video — never reorders
    # or touches anything above. Remotion composites assemble_video's already
    # exact-duration per-row segments with transitions + animated word-by-word
    # captions in one pass, replacing the old ffmpeg concat + subtitle burn-in
    # as the primary deliverable (final_video_path above is kept too — still
    # useful as a silent/no-caption artifact and for render comparisons, see
    # scripts/benchmark_render.py). Caption timing is real Whisper ASR when
    # audio_path was supplied (whisper_words above), otherwise still the
    # word-count pace estimate — see app/subtitles.py::build_word_timings.
    # captions.srt below (build_cues/render_srt) is NOT wired to whisper_words
    # yet — known follow-up, out of this change's scope.
    style_decision: StyleDecision | None = None
    subtitles_path: str | None = None
    subtitled_video_path: str | None = None
    subtitle_error: str | None = None
    if final_video_path is not None:
        style_decision = await asyncio.to_thread(decide_style, script)

        cues = build_cues(timeline)
        if cues:
            srt_path = project_dir / "captions.srt"
            srt_path.write_text(render_srt(cues), encoding="utf-8")
            subtitles_path = str(srt_path)

        try:
            remotion_path = await asyncio.to_thread(
                render_final_video, project_dir, timeline, style_decision, whisper_words
            )
            subtitled_video_path = str(remotion_path)
        except RemotionRenderError as exc:
            logger.error("documentary_pipeline: remotion render failed: %s", exc)
            subtitle_error = str(exc)

    # Mux the uploaded narration audio into the rendered video, in full and
    # unmodified — Remotion's own output above has no audio track at all
    # (every clip segment is rendered -an). This is the actual final
    # deliverable whenever manual narration audio was supplied; visuals are
    # already synced to it because clip durations/captions were timed from
    # this exact audio's own Whisper timestamps (see app/transcription.py).
    final_video_with_narration_path: str | None = None
    audio_mux_error: str | None = None
    if audio_path and subtitled_video_path is not None:
        try:
            narrated_path = await asyncio.to_thread(
                mux_narration_audio, Path(subtitled_video_path), audio_path,
                project_dir / "final_video_with_narration.mp4",
            )
            final_video_with_narration_path = str(narrated_path)
        except AudioMuxError as exc:
            logger.error("documentary_pipeline: audio mux failed: %s", exc)
            audio_mux_error = str(exc)

    (project_dir / "timeline_table.md").write_text(render_table_markdown(table), encoding="utf-8")
    (project_dir / "timeline.json").write_text(
        json.dumps([entry.model_dump() for entry in timeline], indent=2), encoding="utf-8"
    )

    return DocumentaryResult(
        project_dir=str(project_dir), table=table, timeline=timeline,
        final_video_path=final_video_path, assembly_error=assembly_error,
        subtitles_path=subtitles_path, subtitled_video_path=subtitled_video_path,
        subtitle_error=subtitle_error,
        style_decision=style_decision.model_dump() if style_decision else None,
        transcript_alignment_warning=transcript_alignment_warning,
        final_video_with_narration_path=final_video_with_narration_path,
        audio_mux_error=audio_mux_error,
    )
