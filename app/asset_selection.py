import asyncio
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.clip_ingest import probe
from app.config import settings
from app.downloader import download
from app.embeddings import embed
from app.llm_client import sample_local_frames
from app.models import ClipRecord
from app.scene_understanding import describe_candidate
from app.stock.base import StockHit

logger = logging.getLogger(__name__)

_TARGET_WIDTH, _TARGET_HEIGHT = 1920, 1080
_TARGET_RATIO = 16 / 9
# ponytail: covers minor container/encode variance (e.g. 1920x1088 from mod-16
# padding is ratio 1.7647, diff 0.013) without letting through a genuinely
# different aspect (4:3 = 1.33, 1:1 = 1.0, 9:16 vertical = 0.56).
_RATIO_TOLERANCE = 0.05
# "480p equivalent" floor on the smaller dimension — below this, upscaling
# during normalization (crop or blur-pad) would look unusably blurry/blocky,
# so the candidate is rejected outright instead of normalized.
_MIN_ACCEPTABLE_DIMENSION = 480


def _parse_resolution(resolution: str) -> tuple[int, int]:
    try:
        width_str, height_str = resolution.split("x")
        return int(width_str), int(height_str)
    except ValueError:
        return 0, 0


def _is_horizontal_16_9(width: int, height: int) -> bool:
    if width <= 0 or height <= 0 or width <= height:
        return False
    return abs((width / height) - _TARGET_RATIO) <= _RATIO_TOLERANCE


def _is_exact_16_9(width: int, height: int) -> bool:
    return width > 0 and height > 0 and abs((width / height) - _TARGET_RATIO) < 1e-3


def _is_too_low_res(width: int, height: int) -> bool:
    """The only remaining reason to reject a candidate outright instead of
    normalizing it to 16:9 — see normalize_to_16_9's docstring."""
    return width <= 0 or height <= 0 or min(width, height) < _MIN_ACCEPTABLE_DIMENSION


def _normalize_to_16_9_sync(path: Path, media_type: Literal["video", "image"]) -> bool:
    """Center-crops a near-16:9 file to exactly 1920x1080. Only called on
    candidates already confirmed within _RATIO_TOLERANCE, so the crop is
    always small — this is not for salvaging vertical/square sources."""
    tmp_dest = path.with_name(f"{path.stem}_norm{path.suffix}")
    crop_filter = (
        f"scale={_TARGET_WIDTH}:{_TARGET_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={_TARGET_WIDTH}:{_TARGET_HEIGHT}"
    )
    args = ["ffmpeg", "-y", "-i", str(path), "-vf", crop_filter]
    args += ["-frames:v", "1", "-q:v", "2"] if media_type == "image" else ["-an", "-c:v", "libx264"]
    args.append(str(tmp_dest))
    result = subprocess.run(args, capture_output=True)
    if result.returncode != 0 or not tmp_dest.exists() or tmp_dest.stat().st_size == 0:
        tmp_dest.unlink(missing_ok=True)
        return False
    tmp_dest.replace(path)
    return True


def _normalize_to_16_9_with_blur_pad_sync(path: Path, media_type: Literal["video", "image"]) -> bool:
    """For sources far from 16:9 (portrait/vertical, height > width) where a
    center-crop would cut away most of the frame, this instead scales the
    source to fit fully inside 1920x1080 with nothing cropped, and fills the
    remaining space with a blurred, cover-scaled copy of the same source as a
    background — the standard 'blurred bars' technique — so the whole
    subject stays visible instead of being lost to an aggressive crop."""
    tmp_dest = path.with_name(f"{path.stem}_norm{path.suffix}")
    pad_filter = (
        f"split=2[bg][fg];"
        f"[bg]scale={_TARGET_WIDTH}:{_TARGET_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={_TARGET_WIDTH}:{_TARGET_HEIGHT},gblur=sigma=20[bg];"
        f"[fg]scale={_TARGET_WIDTH}:{_TARGET_HEIGHT}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
    )
    args = ["ffmpeg", "-y", "-i", str(path), "-vf", pad_filter]
    args += ["-frames:v", "1", "-q:v", "2"] if media_type == "image" else ["-an", "-c:v", "libx264"]
    args.append(str(tmp_dest))
    result = subprocess.run(args, capture_output=True)
    if result.returncode != 0 or not tmp_dest.exists() or tmp_dest.stat().st_size == 0:
        tmp_dest.unlink(missing_ok=True)
        return False
    tmp_dest.replace(path)
    return True


def normalize_to_16_9(path: Path, media_type: Literal["video", "image"], width: int, height: int) -> bool:
    """Single entry point replacing the old hard-reject-if-not-16:9 behavior:
    every non-exact-16:9 candidate is now normalized instead of thrown away.
    Crops when the source is landscape-oriented or square (width >= height,
    e.g. 4:3, 1:1) since a cover-crop only trims a modest amount off one
    axis; pads with a blurred background fill when it's portrait (height >
    width, e.g. a vertical phone-shot clip) since cropping that down to 16:9
    would cut away nearly the entire frame. Returns False (never raises) on
    an ffmpeg failure — callers already treat that as "unusable", same as
    the old rejection path."""
    if width >= height:
        return _normalize_to_16_9_sync(path, media_type)
    return _normalize_to_16_9_with_blur_pad_sync(path, media_type)


@dataclass
class Candidate:
    source: str  # "local" or a stock provider name (pexels, pexels_images, ...)
    media_type: Literal["video", "image"]
    description: str
    keywords: list[str]
    analysis: dict  # full analysis dict — reused verbatim at ingest time if this wins
    width: int
    height: int
    duration: float
    embedding: list[float]
    hit: StockHit | None = None  # set for stock candidates
    clip: ClipRecord | None = None  # set for local-library candidates
    # Full-download scratch path + its sha256, set for stock candidates only.
    # The caller moves the winner's local_path into permanent storage and
    # deletes every other candidate's (see cleanup_rejected below).
    local_path: Path | None = None
    fingerprint: str = ""


@dataclass
class Selection:
    candidate: Candidate
    score: float
    reason: str


async def _download_candidate_full(hit: StockHit, scratch_dir: Path) -> tuple[Path, str]:
    """Downloads a candidate's full file (not a thumbnail, not a partial seek)
    into scratch_dir. Returns (local_path, sha256_fingerprint) — the fingerprint
    comes free from downloader.download's streaming hash, so the winner's dedup
    check downstream doesn't need to re-read the file."""
    ext = Path(hit.download_url.split("?")[0]).suffix or (".mp4" if hit.media_type == "video" else ".jpg")
    dest_filename = f"{hit.source}_{hit.source_id}{ext}"
    path, fingerprint = await download(hit.download_url, dest_filename, dest_dir=scratch_dir)
    return Path(path), fingerprint


async def _describe_stock_candidate(hit: StockHit, scene_text: str, scratch_dir: Path) -> Candidate | None:
    local_path: Path | None = None
    try:
        local_path, fingerprint = await _download_candidate_full(hit, scratch_dir)
        logger.info(
            "asset_selection: downloaded %s:%s (%s, %d bytes) -> %s",
            hit.source, hit.source_id, hit.media_type, local_path.stat().st_size, local_path,
        )

        # Checked on the actual downloaded file, not the provider's (sometimes
        # stale/wrong) metadata, and BEFORE the Groq call so a rejected
        # candidate never costs a vision request. Non-16:9 sources are
        # normalized (cropped or blur-padded) rather than rejected — only an
        # unusably low resolution is a hard reject now (see normalize_to_16_9).
        _duration, resolution = await asyncio.to_thread(probe, str(local_path))
        width, height = _parse_resolution(resolution)

        if not _is_exact_16_9(width, height):
            if _is_too_low_res(width, height):
                logger.info(
                    "asset_selection: rejecting %s:%s — %dx%d is below the %dp-equivalent floor, "
                    "not worth normalizing",
                    hit.source, hit.source_id, width, height, _MIN_ACCEPTABLE_DIMENSION,
                )
                local_path.unlink(missing_ok=True)
                return None
            if not await asyncio.to_thread(normalize_to_16_9, local_path, hit.media_type, width, height):
                logger.warning(
                    "asset_selection: rejecting %s:%s — failed to normalize %dx%d to 16:9",
                    hit.source, hit.source_id, width, height,
                )
                local_path.unlink(missing_ok=True)
                return None
            logger.info(
                "asset_selection: normalized %s:%s from %dx%d to %dx%d (16:9)",
                hit.source, hit.source_id, width, height, _TARGET_WIDTH, _TARGET_HEIGHT,
            )
            width, height = _TARGET_WIDTH, _TARGET_HEIGHT

        if hit.media_type == "video":
            images = await asyncio.to_thread(sample_local_frames, str(local_path))
        else:
            images = [local_path.read_bytes()]
        if not images:
            raise RuntimeError("no frames/image bytes obtained")
        analysis = await asyncio.to_thread(describe_candidate, scene_text, images)
    except Exception as exc:
        # ponytail: one flaky candidate (dead link, decode failure, transient
        # LLM error) shouldn't sink the whole batch — skip it, and since it can
        # never win, delete its scratch download right away.
        logger.warning("asset_selection: rejecting %s:%s (%s) — deleting scratch download", hit.source, hit.source_id, exc)
        if local_path is not None:
            local_path.unlink(missing_ok=True)
        return None

    description = analysis.get("description", "")
    keywords = analysis.get("keywords", [])
    return Candidate(
        source=hit.source, media_type=hit.media_type, description=description, keywords=keywords,
        analysis=analysis, width=width, height=height, duration=hit.duration,
        embedding=embed(f"{description} {' '.join(keywords)}"), hit=hit,
        local_path=local_path, fingerprint=fingerprint,
    )


async def describe_candidates(hits: list[StockHit], scene_text: str, scratch_dir: Path) -> list[Candidate]:
    """Downloads every candidate in hits to scratch_dir in full, then scores
    each from its actual downloaded content (full image bytes, or frames
    sampled from the fully-downloaded video) — no thumbnails, no remote
    seeking. Caller must call cleanup_rejected() after picking a winner."""
    logger.info("asset_selection: downloading and describing %d stock candidates into %s", len(hits), scratch_dir)
    described = await asyncio.gather(*(_describe_stock_candidate(hit, scene_text, scratch_dir) for hit in hits))
    accepted = [c for c in described if c is not None]
    logger.info(
        "asset_selection: %d/%d candidates accepted after normalization (%d rejected — too low-res "
        "or a real download/describe failure)",
        len(accepted), len(hits), len(hits) - len(accepted),
    )
    return accepted


def cleanup_rejected(candidates: list[Candidate], winner: Candidate) -> None:
    """Deletes every downloaded candidate's scratch file except the winner's.
    The winner's file is the caller's responsibility to move out of scratch."""
    for candidate in candidates:
        if candidate is winner or candidate.local_path is None:
            continue
        candidate.local_path.unlink(missing_ok=True)
        logger.info(
            "asset_selection: deleted rejected candidate %s:%s -> %s",
            candidate.source, _candidate_id(candidate), candidate.local_path,
        )


def local_candidates(matches: list[tuple[ClipRecord, float]]) -> list[Candidate]:
    candidates = []
    for clip, _similarity in matches:
        width, height = _parse_resolution(clip.resolution)
        if not _is_horizontal_16_9(width, height):
            logger.info(
                "asset_selection: skipping local clip %s (%s) — not 16:9 horizontal",
                clip.id, clip.resolution or "unknown resolution",
            )
            continue
        candidates.append(Candidate(
            source="local", media_type=clip.media_type, description=clip.description,
            keywords=clip.keywords, analysis={}, width=width, height=height, duration=clip.duration,
            embedding=embed(f"{clip.description} {' '.join(clip.keywords)}"), clip=clip,
        ))
    return candidates


def _cosine(a: list[float], b: list[float]) -> float:
    return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b))))


def select_best_asset(
    candidates: list[Candidate],
    scene_text: str,
    preferred_media_type: Literal["video", "image"],
    target_duration: float | None = None,
) -> Selection | None:
    """Rank candidates by semantic similarity to scene_text first, prefer the
    decided media type unless another candidate scores significantly higher,
    and only use resolution to break near-ties. If target_duration is given,
    video candidates shorter than it are filtered out first (a scene-level
    duration isn't produced anywhere in this codebase yet — this stays inert
    until a caller has one to pass)."""
    if not candidates:
        return None

    pool = candidates
    if target_duration is not None:
        filtered = [c for c in pool if c.media_type != "video" or c.duration >= target_duration]
        if filtered:  # don't discard every candidate if none clears the bar
            pool = filtered

    query_embedding = embed(scene_text)
    scored = sorted(
        ((c, _cosine(query_embedding, c.embedding)) for c in pool),
        key=lambda pair: pair[1], reverse=True,
    )
    for candidate, score in scored:
        logger.debug("asset_selection: candidate %s:%s (%s) scored %.3f",
                     candidate.source, _candidate_id(candidate), candidate.media_type, score)

    top_score = scored[0][1]
    margin = settings.asset_selection_score_margin
    preferred_pool = [(c, s) for c, s in scored if c.media_type == preferred_media_type and s >= top_score - margin]

    if preferred_pool:
        ranked = preferred_pool
        reason_prefix = f"preferred media type '{preferred_media_type}'"
    else:
        ranked = scored[:1]
        reason_prefix = f"no '{preferred_media_type}' candidate within {margin:.2f} of the top score ({top_score:.2f})"

    best_score = ranked[0][1]
    epsilon = settings.asset_selection_tie_epsilon
    tied = [(c, s) for c, s in ranked if abs(s - best_score) <= epsilon]
    chosen, score = max(tied, key=lambda pair: pair[0].width * pair[0].height)

    reason = f"{reason_prefix}; semantic score {score:.2f}"
    if len(tied) > 1:
        reason += "; resolution used as tiebreaker"

    logger.info("asset_selection: selected %s:%s (%s) — %s",
                chosen.source, _candidate_id(chosen), chosen.media_type, reason)
    return Selection(candidate=chosen, score=score, reason=reason)


def _candidate_id(candidate: Candidate) -> str:
    if candidate.hit is not None:
        return candidate.hit.source_id
    if candidate.clip is not None:
        return candidate.clip.id
    return "?"
