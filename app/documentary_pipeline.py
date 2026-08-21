import asyncio
import json
import logging
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from app import progress
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
from app.documentary_assembly import (
    _DURATION_EPSILON,
    _HEIGHT,
    _SPEED_MATCH_MAX,
    _SPEED_MATCH_MIN,
    _WIDTH,
    AssemblyValidationError,
    assemble_video,
    render_all_segments,
    rerender_single_clip_segment,
)
from app.documentary_table import generate_table, render_table_markdown
from app.downloader import download, download_video_trimmed
from app.llm_client import generate_json
from app.models import (
    AlternateCandidate,
    AssetInfo,
    AvailabilityReport,
    ClipAvailability,
    DocumentaryResult,
    ProjectMeta,
    TimelineClip,
    TimelineEntry,
)
from app.niches import DEFAULT_NICHE, NicheConfig, candidate_violates_niche, get_niche, resolve_sub_niches, violates_niche
from app.remotion_render import RemotionRenderError, render_final_video
from app.scoring import _semantic_similarity, keyword_overlap_ratio, query_embedding_for, score_asset
from app.stock import coverr, internet_archive, local_library, nasa, pexels_images, pixabay_images
from app.stock import pexels as pexels_video
from app.stock import pixabay as pixabay_video
from app.stock.openai_image import generate_fallback_image_openai
from app.stock.base import StockHit
from app.style_decision import StyleDecision, decide_style
from app.subtitles import WordTiming, build_cues, render_srt
from app.timecode import duration_seconds as _duration_seconds
from app.transcription import check_alignment, get_narration_timing
from app.visual_verification import passes_visual_verification

logger = logging.getLogger(__name__)


def resolve_project_name(project_name: str | None) -> str:
    """Same fallback run() applies internally — exposed so callers that need
    to know the project_name before the pipeline finishes (see app/main.py's
    /generate-documentary-timeline, which returns it immediately so the
    frontend can start polling GET /progress/{project_name}) can resolve it
    up front instead of guessing what run() would have generated."""
    return project_name or f"project_{int(time.time())}"


# Matches ONLY resolve_project_name()'s own auto-generated fallback shape
# (project_<unix timestamp>) — used by final_video_filename below to tell
# "user typed no name" apart from "user typed a name", since project_dir's
# name alone doesn't otherwise carry that distinction (resolve_project_name
# collapses both cases into one string, with no separate flag persisted).
_AUTO_PROJECT_NAME_RE = re.compile(r"^project_\d+$")

_INVALID_FILENAME_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _sanitize_filename(name: str) -> str:
    """Filesystem-safe basename (no extension): strips characters invalid on
    Windows (also unusual/unsafe on POSIX), collapses whitespace runs to a
    single underscore, and trims leading/trailing underscores left over from
    stripped characters — e.g. "Alex Eala: Toronto Recap!" -> "Alex_Eala_Toronto_Recap!"
    (":" stripped, space -> underscore; "!" is valid so it's kept)."""
    cleaned = _INVALID_FILENAME_CHARS_RE.sub("", name)
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    return cleaned.strip("_")


def final_video_filename(project_dir: Path) -> str:
    """Basename for the narrated final video (see _finalize_and_render, which
    writes it, and app/main.py's GET /project, which needs to find it again
    for the editor's download links — both call this so they can never
    disagree on the name): final_video_with_narration.mp4 by default, or a
    sanitized version of the user's typed project name when they gave one,
    so multiple projects don't all download under the same generic filename."""
    if _AUTO_PROJECT_NAME_RE.match(project_dir.name):
        return "final_video_with_narration.mp4"
    sanitized = _sanitize_filename(project_dir.name)
    return f"{sanitized}.mp4" if sanitized else "final_video_with_narration.mp4"


Niche = Literal["historical", "modern", "general"]

_HISTORICAL_TYPES = {"Archive", "Historical", "Military", "Documents"}
_MODERN_TYPES = {"Industry", "Technology", "Sports", "Exercise Demo"}

# Stock-provider priority order (Pexels/Pixabay/Coverr first). Wikimedia
# dropped from rotation — it consistently failed to connect (ConnectTimeout,
# not just slow) in this environment, burning ~10-15s per clip for zero
# usable candidates. app/stock/wikimedia.py itself is untouched in case this
# is worth revisiting.
PROVIDER_ORDER_DEFAULT = [
    local_library,
    pexels_video, pixabay_video, coverr, pexels_images, pixabay_images,
    internet_archive, nasa,
]
# Historical scripts need real archival sources first — stock sites rarely have them.
PROVIDER_ORDER_HISTORICAL = [
    local_library,
    internet_archive, nasa,
    pexels_video, pixabay_video, coverr, pexels_images, pixabay_images,
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

_SEMANTIC_KEYWORD_PROMPT = """Generate 3 short (2-4 word) stock-footage search \
phrases that could visually represent this documentary beat, without relying \
on specific historical footage that may not exist on stock sites.

{niche_context}

CRITICAL: each phrase must describe a LITERAL, FILMABLE visual — a real \
object, place, person, or action that could actually appear in a stock photo \
or stock video. Do NOT use metaphors, idioms, or abstract concepts as the \
search query itself — stock sites index literal visual content, not figures \
of speech, so a metaphorical query returns nothing usable. Keep phrases \
SHORT — 2 to 4 words, exactly how a person would type a query into a stock \
search bar, not a full descriptive sentence.

A short phrase can still be unfilmable if it names an ABSTRACT concept — a \
measurement, a state, an effectiveness/performance/stability judgment, a \
cost, a range, a calculation, a comparison — instead of a concrete object or \
action. Do not just shorten the abstract word; TRANSLATE it into what that \
concept would actually look like on camera.
  WRONG (short but still abstract): "chlorine effectiveness", "stable pH \
level", "algae treatment costs"
  RIGHT (concrete translation): "pool chlorine test kit", "person testing \
pool pH", "pool chemical receipt"
Ask "what would a camera actually see if someone were filming this exact \
moment?" — never a measurement, a percentage, a judgment word, or a \
financial/statistical concept on its own.

If the beat mentions a SPECIFIC, CONCRETE, NAMEABLE thing — a dollar \
amount, a named event/incident, a specific object, a specific number/\
quantity — at least one phrase MUST directly reference that literal thing, \
not a general paraphrase of the surrounding sentence.
  Beat: "It cost the company fifty thousand dollars to fix."
    WRONG: "business financial loss" (vague paraphrase)
    RIGHT: "stack of cash money", "dollar bills closeup" (the literal \
concrete thing mentioned — money)
  Beat: "The server crash took down the entire platform."
    WRONG: "technical difficulties" (vague)
    RIGHT: "server room crash error", "computer server malfunction" (the \
literal specific event/object mentioned)
  Beat: "Three hundred inmates were affected."
    WRONG: "prison population issue"
    RIGHT: "group of inmates" (the literal countable thing mentioned)
General rule: if the sentence names a specific dollar figure, a named \
technical/mechanical event, a specific counted quantity, or any other \
concrete nameable thing, at least one phrase should be built around THAT \
thing directly, not a summary of what it implies.

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
exactly 3 strings, each a literal, filmable, SHORT (2-4 word) visual \
description that ALSO complies with the niche restriction above.
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


async def _safe_search(
    provider, query: str, per_page: int = 5, timeout: float = 10.0,
    local_library_niches: list[str] | None = None,
) -> list[StockHit] | None:
    """Returns None (distinct from a genuinely empty []) when the provider
    timed out or failed to connect at all — callers use that signal to stop
    retrying further keywords against this same provider for this clip
    instead of repeating the same doomed call (e.g. a provider that's
    unreachable in this environment, which otherwise burns the full timeout
    once per keyword tried).

    local_library has a different search signature (query, niche, per_page)
    than every other provider (query, per_page) — this is the one chokepoint
    all provider calls already funnel through, so it's the one place that
    special-cases it rather than threading niche through every call site.
    local_library_niches can hold more than one folder (a multi-select
    content_niche sharing one parent, see resolve_sub_niches) — each is
    queried concurrently and the results merged into one candidate pool, so
    a single-select run (exactly one path) behaves identically to before."""
    local_library_niches = local_library_niches or []
    try:
        if provider is local_library:
            if not local_library_niches:
                return []
            results = await asyncio.wait_for(
                asyncio.gather(*(
                    provider.search(query, niche, per_page=per_page) for niche in local_library_niches
                )),
                timeout=timeout,
            )
            merged: list[StockHit] = []
            for hits in results:
                merged.extend(hits)
            return merged
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


def _apply_niche_filter_to_keywords(keywords: list[str], niche_config: NicheConfig) -> list[str]:
    """Same safety net as app.documentary_table._apply_niche_filter, applied
    to this module's flat keyword-list shape instead of table rows — the
    semantic fallback search doesn't go through convert_script_to_visual_table
    at all, so it needs its own filter application at this call site."""
    filtered = []
    for keyword in keywords:
        if violates_niche(keyword, niche_config):
            logger.warning(
                "documentary_pipeline: semantic keyword %r violates niche %r denylist — "
                "replacing with safe fallback %r",
                keyword, niche_config.key, niche_config.safe_fallback_keyword,
            )
            filtered.append(niche_config.safe_fallback_keyword)
        else:
            filtered.append(keyword)
    return filtered


def _resolve_content_niche(content_niche: str | list[str]) -> tuple[NicheConfig, list[str]]:
    """Single-key resolution (a plain string, or a list with only one item —
    i.e. a category with no sub-niches selected) goes through get_niche()
    completely unchanged from before this function existed, so every
    existing test that patches app.documentary_pipeline.get_niche still
    intercepts it. 2+ keys (a real multi-select) go through
    app.niches.resolve_sub_niches instead, which raises ValueError if they
    don't all share one parent category — new code path, no prior behavior
    to preserve. Returns (effective NicheConfig, local_library_paths) — the
    second element lets search_clip query every selected sub-niche's local
    library, not just the single one a NicheConfig has room for."""
    if isinstance(content_niche, str) or len(content_niche) <= 1:
        key = content_niche if isinstance(content_niche, str) else (content_niche[0] if content_niche else DEFAULT_NICHE)
        niche_config = get_niche(key)
        return niche_config, ([niche_config.local_library_path] if niche_config.local_library_path else [])
    return resolve_sub_niches(content_niche)


def _generate_semantic_keywords(
    clip: TimelineClip, tried: list[str], content_niche: str | list[str] = DEFAULT_NICHE,
) -> list[str]:
    niche_config, _ = _resolve_content_niche(content_niche)
    data = generate_json(
        _SEMANTIC_KEYWORD_PROMPT.format(
            beat=clip.script_beat, tried=", ".join(tried), niche_context=niche_config.system_context,
        ),
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
            _SEMANTIC_KEYWORD_PROMPT.format(
                beat=clip.script_beat, tried=", ".join(retry_tried), niche_context=niche_config.system_context,
            ),
            settings.llm_text_model,
        )
        keywords = [str(k) for k in data["keywords"]][:3]

    return _apply_niche_filter_to_keywords(keywords, niche_config)


def _satisfied(hits: list[ScoredAsset]) -> bool:
    """True only when at least one candidate already clears
    documentary_high_quality_score — this is the ONLY legitimate reason for
    search_clip's provider loop to stop early. A handful of merely-acceptable
    (>=documentary_min_score) hits is deliberately NOT enough to stop early
    anymore: that was the actual premature-early-stop bug — the search would
    settle for 2 mediocre 50-70-range hits instead of exhausting the full
    provider list (Pixabay images, Wikimedia, NASA included) in search of
    something genuinely better. Without a high-quality hit, the provider loop
    now always runs to the end of the active provider list — it stops
    naturally once every provider has been queried, not before."""
    return any(a.score >= settings.documentary_high_quality_score for a in hits)


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


# Providers that have ever returned a candidate with no usable text metadata
# (empty tags/title/alt/URL-slug) — logged once per source, not once per
# candidate, so a provider that's consistently metadata-poor doesn't spam
# the log once per clip for the entire run.
_niche_metadata_warned_sources: set[str] = set()


def _check_candidate_niche(hit: StockHit, niche_config: NicheConfig) -> str | None:
    """Pre-scoring niche gate for a stock candidate's own provider metadata
    (tags/title/alt-text/URL slug — see each app.stock.* module's `text`
    field). Returns the matched banned term if the candidate should be
    rejected outright, else None. When a provider gives no usable text at
    all, niche compliance can't be checked pre-download for that candidate —
    that's an acknowledged limitation (see candidate_violates_niche's
    docstring), not something this filter can fix, so it's let through
    unfiltered with a one-time-per-source warning instead of silently
    passing forever with no visibility."""
    if not hit.text.strip():
        if hit.source not in _niche_metadata_warned_sources:
            _niche_metadata_warned_sources.add(hit.source)
            logger.warning(
                "documentary_pipeline: %s returned a candidate with no usable text metadata — "
                "niche compliance cannot be verified pre-download for this source; relying on "
                "the keyword-generation-side denylist only",
                hit.source,
            )
        return None
    return candidate_violates_niche(hit.text, niche_config)


def _filter_providers_by_source_toggles(
    providers: list, content_niche_config: NicheConfig,
    enable_local_library: bool, enable_stock_providers: bool,
) -> list:
    """Shared by search_clip and _provider_count_for so the two can never
    drift apart on what counts as an "active" provider for a given run's
    source toggles (see the Add Source Toggle Checkboxes feature)."""
    if not content_niche_config.use_archive_org:
        # Archive.org's catalog is deep for historical/archival content but
        # has ~nothing for the contemporary lifestyle niches this pipeline
        # currently serves — skip the wasted calls (see NicheConfig.use_archive_org).
        providers = [p for p in providers if p is not internet_archive]
    if not enable_stock_providers:
        providers = [p for p in providers if p is local_library]
    if not enable_local_library or not content_niche_config.local_library_path:
        providers = [p for p in providers if p is not local_library]
    elif local_library in providers and providers[0] is not local_library:
        # Curated, on-topic footage for this niche should be checked before
        # any stock API regardless of PROVIDER_ORDER_DEFAULT/HISTORICAL's own
        # ordering (it's already first there today, but this makes the
        # priority explicit rather than incidental to that list's order).
        providers = [local_library] + [p for p in providers if p is not local_library]
    return providers


def _provider_count_for(
    clip: TimelineClip, content_niche: str | list[str],
    enable_local_library: bool = True, enable_stock_providers: bool = True,
) -> int:
    """Same provider-list resolution search_clip itself uses (niche-ordered,
    Internet Archive dropped when the content niche doesn't use it) —
    recomputed here (cheap, deterministic) purely to report an accurate
    provider count in the placeholder note, without changing search_clip's
    return signature just to thread a count through."""
    providers = provider_order_for_niche(niche_for_clip(clip))
    niche_config, _ = _resolve_content_niche(content_niche)
    providers = _filter_providers_by_source_toggles(
        providers, niche_config, enable_local_library, enable_stock_providers,
    )
    return len(providers)


async def search_clip(
    clip: TimelineClip,
    used_assets: dict[tuple[str, str], int],
    allow_duplicates: bool = False,
    content_niche: str | list[str] = DEFAULT_NICHE,
    use_semantic_fallback: bool = True,
    enable_local_library: bool = True,
    enable_stock_providers: bool = True,
) -> tuple[list[ScoredAsset], list[ScoredAsset]]:
    # `content_niche` (trucks/true_crime/etc, see app.niches) is unrelated to
    # the `niche` below (historical/modern/general, this module's own
    # Niche type) — that one drives stock-provider search order, this one
    # locks keyword *content* to a category. content_niche is a plain key for
    # single-select (unchanged) or a list of sub-niche keys sharing one
    # parent for multi-select (see resolve_sub_niches) — either way
    # content_niche_library_paths ends up holding every local library folder
    # that should be searched for this run (just one, for single-select).
    # `use_semantic_fallback=False` (see check_footage_availability) skips the
    # LLM-generated semantic-keyword tier entirely — used by the pre-render
    # availability scan to keep it free of extra OpenAI cost; the real
    # resolve loop (_resolve_clip) always leaves this at its default True.
    niche = niche_for_clip(clip)
    providers = provider_order_for_niche(niche)
    content_niche_config, content_niche_library_paths = _resolve_content_niche(content_niche)
    providers = _filter_providers_by_source_toggles(
        providers, content_niche_config, enable_local_library, enable_stock_providers,
    )
    progress.update("search", f"Searching stock providers for clip {clip.clip_number} ({clip.section})")

    query_embedding = query_embedding_for(clip)
    video_hits: list[ScoredAsset] = []
    image_hits: list[ScoredAsset] = []
    semantic_keywords: list[str] | None = None

    def _clears_threshold(scored: list[ScoredAsset]) -> bool:
        return any(a.score >= settings.documentary_min_score for a in scored)

    def _score_and_filter(hits: list[StockHit]) -> list[ScoredAsset]:
        """Dedup cap -> low-res floor -> niche filter -> score, in that
        order, same checks the old per-hit loop applied — factored out so
        canva_keyword and fallback_keyword can each be scored independently
        and then merged into one pool, instead of only ever scoring whichever
        tier's raw hits happened to win the old first-usable-tier race."""
        scored: list[ScoredAsset] = []
        for hit in hits:
            key = (hit.source, hit.source_id)
            if not allow_duplicates and used_assets.get(key, 0) >= settings.max_asset_repeat_count:
                continue
            if _known_unusably_low_res(hit):
                logger.info(
                    "documentary_pipeline: clip %d rejecting %s:%s — %dx%d is below the low-res floor",
                    clip.clip_number, hit.source, hit.source_id, hit.width, hit.height,
                )
                continue
            niche_violation = _check_candidate_niche(hit, content_niche_config)
            if niche_violation is not None:
                logger.info(
                    "documentary_pipeline: clip %d [%s] rejected candidate %s:%s — "
                    "niche violation (matched: %r, text: %r)",
                    clip.clip_number, clip.section, hit.source, hit.source_id, niche_violation, hit.text,
                )
                continue
            scored.append(ScoredAsset(hit=hit, score=score_asset(hit, clip, query_embedding)))
        return scored

    for provider in providers:
        # A single genuinely excellent hit anywhere (video or image) is
        # reason enough to stop searching more providers — checked as one
        # combined pool, not per-media-type: image candidates structurally
        # cap out around ~87 under the current weights (no motion credit,
        # a lower cinematic ceiling), so requiring an independent
        # high-quality hit in EACH bucket would make this short-circuit
        # unreachable whenever a search returns any images at all.
        if _satisfied(video_hits + image_hits):
            break

        provider_timed_out = False

        logger.info(
            "documentary_pipeline: clip %d [%s] -> %s search using canva_keyword: %r",
            clip.clip_number, clip.section, provider.name, clip.canva_keyword,
        )
        canva_result = await _safe_search(
            provider, clip.canva_keyword, local_library_niches=content_niche_library_paths
        )
        if canva_result is None:
            logger.warning(
                "documentary_pipeline: clip %d [%s] -> %s timed out/failed to connect on "
                "canva_keyword — skipping remaining keywords for this provider on this clip",
                clip.clip_number, clip.section, provider.name,
            )
            provider_timed_out = True
            scored_pool: list[ScoredAsset] = []
        else:
            scored_pool = _score_and_filter(canva_result)

        # fallback_keyword is now tried, and MERGED into the same pool,
        # whenever canva_keyword's own scored/filtered candidates don't
        # clear documentary_min_score — this covers both the original "zero
        # usable hits" case and the new "real hits, but none score high
        # enough" case with one condition, since an empty/no-passing-score
        # pool naturally satisfies both.
        if not provider_timed_out and not _clears_threshold(scored_pool):
            if scored_pool:
                logger.info(
                    "documentary_pipeline: clip %d [%s] -> %s canva_keyword's best candidate scored "
                    "%.2f (below %.0f) — also searching fallback_keyword to widen the candidate pool",
                    clip.clip_number, clip.section, provider.name,
                    max(a.score for a in scored_pool), settings.documentary_min_score,
                )
            else:
                logger.info(
                    "documentary_pipeline: clip %d [%s] -> %s canva_keyword returned nothing usable — "
                    "also searching fallback_keyword",
                    clip.clip_number, clip.section, provider.name,
                )
            logger.info(
                "documentary_pipeline: clip %d [%s] -> %s search using fallback_keyword: %r",
                clip.clip_number, clip.section, provider.name, clip.fallback_keyword,
            )
            fallback_result = await _safe_search(
                provider, clip.fallback_keyword, local_library_niches=content_niche_library_paths
            )
            if fallback_result is None:
                logger.warning(
                    "documentary_pipeline: clip %d [%s] -> %s timed out/failed to connect on "
                    "fallback_keyword — skipping remaining keywords for this provider on this clip",
                    clip.clip_number, clip.section, provider.name,
                )
            else:
                scored_pool = scored_pool + _score_and_filter(fallback_result)

        # Semantic fallback: unchanged internal behavior (try each of the 3
        # LLM-generated keywords in turn, stop at the first that contributes
        # any usable candidate) — only the trigger condition changed, from
        # "canva+fallback raw hits were empty" to "canva+fallback's merged,
        # scored pool still doesn't clear the acceptance threshold".
        if not provider_timed_out and not _clears_threshold(scored_pool) and use_semantic_fallback:
            if semantic_keywords is None:
                semantic_keywords = await asyncio.to_thread(
                    _generate_semantic_keywords, clip, [clip.canva_keyword, clip.fallback_keyword], content_niche,
                )
            for query in semantic_keywords:
                logger.info(
                    "documentary_pipeline: clip %d [%s] -> %s search using semantic_fallback "
                    "(canva/fallback keywords didn't clear the acceptance threshold): %r",
                    clip.clip_number, clip.section, provider.name, query,
                )
                result = await _safe_search(
                    provider, query, local_library_niches=content_niche_library_paths
                )
                if result is None:
                    logger.warning(
                        "documentary_pipeline: clip %d [%s] -> %s timed out/failed to connect on "
                        "semantic_fallback — skipping remaining keywords for this provider on this clip",
                        clip.clip_number, clip.section, provider.name,
                    )
                    break
                semantic_scored = _score_and_filter(result)
                if semantic_scored:
                    scored_pool = scored_pool + semantic_scored
                    break
                if result:
                    logger.info(
                        "documentary_pipeline: clip %d [%s] -> %s semantic_fallback %r returned only "
                        "already-used/filtered candidate(s) — trying next semantic keyword",
                        clip.clip_number, clip.section, provider.name, query,
                    )

        candidates_before = len(video_hits) + len(image_hits)
        for scored in scored_pool:
            (video_hits if scored.hit.media_type == "video" else image_hits).append(scored)
        candidates_after = len(video_hits) + len(image_hits)
        if scored_pool:
            logger.info(
                "documentary_pipeline: clip %d [%s] -> %s contributed %d scored candidate(s) "
                "(running total %d -> %d)",
                clip.clip_number, clip.section, provider.name,
                len(scored_pool), candidates_before, candidates_after,
            )

        if provider is local_library and scored_pool:
            # A strong enough local match is preferable to an unknown stock
            # candidate — accept it directly rather than spending time/API
            # calls searching Pexels/Pixabay/etc. for this clip. Uses the
            # (already local-library-boosted, see score_asset) AI-generation
            # trigger threshold as "good enough" rather than the stricter
            # documentary_high_quality_score _satisfied uses below, since a
            # curated local match clearing the bar that would otherwise
            # trigger an AI-generated guess is exactly the case a real,
            # on-topic clip should win outright.
            best_local_score = max(a.score for a in scored_pool)
            if best_local_score >= settings.ai_generation_trigger_threshold:
                logger.info(
                    "documentary_pipeline: clip %d [%s] -> local_library candidate scored %.2f "
                    "(>= AI-generation trigger threshold %.0f) — accepting directly, skipping "
                    "remaining stock providers for this clip",
                    clip.clip_number, clip.section, best_local_score, settings.ai_generation_trigger_threshold,
                )
                break

    _log_top_scored_candidates(clip, video_hits, image_hits, query_embedding)
    progress.update(
        "score",
        f"Scored {len(video_hits) + len(image_hits)} candidate(s), applying niche filter & visual "
        f"verification for clip {clip.clip_number} ({clip.section})",
    )
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
        keyword_match = keyword_overlap_ratio(clip.canva_keyword, scored.hit.text)
        logger.info(
            "documentary_pipeline: clip %d [%s] top-%d: %s:%s (%s) total_score=%.2f "
            "semantic_similarity=%.3f keyword_match=%.3f text=%r%s",
            clip.clip_number, clip.section, rank, scored.hit.source, scored.hit.source_id,
            scored.hit.media_type, scored.score, similarity, keyword_match, scored.hit.text[:80],
            " <- WINNER" if scored is winner else "",
        )


def _media_type_bias(media_type_counts: dict[str, int]) -> dict[str, float]:
    """Small ranking-only score bonus/penalty per media type, pushing the
    project's running image/video mix toward settings.documentary_target_image_ratio
    over the course of a video. Under-represented type gets a bonus,
    over-represented type gets a penalty. Magnitude is capped at +/-8 (on the
    0-100 score scale) so it can only break a near-tie between candidates,
    never override a real quality gap — see rank_acceptable_assets, which is
    the only place this is added to a candidate's score."""
    target = settings.documentary_target_image_ratio
    total = media_type_counts["image"] + media_type_counts["video"]
    current_image_ratio = target if total == 0 else media_type_counts["image"] / total
    deficit = target - current_image_ratio  # positive = running mix is short on images
    image_bonus = max(-8.0, min(8.0, deficit * 20))
    return {"image": image_bonus, "video": -image_bonus}


def rank_acceptable_assets(
    video_hits: list[ScoredAsset],
    image_hits: list[ScoredAsset],
    media_type_counts: dict[str, int] | None = None,
) -> list[ScoredAsset]:
    """Every acceptable hit (score >= documentary_min_score) across both media
    types, ranked purely by score, best first — no hard preference for video
    over image. score_asset's own weighting (motion + cinematic value) tends
    to score video higher, but a stronger-scoring image still wins here; there
    is no separate rule privileging one media type. _resolve_clip walks this
    list in order so a top pick that fails the 16:9 check falls through to
    the next-best rather than giving up straight to a placeholder.

    media_type_counts, when given, is the project's running count of
    successfully-resolved clips per media type (see run()) — it only nudges
    sort order via _media_type_bias, never changes a.score itself, so
    score_asset's return value (persisted as AssetInfo.score, reported in
    footage_availability_report.json and the keyword-match report) stays a
    clean, project-state-independent number. Callers that don't track a
    running mix (check_footage_availability's simulated scan, pick_best_asset)
    simply omit it and get the unbiased ranking as before."""
    acceptable = [a for a in video_hits + image_hits if a.score >= settings.documentary_min_score]
    if media_type_counts is None:
        return sorted(acceptable, key=lambda a: a.score, reverse=True)
    bias = _media_type_bias(media_type_counts)
    return sorted(acceptable, key=lambda a: a.score + bias[a.hit.media_type], reverse=True)


def pick_best_asset(video_hits: list[ScoredAsset], image_hits: list[ScoredAsset]) -> ScoredAsset | None:
    """Single winner per clip: highest-scoring acceptable hit across both media
    types (see rank_acceptable_assets for the no-video-preference rationale)."""
    ranked = rank_acceptable_assets(video_hits, image_hits)
    return ranked[0] if ranked else None


async def check_footage_availability(
    table: list[TimelineClip], content_niche: str | list[str] = DEFAULT_NICHE,
    enable_local_library: bool = True, enable_stock_providers: bool = True,
) -> AvailabilityReport:
    """Cheap pre-render scan: walks the whole timeline calling search_clip
    (real provider searches, niche filter, scoring) with
    use_semantic_fallback=False, so it costs extra HTTP calls to free stock
    APIs but no extra LLM calls — reaching "would need the semantic
    fallback" is itself treated as a thin-candidate signal, not something
    worth resolving with a paid call during a pre-check. Never downloads
    anything, applies no CLIP verification, and never touches the real
    used_assets a subsequent run() call will use — it keeps its own
    simulated dedup counter (same cap semantics as the real resolve loop, see
    settings.max_asset_repeat_count) so a scarce/popular asset isn't
    double-counted as 'available' for every clip that could use it.

    This is a deliberate approximation, not a full dry-run simulator: since
    it never downloads anything, it can't know about post-download
    rejections (16:9 normalization failure, CLIP score miss — see
    _resolve_clip) that the real run might still hit even on a candidate this
    scan counted as acceptable. It's meant to catch a thin candidate pool
    before render, not to predict every clip's exact outcome."""
    simulated_used: dict[tuple[str, str], int] = {}
    clips: list[ClipAvailability] = []
    thin_clip_numbers: list[int] = []

    for clip in table:
        video_hits, image_hits = await search_clip(
            clip, simulated_used, allow_duplicates=False,
            content_niche=content_niche, use_semantic_fallback=False,
            enable_local_library=enable_local_library, enable_stock_providers=enable_stock_providers,
        )
        acceptable = rank_acceptable_assets(video_hits, image_hits)
        thin = len(acceptable) < settings.footage_availability_min_candidates
        if thin:
            thin_clip_numbers.append(clip.clip_number)
        if acceptable:
            top_key = (acceptable[0].hit.source, acceptable[0].hit.source_id)
            simulated_used[top_key] = simulated_used.get(top_key, 0) + 1
        clips.append(ClipAvailability(
            clip_number=clip.clip_number, section=clip.section,
            candidate_count=len(acceptable),
            top_score=acceptable[0].score if acceptable else None,
            thin=thin,
        ))

    total = len(table)
    if total and len(thin_clip_numbers) / total >= settings.footage_availability_warn_ratio:
        logger.warning(
            "documentary_pipeline: LOW FOOTAGE AVAILABILITY — %d/%d clip(s) (%.0f%%) have fewer than "
            "%d acceptable candidate(s) using canva/fallback keywords alone (before the semantic-keyword "
            "fallback and before any download/CLIP verification); review the niche/keywords before "
            "committing to the full render. Thin clips: %s",
            len(thin_clip_numbers), total, 100 * len(thin_clip_numbers) / total,
            settings.footage_availability_min_candidates, thin_clip_numbers,
        )

    return AvailabilityReport(
        total_clips=total, thin_count=len(thin_clip_numbers),
        thin_clip_numbers=thin_clip_numbers, clips=clips,
    )


async def _download_asset(asset: ScoredAsset, dest_dir: Path, prefix: str, index: int, target_duration: float) -> AssetInfo:
    """target_duration is the table row's own timeline length, not a flat
    cap. download_video_trimmed's -t is a ceiling, not a mandate — if the
    source is shorter than target_duration, ffmpeg just fetches the whole
    (shorter) source; the assembly stage then either speed-adjusts it or
    (see _fetch_fill_asset below) pairs it with a second, distinct clip to
    cover the remainder — it never loops/repeats the same source to fill
    the row."""
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
        selected_text=hit.text,
    )


async def _fetch_fill_asset(
    candidates: list[ScoredAsset], accepted_index: int, accepted_key: tuple[str, str],
    clip: TimelineClip, clip_dir: Path, prefix: str, remaining_duration: float,
    used_assets: dict[tuple[str, str], int], downloads_so_far: int, limit: int | None,
) -> tuple[AssetInfo | None, int]:
    """Walks the ranked candidates AFTER the just-accepted primary one
    (accepted_index is its 1-based position in `candidates`) for the first
    genuinely distinct asset — never the primary's own (source, source_id),
    and never one already at/over its dedup cap in used_assets (see
    settings.max_asset_repeat_count) — to cover the remaining_duration a
    too-short primary clip left unfilled. Downloads and 16:9-normalizes it,
    same bar as the main accept loop above, minus the CLIP re-verification:
    every candidate here already cleared documentary_min_score and niche
    filtering pre-download same as the primary did, and this fill segment
    only covers PART of the row, not the whole thing. Returns (None,
    downloads_so_far) if the download limit is hit or no candidate survives
    — the caller (_resolve_clip) then leaves entry.fill_asset_path unset and
    documentary_assembly falls back to a forced speed adjustment instead of
    a loop (see _render_row_segment)."""
    for index, candidate in enumerate(candidates[accepted_index:], start=accepted_index + 1):
        key = (candidate.hit.source, candidate.hit.source_id)
        if key == accepted_key or used_assets.get(key, 0) >= settings.max_asset_repeat_count:
            continue
        if limit is not None and downloads_so_far >= limit:
            return None, downloads_so_far
        try:
            asset = await _download_asset(candidate, clip_dir, prefix, index, remaining_duration)
        except Exception as exc:
            logger.warning(
                "documentary_pipeline: clip %d fill-candidate %d/%d (%s:%s) failed to download (%s), "
                "trying next-best candidate",
                clip.clip_number, index, len(candidates), key[0], key[1], exc,
            )
            continue
        downloads_so_far += 1

        _duration, resolution = await asyncio.to_thread(probe, asset.path)
        width, height = _parse_resolution(resolution)
        if not _is_exact_16_9(width, height):
            if _is_too_low_res(width, height):
                Path(asset.path).unlink(missing_ok=True)
                continue
            normalized = await asyncio.to_thread(normalize_to_16_9, Path(asset.path), asset.media_type, width, height)
            if not normalized:
                Path(asset.path).unlink(missing_ok=True)
                continue
            asset = asset.model_copy(update={"width": _WIDTH, "height": _HEIGHT})

        used_assets[key] = used_assets.get(key, 0) + 1
        logger.info(
            "documentary_pipeline: clip %d — primary candidate too short for a natural speed "
            "adjustment, filling remaining %.2fs with distinct candidate %s:%s",
            clip.clip_number, remaining_duration, key[0], key[1],
        )
        return asset, downloads_so_far
    return None, downloads_so_far


def _to_alternate_candidates(candidates: list[ScoredAsset], limit: int = 5) -> list[AlternateCandidate]:
    """Top-N ranked-but-not-necessarily-downloaded candidates, persisted onto
    TimelineEntry.alternates for the editor's "swap to alternate" feature
    (see app/main.py's PATCH /project/{project_name}/clip/{clip_number})."""
    return [
        AlternateCandidate(
            source=c.hit.source, source_id=c.hit.source_id, download_url=c.hit.download_url,
            media_type=c.hit.media_type, width=c.hit.width, height=c.hit.height,
            duration=c.hit.duration, text=c.hit.text, score=c.score,
        )
        for c in candidates[:limit]
    ]


def _placeholder_entry(
    clip: TimelineClip, note: str,
    generation_attempted: bool = False, generation_failure_reason: str | None = None,
    alternates: list[AlternateCandidate] | None = None,
) -> TimelineEntry:
    effect, transition = edit_recommendation(clip.visual_type)
    return TimelineEntry(
        clip_number=clip.clip_number, start=clip.start, end=clip.end,
        duration=_duration_seconds(clip.start, clip.end), script=clip.script_beat,
        asset_path=None, asset_metadata=None, alternates=alternates or [],
        recommended_effect=effect, transition=transition, placeholder=True, note=note,
        generation_attempted=generation_attempted, generation_failure_reason=generation_failure_reason,
    )


def _should_try_ai_generation(best_real_candidate: ScoredAsset | None, enable_ai_generation: bool = True) -> bool:
    """Trigger condition: fires whenever there's no real candidate at all, OR
    the best one search found scores below ai_generation_trigger_threshold —
    a quality gate, not a last resort (see settings.ai_generation_trigger_threshold's
    docstring for why most clips are expected to trigger this once enabled).
    enable_ai_generation is the per-request UI toggle; settings.enable_ai_generation_fallback
    remains the separate server-wide kill switch — both must allow it."""
    if not enable_ai_generation or not settings.enable_ai_generation_fallback:
        return False
    return best_real_candidate is None or best_real_candidate.score < settings.ai_generation_trigger_threshold


def _build_generation_prompt(clip: TimelineClip, niche_config: NicheConfig) -> str:
    """canva_keyword alone (2-4 words) is too thin an image-generation prompt
    — combines it with edit_note's richer scene detail (setting/framing/
    lighting, see CONVERSION_PROMPT's edit_note rule in documentary_table.py)
    plus a niche/style suffix so the generated image has more to go on than
    a bare search phrase does. edit_note is otherwise search-inert (only
    canva_keyword/fallback_keyword/semantic-fallback ever reach a stock
    provider, see search_clip) — safe to lean on here with zero effect on
    real search."""
    parts = [clip.canva_keyword]
    if clip.edit_note:
        parts.append(clip.edit_note)
    prompt = ". ".join(parts)
    prompt += f". {niche_config.display_name} context, photorealistic, natural lighting."
    return prompt


async def _enhance_prompt_for_generation(clip: TimelineClip, niche_config: NicheConfig, base_prompt: str) -> str:
    """Expands the thin canva_keyword+edit_note prompt into a richer,
    more specific one before it reaches image generation — GPT fills in
    subject/action/setting/lighting/composition detail a bare keyword
    phrase doesn't carry. Grounds the expansion in the clip's own
    script_beat (not just canva_keyword+edit_note) so the generated image
    reflects THIS narration line specifically, and explicitly asks for
    varied framing/angle/composition — many clips share a section and
    similar keywords (e.g. repeated "Algae covered pool" across a
    documentary), which risks repetitive-looking generated images if every
    prompt defaults to the same generic establishing shot. Falls back to
    base_prompt untouched on any failure/malformed response, same
    non-blocking philosophy as _keywords_are_concrete."""
    try:
        data = await asyncio.to_thread(
            generate_json,
            "Write a detailed, vivid image-generation prompt (2-3 sentences) "
            f"for this specific narration line: \"{clip.script_beat}\"\n"
            f"Keyword: {clip.canva_keyword}. Note: {clip.edit_note}.\n"
            f"Context: {niche_config.display_name}.\n"
            "Make it visually specific to THIS line, not generic — vary "
            "framing/angle/composition rather than defaulting to a standard shot.\n"
            "Return JSON: {\"prompt\": \"...\"}",
            settings.llm_text_model,
        )
        enhanced = data.get("prompt")
        return enhanced if isinstance(enhanced, str) and enhanced.strip() else base_prompt
    except Exception as exc:
        logger.warning("documentary_pipeline: prompt enhancement failed (%s), using base prompt", exc)
        return base_prompt


async def _try_ai_generation_fallback(
    clip: TimelineClip, clip_dir: Path, best_real_candidate: ScoredAsset | None, content_niche: str | list[str],
    used_assets: dict[tuple[str, str], int], alternates: list[AlternateCandidate] | None = None,
) -> tuple[TimelineEntry | None, str | None]:
    """Attempts generation for a clip _should_try_ai_generation already
    flagged. Returns (None, failure_reason) (never raises) when generation
    itself fails, so the caller degrades gracefully: falls back to accepting
    best_real_candidate anyway (a real, if below-threshold, clip beats a
    black placeholder) rather than treating a failed generation call as a
    reason to give up — only produces a placeholder when BOTH real search
    and generation come up empty (see _resolve_clip). failure_reason is
    persisted onto the eventually-accepted entry (generation_failure_reason)
    so a real-candidate-after-failed-generation clip stays distinguishable
    from one that never attempted generation at all, in both timeline.json
    and keyword_match_report.md — not just an ephemeral log line."""
    niche_config, _ = _resolve_content_niche(content_niche)
    base_prompt = _build_generation_prompt(clip, niche_config)
    prompt = await _enhance_prompt_for_generation(clip, niche_config, base_prompt)
    logger.info(
        "documentary_pipeline: clip %d [%s] AI generation prompt: %r",
        clip.clip_number, clip.section, prompt,
    )
    output_path = clip_dir / f"clip{clip.clip_number:03d}_generated.jpg"
    success, failure_reason = await generate_fallback_image_openai(
        prompt, output_path,
        rendering_instructions=(
            "Create a polished photorealistic 16:9 landscape image. "
            "Use realistic lighting, accurate anatomy, natural textures, "
            "strong subject separation, and a clean cinematic composition. "
            "Do not include captions, logos, watermarks, borders, or UI."
        ),
    )
    if not success:
        return None, failure_reason

    # Probe the actual generated file instead of assuming a fixed size —
    # settings.ai_generation_model decides the real output size (gpt-image-2's
    # 2048x1152 is already exact 16:9; gpt-image-1-mini's 1536x1024 is not).
    # Always normalize regardless as a safety net (e.g. an unexpected
    # fallback size) — for an already-exact-16:9 source this is just a clean
    # scale down to 1920x1080, no crop/pad artifacts, same as the
    # normalize_to_16_9 crop used for near-16:9 real candidates (see the
    # main resolve loop below).
    _duration, resolution = await asyncio.to_thread(probe, str(output_path))
    gen_width, gen_height = _parse_resolution(resolution)
    normalized = await asyncio.to_thread(normalize_to_16_9, output_path, "image", gen_width, gen_height)
    if not normalized:
        return None, "failed to normalize AI-generated image to 16:9"

    if best_real_candidate is not None:
        note = (
            f"best real candidate scored {best_real_candidate.score:.1f} "
            f"(below trigger threshold {settings.ai_generation_trigger_threshold:.1f}) — "
            f"AI-generated via OpenAI {settings.ai_generation_model} instead"
        )
    else:
        note = f"no real candidate found — AI-generated via OpenAI {settings.ai_generation_model}"

    # No CLIP verification here — the image was generated FOR canva_keyword,
    # not searched and matched against it, so there's nothing independent to
    # verify it against (unlike a real stock candidate). If real usage shows
    # generated images sometimes miss the intended keyword, a CLIP sanity
    # check could be added later; skipped for now to keep this simple until
    # there's real output quality to evaluate.
    # source_id is the output path, not "" — each generated image is its own
    # unique file (named by clip_number), and an empty source_id would make
    # every AI-generated entry collide on the same ("ai_generated", "") dedup
    # key. Registered in used_assets below for the same reason every other
    # accept path registers its asset: consistency, even though a freshly
    # generated file can never itself be a repeat.
    asset = AssetInfo(
        path=str(output_path), source="ai_generated", source_id=str(output_path), media_type="image",
        width=_WIDTH, height=_HEIGHT, duration=0, score=None,
        selected_text=f"(AI-generated via OpenAI {settings.ai_generation_model})",
    )
    asset_key = (asset.source, asset.source_id)
    used_assets[asset_key] = used_assets.get(asset_key, 0) + 1
    effect, transition = edit_recommendation(clip.visual_type)
    return TimelineEntry(
        clip_number=clip.clip_number, start=clip.start, end=clip.end,
        duration=_duration_seconds(clip.start, clip.end), script=clip.script_beat,
        asset_path=str(output_path), asset_metadata=asset, alternates=alternates or [],
        recommended_effect=effect, transition=transition, placeholder=False,
        note=note, generation_attempted=True,
    ), None


async def _resolve_clip(
    clip: TimelineClip,
    clips_root: Path,
    used_assets: dict[tuple[str, str], int],
    allow_duplicates: bool,
    downloads_so_far: int,
    limit: int | None,
    content_niche: str | list[str] = DEFAULT_NICHE,
    media_type_counts: dict[str, int] | None = None,
    enable_ai_generation: bool = True,
    enable_local_library: bool = True,
    enable_stock_providers: bool = True,
) -> tuple[TimelineEntry, int]:
    if limit is not None and downloads_so_far >= limit:
        return _placeholder_entry(clip, "project download limit reached"), downloads_so_far

    video_hits, image_hits = await search_clip(
        clip, used_assets, allow_duplicates, content_niche,
        enable_local_library=enable_local_library, enable_stock_providers=enable_stock_providers,
    )
    candidates = rank_acceptable_assets(video_hits, image_hits, media_type_counts)
    best_real_candidate = candidates[0] if candidates else None
    alternates = _to_alternate_candidates(candidates)

    clip_dir = clips_root / f"clip_{clip.clip_number:03d}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    generation_attempted = False
    generation_failure_reason: str | None = None

    if _should_try_ai_generation(best_real_candidate, enable_ai_generation):
        generation_attempted = True
        generated, generation_failure_reason = await _try_ai_generation_fallback(
            clip, clip_dir, best_real_candidate, content_niche, used_assets, alternates,
        )
        if generated is not None:
            return generated, downloads_so_far
        if best_real_candidate is not None:
            logger.warning(
                "documentary_pipeline: clip %d [%s] AI generation failed (%s) — falling back to best real "
                "candidate (score %.1f, below trigger threshold %.1f) rather than a placeholder",
                clip.clip_number, clip.section, generation_failure_reason,
                best_real_candidate.score, settings.ai_generation_trigger_threshold,
            )
        # best_real_candidate is None here -> falls through to the "no candidates" placeholder below,
        # same as the disabled/above-threshold path already does.

    if not candidates:
        note = (
            f"no candidate scoring >= {settings.documentary_min_score:.0f} found after searching "
            f"all {_provider_count_for(clip, content_niche, enable_local_library, enable_stock_providers)} "
            "providers across 3 keyword tiers (canva_keyword, fallback_keyword, semantic_fallback)"
        )
        return _placeholder_entry(clip, note, generation_attempted, generation_failure_reason, alternates), downloads_so_far

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
    progress.update("download", f"Downloading candidate(s) for clip {clip.clip_number} ({clip.section})")
    for index, candidate in enumerate(candidates, start=1):
        if limit is not None and downloads_so_far >= limit:
            return _placeholder_entry(clip, "project download limit reached", alternates=alternates), downloads_so_far

        try:
            asset = await _download_asset(candidate, clip_dir, prefix, index, target_duration)
        except Exception as exc:
            logger.warning(
                "documentary_pipeline: clip %d candidate %d/%d (%s:%s) failed to download (%s), "
                "trying next-best candidate",
                clip.clip_number, index, len(candidates), candidate.hit.source, candidate.hit.source_id, exc,
            )
            last_rejection_reason = f"{candidate.hit.source}:{candidate.hit.source_id} failed to download ({exc})"
            continue
        asset_key = (asset.source, asset.source_id)
        used_assets[asset_key] = used_assets.get(asset_key, 0) + 1
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

        # canva_keyword alone, NOT canva_keyword + script_beat — script_beat is
        # full narration prose (dosage amounts, chemistry explanation, etc.),
        # and diluting/truncating past CLIP's 77-token limit with that prose
        # was confirmed (real pipeline run, see git history) to drag genuinely
        # correct visual matches below threshold: three actual weighing-scale
        # videos for a "weighing pool chemicals" beat scored 0.227-0.236
        # against the diluted text and were wrongly rejected. canva_keyword is
        # already a short, concrete, filmable phrase by construction (see the
        # niche-lock keyword-generation prompting), which is exactly what
        # CLIP's text encoder needs.
        passed, visual_score = await asyncio.to_thread(passes_visual_verification, asset.path, clip.canva_keyword)
        if not passed:
            logger.info(
                "documentary_pipeline: clip %d [%s] rejected candidate %d/%d (%s:%s) — visual verification "
                "failed (CLIP similarity %.3f below threshold %.3f)",
                clip.clip_number, clip.section, index, len(candidates), asset.source, asset.source_id,
                visual_score, settings.visual_verification_threshold,
            )
            last_rejection_reason = (
                f"{asset.source}:{asset.source_id} failed visual verification "
                f"(CLIP similarity {visual_score:.3f} < {settings.visual_verification_threshold:.3f})"
            )
            Path(asset.path).unlink(missing_ok=True)
            continue

        # _should_try_ai_generation above only ever ran once, against
        # candidates[0]'s PRE-download score — if that (or any higher-scoring)
        # candidate then failed post-download verification/normalization and
        # the loop fell through to THIS one, that original verdict is stale:
        # this specific candidate may itself be below the trigger threshold
        # even though the best one search found wasn't. Re-check against the
        # candidate actually about to be accepted, once per clip (guarded by
        # generation_attempted, same as every other trigger point), instead
        # of silently accepting whatever real asset happens to survive the
        # rejection cascade regardless of its own score. Real regression:
        # a clip scoring 58 was accepted this way with generation_attempted
        # left False (see git history for the production timeline.json
        # evidence this was found from).
        if not generation_attempted and _should_try_ai_generation(candidate, enable_ai_generation):
            generation_attempted = True
            generated, generation_failure_reason = await _try_ai_generation_fallback(
                clip, clip_dir, candidate, content_niche, used_assets, alternates,
            )
            if generated is not None:
                return generated, downloads_so_far
            logger.warning(
                "documentary_pipeline: clip %d [%s] AI generation failed (%s) — the top candidate(s) were "
                "rejected post-verification and this one (score %.1f, below trigger threshold %.1f) is what "
                "survived; falling back to it rather than a placeholder",
                clip.clip_number, clip.section, generation_failure_reason,
                candidate.score, settings.ai_generation_trigger_threshold,
            )

        logger.info(
            "documentary_pipeline: clip %d accepted candidate %d/%d (%s:%s) — %d rejected before it",
            clip.clip_number, index, len(candidates), asset.source, asset.source_id, index - 1,
        )
        if media_type_counts is not None:
            media_type_counts[asset.media_type] = media_type_counts.get(asset.media_type, 0) + 1
        effect, transition = edit_recommendation(clip.visual_type)
        note = (
            f"AI generation attempted and failed ({generation_failure_reason}) — "
            f"accepted this real candidate instead" if generation_attempted else ""
        )

        # Real duration too short for even a natural-looking speed adjustment
        # (assembly's own _SPEED_MATCH_MIN/MAX range) — fetch a second,
        # distinct candidate to cover the remainder now, while the full
        # ranked `candidates` list is still in scope, rather than let
        # documentary_assembly discover this at render time with nothing
        # left to draw from but a loop (see _fetch_fill_asset).
        fill_asset: AssetInfo | None = None
        if (
            asset.media_type == "video"
            and _actual_duration > _DURATION_EPSILON and target_duration > _DURATION_EPSILON
            and _actual_duration < target_duration - _DURATION_EPSILON
            and not (_SPEED_MATCH_MIN <= _actual_duration / target_duration <= _SPEED_MATCH_MAX)
        ):
            fill_asset, downloads_so_far = await _fetch_fill_asset(
                candidates, index, asset_key, clip, clip_dir, f"{prefix}_fill",
                target_duration - _actual_duration, used_assets, downloads_so_far, limit,
            )

        entry = TimelineEntry(
            clip_number=clip.clip_number, start=clip.start, end=clip.end,
            duration=target_duration, script=clip.script_beat,
            asset_path=asset.path, asset_metadata=asset,
            fill_asset_path=fill_asset.path if fill_asset else None,
            fill_asset_metadata=fill_asset,
            alternates=alternates,
            recommended_effect=effect, transition=transition, note=note,
            generation_attempted=generation_attempted, generation_failure_reason=generation_failure_reason,
        )
        return entry, downloads_so_far

    # Last-resort trigger, distinct from _should_try_ai_generation above: that
    # check only ever runs once, against the best candidate's PRE-download
    # score — a candidate that looked good enough there can still fail
    # post-download CLIP verification (or normalization), and if every other
    # candidate is exhausted too, the pre-download check's "looks fine, skip
    # generation" verdict is now stale. Only fires if generation wasn't
    # already attempted above (that path already covers the case where no
    # real candidate existed at all, or the best one was already below
    # threshold).
    if enable_ai_generation and settings.enable_ai_generation_fallback and not generation_attempted:
        generation_attempted = True
        generated, generation_failure_reason = await _try_ai_generation_fallback(
            clip, clip_dir, None, content_niche, used_assets, alternates,
        )
        if generated is not None:
            note = (
                f"all {len(candidates)} real candidate(s) rejected post-verification"
                + (f" (last: {last_rejection_reason})" if last_rejection_reason else "")
                + f" — AI-generated via OpenAI {settings.ai_generation_model} as last resort"
            )
            return generated.model_copy(update={"note": note}), downloads_so_far
        logger.warning(
            "documentary_pipeline: clip %d [%s] last-resort AI generation failed (%s) — falling back to placeholder",
            clip.clip_number, clip.section, generation_failure_reason,
        )

    note = f"all {len(candidates)} acceptable candidate(s) were rejected (resolution and/or visual verification)"
    if last_rejection_reason:
        note += f" (last: {last_rejection_reason})"
    return _placeholder_entry(clip, note, generation_attempted, generation_failure_reason, alternates), downloads_so_far


def resolve_script_text(script: str, script_path: str | None) -> str:
    """Shared by run() and the /generate-documentary-timeline/preview
    endpoint (see app/main.py) — reads script_path if given, and rejects
    empty script text either way, before anything else happens."""
    if script_path:
        script = Path(script_path).read_text(encoding="utf-8")
    if not script or not script.strip():
        raise ValueError("documentary_pipeline.run requires non-empty script text (via `script` or `script_path`)")
    return script


@dataclass
class DocumentaryPlan:
    project_name: str
    table: list[TimelineClip]
    footage_availability: AvailabilityReport
    whisper_words: list[WordTiming] | None
    transcript_alignment_warning: str | None


async def plan_documentary(
    script: str,
    project_name: str,
    audio_path: str | None = None,
    content_niche: str | list[str] = DEFAULT_NICHE,
    enable_local_library: bool = True,
    enable_stock_providers: bool = True,
) -> DocumentaryPlan:
    """Everything run() does before any per-clip download/CLIP-verify/render
    work: transcribes narration audio if supplied, generates the visual
    timeline table, and runs the pre-render footage availability scan (see
    check_footage_availability). Factored out so
    /generate-documentary-timeline/preview can produce the exact same table
    and availability report a real run would, without duplicating this logic
    or paying for any of the expensive per-clip work.

    Caller is responsible for resolving/validating script text (see
    resolve_script_text), resolving project_name, and progress-tracking
    context (progress.set_context) — this function only labels the returned
    plan with project_name and otherwise assumes progress.update calls here
    are safe to no-op (e.g. when called from the preview endpoint, which
    doesn't set a progress context)."""
    # Manually-supplied narration audio (no TTS integration — the user
    # provides this file themselves): real Whisper word timestamps replace
    # the word-count pacing estimate as the source of truth for both caption
    # text/timing and per-clip narration duration. check_alignment is a
    # sanity check only — it never changes what gets used, just flags
    # significant script/transcript deviation for manual review.
    whisper_words: list[WordTiming] | None = None
    transcript_alignment_warning: str | None = None
    if audio_path:
        progress.update("input", "Transcribing narration audio")
        whisper_words = await asyncio.to_thread(get_narration_timing, audio_path)
        transcript_alignment_warning = await asyncio.to_thread(check_alignment, whisper_words, script)

    progress.update("table", "Generating visual timeline table (LLM)")
    table = await asyncio.to_thread(generate_table, script, whisper_words, content_niche)

    progress.update("availability", "Scanning footage availability across the full timeline")
    footage_availability = await check_footage_availability(
        table, content_niche, enable_local_library=enable_local_library, enable_stock_providers=enable_stock_providers,
    )

    return DocumentaryPlan(
        project_name=project_name, table=table, footage_availability=footage_availability,
        whisper_words=whisper_words, transcript_alignment_warning=transcript_alignment_warning,
    )


def render_keyword_match_report(table: list[TimelineClip], timeline: list[TimelineEntry]) -> str:
    """One row per clip: canva_keyword vs. the actual selected candidate's own
    text metadata and keyword_match ratio — persisted so future match-quality
    review never again requires reconstructing this after the fact via manual
    per-ID provider lookups (see docs/clip_match_diagnostic.md, which had to
    do exactly that). Reuses keyword_overlap_ratio directly; no new scoring.

    The Outcome column distinguishes three cases that would otherwise be
    indistinguishable from asset_metadata alone: "real" (score cleared
    ai_generation_trigger_threshold, generation never attempted),
    "ai_generated" (generation succeeded), and "real (generation failed:
    ...)" (score was below threshold, generation was attempted and failed,
    and this real candidate was accepted as the graceful-degradation
    fallback) — see TimelineEntry.generation_attempted/
    generation_failure_reason."""
    header = "| Clip # | Canva Keyword | Selected (source:id) | Selected Text | Score | keyword_match | Outcome |\n"
    sep = "|---|---|---|---|---|---|---|\n"
    entries_by_clip = {e.clip_number: e for e in timeline}
    rows = []
    for clip in table:
        entry = entries_by_clip.get(clip.clip_number)
        asset = entry.asset_metadata if entry else None
        if entry is None or entry.placeholder or asset is None:
            reason = entry.note if entry else "no timeline entry"
            rows.append(
                f"| {clip.clip_number} | {clip.canva_keyword} | — | "
                f"(placeholder — no candidate accepted: {reason}) | — | — | placeholder |\n"
            )
            continue
        text = asset.selected_text.replace("|", "/")
        keyword_match = keyword_overlap_ratio(clip.canva_keyword, asset.selected_text)
        if asset.source == "ai_generated":
            outcome = "ai_generated"
        elif entry.generation_attempted:
            outcome = f"real (generation failed: {entry.generation_failure_reason})"
        else:
            outcome = "real"
        rows.append(
            f"| {clip.clip_number} | {clip.canva_keyword} | {asset.source}:{asset.source_id} | "
            f"{text} | {asset.score} | {keyword_match:.2f} | {outcome} |\n"
        )
    return header + sep + "".join(rows)


def _load_timeline(project_dir: Path) -> list[TimelineEntry]:
    path = project_dir / "timeline.json"
    if not path.exists():
        raise ValueError(f"no timeline.json found in {project_dir} — run the full pipeline at least once first")
    data = json.loads(path.read_text(encoding="utf-8"))
    return [TimelineEntry(**item) for item in data]


def _write_timeline(project_dir: Path, timeline: list[TimelineEntry]) -> None:
    (project_dir / "timeline.json").write_text(
        json.dumps([entry.model_dump() for entry in timeline], indent=2), encoding="utf-8"
    )


def _load_project_meta(project_dir: Path) -> ProjectMeta:
    path = project_dir / "project_meta.json"
    if not path.exists():
        raise ValueError(f"no project_meta.json found in {project_dir} — run the full pipeline at least once first")
    return ProjectMeta.model_validate_json(path.read_text(encoding="utf-8"))


def _write_project_meta(project_dir: Path, meta: ProjectMeta) -> None:
    (project_dir / "project_meta.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")


async def _materialize_alternate_asset(
    alt: AlternateCandidate, clip_dir: Path, clip_number: int, target_duration: float,
) -> AssetInfo:
    """Downloads and 16:9-normalizes ONE editor-chosen alternate — unlike the
    main _resolve_clip loop, there's no "try the next candidate" fallback
    here: this candidate was explicitly picked by a human, so a failure is
    raised straight to the caller instead of silently substituting something
    else. No CLIP re-verification either, for the same reason — the human
    looking at it IS the verification.

    local_library alternates carry a plain filesystem path as download_url
    (see app.stock.local_library.search) — checked explicitly here rather
    than letting a missing file surface as whatever ffmpeg's stderr happens
    to say (see downloader.download_video_trimmed's RuntimeError for a local
    source), so the editor gets one clear, user-facing reason instead of an
    opaque "ffmpeg trim failed" 500. local_library.search() itself already
    filters out stale entries going forward (root-cause fix); this covers
    alternates that were persisted to timeline.json before that filter
    existed, or a file removed between search and the user clicking it."""
    if alt.source == "local_library" and not Path(alt.download_url).exists():
        raise ValueError(
            "this alternate is no longer available — its source file was removed from the library "
            f"({alt.source}:{alt.source_id})"
        )
    hit = StockHit(
        source=alt.source, source_id=alt.source_id, download_url=alt.download_url,
        width=alt.width, height=alt.height, duration=alt.duration,
        media_type=alt.media_type, text=alt.text,
    )
    scored = ScoredAsset(hit=hit, score=alt.score)
    asset = await _download_asset(scored, clip_dir, f"clip{clip_number:03d}", 0, target_duration)

    _duration, resolution = await asyncio.to_thread(probe, asset.path)
    width, height = _parse_resolution(resolution)
    if not _is_exact_16_9(width, height):
        if _is_too_low_res(width, height):
            raise RuntimeError(f"alternate {alt.source}:{alt.source_id} is {width}x{height} — too low-res to use")
        normalized = await asyncio.to_thread(normalize_to_16_9, Path(asset.path), asset.media_type, width, height)
        if not normalized:
            raise RuntimeError(
                f"alternate {alt.source}:{alt.source_id} ({width}x{height}) failed to normalize to 16:9"
            )
        asset = asset.model_copy(update={"width": _WIDTH, "height": _HEIGHT})
    return asset


def _reject_if_asset_used_elsewhere(
    timeline: list[TimelineEntry], clip_number: int, source: str, source_id: str,
) -> None:
    """Guards the editor's alternate-swap path (see rerender_single_clip)
    against the exact same "clip used twice" bug the automatic resolve loop
    already prevents via used_assets — but there is no persisted used_assets
    dict for a single-clip edit against an already-generated project, so the
    check has to be done fresh here, against the CURRENT timeline.json, every
    time. Without this, swapping clip A to an alternate that some other clip
    B in the same project already has as its accepted asset would silently
    put the same real footage on screen twice, with no dedup mechanism ever
    seeing it happen — the automatic-run bug this whole priority is about,
    just reachable via the editor instead of a fresh generate."""
    for other in timeline:
        if other.clip_number == clip_number or other.asset_metadata is None:
            continue
        if (other.asset_metadata.source, other.asset_metadata.source_id) == (source, source_id):
            raise ValueError(
                f"this alternate ({source}:{source_id}) is already used by clip "
                f"{other.clip_number} in this project — pick a different alternate to avoid "
                "using the same footage twice"
            )


async def rerender_single_clip(
    project_name: str,
    clip_number: int,
    *,
    alternate_index: int | None = None,
    ai_regenerate: bool = False,
    upload_path: str | None = None,
) -> TimelineEntry:
    """Tier-1 fast edit (see the editor's PATCH /project/{project_name}/clip/
    {clip_number}): replaces ONE clip's asset and re-renders only that clip's
    segment + re-concats final_video.mp4 — no search/download/CLIP for any
    other clip, and no Remotion/audio-mux (that's Tier 2, see
    generate_full_video). Exactly one of alternate_index/ai_regenerate/
    upload_path must be given."""
    chosen = sum(x is not None for x in (alternate_index, upload_path)) + int(ai_regenerate)
    if chosen != 1:
        raise ValueError("exactly one of alternate_index, ai_regenerate, upload_path must be given")

    project_dir = settings.documentary_projects_dir / project_name
    timeline = _load_timeline(project_dir)
    entry_index = next((i for i, e in enumerate(timeline) if e.clip_number == clip_number), None)
    if entry_index is None:
        raise ValueError(f"clip {clip_number} not found in project {project_name!r}")
    entry = timeline[entry_index]
    clip_dir = project_dir / "clips" / f"clip_{clip_number:03d}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    target_duration = _duration_seconds(entry.start, entry.end)

    if alternate_index is not None:
        if not (0 <= alternate_index < len(entry.alternates)):
            raise ValueError(
                f"alternate_index {alternate_index} out of range — clip {clip_number} has "
                f"{len(entry.alternates)} alternate(s)"
            )
        chosen_alt = entry.alternates[alternate_index]
        _reject_if_asset_used_elsewhere(timeline, clip_number, chosen_alt.source, chosen_alt.source_id)
        asset = await _materialize_alternate_asset(chosen_alt, clip_dir, clip_number, target_duration)
        note = f"swapped to alternate {alternate_index} ({asset.source}:{asset.source_id}) via editor"
    elif ai_regenerate:
        meta = _load_project_meta(project_dir)
        table_clip = next((c for c in meta.table if c.clip_number == clip_number), None)
        if table_clip is None:
            raise ValueError(
                f"clip {clip_number} has no table entry in project_meta.json — cannot rebuild its generation prompt"
            )
        generated, failure_reason = await _try_ai_generation_fallback(table_clip, clip_dir, None, meta.content_niche, {})
        if generated is None:
            raise RuntimeError(f"AI regeneration failed: {failure_reason}")
        asset = generated.asset_metadata
        note = "regenerated via AI (OpenAI) from the editor"
    else:
        upload = Path(upload_path)
        ext = upload.suffix.lower() or ".mp4"
        media_type: Literal["video", "image"] = "image" if ext in {".jpg", ".jpeg", ".png", ".webp"} else "video"
        dest = clip_dir / f"clip{clip_number:03d}_uploaded{ext}"
        shutil.copy(upload, dest)
        _duration, resolution = await asyncio.to_thread(probe, str(dest))
        width, height = _parse_resolution(resolution)
        if not _is_exact_16_9(width, height) and not _is_too_low_res(width, height):
            if await asyncio.to_thread(normalize_to_16_9, dest, media_type, width, height):
                width, height = _WIDTH, _HEIGHT
        asset = AssetInfo(
            path=str(dest), source="upload", source_id=dest.name, media_type=media_type,
            width=width, height=height, duration=_duration if media_type == "video" else 0,
            score=None, selected_text="(uploaded by editor)",
        )
        note = "manually uploaded via editor"

    timeline[entry_index] = entry.model_copy(update={
        "asset_path": asset.path, "asset_metadata": asset, "placeholder": False, "note": note,
        "generation_attempted": entry.generation_attempted or ai_regenerate,
    })

    _final_path, updated_timeline = await asyncio.to_thread(
        rerender_single_clip_segment, project_dir, timeline, clip_number,
    )
    _write_timeline(project_dir, updated_timeline)
    return next(e for e in updated_timeline if e.clip_number == clip_number)


async def generate_full_video(project_name: str) -> DocumentaryResult:
    """Tier-2 full pipeline render (see the editor's "Generate Full Video"
    button): re-runs assemble_video -> render_final_video -> audio mux over
    the CURRENT (possibly Tier-1-edited) timeline.json, reusing the exact
    same finalization path run() uses — never re-runs script->table
    generation, search, download, or CLIP verification."""
    project_dir = settings.documentary_projects_dir / project_name
    timeline = _load_timeline(project_dir)
    meta = _load_project_meta(project_dir)
    progress.set_context(project_name)
    whisper_words = [WordTiming(**w) for w in meta.whisper_words] if meta.whisper_words else None
    return await _finalize_and_render(
        project_dir, meta.table, timeline, meta.script, meta.audio_path, whisper_words,
        meta.transcript_alignment_warning, meta.footage_availability, meta.content_niche,
    )


async def _finalize_and_render(
    project_dir: Path,
    table: list[TimelineClip],
    timeline: list[TimelineEntry],
    script: str,
    audio_path: str | None,
    whisper_words: list[WordTiming] | None,
    transcript_alignment_warning: str | None,
    footage_availability: AvailabilityReport,
    content_niche: str | list[str],
) -> DocumentaryResult:
    """Shared tail of run() and generate_full_video(): assemble_video ->
    Remotion render -> audio mux -> write timeline.json/project_meta.json/
    reports. Factored out so a later editor-triggered "Generate Full Video"
    reuses this EXACT path instead of a second copy of it."""
    progress.update("timeline", "Concatenating rendered segments into final_video.mp4")

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
        progress.update("timeline", "Deciding caption/transition style (LLM)")
        style_decision = await asyncio.to_thread(decide_style, script)

        cues = build_cues(timeline)
        if cues:
            srt_path = project_dir / "captions.srt"
            srt_path.write_text(render_srt(cues), encoding="utf-8")
            subtitles_path = str(srt_path)

        try:
            progress.update("timeline", "Rendering captions/transitions (Remotion)")
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
            progress.update("timeline", "Muxing narration audio into final render")
            narrated_path = await asyncio.to_thread(
                mux_narration_audio, Path(subtitled_video_path), audio_path,
                project_dir / final_video_filename(project_dir),
            )
            final_video_with_narration_path = str(narrated_path)
        except AudioMuxError as exc:
            logger.error("documentary_pipeline: audio mux failed: %s", exc)
            audio_mux_error = str(exc)

    (project_dir / "timeline_table.md").write_text(render_table_markdown(table), encoding="utf-8")
    _write_timeline(project_dir, timeline)
    (project_dir / "keyword_match_report.md").write_text(
        render_keyword_match_report(table, timeline), encoding="utf-8"
    )

    # A stable, project-relative copy of the narration audio for the editor's
    # preview playback (see app/main.py's GET /project route) — audio_path
    # itself may point into scratch upload space with no survival guarantee,
    # and isn't servable via the /project-files mount either way (outside
    # documentary_projects_dir). The mux call above intentionally still uses
    # the original audio_path unchanged — this is a separate, additive
    # persistence step, not a substitute for it.
    stable_audio_path: str | None = None
    if audio_path:
        try:
            src = Path(audio_path)
            if src.exists():
                dest = project_dir / f"narration{src.suffix or '.mp3'}"
                if not dest.exists():
                    shutil.copy(src, dest)
                stable_audio_path = str(dest)
        except OSError as exc:
            logger.warning("documentary_pipeline: failed to persist a stable narration audio copy: %s", exc)

    _write_project_meta(project_dir, ProjectMeta(
        script=script, content_niche=content_niche, audio_path=stable_audio_path,
        whisper_words=[{"text": w.text, "start": w.start, "end": w.end} for w in whisper_words] if whisper_words else [],
        transcript_alignment_warning=transcript_alignment_warning,
        table=table, footage_availability=footage_availability,
    ))

    return DocumentaryResult(
        project_dir=str(project_dir), table=table, timeline=timeline,
        final_video_path=final_video_path, assembly_error=assembly_error,
        subtitles_path=subtitles_path, subtitled_video_path=subtitled_video_path,
        subtitle_error=subtitle_error,
        style_decision=style_decision.model_dump() if style_decision else None,
        transcript_alignment_warning=transcript_alignment_warning,
        final_video_with_narration_path=final_video_with_narration_path,
        audio_mux_error=audio_mux_error,
        footage_availability=footage_availability,
    )


@dataclass
class _ResolvedProject:
    project_dir: Path
    table: list[TimelineClip]
    timeline: list[TimelineEntry]
    plan: DocumentaryPlan
    script: str


async def _resolve_all_clips(
    script: str,
    project_name: str | None,
    allow_duplicates: bool,
    max_downloads: int | None,
    script_path: str | None,
    audio_path: str | None,
    content_niche: str | list[str],
    enable_ai_generation: bool = True,
    enable_local_library: bool = True,
    enable_stock_providers: bool = True,
) -> _ResolvedProject:
    """Shared prefix of run() and generate_and_edit(): table generation +
    per-clip search/download/CLIP-verify (stages 1-5 of the pipeline strip).
    No assembly/render/mux — callers decide how much of the finalization
    tail (see _finalize_and_render) they need."""
    script = resolve_script_text(script, script_path)

    project_name = resolve_project_name(project_name)
    progress.set_context(project_name)
    project_dir = settings.documentary_projects_dir / project_name
    clips_root = project_dir / "clips"
    clips_root.mkdir(parents=True, exist_ok=True)

    progress.update("input", "Received script")

    plan = await plan_documentary(
        script, project_name, audio_path, content_niche,
        enable_local_library=enable_local_library, enable_stock_providers=enable_stock_providers,
    )
    table = plan.table
    (project_dir / "footage_availability_report.json").write_text(
        plan.footage_availability.model_dump_json(indent=2), encoding="utf-8"
    )

    limit = max_downloads if max_downloads is not None else settings.documentary_max_downloads_per_project
    used_assets: dict[tuple[str, str], int] = {}
    media_type_counts: dict[str, int] = {"image": 0, "video": 0}
    downloads_so_far = 0

    # Sequential per clip: dedup bookkeeping (used_assets), the running
    # media-type mix (media_type_counts — see _media_type_bias), and the
    # download-limit counter are shared mutable state, so clips can't safely
    # search concurrently.
    # ponytail: revisit with a lock-protected shared set if project size makes
    # sequential search a bottleneck.
    total_clips = len(table)
    timeline: list[TimelineEntry] = []
    for clip in table:
        progress.set_context(project_name, clip.clip_number, total_clips)
        entry, downloads_so_far = await _resolve_clip(
            clip, clips_root, used_assets, allow_duplicates, downloads_so_far, limit, content_niche,
            media_type_counts, enable_ai_generation=enable_ai_generation,
            enable_local_library=enable_local_library, enable_stock_providers=enable_stock_providers,
        )
        timeline.append(entry)

    progress.set_context(project_name)
    return _ResolvedProject(project_dir=project_dir, table=table, timeline=timeline, plan=plan, script=script)


async def run(
    script: str,
    project_name: str | None = None,
    allow_duplicates: bool = False,
    max_downloads: int | None = None,
    script_path: str | None = None,
    audio_path: str | None = None,
    # Content-category niche lock (see app.niches) — passed through from the
    # request model's Category selector, same as script_path/audio_path above.
    content_niche: str | list[str] = DEFAULT_NICHE,
    enable_ai_generation: bool = True,
    enable_local_library: bool = True,
    enable_stock_providers: bool = True,
) -> DocumentaryResult:
    resolved = await _resolve_all_clips(
        script, project_name, allow_duplicates, max_downloads, script_path, audio_path, content_niche,
        enable_ai_generation, enable_local_library, enable_stock_providers,
    )
    return await _finalize_and_render(
        resolved.project_dir, resolved.table, resolved.timeline, resolved.script, audio_path,
        resolved.plan.whisper_words, resolved.plan.transcript_alignment_warning,
        resolved.plan.footage_availability, content_niche,
    )


async def generate_and_edit(
    script: str,
    project_name: str | None = None,
    allow_duplicates: bool = False,
    max_downloads: int | None = None,
    script_path: str | None = None,
    audio_path: str | None = None,
    content_niche: str | list[str] = DEFAULT_NICHE,
    enable_ai_generation: bool = True,
    enable_local_library: bool = True,
    enable_stock_providers: bool = True,
) -> DocumentaryResult:
    """"Generate & Edit" mode (see app/main.py's /generate-documentary-timeline/edit):
    identical to run() through table generation + search/download/CLIP-verify,
    plus render_all_segments' per-clip segment render (rendered_clip_path —
    the editor's preview playback and single-clip re-render both depend on
    this existing) — but stops there. No final ffmpeg concat/final_video.mp4,
    no Remotion composite, no caption burn-in, no audio mux: those stay
    available on demand via generate_full_video() once the user is done
    editing in the browser, same as they already are for any Tier-1-edited
    project. Deliberately calls render_all_segments (the cheap per-row half
    of assemble_video), not assemble_video itself — the concat pass produces
    a final_video.mp4 that generate_full_video() immediately re-renders from
    scratch anyway, so running it here was pure wasted ffmpeg time; worse,
    a genuinely broken table (bad contiguity/orientation on any one row) or
    even just a final duration/resolution mismatch used to make
    assemble_video raise AFTER already rendering every row, discarding every
    already-rendered segment along with it and leaving EVERY clip (not just
    the offending one) with no rendered_clip_path — breaking preview/editing
    for the whole project on a failure that only actually matters for the
    final concat, not per-clip editing.

    Still writes timeline.json/project_meta.json (mirroring the tail of
    _finalize_and_render, minus the render/mux-only fields) so
    GET /project/{project_name} — and therefore the editor — can load the
    result the moment this returns."""
    resolved = await _resolve_all_clips(
        script, project_name, allow_duplicates, max_downloads, script_path, audio_path, content_niche,
        enable_ai_generation, enable_local_library, enable_stock_providers,
    )
    project_dir, table, timeline, plan, script = (
        resolved.project_dir, resolved.table, resolved.timeline, resolved.plan, resolved.script,
    )

    progress.update("timeline", "Rendering per-clip segments")
    assembly_error: str | None = None
    try:
        timeline = await asyncio.to_thread(render_all_segments, project_dir, timeline)
    except AssemblyValidationError as exc:
        logger.error("documentary_pipeline: per-clip segment render failed: %s", exc)
        assembly_error = str(exc)

    (project_dir / "timeline_table.md").write_text(render_table_markdown(table), encoding="utf-8")
    _write_timeline(project_dir, timeline)
    (project_dir / "keyword_match_report.md").write_text(
        render_keyword_match_report(table, timeline), encoding="utf-8"
    )

    # Same stable-copy persistence _finalize_and_render does — the editor's
    # narration preview track reads this, not the original (possibly
    # scratch-space) audio_path.
    stable_audio_path: str | None = None
    if audio_path:
        try:
            src = Path(audio_path)
            if src.exists():
                dest = project_dir / f"narration{src.suffix or '.mp3'}"
                if not dest.exists():
                    shutil.copy(src, dest)
                stable_audio_path = str(dest)
        except OSError as exc:
            logger.warning("documentary_pipeline: failed to persist a stable narration audio copy: %s", exc)

    _write_project_meta(project_dir, ProjectMeta(
        script=script, content_niche=content_niche, audio_path=stable_audio_path,
        whisper_words=[{"text": w.text, "start": w.start, "end": w.end} for w in plan.whisper_words] if plan.whisper_words else [],
        transcript_alignment_warning=plan.transcript_alignment_warning,
        table=table, footage_availability=plan.footage_availability,
    ))

    return DocumentaryResult(
        project_dir=str(project_dir), table=table, timeline=timeline,
        # final_video_path is intentionally always None here — Mode A never
        # runs the concat pass that produces it (see the docstring above);
        # it only exists once generate_full_video() has actually run.
        final_video_path=None, assembly_error=assembly_error,
        transcript_alignment_warning=plan.transcript_alignment_warning,
        footage_availability=plan.footage_availability,
    )
