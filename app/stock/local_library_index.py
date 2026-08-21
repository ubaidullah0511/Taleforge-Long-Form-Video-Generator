import argparse
import hashlib
import json
import logging
import re
import time
from pathlib import Path

from openai import RateLimitError

from app.clip_ingest import probe
from app.config import settings
from app.llm_client import generate_json_from_video
from app.stock.find_near_duplicates import QUARANTINE_DIRNAME

logger = logging.getLogger(__name__)

_VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv")

# Own prompt rather than clip_ingest.ANALYSIS_PROMPT: same "detailed visual
# description" instruction (caption content/quality unchanged), plus a
# "slug" field ANALYSIS_PROMPT has no use for elsewhere (it feeds
# ClipRecord/vector-search ingestion, not filenames). Still reuses the same
# underlying vision-LLM call (generate_json_from_video) — only the prompt
# text passed into it differs, so this isn't duplicating the captioning
# machinery, just tailoring the schema it's asked to return.
_CAPTION_AND_SLUG_PROMPT = """Analyze this video clip for use as B-roll footage. Return \
a JSON object with keys:
"caption": a detailed visual description of what is shown — objects, actions, setting, mood.
"slug": a short filename slug (3-6 words, lowercase, underscore-separated) capturing the \
MOST SPECIFIC/DISTINCTIVE content in the clip. No filler words like "the video clip", \
"features", or "a person" — start directly with the distinctive subject/action. \
Example: for a bodybuilder flexing in a gym, use "muscular_bodybuilder_flexing_gym", \
not "the_video_features_a_man"."""

# generate_json_from_video's own rate-limit retries (see llm_client._throttle /
# _call_with_rate_limit_retry) only pace REQUEST count, not tokens — a batch
# of many back-to-back 3-image vision calls empirically exhausts the account's
# tokens-per-minute cap well before that request-count throttle kicks in, and
# llm_client's short internal retries aren't enough to outlast a TPM window.
# One extra retry here, after a wait long enough to clear that rolling window,
# recovers the file instead of permanently skipping it.
# ponytail: flat backoff, not adaptive to the API's actual retry-after value —
# tune down/adaptive if this still leaves files skipped in practice.
_RATE_LIMIT_RETRIES = 2
_RATE_LIMIT_BACKOFF_SECONDS = 65

# Niches with a stop requested via request_stop() (see the "Sync Library"
# stop button in main.py) — checked by index_niche() between files, never
# mid-caption-call. A plain module-level set is enough here for the same
# reason app.progress uses a plain dict: single-process local tool.
_stop_requested: set[str] = set()


def request_stop(niche: str) -> None:
    """Marks niche's in-progress index_niche() run to stop before its NEXT
    file — the file currently being captioned always finishes, since
    index_niche only checks this at the top of each loop iteration."""
    _stop_requested.add(niche)


def _caption(path: Path) -> dict:
    for attempt in range(_RATE_LIMIT_RETRIES + 1):
        try:
            return generate_json_from_video(_CAPTION_AND_SLUG_PROMPT, str(path), settings.llm_vision_model)
        except RateLimitError:
            if attempt == _RATE_LIMIT_RETRIES:
                raise
            logger.warning(
                "local_library_index: tokens-per-minute cap hit captioning %s — waiting %ds (retry %d/%d)",
                path.name, _RATE_LIMIT_BACKOFF_SECONDS, attempt + 1, _RATE_LIMIT_RETRIES,
            )
            time.sleep(_RATE_LIMIT_BACKOFF_SECONDS)


def _slugify_caption(text: str, max_words: int = 6) -> str:
    """Fallback only — used when GPT doesn't return a usable "slug" (see
    _CAPTION_AND_SLUG_PROMPT). Also doubles as a sanitizer for GPT's own
    slug (word-splits on underscores too, not just whitespace/punctuation),
    in case it doesn't perfectly follow the requested format."""
    cleaned = re.sub(r"[^a-zA-Z0-9\s_]", "", text).lower()
    words = [w for w in re.split(r"[\s_]+", cleaned) if w][:max_words]
    return "_".join(words) if words else "clip"


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _rename_with_unique_suffix(old_path: Path, new_stem: str) -> Path:
    """Renames old_path to new_stem + original extension, appending _2, _3,
    etc. if a file with that name already exists — never silently overwrites."""
    candidate = old_path.with_name(f"{new_stem}{old_path.suffix}")
    counter = 2
    while candidate.exists() and candidate != old_path:
        candidate = old_path.with_name(f"{new_stem}_{counter}{old_path.suffix}")
        counter += 1
    old_path.rename(candidate)
    return candidate


def index_niche(niche: str, stats: dict | None = None) -> dict:
    """Scans local_clips/<niche>/ RECURSIVELY (e.g. clips/local/tennis/Alex
    Eala/*.mp4 is found exactly like a flat clips/local/tennis/*.mp4 file
    would be — subfolders are just an organizing convenience, e.g. one per
    player/subject) for video files, captions each NEW one via the same
    vision-LLM call machinery app.clip_ingest uses to ingest real footage
    (generate_json_from_video: samples start/middle/end frames + one
    JSON-schema vision call — see _CAPTION_AND_SLUG_PROMPT for the schema
    used here), renames it on disk (in place, same subfolder) to a clean
    filename slug that GPT generates directly alongside the caption (for
    human browsing — index.json's keys are the real source of truth for
    search matching either way), and persists a niche_dir-relative path
    (e.g. "Alex Eala/muscular_serve.mp4") -> {caption, width, height,
    duration, hash} to local_clips/<niche>/index.json — the relative path
    keeps clips from different subfolders that happen to get the same slug
    from colliding in the index. The find_near_duplicates.py quarantine
    folder (_duplicates_removed/) is skipped so files deliberately removed
    from the index by that tool don't get silently re-indexed here.
    Already-indexed files are skipped (neither re-captioned nor renamed). A
    new file whose content hash exactly matches an already-indexed file's
    hash is deleted outright instead of captioned again (see _file_hash) —
    only byte-identical duplicates are caught this way; existing entries
    indexed before this check existed have no "hash" field and are never
    retroactively touched. Written incrementally (after each file) so a
    crash partway through a large batch doesn't lose already-captioned work.

    stats, if given, is filled in place with {"indexed": int, "total_new":
    int, "stopped": bool} as the run progresses/finishes — used by the
    background sync-library task in main.py to report live progress and
    whether a request_stop() call cut the run short. Existing callers that
    don't pass it (including every test in test_local_library.py) see no
    behavior change; the return value is still just the index dict."""
    niche_dir = settings.local_clips_dir / niche
    niche_dir.mkdir(parents=True, exist_ok=True)
    index_path = niche_dir / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    hashes_seen = {entry["hash"]: name for name, entry in index.items() if "hash" in entry}

    files = sorted(
        p for p in niche_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTS
        and QUARANTINE_DIRNAME not in p.relative_to(niche_dir).parts
    )
    new_files = [p for p in files if p.relative_to(niche_dir).as_posix() not in index]
    logger.info("local_library_index: %d file(s) in %s, %d new", len(files), niche_dir, len(new_files))

    _stop_requested.discard(niche)  # clear any stale flag left by a prior run
    if stats is not None:
        stats.update(indexed=0, total_new=len(new_files), stopped=False)

    indexed_count = 0
    duplicate_count = 0
    failed_count = 0

    for path in new_files:
        if niche in _stop_requested:
            logger.info(
                "local_library_index: stop requested for %s — halting before next file (%d/%d indexed so far)",
                niche, indexed_count, len(new_files),
            )
            _stop_requested.discard(niche)
            if stats is not None:
                stats["stopped"] = True
            break
        try:
            file_hash = _file_hash(path)
            duplicate_of = hashes_seen.get(file_hash)
            if duplicate_of is not None:
                logger.info(
                    "local_library_index: %s is a byte-identical duplicate of already-indexed %s — deleting",
                    path.name, duplicate_of,
                )
                path.unlink()
                duplicate_count += 1
                continue

            duration, resolution = probe(str(path))
            width_str, height_str = resolution.split("x")
            analysis = _caption(path)
            time.sleep(4.0)  # ponytail: flat pacing between calls, cuts how often the TPM backoff above triggers

            # A vision-LLM call returning valid JSON that's missing "caption"
            # isn't necessarily a format bug — response_format=json_object
            # only guarantees valid JSON syntax, not schema compliance, so a
            # content-policy soft-decline can come back as e.g. {"error":
            # "..."} instead of raising. Logging the full raw response (not
            # just the KeyError) is what actually distinguishes "the model
            # refused" from "the model sent a differently-shaped JSON object"
            # — a bare traceback shows neither.
            try:
                caption = analysis["caption"]
            except (KeyError, TypeError) as exc:
                logger.warning(
                    "local_library_index: no usable 'caption' key for %s (%s) — full raw response: %r — "
                    "skipping this file (not retried within this run)",
                    path.name, exc, analysis,
                )
                failed_count += 1
                continue

            slug = _slugify_caption(analysis.get("slug") or caption)

            renamed_path = _rename_with_unique_suffix(path, slug)
            rel_name = renamed_path.relative_to(niche_dir).as_posix()

            index[rel_name] = {
                "caption": caption,
                "width": int(width_str),
                "height": int(height_str),
                "duration": duration,
                "hash": file_hash,
            }
            hashes_seen[file_hash] = rel_name
            index_path.write_text(json.dumps(index, indent=2))
            logger.info("local_library_index: indexed %s -> %s (%s)", path.name, rel_name, caption)
            indexed_count += 1
            if stats is not None:
                stats["indexed"] = indexed_count
        except Exception:
            # ponytail: one corrupt/unreadable file shouldn't sink an 80-file
            # batch — skip it and keep going, same as asset_selection's
            # per-candidate try/except.
            logger.exception("local_library_index: failed to index %s — skipping", path.name)
            failed_count += 1

    logger.info(
        "local_library_index: run summary for %s — %d indexed, %d duplicate(s) deleted, %d failed/skipped "
        "(%d already indexed before this run)",
        niche, indexed_count, duplicate_count, failed_count, len(files) - len(new_files),
    )
    return index


def verify_niche(niche: str, dry_run: bool = False) -> dict:
    """Reconciles local_clips/<niche>/index.json against what's actually on
    disk — catches the recurring desync where a file gets deleted (manually,
    or by a near-duplicate cleanup pass) but its index.json entry doesn't,
    leaving the editor's alternates ("swap to this") pointing at a file that
    no longer exists (see _materialize_alternate_asset's graceful handling
    for the case where a stale entry gets clicked anyway before the next
    --verify pass catches it). Stale entries are removed from index.json by
    default (dry_run=False) — an index that lies about what's available is
    strictly worse than a smaller-but-accurate one, since local_library's own
    search() only ever offers what's indexed. dry_run=True reports without
    writing, for a look-before-you-clean pass."""
    niche_dir = settings.local_clips_dir / niche
    index_path = niche_dir / "index.json"
    if not index_path.exists():
        return {"niche": niche, "checked": 0, "stale": []}

    index = json.loads(index_path.read_text(encoding="utf-8"))
    stale = [filename for filename in index if not (niche_dir / filename).exists()]
    checked = len(index)

    if stale and not dry_run:
        for filename in stale:
            del index[filename]
        index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    return {"niche": niche, "checked": checked, "stale": stale, "removed": bool(stale and not dry_run)}


def verify_all_niches(dry_run: bool = False) -> list[dict]:
    root = settings.local_clips_dir
    if not root.exists():
        return []
    niches = sorted(p.name for p in root.iterdir() if p.is_dir())
    return [verify_niche(niche, dry_run=dry_run) for niche in niches]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index/verify a local_clips/<niche>/ folder for local_library search.")
    parser.add_argument("--niche", help="Limit to one niche folder. Omit with --verify to check every niche.")
    parser.add_argument(
        "--verify", action="store_true",
        help="Check index.json entries against files actually on disk instead of indexing new files.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="With --verify, report stale entries without removing them from index.json.",
    )
    args = parser.parse_args()

    if args.verify:
        results = [verify_niche(args.niche, dry_run=args.dry_run)] if args.niche else verify_all_niches(dry_run=args.dry_run)
        total_stale = 0
        for result in results:
            if result["checked"] == 0 and not result["stale"]:
                continue
            action = "would remove" if args.dry_run else "removed"
            print(f"{result['niche']}: {result['checked']} indexed, {len(result['stale'])} stale ({action})")
            for filename in result["stale"]:
                print(f"  - {filename}")
            total_stale += len(result["stale"])
        print(f"\n{total_stale} stale entr{'y' if total_stale == 1 else 'ies'} across {len(results)} niche(s).")
    else:
        if not args.niche:
            parser.error("--niche is required unless --verify is scanning all niches")
        result = index_niche(args.niche)
        print(f"{len(result)} clip(s) indexed total in local_clips/{args.niche}/index.json")
