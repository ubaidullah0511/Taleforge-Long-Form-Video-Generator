"""Scan of a local_library niche for near-duplicate clips — compares every
caption pairwise via the same embedding model app.scoring already uses for
search ranking (app.embeddings), since captions describing the same scene in
different words (e.g. "communal area" vs "common area") share little literal
text overlap but embed close together.

Default mode is report-only (near_duplicates_report.md + a removal preview,
duplicates_removal_plan.md — nothing on disk is touched). Pass --apply to
actually act on the plan: for each near-duplicate cluster, the largest file
(by byte size) is kept and every other file in that cluster is MOVED (never
deleted) to clips/local/<niche>/_duplicates_removed/, with index.json's
entry for each moved file removed (the kept file's entry is untouched).

Usage:
  python -m app.stock.find_near_duplicates --niche prison                  # report + plan preview, no changes
  python -m app.stock.find_near_duplicates --niche prison --apply          # actually quarantine duplicates
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# Suggested by the original task, but calibration against the real prison
# library (1210 clips) showed 0.75 flags ~55k of ~732k pairs — most of that
# is just shared domain vocabulary ("prison", "inmate", "correctional
# facility"), not real duplicates. Kept as the report-only default since
# it's still a legitimate (if noisy) cut and --threshold overrides it; the
# --apply path defaults to APPLY_THRESHOLD instead (see main()), which is
# the value manual review of that same run actually confirmed as meaningful.
DEFAULT_THRESHOLD = 0.75
APPLY_THRESHOLD = 0.9

QUARANTINE_DIRNAME = "_duplicates_removed"


def _load_index(niche: str) -> dict:
    index_path = settings.local_clips_dir / niche / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"no index.json for niche {niche!r} at {index_path}")
    return json.loads(index_path.read_text(encoding="utf-8"))


def _embed_captions(captions: list[str]) -> np.ndarray:
    """Split out from find_near_duplicates so tests can patch this one call
    instead of loading the real (large, network-fetched) embedding model."""
    from app.embeddings import _model
    return np.array(_model().encode(captions, normalize_embeddings=True, show_progress_bar=False))


def find_near_duplicates(niche: str, threshold: float = DEFAULT_THRESHOLD) -> list[tuple[float, str, str, str, str]]:
    """Returns (score, filename_a, caption_a, filename_b, caption_b) tuples,
    highest similarity first, for every pair scoring >= threshold."""
    index = _load_index(niche)
    names = list(index.keys())
    if len(names) < 2:
        return []
    captions = [index[n].get("caption", "") for n in names]

    embeddings = _embed_captions(captions)
    # normalize_embeddings=True (same flag app.embeddings.embed uses) makes
    # every row unit-length, so the dot product IS the cosine similarity —
    # one matmul scores every pair at once instead of n^2 individual calls.
    similarity = embeddings @ embeddings.T

    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            score = float(similarity[i, j])
            if score >= threshold:
                pairs.append((score, names[i], captions[i], names[j], captions[j]))
    pairs.sort(key=lambda p: p[0], reverse=True)
    return pairs


def render_report(niche: str, threshold: float, pairs: list[tuple[float, str, str, str, str]]) -> str:
    lines = [
        f"# Near-Duplicate Report — {niche}",
        "",
        f"Similarity threshold: {threshold}",
        f"Flagged pairs: {len(pairs)}",
        "",
        "Report only — nothing was deleted or renamed. Review manually.",
        "",
    ]
    for score, name_a, caption_a, name_b, caption_b in pairs:
        lines.append(f"## {score:.3f} — `{name_a}` vs `{name_b}`")
        lines.append("")
        lines.append(f"- **{name_a}**: {caption_a}")
        lines.append(f"- **{name_b}**: {caption_b}")
        lines.append("")
    return "\n".join(lines)


def _cluster(pairs: list[tuple[float, str, str, str, str]]) -> list[set[str]]:
    """Union-find over flagged pairs -> connected components. A chain of
    near-duplicates (A~B, B~C, but A~C never directly flagged) must resolve
    to one shared keep/remove decision instead of two pairs independently
    picking possibly-inconsistent keepers."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for _, name_a, _, name_b, _ in pairs:
        union(name_a, name_b)

    clusters: dict[str, set[str]] = {}
    for name in parent:
        clusters.setdefault(find(name), set()).add(name)
    return list(clusters.values())


def plan_removals(niche: str, pairs: list[tuple[float, str, str, str, str]]) -> list[dict]:
    """One entry per file slated for removal: {keep, keep_size, remove,
    remove_size}. Per cluster of near-duplicates, the single largest file on
    disk (by byte size, per the task) is kept; every other file in that
    cluster is slated for removal. A file missing from disk (index.json
    stale/out of sync) is logged and skipped rather than crashing the run."""
    niche_dir = settings.local_clips_dir / niche
    plan = []
    for cluster in _cluster(pairs):
        sized = []
        for name in cluster:
            path = niche_dir / name
            if not path.exists():
                logger.warning("find_near_duplicates: %s listed in a near-duplicate pair but missing on disk — skipping", name)
                continue
            sized.append((name, path.stat().st_size))
        if len(sized) < 2:
            continue
        sized.sort(key=lambda t: t[1], reverse=True)
        keep_name, keep_size = sized[0]
        for name, size in sized[1:]:
            plan.append({"keep": keep_name, "keep_size": keep_size, "remove": name, "remove_size": size})
    plan.sort(key=lambda p: p["remove"])
    return plan


def render_removal_plan(niche: str, threshold: float, plan: list[dict], applied: bool) -> str:
    status = "APPLIED — files below were moved to quarantine" if applied else "DRY RUN — nothing moved. Pass --apply to execute."
    lines = [
        f"# Duplicate Removal Plan — {niche}",
        "",
        f"Similarity threshold: {threshold}",
        f"Files slated for removal: {len(plan)}",
        "",
        f"**{status}**",
        "",
    ]
    for item in plan:
        lines.append(
            f"- keep `{item['keep']}` ({item['keep_size']} bytes) — "
            f"remove `{item['remove']}` ({item['remove_size']} bytes, smaller)"
        )
    return "\n".join(lines) + "\n"


def apply_removals(niche: str, plan: list[dict]) -> None:
    """Moves every plan['remove'] file to clips/local/<niche>/_duplicates_removed/
    (never deletes) and drops its index.json entry. The kept file's entry
    and file are never touched. Re-reads/writes index.json directly (rather
    than reusing an in-memory copy from find_near_duplicates) so this always
    acts on the current on-disk state."""
    niche_dir = settings.local_clips_dir / niche
    quarantine_dir = niche_dir / QUARANTINE_DIRNAME
    quarantine_dir.mkdir(exist_ok=True)
    index_path = niche_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))

    for item in plan:
        remove_name = item["remove"]
        src = niche_dir / remove_name
        if not src.exists():
            logger.warning("find_near_duplicates: %s already missing, skipping move", remove_name)
            continue

        dest = quarantine_dir / remove_name
        counter = 2
        while dest.exists():  # don't clobber a same-named file quarantined by a prior run
            dest = quarantine_dir / f"{Path(remove_name).stem}_{counter}{Path(remove_name).suffix}"
            counter += 1
        src.rename(dest)
        index.pop(remove_name, None)
        logger.info(
            "find_near_duplicates: kept %s (%d bytes) — moved %s (%d bytes) to %s "
            "(near-duplicate, smaller of the two)",
            item["keep"], item["keep_size"], remove_name, item["remove_size"], dest,
        )

    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Find (and optionally quarantine) near-duplicate local_library clips.")
    parser.add_argument("--niche", required=True)
    parser.add_argument("--threshold", type=float, default=None, help=f"default {DEFAULT_THRESHOLD}, or {APPLY_THRESHOLD} with --apply")
    parser.add_argument("--apply", action="store_true", help="Move duplicates to quarantine (default: report only)")
    args = parser.parse_args()
    threshold = args.threshold if args.threshold is not None else (APPLY_THRESHOLD if args.apply else DEFAULT_THRESHOLD)

    try:
        pairs = find_near_duplicates(args.niche, threshold)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    niche_dir = settings.local_clips_dir / args.niche
    (niche_dir / "near_duplicates_report.md").write_text(render_report(args.niche, threshold, pairs), encoding="utf-8")

    plan = plan_removals(args.niche, pairs)
    if args.apply:
        apply_removals(args.niche, plan)
    (niche_dir / "duplicates_removal_plan.md").write_text(
        render_removal_plan(args.niche, threshold, plan, applied=args.apply), encoding="utf-8"
    )

    print(f"Flagged {len(pairs)} near-duplicate pair(s) (threshold {threshold}) -> {niche_dir / 'near_duplicates_report.md'}")
    if args.apply:
        print(f"Moved {len(plan)} file(s) to {niche_dir / QUARANTINE_DIRNAME} -> {niche_dir / 'duplicates_removal_plan.md'}")
    else:
        print(f"DRY RUN: {len(plan)} file(s) would be moved. Re-run with --apply to execute. See {niche_dir / 'duplicates_removal_plan.md'}")


if __name__ == "__main__":
    main()
