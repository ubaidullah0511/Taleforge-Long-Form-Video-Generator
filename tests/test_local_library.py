"""Offline tests for app.stock.local_library / app.stock.local_library_index —
no network/API keys needed. Run with: pytest tests/test_local_library.py
"""
import asyncio
import json
from pathlib import Path
from unittest.mock import patch

from app.config import settings
from app.stock import local_library, local_library_index


def _write_index(niche_dir: Path, index: dict) -> None:
    niche_dir.mkdir(parents=True, exist_ok=True)
    (niche_dir / "index.json").write_text(json.dumps(index))


def test_search_returns_no_hits_without_index(tmp_path):
    with patch.object(settings, "clips_dir", tmp_path):
        hits = asyncio.run(local_library.search("deadlift", "bodybuilding"))
    assert hits == []


def test_search_ranks_by_caption_overlap_and_shapes_stockhit(tmp_path):
    niche_dir = tmp_path / "local" / "bodybuilding"
    _write_index(niche_dir, {
        "off_topic.mp4": {"caption": "a chef chopping vegetables in a kitchen", "width": 1920, "height": 1080, "duration": 5.0},
        "deadlift.mp4": {"caption": "a bodybuilder performing a heavy barbell deadlift in a gym", "width": 1280, "height": 720, "duration": 8.2},
    })
    (niche_dir / "off_topic.mp4").write_bytes(b"fake")
    (niche_dir / "deadlift.mp4").write_bytes(b"fake")
    with patch.object(settings, "clips_dir", tmp_path):
        hits = asyncio.run(local_library.search("person deadlifting barbell", "bodybuilding"))

    assert hits[0].source_id == "deadlift.mp4"
    assert hits[0].source == "local_library"
    assert hits[0].text == "a bodybuilder performing a heavy barbell deadlift in a gym"
    assert hits[0].width == 1280 and hits[0].height == 720 and hits[0].duration == 8.2
    assert hits[0].download_url == str((niche_dir / "deadlift.mp4").resolve())


def test_search_skips_entries_whose_file_is_missing_from_disk(tmp_path):
    """Root-cause fix for the recurring index/disk desync (files deleted
    after indexing, e.g. by the near-duplicate cleanup pass): search() must
    never hand back a StockHit for a file that isn't actually there, since
    that hit can become either a primary candidate or an editor alternate
    that later fails at download/materialize time with a confusing error."""
    niche_dir = tmp_path / "local" / "prison"
    niche_dir.mkdir(parents=True)
    (niche_dir / "present.mp4").write_bytes(b"fake")
    _write_index(niche_dir, {
        "present.mp4": {"caption": "a prison cell block", "width": 1920, "height": 1080, "duration": 5.0},
        "deleted.mp4": {"caption": "a prison yard aerial view", "width": 1920, "height": 1080, "duration": 5.0},
    })
    with patch.object(settings, "clips_dir", tmp_path):
        hits = asyncio.run(local_library.search("prison", "prison"))
    assert [h.source_id for h in hits] == ["present.mp4"]


def test_search_still_fills_per_page_when_top_ranked_entries_are_missing(tmp_path):
    """A missing entry must be skipped, not counted against per_page —
    otherwise a stale top-ranked entry silently shrinks the candidate pool
    instead of being backfilled by the next-best entry that IS on disk.
    a.mp4/b.mp4/c.mp4 all tie on overlap ratio (every query word present in
    each caption) — sorted() is stable, so a.mp4 (missing on disk) would
    rank first if it weren't filtered out, leaving only b.mp4/c.mp4 to fill
    per_page=2."""
    niche_dir = tmp_path / "local" / "prison"
    niche_dir.mkdir(parents=True)
    (niche_dir / "b.mp4").write_bytes(b"fake")
    (niche_dir / "c.mp4").write_bytes(b"fake")
    _write_index(niche_dir, {
        "a.mp4": {"caption": "prison yard", "width": 0, "height": 0, "duration": 0},  # missing on disk
        "b.mp4": {"caption": "prison yard", "width": 0, "height": 0, "duration": 0},
        "c.mp4": {"caption": "prison yard fence", "width": 0, "height": 0, "duration": 0},
    })
    with patch.object(settings, "clips_dir", tmp_path):
        hits = asyncio.run(local_library.search("prison yard", "prison", per_page=2))
    assert len(hits) == 2
    assert "a.mp4" not in {h.source_id for h in hits}


def test_search_respects_per_page(tmp_path):
    niche_dir = tmp_path / "local" / "bodybuilding"
    _write_index(niche_dir, {f"clip{i}.mp4": {"caption": "gym workout", "width": 0, "height": 0, "duration": 0} for i in range(5)})
    for i in range(5):
        (niche_dir / f"clip{i}.mp4").write_bytes(b"fake")
    with patch.object(settings, "clips_dir", tmp_path):
        hits = asyncio.run(local_library.search("gym", "bodybuilding", per_page=2))
    assert len(hits) == 2


def test_index_niche_skips_already_indexed(tmp_path):
    niche_dir = tmp_path / "local" / "bodybuilding"
    niche_dir.mkdir(parents=True)
    (niche_dir / "already.mp4").write_bytes(b"fake")
    _write_index(niche_dir, {"already.mp4": {"caption": "existing caption", "width": 100, "height": 100, "duration": 1.0}})

    with patch.object(settings, "clips_dir", tmp_path), \
         patch.object(local_library_index, "probe", return_value=(2.0, "640x360")), \
         patch.object(local_library_index, "generate_json_from_video") as mock_caption:
        result = local_library_index.index_niche("bodybuilding")

    mock_caption.assert_not_called()
    assert result["already.mp4"]["caption"] == "existing caption"
    # Do Not: never rename a file that's already indexed from a prior run.
    assert (niche_dir / "already.mp4").exists()


def test_index_niche_uses_gpt_slug_directly_not_caption_word_position(tmp_path):
    """The bug being fixed: deriving the slug from the caption's first N
    words grabs generic scene-setting filler ("the_video_clip_features_a_")
    instead of the actual distinctive content. GPT now returns the slug
    directly alongside the caption — this must be used as-is, not
    re-derived from caption text."""
    niche_dir = tmp_path / "local" / "bodybuilding"
    niche_dir.mkdir(parents=True)
    (niche_dir / "17.mp4").write_bytes(b"fake")

    with patch.object(settings, "clips_dir", tmp_path), \
         patch.object(local_library_index, "probe", return_value=(3.5, "1920x1080")), \
         patch.object(local_library_index, "generate_json_from_video", return_value={
             "caption": "The video clip features a muscular male bodybuilder displaying his physique against a dark, moody gym backdrop.",
             "slug": "muscular_bodybuilder_flexing_gym",
         }):
        result = local_library_index.index_niche("bodybuilding")

    assert not (niche_dir / "17.mp4").exists()
    assert (niche_dir / "muscular_bodybuilder_flexing_gym.mp4").exists()
    entry = result["muscular_bodybuilder_flexing_gym.mp4"]
    assert entry["caption"] == (
        "The video clip features a muscular male bodybuilder displaying his physique against a dark, moody gym backdrop."
    )
    assert entry["width"] == 1920 and entry["height"] == 1080 and entry["duration"] == 3.5
    assert entry["hash"] == local_library_index._file_hash(niche_dir / "muscular_bodybuilder_flexing_gym.mp4")
    on_disk = json.loads((niche_dir / "index.json").read_text())
    assert "muscular_bodybuilder_flexing_gym.mp4" in on_disk
    assert "17.mp4" not in on_disk


def test_index_niche_falls_back_to_derived_slug_when_gpt_omits_slug(tmp_path):
    niche_dir = tmp_path / "local" / "bodybuilding"
    niche_dir.mkdir(parents=True)
    (niche_dir / "17.mp4").write_bytes(b"fake")

    with patch.object(settings, "clips_dir", tmp_path), \
         patch.object(local_library_index, "probe", return_value=(3.5, "1920x1080")), \
         patch.object(local_library_index, "generate_json_from_video", return_value={"caption": "A man bench-pressing in a gym."}):
        result = local_library_index.index_niche("bodybuilding")

    assert (niche_dir / "a_man_benchpressing_in_a_gym.mp4").exists()
    assert "a_man_benchpressing_in_a_gym.mp4" in result


def test_index_niche_never_overwrites_on_slug_collision(tmp_path):
    """Two new files whose slugs collide must both survive on disk, with the
    second one getting a numeric suffix rather than clobbering the first."""
    niche_dir = tmp_path / "local" / "bodybuilding"
    niche_dir.mkdir(parents=True)
    (niche_dir / "a.mp4").write_bytes(b"fake-a")
    (niche_dir / "b.mp4").write_bytes(b"fake-b")

    with patch.object(settings, "clips_dir", tmp_path), \
         patch.object(local_library_index, "probe", return_value=(3.0, "1920x1080")), \
         patch.object(local_library_index, "generate_json_from_video", return_value={"caption": "gym workout", "slug": "gym_workout"}):
        result = local_library_index.index_niche("bodybuilding")

    assert (niche_dir / "gym_workout.mp4").read_bytes() == b"fake-a"
    assert (niche_dir / "gym_workout_2.mp4").read_bytes() == b"fake-b"
    assert set(result) == {"gym_workout.mp4", "gym_workout_2.mp4"}


def test_index_niche_deletes_byte_identical_duplicate_without_captioning(tmp_path):
    niche_dir = tmp_path / "local" / "bodybuilding"
    niche_dir.mkdir(parents=True)
    (niche_dir / "original.mp4").write_bytes(b"identical video content")
    (niche_dir / "duplicate.mp4").write_bytes(b"identical video content")

    with patch.object(settings, "clips_dir", tmp_path), \
         patch.object(local_library_index, "probe", return_value=(3.0, "1920x1080")), \
         patch.object(local_library_index, "generate_json_from_video", return_value={
             "caption": "a bodybuilder deadlifting", "slug": "bodybuilder_deadlifting",
         }) as mock_caption:
        result = local_library_index.index_niche("bodybuilding")

    # Exactly one entry: the duplicate was deleted, not captioned/indexed a second time.
    assert len(result) == 1
    mock_caption.assert_called_once()
    assert not (niche_dir / "duplicate.mp4").exists()
    assert (niche_dir / "bodybuilder_deadlifting.mp4").exists()
    assert "hash" in result["bodybuilder_deadlifting.mp4"]


def test_index_niche_does_not_delete_files_with_different_content(tmp_path):
    """Do Not: never delete anything that isn't byte-identical — same
    caption/slug response for both (a flaky/generic vision answer) must not
    be mistaken for a content match; only the hash decides."""
    niche_dir = tmp_path / "local" / "bodybuilding"
    niche_dir.mkdir(parents=True)
    (niche_dir / "a.mp4").write_bytes(b"video content A")
    (niche_dir / "b.mp4").write_bytes(b"video content B, not identical to A")

    with patch.object(settings, "clips_dir", tmp_path), \
         patch.object(local_library_index, "probe", return_value=(3.0, "1920x1080")), \
         patch.object(local_library_index, "generate_json_from_video", return_value={
             "caption": "gym workout", "slug": "gym_workout",
         }):
        result = local_library_index.index_niche("bodybuilding")

    assert len(result) == 2
    assert (niche_dir / "gym_workout.mp4").exists()
    assert (niche_dir / "gym_workout_2.mp4").exists()


def test_index_niche_duplicate_check_ignores_already_indexed_files_without_hash(tmp_path):
    """Do Not: no retroactive sweep — an already-indexed file predating this
    feature (no "hash" field) must never be deleted just because a new file
    happens to share its content; it's simply invisible to the dup check."""
    niche_dir = tmp_path / "local" / "bodybuilding"
    niche_dir.mkdir(parents=True)
    (niche_dir / "legacy.mp4").write_bytes(b"identical video content")
    (niche_dir / "new_upload.mp4").write_bytes(b"identical video content")
    _write_index(niche_dir, {
        "legacy.mp4": {"caption": "existing caption", "width": 100, "height": 100, "duration": 1.0},
    })

    with patch.object(settings, "clips_dir", tmp_path), \
         patch.object(local_library_index, "probe", return_value=(3.0, "1920x1080")), \
         patch.object(local_library_index, "generate_json_from_video", return_value={
             "caption": "a bodybuilder deadlifting", "slug": "bodybuilder_deadlifting",
         }) as mock_caption:
        result = local_library_index.index_niche("bodybuilding")

    mock_caption.assert_called_once()
    assert (niche_dir / "legacy.mp4").exists()
    assert (niche_dir / "bodybuilder_deadlifting.mp4").exists()
    assert len(result) == 2


def test_index_niche_missing_caption_key_is_skipped_not_crashed(tmp_path, caplog):
    """Bug report: a vision-LLM response missing "caption" (e.g. a content-
    policy soft-decline shaped as {"error": "..."} instead of raising)
    crashed the whole run with an uncaught KeyError. Must now log the full
    raw response and move on to the next file instead."""
    niche_dir = tmp_path / "local" / "prison"
    niche_dir.mkdir(parents=True)
    (niche_dir / "clip_0005.mp4").write_bytes(b"fake-bad")
    (niche_dir / "clip_0006.mp4").write_bytes(b"fake-good")

    responses = iter([
        {"error": "I can't assist with analyzing this content."},  # clip_0005: no "caption" key
        {"caption": "a prison cell block", "slug": "prison_cell_block"},  # clip_0006: fine
    ])

    with patch.object(settings, "clips_dir", tmp_path), \
         patch.object(local_library_index, "probe", return_value=(3.0, "1920x1080")), \
         patch.object(local_library_index, "generate_json_from_video", side_effect=lambda *a, **k: next(responses)), \
         caplog.at_level("WARNING", logger="app.stock.local_library_index"):
        result = local_library_index.index_niche("prison")

    # Bad file: not indexed, not renamed, but the run kept going and the
    # good file right after it was still indexed normally.
    assert "clip_0005.mp4" not in result
    assert (niche_dir / "clip_0005.mp4").exists()
    assert "prison_cell_block.mp4" in result
    assert (niche_dir / "prison_cell_block.mp4").exists()

    warning_text = " ".join(r.message for r in caplog.records if r.levelname == "WARNING")
    assert "clip_0005.mp4" in warning_text
    assert "I can't assist with analyzing this content." in warning_text


def test_index_niche_logs_run_summary_with_success_and_failure_counts(tmp_path, caplog):
    niche_dir = tmp_path / "local" / "prison"
    niche_dir.mkdir(parents=True)
    (niche_dir / "bad.mp4").write_bytes(b"fake-bad")
    (niche_dir / "good.mp4").write_bytes(b"fake-good")

    responses = iter([
        {"message": "unable to process"},
        {"caption": "a prison yard", "slug": "prison_yard"},
    ])

    with patch.object(settings, "clips_dir", tmp_path), \
         patch.object(local_library_index, "probe", return_value=(3.0, "1920x1080")), \
         patch.object(local_library_index, "generate_json_from_video", side_effect=lambda *a, **k: next(responses)), \
         caplog.at_level("INFO", logger="app.stock.local_library_index"):
        local_library_index.index_niche("prison")

    summary = next(r.message for r in caplog.records if "run summary" in r.message)
    assert "1 indexed" in summary
    assert "1 failed/skipped" in summary


def test_slugify_caption_strips_punctuation_and_caps_word_count():
    assert local_library_index._slugify_caption("A Man, Bench-Pressing (heavy) in a gym!!") == "a_man_benchpressing_heavy_in_a"


def test_slugify_caption_falls_back_when_caption_has_no_words():
    assert local_library_index._slugify_caption("!!!") == "clip"


def test_slugify_caption_treats_underscores_as_word_separators():
    """Doubles as the sanitizer for GPT's own already-underscored slug —
    underscores must split words, not get silently stripped into one
    run-together word."""
    assert local_library_index._slugify_caption("muscular_bodybuilder_flexing_gym") == "muscular_bodybuilder_flexing_gym"


def test_verify_niche_removes_stale_entries_by_default(tmp_path):
    niche_dir = tmp_path / "local" / "prison"
    niche_dir.mkdir(parents=True)
    (niche_dir / "present.mp4").write_bytes(b"fake")
    _write_index(niche_dir, {
        "present.mp4": {"caption": "a", "width": 0, "height": 0, "duration": 0},
        "deleted1.mp4": {"caption": "b", "width": 0, "height": 0, "duration": 0},
        "deleted2.mp4": {"caption": "c", "width": 0, "height": 0, "duration": 0},
    })

    with patch.object(settings, "clips_dir", tmp_path):
        result = local_library_index.verify_niche("prison")

    assert result["checked"] == 3
    assert set(result["stale"]) == {"deleted1.mp4", "deleted2.mp4"}
    assert result["removed"] is True
    on_disk = json.loads((niche_dir / "index.json").read_text())
    assert set(on_disk) == {"present.mp4"}


def test_verify_niche_dry_run_reports_without_removing(tmp_path):
    niche_dir = tmp_path / "local" / "prison"
    niche_dir.mkdir(parents=True)
    _write_index(niche_dir, {"deleted.mp4": {"caption": "a", "width": 0, "height": 0, "duration": 0}})

    with patch.object(settings, "clips_dir", tmp_path):
        result = local_library_index.verify_niche("prison", dry_run=True)

    assert result["stale"] == ["deleted.mp4"]
    assert result["removed"] is False
    on_disk = json.loads((niche_dir / "index.json").read_text())
    assert "deleted.mp4" in on_disk  # untouched


def test_verify_niche_no_index_is_a_noop(tmp_path):
    with patch.object(settings, "clips_dir", tmp_path):
        result = local_library_index.verify_niche("no_such_niche")
    assert result == {"niche": "no_such_niche", "checked": 0, "stale": []}


def test_verify_all_niches_scans_every_niche_dir(tmp_path):
    local_dir = tmp_path / "local"
    _write_index(local_dir / "prison", {"deleted.mp4": {"caption": "a", "width": 0, "height": 0, "duration": 0}})
    _write_index(local_dir / "bodybuilding", {"deleted.mp4": {"caption": "b", "width": 0, "height": 0, "duration": 0}})

    with patch.object(settings, "clips_dir", tmp_path):
        results = local_library_index.verify_all_niches()

    assert {r["niche"] for r in results} == {"prison", "bodybuilding"}
    assert all(r["stale"] == ["deleted.mp4"] for r in results)


def test_index_niche_finds_files_in_subfolders(tmp_path):
    """Bug report: clips organized into per-subject subfolders (e.g.
    clips/local/tennis/Alex Eala/*.mp4) were invisible to indexing — only
    files directly in niche_dir were scanned. Must recurse, and key the
    index by the niche_dir-relative path (not bare filename) so files from
    different subfolders can't collide in index.json."""
    niche_dir = tmp_path / "local" / "tennis"
    subfolder = niche_dir / "Alex Eala"
    subfolder.mkdir(parents=True)
    (subfolder / "clip1.mp4").write_bytes(b"fake")

    with patch.object(settings, "clips_dir", tmp_path), \
         patch.object(local_library_index, "probe", return_value=(3.0, "1920x1080")), \
         patch.object(local_library_index, "generate_json_from_video", return_value={
             "caption": "a tennis player serving", "slug": "tennis_player_serving",
         }):
        result = local_library_index.index_niche("tennis")

    assert "Alex Eala/tennis_player_serving.mp4" in result
    assert (subfolder / "tennis_player_serving.mp4").exists()


def test_index_niche_skips_quarantine_folder(tmp_path):
    """Files find_near_duplicates.py --apply already moved to
    _duplicates_removed/ (and dropped from index.json) must not get
    silently picked back up and re-indexed on the next indexing run."""
    niche_dir = tmp_path / "local" / "bodybuilding"
    quarantine = niche_dir / local_library_index.QUARANTINE_DIRNAME
    quarantine.mkdir(parents=True)
    (quarantine / "removed.mp4").write_bytes(b"fake")

    with patch.object(settings, "clips_dir", tmp_path), \
         patch.object(local_library_index, "generate_json_from_video") as mock_caption:
        result = local_library_index.index_niche("bodybuilding")

    mock_caption.assert_not_called()
    assert result == {}


def test_index_niche_stops_after_current_file_completes(tmp_path):
    """request_stop() (the "Sync Library" stop button's backend) must never
    cut off an in-flight captioning call — only take effect before the NEXT
    file starts. Here request_stop() fires as a side effect of the FIRST
    file's captioning call itself (simulating a stop click landing while
    that call is in flight); that file must still finish and get indexed,
    while the second/third files' captioning calls never happen at all.
    Also confirms the interrupted run's un-indexed files are picked up
    normally by a later, uninterrupted sync (see test plan step 3)."""
    niche_dir = tmp_path / "local" / "stoptest"
    niche_dir.mkdir(parents=True)
    for name, content in (("a.mp4", b"aaa"), ("b.mp4", b"bbb"), ("c.mp4", b"ccc")):
        (niche_dir / name).write_bytes(content)

    calls = []
    stop_on_first_call = True

    def _caption_side_effect(prompt, path, model):
        calls.append(path)
        if stop_on_first_call:
            local_library_index.request_stop("stoptest")
        return {"caption": f"caption {len(calls)}", "slug": f"clip_{len(calls)}"}

    stats: dict = {}
    with patch.object(settings, "clips_dir", tmp_path), \
         patch.object(local_library_index, "probe", return_value=(1.0, "640x360")), \
         patch.object(local_library_index, "generate_json_from_video", side_effect=_caption_side_effect), \
         patch("time.sleep"):
        result = local_library_index.index_niche("stoptest", stats=stats)

    assert len(calls) == 1  # the second/third files' captioning calls never started
    assert list(result) == ["clip_1.mp4"]
    assert stats == {"indexed": 1, "total_new": 3, "stopped": True}
    assert "stoptest" not in local_library_index._stop_requested  # flag cleared, doesn't leak into the next run

    # A later sync (no stop requested) picks up the two files left over.
    stop_on_first_call = False
    with patch.object(settings, "clips_dir", tmp_path), \
         patch.object(local_library_index, "probe", return_value=(1.0, "640x360")), \
         patch.object(local_library_index, "generate_json_from_video", side_effect=_caption_side_effect), \
         patch("time.sleep"):
        resumed = local_library_index.index_niche("stoptest")

    assert len(calls) == 3
    assert set(resumed) == {"clip_1.mp4", "clip_2.mp4", "clip_3.mp4"}


def test_rename_with_unique_suffix_avoids_overwrite(tmp_path):
    existing = tmp_path / "gym.mp4"
    existing.write_bytes(b"existing")
    target = tmp_path / "new.mp4"
    target.write_bytes(b"to_rename")

    renamed = local_library_index._rename_with_unique_suffix(target, "gym")

    assert renamed == tmp_path / "gym_2.mp4"
    assert existing.read_bytes() == b"existing"
    assert renamed.read_bytes() == b"to_rename"


if __name__ == "__main__":
    import tempfile

    for test in [
        test_search_returns_no_hits_without_index,
        test_search_ranks_by_caption_overlap_and_shapes_stockhit,
        test_search_respects_per_page,
        test_index_niche_skips_already_indexed,
        test_index_niche_uses_gpt_slug_directly_not_caption_word_position,
        test_index_niche_falls_back_to_derived_slug_when_gpt_omits_slug,
        test_index_niche_never_overwrites_on_slug_collision,
        test_index_niche_deletes_byte_identical_duplicate_without_captioning,
        test_index_niche_does_not_delete_files_with_different_content,
        test_index_niche_duplicate_check_ignores_already_indexed_files_without_hash,
        test_index_niche_finds_files_in_subfolders,
        test_index_niche_skips_quarantine_folder,
        test_index_niche_stops_after_current_file_completes,
    ]:
        with tempfile.TemporaryDirectory() as d:
            test(Path(d))
    test_slugify_caption_strips_punctuation_and_caps_word_count()
    test_slugify_caption_falls_back_when_caption_has_no_words()
    test_slugify_caption_treats_underscores_as_word_separators()
    with tempfile.TemporaryDirectory() as d:
        test_rename_with_unique_suffix_avoids_overwrite(Path(d))
    print("OK")
