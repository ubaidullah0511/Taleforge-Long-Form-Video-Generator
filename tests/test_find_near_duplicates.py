"""Offline tests for app.stock.find_near_duplicates — the embedding call
(_embed_captions) is patched with fixed vectors so no model download/GPU is
needed; the real model is only exercised by the standalone CLI run against
the actual local library (see ticket's manual "Test" step, not covered here).
"""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.config import settings
from app.stock.find_near_duplicates import apply_removals, find_near_duplicates, main, plan_removals, render_report


def _write_index(niche_dir: Path, index: dict) -> None:
    niche_dir.mkdir(parents=True, exist_ok=True)
    (niche_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")


def test_find_near_duplicates_flags_only_pairs_above_threshold(tmp_path):
    niche_dir = tmp_path / "local" / "prison"
    _write_index(niche_dir, {
        "a.mp4": {"caption": "caption a"},
        "b.mp4": {"caption": "caption b"},  # near-identical embedding to a
        "c.mp4": {"caption": "caption c"},  # unrelated
    })
    # a/b share direction [1,0] (similarity 1.0), c is orthogonal ([0,1], similarity 0.0 to both).
    fake_vectors = np.array([[1.0, 0.0], [0.99, np.sqrt(1 - 0.99 ** 2)], [0.0, 1.0]])

    with patch.object(settings, "clips_dir", tmp_path), \
         patch("app.stock.find_near_duplicates._embed_captions", return_value=fake_vectors):
        pairs = find_near_duplicates("prison", threshold=0.9)

    assert len(pairs) == 1
    score, name_a, caption_a, name_b, caption_b = pairs[0]
    assert {name_a, name_b} == {"a.mp4", "b.mp4"}
    assert score >= 0.9


def test_find_near_duplicates_sorts_highest_similarity_first(tmp_path):
    niche_dir = tmp_path / "local" / "prison"
    _write_index(niche_dir, {
        "a.mp4": {"caption": "caption a"},
        "b.mp4": {"caption": "caption b"},
        "c.mp4": {"caption": "caption c"},
    })
    # a-b similarity 0.99, a-c similarity 0.95, b-c similarity ~0.94 -- all above a low threshold.
    fake_vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.99, np.sqrt(1 - 0.99 ** 2), 0.0],
        [0.95, 0.0, np.sqrt(1 - 0.95 ** 2)],
    ])

    with patch.object(settings, "clips_dir", tmp_path), \
         patch("app.stock.find_near_duplicates._embed_captions", return_value=fake_vectors):
        pairs = find_near_duplicates("prison", threshold=0.5)

    scores = [p[0] for p in pairs]
    assert scores == sorted(scores, reverse=True)


def test_find_near_duplicates_returns_empty_below_threshold(tmp_path):
    niche_dir = tmp_path / "local" / "prison"
    _write_index(niche_dir, {
        "a.mp4": {"caption": "caption a"},
        "b.mp4": {"caption": "caption b"},
    })
    fake_vectors = np.array([[1.0, 0.0], [0.0, 1.0]])  # orthogonal -> similarity 0.0

    with patch.object(settings, "clips_dir", tmp_path), \
         patch("app.stock.find_near_duplicates._embed_captions", return_value=fake_vectors):
        pairs = find_near_duplicates("prison", threshold=0.75)

    assert pairs == []


def test_find_near_duplicates_raises_clear_error_when_niche_not_indexed(tmp_path):
    with patch.object(settings, "clips_dir", tmp_path):
        try:
            find_near_duplicates("nonexistent")
            assert False, "expected FileNotFoundError"
        except FileNotFoundError as exc:
            assert "nonexistent" in str(exc)


def test_find_near_duplicates_handles_single_clip_without_crashing(tmp_path):
    niche_dir = tmp_path / "local" / "prison"
    _write_index(niche_dir, {"a.mp4": {"caption": "caption a"}})

    with patch.object(settings, "clips_dir", tmp_path):
        pairs = find_near_duplicates("prison")

    assert pairs == []


def test_render_report_includes_both_filenames_and_captions_and_states_report_only():
    pairs = [(0.845, "a.mp4", "caption a text", "b.mp4", "caption b text")]
    report = render_report("prison", 0.75, pairs)

    assert "a.mp4" in report and "b.mp4" in report
    assert "caption a text" in report and "caption b text" in report
    assert "0.845" in report
    assert "Flagged pairs: 1" in report
    assert "nothing was deleted or renamed" in report.lower()


def test_plan_removals_keeps_the_larger_file(tmp_path):
    niche_dir = tmp_path / "local" / "prison"
    _write_index(niche_dir, {"big.mp4": {"caption": "caption a"}, "small.mp4": {"caption": "caption b"}})
    (niche_dir / "big.mp4").write_bytes(b"x" * 1000)
    (niche_dir / "small.mp4").write_bytes(b"x" * 10)
    pairs = [(0.95, "big.mp4", "caption a", "small.mp4", "caption b")]

    with patch.object(settings, "clips_dir", tmp_path):
        plan = plan_removals("prison", pairs)

    assert len(plan) == 1
    assert plan[0]["keep"] == "big.mp4"
    assert plan[0]["remove"] == "small.mp4"
    assert plan[0]["keep_size"] == 1000
    assert plan[0]["remove_size"] == 10


def test_plan_removals_clusters_a_chain_and_keeps_only_the_single_largest(tmp_path):
    """A~B and B~C flagged, but A~C never directly flagged (a chain, not a
    fully-connected triangle) — must still resolve to ONE survivor for the
    whole cluster (the largest of the three), not two independent per-pair
    decisions that could each 'keep' a different, smaller file."""
    niche_dir = tmp_path / "local" / "prison"
    _write_index(niche_dir, {
        "a.mp4": {"caption": "a"}, "b.mp4": {"caption": "b"}, "c.mp4": {"caption": "c"},
    })
    (niche_dir / "a.mp4").write_bytes(b"x" * 100)   # largest overall
    (niche_dir / "b.mp4").write_bytes(b"x" * 90)
    (niche_dir / "c.mp4").write_bytes(b"x" * 80)    # smallest, only chained to b via similarity
    pairs = [
        (0.95, "a.mp4", "a", "b.mp4", "b"),
        (0.93, "b.mp4", "b", "c.mp4", "c"),
    ]

    with patch.object(settings, "clips_dir", tmp_path):
        plan = plan_removals("prison", pairs)

    removed = {item["remove"] for item in plan}
    kept = {item["keep"] for item in plan}
    assert removed == {"b.mp4", "c.mp4"}
    assert kept == {"a.mp4"}  # single survivor for the whole 3-file cluster


def test_plan_removals_skips_missing_file_without_crashing(tmp_path, caplog):
    niche_dir = tmp_path / "local" / "prison"
    _write_index(niche_dir, {"a.mp4": {"caption": "a"}, "b.mp4": {"caption": "b"}})
    (niche_dir / "a.mp4").write_bytes(b"x" * 100)
    # b.mp4 referenced in index.json/pairs but not actually present on disk.
    pairs = [(0.95, "a.mp4", "a", "b.mp4", "b")]

    with patch.object(settings, "clips_dir", tmp_path), caplog.at_level("WARNING"):
        plan = plan_removals("prison", pairs)

    assert plan == []  # can't compare sizes with only one real file -> nothing actionable
    assert "b.mp4" in caplog.text


def test_apply_removals_moves_to_quarantine_and_updates_index_leaving_files_recoverable(tmp_path):
    niche_dir = tmp_path / "local" / "prison"
    _write_index(niche_dir, {
        "big.mp4": {"caption": "caption a", "width": 1920},
        "small.mp4": {"caption": "caption b", "width": 1280},
    })
    (niche_dir / "big.mp4").write_bytes(b"x" * 1000)
    (niche_dir / "small.mp4").write_bytes(b"x" * 10)
    plan = [{"keep": "big.mp4", "keep_size": 1000, "remove": "small.mp4", "remove_size": 10}]

    with patch.object(settings, "clips_dir", tmp_path):
        apply_removals("prison", plan)

    # Moved, not deleted -> still recoverable on disk, just relocated.
    assert not (niche_dir / "small.mp4").exists()
    quarantined = niche_dir / "_duplicates_removed" / "small.mp4"
    assert quarantined.exists()
    assert quarantined.read_bytes() == b"x" * 10

    # Kept file completely untouched.
    assert (niche_dir / "big.mp4").exists()
    assert (niche_dir / "big.mp4").read_bytes() == b"x" * 1000

    index = json.loads((niche_dir / "index.json").read_text(encoding="utf-8"))
    assert "small.mp4" not in index
    assert index["big.mp4"] == {"caption": "caption a", "width": 1920}  # kept entry unchanged


def test_apply_removals_does_not_clobber_a_prior_run_already_in_quarantine(tmp_path):
    niche_dir = tmp_path / "local" / "prison"
    _write_index(niche_dir, {"big.mp4": {"caption": "a"}, "small.mp4": {"caption": "b"}})
    (niche_dir / "big.mp4").write_bytes(b"new-big")
    (niche_dir / "small.mp4").write_bytes(b"new-small")
    quarantine_dir = niche_dir / "_duplicates_removed"
    quarantine_dir.mkdir()
    (quarantine_dir / "small.mp4").write_bytes(b"old-quarantined-content")  # from an earlier run
    plan = [{"keep": "big.mp4", "keep_size": 100, "remove": "small.mp4", "remove_size": 10}]

    with patch.object(settings, "clips_dir", tmp_path):
        apply_removals("prison", plan)

    assert (quarantine_dir / "small.mp4").read_bytes() == b"old-quarantined-content"  # untouched
    assert (quarantine_dir / "small_2.mp4").read_bytes() == b"new-small"  # new one got a suffix instead


def test_main_dry_run_does_not_apply_even_when_threshold_matches_apply_default(tmp_path):
    """--apply is the only thing that triggers file movement — passing
    --threshold 0.9 alone (matching APPLY_THRESHOLD) must NOT be treated as
    implicit consent to move files."""
    niche_dir = tmp_path / "local" / "prison"
    _write_index(niche_dir, {"a.mp4": {"caption": "caption a"}, "b.mp4": {"caption": "caption b"}})
    (niche_dir / "a.mp4").write_bytes(b"x" * 1000)
    (niche_dir / "b.mp4").write_bytes(b"x" * 10)
    fake_vectors = np.array([[1.0, 0.0], [0.99, np.sqrt(1 - 0.99 ** 2)]])

    with patch.object(settings, "clips_dir", tmp_path), \
         patch("app.stock.find_near_duplicates._embed_captions", return_value=fake_vectors), \
         patch("sys.argv", ["find_near_duplicates.py", "--niche", "prison", "--threshold", "0.9"]):
        main()

    assert (niche_dir / "a.mp4").exists() and (niche_dir / "b.mp4").exists()
    assert not (niche_dir / "_duplicates_removed").exists()
    plan_path = niche_dir / "duplicates_removal_plan.md"
    assert plan_path.exists()
    assert "DRY RUN" in plan_path.read_text(encoding="utf-8")


def test_main_apply_flag_actually_moves_files_and_uses_090_default_threshold(tmp_path):
    niche_dir = tmp_path / "local" / "prison"
    _write_index(niche_dir, {"a.mp4": {"caption": "caption a"}, "b.mp4": {"caption": "caption b"}})
    (niche_dir / "a.mp4").write_bytes(b"x" * 1000)
    (niche_dir / "b.mp4").write_bytes(b"x" * 10)
    fake_vectors = np.array([[1.0, 0.0], [0.99, np.sqrt(1 - 0.99 ** 2)]])  # similarity ~0.99 -> above 0.9 default

    with patch.object(settings, "clips_dir", tmp_path), \
         patch("app.stock.find_near_duplicates._embed_captions", return_value=fake_vectors), \
         patch("sys.argv", ["find_near_duplicates.py", "--niche", "prison", "--apply"]):
        main()

    assert (niche_dir / "a.mp4").exists()  # larger file kept
    assert not (niche_dir / "b.mp4").exists()  # smaller file moved out
    assert (niche_dir / "_duplicates_removed" / "b.mp4").exists()
    plan_path = niche_dir / "duplicates_removal_plan.md"
    assert "APPLIED" in plan_path.read_text(encoding="utf-8")


def test_main_writes_report_file_and_never_touches_source_clips(tmp_path):
    """Do Not (per ticket): report-only, no auto-delete — confirm the CLI
    entrypoint only ever writes the report file, the index.json and clip
    files are untouched."""
    niche_dir = tmp_path / "local" / "prison"
    _write_index(niche_dir, {
        "a.mp4": {"caption": "caption a"},
        "b.mp4": {"caption": "caption b"},
    })
    (niche_dir / "a.mp4").write_bytes(b"fake video a")
    (niche_dir / "b.mp4").write_bytes(b"fake video b")
    fake_vectors = np.array([[1.0, 0.0], [0.99, np.sqrt(1 - 0.99 ** 2)]])

    with patch.object(settings, "clips_dir", tmp_path), \
         patch("app.stock.find_near_duplicates._embed_captions", return_value=fake_vectors), \
         patch("sys.argv", ["find_near_duplicates.py", "--niche", "prison", "--threshold", "0.9"]):
        main()

    report_path = niche_dir / "near_duplicates_report.md"
    assert report_path.exists()
    assert "a.mp4" in report_path.read_text(encoding="utf-8")
    assert (niche_dir / "a.mp4").read_bytes() == b"fake video a"  # untouched
    assert (niche_dir / "b.mp4").read_bytes() == b"fake video b"  # untouched
    assert json.loads((niche_dir / "index.json").read_text(encoding="utf-8")) == {
        "a.mp4": {"caption": "caption a"}, "b.mp4": {"caption": "caption b"},
    }


if __name__ == "__main__":
    import tempfile

    for test in [
        test_find_near_duplicates_flags_only_pairs_above_threshold,
        test_find_near_duplicates_sorts_highest_similarity_first,
        test_find_near_duplicates_returns_empty_below_threshold,
        test_find_near_duplicates_raises_clear_error_when_niche_not_indexed,
        test_find_near_duplicates_handles_single_clip_without_crashing,
        test_plan_removals_keeps_the_larger_file,
        test_plan_removals_clusters_a_chain_and_keeps_only_the_single_largest,
        test_apply_removals_moves_to_quarantine_and_updates_index_leaving_files_recoverable,
        test_apply_removals_does_not_clobber_a_prior_run_already_in_quarantine,
        test_main_dry_run_does_not_apply_even_when_threshold_matches_apply_default,
        test_main_apply_flag_actually_moves_files_and_uses_090_default_threshold,
        test_main_writes_report_file_and_never_touches_source_clips,
    ]:
        with tempfile.TemporaryDirectory() as d:
            test(Path(d))
    # test_plan_removals_skips_missing_file_without_crashing needs pytest's
    # caplog fixture (like test_local_library.py's caplog tests) -- run via
    # pytest, not this __main__ block.
    test_render_report_includes_both_filenames_and_captions_and_states_report_only()
    print("OK")
