"""Offline tests for app.progress. Run with: pytest tests/test_progress.py"""
from app import progress


def test_start_creates_entry_in_input_stage():
    progress.start("proj_a")
    entry = progress.get("proj_a")
    assert entry["stage"] == "input"
    assert entry["done"] is False


def test_update_without_context_is_a_no_op():
    progress.set_context(None)
    progress.update("table", "should not be recorded anywhere")
    assert progress.get("no_such_project") is None


def test_update_records_stage_detail_and_clip_counters():
    progress.set_context("proj_b", clip_index=3, total_clips=10)
    progress.update("search", "Searching clip 3 of 10")
    entry = progress.get("proj_b")
    assert entry["stage"] == "search"
    assert entry["detail"] == "Searching clip 3 of 10"
    assert entry["current"] == 3
    assert entry["total"] == 10


def test_finish_marks_done_with_result():
    progress.start("proj_c")
    progress.finish("proj_c", {"final_video_path": "x.mp4"})
    entry = progress.get("proj_c")
    assert entry["done"] is True
    assert entry["error"] is None
    assert entry["result"] == {"final_video_path": "x.mp4"}


def test_fail_marks_done_with_error_and_no_result():
    progress.start("proj_d")
    progress.fail("proj_d", "something broke")
    entry = progress.get("proj_d")
    assert entry["done"] is True
    assert entry["error"] == "something broke"
    assert entry["result"] is None


def test_get_returns_none_for_unknown_project():
    assert progress.get("never_started") is None


if __name__ == "__main__":
    test_start_creates_entry_in_input_stage()
    test_update_without_context_is_a_no_op()
    test_update_records_stage_detail_and_clip_counters()
    test_finish_marks_done_with_result()
    test_fail_marks_done_with_error_and_no_result()
    test_get_returns_none_for_unknown_project()
    print("OK")
