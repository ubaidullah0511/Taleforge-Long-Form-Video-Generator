import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.config import settings
from app.documentary_pipeline import DocumentaryPlan
from app.main import app
from app.models import AvailabilityReport, DocumentaryResult, TimelineClip


client = TestClient(app)


def test_process_script_accepts_raw_data():
    with patch("app.main.pipeline.run", new=AsyncMock(return_value=[])) as mock_run:
        response = client.post("/process-script", json={"raw_data": "some raw narration"})

    assert response.status_code == 200
    assert mock_run.await_count == 1
    assert mock_run.await_args.args[0] == "some raw narration"


def _fake_result():
    return DocumentaryResult(project_dir="projects/x", table=[], timeline=[])


def test_generate_documentary_timeline_upload_saves_files_and_calls_run():
    script_bytes = b"A script uploaded as a file."
    audio_bytes = b"fake audio bytes"

    with tempfile.TemporaryDirectory() as tmpdir, \
         patch.object(settings, "uploads_dir", Path(tmpdir)), \
         patch("app.main.documentary_pipeline.run", new=AsyncMock(return_value=_fake_result())) as mock_run:
        response = client.post(
            "/generate-documentary-timeline/upload",
            data={"project_name": "upload_proj", "max_downloads": "2"},
            files={
                "script_file": ("script.txt", script_bytes, "text/plain"),
                "audio_file": ("narration.wav", audio_bytes, "audio/wav"),
            },
        )

        assert response.status_code == 200
        mock_run.assert_awaited_once()
        call_args = mock_run.await_args
        assert call_args.args == ("", "upload_proj", False, 2)
        script_path = Path(call_args.kwargs["script_path"])
        audio_path = Path(call_args.kwargs["audio_path"])
        assert script_path.read_bytes() == script_bytes
        assert audio_path.read_bytes() == audio_bytes


def test_generate_documentary_timeline_upload_works_with_pasted_script_only():
    with patch("app.main.documentary_pipeline.run", new=AsyncMock(return_value=_fake_result())) as mock_run:
        response = client.post(
            "/generate-documentary-timeline/upload",
            data={"script": "pasted script text, no files"},
        )

    assert response.status_code == 200
    call_args = mock_run.await_args
    assert call_args.args[0] == "pasted script text, no files"
    assert call_args.kwargs["script_path"] is None
    assert call_args.kwargs["audio_path"] is None


def test_get_project_returns_404_when_timeline_missing():
    with tempfile.TemporaryDirectory() as tmpdir, \
         patch.object(settings, "documentary_projects_dir", Path(tmpdir)):
        response = client.get("/project/does_not_exist")
    assert response.status_code == 404


def test_get_project_returns_timeline_with_asset_urls():
    from app.models import AssetInfo, TimelineEntry

    with tempfile.TemporaryDirectory() as tmpdir:
        projects_dir = Path(tmpdir)
        project_dir = projects_dir / "proj1"
        clip_dir = project_dir / "clips" / "clip_001"
        clip_dir.mkdir(parents=True)
        asset_path = clip_dir / "asset.mp4"
        asset_path.write_bytes(b"fake video bytes")
        entry = TimelineEntry(
            clip_number=1, start="00:00:00", end="00:00:03", duration=3.0, script="beat",
            asset_path=str(asset_path),
            asset_metadata=AssetInfo(path=str(asset_path), source="pexels", source_id="1",
                                      media_type="video", width=1920, height=1080, duration=3.0, score=80.0),
            recommended_effect="e", transition="t",
        )
        (project_dir / "timeline.json").write_text(f"[{entry.model_dump_json()}]", encoding="utf-8")

        with patch.object(settings, "documentary_projects_dir", projects_dir):
            response = client.get("/project/proj1")

    assert response.status_code == 200
    data = response.json()
    assert data["timeline"][0]["clip_number"] == 1
    assert data["timeline"][0]["asset_url"] == "/project-files/proj1/clips/clip_001/asset.mp4"


def test_get_project_finds_final_video_under_its_sanitized_project_name_filename():
    """GET /project must look for the narrated final video under the same
    sanitized-project-name filename documentary_pipeline.final_video_filename
    computes (see _finalize_and_render, which writes it there) — not the old
    hardcoded final_video_with_narration.mp4, which a real project named like
    this would never actually have on disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        projects_dir = Path(tmpdir)
        project_dir = projects_dir / "Alex Eala Toronto Recap"
        project_dir.mkdir(parents=True)
        (project_dir / "timeline.json").write_text("[]", encoding="utf-8")
        (project_dir / "Alex_Eala_Toronto_Recap.mp4").write_bytes(b"fake video bytes")

        with patch.object(settings, "documentary_projects_dir", projects_dir):
            response = client.get("/project/Alex%20Eala%20Toronto%20Recap")

    assert response.status_code == 200
    data = response.json()
    assert data["final_video_with_narration_url"] == (
        "/project-files/Alex Eala Toronto Recap/Alex_Eala_Toronto_Recap.mp4"
    )
    assert data["audio_url"] == data["final_video_with_narration_url"]  # falls back to it, same as before


def test_rerender_clip_route_requires_alternate_index_for_alternate_mode():
    response = client.patch("/project/anyproj/clip/1", json={"mode": "alternate"})
    assert response.status_code == 400
    assert "alternate_index" in response.json()["detail"]


def test_rerender_clip_route_delegates_to_pipeline_and_returns_entry():
    from app.models import AssetInfo, TimelineEntry

    fake_entry = TimelineEntry(
        clip_number=1, start="00:00:00", end="00:00:03", duration=3.0, script="beat",
        asset_path="a.mp4",
        asset_metadata=AssetInfo(path="a.mp4", source="pexels", source_id="99", media_type="video",
                                  width=1920, height=1080, duration=3.0, score=70.0),
        recommended_effect="e", transition="t", note="swapped to alternate 0 (pexels:99) via editor",
    )
    with patch("app.main.documentary_pipeline.rerender_single_clip", new=AsyncMock(return_value=fake_entry)) as mock_rerender:
        response = client.patch("/project/proj1/clip/1", json={"mode": "alternate", "alternate_index": 0})

    assert response.status_code == 200
    assert response.json()["asset_metadata"]["source_id"] == "99"
    mock_rerender.assert_awaited_once_with("proj1", 1, alternate_index=0)


def test_rerender_clip_route_maps_value_error_to_400():
    with patch("app.main.documentary_pipeline.rerender_single_clip",
               new=AsyncMock(side_effect=ValueError("clip 99 not found in project 'proj1'"))):
        response = client.patch("/project/proj1/clip/99", json={"mode": "ai"})
    assert response.status_code == 400
    assert "clip 99 not found" in response.json()["detail"]


def test_generate_full_video_route_starts_background_job():
    with patch("app.main.documentary_pipeline.generate_full_video",
               new=AsyncMock(return_value=_fake_result())) as mock_generate:
        response = client.post("/project/proj1/generate-full-video")
    assert response.status_code == 200
    assert response.json() == {"project_name": "proj1"}
    # Background task runs synchronously under TestClient by the time the response is returned.
    mock_generate.assert_awaited_once_with("proj1")


def test_preview_documentary_timeline_returns_table_and_availability_without_running():
    """The preview endpoint must call plan_documentary (table + availability
    scan only) and never documentary_pipeline.run (which would download/
    CLIP-verify/render) — that's the entire point of the cheap preview."""
    fake_table = [TimelineClip(
        clip_number=1, section="INTRO", script_beat="beat", canva_keyword="k",
        fallback_keyword="f", visual_type="Cinematic", edit_note="n",
    )]
    fake_plan = DocumentaryPlan(
        project_name="preview_proj", table=fake_table,
        footage_availability=AvailabilityReport(total_clips=1, thin_count=0, thin_clip_numbers=[], clips=[]),
        whisper_words=None, transcript_alignment_warning=None,
    )

    with patch("app.main.documentary_pipeline.plan_documentary", new=AsyncMock(return_value=fake_plan)) as mock_plan, \
         patch("app.main.documentary_pipeline.run", new=AsyncMock()) as mock_run:
        response = client.post(
            "/generate-documentary-timeline/preview",
            json={"script": "some script", "project_name": "preview_proj", "content_niche": "trucks"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["project_name"] == "preview_proj"
    assert data["table"][0]["clip_number"] == 1
    assert data["footage_availability"]["total_clips"] == 1
    mock_plan.assert_awaited_once()
    mock_run.assert_not_awaited()


def test_preview_documentary_timeline_rejects_multi_select_with_mismatched_parents():
    """Two sub-niche keys from different parent categories must be rejected
    with a 400 before plan_documentary is even called (see
    app.main._validate_content_niche / app.niches.resolve_sub_niches) — a
    single-select content_niche (str or 1-item list) is never validated this
    way, only a real 2+-key multi-select."""
    from app.niches import NICHES, NicheConfig

    NICHES["ml_parent_a"] = NicheConfig(
        key="ml_parent_a", display_name="A", system_context="", banned_terms=[], positive_terms=[], safe_fallback_keyword="x",
    )
    NICHES["ml_child_a"] = NicheConfig(
        key="ml_child_a", display_name="A child", system_context="", banned_terms=[], positive_terms=[],
        safe_fallback_keyword="x", parent_key="ml_parent_a",
    )
    NICHES["ml_parent_b"] = NicheConfig(
        key="ml_parent_b", display_name="B", system_context="", banned_terms=[], positive_terms=[], safe_fallback_keyword="x",
    )
    NICHES["ml_child_b"] = NicheConfig(
        key="ml_child_b", display_name="B child", system_context="", banned_terms=[], positive_terms=[],
        safe_fallback_keyword="x", parent_key="ml_parent_b",
    )
    try:
        with patch("app.main.documentary_pipeline.plan_documentary", new=AsyncMock()) as mock_plan:
            response = client.post(
                "/generate-documentary-timeline/preview",
                json={"script": "some script", "content_niche": ["ml_child_a", "ml_child_b"]},
            )
        assert response.status_code == 400
        assert "same parent" in response.json()["detail"]
        mock_plan.assert_not_awaited()
    finally:
        for key in ["ml_parent_a", "ml_child_a", "ml_parent_b", "ml_child_b"]:
            NICHES.pop(key, None)


def test_preview_documentary_timeline_accepts_multi_select_with_shared_parent():
    """The happy path: 2+ sub-niche keys sharing one parent pass validation
    untouched and reach plan_documentary as the same list the client sent."""
    from app.niches import NICHES, NicheConfig

    NICHES["ml_parent_ok"] = NicheConfig(
        key="ml_parent_ok", display_name="OK Parent", system_context="", banned_terms=[], positive_terms=[],
        safe_fallback_keyword="x",
    )
    NICHES["ml_child_ok_a"] = NicheConfig(
        key="ml_child_ok_a", display_name="OK Child A", system_context="", banned_terms=[], positive_terms=[],
        safe_fallback_keyword="x", parent_key="ml_parent_ok",
    )
    NICHES["ml_child_ok_b"] = NicheConfig(
        key="ml_child_ok_b", display_name="OK Child B", system_context="", banned_terms=[], positive_terms=[],
        safe_fallback_keyword="x", parent_key="ml_parent_ok",
    )
    fake_plan = DocumentaryPlan(
        project_name="preview_proj2", table=[],
        footage_availability=AvailabilityReport(total_clips=0, thin_count=0, thin_clip_numbers=[], clips=[]),
        whisper_words=None, transcript_alignment_warning=None,
    )
    try:
        with patch("app.main.documentary_pipeline.plan_documentary", new=AsyncMock(return_value=fake_plan)) as mock_plan:
            response = client.post(
                "/generate-documentary-timeline/preview",
                json={
                    "script": "some script", "project_name": "preview_proj2",
                    "content_niche": ["ml_child_ok_a", "ml_child_ok_b"],
                },
            )
        assert response.status_code == 200
        mock_plan.assert_awaited_once()
        assert mock_plan.await_args.args[3] == ["ml_child_ok_a", "ml_child_ok_b"]
    finally:
        for key in ["ml_parent_ok", "ml_child_ok_a", "ml_child_ok_b"]:
            NICHES.pop(key, None)
