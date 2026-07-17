import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.models import DocumentaryResult


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
