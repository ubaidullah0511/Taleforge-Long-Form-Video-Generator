import uuid
from typing import Optional

from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app import documentary_pipeline, pipeline
from app.clip_ingest import analyze_and_store
from app.config import settings
from app.models import DocumentaryResult, SceneResult
from app.vector_store import query_similar

app = FastAPI(title="B-Roll Retrieval System")


class ScriptRequest(BaseModel):
    script: Optional[str] = Field(default=None)
    raw_data: Optional[str] = Field(default=None)

    @property
    def resolved_script(self) -> str:
        return self.script or self.raw_data or ""


class DocumentaryRequest(BaseModel):
    script: str = ""
    script_path: str | None = None
    project_name: str | None = None
    allow_duplicate_assets: bool = False
    max_downloads: int | None = None
    # Manually-supplied voiceover recording (no TTS integration) — if given,
    # real Whisper timestamps replace the word-count timing estimate for
    # both captions and per-clip duration. See app/transcription.py.
    audio_path: str | None = None


class IngestRequest(BaseModel):
    video_path: str
    start: float
    end: float


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).resolve().parent.parent / "templates" / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.post("/process-script", response_model=list[SceneResult])
async def process_script(req: ScriptRequest):
    return await pipeline.run(req.resolved_script)


@app.post("/generate-documentary-timeline", response_model=DocumentaryResult)
async def generate_documentary_timeline(req: DocumentaryRequest):
    return await documentary_pipeline.run(
        req.script, req.project_name, req.allow_duplicate_assets, req.max_downloads,
        script_path=req.script_path, audio_path=req.audio_path,
    )


@app.post("/generate-documentary-timeline/upload", response_model=DocumentaryResult)
async def generate_documentary_timeline_upload(
    script: str = Form(""),
    project_name: str | None = Form(None),
    allow_duplicate_assets: bool = Form(False),
    max_downloads: int | None = Form(None),
    script_file: UploadFile | None = File(None),
    audio_file: UploadFile | None = File(None),
):
    """multipart/form-data counterpart to /generate-documentary-timeline, for
    the frontend's file pickers (upload a script .txt and/or a voiceover
    recording). Saves whatever files were sent to a per-request scratch
    directory, then calls documentary_pipeline.run() — the same single entry
    point the JSON route uses — with script_path/audio_path pointing at them."""
    upload_dir = settings.uploads_dir / uuid.uuid4().hex
    upload_dir.mkdir(parents=True, exist_ok=True)

    script_path = None
    if script_file is not None and script_file.filename:
        script_path = upload_dir / script_file.filename
        script_path.write_bytes(await script_file.read())

    audio_path = None
    if audio_file is not None and audio_file.filename:
        audio_path = upload_dir / audio_file.filename
        audio_path.write_bytes(await audio_file.read())

    return await documentary_pipeline.run(
        script, project_name, allow_duplicate_assets, max_downloads,
        script_path=str(script_path) if script_path else None,
        audio_path=str(audio_path) if audio_path else None,
    )


@app.post("/clips/ingest")
async def ingest_clip(req: IngestRequest):
    clip = analyze_and_store(req.video_path, req.start, req.end, source="local")
    return clip


@app.get("/clips/search")
async def search_clips(query: str, top_k: int = 5):
    matches = query_similar(query, top_k=top_k)
    return [{"clip": clip, "similarity": similarity} for clip, similarity in matches]
