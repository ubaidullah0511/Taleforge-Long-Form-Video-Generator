import logging
import re
import uuid
from typing import Literal, Optional

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app import documentary_pipeline, pipeline, progress
from app.clip_ingest import analyze_and_store
from app.config import settings
from app.llm_client import generate_json
from app.models import DocumentaryPreview, SceneResult, TimelineEntry
from app.niches import (
    DEFAULT_NICHE,
    NICHES,
    add_custom_niche,
    delete_custom_niche,
    is_custom_niche,
    rename_custom_niche,
    resolve_sub_niches,
    sorted_niches,
)
from app.stock.local_library_index import index_niche, request_stop
from app.vector_store import query_similar

logger = logging.getLogger(__name__)

app = FastAPI(title="B-Roll Retrieval System")

# Lets the editor's browser <video>/<img> tags play rendered segments,
# alternates-swap downloads, and final videos straight off disk — e.g.
# /project-files/{project_name}/clips/clip_001/clip001_rendered.mp4. Read-only,
# same tree documentary_pipeline.run() already writes to.
app.mount("/project-files", StaticFiles(directory=settings.documentary_projects_dir), name="project-files")


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
    # A plain key (today's single-select, unchanged) or a list of sub-niche
    # keys sharing one parent category (multi-select — see
    # app.niches.resolve_sub_niches and the niche dropdown in
    # templates/index.html). Validated in the endpoints below, not here —
    # Pydantic just needs to accept either shape.
    content_niche: str | list[str] = DEFAULT_NICHE
    # Footage-source toggles (see templates/index.html's 3 checkboxes) — all
    # default True, matching today's full-search behavior unchanged.
    enable_ai_generation: bool = True
    enable_local_library: bool = True
    enable_stock_providers: bool = True


def _validate_content_niche(content_niche: str | list[str]) -> None:
    """Only a real multi-select (2+ keys) needs validating — a plain string
    or a single-item list is exactly today's single-select path, which
    already tolerates an unknown key by silently falling back to
    DEFAULT_NICHE (see get_niche) and isn't being changed here. 2+ keys must
    share one parent category (see resolve_sub_niches) — surfaced as a 400
    immediately, before the background task is even queued, rather than as a
    buried progress.fail() the client would have to poll for."""
    if isinstance(content_niche, list) and len(content_niche) > 1:
        try:
            resolve_sub_niches(content_niche)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


class IngestRequest(BaseModel):
    video_path: str
    start: float
    end: float


class AddNicheRequest(BaseModel):
    name: str
    parent_key: str | None = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).resolve().parent.parent / "templates" / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/theme.css")
async def theme_css():
    """Shared stylesheet for index.html and editor.html — one file so the
    two pages can't drift out of visual sync with each other."""
    css_path = Path(__file__).resolve().parent.parent / "templates" / "theme.css"
    return Response(content=css_path.read_text(encoding="utf-8"), media_type="text/css")


@app.post("/process-script", response_model=list[SceneResult])
async def process_script(req: ScriptRequest):
    return await pipeline.run(req.resolved_script)


def _slugify_niche_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _niche_key_or_name_taken(key: str, display_name: str) -> bool:
    return key in NICHES or any(n.display_name.lower() == display_name.lower() for n in NICHES.values())


@app.get("/niches")
async def list_niches():
    """Full niche list (built-in + custom, from the live NICHES registry —
    which itself merges custom_niches.json at import time and on every
    add_custom_niche() call), grouped so sub-niches sort directly under
    their parent. Read on every request, so a page reload always reflects
    niches added in a prior process/session, not just the current one.
    is_custom/local_library_path let the frontend show Sync/Rename/Delete
    controls only for user-added niches that actually have a local library
    to sync (see the niche-management panel in templates/index.html)."""
    return [
        {
            "key": n.key,
            "display_name": n.display_name,
            "parent_key": n.parent_key,
            "is_custom": is_custom_niche(n.key),
            "local_library_path": n.local_library_path,
        }
        for n in sorted_niches()
    ]


class RenameNicheRequest(BaseModel):
    display_name: str


@app.patch("/niches/{key}")
async def rename_niche(key: str, req: RenameNicheRequest):
    """Edits a custom niche's display_name only — see
    app.niches.rename_custom_niche for why key renames aren't supported."""
    display_name = req.display_name.strip()
    try:
        config = rename_custom_niche(key, display_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"key": config.key, "display_name": config.display_name}


@app.delete("/niches/{key}")
async def delete_niche(key: str):
    """Deletes a custom niche's configuration entry — see
    app.niches.delete_custom_niche for the built-in-niche guard and the
    block-on-sub-niches policy."""
    try:
        delete_custom_niche(key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"deleted": key}


def _sync_progress_key(key: str) -> str:
    return f"niche-sync:{key}"


async def _run_sync_tracked(sync_key: str, key: str, local_library_path: str) -> None:
    stats: dict = {}
    try:
        await run_in_threadpool(index_niche, local_library_path, stats)
        progress.finish(sync_key, {
            "indexed_new": stats.get("indexed", 0),
            "total_new": stats.get("total_new", 0),
            "stopped": stats.get("stopped", False),
        })
    except Exception as exc:
        logger.exception("index_niche failed for niche %s", key)
        progress.fail(sync_key, str(exc))


@app.post("/niches/{key}/sync-library")
async def sync_niche_library(key: str, background_tasks: BackgroundTasks):
    """Manually-triggered re-index of a niche's local library folder (see
    app.stock.local_library_index.index_niche) — captions any new files
    dropped into clips/local/<local_library_path>/ since the last sync and
    skips everything already indexed. Deliberately NOT run automatically on
    server start or file changes: each new file costs a real OpenAI vision
    call, so indexing only happens when the user explicitly asks for it
    here. Runs as a background task (same progress-polling shape as
    /generate-full-video, see app.progress) rather than blocking the
    request, so the frontend can poll GET /progress/{sync_key} and show a
    Stop button (POST .../sync-library/stop) while it's running."""
    niche = NICHES.get(key)
    if niche is None:
        raise HTTPException(status_code=404, detail=f"unknown niche '{key}'")
    if not niche.local_library_path:
        raise HTTPException(status_code=400, detail=f"niche '{key}' has no local library folder configured")

    sync_key = _sync_progress_key(key)
    progress.start(sync_key)
    background_tasks.add_task(_run_sync_tracked, sync_key, key, niche.local_library_path)
    return {"sync_key": sync_key}


@app.post("/niches/{key}/sync-library/stop")
async def stop_niche_library_sync(key: str):
    """Requests the in-progress sync for this niche stop before its NEXT
    file (see app.stock.local_library_index.request_stop) — the file
    currently being captioned always finishes; already-indexed files from
    earlier in the run stay indexed. No-op (still 200) if no sync is
    actually running, same as any other best-effort cancel request."""
    niche = NICHES.get(key)
    if niche is None:
        raise HTTPException(status_code=404, detail=f"unknown niche '{key}'")
    if not niche.local_library_path:
        raise HTTPException(status_code=400, detail=f"niche '{key}' has no local library folder configured")
    request_stop(niche.local_library_path)
    return {"stopping": True}


@app.post("/niches")
async def add_niche(req: AddNicheRequest):
    """Generates a starting NicheConfig for a user-typed niche name via LLM
    and persists it (see app.niches.add_custom_niche) so it's immediately
    usable in the category dropdown, no manual app/niches.py editing or
    restart required. If parent_key is given, generates a narrower
    SUB-niche scoped within that parent's context instead (see the
    parent-context prompt below)."""
    name = req.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    if _niche_key_or_name_taken(_slugify_niche_key(name), name):
        raise HTTPException(status_code=409, detail=f"a niche matching '{name}' already exists")

    parent = None
    if req.parent_key:
        parent = NICHES.get(req.parent_key)
        if parent is None:
            raise HTTPException(status_code=400, detail=f"unknown parent niche '{req.parent_key}'")

    if parent:
        prompt = (
            f"Generate a content-niche configuration for a documentary/B-roll "
            f"video pipeline about: {name}, as a SUB-CATEGORY within the broader "
            f"'{parent.display_name}' niche.\n\n"
            f"Parent niche context for reference:\n{parent.system_context}\n\n"
            f"The sub-niche's system_context should:\n"
            f"- Stay within the parent niche's overall scope (still fundamentally "
            f"about {parent.display_name}), but narrow the focus specifically to: "
            f"{name}.\n"
            f"- Follow the same pattern as the parent: what IS in scope, what "
            f"must every keyword depict, what should NEVER be generated.\n\n"
            f"Return JSON: {{\"key\": ..., \"display_name\": ..., "
            f"\"system_context\": ..., \"banned_terms\": [...]}}"
        )
    else:
        prompt = (
            f"Generate a content-niche configuration for a documentary/B-roll "
            f"video pipeline about: {name}.\n\n"
            f"Return JSON with:\n"
            f"- key: a lowercase_snake_case identifier\n"
            f"- display_name: a clean display name\n"
            f"- system_context: 3-5 sentences defining what visual content IS and "
            f"IS NOT in scope for this niche, following this exact pattern: state "
            f"what the content is strictly about, list relevant subjects, state "
            f"every generated keyword MUST clearly depict [niche] content, and "
            f"explicitly say what should NEVER be generated (unrelated adjacent "
            f"topics a naive search might drift into).\n"
            f"- banned_terms: 5-10 lowercase terms/phrases that are NEVER valid "
            f"for this niche regardless of context (unambiguous exclusions only — "
            f"not context-dependent terms).\n"
            f"Return JSON: {{\"key\": ..., \"display_name\": ..., "
            f"\"system_context\": ..., \"banned_terms\": [...]}}"
        )
    try:
        data = generate_json(prompt, settings.llm_text_model)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"niche generation failed: {exc}") from exc

    key = _slugify_niche_key(str(data.get("key") or name))
    display_name = str(data.get("display_name") or name).strip()
    if _niche_key_or_name_taken(key, display_name):
        raise HTTPException(status_code=409, detail=f"niche '{key}' already exists")

    system_context = str(data.get("system_context") or "").strip()
    banned_terms = [str(t).strip().lower() for t in (data.get("banned_terms") or []) if str(t).strip()]
    if parent:
        # Union with the parent's banned_terms, deduplicated — a sub-niche
        # is never less restrictive than its parent.
        merged = dict.fromkeys(parent.banned_terms)
        merged.update(dict.fromkeys(banned_terms))
        banned_terms = list(merged)

    config = add_custom_niche(
        key, display_name, system_context, banned_terms,
        parent_key=parent.key if parent else None,
    )
    return {"key": config.key, "display_name": config.display_name, "parent_key": config.parent_key}


async def _run_tracked(project_name: str, target, *run_args, **run_kwargs) -> None:
    """Runs a documentary_pipeline entry point (run() for the full pipeline,
    generate_and_edit() for the "Generate & Edit" mode — see below) as a
    FastAPI BackgroundTasks job and records the outcome in app.progress so
    GET /progress/{project_name} has something to report. The endpoint has
    already returned to the client by the time this executes (or, under
    TestClient, runs to completion before the test gets control back — see
    Starlette's BackgroundTasks docs), so any exception here would otherwise
    vanish silently instead of reaching the caller as an HTTP error.

    project_name is its own leading parameter (not read out of run_args/
    run_kwargs) purely for progress tracking here — it's passed a second
    time inside run_args/run_kwargs too, exactly as target itself expects
    it, so call sites still invoke target() with the same positional shape
    existing tests assert on (target is resolved to documentary_pipeline.run
    at each call site, so patching that attribute still intercepts it)."""
    try:
        result = await target(*run_args, **run_kwargs)
        progress.finish(project_name, result.model_dump())
    except Exception as exc:
        logger.exception("documentary_pipeline job failed for project %s", project_name)
        progress.fail(project_name, str(exc))


@app.post("/generate-documentary-timeline")
async def generate_documentary_timeline(req: DocumentaryRequest, background_tasks: BackgroundTasks):
    _validate_content_niche(req.content_niche)
    project_name = documentary_pipeline.resolve_project_name(req.project_name)
    progress.start(project_name)
    background_tasks.add_task(
        _run_tracked, project_name, documentary_pipeline.run,
        req.script, project_name, req.allow_duplicate_assets, req.max_downloads,
        script_path=req.script_path, audio_path=req.audio_path, content_niche=req.content_niche,
        enable_ai_generation=req.enable_ai_generation, enable_local_library=req.enable_local_library,
        enable_stock_providers=req.enable_stock_providers,
    )
    return {"project_name": project_name}


@app.post("/generate-documentary-timeline/edit")
async def generate_documentary_timeline_edit(req: DocumentaryRequest, background_tasks: BackgroundTasks):
    """"Generate & Edit" mode (frontend's Run-mode dropdown): same background-
    task/progress shape as /generate-documentary-timeline, but runs
    documentary_pipeline.generate_and_edit() instead of run() — stops after
    per-clip asset resolution + segment render rather than also paying for
    the Remotion/audio-mux tail, since the frontend redirects straight to
    the editor once this finishes rather than showing results on this page."""
    _validate_content_niche(req.content_niche)
    project_name = documentary_pipeline.resolve_project_name(req.project_name)
    progress.start(project_name)
    background_tasks.add_task(
        _run_tracked, project_name, documentary_pipeline.generate_and_edit,
        req.script, project_name, req.allow_duplicate_assets, req.max_downloads,
        script_path=req.script_path, audio_path=req.audio_path, content_niche=req.content_niche,
        enable_ai_generation=req.enable_ai_generation, enable_local_library=req.enable_local_library,
        enable_stock_providers=req.enable_stock_providers,
    )
    return {"project_name": project_name}


@app.post("/generate-documentary-timeline/preview", response_model=DocumentaryPreview)
async def preview_documentary_timeline(req: DocumentaryRequest):
    """Same table + pre-render footage availability scan a real run() would
    produce (see documentary_pipeline.plan_documentary), returned
    immediately with no download/CLIP-verify/render work — lets a thin or
    bad batch get caught before committing to the full, expensive pipeline.
    JSON-only, mirroring /generate-documentary-timeline (not its /upload
    counterpart) — file-based script/audio inputs aren't supported here yet."""
    _validate_content_niche(req.content_niche)
    project_name = documentary_pipeline.resolve_project_name(req.project_name)
    try:
        script = documentary_pipeline.resolve_script_text(req.script, req.script_path)
    except ValueError as exc:
        # Unlike /generate-documentary-timeline (a background task, whose
        # errors go through progress.fail instead), this endpoint runs
        # synchronously in the request — a bad request must surface as 400,
        # not an unhandled 500.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    plan = await documentary_pipeline.plan_documentary(
        script, project_name, req.audio_path, req.content_niche,
        enable_local_library=req.enable_local_library, enable_stock_providers=req.enable_stock_providers,
    )
    return DocumentaryPreview(
        project_name=plan.project_name, table=plan.table, footage_availability=plan.footage_availability,
    )


async def _save_upload(upload_dir: Path, file: UploadFile | None) -> Path | None:
    """Shared by both multipart routes below — saves an optional uploaded
    file into a per-request scratch directory, or does nothing if the field
    was omitted (FastAPI still hands back an UploadFile with an empty
    filename for an unset optional file field, not None)."""
    if file is None or not file.filename:
        return None
    path = upload_dir / file.filename
    path.write_bytes(await file.read())
    return path


async def _generate_documentary_timeline_upload(
    target,
    background_tasks: BackgroundTasks,
    script: str,
    project_name: str | None,
    allow_duplicate_assets: bool,
    max_downloads: int | None,
    content_niche: str | list[str],
    script_file: UploadFile | None,
    audio_file: UploadFile | None,
    enable_ai_generation: bool = True,
    enable_local_library: bool = True,
    enable_stock_providers: bool = True,
) -> dict:
    """Shared body of both multipart routes below — only the target pipeline
    entry point (run() vs generate_and_edit()) differs between them."""
    _validate_content_niche(content_niche)
    upload_dir = settings.uploads_dir / uuid.uuid4().hex
    upload_dir.mkdir(parents=True, exist_ok=True)

    script_path = await _save_upload(upload_dir, script_file)
    audio_path = await _save_upload(upload_dir, audio_file)

    resolved_project_name = documentary_pipeline.resolve_project_name(project_name)
    progress.start(resolved_project_name)
    background_tasks.add_task(
        _run_tracked, resolved_project_name, target,
        script, resolved_project_name, allow_duplicate_assets, max_downloads,
        script_path=str(script_path) if script_path else None,
        audio_path=str(audio_path) if audio_path else None,
        content_niche=content_niche,
        enable_ai_generation=enable_ai_generation, enable_local_library=enable_local_library,
        enable_stock_providers=enable_stock_providers,
    )
    return {"project_name": resolved_project_name}


@app.post("/generate-documentary-timeline/upload")
async def generate_documentary_timeline_upload(
    background_tasks: BackgroundTasks,
    script: str = Form(""),
    project_name: str | None = Form(None),
    allow_duplicate_assets: bool = Form(False),
    max_downloads: int | None = Form(None),
    content_niche: str = Form(DEFAULT_NICHE),
    # Repeated form field for multi-select sub-niches (the frontend appends
    # one content_niche_list entry per selected sub-niche) — kept separate
    # from content_niche above rather than typing that field as list[str],
    # since a Form field's type decides how FastAPI parses it and content_niche
    # must stay a plain single value for old callers that still send one.
    content_niche_list: list[str] | None = Form(None),
    enable_ai_generation: bool = Form(True),
    enable_local_library: bool = Form(True),
    enable_stock_providers: bool = Form(True),
    script_file: UploadFile | None = File(None),
    audio_file: UploadFile | None = File(None),
):
    """multipart/form-data counterpart to /generate-documentary-timeline, for
    the frontend's file pickers (upload a script .txt and/or a voiceover
    recording). Saves whatever files were sent to a per-request scratch
    directory, then runs documentary_pipeline.run() — the same single entry
    point the JSON route uses — with script_path/audio_path pointing at them."""
    return await _generate_documentary_timeline_upload(
        documentary_pipeline.run, background_tasks, script, project_name,
        allow_duplicate_assets, max_downloads, content_niche_list or content_niche, script_file, audio_file,
        enable_ai_generation, enable_local_library, enable_stock_providers,
    )


@app.post("/generate-documentary-timeline/edit/upload")
async def generate_documentary_timeline_edit_upload(
    background_tasks: BackgroundTasks,
    script: str = Form(""),
    project_name: str | None = Form(None),
    allow_duplicate_assets: bool = Form(False),
    max_downloads: int | None = Form(None),
    content_niche: str = Form(DEFAULT_NICHE),
    content_niche_list: list[str] | None = Form(None),
    enable_ai_generation: bool = Form(True),
    enable_local_library: bool = Form(True),
    enable_stock_providers: bool = Form(True),
    script_file: UploadFile | None = File(None),
    audio_file: UploadFile | None = File(None),
):
    """multipart/form-data counterpart to /generate-documentary-timeline/edit
    — see that route's docstring for what "Generate & Edit" mode skips."""
    return await _generate_documentary_timeline_upload(
        documentary_pipeline.generate_and_edit, background_tasks, script, project_name,
        allow_duplicate_assets, max_downloads, content_niche_list or content_niche, script_file, audio_file,
        enable_ai_generation, enable_local_library, enable_stock_providers,
    )


@app.get("/progress/{project_name}")
async def get_progress(project_name: str):
    entry = progress.get(project_name)
    if entry is None:
        raise HTTPException(status_code=404, detail="unknown project_name")
    return entry


@app.get("/editor/{project_name}", response_class=HTMLResponse)
async def editor(project_name: str):
    """Post-generation timeline editor — fast per-clip swap/regenerate/upload
    editing over an already-generated project's timeline.json, with a
    separate "Generate Full Video" button for the slow Remotion+audio-mux
    pass (see documentary_pipeline.rerender_single_clip / generate_full_video)."""
    html_path = Path(__file__).resolve().parent.parent / "templates" / "editor.html"
    return html_path.read_text(encoding="utf-8")


def _project_file_url(project_dir: Path, project_name: str, path: str | None) -> str | None:
    """Converts an absolute on-disk path under project_dir into a URL the
    browser can actually fetch, via the /project-files static mount above.
    None for anything outside project_dir (shouldn't happen for pipeline
    output, but never leaks a raw filesystem path to the frontend either way)."""
    if not path:
        return None
    try:
        rel = Path(path).resolve().relative_to(project_dir.resolve()).as_posix()
    except ValueError:
        return None
    return f"/project-files/{project_name}/{rel}"


@app.get("/project/{project_name}")
async def get_project(project_name: str):
    """Loads an already-generated project's timeline for the editor —
    timeline.json is otherwise write-only (see documentary_pipeline.run)."""
    project_dir = settings.documentary_projects_dir / project_name
    try:
        timeline = documentary_pipeline._load_timeline(project_dir)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        meta = documentary_pipeline._load_project_meta(project_dir)
        table = meta.table
        content_niche = meta.content_niche
        stable_audio_path = meta.audio_path
    except ValueError:
        table = []  # project_meta.json predates this feature or is missing — editor still works, just no AI-regenerate
        content_niche = None
        stable_audio_path = None

    entries = []
    for entry in sorted(timeline, key=lambda e: e.clip_number):
        data = entry.model_dump()
        data["asset_url"] = _project_file_url(project_dir, project_name, entry.asset_path)
        data["rendered_url"] = _project_file_url(project_dir, project_name, entry.rendered_clip_path)
        entries.append(data)

    def final_url(filename: str) -> str | None:
        path = project_dir / filename
        return _project_file_url(project_dir, project_name, str(path)) if path.exists() else None

    # Preview audio (see the editor's parallel <audio> track): prefer the
    # stable narration copy persisted alongside project_meta.json, since it's
    # the raw audio with no video-container overhead. Older projects (from
    # before that copy existed) fall back to the narrated video file — a
    # browser <audio> element can play just the audio track of a video file.
    narration_filename = documentary_pipeline.final_video_filename(project_dir)
    audio_url = _project_file_url(project_dir, project_name, stable_audio_path) if stable_audio_path else None
    if not audio_url:
        audio_url = final_url(narration_filename)

    return {
        "project_name": project_name,
        "content_niche": content_niche,
        "table": [c.model_dump() for c in table],
        "timeline": entries,
        "audio_url": audio_url,
        "final_video_url": final_url("final_video.mp4"),
        "subtitled_video_url": final_url("final_video_remotion.mp4"),
        "final_video_with_narration_url": final_url(narration_filename),
        "captions_url": final_url("captions.srt"),
    }


class ClipEditRequest(BaseModel):
    mode: Literal["alternate", "ai"]
    alternate_index: int | None = None


@app.patch("/project/{project_name}/clip/{clip_number}", response_model=TimelineEntry)
async def rerender_clip(project_name: str, clip_number: int, req: ClipEditRequest):
    """Tier-1 fast edit: swap to a persisted alternate, or regenerate via AI.
    File uploads go through the multipart /upload route below instead —
    FastAPI can't mix a JSON body and File/Form params on one route."""
    try:
        if req.mode == "alternate":
            if req.alternate_index is None:
                raise ValueError("alternate_index is required when mode='alternate'")
            entry = await documentary_pipeline.rerender_single_clip(
                project_name, clip_number, alternate_index=req.alternate_index,
            )
        else:
            entry = await documentary_pipeline.rerender_single_clip(
                project_name, clip_number, ai_regenerate=True,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("rerender_single_clip failed for project %s clip %d", project_name, clip_number)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return entry


@app.patch("/project/{project_name}/clip/{clip_number}/upload", response_model=TimelineEntry)
async def rerender_clip_upload(project_name: str, clip_number: int, file: UploadFile = File(...)):
    upload_dir = settings.uploads_dir / uuid.uuid4().hex
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / (file.filename or "upload")
    upload_path.write_bytes(await file.read())

    try:
        entry = await documentary_pipeline.rerender_single_clip(
            project_name, clip_number, upload_path=str(upload_path),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("rerender_single_clip (upload) failed for project %s clip %d", project_name, clip_number)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return entry


async def _run_full_video_tracked(project_name: str) -> None:
    try:
        result = await documentary_pipeline.generate_full_video(project_name)
        progress.finish(project_name, result.model_dump())
    except Exception as exc:
        logger.exception("documentary_pipeline.generate_full_video failed for project %s", project_name)
        progress.fail(project_name, str(exc))


@app.post("/project/{project_name}/generate-full-video")
async def generate_full_video_route(project_name: str, background_tasks: BackgroundTasks):
    """Tier-2 full render: Remotion composite -> concat -> audio mux over the
    current (possibly Tier-1-edited) timeline.json. Same background-task +
    progress-polling shape as /generate-documentary-timeline, so the editor
    reuses the frontend's existing pollProgress()."""
    progress.start(project_name)
    background_tasks.add_task(_run_full_video_tracked, project_name)
    return {"project_name": project_name}


@app.post("/clips/ingest")
async def ingest_clip(req: IngestRequest):
    clip = analyze_and_store(req.video_path, req.start, req.end, source="local")
    return clip


@app.get("/clips/search")
async def search_clips(query: str, top_k: int = 5):
    matches = query_similar(query, top_k=top_k)
    return [{"clip": clip, "similarity": similarity} for clip, similarity in matches]
