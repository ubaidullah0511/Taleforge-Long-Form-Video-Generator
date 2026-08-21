"""In-memory per-project pipeline progress, polled by the frontend via
GET /progress/{project_name} (see app/main.py). A plain module-level dict is
enough for this single-process local tool.
# ponytail: process-local dict, lost on restart and not shared across
# workers — move to Redis/a DB if this ever runs with uvicorn --workers > 1.

Context (which project/clip a progress update belongs to) is carried via
contextvars instead of threading project_name/clip_index/total_clips
parameters through every function in the call chain (search_clip,
_resolve_clip, etc.) — the whole per-clip pipeline runs as one continuous
sequential coroutine chain (see documentary_pipeline.run's "Sequential per
clip" note), so a plain contextvars.ContextVar set once per clip iteration
is visible to everything called from within that iteration.
"""
import contextvars
import time

_progress: dict[str, dict] = {}

_current_project: contextvars.ContextVar[str | None] = contextvars.ContextVar("_current_project", default=None)
_current_clip: contextvars.ContextVar[tuple[int, int] | None] = contextvars.ContextVar("_current_clip", default=None)


def set_context(project_name: str | None, clip_index: int | None = None, total_clips: int | None = None) -> None:
    _current_project.set(project_name)
    _current_clip.set((clip_index, total_clips) if clip_index is not None else None)


def start(project_name: str) -> None:
    _progress[project_name] = {
        "project_name": project_name, "stage": "input", "detail": "Starting",
        "current": None, "total": None, "done": False, "error": None, "result": None,
        "updated_at": time.time(),
    }


def update(stage: str, detail: str) -> None:
    """No-op if called with no project in context (e.g. a test calling
    documentary_pipeline.run() directly, bypassing the /generate-* endpoints
    that call set_context) — there's nowhere to record the update."""
    project_name = _current_project.get()
    if not project_name:
        return
    current, total = _current_clip.get() or (None, None)
    entry = _progress.setdefault(project_name, {"done": False, "error": None, "result": None})
    entry.update(project_name=project_name, stage=stage, detail=detail, current=current, total=total, updated_at=time.time())


def finish(project_name: str, result: dict) -> None:
    entry = _progress.setdefault(project_name, {})
    entry.update(
        project_name=project_name, stage="timeline", detail="Complete",
        done=True, error=None, result=result, updated_at=time.time(),
    )


def fail(project_name: str, error: str) -> None:
    entry = _progress.setdefault(project_name, {})
    entry.update(project_name=project_name, done=True, error=error, updated_at=time.time())


def get(project_name: str) -> dict | None:
    return _progress.get(project_name)
