from typing import Literal

from pydantic import BaseModel


class Scene(BaseModel):
    scene: int
    text: str


class SceneAnalysis(BaseModel):
    description: str
    keywords: list[str] = []
    objects: list[str] = []
    actions: list[str] = []
    mood: str = ""
    camera: str = ""


class ClipRecord(BaseModel):
    id: str
    video_path: str  # holds an image file path too when media_type == "image"
    start: float
    end: float
    duration: float
    description: str
    keywords: list[str] = []
    resolution: str = ""
    source: Literal["local", "pexels", "pixabay", "pexels_images", "pixabay_images"]
    source_id: str = ""
    fingerprint: str = ""
    media_type: Literal["video", "image"] = "video"


class SceneResult(BaseModel):
    scene: int
    script: str
    selected_clip: str | None
    similarity: float | None = None
    source: Literal["local", "pexels", "pixabay", "pexels_images", "pixabay_images", "none"]
    start: float | None = None
    end: float | None = None
    downloaded: bool = False
    cached: bool = False
    # NEW: multi-candidate selection extends the response with what was picked
    # and why, rather than silently taking the highest-resolution hit.
    media_type: Literal["video", "image"] | None = None
    selection_reason: str = ""


VisualType = Literal[
    "Archive", "Cinematic", "Sports", "Exercise Demo", "Documents", "Laboratory",
    "B-roll", "Animation", "Map", "Military", "Emotion", "Industry", "Historical",
    "Interview", "Montage", "Nature", "Technology", "Combat",
]


class TimelineClip(BaseModel):
    clip_number: int
    start: str = ""
    end: str = ""
    section: str
    script_beat: str
    canva_keyword: str
    fallback_keyword: str
    visual_type: VisualType
    edit_note: str


class ClipAvailability(BaseModel):
    clip_number: int
    section: str
    # Acceptable (score >= documentary_min_score) candidates found using
    # canva/fallback keywords alone — see
    # documentary_pipeline.check_footage_availability.
    candidate_count: int
    top_score: float | None = None
    thin: bool


class AvailabilityReport(BaseModel):
    """Pre-render footage availability scan result (see
    documentary_pipeline.check_footage_availability) — a cheap approximation
    of how the real resolve loop will fare, not a guarantee: it never
    downloads anything, so it can't know about post-download rejections
    (16:9 normalization failure, CLIP score miss) the real run might still
    hit even on a candidate this scan counted as acceptable."""
    total_clips: int
    thin_count: int
    thin_clip_numbers: list[int]
    clips: list[ClipAvailability]


class AlternateCandidate(BaseModel):
    """A ranked-but-not-chosen search candidate for a clip, persisted so the
    editor's "swap to alternate" feature has something to show without
    re-running search. Deliberately lighter than AssetInfo (no local `path`)
    — these are never downloaded unless picked; the editor previews them
    straight from `download_url`, and only a real swap (see
    documentary_pipeline.rerender_single_clip) downloads one for real."""
    source: str
    source_id: str
    download_url: str
    media_type: Literal["video", "image"]
    width: int = 0
    height: int = 0
    duration: float = 0
    text: str = ""
    score: float


class AssetInfo(BaseModel):
    path: str
    source: str
    source_id: str
    media_type: Literal["video", "image"]
    width: int = 0
    height: int = 0
    duration: float = 0
    # None only for AI-generated fallback assets (source == "ai_generated") —
    # there's no real candidate score to report since nothing was searched
    # or ranked, see app.stock.ai_generated.
    score: float | None
    # The winning StockHit's own text (tags/title/alt) at accept time — same
    # text CLIP verification and niche filtering already judged the candidate
    # against. Persisted so match-quality review never again requires
    # reconstructing it after the fact via per-ID provider lookups.
    selected_text: str = ""


class TimelineEntry(BaseModel):
    clip_number: int
    start: str
    end: str
    duration: float
    script: str
    asset_path: str | None
    asset_metadata: AssetInfo | None
    # NEW: a second, genuinely distinct asset to cover the remainder of this
    # row's slot when asset_path's own real duration is too short for a
    # natural-looking speed adjustment alone (see
    # documentary_pipeline._resolve_clip's fill-candidate search and
    # documentary_assembly._render_row_segment's "split-fill" status) — never
    # looped/repeated to fill a gap. None when no second distinct candidate
    # was available (the row falls back to a forced speed adjustment instead).
    fill_asset_path: str | None = None
    fill_asset_metadata: AssetInfo | None = None
    alternates: list[AlternateCandidate] = []
    recommended_effect: str
    transition: str
    placeholder: bool = False
    note: str = ""
    # NEW: populated by the assembly stage (app/documentary_assembly.py) after
    # each row's source asset is normalized to exactly `duration` seconds.
    # "looped-to-fill" is kept only so old persisted timeline.json/project_meta.json
    # files (from before this status was retired) still load — current code
    # never produces it again; a short clip now gets "split-fill" (a second
    # distinct asset covers the remainder) or "speed-matched" (forced, no
    # second candidate available) instead. See documentary_assembly.py.
    status: Literal["ok", "trimmed", "looped-to-fill", "speed-matched", "split-fill", "missing"] = "ok"
    rendered_clip_path: str | None = None
    # NEW: distinguishes "real candidate accepted directly (score >= trigger
    # threshold, generation never attempted)" from "generation was attempted
    # (score < threshold) but failed, falling back to this real candidate" —
    # both previously looked identical once persisted (asset_metadata.source
    # is the real candidate's source either way). See
    # documentary_pipeline._resolve_clip / render_keyword_match_report.
    generation_attempted: bool = False
    generation_failure_reason: str | None = None


class DocumentaryResult(BaseModel):
    project_dir: str
    table: list[TimelineClip]
    timeline: list[TimelineEntry]
    # NEW: set once assembly succeeds; assembly_error is set instead (never
    # both) if the pre-render contiguity check or ffmpeg concat failed.
    final_video_path: str | None = None
    assembly_error: str | None = None
    # NEW: caption burn-in, only attempted once final_video_path exists.
    # Cue timing is a word-count/pace estimate (no TTS narration audio exists
    # in this pipeline) — see README's design note.
    subtitles_path: str | None = None
    subtitled_video_path: str | None = None
    subtitle_error: str | None = None
    # NEW: the LLM-driven style decision (pacing/transition/caption emphasis)
    # fed into the Remotion render — surfaced here for debugging, in addition
    # to being logged by app.style_decision.decide_style.
    style_decision: dict | None = None
    # NEW: set only when audio_path was supplied and the manually-provided
    # script deviates significantly from Whisper's actual transcription (see
    # app.transcription.check_alignment) — a sanity-check signal only, never
    # blocks the pipeline and never changes what timing/text gets used.
    transcript_alignment_warning: str | None = None
    # NEW: only produced when audio_path was supplied — subtitled_video_path
    # above is Remotion's own output, which has no audio track at all;
    # this is that same video with the full, unmodified uploaded narration
    # muxed in (see app/audio_mux.py). This is the actual final deliverable
    # whenever manual narration audio was provided.
    final_video_with_narration_path: str | None = None
    audio_mux_error: str | None = None
    # NEW: pre-render footage availability scan (see
    # documentary_pipeline.check_footage_availability), run once before any
    # per-clip download/CLIP-verify/render work begins.
    footage_availability: AvailabilityReport | None = None


class ProjectMeta(BaseModel):
    """Inputs the original run() had in memory but never wrote to disk —
    persisted so a later, editor-triggered re-render (see
    documentary_pipeline.generate_full_video / rerender_single_clip) can redo
    the table-aware parts of the pipeline (AI-regeneration prompts, style
    decision, captions) without re-running script->table LLM generation or
    Whisper transcription. whisper_words is stored as plain dicts
    (text/start/end) rather than WordTiming to avoid models.py importing
    app.subtitles, which itself imports models.py."""
    script: str
    content_niche: str | list[str]
    audio_path: str | None = None
    whisper_words: list[dict] = []
    transcript_alignment_warning: str | None = None
    table: list[TimelineClip] = []
    footage_availability: AvailabilityReport | None = None


class DocumentaryPreview(BaseModel):
    """Response for POST /generate-documentary-timeline/preview — the same
    table + availability scan a real run() would produce, without any
    download/CLIP-verify/render work, so a thin/bad batch can be caught
    before committing to the expensive part."""
    project_name: str
    table: list[TimelineClip]
    footage_availability: AvailabilityReport
