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


class AssetInfo(BaseModel):
    path: str
    source: str
    source_id: str
    media_type: Literal["video", "image"]
    width: int = 0
    height: int = 0
    duration: float = 0
    score: float


class TimelineEntry(BaseModel):
    clip_number: int
    start: str
    end: str
    duration: float
    script: str
    asset_path: str | None
    asset_metadata: AssetInfo | None
    alternates: list[AssetInfo] = []
    recommended_effect: str
    transition: str
    placeholder: bool = False
    note: str = ""
    # NEW: populated by the assembly stage (app/documentary_assembly.py) after
    # each row's source asset is normalized to exactly `duration` seconds.
    status: Literal["ok", "trimmed", "looped-to-fill", "speed-matched", "missing"] = "ok"
    rendered_clip_path: str | None = None


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
    # NEW: the Groq-driven style decision (pacing/transition/caption emphasis)
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
