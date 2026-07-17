from typing import Literal

from app.config import settings
from app.llm_client import generate_json, generate_json_from_images
from app.models import SceneAnalysis

PROMPT = """Analyze this video script scene for B-roll footage selection. \
Return a JSON object with keys: "description" (a detailed visual description \
of what B-roll footage should show), "keywords" (list of 5-8 search terms), \
"objects" (list of visible objects/subjects), "actions" (list of actions/motion \
happening in the shot), "mood" (one or two words), "camera" (suggested camera \
angle/movement, e.g. "cinematic drone shot").

Scene: \"\"\"{text}\"\"\"
"""

MEDIA_TYPE_PROMPT = """Does this documentary/video script scene primarily need \
MOTION footage (video showing action, movement, or change over time) or would \
a STATIC image suffice (a portrait, a document, a landscape, a concept shot)? \
Return a JSON object with key "media_type" set to exactly "video" or "image".

Scene: \"\"\"{text}\"\"\"
"""

# Shared schema with clip_ingest.ANALYSIS_PROMPT so a candidate's description,
# generated here during selection, can be reused as-is at ingest time instead
# of re-analyzing the same asset a second time.
CANDIDATE_ANALYSIS_PROMPT = """You are evaluating a stock B-roll candidate for a \
video script scene. {frame_note}

Scene this candidate might be used for: \"\"\"{scene_text}\"\"\"

Return a JSON object with keys: "description" (detailed visual description of \
what is shown), "keywords" (list of 5-8 search terms describing the visual \
content), "objects" (list of visible objects/subjects), "camera" (camera \
angle/movement, or "static" for a still image), "motion" (description of any \
movement across the provided frames, or "none" for a still image), \
"scene_type" (e.g. "nature", "urban", "office"), "mood" (one or two words).
"""


def analyze(text: str) -> SceneAnalysis:
    data = generate_json(PROMPT.format(text=text), settings.llm_text_model)
    return SceneAnalysis(**data)


def decide_media_type(text: str) -> Literal["video", "image"]:
    data = generate_json(MEDIA_TYPE_PROMPT.format(text=text), settings.llm_text_model)
    media_type = data.get("media_type", "video")
    return media_type if media_type in ("video", "image") else "video"


def describe_candidate(scene_text: str, images: list[bytes]) -> dict:
    """Vision-describe a candidate asset from 1 (still image) or 2-3 (sampled
    video frames: start/middle/end) images, for both scoring and ingest reuse."""
    frame_note = (
        "You are shown a single still image."
        if len(images) == 1
        else f"You are shown {len(images)} frames sampled across the clip "
             "(start, middle, end) — describe the motion/change across them, not just one frame."
    )
    prompt = CANDIDATE_ANALYSIS_PROMPT.format(frame_note=frame_note, scene_text=scene_text)
    return generate_json_from_images(prompt, images, settings.llm_vision_model)
