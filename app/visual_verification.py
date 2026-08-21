import logging
from functools import lru_cache

import clip
import cv2
import torch
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_NAME = "ViT-B/32"  # Small, fast, fits comfortably on a 4GB GPU


@lru_cache(maxsize=1)
def _load_clip_model():
    logger.info("visual_verification: loading CLIP model '%s' on %s", _MODEL_NAME, _DEVICE)
    model, preprocess = clip.load(_MODEL_NAME, device=_DEVICE)
    return model, preprocess


def _extract_sample_frame(video_path: str) -> Image.Image | None:
    """Grabs a frame from the middle of the clip (avoids black-frame fades
    at the very start/end that could give a misleading null match)."""
    cap = cv2.VideoCapture(video_path)
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            logger.warning("visual_verification: could not read frame count for %s", video_path)
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        success, frame = cap.read()
        if not success:
            logger.warning("visual_verification: failed to read middle frame from %s", video_path)
            return None
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()


def visual_match_score(video_path: str, visual_query: str) -> float | None:
    """Returns a 0.0-1.0 cosine similarity between visual_query and a sampled
    frame from the candidate video. Returns None if the frame couldn't be
    extracted (caller should treat this as 'unverifiable, use existing
    text-based judgment' rather than an automatic rejection).

    visual_query must be a SHORT, CONCRETE visual phrase (a few words), not
    narration prose — confirmed via a real pipeline run that passing full
    script_beat text (20-40 words of dosage amounts/chemistry explanation
    with little directly visual content) dilutes/truncates past CLIP's
    77-token limit and drags genuinely correct matches below threshold (see
    app.documentary_pipeline._resolve_clip's call site, which passes
    clip.canva_keyword alone for exactly this reason)."""
    frame = _extract_sample_frame(video_path)
    if frame is None:
        return None

    logger.debug(
        "visual_verification: comparing frame from %s against text: %r (length: %d chars, ~%d words)",
        video_path, visual_query, len(visual_query), len(visual_query.split()),
    )

    model, preprocess = _load_clip_model()
    image_input = preprocess(frame).unsqueeze(0).to(_DEVICE)
    text_input = clip.tokenize([visual_query], truncate=True).to(_DEVICE)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
        similarity = torch.cosine_similarity(image_features, text_features).item()

    logger.debug(
        "visual_verification: %s vs '%s' -> similarity=%.3f",
        video_path, visual_query, similarity,
    )
    return similarity


def passes_visual_verification(video_path: str, visual_query: str) -> tuple[bool, float | None]:
    """Returns (passed, score). If score is None (frame extraction failed),
    passed defaults to True — we don't want a corrupted/unreadable video file
    to be treated as a content mismatch; that's a different failure mode
    handled elsewhere (asset download validation). Reads the threshold from
    settings at call time (not a module-level constant) so
    settings.visual_verification_threshold can be tuned/tested without a
    reload, matching how the rest of the pipeline reads settings.*
    directly (e.g. documentary_min_score in rank_acceptable_assets)."""
    if not settings.enable_visual_verification:
        return True, None
    score = visual_match_score(video_path, visual_query)
    if score is None:
        return True, None
    return score >= settings.visual_verification_threshold, score
