import logging
from typing import Literal

from pydantic import BaseModel

from app.config import settings
from app.llm_client import generate_json

logger = logging.getLogger(__name__)

Pacing = Literal["fast", "medium", "slow"]
TransitionStyle = Literal["fade", "slide", "zoom", "hard_cut"]

_VALID_PACING = {"fast", "medium", "slow"}
_VALID_TRANSITIONS = {"fade", "slide", "zoom", "hard_cut"}
_DEFAULT_PACING: Pacing = "medium"
_DEFAULT_TRANSITION: TransitionStyle = "fade"

STYLE_PROMPT = """Analyze this video script and decide a visual editing style for it. \
Return a JSON object with exactly these keys:
"pacing": one of "fast", "medium", "slow" - how quickly cuts and captions should feel.
"transition_style": one of "fade", "slide", "zoom", "hard_cut" - the transition \
between clips that best fits this script's tone.
"caption_emphasis": an array of 5-15 individual words (exact words as they appear \
in the script, not phrases) that are the most important or dramatic and should be \
visually emphasized in captions.

Script:
\"\"\"{script}\"\"\"
"""


class StyleDecision(BaseModel):
    pacing: Pacing = _DEFAULT_PACING
    transition_style: TransitionStyle = _DEFAULT_TRANSITION
    caption_emphasis: list[str] = []


def _normalize(data: dict) -> StyleDecision:
    # ponytail: json_object mode doesn't enforce an enum (see
    # documentary_table.assign_timings for the same issue with visual_type)
    # — normalize instead of trusting the raw value.
    pacing = data.get("pacing")
    if pacing not in _VALID_PACING:
        pacing = _DEFAULT_PACING

    transition_style = data.get("transition_style")
    if transition_style not in _VALID_TRANSITIONS:
        transition_style = _DEFAULT_TRANSITION

    emphasis = data.get("caption_emphasis", [])
    if not isinstance(emphasis, list):
        emphasis = []
    emphasis = [str(w) for w in emphasis if isinstance(w, (str, int, float))][:15]

    return StyleDecision(pacing=pacing, transition_style=transition_style, caption_emphasis=emphasis)


def decide_style(script: str) -> StyleDecision:
    """LLM-driven pacing/transition/caption-emphasis decision fed into the
    Remotion final render. Never raises: any failure (network error, bad
    JSON, wrong types/values) falls back to hardcoded defaults so a style
    problem can never block or fail the render itself."""
    try:
        data = generate_json(STYLE_PROMPT.format(script=script), settings.llm_text_model)
        decision = _normalize(data)
    except Exception as exc:
        logger.warning("style_decision: LLM call failed or returned invalid JSON (%s) — using defaults", exc)
        decision = StyleDecision()

    logger.info(
        "style_decision: pacing=%s transition_style=%s caption_emphasis=%s",
        decision.pacing, decision.transition_style, decision.caption_emphasis,
    )
    return decision
