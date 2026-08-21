import logging
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    openai_api_key: str = ""
    pexels_api_key: str = ""
    pixabay_api_key: str = ""
    coverr_api_key: str = ""

    similarity_threshold: float = 0.75
    chroma_dir: Path = Path("./data/chroma")
    clips_dir: Path = Path("./clips")

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    llm_text_model: str = "gpt-4o-mini"
    llm_vision_model: str = "gpt-4o-mini"  # gpt-4o-mini also accepts image_url content, no separate vision model needed
    # OpenAI's real rate limits depend on your account/usage tier — this is a
    # conservative starting point, raise it if your tier allows more.
    llm_max_requests_per_minute: int = 30
    # Local (offline, no API key) Whisper model size — see openai-whisper's
    # model card for size/speed/accuracy tradeoffs. These timestamps drive
    # real per-clip durations (app/documentary_table.py), not just captions,
    # so bumping this up (small/medium/large) trades CPU time for accuracy.
    whisper_model_size: str = "base"

    documentary_projects_dir: Path = Path("./projects")
    documentary_high_quality_score: float = 90
    documentary_min_score: float = 50
    # Project-wide running quota (see documentary_pipeline._media_type_bias) —
    # a soft ranking nudge toward this image/video mix across a whole project,
    # not a per-clip rule. Never excludes a candidate, only breaks near-ties.
    documentary_target_image_ratio: float = 0.65
    documentary_niche_min_assets: dict[str, dict[str, int]] = {
        "historical": {"video": 3, "image": 3},
        "modern": {"video": 1, "image": 1},
        "general": {"video": 2, "image": 2},
    }
    documentary_max_downloads_per_project: int | None = None

    # Post-download pixel-content check (see app/visual_verification.py) —
    # text metadata alone (banned_terms/positive_terms) can't catch a
    # candidate whose tags lie about what's actually in frame. Threshold is
    # a cosine similarity, not a percentage — CLIP scores run lower than
    # intuition suggests.
    #
    # Recalibrated 2026-07-23 (was 0.26): the original 0.26 was set against
    # scene_description = canva_keyword + full script_beat (diluted/truncated
    # narration prose), which a real pipeline run confirmed was dragging
    # genuinely correct matches down into the 0.20-0.27 range — three real
    # analytical-balance/scale videos for a "weighing pool chemicals" beat
    # scored 0.227-0.236 and were all wrongly rejected. After fixing the text
    # side to pass canva_keyword alone (see documentary_pipeline._resolve_clip),
    # a controlled real-CLIP A/B (same real candidates, both text variants)
    # showed: confirmed-good topical matches now score 0.220-0.267, while a
    # clearly off-topic negative control (ocean/beach footage against
    # pool-chemical queries) scores 0.095-0.161 — a real, if modest-sample,
    # separation. 0.20 sits with margin below the observed good-match floor
    # and above the observed bad-match ceiling. Revisit with a larger sample
    # if either boundary turns out to be too tight in production.
    enable_visual_verification: bool = True
    visual_verification_threshold: float = 0.20

    # ponytail: scratch space for full-download candidate comparison (see
    # asset_selection.py) — every stock candidate lands here in full before
    # scoring, and everything but the winner gets deleted after selection.
    candidate_scratch_dir: Path = Path("./.cache/candidates")

    # Final-render stage: Remotion project (see remotion/), invoked via
    # `node render.mjs` from app/remotion_render.py.
    remotion_dir: Path = Path("./remotion")
    remotion_render_timeout_seconds: int = 2400

    # Script/audio files uploaded via the frontend's file pickers (see
    # app/main.py's /generate-documentary-timeline/upload) land here, one
    # subdirectory per request — documentary_pipeline.run() reads them via
    # its existing script_path/audio_path params, nothing else changes.
    uploads_dir: Path = Path("./uploads")

    candidate_pool_size: int = 8
    asset_selection_score_margin: float = 0.05
    asset_selection_tie_epsilon: float = 0.01

    # Cross-clip anti-fatigue cap (see documentary_pipeline.search_clip's
    # used_assets dedup) — the same stock asset may win for up to this many
    # clips in one video before it's excluded and search falls through to the
    # next-best alternative. allow_duplicates=True still bypasses this
    # entirely (no cap at all) for the rare explicit override.
    #
    # Set to 1 (true never-repeat) 2026-07-24: a preview-endpoint scan
    # (check_footage_availability) on a 37-clip pool_maintenance script
    # showed thin_count=0/37 under both cap=1 and cap=2 post-CLIP-fix — the
    # candidate pool has enough real depth that the stricter cap costs
    # nothing pre-download. Verified with a real full pipeline run too (see
    # handoff.md for the placeholder-count/asset-reuse evidence). Revert to 2
    # if a future niche/script shows thin_count rising under cap=1.
    max_asset_repeat_count: int = 1

    # Pre-render footage availability scan (see
    # documentary_pipeline.check_footage_availability) — a clip counts as
    # "thin" when its acceptable-candidate count (canva/fallback keywords
    # only, no semantic-keyword LLM fallback, no download/CLIP-verify) falls
    # below this floor. Looser than the real per-media-type
    # documentary_niche_min_assets requirement on purpose: this scan is meant
    # to catch "would become a placeholder" risk cheaply, not to reproduce
    # the real resolve loop's full escalation.
    footage_availability_min_candidates: int = 1
    # Aggregate warning threshold: fraction of clips flagged thin before
    # run() logs a prominent "LOW FOOTAGE AVAILABILITY" warning.
    footage_availability_warn_ratio: float = 0.15

    # AI-generated fallback (see app/stock/openai_image.py — OpenAI
    # gpt-image-1-mini) — fires whenever the BEST real candidate search
    # found for a clip scores below ai_generation_trigger_threshold, not
    # only when real search comes up completely empty. This is a QUALITY
    # gate, not a last resort: real diagnostic data on this pipeline shows
    # most clips' best real candidate scores land in the 58-75 range, so at
    # threshold=75 most clips in a typical video are expected to become
    # AI-generated images rather than real stock footage once this is
    # enabled — confirm the actual ratio with a real run rather than
    # assuming it, see handoff.md.
    #
    # Defaults to OFF on purpose: since this can affect the MAJORITY of a
    # video's visual content rather than a handful of gap-fill clips, verify
    # output quality with a real run before flipping this to True for
    # monetized production.
    enable_ai_generation_fallback: bool = True
    ai_generation_trigger_threshold: float = 75.0
    ai_generation_model: str = "gpt-image-2"
    ai_generation_quality: str = "medium"  # "low" is the cheapest tier

    @property
    def local_clips_dir(self) -> Path:
        return self.clips_dir / "local"

    @property
    def downloaded_clips_dir(self) -> Path:
        return self.clips_dir / "downloaded"


settings = Settings()
settings.chroma_dir.mkdir(parents=True, exist_ok=True)
settings.local_clips_dir.mkdir(parents=True, exist_ok=True)
settings.downloaded_clips_dir.mkdir(parents=True, exist_ok=True)
settings.documentary_projects_dir.mkdir(parents=True, exist_ok=True)
settings.candidate_scratch_dir.mkdir(parents=True, exist_ok=True)  
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
 