import logging
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    groq_api_key: str = ""
    pexels_api_key: str = ""
    pixabay_api_key: str = ""

    similarity_threshold: float = 0.75
    chroma_dir: Path = Path("./data/chroma")
    clips_dir: Path = Path("./clips")

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    llm_text_model: str = "llama-3.3-70b-versatile"
    llm_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    # Accuracy over the ~2.8x cheaper "whisper-large-v3-turbo" — these
    # timestamps now drive real clip durations (app/documentary_table.py),
    # not just cosmetic captions, so transcription accuracy matters more here.
    llm_whisper_model: str = "whisper-large-v3"
    llm_max_requests_per_minute: int = 30  # Groq free tier: 30 RPM on both models above

    documentary_projects_dir: Path = Path("./projects")
    documentary_high_quality_score: float = 90
    documentary_min_score: float = 50
    documentary_niche_min_assets: dict[str, dict[str, int]] = {
        "historical": {"video": 3, "image": 3},
        "modern": {"video": 1, "image": 1},
        "general": {"video": 2, "image": 2},
    }
    documentary_max_downloads_per_project: int | None = None

    # ponytail: scratch space for full-download candidate comparison (see
    # asset_selection.py) — every stock candidate lands here in full before
    # scoring, and everything but the winner gets deleted after selection.
    candidate_scratch_dir: Path = Path("./.cache/candidates")

    # Final-render stage: Remotion project (see remotion/), invoked via
    # `node render.mjs` from app/remotion_render.py.
    remotion_dir: Path = Path("./remotion")
    remotion_render_timeout_seconds: int = 900

    # Script/audio files uploaded via the frontend's file pickers (see
    # app/main.py's /generate-documentary-timeline/upload) land here, one
    # subdirectory per request — documentary_pipeline.run() reads them via
    # its existing script_path/audio_path params, nothing else changes.
    uploads_dir: Path = Path("./uploads")

    candidate_pool_size: int = 8
    asset_selection_score_margin: float = 0.05
    asset_selection_tie_epsilon: float = 0.01

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
