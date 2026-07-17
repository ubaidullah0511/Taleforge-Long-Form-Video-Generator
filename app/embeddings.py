from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


@lru_cache(maxsize=4096)
def embed(text: str) -> list[float]:
    return _model().encode(text, normalize_embeddings=True).tolist()
