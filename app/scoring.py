from app.embeddings import embed
from app.models import TimelineClip
from app.stock.base import StockHit

_ARCHIVE_SOURCES = {"internet_archive", "wikimedia", "nasa"}
_HISTORICAL_VISUAL_TYPES = {"Archive", "Historical", "Military", "Documents"}

_WEIGHTS = {
    "semantic": 0.40,
    "historical": 0.20,
    "quality": 0.15,
    "cinematic": 0.15,
    "motion": 0.10,
}


def query_embedding_for(clip: TimelineClip) -> list[float]:
    return embed(f"{clip.canva_keyword} {clip.script_beat}")


def _semantic_similarity(hit: StockHit, query_embedding: list[float]) -> float:
    if not hit.text:
        return 0.5
    hit_embedding = embed(hit.text)
    similarity = sum(a * b for a, b in zip(query_embedding, hit_embedding))
    return max(0.0, min(1.0, similarity))


def _historical_accuracy(hit: StockHit, clip: TimelineClip) -> float:
    is_archive_source = hit.source in _ARCHIVE_SOURCES
    is_historical_clip = clip.visual_type in _HISTORICAL_VISUAL_TYPES
    if is_archive_source and is_historical_clip:
        return 1.0
    if is_archive_source or is_historical_clip:
        return 0.6
    return 0.5


def _visual_quality(hit: StockHit) -> float:
    return min(1.0, (hit.width * hit.height) / (1920 * 1080))


def _cinematic_value(hit: StockHit) -> float:
    base = 0.8 if hit.media_type == "video" else 0.4
    return min(1.0, base + 0.2 * _visual_quality(hit))


def _motion_potential(hit: StockHit) -> float:
    return 1.0 if hit.media_type == "video" else 0.0


# ponytail: heuristic scorer using metadata already returned by the provider
# APIs, not a real visual-content model — upgrade to CLIP-style image/query
# similarity if ranking quality proves insufficient in practice.
def score_asset(hit: StockHit, clip: TimelineClip, query_embedding: list[float]) -> float:
    total = (
        _WEIGHTS["semantic"] * _semantic_similarity(hit, query_embedding)
        + _WEIGHTS["historical"] * _historical_accuracy(hit, clip)
        + _WEIGHTS["quality"] * _visual_quality(hit)
        + _WEIGHTS["cinematic"] * _cinematic_value(hit)
        + _WEIGHTS["motion"] * _motion_potential(hit)
    )
    return round(total * 100, 2)
