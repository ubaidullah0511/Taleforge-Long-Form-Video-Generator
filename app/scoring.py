import math
import re

from wordfreq import word_frequency

from app.embeddings import embed
from app.models import TimelineClip
from app.stock.base import StockHit

_ARCHIVE_SOURCES = {"internet_archive", "wikimedia", "nasa"}
_HISTORICAL_VISUAL_TYPES = {"Archive", "Historical", "Military", "Documents"}

_WEIGHTS = {
    "semantic": 0.27,
    "keyword_match": 0.18,
    "historical": 0.135,
    "quality": 0.135,
    "cinematic": 0.108,
    "motion": 0.072,
    "duration": 0.10,
}

# Common grammatical words that carry little discriminating value.
#
# Keep this list small and niche-agnostic. Do not add niche-specific nouns
# such as "pool", "truck", "factory", or similar words. Those words should
# continue to count as one matched word among the complete keyword phrase.
_LOW_SIGNAL_WORDS = {"the", "a", "an", "in", "on", "at", "of", "for", "with", "and", "or"}
# Same punctuation-stripping style as app.documentary_table's whisper-alignment
# tokenizer (_PUNCT_RE/_normalize_token) — duplicated rather than imported,
# since app.scoring has no existing dependency on app.documentary_table.
_OVERLAP_PUNCT_RE = re.compile(r"[.,!?;:\"'()\[\]—–]")


def _normalize_for_overlap(text: str) -> set[str]:
    return {
        word
        for word in (_OVERLAP_PUNCT_RE.sub("", w).lower() for w in text.split())
        if word and word not in _LOW_SIGNAL_WORDS and len(word) > 1
    }


# Rarity ceiling the log-frequency is normalized against — a word this rare
# (frequency ~1e-7) or rarer saturates at the max weight of 1.0.
_RARITY_CEILING = 7.0


def _word_rarity_weight(word: str) -> float:
    """Higher weight = rarer/more specific word. Common words like 'person'
    or 'being' get a low weight; specific/technical words like 'chlorine'
    or 'algaecide' get a high weight. Words not found in the frequency
    dictionary at all (very likely niche/technical/brand terms not in
    general usage, e.g. 'algaecide') are treated as maximally rare — this
    is a reasonable default, not a workaround, since a word too obscure to
    appear in a general-English frequency corpus is exactly the kind of
    word that should carry strong discriminating signal here."""
    freq = word_frequency(word, "en")
    if freq <= 0:
        return 1.0  # unknown word — treat as maximally specific/rare
    rarity = -math.log10(freq)
    return min(1.0, rarity / _RARITY_CEILING)


def keyword_overlap_ratio(canva_keyword: str, candidate_text: str) -> float:
    """Rarity-weighted fraction of canva_keyword's meaning present in
    candidate_text. Unlike a plain word-count ratio, a match on a rare/
    specific word (e.g. "chlorine") contributes much more than a match on a
    common word (e.g. "person") — this directly addresses cases where
    generic shared words (person, holding, being, box, shelf) previously
    produced a misleadingly high ratio despite the one word that actually
    mattered being absent. _LOW_SIGNAL_WORDS/_normalize_for_overlap still
    filter pure grammatical filler first; rarity weighting is an additional
    layer on top of that, not a replacement for it.

    1.0 means every meaningful word from canva_keyword is present in the
    candidate's metadata; 0.0 means none are (or the keyword had no
    meaningful words at all). Still a ranking signal, not a filter — a
    candidate with a low or zero ratio remains eligible via the other score
    components; only the final combined score is compared against
    documentary_min_score."""
    keyword_words = _normalize_for_overlap(canva_keyword)
    if not keyword_words:
        return 0.0
    candidate_words = _normalize_for_overlap(candidate_text)

    weights = {w: _word_rarity_weight(w) for w in keyword_words}
    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0
    matched_weight = sum(weights[w] for w in keyword_words if w in candidate_words)
    return matched_weight / total_weight


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


# No scene-level target duration reaches score_asset (see
# asset_selection.select_best_asset's docstring — that data isn't produced
# until after asset selection, elsewhere in the pipeline), so this can't
# score against the specific beat's length. What it CAN catch: every row in
# this pipeline runs several seconds (real projects observed at 2-5s), so a
# candidate under this floor has to loop many times to fill even the
# shortest realistic beat — e.g. a real 0.14s local_library clip looped
# ~20x to fill a 3s row was reported as "repeats a 1-second segment
# instead of playing full content" (duration data itself was verified
# correct; the bug was this scorer having no opinion on it at all). Not a
# hard cutoff — a short clip below the floor still scores > 0, so it can
# still win when it's the only usable candidate.
_MIN_COMFORTABLE_VIDEO_DURATION = 2.0


def _duration_adequacy(hit: StockHit) -> float:
    if hit.media_type != "video" or hit.duration <= 0:
        return 1.0
    return min(1.0, hit.duration / _MIN_COMFORTABLE_VIDEO_DURATION)


# Curated, on-topic footage the user indexed themselves for this exact niche
# — a decent match here should usually beat an unknown stock candidate, so it
# gets a flat priority bump on top of the normal weighted score (applied
# after the *100 conversion, capped at 100 — see search_clip's local-library
# early-accept, which uses this boosted score against
# ai_generation_trigger_threshold).
LOCAL_LIBRARY_BONUS = 10.0


# ponytail: heuristic scorer using metadata already returned by the provider
# APIs, not a real visual-content model — upgrade to CLIP-style image/query
# similarity if ranking quality proves insufficient in practice.
def score_asset(hit: StockHit, clip: TimelineClip, query_embedding: list[float]) -> float:
    total = (
        _WEIGHTS["semantic"] * _semantic_similarity(hit, query_embedding)
        + _WEIGHTS["keyword_match"] * keyword_overlap_ratio(clip.canva_keyword, hit.text)
        + _WEIGHTS["historical"] * _historical_accuracy(hit, clip)
        + _WEIGHTS["quality"] * _visual_quality(hit)
        + _WEIGHTS["cinematic"] * _cinematic_value(hit)
        + _WEIGHTS["motion"] * _motion_potential(hit)
        + _WEIGHTS["duration"] * _duration_adequacy(hit)
    )
    score = round(total * 100, 2)
    if hit.source == "local_library":
        score = min(100.0, score + LOCAL_LIBRARY_BONUS)
    return score
