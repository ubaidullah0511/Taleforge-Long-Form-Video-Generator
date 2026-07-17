import difflib
import logging
import math
import re
from typing import get_args

from app.config import settings
from app.llm_client import generate_json
from app.models import TimelineClip, VisualType
from app.subtitles import WordTiming
from app.timecode import parse_timestamp

logger = logging.getLogger(__name__)

_VALID_VISUAL_TYPES = set(get_args(VisualType))
_FALLBACK_VISUAL_TYPE: VisualType = "B-roll"

_PUNCT_RE = re.compile(r"[.,!?;:\"'()\[\]—–]")  # includes em/en dash

CONVERSION_PROMPT = """You are an expert Documentary Visual Research Agent.

Convert the following narration script into a structured visual timeline table.

Break the script into logical visual beats of approximately 3-7 seconds each.

Return a JSON object with a single key "clips" whose value is an array. \
For every clip in that array return:
- clip_number
- section
- script_beat
- canva_keyword
- fallback_keyword
- visual_type
- edit_note
- word_count

Visual types allowed:
Archive, Cinematic, Sports, Exercise Demo, Documents, Laboratory, B-roll, Animation, Map, Military, Emotion, Industry, Historical, Interview, Montage, Nature, Technology, Combat.

Rules:
- Adapt automatically to any niche (history, business, geopolitics, technology, science, finance, crime, sports, biography, etc.).
- Generate highly searchable visual keywords.
- Use broad fallback keywords if specific footage may not exist.
- Return valid JSON only.
- Do not include markdown.
- Do not include explanations.

Script:
\"\"\"{script}\"\"\"
"""

WORDS_PER_SECOND = 2.5
MIN_CLIP_SECONDS = 3
MAX_CLIP_SECONDS = 7
# Below this much uncovered narration at the end of the table, generating a
# whole extra row isn't worth it (floating-point/rounding noise territory).
_UNCOVERED_TIME_EPSILON = 0.5
# Safety cap on how many times we'll ask the model for more coverage rows in
# a row — guards against a pathological response that never fully consumes
# the leftover transcript text.
_MAX_COVERAGE_GAP_PASSES = 3
# Hard ceiling on how long any single VIDEO clip may stay on screen. Named
# distinctly from the pre-existing MAX_CLIP_SECONDS (=7, only used above as
# the word-count-pacing cap for the no-whisper fallback) so lowering this
# doesn't silently change that unrelated behavior — this cap instead applies
# as a final splitting pass over every row regardless of how its duration was
# computed (whisper-timed or coverage rows included).
MAX_ON_SCREEN_CLIP_SECONDS = 5.0


def _format_timestamp(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


_ALIGNMENT_MATCH_THRESHOLD = 0.55  # below this, fall back to word-count for this beat only
_WINDOW_SLACK = 6  # how many extra/fewer whisper words to try around the naive word count


def _normalize_token(token: str) -> str:
    return _PUNCT_RE.sub("", token).lower()


def _normalize_tokens(text: str) -> list[str]:
    return [t for t in (_normalize_token(tok) for tok in text.split()) if t]


def _find_best_whisper_span(
    whisper_words: list[WordTiming], cursor_index: int, beat_text: str, naive_count: int,
) -> tuple[int, float] | None:
    """Finds where this beat's text actually ends in the whisper transcript
    by CONTENT, not by assuming it's exactly len(beat_text.split()) words
    ahead — that assumption breaks whenever real speech pacing/pauses (or
    even just punctuation tokens like an em-dash counting as a "word" in
    the written script but never appearing as a transcribed word) differ
    from the written script. Tries a range of window lengths around
    naive_count and scores each against beat_text with difflib's token-
    sequence similarity; returns (end_index, ratio) for the best-scoring
    window, or None if nothing clears _ALIGNMENT_MATCH_THRESHOLD."""
    beat_tokens = _normalize_tokens(beat_text)
    if not beat_tokens or cursor_index >= len(whisper_words):
        return None

    remaining = len(whisper_words) - cursor_index
    low = max(1, naive_count - _WINDOW_SLACK)
    high = min(remaining, naive_count + _WINDOW_SLACK)

    best_end_index, best_ratio = None, 0.0
    for length in range(low, high + 1):
        end_index = cursor_index + length
        window_tokens = [_normalize_token(w.text) for w in whisper_words[cursor_index:end_index]]
        window_tokens = [t for t in window_tokens if t]
        ratio = difflib.SequenceMatcher(a=beat_tokens, b=window_tokens, autojunk=False).ratio()
        if ratio > best_ratio:
            best_ratio, best_end_index = ratio, end_index

    if best_end_index is None or best_ratio < _ALIGNMENT_MATCH_THRESHOLD:
        return None
    return best_end_index, best_ratio


def _consume_whisper_span(
    whisper_words: list[WordTiming], cursor_index: int, beat_text: str, naive_count: int,
    floor_start: float, clip_number: int | None,
) -> tuple[float, float, int]:
    """Aligns one clip's real narration span in the whisper transcript by
    matching beat_text's actual content (see _find_best_whisper_span), not
    by blindly consuming naive_count words in sequence — the latter drifts
    whenever the script's own word count per beat doesn't match how many
    words Whisper actually transcribed for that stretch of speech (different
    pacing/pauses, punctuation tokens like an em-dash that split() counts
    but Whisper never transcribes, "40" vs "forty", etc.). Falls back to the
    old word-count method only when no window scores above threshold, or
    when the transcript has run out of words entirely.

    Returns (start, end, new_cursor_index). floor_start is the previous
    clip's end — this clip is never allowed to start before it (guards
    against a prior clip's own padding pushing into this one).

    Real absolute Whisper timestamps are used untouched whenever the span is
    long enough — a normal-length clip gets exact real sync with no drift.
    Two things force padding instead: the transcript running out of words
    entirely, or a real span that's degenerately short (e.g. a fast 2-3 word
    beat spoken in well under a second, which rounds start/end to the SAME
    whole-second string in _format_timestamp — a zero-duration row that
    breaks ffmpeg/Remotion rendering, not just a cosmetic issue: this exact
    case crashed a real render mid-way through). Padding only pushes this one
    clip and cascades to later ones via floor_start on the next call — it
    never affects clips elsewhere in the table."""
    if cursor_index >= len(whisper_words):
        logger.warning(
            "documentary_table: clip %s — whisper transcript ran out of words; "
            "flooring this clip to %ss from where the previous one ended",
            clip_number, MIN_CLIP_SECONDS,
        )
        return floor_start, floor_start + MIN_CLIP_SECONDS, cursor_index

    match = _find_best_whisper_span(whisper_words, cursor_index, beat_text, naive_count)
    naive_end_index = min(len(whisper_words), cursor_index + naive_count)

    if match is not None:
        end_index, ratio = match
        method = "content-match"
    else:
        end_index, ratio = naive_end_index, 0.0
        method = "word-count-fallback"
        logger.warning(
            "documentary_table: clip %s — could not confidently match beat text against the "
            "whisper transcript (best ratio below %.2f); falling back to word-count consumption "
            "for this beat only",
            clip_number, _ALIGNMENT_MATCH_THRESHOLD,
        )

    span = whisper_words[cursor_index:end_index]
    start = max(span[0].start, floor_start)
    end = max(span[-1].end, start + MIN_CLIP_SECONDS)
    if end > span[-1].end:
        logger.warning(
            "documentary_table: clip %s — real narration span was only %.2fs, "
            "flooring to %ss to avoid a degenerate near-zero-duration clip",
            clip_number, span[-1].end - span[0].start, MIN_CLIP_SECONDS,
        )

    naive_span = whisper_words[cursor_index:naive_end_index] if naive_end_index > cursor_index else []
    logger.debug(
        "documentary_table: clip %s — %s span %.2fs-%.2fs (ratio %.2f) vs old word-count-only "
        "span would have been %s",
        clip_number, method, span[0].start, span[-1].end, ratio,
        f"{naive_span[0].start:.2f}s-{naive_span[-1].end:.2f}s" if naive_span else "n/a",
    )
    return start, end, end_index


def _build_timeline_clip(
    item: dict, whisper_words: list[WordTiming] | None, cursor: float, whisper_cursor: int,
) -> tuple[TimelineClip, float, int]:
    """Builds one TimelineClip from one raw row dict — visual_type
    normalization and start/end timing, shared between the main per-script
    loop in assign_timings and _cover_trailing_narration's leftover-coverage
    rows so neither path can drift from the other."""
    item = dict(item)
    word_count = item.pop("word_count", 0) or 0
    # Groq's plain json_object mode doesn't enforce the enum listed in the
    # prompt — the model occasionally free-forms a visual_type that isn't
    # in the allowed set. Normalize instead of letting one bad clip 500 the
    # whole request.
    if item.get("visual_type") not in _VALID_VISUAL_TYPES:
        logger.warning(
            "documentary_table: clip %s had invalid visual_type %r, defaulting to %r",
            item.get("clip_number"), item.get("visual_type"), _FALLBACK_VISUAL_TYPE,
        )
        item["visual_type"] = _FALLBACK_VISUAL_TYPE

    if whisper_words is not None:
        beat_text = item.get("script_beat", "")
        naive_count = len(beat_text.split()) or word_count or 1
        start, end, whisper_cursor = _consume_whisper_span(
            whisper_words, whisper_cursor, beat_text, naive_count, cursor, item.get("clip_number"),
        )
    else:
        duration = min(MAX_CLIP_SECONDS, max(MIN_CLIP_SECONDS, word_count / WORDS_PER_SECOND))
        start, end = cursor, cursor + duration

    clip = TimelineClip(**item, start=_format_timestamp(start), end=_format_timestamp(end))
    logger.debug(
        "documentary_table: clip %s computed start=%s end=%s (raw %.2fs-%.2fs)",
        item.get("clip_number"), clip.start, clip.end, start, end,
    )
    return clip, end, whisper_cursor


# ponytail: word-count/pace heuristic — now only the fallback used when no
# real narration audio was supplied (see whisper_words below and
# documentary_pipeline.run's audio_path param). Real per-clip duration comes
# from where each clip's own words actually land in the Whisper transcript.
def assign_timings(raw_clips: list[dict], whisper_words: list[WordTiming] | None = None) -> list[TimelineClip]:
    clips = []
    cursor = 0.0
    whisper_cursor = 0
    for item in raw_clips:
        clip, cursor, whisper_cursor = _build_timeline_clip(item, whisper_words, cursor, whisper_cursor)
        clips.append(clip)

    if whisper_words is not None:
        if whisper_cursor < len(whisper_words):
            logger.warning(
                "documentary_table: %d trailing whisper word(s) were not assigned to any clip "
                "(narration audio runs longer than the script's clip table)",
                len(whisper_words) - whisper_cursor,
            )
        clips, cursor, whisper_cursor = _cover_trailing_narration(clips, whisper_words, cursor, whisper_cursor)

    _force_contiguity(clips)

    clips = _enforce_max_clip_duration(clips)
    _force_contiguity(clips)

    return clips


def _generate_coverage_rows_for_gap(leftover_words: list[WordTiming], start_clip_number: int, section: str) -> list[dict]:
    """Turns a leftover slice of the whisper transcript into new raw table
    rows the same way the initial script->table conversion does — reuses
    convert_script_to_visual_table (same prompt, same canva/fallback keyword
    generation) on just the leftover narration text instead of duplicating
    that logic, then renumbers clip_number to continue after the existing
    table. Returns [] (never raises) on an LLM failure — this is a
    best-effort enhancement; a flaky call here shouldn't crash the whole
    project when the pre-existing behavior was already just a warning."""
    leftover_text = " ".join(w.text for w in leftover_words)
    try:
        raw_rows = convert_script_to_visual_table(leftover_text)
    except Exception as exc:
        logger.warning(
            "documentary_table: coverage-gap table conversion failed (%s) — giving up on "
            "generating rows for this leftover narration segment",
            exc,
        )
        return []
    for offset, row in enumerate(raw_rows):
        row["clip_number"] = start_clip_number + offset
        row.setdefault("section", section)
    return raw_rows


def _cover_trailing_narration(
    clips: list[TimelineClip], whisper_words: list[WordTiming], cursor: float, whisper_cursor: int,
) -> tuple[list[TimelineClip], float, int]:
    """If the narration audio (per whisper_words' own final timestamp) runs
    past the last generated row's end, the rendered video would otherwise
    fall short of the audio — previously papered over at mux time by
    freezing the last frame to cover the gap (see audio_mux.py), which
    covers the audio but shows no real content. Instead, generate real
    additional rows: the leftover transcript text is run back through the
    same script->table conversion used for the original table, naturally
    chunked into more ~3-7s beats the same way, each with its own real
    canva/fallback keywords, then timed against the actual leftover whisper
    words exactly like every other row (via _build_timeline_clip)."""
    if not whisper_words or whisper_cursor >= len(whisper_words):
        return clips, cursor, whisper_cursor

    audio_duration = whisper_words[-1].end
    initial_uncovered = audio_duration - cursor
    if initial_uncovered <= _UNCOVERED_TIME_EPSILON:
        return clips, cursor, whisper_cursor

    next_clip_number = (clips[-1].clip_number + 1) if clips else 1
    section = clips[-1].section if clips else "OUTRO"
    rows_added = 0

    for _ in range(_MAX_COVERAGE_GAP_PASSES):
        uncovered = audio_duration - cursor
        if uncovered <= _UNCOVERED_TIME_EPSILON or whisper_cursor >= len(whisper_words):
            break

        raw_rows = _generate_coverage_rows_for_gap(whisper_words[whisper_cursor:], next_clip_number, section)
        if not raw_rows:
            logger.warning(
                "documentary_table: coverage-gap generation returned no new rows for %.2fs of "
                "uncovered narration at the end — giving up on further coverage",
                uncovered,
            )
            break

        for item in raw_rows:
            clip, cursor, whisper_cursor = _build_timeline_clip(item, whisper_words, cursor, whisper_cursor)
            clips.append(clip)
            next_clip_number = clip.clip_number + 1
            rows_added += 1

    if rows_added:
        logger.info(
            "documentary_table: narration ran %.2fs past the last generated row's end — added "
            "%d new row(s) to cover the missing time instead of stretching the final clip",
            initial_uncovered, rows_added,
        )
    else:
        logger.warning(
            "documentary_table: narration runs %.2fs past the last row but no coverage rows "
            "could be generated for it",
            initial_uncovered,
        )
    return clips, cursor, whisper_cursor


def _script_beat_slice(script_beat: str, part_index: int, n_parts: int) -> str:
    """Divides the parent row's own script_beat text into n_parts word-count
    chunks, one per sub-clip, so each split sub-row's keyword gets generated
    from ONLY its own slice of the narration rather than the whole beat.
    Too few words to meaningfully subdivide -> every part gets the full
    text back (distinctness is then guaranteed downstream instead)."""
    words = script_beat.split()
    if len(words) < n_parts:
        return script_beat
    per_part = math.ceil(len(words) / n_parts)
    start = part_index * per_part
    end = min(len(words), start + per_part)
    return " ".join(words[start:end]) or script_beat


def _distinct_keyword_variant(base: str, part_index: int) -> str:
    return f"{base} (part {part_index + 1})"


def _keywords_for_subclip(
    text_slice: str, base_canva: str, base_fallback: str, part_index: int,
) -> tuple[str, str]:
    """Generates an independent canva/fallback keyword pair for one split
    sub-clip from ONLY its own narration text slice, reusing the exact same
    LLM-based generator ordinary rows use (convert_script_to_visual_table)
    instead of duplicating that prompt/logic. Falls back to a clearly
    distinguished variant of the parent row's own keywords (never silently
    identical) if the call fails or returns nothing usable — e.g. the slice
    was too short to produce a meaningfully different result."""
    try:
        rows = convert_script_to_visual_table(text_slice)
    except Exception as exc:
        logger.warning(
            "documentary_table: keyword generation for split sub-clip part %d failed (%s) — "
            "falling back to a distinguished variant of the original keyword",
            part_index + 1, exc,
        )
        rows = []

    if rows:
        canva = str(rows[0].get("canva_keyword") or "").strip()
        fallback = str(rows[0].get("fallback_keyword") or "").strip()
        if canva and fallback:
            return canva, fallback

    return _distinct_keyword_variant(base_canva, part_index), _distinct_keyword_variant(base_fallback, part_index)


def _split_long_clip(clip: TimelineClip) -> list[TimelineClip]:
    """Splits one over-cap row into consecutive equal-length sub-rows that
    together cover the exact same start-end range, each <= the cap. Each
    sub-row gets its OWN independently-generated canva/fallback keyword
    (and its own narration text slice) instead of inheriting the parent's
    unchanged — so each one runs its own fresh, distinctly-targeted search
    downstream instead of the whole split effectively searching for the
    same clip twice. clip_number/section/visual_type/edit_note are copied
    unchanged; only start/end/script_beat/canva_keyword/fallback_keyword
    are sub-row-specific."""
    start = float(parse_timestamp(clip.start))
    end = float(parse_timestamp(clip.end))
    duration = end - start
    if duration <= MAX_ON_SCREEN_CLIP_SECONDS:
        return [clip]

    n_parts = math.ceil(duration / MAX_ON_SCREEN_CLIP_SECONDS)
    chunk = duration / n_parts
    sub_clips = []
    used_canva_keywords: set[str] = set()
    for i in range(n_parts):
        sub_start = start + i * chunk
        sub_end = end if i == n_parts - 1 else start + (i + 1) * chunk
        text_slice = _script_beat_slice(clip.script_beat, i, n_parts)
        canva_keyword, fallback_keyword = _keywords_for_subclip(
            text_slice, clip.canva_keyword, clip.fallback_keyword, i,
        )
        # Hard guarantee, regardless of what the LLM returned: no two
        # sub-rows from the same split ever carry a byte-identical
        # canva_keyword (required even when the slice was too degenerate
        # for the model to meaningfully differentiate).
        if canva_keyword in used_canva_keywords:
            canva_keyword = _distinct_keyword_variant(canva_keyword, i)
        used_canva_keywords.add(canva_keyword)

        sub_clips.append(clip.model_copy(update={
            "start": _format_timestamp(sub_start),
            "end": _format_timestamp(sub_end),
            "script_beat": text_slice,
            "canva_keyword": canva_keyword,
            "fallback_keyword": fallback_keyword,
        }))
    logger.info(
        "documentary_table: clip %s duration %.2fs exceeds the %.1fs on-screen cap — "
        "split into %d sub-clip(s), each with its own independently-generated keyword",
        clip.clip_number, duration, MAX_ON_SCREEN_CLIP_SECONDS, n_parts,
    )
    return sub_clips


def _enforce_max_clip_duration(clips: list[TimelineClip]) -> list[TimelineClip]:
    """Final pass, run after the table is fully built and contiguous: splits
    any row longer than MAX_ON_SCREEN_CLIP_SECONDS into multiple sub-rows.
    Renumbers clip_number sequentially across the whole result since
    splitting changes the row count. Caller must re-run _force_contiguity
    afterward — chunk boundaries are computed independently per clip and
    aren't guaranteed bit-exact against neighboring rows."""
    result = []
    for clip in clips:
        result.extend(_split_long_clip(clip))
    for index, clip in enumerate(result, start=1):
        clip.clip_number = index
    return result


def _force_contiguity(clips: list[TimelineClip]) -> None:
    """Unconditional final pass: no consecutive pair may have a gap or overlap
    between prev.end and cur.start, no matter how start/end were computed
    upstream. _consume_whisper_span's floor_start only guards against a
    previous clip's padding overlapping the next one — it does not (and, by
    the nature of content-matched whisper spans, cannot always) guard against
    a genuine gap, e.g. a real pause in the narration between two beats. This
    pass is the actual source of truth for contiguity: it forces every
    cur.start to exactly equal the previous clip's end, so a gap/overlap can
    never reach _validate_contiguity in documentary_assembly.py regardless of
    what upstream logic does or how it changes in the future."""
    for prev, cur in zip(clips, clips[1:]):
        if cur.start != prev.end:
            logger.warning(
                "documentary_table: forcing contiguity — clip %s start %s -> %s "
                "(previous clip %s ends at %s)",
                cur.clip_number, cur.start, prev.end, prev.clip_number, prev.end,
            )
            cur.start = prev.end


def convert_script_to_visual_table(script: str) -> list[dict]:
    data = generate_json(CONVERSION_PROMPT.format(script=script), settings.llm_text_model)
    return data["clips"]


def generate_table(script: str, whisper_words: list[WordTiming] | None = None) -> list[TimelineClip]:
    raw = convert_script_to_visual_table(script)
    return assign_timings(raw, whisper_words=whisper_words)


def render_table_markdown(clips: list[TimelineClip]) -> str:
    header = (
        "| Clip # | Start | End | Section | Script Beat Covered | "
        "Canva Keyword | Fallback Keyword | Visual Type | Edit Note |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|"
    rows = [
        f"|{c.clip_number}|{c.start}|{c.end}|{c.section}|{c.script_beat}|"
        f"{c.canva_keyword}|{c.fallback_keyword}|{c.visual_type}|{c.edit_note}|"
        for c in clips
    ]
    return "\n".join([header, sep, *rows])
