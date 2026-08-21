"""Offline/mocked tests for app.documentary_pipeline, app.documentary_table, and
app.scoring. No network/API keys required. Run with: pytest tests/test_documentary_pipeline.py
"""
import asyncio
import json
import subprocess
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app import documentary_pipeline
from app.audio_mux import AudioMuxError
from app.clip_ingest import probe
from app.config import settings
from app.documentary_pipeline import (
    ScoredAsset,
    _download_asset,
    _media_type_bias,
    _generate_semantic_keywords,
    _keywords_are_concrete,
    _known_and_not_horizontal,
    _resolve_clip,
    _safe_search,
    _satisfied,
    check_footage_availability,
    generate_and_edit,
    generate_full_video,
    niche_for_clip,
    pexels_video,
    pick_best_asset,
    plan_documentary,
    provider_order_for_niche,
    rank_acceptable_assets,
    render_keyword_match_report,
    rerender_single_clip,
    resolve_script_text,
    run,
    search_clip,
)
from app.documentary_table import (
    CONVERSION_PROMPT,
    MAX_ON_SCREEN_CLIP_SECONDS,
    MIN_CLIP_SECONDS,
    _MIN_REAL_SPAN_SECONDS,
    _enforce_max_clip_duration,
    _force_contiguity,
    _format_clip_timestamps,
    _split_long_clip,
    assign_timings,
    convert_script_to_visual_table,
    generate_table,
)
from app.models import AlternateCandidate, AssetInfo, AvailabilityReport, ProjectMeta, TimelineClip, TimelineEntry
from app.niches import DEFAULT_NICHE, get_niche
from app.scoring import _word_rarity_weight, keyword_overlap_ratio, score_asset
from app.stock.base import StockHit
from app.style_decision import StyleDecision
from app.subtitles import WordTiming
from app.timecode import parse_timestamp
from app.transcription import TranscriptionError

_PROVIDER_NAMES = [
    "local_library", "pexels_video", "pixabay_video", "coverr", "pexels_images", "pixabay_images",
    "internet_archive", "nasa",
]


@contextmanager
def patch_providers(**overrides):
    """Patches all 9 provider search() functions. Providers not named in overrides return []."""
    with ExitStack() as stack:
        mocks = {}
        for provider_name in _PROVIDER_NAMES:
            mock = AsyncMock(return_value=overrides.get(provider_name, []))
            stack.enter_context(patch(f"app.documentary_pipeline.{provider_name}.search", new=mock))
            mocks[provider_name] = mock
        stack.enter_context(patch("app.documentary_pipeline.generate_json", return_value={"keywords": ["generic visual footage", "broad documentary scene", "archive style imagery"]}))
        yield mocks


def _clip(clip_number=1, visual_type="Cinematic", start="00:00:00", end="00:00:05", canva_keyword="k"):
    return TimelineClip(
        clip_number=clip_number, section="INTRO", script_beat="beat", canva_keyword=canva_keyword,
        fallback_keyword="f", visual_type=visual_type, edit_note="n", start=start, end=end,
    )


def _hit(source="pexels", source_id="1", media_type="video", width=1920, height=1080, text="t"):
    return StockHit(
        source=source, source_id=source_id, download_url=f"http://x/{source_id}.mp4",
        width=width, height=height, duration=5, media_type=media_type, text=text,
    )


def test_assign_timings_clamps_and_accumulates():
    raw = [
        {"clip_number": 1, "section": "INTRO", "script_beat": "a", "canva_keyword": "k1",
         "fallback_keyword": "f1", "visual_type": "Archive", "edit_note": "n1", "word_count": 2},
        {"clip_number": 2, "section": "INTRO", "script_beat": "b", "canva_keyword": "k2",
         "fallback_keyword": "f2", "visual_type": "Archive", "edit_note": "n2", "word_count": 30},
    ]
    clips = assign_timings(raw)
    assert (clips[0].start, clips[0].end) == ("00:00:00", f"00:00:0{MIN_CLIP_SECONDS}")  # 2 words clamped up to the MIN_CLIP_SECONDS floor
    # 30 words clamped down to the 5s word-count ceiling, which now equals MAX_ON_SCREEN_CLIP_SECONDS
    # itself, so it lands exactly at the cap and does not need splitting (see
    # test_enforce_max_clip_duration_splits_an_8s_row_into_multiple_sub_5s_rows for split coverage).
    assert len(clips) == 2
    assert clips[1].start == clips[0].end
    assert clips[1].end == f"00:00:0{MIN_CLIP_SECONDS + 5}"


def test_assign_timings_normalizes_invalid_visual_type_instead_of_raising():
    raw = [{"clip_number": 1, "section": "INTRO", "script_beat": "a", "canva_keyword": "k",
            "fallback_keyword": "f", "visual_type": "Metaphor", "edit_note": "n", "word_count": 5}]
    clips = assign_timings(raw)  # would raise a pydantic ValidationError without normalization
    assert clips[0].visual_type == "B-roll"


def test_assign_timings_uses_real_whisper_durations_when_provided():
    raw = [
        {"clip_number": 1, "section": "INTRO", "script_beat": "hello there world how are you today", "canva_keyword": "k1",
         "fallback_keyword": "f1", "visual_type": "Archive", "edit_note": "n1", "word_count": 7},
        {"clip_number": 2, "section": "INTRO", "script_beat": "goodbye now my good old dear friend", "canva_keyword": "k2",
         "fallback_keyword": "f2", "visual_type": "Archive", "edit_note": "n2", "word_count": 7},
    ]
    # Realistic pace (0.6s/word) across enough words that both real spans
    # (4.2s each) comfortably exceed _MIN_REAL_SPAN_SECONDS(2.0) — nothing here should get floored.
    words = ("hello", "there", "world", "how", "are", "you", "today",
             "goodbye", "now", "my", "good", "old", "dear", "friend")
    whisper_words = [WordTiming(text=w, start=i * 0.6, end=(i + 1) * 0.6) for i, w in enumerate(words)]

    clips = assign_timings(raw, whisper_words=whisper_words)
    # real timestamps used instead of word_count/2.5 -> real durations, not the 4s floor
    assert (clips[0].start, clips[0].end) == ("00:00:00", "00:00:04")  # 7 words * 0.6s = 4.2s, real
    assert (clips[1].start, clips[1].end) == ("00:00:04", "00:00:08")  # continues to 14*0.6=8.4s, real


def test_assign_timings_generates_coverage_rows_when_narration_runs_past_the_table():
    """Regression test: previously, if the narration audio ran longer than
    the generated table, the final row would just get stretched/frozen at
    mux time to cover the gap (see audio_mux.py's tpad hold) instead of
    showing real content. assign_timings must now generate additional
    table row(s) — with real canva/fallback keywords from the same
    script->table conversion, not a special-cased stub — to cover any such
    leftover narration, and the full table (original + new rows) must stay
    contiguous end to end."""
    raw = [
        {"clip_number": i + 1, "section": "S", "canva_keyword": f"k{i}", "fallback_keyword": f"f{i}",
         "visual_type": "Archive", "edit_note": "n",
         "script_beat": " ".join(f"w{i}_{j}" for j in range(5)), "word_count": 5}
        for i in range(8)
    ]
    # 8 rows * 5 words * 1s/word = 40s of table, tiling the transcript exactly.
    words = []
    t = 0.0
    for i in range(8):
        for j in range(5):
            words.append(WordTiming(text=f"w{i}_{j}", start=t, end=t + 1.0))
            t += 1.0
    # 5 more real seconds of narration (40s-45s) that no raw row covers at all.
    leftover_tokens = [f"w8_{j}" for j in range(5)]
    for tok in leftover_tokens:
        words.append(WordTiming(text=tok, start=t, end=t + 1.0))
        t += 1.0

    coverage_response = {
        "clips": [
            {"clip_number": 999, "section": "OUTRO", "script_beat": " ".join(leftover_tokens),
             "canva_keyword": "leftover keyword", "fallback_keyword": "leftover fallback",
             "visual_type": "B-roll", "edit_note": "n", "word_count": 5},
        ]
    }

    with patch("app.documentary_table.generate_json", return_value=coverage_response) as mock_generate_json:
        clips = assign_timings(raw, whisper_words=words)

    assert len(clips) == 9  # 8 original rows + 1 real coverage row, not a stretched/frozen last clip
    assert clips[0].start == "00:00:00"
    assert clips[7].end == "00:00:40"  # original table's last row unchanged

    new_row = clips[8]
    assert new_row.clip_number == 9  # renumbered to continue after the original table
    assert new_row.section == "OUTRO"
    assert new_row.canva_keyword == "leftover keyword"
    assert new_row.fallback_keyword == "leftover fallback"
    assert new_row.start == clips[7].end == "00:00:40"  # contiguous with the original table
    assert new_row.end == "00:00:45"  # covers all the way to the real narration end

    mock_generate_json.assert_called_once()  # reused the same conversion function, not duplicated logic
    prompt_arg = mock_generate_json.call_args.args[0]
    assert "w8_0" in prompt_arg and "w8_4" in prompt_arg  # ran on exactly the leftover narration text

    for prev, cur in zip(clips, clips[1:]):
        assert cur.start == prev.end  # full table (original + new rows) stays gap-free end to end


def test_assign_timings_floors_to_min_duration_when_whisper_words_run_out():
    raw = [
        {"clip_number": 1, "section": "INTRO", "script_beat": "one two three four five six", "canva_keyword": "k1",
         "fallback_keyword": "f1", "visual_type": "Archive", "edit_note": "n1", "word_count": 6},
        {"clip_number": 2, "section": "INTRO", "script_beat": "seven eight nine", "canva_keyword": "k2",
         "fallback_keyword": "f2", "visual_type": "Archive", "edit_note": "n2", "word_count": 3},
    ]
    # Transcript only covers the first clip's 6 words, at a pace (0.8s/word)
    # comfortably clearing the MIN_CLIP_SECONDS floor (real pace, no floor
    # needed there) — the second clip's words never arrive at all.
    whisper_words = [WordTiming(text=w, start=i * 0.8, end=(i + 1) * 0.8)
                      for i, w in enumerate(("one", "two", "three", "four", "five", "six"))]
    clips = assign_timings(raw, whisper_words=whisper_words)
    assert clips[0].start == "00:00:00" and clips[0].end == "00:00:05"  # 6*0.8=4.8s, real, not floored
    # second clip has no whisper words left at all -> floors to MIN_CLIP_SECONDS from where clip 1 ended
    assert clips[1].start == clips[0].end == "00:00:05"
    assert clips[1].end == f"00:00:0{5 + MIN_CLIP_SECONDS}"  # 4.8 + MIN_CLIP_SECONDS, rounds up


def test_format_clip_timestamps_avoids_rounding_collision():
    # 1.5 and 2.5 both round to 2 under Python's banker's rounding despite a
    # full 1.0s real gap between them — would previously render as a fake
    # zero-duration "00:00:02"-"00:00:02" clip.
    start, end = _format_clip_timestamps(1.5, 2.5)
    assert start == "00:00:02"
    assert end == "00:00:03"
    assert start != end

    # Normal, non-colliding case is untouched.
    start, end = _format_clip_timestamps(0.0, 4.8)
    assert (start, end) == ("00:00:00", "00:00:05")


def test_assign_timings_floors_degenerate_short_real_span_and_cascades_forward():
    """Regression test for a real production crash: a fast 3-word beat's real
    whisper span was under a second, so its start/end rounded to the exact
    same whole-second string in _format_timestamp — a zero-duration table
    row that broke ffmpeg/Remotion rendering outright (not just cosmetic).
    Normal-length clips before and after must keep exact real sync; only the
    short one (and the contiguity boundary right after it) should shift."""
    raw = [
        {"clip_number": 1, "section": "S", "script_beat": "one two three four five six seven", "canva_keyword": "k1",
         "fallback_keyword": "f1", "visual_type": "Archive", "edit_note": "n1", "word_count": 7},
        {"clip_number": 2, "section": "S", "script_beat": "fast short beat", "canva_keyword": "k2",
         "fallback_keyword": "f2", "visual_type": "Archive", "edit_note": "n2", "word_count": 3},
        {"clip_number": 3, "section": "S", "script_beat": "eleven twelve thirteen fourteen fifteen sixteen", "canva_keyword": "k3",
         "fallback_keyword": "f3", "visual_type": "Archive", "edit_note": "n3", "word_count": 6},
    ]
    words, t = [], 0.0
    for w in raw[0]["script_beat"].split():  # normal pace: 0.6s/word -> 4.2s real span
        words.append(WordTiming(text=w, start=t, end=t + 0.6)); t += 0.6
    for w in raw[1]["script_beat"].split():  # fast beat: 0.2s/word -> only 0.6s real span
        words.append(WordTiming(text=w, start=t, end=t + 0.2)); t += 0.2
    for w in raw[2]["script_beat"].split():  # normal pace resumes
        words.append(WordTiming(text=w, start=t, end=t + 0.6)); t += 0.6

    clips = assign_timings(raw, whisper_words=words)

    assert clips[0].start == "00:00:00" and clips[0].end == "00:00:04"  # 7*0.6=4.2s, real, untouched
    assert clips[1].start == clips[0].end  # perfectly contiguous, no gap/overlap
    start_s, end_s = parse_timestamp(clips[1].start), parse_timestamp(clips[1].end)
    assert end_s - start_s >= _MIN_REAL_SPAN_SECONDS  # floored, never a zero/near-zero duration row
    assert clips[2].start == clips[1].end  # clip 3 cascades from the padded clip 2, still contiguous


def test_assign_timings_content_match_avoids_drift_from_punctuation_token_mismatch():
    """Regression test for a real production bug: a script beat containing
    an em-dash ("What you never saw — what nobody...") gets one extra token
    from str.split() that Whisper never actually transcribes as a word (real
    speech has no spoken token for a dash). The old word-count-per-beat
    method would consume one extra REAL whisper word belonging to the next
    beat as a result — drifting the boundary. Content-matching by text
    similarity must find the true (undrifted) boundary instead. Numbers are
    chosen so the drift would be large enough to survive whole-second string
    rounding, matching the ~2s real drift measured against actual project data."""
    raw = [
        {"clip_number": 1, "section": "S",
         "script_beat": "alpha beta gamma — delta epsilon — zeta eta — theta iota kappa",
         "canva_keyword": "k1", "fallback_keyword": "f1", "visual_type": "Archive", "edit_note": "n1",
         "word_count": 13},
        {"clip_number": 2, "section": "S", "script_beat": "lambda mu nu xi omicron",
         "canva_keyword": "k2", "fallback_keyword": "f2", "visual_type": "Archive", "edit_note": "n2",
         "word_count": 5},
    ]
    # Real spoken words: beat 1's 10 real words (dashes are never spoken),
    # then beat 2's 5 real words — 15 words total, continuous, 0.6s each.
    real_words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron".split()
    whisper_words = [WordTiming(text=w, start=i * 0.6, end=(i + 1) * 0.6) for i, w in enumerate(real_words)]

    with patch("app.documentary_table.generate_json",
               return_value={"clips": [{"canva_keyword": "split keyword", "fallback_keyword": "split fallback"}]}):
        clips = assign_timings(raw, whisper_words=whisper_words)

    # Correct real boundary: beat 1 has exactly 10 real spoken words -> ends at 10*0.6=6.0s.
    # That 6s span exceeds the 5s on-screen cap, so it splits into two 3s sub-rows.
    assert clips[0].start == "00:00:00" and clips[0].end == "00:00:03"
    assert clips[1].start == "00:00:03" and clips[1].end == "00:00:06"
    # Beat 2's own real span is 3.0s (5 words * 0.6s), comfortably over
    # _MIN_REAL_SPAN_SECONDS(2.0) -> real span used as-is, no flooring: 6s + 3s = 9s.
    assert clips[2].start == "00:00:06" and clips[2].end == "00:00:09"


def test_assign_timings_falls_back_to_word_count_when_content_match_confidence_is_too_low():
    """When a beat's text has essentially nothing in common with the actual
    whisper words available (e.g. the table's script_beat doesn't match the
    real narration for that stretch at all), content-matching must not force
    a low-confidence guess — it should gracefully fall back to the old
    word-count method for that beat only, per spec."""
    raw = [
        {"clip_number": 1, "section": "S", "script_beat": "one two three four five six seven",
         "canva_keyword": "k1", "fallback_keyword": "f1", "visual_type": "Archive", "edit_note": "n1",
         "word_count": 7},
    ]
    unrelated = "completely unrelated narration content spoken instead here now".split()
    whisper_words = [WordTiming(text=w, start=i * 0.6, end=(i + 1) * 0.6) for i, w in enumerate(unrelated)]

    clips = assign_timings(raw, whisper_words=whisper_words)

    # falls back to naive word-count (7 words) -> consumes all 7 available real words
    assert clips[0].start == "00:00:00" and clips[0].end == "00:00:04"  # 7*0.6=4.2s -> rounds to 4


def test_force_contiguity_corrects_a_gap_between_consecutive_clips():
    """Direct unit test of _force_contiguity: given clips with a real gap
    (clip 5 starting after clip 4 ends, exactly the reported production bug),
    confirm the gap exists beforehand and is corrected afterward."""
    make = lambda n, start, end: TimelineClip(
        clip_number=n, section="S", script_beat="b", canva_keyword="k",
        fallback_keyword="f", visual_type="Archive", edit_note="n", start=start, end=end,
    )
    clips = [make(4, "00:00:14", "00:00:17"), make(5, "00:00:18", "00:00:22")]
    assert clips[1].start != clips[0].end  # sanity-check: the gap is really there beforehand

    _force_contiguity(clips)

    assert clips[1].start == clips[0].end == "00:00:17"  # forced contiguous, zero gap


def test_enforce_max_clip_duration_splits_an_8s_row_into_multiple_sub_5s_rows():
    """New requirement: no single VIDEO clip may stay on screen longer than
    MAX_ON_SCREEN_CLIP_SECONDS (5s), regardless of the table row's own
    duration. An 8s row must split into multiple sub-rows, each <=5s,
    covering the exact same 0-8s range with zero gaps once _force_contiguity
    is re-run — and visual_type/section/edit_note are still carried over
    unchanged (only start/end/script_beat/keywords are sub-row-specific,
    see the dedicated keyword-distinctness test below)."""
    make = lambda n, start, end: TimelineClip(
        clip_number=n, section="S", script_beat="one two three four five six seven eight", canva_keyword="k",
        fallback_keyword="f", visual_type="Archive", edit_note="n", start=start, end=end,
    )
    clips = [make(1, "00:00:00", "00:00:08")]

    with patch("app.documentary_table.generate_json",
               return_value={"clips": [{"canva_keyword": "gen keyword", "fallback_keyword": "gen fallback"}]}):
        split = _enforce_max_clip_duration(clips)
    _force_contiguity(split)

    assert len(split) > 1  # the single 8s row became multiple sub-rows
    for clip in split:
        duration = parse_timestamp(clip.end) - parse_timestamp(clip.start)
        assert duration <= MAX_ON_SCREEN_CLIP_SECONDS

    assert split[0].start == "00:00:00"
    assert split[-1].end == "00:00:08"  # covers the exact same original range
    for prev, cur in zip(split, split[1:]):
        assert cur.start == prev.end  # zero gaps

    for clip in split:  # visual_type/section/edit_note untouched by the split
        assert clip.visual_type == "Archive"
        assert clip.section == "S"
        assert clip.edit_note == "n"

    assert [c.clip_number for c in split] == list(range(1, len(split) + 1))  # renumbered sequentially


def test_split_long_clip_gives_each_sub_row_its_own_distinct_keyword():
    """FIX 1: a split sub-row must not just inherit its sibling's keyword —
    each sub-row's canva_keyword is generated independently from ONLY that
    sub-row's own narration text slice (via the same convert_script_to_visual_table
    generator ordinary rows use), and the two must not be byte-identical
    strings even when the underlying LLM calls happen to agree."""
    clip = TimelineClip(
        clip_number=1, section="S", script_beat="one two three four five six seven eight",
        canva_keyword="original keyword", fallback_keyword="original fallback",
        visual_type="Archive", edit_note="n", start="00:00:00", end="00:00:08",
    )
    # Two distinct real per-slice generations, proving each sub-row got its
    # own separate keyword-generation call rather than reusing one result.
    responses = [
        {"clips": [{"canva_keyword": "rocket launch on the pad", "fallback_keyword": "space rocket liftoff"}]},
        {"clips": [{"canva_keyword": "astronaut walking on the moon", "fallback_keyword": "moon surface exploration"}]},
    ]
    with patch("app.documentary_table.generate_json", side_effect=responses) as mock_generate_json:
        sub_clips = _split_long_clip(clip)

    assert len(sub_clips) == 2
    assert mock_generate_json.call_count == 2  # generated separately per sub-row, not once and copied
    assert sub_clips[0].canva_keyword == "rocket launch on the pad"
    assert sub_clips[1].canva_keyword == "astronaut walking on the moon"
    assert sub_clips[0].canva_keyword != sub_clips[1].canva_keyword  # never byte-identical
    assert sub_clips[0].script_beat != sub_clips[1].script_beat  # each generated from its own text slice


def test_split_long_clip_does_not_leak_part_suffix_into_keyword_when_generation_is_degenerate():
    """Regression test for the "(part N)" search-quality bug: when the
    original row's text is too short to meaningfully subdivide (or the LLM
    call fails/returns nothing usable for every sub-row), the sub-row keyword
    must fall back to the parent's clean keyword unmodified — never a
    "(part N)"-suffixed variant, since canva_keyword/fallback_keyword are the
    literal strings searched against stock providers and shown in
    timeline_table.md. Two sub-rows landing on the same clean keyword in this
    degenerate case is fine; a polluted search string is not."""
    clip = TimelineClip(
        clip_number=1, section="S", script_beat="wow", canva_keyword="original keyword",
        fallback_keyword="original fallback", visual_type="Archive", edit_note="n",
        start="00:00:00", end="00:00:08",
    )
    with patch("app.documentary_table.generate_json", side_effect=RuntimeError("llm unavailable")):
        sub_clips = _split_long_clip(clip)

    assert len(sub_clips) == 2
    for sub in sub_clips:
        assert sub.canva_keyword == "original keyword"
        assert sub.fallback_keyword == "original fallback"
        assert "part" not in sub.canva_keyword.lower()


def test_assign_timings_forces_contiguity_when_a_real_narration_pause_creates_a_gap():
    """Regression test for the exact reported production bug: 'timeline
    gap/overlap between clip 4 (ends 00:00:17) and clip 5 (starts 00:00:18)'.

    The prior fix (start = max(natural_start, floor_start) in
    _consume_whisper_span) only guards against a previous clip's padding
    overlapping the next one. It does NOT guard against a genuine pause in
    the narration — a real gap in the whisper transcript's timestamps
    between clip 4's last word and clip 5's first word — because in that
    case natural_start > floor_start and max() picks the (larger, gappy)
    natural_start. This reproduces that exact scenario with a real 1-second
    pause and asserts assign_timings' final output has zero gap regardless,
    via the unconditional _force_contiguity pass."""
    raw = [
        {"clip_number": 1, "section": "S", "script_beat": "w0 w1 w2 w3 w4 w5", "canva_keyword": "k1",
         "fallback_keyword": "f1", "visual_type": "Archive", "edit_note": "n1", "word_count": 6},
        {"clip_number": 2, "section": "S", "script_beat": "w6 w7 w8 w9 w10", "canva_keyword": "k2",
         "fallback_keyword": "f2", "visual_type": "Archive", "edit_note": "n2", "word_count": 5},
        {"clip_number": 3, "section": "S", "script_beat": "w11 w12 w13", "canva_keyword": "k3",
         "fallback_keyword": "f3", "visual_type": "Archive", "edit_note": "n3", "word_count": 3},
        {"clip_number": 4, "section": "S", "script_beat": "w14 w15 w16", "canva_keyword": "k4",
         "fallback_keyword": "f4", "visual_type": "Archive", "edit_note": "n4", "word_count": 3},
        {"clip_number": 5, "section": "S", "script_beat": "w17 w18 w19 w20", "canva_keyword": "k5",
         "fallback_keyword": "f5", "visual_type": "Archive", "edit_note": "n5", "word_count": 4},
    ]
    # Clips 1-4 tile the transcript back-to-back with no gap at all (times
    # 0-17s, 1s/word). Clip 5's words then start at t=18 instead of t=17 — a
    # real 1-second pause in the narration audio, exactly like a beat
    # boundary that doesn't perfectly tile the transcript in production.
    words = []
    for i, w in enumerate(("w0", "w1", "w2", "w3", "w4", "w5")):
        words.append(WordTiming(text=w, start=float(i), end=float(i + 1)))
    for i, w in enumerate(("w6", "w7", "w8", "w9", "w10")):
        words.append(WordTiming(text=w, start=float(6 + i), end=float(7 + i)))
    for i, w in enumerate(("w11", "w12", "w13")):
        words.append(WordTiming(text=w, start=float(11 + i), end=float(12 + i)))
    for i, w in enumerate(("w14", "w15", "w16")):
        words.append(WordTiming(text=w, start=float(14 + i), end=float(15 + i)))
    for i, w in enumerate(("w17", "w18", "w19", "w20")):
        words.append(WordTiming(text=w, start=float(18 + i), end=float(19 + i)))  # +1s pause here

    with patch("app.documentary_table.generate_json",
               return_value={"clips": [{"canva_keyword": "split keyword", "fallback_keyword": "split fallback"}]}):
        clips = assign_timings(raw, whisper_words=words)

    # Original clip 1's 6s span (0-6s) exceeds the 5s on-screen cap and splits into two
    # 3s sub-rows, shifting every later clip's index by one (clip_number renumbered too) —
    # what was "clip 4" is now clips[4], not clips[3]. Its own real span (3s, "w14 w15
    # w16" 1s/word) comfortably clears _MIN_REAL_SPAN_SECONDS(2.0), so it keeps its
    # real, unpadded timestamps — no cascade.
    assert clips[4].start == "00:00:14" and clips[4].end == "00:00:17"  # real span, untouched
    # without _force_contiguity, clip 6's natural content-matched span would
    # start at 00:00:18 (the real 1s pause) -> a gap against clip 5's end.
    # The final pass must force it back to clip 5's end instead.
    assert clips[5].start == clips[4].end == "00:00:17"
    # Clip 6's own real span (4 words * 1s = 4s) also clears the floor -> real span used as-is.
    assert clips[5].end == "00:00:22"


def test_generate_table_uses_structured_conversion_prompt():
    raw = {"clips": [{"clip_number": 1, "section": "INTRO", "script_beat": "hello world",
                       "canva_keyword": "k", "fallback_keyword": "f", "visual_type": "Cinematic",
                       "edit_note": "n", "word_count": 2}]}
    with patch("app.documentary_table.generate_json", return_value=raw) as mock_generate_json:
        generate_table("some script")
    prompt = mock_generate_json.call_args.args[0]
    assert "Convert the following narration script" in prompt
    assert "Return valid JSON only" in prompt


def test_conversion_prompt_targets_five_second_beats():
    assert "approximately 5 seconds" in CONVERSION_PROMPT
    assert "3-7 seconds" not in CONVERSION_PROMPT


def test_convert_script_to_visual_table_single_call_for_short_script():
    script = "some short script"
    raw = {"clips": [{"clip_number": 1, "section": "INTRO", "script_beat": script,
                       "canva_keyword": "k", "fallback_keyword": "f", "visual_type": "Cinematic",
                       "edit_note": "n", "word_count": 3}]}
    with patch("app.documentary_table.generate_json", return_value=raw) as mock_generate_json:
        rows = convert_script_to_visual_table(script)
    expected_prompt = CONVERSION_PROMPT.format(script=script, niche_context=get_niche(DEFAULT_NICHE).system_context)
    mock_generate_json.assert_called_once_with(expected_prompt, settings.llm_text_model)
    assert rows == raw["clips"]


def test_convert_script_to_visual_table_replaces_niche_violating_keywords():
    niche_config = get_niche(DEFAULT_NICHE)
    raw = {"clips": [{"clip_number": 1, "section": "INTRO", "script_beat": "text",
                       "canva_keyword": "a car driving down the road", "fallback_keyword": "semi truck highway",
                       "visual_type": "Cinematic", "edit_note": "n", "word_count": 3}]}
    with patch("app.documentary_table.generate_json", return_value=raw):
        rows = convert_script_to_visual_table("some short script")
    assert rows[0]["canva_keyword"] == niche_config.safe_fallback_keyword  # violated -> replaced
    assert rows[0]["fallback_keyword"] == "semi truck highway"  # clean -> untouched


def test_convert_script_to_visual_table_chunks_long_script_and_renumbers():
    script = "One two three. Four five six. Seven eight nine."
    responses = [
        {"clips": [{"clip_number": 1, "section": "S", "script_beat": "One two three."}]},
        {"clips": [{"clip_number": 1, "section": "S", "script_beat": "Four five six."}]},
        {"clips": [{"clip_number": 1, "section": "S", "script_beat": "Seven eight nine."}]},
    ]
    with patch("app.documentary_table._MAX_WORDS_PER_CONVERSION_CALL", 5), \
         patch("app.documentary_table.generate_json", side_effect=responses) as mock_generate_json:
        rows = convert_script_to_visual_table(script)

    assert mock_generate_json.call_count == 3
    assert [r["script_beat"] for r in rows] == ["One two three.", "Four five six.", "Seven eight nine."]
    assert [r["clip_number"] for r in rows] == [1, 2, 3]


def test_score_asset_weighting():
    # canva_keyword matches hit.text exactly so keyword_match is also 1.0 —
    # every one of the six weighted components is perfect, total stays 100.
    clip = _clip(visual_type="Archive", canva_keyword="archival footage")
    hit = _hit(source="internet_archive", text="archival footage")
    with patch("app.scoring.embed", return_value=[1.0, 0.0]):
        score = score_asset(hit, clip, [1.0, 0.0])
    assert score == 100.0  # perfect semantic + keyword_match + historical + quality + cinematic + motion


def test_score_asset_local_library_bonus_boosts_partial_match():
    clip = _clip(visual_type="Cinematic", canva_keyword="archival footage")
    local_hit = _hit(source="local_library", text="archival footage")
    stock_hit = _hit(source="pexels", text="archival footage")
    with patch("app.scoring.embed", return_value=[1.0, 0.0]):
        local_score = score_asset(local_hit, clip, [0.5, 0.5])
        stock_score = score_asset(stock_hit, clip, [0.5, 0.5])
    assert local_score == pytest.approx(stock_score + 10.0, abs=0.01)


def test_score_asset_local_library_bonus_caps_at_100():
    clip = _clip(visual_type="Archive", canva_keyword="archival footage")
    hit = _hit(source="local_library", text="archival footage")
    with patch("app.scoring.embed", return_value=[1.0, 0.0]):
        score = score_asset(hit, clip, [1.0, 0.0])
    assert score == 100.0


def test_keyword_overlap_ratio_full_match():
    assert keyword_overlap_ratio("Pool pH testing", "pool pH testing water") == 1.0


def test_keyword_overlap_ratio_partial_match():
    # Rarity-weighted, not a plain 1/3 word-count ratio: "pool" contributes
    # only its own (below-average) rarity weight out of the three keyword
    # words' combined weight, not an equal one-third share.
    assert keyword_overlap_ratio("Pool pH testing", "swimming pool water") == pytest.approx(0.3163, abs=0.001)


def test_keyword_overlap_ratio_no_match():
    assert keyword_overlap_ratio("Pool pH testing", "balloon splash summer fun") == 0.0


def test_keyword_overlap_ratio_ignores_filler_words_in_denominator():
    assert keyword_overlap_ratio("Testing the pH of a pool", "pool pH testing") == 1.0


def test_keyword_overlap_ratio_empty_or_filler_only_keyword():
    assert keyword_overlap_ratio("", "pool testing") == 0.0
    assert keyword_overlap_ratio("the and of", "pool testing") == 0.0


def test_keyword_overlap_ratio_strips_metadata_punctuation():
    # Rarity-weighted: "pool" + "testing" matched, "pH" absent — not a plain 2/3.
    assert keyword_overlap_ratio("Pool pH testing", "pool, testing, chlorine") == pytest.approx(0.6326, abs=0.001)


def test_word_rarity_weight_common_words_score_low():
    """Common, non-discriminating words should get low rarity weight —
    this is what replaces the old fixed stopword list's inability to catch
    generic-but-non-grammatical words like "person"/"holding"/"being"."""
    for word in ("person", "being", "holding", "box"):
        assert _word_rarity_weight(word) < 0.65, f"{word!r} should score as common/low-weight"


def test_word_rarity_weight_specific_words_score_high():
    """Specific/technical/niche words should get high rarity weight,
    including words absent from the frequency corpus entirely (e.g. a brand/
    niche term like 'algaecide'), which are treated as maximally rare."""
    for word in ("chlorine", "borax", "algaecide", "ph"):
        assert _word_rarity_weight(word) > 0.7, f"{word!r} should score as rare/high-weight"


def test_keyword_overlap_ratio_real_motivating_examples():
    """Direct regression test for the four real keyword_match_report.md rows
    that motivated this change — a shared generic word (being/holding/
    adjusting/box+shelf) should no longer inflate the ratio as much as the
    old uniform word-count formula did; the specific/absent word (chlorine/
    borax/pH/algaecide) should dominate the result instead."""
    # "being" matches (common, low weight); "chlorine"/"added" absent — ratio
    # well under the old uniform 1/3.
    assert keyword_overlap_ratio(
        "Chlorine being added", "stomachache, pain, intestine, intestinal pain, human being"
    ) == pytest.approx(0.245, abs=0.01)
    # "person"+"holding" match (common); "borax" (the word that matters) absent
    # — ratio meaningfully under the old uniform 2/3.
    assert keyword_overlap_ratio(
        "Person holding borax", "a person wearing an apron holding a burning candle"
    ) == pytest.approx(0.537, abs=0.01)
    # "person" matches (common); "pH" (rare) absent, but "adjusting" itself
    # also carries a comparatively high rarity weight in general English
    # frequency data (gerund verb forms are simply uncommon), so this ratio
    # moves only modestly from the old uniform 2/3 — an honest limitation of
    # domain-agnostic word-frequency weighting, not a test bug.
    assert keyword_overlap_ratio(
        "Person adjusting pH", "a person adjusting a microscope"
    ) == pytest.approx(0.635, abs=0.01)
    # "box"+"shelf" match (moderately common); "algaecide" (the word that
    # matters, and unknown to the frequency corpus) absent.
    assert keyword_overlap_ratio(
        "Algaecide box shelf",
        "amazon, box, parcels, warehouse, post, shelf, rack, shipping, transport, "
        "cargo, delivery, storage, order, willbot studios",
    ) == pytest.approx(0.561, abs=0.01)


def test_score_asset_full_vs_single_word_keyword_match():
    """Candidate B (matches all 3 meaningful keyword words) must score
    meaningfully higher than Candidate A (matches only the one broad word
    "pool") purely because of keyword_match — every other score component is
    controlled/identical between the two so keyword_match is isolated as the
    only source of the difference."""
    clip = _clip(visual_type="Cinematic", canva_keyword="Pool pH testing")
    hit_a = _hit(source_id="a", text="swimming pool water fun")
    hit_b = _hit(source_id="b", text="pool pH testing water analysis")

    # Same embedding regardless of input text -> semantic_similarity identical
    # for both candidates, isolating keyword_match as the only variable.
    with patch("app.scoring.embed", return_value=[1.0, 0.0]):
        score_a = score_asset(hit_a, clip, [1.0, 0.0])
        score_b = score_asset(hit_b, clip, [1.0, 0.0])

    assert score_b > score_a
    # keyword_match: A ~= 0.3163 (rarity-weighted "pool" alone), B = 1.0 (all
    # 3 words present) -> weighted gap = 0.18 * (1.0 - 0.3163) ~= 12.31 points
    # (weight is 0.18, not 0.20 — rebalanced when a "duration" component was
    # added to score_asset, see app.scoring._WEIGHTS)
    assert score_b - score_a == pytest.approx(12.31, abs=0.05)


def test_download_asset_persists_selected_text_from_hit():
    """The winning StockHit's own .text is already in memory at accept time
    (the same text CLIP verification and niche filtering already judged the
    candidate against) — confirms it's persisted onto AssetInfo.selected_text
    with zero new API/CLIP calls, not silently discarded after acceptance."""
    hit = _hit(source="pexels", source_id="42", media_type="video", text="a real candidate description")
    asset = ScoredAsset(hit=hit, score=77.0)
    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.download_video_trimmed", new=AsyncMock(return_value="out.mp4")):
        info = asyncio.run(_download_asset(asset, Path(tmpdir), "clip001", 1, target_duration=5.0))
    assert info.selected_text == "a real candidate description"


def test_render_keyword_match_report_includes_selected_text_and_keyword_match():
    table = [
        _clip(clip_number=1, canva_keyword="Pool pH testing"),
        _clip(clip_number=2, canva_keyword="Algae growth pool"),
    ]
    entry1 = TimelineEntry(
        clip_number=1, start="00:00:00", end="00:00:05", duration=5.0, script="beat one",
        asset_path="p.mp4", asset_metadata=AssetInfo(
            path="p.mp4", source="pexels", source_id="1", media_type="video",
            score=70.0, selected_text="pool pH testing kit",
        ), recommended_effect="e", transition="t",
    )
    entry2 = TimelineEntry(
        clip_number=2, start="00:00:05", end="00:00:10", duration=5.0, script="beat two",
        asset_path=None, asset_metadata=None, recommended_effect="e", transition="t",
        placeholder=True, note="no acceptable asset found across all providers",
    )

    report = render_keyword_match_report(table, [entry1, entry2])

    assert "Pool pH testing" in report
    assert "pexels:1" in report
    assert "pool pH testing kit" in report
    assert "1.00" in report  # keyword_match: all 3 meaningful words ("pool","ph","testing") matched
    assert "placeholder — no candidate accepted" in report
    assert "no acceptable asset found across all providers" in report


def test_niche_for_clip():
    assert niche_for_clip(_clip(visual_type="Historical")) == "historical"
    assert niche_for_clip(_clip(visual_type="Technology")) == "modern"
    assert niche_for_clip(_clip(visual_type="Cinematic")) == "general"


def test_provider_order_for_niche_prioritizes_archives_for_historical():
    # local_library leads both orderings (see PROVIDER_ORDER_DEFAULT/HISTORICAL)
    # but is filtered out of the actual search_clip run unless the active
    # content niche has a local_library_path set — see the local_library
    # gating tests below.
    assert provider_order_for_niche("historical")[1] is not pexels_video
    assert provider_order_for_niche("modern")[1] is pexels_video
    assert provider_order_for_niche("general")[1] is pexels_video


def test_satisfied_only_stops_early_on_a_genuinely_high_quality_hit():
    """A merely-acceptable (>=documentary_min_score) hit is no longer enough
    to stop the provider loop early — that was the actual premature-early-
    stop bug (settling for 2 mediocre 50-70-range hits instead of exhausting
    the full provider list). Only a documentary_high_quality_score+ hit is a
    legitimate reason to stop searching more providers."""
    mediocre_asset = ScoredAsset(hit=_hit(), score=55)
    high_quality_asset = ScoredAsset(hit=_hit(), score=95)
    with patch.object(settings, "documentary_high_quality_score", 90), \
         patch.object(settings, "documentary_min_score", 50):
        assert _satisfied([mediocre_asset]) is False  # acceptable, but not excellent -> keep searching
        assert _satisfied([mediocre_asset, mediocre_asset]) is False  # multiple mediocre hits still don't stop it
        assert _satisfied([high_quality_asset]) is True  # one genuinely excellent hit is enough
    with patch.object(settings, "documentary_high_quality_score", 50):
        assert _satisfied([mediocre_asset]) is True  # now counts as high-quality on its own


def test_high_quality_hit_stops_search_early():
    """A single genuinely excellent (>=documentary_high_quality_score) hit —
    anywhere, video or image — is enough to stop the provider loop early.
    Checked as one combined pool rather than requiring independent
    excellence in EACH media type: an image candidate structurally cannot
    reach 90 under the current weights (no motion credit, lower cinematic
    ceiling — max ~87), so a per-bucket requirement would make this
    short-circuit unreachable in practice whenever any images are returned.
    Constructed to actually clear 90 (canva_keyword matches hit.text
    exactly, archive source, full resolution) — unlike the old version of
    this test, which used merely-decent (~72-score) hits that only
    satisfied the old, now-removed count-based check."""
    clip = _clip(visual_type="Cinematic", canva_keyword="archival footage")
    excellent_video_hit = _hit(source="internet_archive", source_id="v1", media_type="video", text="archival footage")

    with patch_providers(pexels_video=[excellent_video_hit]) as mocks, \
         patch("app.scoring.embed", return_value=[1.0, 0.0]):
        video_hits, _image_hits = asyncio.run(search_clip(clip, used_assets={}))

    assert video_hits
    assert video_hits[0].score >= settings.documentary_high_quality_score
    # The loop must stop right after the very first provider — nothing else queried.
    for later_provider in ["pixabay_video", "pexels_images", "pixabay_images", "internet_archive", "nasa"]:
        mocks[later_provider].assert_not_called()


def test_mediocre_hits_no_longer_stop_the_provider_loop_early():
    """Regression test for the actual premature-early-stop bug this task
    fixes: two merely-acceptable (not excellent) hits used to satisfy the
    old count-based _satisfied() check and stop the search after only 2-3
    providers, silently never querying Pixabay images/Wikimedia/NASA at all.
    The loop must now keep going through every configured provider unless
    something actually clears documentary_high_quality_score."""
    clip = _clip(visual_type="Cinematic")
    decent_video_hit = _hit(source="pexels", source_id="v1", media_type="video", text="some decent footage")
    decent_image_hit = _hit(source="pexels_images", source_id="i1", media_type="image", text="some decent photo")

    with patch_providers(pexels_video=[decent_video_hit], pexels_images=[decent_image_hit]) as mocks, \
         patch("app.scoring.embed", return_value=[1.0, 0.0]):
        video_hits, image_hits = asyncio.run(search_clip(clip, used_assets={}))

    assert video_hits and image_hits
    assert all(a.score < settings.documentary_high_quality_score for a in video_hits + image_hits)
    # Unlike the old count-based early stop, every remaining provider (other
    # than internet_archive, legitimately excluded for the default "trucks"
    # content niche) must still be queried since nothing excellent was found.
    for later_provider in ["pixabay_video", "pixabay_images", "nasa"]:
        mocks[later_provider].assert_called()


def test_search_clip_uses_table_canva_keyword_as_primary_query():
    """Regression guard for a real-run investigation: confirmed by direct
    trace against a live run that search_clip already reads clip.canva_keyword
    correctly as its first query on every provider — this pins that behavior
    so it can't silently drift (e.g. to raw script_beat text or a stale
    keyword) in the future."""
    clip = _clip(visual_type="Technology").model_copy(update={
        "canva_keyword": "Windows logo with cash or money", "fallback_keyword": "Microsoft logo",
    })
    hit = _hit(source="pexels", source_id="1")

    with patch_providers(pexels_video=[hit]) as mocks, \
         patch("app.scoring.embed", return_value=[1.0, 0.0]):
        asyncio.run(search_clip(clip, used_assets={}))

    mocks["pexels_video"].assert_any_call("Windows logo with cash or money", per_page=5)


def test_search_clip_uses_fallback_keyword_when_canva_keyword_returns_nothing():
    clip = _clip(visual_type="Technology").model_copy(update={
        "canva_keyword": "Windows logo with cash or money", "fallback_keyword": "Microsoft logo",
    })
    hit = _hit(source="pexels", source_id="1")

    async def fake_search(query, per_page=5):
        return [hit] if query == "Microsoft logo" else []

    with patch_providers(), \
         patch("app.documentary_pipeline.pexels_video.search", new=fake_search), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]):
        video_hits, _image_hits = asyncio.run(search_clip(clip, used_assets={}))

    assert any(h.hit.source_id == "1" for h in video_hits)  # only found via fallback_keyword, not canva_keyword


def test_known_and_not_horizontal_rejects_only_confirmed_non_16_9():
    assert _known_and_not_horizontal(_hit(width=1080, height=1920)) is True  # vertical, known
    assert _known_and_not_horizontal(_hit(width=1920, height=1080)) is False  # horizontal, known
    assert _known_and_not_horizontal(_hit(width=0, height=0)) is False  # unknown (e.g. nasa) — deferred, not rejected


def test_search_clip_keeps_non_16_9_hit_with_known_metadata_for_post_download_normalization():
    """Non-16:9 metadata is no longer a pre-download rejection reason — those
    candidates get normalized (cropped/blur-padded) after download instead
    (see _resolve_clip). Only a confirmed too-low-res candidate is rejected
    at this metadata stage (see the low-res test below)."""
    clip = _clip(visual_type="Technology")  # modern niche: required video=1
    vertical_hit = _hit(source="pexels", source_id="vert", width=1080, height=1920)
    horizontal_hit = _hit(source="pixabay", source_id="horiz", width=1920, height=1080)

    with patch_providers(pexels_video=[vertical_hit, horizontal_hit]), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]):
        video_hits, _image_hits = asyncio.run(search_clip(clip, used_assets={}))

    source_ids = {a.hit.source_id for a in video_hits}
    assert source_ids == {"vert", "horiz"}  # both kept — orientation no longer gates sourcing


def test_search_clip_rejects_known_too_low_res_hit_before_download():
    clip = _clip(visual_type="Technology")
    tiny_hit = _hit(source="pexels", source_id="tiny", width=320, height=240)  # below the 480 floor
    good_hit = _hit(source="pixabay", source_id="good", width=1920, height=1080)

    with patch_providers(pexels_video=[tiny_hit, good_hit]), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]):
        video_hits, _image_hits = asyncio.run(search_clip(clip, used_assets={}))

    source_ids = {a.hit.source_id for a in video_hits}
    assert source_ids == {"good"}  # too-low-res rejected pre-download, nothing to normalize


def test_search_clip_rejects_niche_violating_hit_before_scoring():
    """Candidate selection stage: a hit whose provider metadata (tags/title)
    shows no truck signal at all must never enter video_hits/image_hits — so
    it can never become top-1 regardless of its semantic_similarity score,
    per the niche-lock-at-selection fix."""
    clip = _clip(visual_type="Technology")
    car_hit = _hit(source="pexels", source_id="cars", text="cars in the road")
    truck_hit = _hit(source="pixabay", source_id="truck", text="semi truck climbing a mountain")

    with patch_providers(pexels_video=[car_hit, truck_hit]), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]):
        video_hits, _image_hits = asyncio.run(search_clip(clip, used_assets={}, content_niche="trucks"))

    source_ids = {a.hit.source_id for a in video_hits}
    assert source_ids == {"truck"}  # niche-violating candidate rejected before scoring


def test_search_clip_keeps_mixed_metadata_hit_with_truck_signal():
    """Real-world stock tags commonly co-mention "cars" alongside "truck" for
    ordinary highway footage — this must NOT be rejected (see
    candidate_violates_niche's positive-term exception)."""
    clip = _clip(visual_type="Technology")
    mixed_hit = _hit(source="pixabay", source_id="mixed", text="highway, traffic, cars, road, truck, speed")

    with patch_providers(pexels_video=[mixed_hit]), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]):
        video_hits, _image_hits = asyncio.run(search_clip(clip, used_assets={}, content_niche="trucks"))

    assert {a.hit.source_id for a in video_hits} == {"mixed"}


def test_search_clip_skips_internet_archive_when_niche_disallows_it():
    """trucks/pool_maintenance set use_archive_org=False (see app.niches) —
    internet_archive must never be queried for them, even when the clip's
    own visual_type would otherwise put it first in the provider order."""
    clip = _clip(visual_type="Archive")  # historical -> PROVIDER_ORDER_HISTORICAL (internet_archive first)
    with patch_providers() as mocks, patch("app.scoring.embed", return_value=[1.0, 0.0]):
        asyncio.run(search_clip(clip, used_assets={}, content_niche="trucks"))
    mocks["internet_archive"].assert_not_called()


def test_search_clip_queries_internet_archive_when_niche_allows_it():
    clip = _clip(visual_type="Archive")
    archival_niche = get_niche("trucks")._replace(key="historical_test", use_archive_org=True)
    with patch_providers() as mocks, \
         patch("app.scoring.embed", return_value=[1.0, 0.0]), \
         patch("app.documentary_pipeline.get_niche", return_value=archival_niche):
        asyncio.run(search_clip(clip, used_assets={}, content_niche="historical_test"))
    mocks["internet_archive"].assert_called()


def test_search_clip_skips_local_library_when_niche_has_no_local_path():
    """trucks/pool_maintenance leave local_library_path unset (see app.niches)
    — local_library must never be queried for them, even though it's first
    in both PROVIDER_ORDER_DEFAULT and PROVIDER_ORDER_HISTORICAL."""
    clip = _clip(visual_type="Technology")
    with patch_providers() as mocks, patch("app.scoring.embed", return_value=[1.0, 0.0]):
        asyncio.run(search_clip(clip, used_assets={}, content_niche="trucks"))
    mocks["local_library"].assert_not_called()


def test_search_clip_queries_local_library_with_niche_subfolder_when_configured():
    clip = _clip(visual_type="Technology", canva_keyword="deadlift")
    body_niche = get_niche("trucks")._replace(key="bodybuilding_test", local_library_path="bodybuilding")
    with patch_providers() as mocks, \
         patch("app.scoring.embed", return_value=[1.0, 0.0]), \
         patch("app.documentary_pipeline.get_niche", return_value=body_niche):
        asyncio.run(search_clip(clip, used_assets={}, content_niche="bodybuilding_test"))
    mocks["local_library"].assert_any_call("deadlift", "bodybuilding", per_page=5)


@contextmanager
def _temp_niches(*configs):
    """Installs real NicheConfig entries into the live app.niches.NICHES
    registry for the duration of the block, then removes them — used for the
    multi-select tests below, which go through resolve_sub_niches (a real
    NICHES lookup + parent-matching validation), not get_niche(), so patching
    app.documentary_pipeline.get_niche (as the single-select tests above do)
    wouldn't reach this code path at all."""
    from app.niches import NICHES as niches_registry

    for config in configs:
        niches_registry[config.key] = config
    try:
        yield
    finally:
        for config in configs:
            niches_registry.pop(config.key, None)


def test_search_clip_multi_select_queries_every_selected_sub_niches_local_library():
    """Multi-select's whole point (see app.niches.resolve_sub_niches): picking
    2+ sub-niches that share one parent must query ALL of their
    local_library_path folders in a single search_clip run, not just one."""
    parent = get_niche("trucks")._replace(key="ml_parent", banned_terms=[], positive_terms=[], local_library_path="")
    child_a = get_niche("trucks")._replace(
        key="ml_child_a", banned_terms=[], positive_terms=[], local_library_path="lib_a", parent_key="ml_parent",
    )
    child_b = get_niche("trucks")._replace(
        key="ml_child_b", banned_terms=[], positive_terms=[], local_library_path="lib_b", parent_key="ml_parent",
    )
    clip = _clip(visual_type="Technology", canva_keyword="deadlift")
    with _temp_niches(parent, child_a, child_b), \
         patch_providers() as mocks, \
         patch("app.scoring.embed", return_value=[1.0, 0.0]):
        asyncio.run(search_clip(clip, used_assets={}, content_niche=["ml_child_a", "ml_child_b"]))
    mocks["local_library"].assert_any_call("deadlift", "lib_a", per_page=5)
    mocks["local_library"].assert_any_call("deadlift", "lib_b", per_page=5)


def test_search_clip_multi_select_rejects_hit_violating_either_selected_sub_niches_denylist():
    """Merge contract check: a candidate violating EITHER selected sub-niche's
    banned_terms must be rejected, not just the first one's — proves the
    union (not just the first niche's list) is actually what gets enforced."""
    parent = get_niche("trucks")._replace(key="ml_parent2", banned_terms=[], positive_terms=[], local_library_path="")
    child_a = get_niche("trucks")._replace(
        key="ml_child2_a", banned_terms=["forbidden_a"], positive_terms=[], local_library_path="", parent_key="ml_parent2",
    )
    child_b = get_niche("trucks")._replace(
        key="ml_child2_b", banned_terms=["forbidden_b"], positive_terms=[], local_library_path="", parent_key="ml_parent2",
    )
    clip = _clip(visual_type="Technology")
    bad_hit = _hit(source="pexels", source_id="bad", text="forbidden_b content here")
    good_hit = _hit(source="pixabay", source_id="good", text="totally fine footage")
    with _temp_niches(parent, child_a, child_b), \
         patch_providers(pexels_video=[bad_hit], pixabay_video=[good_hit]), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]):
        video_hits, _image_hits = asyncio.run(
            search_clip(clip, used_assets={}, content_niche=["ml_child2_a", "ml_child2_b"])
        )
    assert {a.hit.source_id for a in video_hits} == {"good"}


def test_search_clip_multi_select_falls_back_to_get_niche_for_single_item_list():
    """A category with no sub-niches selected sends a 1-item list (see
    getSelectedContentNiche() in templates/index.html) — must behave exactly
    like today's plain-string single-select, still via get_niche()."""
    with patch_providers() as mocks, patch("app.scoring.embed", return_value=[1.0, 0.0]):
        asyncio.run(search_clip(clip=_clip(visual_type="Technology"), used_assets={}, content_niche=["trucks"]))
    mocks["local_library"].assert_not_called()  # trucks has no local_library_path, same as content_niche="trucks"


def test_search_clip_skips_local_library_when_enable_local_library_false():
    """Source toggle checkbox: even with a niche-configured local_library_path,
    enable_local_library=False must still skip it entirely (distinct from the
    niche-has-no-local-path case above — this is a per-request override)."""
    clip = _clip(visual_type="Technology", canva_keyword="deadlift")
    body_niche = get_niche("trucks")._replace(key="bodybuilding_test", local_library_path="bodybuilding")
    with patch_providers() as mocks, \
         patch("app.scoring.embed", return_value=[1.0, 0.0]), \
         patch("app.documentary_pipeline.get_niche", return_value=body_niche):
        asyncio.run(search_clip(
            clip, used_assets={}, content_niche="bodybuilding_test", enable_local_library=False,
        ))
    mocks["local_library"].assert_not_called()


def test_search_clip_skips_stock_providers_when_enable_stock_providers_false():
    """Source toggle checkbox: enable_stock_providers=False must skip every
    stock provider while still allowing local_library through."""
    clip = _clip(visual_type="Technology", canva_keyword="deadlift")
    body_niche = get_niche("trucks")._replace(key="bodybuilding_test", local_library_path="bodybuilding")
    with patch_providers() as mocks, \
         patch("app.scoring.embed", return_value=[1.0, 0.0]), \
         patch("app.documentary_pipeline.get_niche", return_value=body_niche):
        asyncio.run(search_clip(
            clip, used_assets={}, content_niche="bodybuilding_test", enable_stock_providers=False,
        ))
    mocks["local_library"].assert_called()
    for name in ("pexels_video", "pixabay_video", "coverr", "pexels_images", "pixabay_images", "internet_archive", "nasa"):
        mocks[name].assert_not_called()


def test_resolve_clip_skips_ai_generation_when_enable_ai_generation_false():
    """Source toggle checkbox: with no real candidate and enable_ai_generation=False,
    _resolve_clip must go straight to a placeholder rather than ever calling
    generate_fallback_image_openai — regardless of score/threshold."""
    clip = _clip()
    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.search_clip", new=AsyncMock(return_value=([], []))), \
         patch.object(settings, "enable_ai_generation_fallback", True), \
         patch("app.documentary_pipeline.generate_fallback_image_openai", new=AsyncMock(return_value=(True, None))) as mock_generate:
        entry, _ = asyncio.run(_resolve_clip(
            clip, Path(tmpdir), used_assets={}, allow_duplicates=False,
            downloads_so_far=0, limit=None, enable_ai_generation=False,
        ))

    mock_generate.assert_not_called()
    assert entry.placeholder is True


def test_search_clip_accepts_strong_local_library_match_and_skips_stock_providers():
    """Score lands between ai_generation_trigger_threshold (67) and
    documentary_high_quality_score (90) on purpose — isolates the NEW
    local-library-specific early-accept from the pre-existing _satisfied
    high-quality early-stop, which only fires at >=90."""
    clip = _clip(visual_type="Technology", canva_keyword="deadlift")
    body_niche = get_niche("trucks")._replace(key="bodybuilding_test", local_library_path="bodybuilding")
    strong_hit = _hit(source="local_library", media_type="video", width=480, height=480, text="deadlift")

    with patch_providers(local_library=[strong_hit]) as mocks, \
         patch("app.scoring.embed", return_value=[1.0, 0.0]), \
         patch("app.documentary_pipeline.get_niche", return_value=body_niche):
        video_hits, _image_hits = asyncio.run(search_clip(clip, used_assets={}, content_niche="bodybuilding_test"))

    assert 67.0 <= video_hits[0].score < 90.0
    assert {a.hit.source for a in video_hits} == {"local_library"}
    mocks["pexels_video"].assert_not_called()
    mocks["pixabay_video"].assert_not_called()


def test_search_clip_falls_through_to_stock_when_local_library_match_is_weak():
    clip = _clip(visual_type="Technology", canva_keyword="deadlift")
    body_niche = get_niche("trucks")._replace(key="bodybuilding_test", local_library_path="bodybuilding")
    weak_hit = _hit(source="local_library", media_type="image", width=480, height=480, text="unrelated content")
    good_stock_hit = _hit(source="pexels", media_type="video", text="deadlift")

    with patch_providers(local_library=[weak_hit], pexels_video=[good_stock_hit]) as mocks, \
         patch("app.scoring.embed", return_value=[1.0, 0.0]), \
         patch("app.documentary_pipeline.get_niche", return_value=body_niche):
        asyncio.run(search_clip(clip, used_assets={}, content_niche="bodybuilding_test"))

    mocks["pexels_video"].assert_called()


def test_local_library_candidate_below_trigger_threshold_falls_through_to_ai_generation():
    """Bug report: a local_library candidate that scores below
    ai_generation_trigger_threshold (post-bonus) must NOT be accepted
    directly — it must go through _should_try_ai_generation exactly like a
    pexels/pixabay candidate would (mirrors
    test_ai_generation_fallback_triggers_when_best_real_candidate_scores_below_threshold,
    same mechanism, local_library source instead of pexels). This is the
    _resolve_clip level (post search_clip) — confirms _should_try_ai_generation
    itself already keys off ai_generation_trigger_threshold, not
    documentary_min_score, regardless of source."""
    clip = _clip()
    hit = _hit(source="local_library", source_id="weak.mp4", text="unrelated")
    below_threshold_score = settings.ai_generation_trigger_threshold - 2.0
    video_hits = [ScoredAsset(hit=hit, score=below_threshold_score)]

    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.search_clip", new=AsyncMock(return_value=(video_hits, []))), \
         patch.object(settings, "enable_ai_generation_fallback", True), \
         patch("app.documentary_pipeline.generate_fallback_image_openai", new=AsyncMock(return_value=(True, None))) as mock_generate, \
         patch("app.documentary_pipeline.generate_json", return_value={"prompt": "enhanced prompt"}), \
         patch("app.documentary_pipeline.probe", return_value=(0.0, "2048x1152")), \
         patch("app.documentary_pipeline.normalize_to_16_9", return_value=True):
        entry, _ = asyncio.run(_resolve_clip(
            clip, Path(tmpdir), used_assets={}, allow_duplicates=False,
            downloads_so_far=0, limit=None,
        ))

    mock_generate.assert_called_once()
    assert entry.placeholder is False
    assert entry.asset_metadata.source == "ai_generated"


def test_search_clip_registers_local_library_dedup_key_and_excludes_on_reuse():
    """Bug report: duplicate local_library file used across multiple clips —
    verifies used_assets exclusion applies to local_library hits identically
    to pexels/pixabay (mirrors test_cross_clip_dedup_excludes_used_asset_unless_allowed).
    source_id is the filename (see app.stock.local_library.search), so the
    dedup key (hit.source, hit.source_id) is well-defined for local_library
    exactly like any other provider — no special-casing needed or present."""
    clip = _clip(canva_keyword="deadlift")
    body_niche = get_niche("trucks")._replace(key="bodybuilding_test", local_library_path="bodybuilding")
    hit = _hit(source="local_library", source_id="deadlift.mp4", text="deadlift")
    used_at_cap = {("local_library", "deadlift.mp4"): settings.max_asset_repeat_count}

    with patch_providers(local_library=[hit]), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]), \
         patch("app.documentary_pipeline.get_niche", return_value=body_niche):
        video_hits, _ = asyncio.run(search_clip(
            clip, used_assets=used_at_cap, allow_duplicates=False, content_niche="bodybuilding_test",
        ))
    assert video_hits == []

    with patch_providers(local_library=[hit]), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]), \
         patch("app.documentary_pipeline.get_niche", return_value=body_niche):
        video_hits, _ = asyncio.run(search_clip(
            clip, used_assets=used_at_cap, allow_duplicates=True, content_niche="bodybuilding_test",
        ))
    assert len(video_hits) == 1


def test_resolve_clip_registers_local_library_asset_in_used_assets_after_accept():
    """The other half of the dedup story: once a local_library candidate is
    actually accepted, _resolve_clip must register it in used_assets under
    (source, source_id) the same way it does for any other source — this is
    the generic post-download registration at _resolve_clip's asset_key
    line, unaffected by which provider the winning candidate came from."""
    clip = _clip()
    hit = _hit(source="local_library", source_id="deadlift.mp4", text="deadlift")
    video_hits = [ScoredAsset(hit=hit, score=90.0)]

    async def fake_download_asset(asset, dest_dir, prefix, index, target_duration):
        return AssetInfo(path="local.mp4", source="local_library", source_id="deadlift.mp4",
                          media_type="video", width=1920, height=1080, duration=5.0, score=90.0)

    used_assets: dict[tuple[str, str], int] = {}
    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.search_clip", new=AsyncMock(return_value=(video_hits, []))), \
         patch("app.documentary_pipeline._download_asset", side_effect=fake_download_asset), \
         patch("app.documentary_pipeline.probe", return_value=(5.0, "1920x1080")), \
         patch("app.documentary_pipeline.passes_visual_verification", return_value=(True, 0.5)):
        asyncio.run(_resolve_clip(
            clip, Path(tmpdir), used_assets=used_assets, allow_duplicates=False,
            downloads_so_far=0, limit=None,
        ))

    assert used_assets[("local_library", "deadlift.mp4")] == 1


def test_resolve_clip_normalizes_downloaded_asset_that_turns_out_vertical():
    """Covers the metadata-lied case: a hit with unknown/passing metadata that
    turns out to be vertical only once the actual file is probed after
    download. Non-16:9 is no longer a rejection reason — this must be
    normalized (blur-padded, since it's portrait and well above the low-res
    floor) to 16:9 in place and used directly, not discarded to a
    placeholder."""
    import subprocess

    clip = _clip()
    hit = _hit(source="nasa", source_id="v1", width=0, height=0)  # unknown metadata, passes pre-filter

    with tempfile.TemporaryDirectory() as tmpdir:
        clips_root = Path(tmpdir)
        vertical_path = clips_root / "vertical_source.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=1080x1920:r=25:d=1",
             "-pix_fmt", "yuv420p", str(vertical_path)],
            capture_output=True, check=True,
        )

        fake_asset = AssetInfo(
            path=str(vertical_path), source="nasa", source_id="v1", media_type="video",
            width=0, height=0, duration=5.0, score=90.0,
        )

        with patch_providers(nasa=[hit]), \
             patch("app.scoring.embed", return_value=[1.0, 0.0]), \
             patch("app.documentary_pipeline._download_asset", new=AsyncMock(return_value=fake_asset)), \
             patch("app.documentary_pipeline.passes_visual_verification", return_value=(True, 0.5)):
            entry, downloads_so_far = asyncio.run(
                _resolve_clip(clip, clips_root, used_assets={}, allow_duplicates=False,
                              downloads_so_far=0, limit=None)
            )

        assert entry.placeholder is False
        assert entry.asset_metadata.width == 1920 and entry.asset_metadata.height == 1080
        assert downloads_so_far == 1
        assert vertical_path.exists()  # normalized in place, not discarded
        _duration, resolution = probe(str(vertical_path))
        assert resolution == "1920x1080"


def test_resolve_clip_verifies_against_canva_keyword_not_full_script_beat():
    """Regression guard for a real-run investigation: passing canva_keyword +
    full script_beat prose into CLIP verification diluted/truncated the text
    past CLIP's 77-token limit and rejected genuinely correct matches (three
    real weighing-scale videos scored 0.227-0.236 against a 19-word diluted
    string for a 'weighing pool chemicals' beat, in a live run — see
    app/visual_verification.py's docstring). Pins that only clip.canva_keyword
    (a short, concrete, filmable phrase) is passed to passes_visual_verification,
    never concatenated with script_beat."""
    clip = _clip().model_copy(update={
        "canva_keyword": "weighing white powder on a scale",
        "script_beat": (
            "Start by weighing out small amounts, typically a few ounces "
            "per ten thousand gallons of water, before dissolving it."
        ),
    })
    hit = _hit(source="pexels", source_id="v1", width=1920, height=1080)
    fake_asset = AssetInfo(
        path="fake.mp4", source="pexels", source_id="v1", media_type="video",
        width=1920, height=1080, duration=5.0, score=90.0,
    )

    with tempfile.TemporaryDirectory() as tmpdir, \
         patch_providers(pexels_video=[hit]), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]), \
         patch("app.documentary_pipeline._download_asset", new=AsyncMock(return_value=fake_asset)), \
         patch("app.documentary_pipeline.probe", return_value=(5.0, "1920x1080")), \
         patch("app.documentary_pipeline.passes_visual_verification", return_value=(True, 0.5)) as mock_verify:
        asyncio.run(
            _resolve_clip(clip, Path(tmpdir), used_assets={}, allow_duplicates=False,
                          downloads_so_far=0, limit=None)
        )

    mock_verify.assert_called_once()
    called_text = mock_verify.call_args.args[1]
    assert called_text == "weighing white powder on a scale"
    assert "ten thousand gallons" not in called_text


def test_resolve_clip_rejects_and_falls_through_when_top_candidate_is_too_low_res():
    """The core fix, updated: orientation alone no longer rejects a
    candidate (it gets normalized instead) — only a confirmed too-low-res
    file still falls through to the next-best already-scored candidate
    rather than giving up straight to a placeholder."""
    import subprocess

    clip = _clip()
    top_hit = _hit(source="pexels", source_id="top")
    second_hit = _hit(source="pixabay", source_id="second")
    video_hits = [ScoredAsset(hit=top_hit, score=95.0), ScoredAsset(hit=second_hit, score=80.0)]

    with tempfile.TemporaryDirectory() as tmpdir:
        clips_root = Path(tmpdir)
        tiny_path = clips_root / "tiny_source.mp4"
        horizontal_path = clips_root / "horizontal_source.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=320x240:r=25:d=1",
             "-pix_fmt", "yuv420p", str(tiny_path)],
            capture_output=True, check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=blue:s=1920x1080:r=25:d=1",
             "-pix_fmt", "yuv420p", str(horizontal_path)],
            capture_output=True, check=True,
        )

        async def fake_download_asset(asset, dest_dir, prefix, index, target_duration):
            if asset.hit.source_id == "top":
                return AssetInfo(path=str(tiny_path), source="pexels", source_id="top",
                                  media_type="video", width=0, height=0, duration=5.0, score=95.0)
            return AssetInfo(path=str(horizontal_path), source="pixabay", source_id="second",
                              media_type="video", width=0, height=0, duration=5.0, score=80.0)

        with patch("app.documentary_pipeline.search_clip", new=AsyncMock(return_value=(video_hits, []))), \
             patch("app.documentary_pipeline._download_asset", side_effect=fake_download_asset), \
             patch("app.documentary_pipeline.passes_visual_verification", return_value=(True, 0.5)):
            entry, downloads_so_far = asyncio.run(
                _resolve_clip(clip, clips_root, used_assets={}, allow_duplicates=False,
                              downloads_so_far=0, limit=None)
            )

        assert entry.placeholder is False
        assert entry.asset_metadata.source_id == "second"  # top candidate (too low-res) skipped, next-best used
        assert downloads_so_far == 2  # both download attempts counted
        assert not tiny_path.exists()  # rejected candidate's file discarded
        assert horizontal_path.exists()  # winning candidate's file kept


def _p4_top_and_fallen_through_setup(tmpdir):
    """Shared setup for the two regression tests below: candidates[0] scores
    90 (above ai_generation_trigger_threshold, so the pre-download check
    correctly skips generation) but fails CLIP visual verification;
    candidates[1] scores 58 (below threshold) and passes. Real production
    bug: the pre-download check's verdict was never re-checked against
    whichever candidate actually survives the post-download rejection
    cascade — see the accept-point re-check in _resolve_clip."""
    clip = _clip()
    top_hit = _hit(source="pexels", source_id="top")
    low_hit = _hit(source="pixabay", source_id="low")
    video_hits = [ScoredAsset(hit=top_hit, score=90.0), ScoredAsset(hit=low_hit, score=58.0)]

    clips_root = Path(tmpdir)
    top_path = clips_root / "top_source.mp4"
    low_path = clips_root / "low_source.mp4"
    for path, color in ((top_path, "blue"), (low_path, "red")):
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=1920x1080:r=25:d=1",
             "-pix_fmt", "yuv420p", str(path)],
            capture_output=True, check=True,
        )

    async def fake_download_asset(asset, dest_dir, prefix, index, target_duration):
        path = top_path if asset.hit.source_id == "top" else low_path
        return AssetInfo(path=str(path), source=asset.hit.source, source_id=asset.hit.source_id,
                          media_type="video", width=1920, height=1080, duration=5.0, score=asset.score)

    call_count = {"n": 0}

    def fake_verify(path, keyword):
        call_count["n"] += 1
        return (False, 0.15) if call_count["n"] == 1 else (True, 0.85)  # top rejected, low accepted

    return clip, video_hits, fake_download_asset, fake_verify


def test_resolve_clip_retries_ai_generation_for_fallen_through_candidate_below_threshold():
    """Regression test for the real production bug (found via
    projects/*/timeline.json: a clip accepted a real candidate scoring 58,
    generation_attempted=False, even though other clips in the same project
    show AI generation was enabled and working): _should_try_ai_generation
    only ran once, against candidates[0] (score 90, above threshold, so it
    correctly decided not to generate) — when that candidate then failed
    CLIP verification and the loop fell through to a lower-scoring (58,
    below threshold) candidate, that stale verdict was never re-checked.
    Generation must now be (re-)attempted against whichever candidate is
    actually about to be accepted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        clip, video_hits, fake_download_asset, fake_verify = _p4_top_and_fallen_through_setup(tmpdir)

        with patch("app.documentary_pipeline.search_clip", new=AsyncMock(return_value=(video_hits, []))), \
             patch("app.documentary_pipeline._download_asset", side_effect=fake_download_asset), \
             patch("app.documentary_pipeline.passes_visual_verification", side_effect=fake_verify), \
             patch.object(settings, "enable_ai_generation_fallback", True), \
             patch("app.documentary_pipeline.generate_fallback_image_openai", new=AsyncMock(return_value=(True, None))), \
             patch("app.documentary_pipeline.probe", return_value=(0.0, "2048x1152")), \
             patch("app.documentary_pipeline.normalize_to_16_9", return_value=True):
            entry, _ = asyncio.run(
                _resolve_clip(clip, Path(tmpdir), used_assets={}, allow_duplicates=False,
                              downloads_so_far=0, limit=None)
            )

    assert entry.generation_attempted is True
    assert entry.asset_metadata.source == "ai_generated"  # never silently accepted the score-58 real candidate


def test_resolve_clip_accepts_fallen_through_below_threshold_candidate_only_after_generation_fails():
    """Same scenario as above, but generation fails — the real, below-
    threshold candidate is still an acceptable graceful-degradation outcome
    (a real clip beats a placeholder), but ONLY after generation was
    actually attempted; generation_attempted must be True and the note must
    say so, unlike the pre-fix behavior (generation_attempted=False, empty
    note, no attempt ever made)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        clip, video_hits, fake_download_asset, fake_verify = _p4_top_and_fallen_through_setup(tmpdir)

        with patch("app.documentary_pipeline.search_clip", new=AsyncMock(return_value=(video_hits, []))), \
             patch("app.documentary_pipeline._download_asset", side_effect=fake_download_asset), \
             patch("app.documentary_pipeline.passes_visual_verification", side_effect=fake_verify), \
             patch.object(settings, "enable_ai_generation_fallback", True), \
             patch("app.documentary_pipeline.generate_fallback_image_openai", new=AsyncMock(return_value=(False, "mocked failure"))):
            entry, _ = asyncio.run(
                _resolve_clip(clip, Path(tmpdir), used_assets={}, allow_duplicates=False,
                              downloads_so_far=0, limit=None)
            )

    assert entry.generation_attempted is True
    assert entry.asset_metadata.source_id == "low"
    assert entry.asset_metadata.score == 58.0
    assert "AI generation attempted and failed" in entry.note


def test_rank_acceptable_assets_orders_by_score_no_video_preference():
    video = ScoredAsset(hit=_hit(source_id="v", media_type="video"), score=70)
    image = ScoredAsset(hit=_hit(source_id="i", media_type="image"), score=95)  # scores higher than the video
    below_min = ScoredAsset(hit=_hit(source_id="low", media_type="video"), score=1)

    with patch.object(settings, "documentary_min_score", 50):
        ranked = rank_acceptable_assets([video, below_min], [image])

    assert [a.hit.source_id for a in ranked] == ["i", "v"]  # image wins purely on score, below_min excluded


def test_media_type_bias_favors_underrepresented_image_and_flips_near_tie():
    video_heavy = {"image": 2, "video": 10}
    with patch.object(settings, "documentary_target_image_ratio", 0.65):
        bias = _media_type_bias(video_heavy)
    assert bias["image"] > 0
    assert bias["video"] < 0

    # Near-tie: video scores a hair higher on raw score_asset output, but the
    # image bias should be enough to flip which candidate ranks first.
    video = ScoredAsset(hit=_hit(source_id="v", media_type="video"), score=61)
    image = ScoredAsset(hit=_hit(source_id="i", media_type="image"), score=60)
    with patch.object(settings, "documentary_min_score", 50), \
         patch.object(settings, "documentary_target_image_ratio", 0.65):
        ranked = rank_acceptable_assets([video], [image], video_heavy)
    assert ranked[0].hit.source_id == "i"


def test_media_type_bias_cannot_override_a_large_quality_gap():
    video_heavy = {"image": 2, "video": 10}
    video = ScoredAsset(hit=_hit(source_id="v", media_type="video"), score=85)
    image = ScoredAsset(hit=_hit(source_id="i", media_type="image"), score=60)
    with patch.object(settings, "documentary_min_score", 50), \
         patch.object(settings, "documentary_target_image_ratio", 0.65):
        ranked = rank_acceptable_assets([video], [image], video_heavy)
    assert ranked[0].hit.source_id == "v"  # 25-point gap survives the +/-8 bias cap


def test_media_type_bias_never_excludes_the_only_available_candidate():
    # Running mix is already heavily video, so the bias favors images — but a
    # clip whose only acceptable candidate is a video must still resolve to it.
    video_heavy = {"image": 2, "video": 10}
    video = ScoredAsset(hit=_hit(source_id="only-video", media_type="video"), score=55)
    with patch.object(settings, "documentary_min_score", 50):
        ranked = rank_acceptable_assets([video], [], video_heavy)
    assert len(ranked) == 1
    assert ranked[0].hit.source_id == "only-video"


def test_pick_best_asset_returns_single_highest_scorer():
    videos = [ScoredAsset(hit=_hit(source_id="v0", media_type="video"), score=70),
              ScoredAsset(hit=_hit(source_id="v1", media_type="video"), score=95)]
    images = [ScoredAsset(hit=_hit(source_id="i0", media_type="image"), score=80)]
    best = pick_best_asset(videos, images)
    assert best.hit.source_id == "v1"  # single winner, not a list


def test_pick_best_asset_ignores_below_min_score():
    videos = [ScoredAsset(hit=_hit(source_id="v0", media_type="video"), score=10)]
    with patch.object(settings, "documentary_min_score", 50):
        assert pick_best_asset(videos, []) is None


def test_pick_best_asset_returns_none_when_no_candidates():
    assert pick_best_asset([], []) is None


def test_check_footage_availability_flags_thin_clips():
    # _clip()'s default canva_keyword/fallback_keyword ("k"/"f") are the same
    # for every clip number, so each clip needs its own distinct keywords
    # here for fake_search to tell them apart.
    good_clip = _clip(clip_number=1).model_copy(update={"canva_keyword": "goodkw", "fallback_keyword": "goodfb"})
    thin_clip = _clip(clip_number=2).model_copy(update={"canva_keyword": "thinkw", "fallback_keyword": "thinfb"})
    good_hit = _hit(source="pexels", source_id="good")

    async def fake_search(query, per_page=5):
        return [good_hit] if query in (good_clip.canva_keyword, good_clip.fallback_keyword) else []

    with patch_providers(), \
         patch("app.documentary_pipeline.pexels_video.search", new=fake_search), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]):
        report = asyncio.run(check_footage_availability([good_clip, thin_clip]))

    assert report.total_clips == 2
    assert report.thin_count == 1
    assert report.thin_clip_numbers == [2]
    assert [c.thin for c in report.clips] == [False, True]
    assert report.clips[0].candidate_count == 1
    assert report.clips[1].candidate_count == 0


def test_check_footage_availability_never_calls_llm_semantic_fallback():
    """Cost control: reaching 'no candidates from canva/fallback keywords'
    during the scan must NOT trigger the semantic-keyword LLM fallback (see
    search_clip's use_semantic_fallback=False) — that's a paid call this
    pre-render scan is deliberately not willing to spend."""
    clip = _clip()

    with patch_providers(), \
         patch("app.documentary_pipeline.generate_json") as mock_generate_json, \
         patch("app.scoring.embed", return_value=[1.0, 0.0]):
        report = asyncio.run(check_footage_availability([clip]))

    mock_generate_json.assert_not_called()
    assert report.thin_count == 1


def test_check_footage_availability_simulates_cross_clip_contention():
    """Two clips competing for the same single candidate: with the repeat
    cap forced down to 1, the scan's own simulated dedup must reflect that
    contention (clip 2 can't also count the same asset as available) instead
    of double-counting it as 'available' for both clips."""
    clip1 = _clip(clip_number=1)
    clip2 = _clip(clip_number=2)
    hit = _hit(source="pexels", source_id="scarce")

    with patch_providers(pexels_video=[hit]), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]), \
         patch.object(settings, "max_asset_repeat_count", 1):
        report = asyncio.run(check_footage_availability([clip1, clip2]))

    assert report.clips[0].candidate_count == 1
    assert report.clips[0].thin is False
    assert report.clips[1].candidate_count == 0  # scarce already claimed by clip 1's simulated dedup
    assert report.clips[1].thin is True


def test_resolve_script_text_reads_from_path_and_rejects_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        script_file = Path(tmpdir) / "s.txt"
        script_file.write_text("real script text", encoding="utf-8")
        assert resolve_script_text("", str(script_file)) == "real script text"

    try:
        resolve_script_text("   ", None)
        raise AssertionError("expected ValueError for empty script")
    except ValueError as exc:
        assert "non-empty script text" in str(exc)


def test_plan_documentary_returns_table_and_availability_report():
    clip = _clip()

    with patch("app.documentary_pipeline.generate_table", return_value=[clip]), \
         patch_providers(), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]):
        plan = asyncio.run(plan_documentary("some script", "proj_x"))

    assert plan.project_name == "proj_x"
    assert plan.table == [clip]
    assert plan.whisper_words is None
    assert plan.transcript_alignment_warning is None
    assert plan.footage_availability.total_clips == 1


def test_cross_clip_dedup_excludes_used_asset_unless_allowed():
    """At or above the repeat cap (settings.max_asset_repeat_count), an
    asset is excluded same as the old all-or-nothing dedup — see
    test_cross_clip_dedup_allows_reuse_up_to_repeat_cap for the new
    under-the-cap behavior this replaces. count=2 is at/above the cap
    regardless of whether the cap is 1 (current default) or 2, so this
    test needs no explicit patch."""
    clip = _clip()
    hit = _hit(source="pexels", source_id="dup1")
    used = {("pexels", "dup1"): 2}

    with patch_providers(pexels_video=[hit]), patch("app.scoring.embed", return_value=[1.0, 0.0]):
        video_hits, _ = asyncio.run(search_clip(clip, used_assets=used, allow_duplicates=False))
    assert video_hits == []

    with patch_providers(pexels_video=[hit]), patch("app.scoring.embed", return_value=[1.0, 0.0]):
        video_hits, _ = asyncio.run(search_clip(clip, used_assets=used, allow_duplicates=True))
    assert len(video_hits) == 1


def test_cross_clip_dedup_allows_reuse_up_to_repeat_cap():
    """New bounded-repeat behavior: an asset used once (count 1) is still
    below a cap of 2, so it must remain selectable — only fully excluded
    once it actually hits the cap (see the test above). Forces the cap to 2
    explicitly since production default is now 1 (true never-repeat, see
    settings.max_asset_repeat_count) — this test is about the bounded-cap
    mechanism itself, not the current default value."""
    clip = _clip()
    hit = _hit(source="pexels", source_id="dup1")
    used = {("pexels", "dup1"): 1}

    with patch_providers(pexels_video=[hit]), patch("app.scoring.embed", return_value=[1.0, 0.0]), \
         patch.object(settings, "max_asset_repeat_count", 2):
        video_hits, _ = asyncio.run(search_clip(clip, used_assets=used, allow_duplicates=False))
    assert len(video_hits) == 1


def test_search_clip_advances_to_next_tier_when_all_hits_from_a_tier_are_already_used():
    """FIX2 gap: a keyword tier whose raw hits are non-empty but EVERY one
    is already at the repeat cap must be treated the same as an empty tier
    and escalate to the next keyword tier (canva -> fallback -> semantic),
    instead of stopping there (because raw_hits was technically non-empty)
    and silently contributing zero real candidates for this provider."""
    clip = _clip()
    used_hit = _hit(source="pexels", source_id="dup", text="")
    fresh_hit = _hit(source="pexels", source_id="fresh", text="")
    used = {("pexels", "dup"): 2}  # at the cap

    async def fake_pexels_search(query, per_page=5):
        # canva_keyword ("k") returns ONLY the already-used hit; fallback_keyword
        # ("f") returns a fresh, never-used one.
        return [used_hit] if query == clip.canva_keyword else [fresh_hit]

    with patch_providers(), \
         patch("app.documentary_pipeline.pexels_video.search", new=fake_pexels_search), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]):
        video_hits, _ = asyncio.run(search_clip(clip, used_assets=used, allow_duplicates=False))

    assert len(video_hits) == 1
    assert video_hits[0].hit.source_id == "fresh"


def test_search_clip_widens_to_fallback_when_canva_keyword_scores_below_threshold():
    """The actual gap this fix addresses: canva_keyword returns REAL hits (not
    zero, not all-already-used), but none of them score >= documentary_min_score
    once scored. fallback_keyword must now ALSO be queried and merged into the
    same pool, instead of the old behavior of stopping as soon as canva_keyword
    returned *anything*, regardless of how badly it scored."""
    clip = _clip(visual_type="Cinematic")
    low_score_hit = _hit(source="pexels", source_id="low", media_type="image", width=0, height=0, text="")
    good_hit = _hit(source="pixabay", source_id="good", media_type="video", width=1920, height=1080, text="t")

    async def fake_search(query, per_page=5):
        if query == clip.canva_keyword:
            return [low_score_hit]
        if query == clip.fallback_keyword:
            return [good_hit]
        return []

    with patch_providers(), \
         patch("app.documentary_pipeline.pexels_video.search", new=fake_search), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]):
        video_hits, image_hits = asyncio.run(search_clip(clip, used_assets={}))

    all_ids = {a.hit.source_id for a in video_hits + image_hits}
    assert "low" in all_ids  # canva_keyword's (weak) candidate is not discarded
    assert "good" in all_ids  # fallback_keyword's candidate was also fetched and merged in


def test_search_clip_merged_pool_ranks_best_candidate_regardless_of_tier():
    """The merge must not prefer canva_keyword's own winner just because it
    arrived first — the final ranking is purely by score across the whole
    merged pool, so fallback_keyword's candidate can legitimately outrank
    canva_keyword's."""
    clip = _clip(visual_type="Cinematic")
    # Both score under documentary_min_score(50) by construction (triggering
    # the merge), but fallback_keyword's hit is the clearly stronger of the two.
    weak_canva_hit = _hit(source="pexels", source_id="weak", media_type="image", width=640, height=480, text="")
    strong_fallback_hit = _hit(source="pixabay", source_id="strong", media_type="image", width=1920, height=1080, text="")

    async def fake_search(query, per_page=5):
        if query == clip.canva_keyword:
            return [weak_canva_hit]
        if query == clip.fallback_keyword:
            return [strong_fallback_hit]
        return []

    with patch_providers(), \
         patch("app.documentary_pipeline.pexels_video.search", new=fake_search), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]):
        video_hits, image_hits = asyncio.run(search_clip(clip, used_assets={}))

    all_scored = sorted(video_hits + image_hits, key=lambda a: a.score, reverse=True)
    assert all_scored[0].hit.source_id == "strong"  # fallback_keyword's hit outranks canva_keyword's, despite arriving second


def test_resolve_clip_reuses_asset_up_to_repeat_cap_then_falls_through():
    """End-to-end for the bounded repeat cap (settings.max_asset_repeat_count):
    three rows whose keywords all return the same overlapping stock results,
    with dupA always the higher-scored candidate. Row 1 and row 2 both win
    dupA (reuse up to the cap — the anti-fatigue behavior this change adds,
    replacing the old all-or-nothing dedup that would have forced row 2 to
    dupB immediately). Row 3 finds dupA at the cap and falls through to dupB,
    sharing one used_assets counter across all three calls exactly like the
    real run() loop does. Forces the cap to 2 explicitly since production
    default is now 1 (true never-repeat) — this test is about the bounded-cap
    mechanism itself, not the current default value."""
    import subprocess

    clip1 = _clip(clip_number=1)
    clip2 = _clip(clip_number=2)
    clip3 = _clip(clip_number=3)
    hit_a = _hit(source="pexels", source_id="dupA", width=1920, height=1080)
    hit_b = _hit(source="pexels", source_id="dupB", width=1920, height=1080)

    def fake_score(hit, clip, query_embedding):
        return 90.0 if hit.source_id == "dupA" else 50.0  # dupA always the top-scored candidate

    with tempfile.TemporaryDirectory() as tmpdir:
        clips_root = Path(tmpdir)
        path_a = clips_root / "a.mp4"
        path_b = clips_root / "b.mp4"
        for path in (path_a, path_b):
            # d=5 matches _clip()'s default 5.0s target duration exactly — this test
            # is about the used_assets repeat-cap mechanism, not fill-clip logic
            # (see test_resolve_clip_fetches_distinct_fill_asset_when_primary_too_short
            # for that); a shorter source would now also trigger a fill-candidate
            # fetch here, consuming dupB before row 3's dedup fallback gets to it.
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=1920x1080:r=25:d=5",
                 "-pix_fmt", "yuv420p", str(path)],
                capture_output=True, check=True,
            )

        asset_a = AssetInfo(path=str(path_a), source="pexels", source_id="dupA", media_type="video",
                             width=1920, height=1080, duration=5.0, score=90.0)
        asset_b = AssetInfo(path=str(path_b), source="pexels", source_id="dupB", media_type="video",
                             width=1920, height=1080, duration=5.0, score=50.0)

        async def fake_download(candidate, dest_dir, prefix, index, target_duration):
            return asset_a if candidate.hit.source_id == "dupA" else asset_b

        used_assets: dict[tuple[str, str], int] = {}

        with patch_providers(pexels_video=[hit_a, hit_b]), \
             patch("app.documentary_pipeline.score_asset", side_effect=fake_score), \
             patch("app.documentary_pipeline._download_asset", new=AsyncMock(side_effect=fake_download)), \
             patch("app.documentary_pipeline.passes_visual_verification", return_value=(True, 0.5)), \
             patch.object(settings, "max_asset_repeat_count", 2):
            entry1, _ = asyncio.run(
                _resolve_clip(clip1, clips_root, used_assets, allow_duplicates=False, downloads_so_far=0, limit=None)
            )
            entry2, _ = asyncio.run(
                _resolve_clip(clip2, clips_root, used_assets, allow_duplicates=False, downloads_so_far=0, limit=None)
            )
            entry3, _ = asyncio.run(
                _resolve_clip(clip3, clips_root, used_assets, allow_duplicates=False, downloads_so_far=0, limit=None)
            )

    assert entry1.asset_metadata.source_id == "dupA"  # row 1: the higher-scored candidate wins
    assert entry2.asset_metadata.source_id == "dupA"  # row 2: still under the cap (1 < 2) -> reused
    assert entry3.asset_metadata.source_id == "dupB"  # row 3: dupA hit the cap (2) -> falls through to dupB


def test_safe_search_swallows_errors_and_timeouts():
    class RaisesHttpError:
        async def search(self, query, per_page=5):
            raise httpx.ConnectError("boom")

    class Slow:
        async def search(self, query, per_page=5):
            await asyncio.sleep(1)
            return ["unreachable"]

    # None (not []) on a real failure/timeout — distinct from a genuinely
    # empty result, so search_clip can stop retrying a provider that's
    # actually unreachable instead of burning the full timeout on every
    # remaining keyword for that clip.
    assert asyncio.run(_safe_search(RaisesHttpError(), "q")) is None
    assert asyncio.run(_safe_search(Slow(), "q", timeout=0.01)) is None


def test_generate_semantic_keywords_regenerates_once_when_first_attempt_is_abstract():
    """Reuses the exact abstract phrases from a real production log
    ("invisible hand rising", "silent giant emerging", "shadow market
    control") to prove the concreteness self-check catches them and
    triggers exactly one regeneration, landing on literal/filmable queries."""
    clip = _clip(visual_type="Industry").model_copy(update={
        "script_beat": "That invisibility was not an accident. It was the foundational "
                       "architecture of Microsoft's entire market dominance.",
    })
    abstract_response = {"keywords": ["invisible hand rising", "silent giant emerging", "shadow market control"]}
    concreteness_check = {"concrete": [False, False, False]}
    concrete_response = {"keywords": ["corporate office building exterior", "tech company headquarters",
                                       "business executives in boardroom meeting"]}

    with patch("app.documentary_pipeline.generate_json",
               side_effect=[abstract_response, concreteness_check, concrete_response]) as mock_generate:
        keywords = _generate_semantic_keywords(clip, tried=["k1", "f1"])

    assert keywords == concrete_response["keywords"]
    assert mock_generate.call_count == 3  # generate, self-check, regenerate


def test_generate_semantic_keywords_skips_regeneration_when_first_attempt_is_already_concrete():
    clip = _clip(visual_type="Industry")
    concrete_response = {"keywords": ["office building exterior", "server room with racks", "delivery truck on highway"]}
    concreteness_check = {"concrete": [True, True, True]}

    with patch("app.documentary_pipeline.generate_json",
               side_effect=[concrete_response, concreteness_check]) as mock_generate:
        keywords = _generate_semantic_keywords(clip, tried=[])

    assert keywords == concrete_response["keywords"]
    assert mock_generate.call_count == 2  # generate, self-check only — no regeneration needed


def test_generate_semantic_keywords_replaces_niche_violating_keywords():
    clip = _clip(visual_type="Industry")
    response = {"keywords": ["car driving on highway", "sedan in traffic", "semi truck at loading dock"]}
    concreteness_check = {"concrete": [True, True, True]}

    with patch("app.documentary_pipeline.generate_json", side_effect=[response, concreteness_check]):
        keywords = _generate_semantic_keywords(clip, tried=[])

    niche_config = get_niche(DEFAULT_NICHE)
    assert keywords == [
        niche_config.safe_fallback_keyword, niche_config.safe_fallback_keyword, "semi truck at loading dock",
    ]


def test_keywords_are_concrete_defaults_to_true_when_self_check_call_fails():
    """The self-check is a soft guard, not a hard gate — if the LLM call
    itself fails, assume the keywords are fine rather than blocking search."""
    with patch("app.documentary_pipeline.generate_json", side_effect=RuntimeError("network down")):
        assert _keywords_are_concrete(["some keyword"]) is True


def test_resolve_clip_placeholder_note_reflects_exhaustive_search():
    """The placeholder note for the 'nothing cleared the threshold' case
    must reflect that the full provider list and all 3 keyword tiers were
    actually exhausted, not the old generic 'no acceptable asset found
    across all providers' wording."""
    clip = _clip()  # default content_niche="trucks" -> internet_archive excluded, local_library excluded -> 6 providers
    with tempfile.TemporaryDirectory() as tmpdir, \
         patch_providers(), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]):
        entry, _ = asyncio.run(_resolve_clip(
            clip, Path(tmpdir), used_assets={}, allow_duplicates=False,
            downloads_so_far=0, limit=None,
        ))

    assert entry.placeholder is True
    assert "no candidate scoring >= 50 found" in entry.note
    assert "6 providers" in entry.note
    assert "3 keyword tiers" in entry.note


def test_resolve_clip_uses_ai_generation_fallback_when_search_is_exhausted_and_enabled():
    """No real candidate at all is one of the two trigger conditions (the
    other being a real candidate that scores below the threshold — see the
    tests below) — fires only when the feature is explicitly enabled, see
    settings.enable_ai_generation_fallback's licensing note."""
    clip = _clip()
    with tempfile.TemporaryDirectory() as tmpdir, \
         patch_providers(), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]), \
         patch.object(settings, "enable_ai_generation_fallback", True), \
         patch("app.documentary_pipeline.generate_fallback_image_openai", new=AsyncMock(return_value=(True, None))), \
         patch("app.documentary_pipeline.probe", return_value=(0.0, "2048x1152")), \
         patch("app.documentary_pipeline.normalize_to_16_9", return_value=True):
        entry, _ = asyncio.run(_resolve_clip(
            clip, Path(tmpdir), used_assets={}, allow_duplicates=False,
            downloads_so_far=0, limit=None,
        ))

    assert entry.placeholder is False
    assert entry.asset_metadata.source == "ai_generated"
    assert entry.asset_metadata.score is None
    assert entry.asset_metadata.width == 1920 and entry.asset_metadata.height == 1080
    assert f"AI-generated via OpenAI {settings.ai_generation_model}" in entry.note


def test_resolve_clip_registers_ai_generated_asset_in_used_assets_after_pre_download_trigger():
    """The pre-download AI-generation accept path (best_real_candidate is
    None or below threshold, generation succeeds, returns immediately) must
    register its generated asset in used_assets too, same as the real-asset
    download loop does — previously this path returned without ever
    touching used_assets."""
    clip = _clip()
    used_assets: dict[tuple[str, str], int] = {}
    with tempfile.TemporaryDirectory() as tmpdir, \
         patch_providers(), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]), \
         patch.object(settings, "enable_ai_generation_fallback", True), \
         patch("app.documentary_pipeline.generate_fallback_image_openai", new=AsyncMock(return_value=(True, None))), \
         patch("app.documentary_pipeline.probe", return_value=(0.0, "2048x1152")), \
         patch("app.documentary_pipeline.normalize_to_16_9", return_value=True):
        entry, _ = asyncio.run(_resolve_clip(
            clip, Path(tmpdir), used_assets=used_assets, allow_duplicates=False,
            downloads_so_far=0, limit=None,
        ))

    assert entry.asset_metadata.source == "ai_generated"
    key = (entry.asset_metadata.source, entry.asset_metadata.source_id)
    assert used_assets.get(key) == 1


def test_resolve_clip_registers_ai_generated_asset_in_used_assets_after_last_resort_trigger():
    """Same registration requirement for the NEW last-resort trigger (fires
    when a real candidate passed the pre-download check but then failed
    post-download CLIP verification, with nothing else left)."""
    clip = _clip()
    hit = _hit(source="local_library", source_id="tactical_officers_prison_yard")
    video_hits = [ScoredAsset(hit=hit, score=95.0)]

    async def fake_download_asset(asset, dest_dir, prefix, index, target_duration):
        return AssetInfo(path="real.mp4", source="local_library", source_id="tactical_officers_prison_yard",
                          media_type="video", width=1920, height=1080, duration=5.0, score=95.0)

    used_assets: dict[tuple[str, str], int] = {}
    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.search_clip", new=AsyncMock(return_value=(video_hits, []))), \
         patch("app.documentary_pipeline._download_asset", side_effect=fake_download_asset), \
         patch("app.documentary_pipeline.probe", return_value=(5.0, "1920x1080")), \
         patch("app.documentary_pipeline.passes_visual_verification", return_value=(False, 0.193)), \
         patch.object(settings, "enable_ai_generation_fallback", True), \
         patch("app.documentary_pipeline.generate_fallback_image_openai", new=AsyncMock(return_value=(True, None))), \
         patch("app.documentary_pipeline.generate_json", return_value={"prompt": "enhanced prompt"}), \
         patch("app.documentary_pipeline.normalize_to_16_9", return_value=True):
        entry, _ = asyncio.run(_resolve_clip(
            clip, Path(tmpdir), used_assets=used_assets, allow_duplicates=False,
            downloads_so_far=0, limit=None,
        ))

    assert entry.asset_metadata.source == "ai_generated"
    key = (entry.asset_metadata.source, entry.asset_metadata.source_id)
    assert used_assets.get(key) == 1
    # the real candidate was downloaded, rejected, and unlinked -> also
    # correctly registered (by the normal download-loop path) so it can't
    # be picked again either.
    assert used_assets.get(("local_library", "tactical_officers_prison_yard")) == 1


def test_two_ai_generated_clips_get_distinct_used_assets_keys():
    """AI-generated images have no real provider source_id — confirm the
    dedup key is derived from each image's own unique output path instead
    of a shared empty string, so two different clips' generated images
    don't collide on ("ai_generated", "") and don't silently bypass
    used_assets tracking either."""
    clip1 = _clip(clip_number=1)
    clip2 = _clip(clip_number=2)
    used_assets: dict[tuple[str, str], int] = {}

    async def resolve(clip):
        with tempfile.TemporaryDirectory() as tmpdir, \
             patch_providers(), \
             patch("app.scoring.embed", return_value=[1.0, 0.0]), \
             patch.object(settings, "enable_ai_generation_fallback", True), \
             patch("app.documentary_pipeline.generate_fallback_image_openai", new=AsyncMock(return_value=(True, None))), \
             patch("app.documentary_pipeline.probe", return_value=(0.0, "2048x1152")), \
             patch("app.documentary_pipeline.normalize_to_16_9", return_value=True):
            entry, _ = await _resolve_clip(
                clip, Path(tmpdir), used_assets=used_assets, allow_duplicates=False,
                downloads_so_far=0, limit=None,
            )
        return entry

    async def run_both():
        return [await resolve(clip1), await resolve(clip2)]

    entry1, entry2 = asyncio.run(run_both())

    key1 = (entry1.asset_metadata.source, entry1.asset_metadata.source_id)
    key2 = (entry2.asset_metadata.source, entry2.asset_metadata.source_id)
    assert key1 != key2
    assert used_assets.get(key1) == 1
    assert used_assets.get(key2) == 1


def test_resolve_clip_falls_through_to_placeholder_when_ai_generation_fails():
    clip = _clip()
    with tempfile.TemporaryDirectory() as tmpdir, \
         patch_providers(), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]), \
         patch.object(settings, "enable_ai_generation_fallback", True), \
         patch("app.documentary_pipeline.generate_fallback_image_openai", new=AsyncMock(return_value=(False, "connection reset"))):
        entry, _ = asyncio.run(_resolve_clip(
            clip, Path(tmpdir), used_assets={}, allow_duplicates=False,
            downloads_so_far=0, limit=None,
        ))

    assert entry.placeholder is True
    assert entry.asset_metadata is None
    assert entry.generation_attempted is True
    assert entry.generation_failure_reason == "connection reset"


def test_resolve_clip_falls_through_to_placeholder_when_normalize_to_16_9_fails():
    """Generated images are always run through normalize_to_16_9 as a safety
    net regardless of the resolved model's output size (see
    _try_ai_generation_fallback) — an ffmpeg failure there must degrade the
    same way a generation failure does, not raise or silently accept the
    un-normalized image."""
    clip = _clip()
    with tempfile.TemporaryDirectory() as tmpdir, \
         patch_providers(), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]), \
         patch.object(settings, "enable_ai_generation_fallback", True), \
         patch("app.documentary_pipeline.generate_fallback_image_openai", new=AsyncMock(return_value=(True, None))), \
         patch("app.documentary_pipeline.probe", return_value=(0.0, "2048x1152")), \
         patch("app.documentary_pipeline.normalize_to_16_9", return_value=False):
        entry, _ = asyncio.run(_resolve_clip(
            clip, Path(tmpdir), used_assets={}, allow_duplicates=False,
            downloads_so_far=0, limit=None,
        ))

    assert entry.placeholder is True
    assert entry.asset_metadata is None
    assert entry.generation_attempted is True
    assert entry.generation_failure_reason == "failed to normalize AI-generated image to 16:9"


def test_resolve_clip_skips_ai_generation_fallback_when_disabled_by_default():
    """settings.enable_ai_generation_fallback defaults to False (unconfirmed
    commercial licensing) — generation must never even be attempted unless
    explicitly turned on."""
    clip = _clip()
    with tempfile.TemporaryDirectory() as tmpdir, \
         patch_providers(), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]), \
         patch("app.documentary_pipeline.generate_fallback_image_openai", new=AsyncMock(return_value=(True, None))) as mock_generate:
        entry, _ = asyncio.run(_resolve_clip(
            clip, Path(tmpdir), used_assets={}, allow_duplicates=False,
            downloads_so_far=0, limit=None,
        ))

    mock_generate.assert_not_called()
    assert entry.placeholder is True


def test_ai_generation_fallback_triggers_when_best_real_candidate_scores_below_threshold():
    """The revised trigger condition: generation fires whenever the BEST
    real candidate scores below ai_generation_trigger_threshold, even though
    a real candidate exists — not only when real search is fully empty."""
    clip = _clip()
    hit = _hit(source="pexels", source_id="low")
    video_hits = [ScoredAsset(hit=hit, score=65.0)]  # below default threshold (75)

    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.search_clip", new=AsyncMock(return_value=(video_hits, []))), \
         patch.object(settings, "enable_ai_generation_fallback", True), \
         patch("app.documentary_pipeline.generate_fallback_image_openai", new=AsyncMock(return_value=(True, None))) as mock_generate, \
         patch("app.documentary_pipeline.generate_json", return_value={"prompt": "enhanced prompt"}), \
         patch("app.documentary_pipeline.probe", return_value=(0.0, "2048x1152")), \
         patch("app.documentary_pipeline.normalize_to_16_9", return_value=True):
        entry, _ = asyncio.run(_resolve_clip(
            clip, Path(tmpdir), used_assets={}, allow_duplicates=False,
            downloads_so_far=0, limit=None,
        ))

    mock_generate.assert_called_once()
    assert entry.placeholder is False
    assert entry.asset_metadata.source == "ai_generated"
    assert "65.0" in entry.note
    assert f"{settings.ai_generation_trigger_threshold:.1f}" in entry.note


def test_ai_generation_fallback_does_not_trigger_when_best_real_candidate_meets_threshold():
    """A real candidate scoring at/above the trigger threshold is accepted
    normally — generation must not even be attempted."""
    clip = _clip()
    hit = _hit(source="pexels", source_id="good")
    video_hits = [ScoredAsset(hit=hit, score=80.0)]  # at/above default threshold (75)

    async def fake_download_asset(asset, dest_dir, prefix, index, target_duration):
        return AssetInfo(path="real.mp4", source="pexels", source_id="good",
                          media_type="video", width=1920, height=1080, duration=5.0, score=80.0)

    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.search_clip", new=AsyncMock(return_value=(video_hits, []))), \
         patch("app.documentary_pipeline._download_asset", side_effect=fake_download_asset), \
         patch("app.documentary_pipeline.probe", return_value=(5.0, "1920x1080")), \
         patch("app.documentary_pipeline.passes_visual_verification", return_value=(True, 0.5)), \
         patch.object(settings, "enable_ai_generation_fallback", True), \
         patch("app.documentary_pipeline.generate_fallback_image_openai", new=AsyncMock(return_value=(True, None))) as mock_generate:
        entry, _ = asyncio.run(_resolve_clip(
            clip, Path(tmpdir), used_assets={}, allow_duplicates=False,
            downloads_so_far=0, limit=None,
        ))

    mock_generate.assert_not_called()
    assert entry.placeholder is False
    assert entry.asset_metadata.source == "pexels"
    assert entry.generation_attempted is False


def test_ai_generation_fallback_degrades_gracefully_to_real_candidate_when_generation_fails():
    """If the below-threshold real candidate exists but generation itself
    fails (network error etc.), the pipeline must accept the real (if
    imperfect) candidate rather than giving up to a placeholder."""
    clip = _clip()
    hit = _hit(source="pexels", source_id="low")
    video_hits = [ScoredAsset(hit=hit, score=65.0)]  # below default threshold (75)

    async def fake_download_asset(asset, dest_dir, prefix, index, target_duration):
        return AssetInfo(path="real.mp4", source="pexels", source_id="low",
                          media_type="video", width=1920, height=1080, duration=5.0, score=65.0)

    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.search_clip", new=AsyncMock(return_value=(video_hits, []))), \
         patch("app.documentary_pipeline._download_asset", side_effect=fake_download_asset), \
         patch("app.documentary_pipeline.probe", return_value=(5.0, "1920x1080")), \
         patch("app.documentary_pipeline.passes_visual_verification", return_value=(True, 0.5)), \
         patch.object(settings, "enable_ai_generation_fallback", True), \
         patch("app.documentary_pipeline.generate_fallback_image_openai", new=AsyncMock(return_value=(False, "connection reset"))) as mock_generate, \
         patch("app.documentary_pipeline.generate_json", return_value={"prompt": "enhanced prompt"}):
        entry, _ = asyncio.run(_resolve_clip(
            clip, Path(tmpdir), used_assets={}, allow_duplicates=False,
            downloads_so_far=0, limit=None,
        ))

    mock_generate.assert_called_once()
    assert entry.placeholder is False
    assert entry.asset_metadata.source == "pexels"  # fell back to the real candidate, not a placeholder
    assert entry.generation_attempted is True
    assert entry.generation_failure_reason == "connection reset"


def test_resolve_clip_tries_ai_generation_as_last_resort_when_candidate_passes_pre_download_but_fails_clip():
    """Clip 7 ('Prison service debt') regression: the only real candidate
    scored above ai_generation_trigger_threshold pre-download (so
    _should_try_ai_generation never fired), but then failed CLIP
    verification post-download with no other candidates left. Previously
    this fell straight to a placeholder since the generation decision was
    only ever made once, before download. Now a second trigger point right
    before the final placeholder must catch this and generate instead."""
    clip = _clip()
    hit = _hit(source="local_library", source_id="tactical_officers_prison_yard")
    video_hits = [ScoredAsset(hit=hit, score=95.0)]  # well above default threshold (75)

    async def fake_download_asset(asset, dest_dir, prefix, index, target_duration):
        return AssetInfo(path="real.mp4", source="local_library", source_id="tactical_officers_prison_yard",
                          media_type="video", width=1920, height=1080, duration=5.0, score=95.0)

    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.search_clip", new=AsyncMock(return_value=(video_hits, []))), \
         patch("app.documentary_pipeline._download_asset", side_effect=fake_download_asset), \
         patch("app.documentary_pipeline.probe", return_value=(5.0, "1920x1080")), \
         patch("app.documentary_pipeline.passes_visual_verification", return_value=(False, 0.193)), \
         patch.object(settings, "enable_ai_generation_fallback", True), \
         patch("app.documentary_pipeline.generate_fallback_image_openai", new=AsyncMock(return_value=(True, None))) as mock_generate, \
         patch("app.documentary_pipeline.generate_json", return_value={"prompt": "enhanced prompt"}), \
         patch("app.documentary_pipeline.normalize_to_16_9", return_value=True):
        entry, _ = asyncio.run(_resolve_clip(
            clip, Path(tmpdir), used_assets={}, allow_duplicates=False,
            downloads_so_far=0, limit=None,
        ))

    mock_generate.assert_called_once()  # never attempted pre-download (score was above threshold) -> exactly one attempt, here
    assert entry.placeholder is False
    assert entry.asset_metadata.source == "ai_generated"
    assert entry.generation_attempted is True
    assert "post-verification" in entry.note


def test_resolve_clip_falls_through_to_placeholder_when_last_resort_ai_generation_also_fails():
    clip = _clip()
    hit = _hit(source="local_library", source_id="tactical_officers_prison_yard")
    video_hits = [ScoredAsset(hit=hit, score=95.0)]

    async def fake_download_asset(asset, dest_dir, prefix, index, target_duration):
        return AssetInfo(path="real.mp4", source="local_library", source_id="tactical_officers_prison_yard",
                          media_type="video", width=1920, height=1080, duration=5.0, score=95.0)

    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.search_clip", new=AsyncMock(return_value=(video_hits, []))), \
         patch("app.documentary_pipeline._download_asset", side_effect=fake_download_asset), \
         patch("app.documentary_pipeline.probe", return_value=(5.0, "1920x1080")), \
         patch("app.documentary_pipeline.passes_visual_verification", return_value=(False, 0.193)), \
         patch.object(settings, "enable_ai_generation_fallback", True), \
         patch("app.documentary_pipeline.generate_fallback_image_openai", new=AsyncMock(return_value=(False, "connection reset"))), \
         patch("app.documentary_pipeline.generate_json", return_value={"prompt": "enhanced prompt"}):
        entry, _ = asyncio.run(_resolve_clip(
            clip, Path(tmpdir), used_assets={}, allow_duplicates=False,
            downloads_so_far=0, limit=None,
        ))

    assert entry.placeholder is True
    assert entry.generation_attempted is True
    assert entry.generation_failure_reason == "connection reset"


def test_resolve_clip_last_resort_ai_generation_skipped_when_disabled():
    """Same exhausted-candidate scenario as the regression test above, but
    with the feature flag off (default) — must still degrade to a
    placeholder, not attempt generation."""
    clip = _clip()
    hit = _hit(source="local_library", source_id="tactical_officers_prison_yard")
    video_hits = [ScoredAsset(hit=hit, score=95.0)]

    async def fake_download_asset(asset, dest_dir, prefix, index, target_duration):
        return AssetInfo(path="real.mp4", source="local_library", source_id="tactical_officers_prison_yard",
                          media_type="video", width=1920, height=1080, duration=5.0, score=95.0)

    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.search_clip", new=AsyncMock(return_value=(video_hits, []))), \
         patch("app.documentary_pipeline._download_asset", side_effect=fake_download_asset), \
         patch("app.documentary_pipeline.probe", return_value=(5.0, "1920x1080")), \
         patch("app.documentary_pipeline.passes_visual_verification", return_value=(False, 0.193)), \
         patch("app.documentary_pipeline.generate_fallback_image_openai", new=AsyncMock(return_value=(True, None))) as mock_generate:
        entry, _ = asyncio.run(_resolve_clip(
            clip, Path(tmpdir), used_assets={}, allow_duplicates=False,
            downloads_so_far=0, limit=None,
        ))

    mock_generate.assert_not_called()
    assert entry.placeholder is True
    assert entry.generation_attempted is False


def test_render_keyword_match_report_labels_ai_generated_entries():
    table = [_clip(clip_number=1, canva_keyword="panic pool owner")]
    entry = TimelineEntry(
        clip_number=1, start="00:00:00", end="00:00:05", duration=5.0, script="beat",
        asset_path="generated.jpg", asset_metadata=AssetInfo(
            path="generated.jpg", source="ai_generated", source_id="", media_type="image",
            score=None, selected_text="(AI-generated via OpenAI gpt-image-1-mini)",
        ), recommended_effect="e", transition="t", placeholder=False,
    )

    report = render_keyword_match_report(table, [entry])

    assert "(AI-generated via OpenAI gpt-image-1-mini)" in report
    assert "ai_generated" in report


def test_render_keyword_match_report_distinguishes_real_after_failed_generation_from_direct_accept():
    """A clip whose score cleared the threshold (generation never attempted)
    and a clip whose score was below threshold but generation failed (real
    candidate accepted as fallback) both end up with a real asset_metadata —
    previously indistinguishable in the report. The Outcome column must
    tell them apart."""
    table = [
        _clip(clip_number=1, canva_keyword="clear pool water"),
        _clip(clip_number=2, canva_keyword="borax powder pool"),
    ]
    direct_accept = TimelineEntry(
        clip_number=1, start="00:00:00", end="00:00:05", duration=5.0, script="beat one",
        asset_path="a.mp4", asset_metadata=AssetInfo(
            path="a.mp4", source="pexels", source_id="1", media_type="video",
            score=80.0, selected_text="clear pool water",
        ), recommended_effect="e", transition="t", placeholder=False,
    )
    failed_generation_fallback = TimelineEntry(
        clip_number=2, start="00:00:05", end="00:00:10", duration=5.0, script="beat two",
        asset_path="b.mp4", asset_metadata=AssetInfo(
            path="b.mp4", source="pexels", source_id="2", media_type="video",
            score=68.1, selected_text="adding powder to glass of water",
        ), recommended_effect="e", transition="t", placeholder=False,
        generation_attempted=True, generation_failure_reason="response too small (0 bytes)",
    )

    report = render_keyword_match_report(table, [direct_accept, failed_generation_fallback])

    assert "| real |" in report
    assert "real (generation failed: response too small (0 bytes))" in report


def test_run_never_fails_a_clip_with_no_assets():
    clip = _clip()
    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.generate_table", return_value=[clip]), \
         patch.object(settings, "documentary_projects_dir", Path(tmpdir)), \
         patch_providers(), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]), \
         patch("app.documentary_pipeline.decide_style", return_value=StyleDecision()), \
         patch("app.documentary_pipeline.render_final_video", return_value=Path(tmpdir) / "final_video_remotion.mp4"):
        result = asyncio.run(run("some script", project_name="empty_proj"))

    assert len(result.timeline) == 1
    entry = result.timeline[0]
    assert entry.placeholder is True
    assert entry.asset_path is None
    assert entry.note


def test_generate_and_edit_renders_per_clip_segments_without_final_concat():
    """Mode A ("Generate & Edit") must produce a real per-clip rendered
    segment for every clip — what the editor's preview playback and
    single-clip re-render both depend on — WITHOUT running the final ffmpeg
    concat pass (that produces final_video.mp4, which generate_full_video()
    re-renders from scratch anyway once the user actually asks for it)."""
    clip = _clip()
    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.generate_table", return_value=[clip]), \
         patch.object(settings, "documentary_projects_dir", Path(tmpdir)), \
         patch_providers(), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]):
        result = asyncio.run(generate_and_edit("some script", project_name="mode_a_proj"))

        assert result.final_video_path is None
        assert result.assembly_error is None
        assert not (Path(tmpdir) / "mode_a_proj" / "final_video.mp4").exists()
        entry = result.timeline[0]
        assert entry.rendered_clip_path
        assert Path(entry.rendered_clip_path).exists()


def test_generate_and_edit_then_single_clip_rerender_succeeds_immediately():
    """Regression test for the real reported bug: Mode A used to call the
    FULL assemble_video() (per-clip render + final concat/duration/
    resolution validation as one all-or-nothing function) — any failure in
    the concat-only portion discarded every row's rendered_clip_path, not
    just the failing step's own output, so the very next PATCH
    .../clip/{N} request in the editor failed with "clip N has no existing
    rendered segment on disk". After generate_and_edit(), every clip must
    already be re-renderable without hitting that error."""
    clips = [_clip(clip_number=1, start="00:00:00", end="00:00:05"),
             _clip(clip_number=2, start="00:00:05", end="00:00:10")]
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("app.documentary_pipeline.generate_table", return_value=clips), \
             patch.object(settings, "documentary_projects_dir", Path(tmpdir)), \
             patch_providers(), \
             patch("app.scoring.embed", return_value=[1.0, 0.0]):
            result = asyncio.run(generate_and_edit("some script", project_name="mode_a_two_clip_proj"))

        assert all(e.rendered_clip_path and Path(e.rendered_clip_path).exists() for e in result.timeline)

        upload_source = Path(tmpdir) / "uploaded.mp4"
        _make_1080p_video(upload_source, "purple")

        with patch.object(settings, "documentary_projects_dir", Path(tmpdir)):
            updated_entry = asyncio.run(
                rerender_single_clip("mode_a_two_clip_proj", 1, upload_path=str(upload_source))
            )

        assert updated_entry.placeholder is False
        assert Path(updated_entry.asset_metadata.path).exists()


def test_generate_and_edit_surfaces_segment_render_error_without_crashing():
    """A genuinely broken table (non-contiguous rows here) must still be
    reported via assembly_error, not raise out of the background job —
    same graceful-degradation contract run() already had via assemble_video."""
    bad_clips = [_clip(clip_number=1, start="00:00:00", end="00:00:05"),
                 _clip(clip_number=2, start="00:00:06", end="00:00:10")]  # gap: 05 != 06
    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.generate_table", return_value=bad_clips), \
         patch.object(settings, "documentary_projects_dir", Path(tmpdir)), \
         patch_providers(), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]):
        result = asyncio.run(generate_and_edit("some script", project_name="mode_a_bad_proj"))

    assert result.assembly_error is not None
    assert "gap/overlap" in result.assembly_error
    assert result.final_video_path is None


def test_max_downloads_limit_stops_later_clips_without_provider_calls():
    """The pre-render availability scan (check_footage_availability) itself
    searches every clip regardless of max_downloads — that's a separate,
    deliberate concern (see its own tests) — so it's stubbed out here to
    keep this test scoped to what it actually checks: the REAL resolve loop
    short-circuits clip 2's provider calls once the download limit is hit."""
    clips = [_clip(clip_number=1, start="00:00:00", end="00:00:05"),
             _clip(clip_number=2, start="00:00:05", end="00:00:10")]
    hit = _hit(source="pexels", source_id="v1")
    no_scan = AvailabilityReport(total_clips=2, thin_count=0, thin_clip_numbers=[], clips=[])

    def fake_assemble(project_dir, timeline):  # pass timeline through untouched
        return project_dir / "final_video.mp4", timeline

    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.generate_table", return_value=clips), \
         patch("app.documentary_pipeline.check_footage_availability", new=AsyncMock(return_value=no_scan)), \
         patch.object(settings, "documentary_projects_dir", Path(tmpdir)), \
         patch("app.documentary_pipeline.download_video_trimmed", new=AsyncMock(return_value="out.mp4")), \
         patch("app.documentary_pipeline.probe", return_value=(5.0, "1920x1080")), \
         patch("app.documentary_pipeline.assemble_video", side_effect=fake_assemble), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]), \
         patch("app.documentary_pipeline.decide_style", return_value=StyleDecision()), \
         patch("app.documentary_pipeline.render_final_video", return_value=Path(tmpdir) / "final_video_remotion.mp4"), \
         patch_providers(pexels_video=[hit]) as mocks:
        result = asyncio.run(run("some script", project_name="limited_proj", max_downloads=1))

    assert result.timeline[0].placeholder is False
    assert result.timeline[1].placeholder is True
    assert "download limit" in result.timeline[1].note
    assert mocks["pexels_video"].await_count == 1  # only called for clip 1, clip 2 short-circuits


def test_run_threads_whisper_words_into_table_and_render_when_audio_provided():
    clip = _clip()
    whisper_words = [WordTiming(text="hello", start=0.0, end=0.4)]

    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.get_narration_timing", return_value=whisper_words) as mock_get_timing, \
         patch("app.documentary_pipeline.check_alignment", return_value=None) as mock_check_alignment, \
         patch("app.documentary_pipeline.generate_table", return_value=[clip]) as mock_generate_table, \
         patch.object(settings, "documentary_projects_dir", Path(tmpdir)), \
         patch_providers(), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]), \
         patch("app.documentary_pipeline.decide_style", return_value=StyleDecision()), \
         patch("app.documentary_pipeline.render_final_video",
               return_value=Path(tmpdir) / "final_video_remotion.mp4") as mock_render, \
         patch("app.documentary_pipeline.mux_narration_audio",
               return_value=Path(tmpdir) / "final_video_with_narration.mp4") as mock_mux:
        result = asyncio.run(run("some script", project_name="audio_proj", audio_path="fake_audio.wav"))

    mock_get_timing.assert_called_once_with("fake_audio.wav")
    mock_check_alignment.assert_called_once_with(whisper_words, "some script")
    mock_generate_table.assert_called_once_with("some script", whisper_words, DEFAULT_NICHE)
    mock_render.assert_called_once()
    assert mock_render.call_args.args[3] is whisper_words  # 4th positional arg threaded through
    assert result.transcript_alignment_warning is None
    mock_mux.assert_called_once()
    assert mock_mux.call_args.args[1] == "fake_audio.wav"
    assert result.final_video_with_narration_path == str(Path(tmpdir) / "final_video_with_narration.mp4")


def test_run_surfaces_alignment_warning_in_result():
    clip = _clip()
    whisper_words = [WordTiming(text="hello", start=0.0, end=0.4)]

    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.get_narration_timing", return_value=whisper_words), \
         patch("app.documentary_pipeline.check_alignment", return_value="mismatch warning text"), \
         patch("app.documentary_pipeline.generate_table", return_value=[clip]), \
         patch.object(settings, "documentary_projects_dir", Path(tmpdir)), \
         patch_providers(), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]), \
         patch("app.documentary_pipeline.decide_style", return_value=StyleDecision()), \
         patch("app.documentary_pipeline.render_final_video", return_value=Path(tmpdir) / "final_video_remotion.mp4"), \
         patch("app.documentary_pipeline.mux_narration_audio",
               return_value=Path(tmpdir) / "final_video_with_narration.mp4"):
        result = asyncio.run(run("some script", project_name="audio_proj2", audio_path="fake_audio.wav"))

    assert result.transcript_alignment_warning == "mismatch warning text"


def test_run_skips_audio_mux_entirely_when_no_audio_path_given():
    clip = _clip()
    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.generate_table", return_value=[clip]), \
         patch.object(settings, "documentary_projects_dir", Path(tmpdir)), \
         patch_providers(), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]), \
         patch("app.documentary_pipeline.decide_style", return_value=StyleDecision()), \
         patch("app.documentary_pipeline.render_final_video", return_value=Path(tmpdir) / "final_video_remotion.mp4"), \
         patch("app.documentary_pipeline.mux_narration_audio") as mock_mux:
        result = asyncio.run(run("some script", project_name="no_audio_proj"))

    mock_mux.assert_not_called()
    assert result.final_video_with_narration_path is None
    assert result.audio_mux_error is None


def test_run_surfaces_audio_mux_error_without_crashing():
    clip = _clip()
    whisper_words = [WordTiming(text="hello", start=0.0, end=0.4)]

    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.get_narration_timing", return_value=whisper_words), \
         patch("app.documentary_pipeline.check_alignment", return_value=None), \
         patch("app.documentary_pipeline.generate_table", return_value=[clip]), \
         patch.object(settings, "documentary_projects_dir", Path(tmpdir)), \
         patch_providers(), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]), \
         patch("app.documentary_pipeline.decide_style", return_value=StyleDecision()), \
         patch("app.documentary_pipeline.render_final_video", return_value=Path(tmpdir) / "final_video_remotion.mp4"), \
         patch("app.documentary_pipeline.mux_narration_audio",
               side_effect=AudioMuxError("ffmpeg failed to mux narration audio")):
        result = asyncio.run(run("some script", project_name="mux_fail_proj", audio_path="fake_audio.wav"))

    assert result.final_video_with_narration_path is None
    assert result.audio_mux_error == "ffmpeg failed to mux narration audio"
    # the rest of the result must still be usable — a mux failure is non-fatal
    assert result.subtitled_video_path is not None


def test_final_video_filename_uses_default_for_auto_generated_project_name():
    """resolve_project_name()'s own timestamp fallback (project_<digits>) —
    i.e. the user left the project name blank — must keep today's filename."""
    from app.documentary_pipeline import final_video_filename

    assert final_video_filename(Path("/tmp/project_1699999999")) == "final_video_with_narration.mp4"


def test_final_video_filename_sanitizes_user_provided_project_name():
    from app.documentary_pipeline import final_video_filename

    assert final_video_filename(Path("/tmp/Alex Eala Toronto Recap")) == "Alex_Eala_Toronto_Recap.mp4"


def test_final_video_filename_strips_filesystem_invalid_characters():
    """"/" is deliberately not tested here — Path treats it as a directory
    separator (so it'd never reach _sanitize_filename as part of .name on a
    real project_dir anyway), not a character within a single path component."""
    from app.documentary_pipeline import final_video_filename

    assert final_video_filename(Path('/tmp/Alex: Eala <Toronto>?')) == "Alex_Eala_Toronto.mp4"


def test_final_video_filename_falls_back_to_default_when_nothing_survives_sanitizing():
    from app.documentary_pipeline import final_video_filename

    assert final_video_filename(Path("/tmp/???")) == "final_video_with_narration.mp4"


def test_run_names_final_video_after_sanitized_project_name():
    """End-to-end wiring: run() must pass a filename built from the sanitized
    project name as mux_narration_audio's output path when a real name was
    given (not the resolve_project_name() timestamp fallback)."""
    clip = _clip()
    whisper_words = [WordTiming(text="hello", start=0.0, end=0.4)]

    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.get_narration_timing", return_value=whisper_words), \
         patch("app.documentary_pipeline.check_alignment", return_value=None), \
         patch("app.documentary_pipeline.generate_table", return_value=[clip]), \
         patch.object(settings, "documentary_projects_dir", Path(tmpdir)), \
         patch_providers(), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]), \
         patch("app.documentary_pipeline.decide_style", return_value=StyleDecision()), \
         patch("app.documentary_pipeline.render_final_video", return_value=Path(tmpdir) / "final_video_remotion.mp4"), \
         patch("app.documentary_pipeline.mux_narration_audio",
               return_value=Path(tmpdir) / "Alex_Eala_Toronto_Recap.mp4") as mock_mux:
        asyncio.run(run("some script", project_name="Alex Eala Toronto Recap", audio_path="fake_audio.wav"))

    output_path = mock_mux.call_args.args[2]
    assert output_path.name == "Alex_Eala_Toronto_Recap.mp4"


def test_run_keeps_default_final_video_name_when_no_project_name_given():
    clip = _clip()
    whisper_words = [WordTiming(text="hello", start=0.0, end=0.4)]

    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.get_narration_timing", return_value=whisper_words), \
         patch("app.documentary_pipeline.check_alignment", return_value=None), \
         patch("app.documentary_pipeline.generate_table", return_value=[clip]), \
         patch.object(settings, "documentary_projects_dir", Path(tmpdir)), \
         patch_providers(), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]), \
         patch("app.documentary_pipeline.decide_style", return_value=StyleDecision()), \
         patch("app.documentary_pipeline.render_final_video", return_value=Path(tmpdir) / "final_video_remotion.mp4"), \
         patch("app.documentary_pipeline.mux_narration_audio",
               return_value=Path(tmpdir) / "final_video_with_narration.mp4") as mock_mux:
        asyncio.run(run("some script", audio_path="fake_audio.wav"))

    output_path = mock_mux.call_args.args[2]
    assert output_path.name == "final_video_with_narration.mp4"


def test_run_propagates_transcription_error_without_falling_back_to_estimate():
    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.get_narration_timing",
               side_effect=TranscriptionError("audio file not found: bad.wav")), \
         patch.object(settings, "documentary_projects_dir", Path(tmpdir)):
        try:
            asyncio.run(run("some script", project_name="bad_audio_proj", audio_path="bad.wav"))
            raise AssertionError("expected TranscriptionError to propagate")
        except TranscriptionError as exc:
            assert "not found" in str(exc)


def test_run_reads_script_from_script_path():
    clip = _clip()
    with tempfile.TemporaryDirectory() as tmpdir:
        script_file = Path(tmpdir) / "script.txt"
        script_file.write_text("A script read from a file on disk.", encoding="utf-8")

        with patch("app.documentary_pipeline.generate_table", return_value=[clip]) as mock_generate_table, \
             patch.object(settings, "documentary_projects_dir", Path(tmpdir)), \
             patch_providers(), \
             patch("app.scoring.embed", return_value=[1.0, 0.0]), \
             patch("app.documentary_pipeline.decide_style", return_value=StyleDecision()), \
             patch("app.documentary_pipeline.render_final_video", return_value=Path(tmpdir) / "final_video_remotion.mp4"):
            asyncio.run(run("", project_name="script_path_proj", script_path=str(script_file)))

    mock_generate_table.assert_called_once_with("A script read from a file on disk.", None, DEFAULT_NICHE)


def test_resolve_clip_populates_alternates_with_top_ranked_candidates_capped_at_five():
    clip = _clip()
    hits = [_hit(source="pexels", source_id=str(i), text=f"candidate {i}") for i in range(7)]
    video_hits = [ScoredAsset(hit=h, score=90.0 - i * 5) for i, h in enumerate(hits)]

    async def fake_download_asset(asset, dest_dir, prefix, index, target_duration):
        return AssetInfo(path="out.mp4", source=asset.hit.source, source_id=asset.hit.source_id,
                          media_type="video", width=1920, height=1080, duration=5.0, score=asset.score)

    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.search_clip", new=AsyncMock(return_value=(video_hits, []))), \
         patch("app.documentary_pipeline._download_asset", side_effect=fake_download_asset), \
         patch("app.documentary_pipeline.probe", return_value=(5.0, "1920x1080")), \
         patch("app.documentary_pipeline.passes_visual_verification", return_value=(True, 0.5)):
        entry, _ = asyncio.run(_resolve_clip(
            clip, Path(tmpdir), used_assets={}, allow_duplicates=False,
            downloads_so_far=0, limit=None,
        ))

    assert len(entry.alternates) == 5  # capped, even though 7 candidates were ranked
    assert [a.source_id for a in entry.alternates] == ["0", "1", "2", "3", "4"]  # best-score-first
    assert entry.alternates[0].score == 90.0
    assert entry.alternates[0].download_url == "http://x/0.mp4"


def test_resolve_clip_fetches_distinct_fill_asset_when_primary_too_short():
    """The actual fix: a primary candidate whose real (post-download)
    duration is far short of the row's target (ratio outside the natural
    speed-match range) must not be left to loop at render time —
    _resolve_clip fetches a second, distinct candidate right here (the full
    ranked list is still in scope) and attaches it as
    fill_asset_path/fill_asset_metadata, registering both in used_assets."""
    clip = _clip()  # target duration 5.0s (see _clip's default start/end)
    hit_a = _hit(source="pexels", source_id="a", text="candidate a")
    hit_b = _hit(source="pexels", source_id="b", text="candidate b")
    video_hits = [ScoredAsset(hit=hit_a, score=90.0), ScoredAsset(hit=hit_b, score=80.0)]

    async def fake_download_asset(asset, dest_dir, prefix, index, target_duration):
        return AssetInfo(path=f"{asset.hit.source_id}.mp4", source=asset.hit.source, source_id=asset.hit.source_id,
                          media_type="video", width=1920, height=1080, duration=target_duration, score=asset.score)

    def fake_probe(path):
        return (1.0, "1920x1080") if path == "a.mp4" else (5.0, "1920x1080")

    used_assets: dict[tuple[str, str], int] = {}
    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.search_clip", new=AsyncMock(return_value=(video_hits, []))), \
         patch("app.documentary_pipeline._download_asset", side_effect=fake_download_asset), \
         patch("app.documentary_pipeline.probe", side_effect=fake_probe), \
         patch("app.documentary_pipeline.passes_visual_verification", return_value=(True, 0.5)):
        entry, downloads_so_far = asyncio.run(_resolve_clip(
            clip, Path(tmpdir), used_assets=used_assets, allow_duplicates=False,
            downloads_so_far=0, limit=None,
        ))

    assert entry.asset_metadata.source_id == "a"
    assert entry.fill_asset_path == "b.mp4"
    assert entry.fill_asset_metadata.source_id == "b"
    assert downloads_so_far == 2  # primary + fill
    assert used_assets == {("pexels", "a"): 1, ("pexels", "b"): 1}


def test_resolve_clip_leaves_fill_unset_when_no_second_candidate_available():
    """No second candidate at all — must not loop anyway (see
    documentary_assembly._render_fill_segment's forced speed-match fallback);
    _resolve_clip's job here is only to leave fill_asset_path unset."""
    clip = _clip()
    hit_a = _hit(source="pexels", source_id="a")
    video_hits = [ScoredAsset(hit=hit_a, score=90.0)]

    async def fake_download_asset(asset, dest_dir, prefix, index, target_duration):
        return AssetInfo(path="a.mp4", source=asset.hit.source, source_id=asset.hit.source_id,
                          media_type="video", width=1920, height=1080, duration=target_duration, score=asset.score)

    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.search_clip", new=AsyncMock(return_value=(video_hits, []))), \
         patch("app.documentary_pipeline._download_asset", side_effect=fake_download_asset), \
         patch("app.documentary_pipeline.probe", return_value=(1.0, "1920x1080")), \
         patch("app.documentary_pipeline.passes_visual_verification", return_value=(True, 0.5)):
        entry, downloads_so_far = asyncio.run(_resolve_clip(
            clip, Path(tmpdir), used_assets={}, allow_duplicates=False,
            downloads_so_far=0, limit=None,
        ))

    assert entry.fill_asset_path is None
    assert entry.fill_asset_metadata is None
    assert downloads_so_far == 1  # only the primary — no fill candidate to try


def test_resolve_clip_skips_fill_fetch_when_within_natural_speed_match_range():
    """4.0s primary in a 5.0s row (ratio 0.8) is well within
    documentary_assembly's natural speed-match range — assembly can retime it
    alone, so _resolve_clip must not even attempt a fill-candidate search."""
    clip = _clip()
    hit_a = _hit(source="pexels", source_id="a")
    hit_b = _hit(source="pexels", source_id="b")
    video_hits = [ScoredAsset(hit=hit_a, score=90.0), ScoredAsset(hit=hit_b, score=80.0)]

    async def fake_download_asset(asset, dest_dir, prefix, index, target_duration):
        return AssetInfo(path=f"{asset.hit.source_id}.mp4", source=asset.hit.source, source_id=asset.hit.source_id,
                          media_type="video", width=1920, height=1080, duration=target_duration, score=asset.score)

    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.search_clip", new=AsyncMock(return_value=(video_hits, []))), \
         patch("app.documentary_pipeline._download_asset", side_effect=fake_download_asset) as mock_download, \
         patch("app.documentary_pipeline.probe", return_value=(4.0, "1920x1080")), \
         patch("app.documentary_pipeline.passes_visual_verification", return_value=(True, 0.5)):
        entry, _ = asyncio.run(_resolve_clip(
            clip, Path(tmpdir), used_assets={}, allow_duplicates=False,
            downloads_so_far=0, limit=None,
        ))

    assert entry.fill_asset_path is None
    assert mock_download.call_count == 1  # only the primary downloaded


def test_resolve_clip_never_fetches_fill_for_image_candidates():
    """Images always fill their row for free (-loop 1, see
    documentary_assembly._render_image_segment) — no duration mismatch is
    possible, so the fill-candidate search must never trigger for one."""
    clip = _clip()
    hit_a = _hit(source="pexels", source_id="a", media_type="image")
    image_hits = [ScoredAsset(hit=hit_a, score=90.0)]

    async def fake_download_asset(asset, dest_dir, prefix, index, target_duration):
        return AssetInfo(path="a.jpg", source=asset.hit.source, source_id=asset.hit.source_id,
                          media_type="image", width=1920, height=1080, duration=0.0, score=asset.score)

    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.search_clip", new=AsyncMock(return_value=([], image_hits))), \
         patch("app.documentary_pipeline._download_asset", side_effect=fake_download_asset) as mock_download, \
         patch("app.documentary_pipeline.probe", return_value=(0.0, "1920x1080")), \
         patch("app.documentary_pipeline.passes_visual_verification", return_value=(True, 0.5)):
        entry, _ = asyncio.run(_resolve_clip(
            clip, Path(tmpdir), used_assets={}, allow_duplicates=False,
            downloads_so_far=0, limit=None,
        ))

    assert entry.fill_asset_path is None
    assert mock_download.call_count == 1


def test_rerender_single_clip_rejects_ambiguous_or_missing_mode():
    with pytest.raises(ValueError, match="exactly one of"):
        asyncio.run(rerender_single_clip("proj", 1))
    with pytest.raises(ValueError, match="exactly one of"):
        asyncio.run(rerender_single_clip("proj", 1, alternate_index=0, ai_regenerate=True))


def test_rerender_single_clip_raises_when_project_has_no_timeline():
    with tempfile.TemporaryDirectory() as tmpdir, \
         patch.object(settings, "documentary_projects_dir", Path(tmpdir)):
        with pytest.raises(ValueError, match="no timeline.json"):
            asyncio.run(rerender_single_clip("missing_proj", 1, ai_regenerate=True))


def _make_1080p_video(path: Path, color: str, duration: float = 3.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s=1920x1080:r=25:d={duration}",
         "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, check=True,
    )


def test_rerender_single_clip_swap_to_alternate_updates_asset_and_reconcats():
    """End-to-end (real ffmpeg, no mocked assembly): swapping clip 1 to its
    recorded alternate must update its own asset/rendered segment, re-concat
    final_video.mp4, persist the change to timeline.json, and leave clip 2's
    already-rendered segment completely untouched."""
    with tempfile.TemporaryDirectory() as tmpdir:
        projects_dir = Path(tmpdir) / "projects"
        project_dir = projects_dir / "swap_proj"
        clips_root = project_dir / "clips"
        source1 = clips_root / "clip_001" / "orig1.mp4"
        source2 = clips_root / "clip_002" / "orig2.mp4"
        _make_1080p_video(source1, "red")
        _make_1080p_video(source2, "blue")

        asset1 = AssetInfo(path=str(source1), source="pexels", source_id="1", media_type="video",
                            width=1920, height=1080, duration=3.0, score=80.0)
        asset2 = AssetInfo(path=str(source2), source="pexels", source_id="2", media_type="video",
                            width=1920, height=1080, duration=3.0, score=75.0)
        alt = AlternateCandidate(source="pexels", source_id="99", download_url="http://x/99.mp4",
                                  media_type="video", width=1920, height=1080, duration=3.0,
                                  text="alt clip", score=70.0)
        entry1 = TimelineEntry(
            clip_number=1, start="00:00:00", end="00:00:03", duration=3.0, script="beat one",
            asset_path=str(source1), asset_metadata=asset1, alternates=[alt],
            recommended_effect="e", transition="t",
        )
        entry2 = TimelineEntry(
            clip_number=2, start="00:00:03", end="00:00:06", duration=3.0, script="beat two",
            asset_path=str(source2), asset_metadata=asset2, recommended_effect="e", transition="t",
        )

        from app.documentary_assembly import assemble_video
        _final_path, built = assemble_video(project_dir, [entry1, entry2])
        (project_dir / "timeline.json").write_text(
            json.dumps([e.model_dump() for e in built], indent=2), encoding="utf-8",
        )
        original_clip2_rendered = built[1].rendered_clip_path

        new_source = clips_root / "clip_001" / "clip001_video0.mp4"
        _make_1080p_video(new_source, "green")
        fake_downloaded = AssetInfo(path=str(new_source), source="pexels", source_id="99",
                                     media_type="video", width=1920, height=1080, duration=3.0, score=70.0)

        with patch.object(settings, "documentary_projects_dir", projects_dir), \
             patch("app.documentary_pipeline._download_asset", new=AsyncMock(return_value=fake_downloaded)):
            updated_entry = asyncio.run(rerender_single_clip("swap_proj", 1, alternate_index=0))

        assert updated_entry.asset_metadata.source_id == "99"
        assert updated_entry.asset_metadata.source == "pexels"
        assert "swapped to alternate" in updated_entry.note
        assert updated_entry.placeholder is False

        reloaded = json.loads((project_dir / "timeline.json").read_text())
        assert reloaded[0]["asset_metadata"]["source_id"] == "99"
        assert reloaded[1]["rendered_clip_path"] == original_clip2_rendered  # untouched by the swap


def test_rerender_single_clip_alternate_index_out_of_range_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        projects_dir = Path(tmpdir) / "projects"
        project_dir = projects_dir / "range_proj"
        source1 = project_dir / "clips" / "clip_001" / "orig1.mp4"
        _make_1080p_video(source1, "red")
        asset1 = AssetInfo(path=str(source1), source="pexels", source_id="1", media_type="video",
                            width=1920, height=1080, duration=3.0, score=80.0)
        entry1 = TimelineEntry(
            clip_number=1, start="00:00:00", end="00:00:03", duration=3.0, script="beat one",
            asset_path=str(source1), asset_metadata=asset1, alternates=[],
            recommended_effect="e", transition="t",
        )
        from app.documentary_assembly import assemble_video
        _final_path, built = assemble_video(project_dir, [entry1])
        (project_dir / "timeline.json").write_text(
            json.dumps([e.model_dump() for e in built], indent=2), encoding="utf-8",
        )

        with patch.object(settings, "documentary_projects_dir", projects_dir):
            with pytest.raises(ValueError, match="out of range"):
                asyncio.run(rerender_single_clip("range_proj", 1, alternate_index=0))


def test_rerender_single_clip_rejects_alternate_already_used_by_another_clip():
    """The automatic resolve loop dedups via used_assets, but a single-clip
    editor swap (rerender_single_clip) has no persisted used_assets dict to
    check against — it must instead check the CURRENT timeline.json fresh.
    Without this guard, swapping clip 1 to an alternate that's already clip
    2's accepted asset would put the same real footage on screen twice with
    no dedup mechanism ever catching it (see _reject_if_asset_used_elsewhere)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        projects_dir = Path(tmpdir) / "projects"
        project_dir = projects_dir / "dup_swap_proj"
        clips_root = project_dir / "clips"
        source1 = clips_root / "clip_001" / "orig1.mp4"
        source2 = clips_root / "clip_002" / "orig2.mp4"
        _make_1080p_video(source1, "red")
        _make_1080p_video(source2, "blue")

        asset1 = AssetInfo(path=str(source1), source="pexels", source_id="1", media_type="video",
                            width=1920, height=1080, duration=3.0, score=80.0)
        # clip 2 already uses source_id "99" — this is exactly what clip 1's
        # alternate below also points at.
        asset2 = AssetInfo(path=str(source2), source="pexels", source_id="99", media_type="video",
                            width=1920, height=1080, duration=3.0, score=75.0)
        alt = AlternateCandidate(source="pexels", source_id="99", download_url="http://x/99.mp4",
                                  media_type="video", width=1920, height=1080, duration=3.0,
                                  text="alt clip", score=70.0)
        entry1 = TimelineEntry(
            clip_number=1, start="00:00:00", end="00:00:03", duration=3.0, script="beat one",
            asset_path=str(source1), asset_metadata=asset1, alternates=[alt],
            recommended_effect="e", transition="t",
        )
        entry2 = TimelineEntry(
            clip_number=2, start="00:00:03", end="00:00:06", duration=3.0, script="beat two",
            asset_path=str(source2), asset_metadata=asset2, recommended_effect="e", transition="t",
        )

        from app.documentary_assembly import assemble_video
        _final_path, built = assemble_video(project_dir, [entry1, entry2])
        (project_dir / "timeline.json").write_text(
            json.dumps([e.model_dump() for e in built], indent=2), encoding="utf-8",
        )

        with patch.object(settings, "documentary_projects_dir", projects_dir), \
             patch("app.documentary_pipeline._download_asset", new=AsyncMock()) as mock_download:
            with pytest.raises(ValueError, match="already used by clip 2"):
                asyncio.run(rerender_single_clip("dup_swap_proj", 1, alternate_index=0))
            mock_download.assert_not_called()

        # timeline.json must be left untouched — the rejected swap never got
        # far enough to write anything back.
        reloaded = json.loads((project_dir / "timeline.json").read_text())
        assert reloaded[0]["asset_metadata"]["source_id"] == "1"


def test_rerender_single_clip_alternate_with_missing_local_library_file_raises_clear_error():
    """Bug report: a local_library index.json entry whose file was deleted
    (e.g. by the near-duplicate cleanup pass) surfaces as a raw ffmpeg-stderr
    500 when the editor clicks that alternate. Must instead raise a ValueError
    (main.py's PATCH route already maps ValueError -> 400 with a clean
    detail message) naming the actual problem, not an opaque ffmpeg failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        projects_dir = Path(tmpdir) / "projects"
        project_dir = projects_dir / "stale_alt_proj"
        source1 = project_dir / "clips" / "clip_001" / "orig1.mp4"
        _make_1080p_video(source1, "red")
        asset1 = AssetInfo(path=str(source1), source="pexels", source_id="1", media_type="video",
                            width=1920, height=1080, duration=3.0, score=80.0)
        missing_path = str(project_dir / "clips" / "local" / "deleted_clip.mp4")
        stale_alt = AlternateCandidate(
            source="local_library", source_id="deleted_clip.mp4", download_url=missing_path,
            media_type="video", width=1920, height=1080, duration=3.0, text="a prison yard", score=70.0,
        )
        entry1 = TimelineEntry(
            clip_number=1, start="00:00:00", end="00:00:03", duration=3.0, script="beat one",
            asset_path=str(source1), asset_metadata=asset1, alternates=[stale_alt],
            recommended_effect="e", transition="t",
        )
        from app.documentary_assembly import assemble_video
        _final_path, built = assemble_video(project_dir, [entry1])
        (project_dir / "timeline.json").write_text(
            json.dumps([e.model_dump() for e in built], indent=2), encoding="utf-8",
        )

        with patch.object(settings, "documentary_projects_dir", projects_dir):
            with pytest.raises(ValueError, match="no longer available"):
                asyncio.run(rerender_single_clip("stale_alt_proj", 1, alternate_index=0))


def test_rerender_single_clip_swap_to_alternate_resolves_a_placeholder_clip():
    """Issue 2: a placeholder clip (no asset_path/asset_metadata, but real
    alternates recorded from search_clip) must be swappable exactly like any
    already-resolved clip — same _materialize_alternate_asset/_download_asset
    path, no placeholder-specific gate blocking it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        projects_dir = Path(tmpdir) / "projects"
        project_dir = projects_dir / "placeholder_swap_proj"
        clips_root = project_dir / "clips"

        alt = AlternateCandidate(source="pexels", source_id="99", download_url="http://x/99.mp4",
                                  media_type="video", width=1920, height=1080, duration=3.0,
                                  text="alt clip", score=70.0)
        entry1 = TimelineEntry(
            clip_number=1, start="00:00:00", end="00:00:03", duration=3.0, script="beat one",
            asset_path=None, asset_metadata=None, alternates=[alt], placeholder=True,
            note="no acceptable asset found across all providers",
            recommended_effect="e", transition="t",
        )

        from app.documentary_assembly import assemble_video
        _final_path, built = assemble_video(project_dir, [entry1])
        assert built[0].placeholder is True
        assert built[0].rendered_clip_path and Path(built[0].rendered_clip_path).exists()  # black segment rendered
        (project_dir / "timeline.json").write_text(
            json.dumps([e.model_dump() for e in built], indent=2), encoding="utf-8",
        )

        new_source = clips_root / "clip_001" / "clip001_video0.mp4"
        _make_1080p_video(new_source, "green")
        fake_downloaded = AssetInfo(path=str(new_source), source="pexels", source_id="99",
                                     media_type="video", width=1920, height=1080, duration=3.0, score=70.0)

        with patch.object(settings, "documentary_projects_dir", projects_dir), \
             patch("app.documentary_pipeline._download_asset", new=AsyncMock(return_value=fake_downloaded)):
            updated_entry = asyncio.run(rerender_single_clip("placeholder_swap_proj", 1, alternate_index=0))

        assert updated_entry.placeholder is False
        assert updated_entry.asset_metadata.source_id == "99"
        assert updated_entry.rendered_clip_path and Path(updated_entry.rendered_clip_path).exists()

        reloaded = json.loads((project_dir / "timeline.json").read_text())
        assert reloaded[0]["placeholder"] is False
        assert reloaded[0]["rendered_clip_path"]


def test_rerender_single_clip_ai_regenerate_uses_table_entry_from_project_meta():
    with tempfile.TemporaryDirectory() as tmpdir:
        projects_dir = Path(tmpdir) / "projects"
        project_dir = projects_dir / "ai_proj"
        source1 = project_dir / "clips" / "clip_001" / "orig1.mp4"
        _make_1080p_video(source1, "red")
        asset1 = AssetInfo(path=str(source1), source="pexels", source_id="1", media_type="video",
                            width=1920, height=1080, duration=3.0, score=80.0)
        entry1 = TimelineEntry(
            clip_number=1, start="00:00:00", end="00:00:03", duration=3.0, script="beat one",
            asset_path=str(source1), asset_metadata=asset1, recommended_effect="e", transition="t",
        )
        from app.documentary_assembly import assemble_video
        _final_path, built = assemble_video(project_dir, [entry1])
        (project_dir / "timeline.json").write_text(
            json.dumps([e.model_dump() for e in built], indent=2), encoding="utf-8",
        )
        table_clip = _clip(clip_number=1, canva_keyword="test keyword")
        meta = ProjectMeta(script="a script", content_niche=DEFAULT_NICHE, table=[table_clip])
        (project_dir / "project_meta.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")

        generated_image = project_dir / "clips" / "clip_001" / "clip001_generated.jpg"
        generated_image.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=yellow:s=1920x1080:d=1",
             "-frames:v", "1", str(generated_image)],
            capture_output=True, check=True,
        )
        fake_asset = AssetInfo(path=str(generated_image), source="ai_generated", source_id=str(generated_image),
                                media_type="image", width=1920, height=1080, duration=0, score=None,
                                selected_text="(AI-generated)")
        fake_entry = TimelineEntry(
            clip_number=1, start="00:00:00", end="00:00:03", duration=3.0, script="beat one",
            asset_path=str(generated_image), asset_metadata=fake_asset,
            recommended_effect="e", transition="t", generation_attempted=True,
        )

        with patch.object(settings, "documentary_projects_dir", projects_dir), \
             patch("app.documentary_pipeline._try_ai_generation_fallback",
                   new=AsyncMock(return_value=(fake_entry, None))):
            updated_entry = asyncio.run(rerender_single_clip("ai_proj", 1, ai_regenerate=True))

        assert updated_entry.asset_metadata.source == "ai_generated"
        assert updated_entry.generation_attempted is True
        assert "regenerated via AI" in updated_entry.note


def test_rerender_single_clip_upload_uses_uploaded_file_directly():
    with tempfile.TemporaryDirectory() as tmpdir:
        projects_dir = Path(tmpdir) / "projects"
        project_dir = projects_dir / "upload_proj"
        source1 = project_dir / "clips" / "clip_001" / "orig1.mp4"
        _make_1080p_video(source1, "red")
        asset1 = AssetInfo(path=str(source1), source="pexels", source_id="1", media_type="video",
                            width=1920, height=1080, duration=3.0, score=80.0)
        entry1 = TimelineEntry(
            clip_number=1, start="00:00:00", end="00:00:03", duration=3.0, script="beat one",
            asset_path=str(source1), asset_metadata=asset1, recommended_effect="e", transition="t",
        )
        from app.documentary_assembly import assemble_video
        _final_path, built = assemble_video(project_dir, [entry1])
        (project_dir / "timeline.json").write_text(
            json.dumps([e.model_dump() for e in built], indent=2), encoding="utf-8",
        )

        upload_source = Path(tmpdir) / "uploaded.mp4"
        _make_1080p_video(upload_source, "purple")

        with patch.object(settings, "documentary_projects_dir", projects_dir):
            updated_entry = asyncio.run(
                rerender_single_clip("upload_proj", 1, upload_path=str(upload_source))
            )

        assert updated_entry.asset_metadata.source == "upload"
        assert "manually uploaded" in updated_entry.note
        assert Path(updated_entry.asset_metadata.path).exists()


def test_generate_full_video_reuses_finalize_and_render_over_persisted_project_state():
    """The Tier-2 "Generate Full Video" path must load table/timeline/script/
    footage_availability from disk (project_meta.json/timeline.json) instead
    of re-running script->table generation, search, download, or CLIP —
    then drive them through the exact same finalize path run() uses."""
    with tempfile.TemporaryDirectory() as tmpdir:
        projects_dir = Path(tmpdir) / "projects"
        project_dir = projects_dir / "regen_proj"
        source1 = project_dir / "clips" / "clip_001" / "orig1.mp4"
        _make_1080p_video(source1, "red")
        asset1 = AssetInfo(path=str(source1), source="pexels", source_id="1", media_type="video",
                            width=1920, height=1080, duration=3.0, score=80.0)
        entry1 = TimelineEntry(
            clip_number=1, start="00:00:00", end="00:00:03", duration=3.0, script="beat one",
            asset_path=str(source1), asset_metadata=asset1, recommended_effect="e", transition="t",
        )
        (project_dir / "clips").mkdir(exist_ok=True)
        (project_dir / "timeline.json").write_text(
            json.dumps([entry1.model_dump()], indent=2), encoding="utf-8",
        )
        table_clip = _clip(clip_number=1)
        meta = ProjectMeta(
            script="a script", content_niche=DEFAULT_NICHE, table=[table_clip],
            footage_availability=AvailabilityReport(total_clips=1, thin_count=0, thin_clip_numbers=[], clips=[]),
        )
        (project_dir / "project_meta.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")

        with patch.object(settings, "documentary_projects_dir", projects_dir), \
             patch("app.documentary_pipeline.decide_style", return_value=StyleDecision()), \
             patch("app.documentary_pipeline.render_final_video",
                   return_value=project_dir / "final_video_remotion.mp4") as mock_render:
            result = asyncio.run(generate_full_video("regen_proj"))

        assert result.final_video_path is not None
        assert result.subtitled_video_path == str(project_dir / "final_video_remotion.mp4")
        mock_render.assert_called_once()
        assert result.footage_availability.total_clips == 1
        assert (project_dir / "project_meta.json").exists()  # re-persisted, not just read


if __name__ == "__main__":
    test_assign_timings_clamps_and_accumulates()
    test_assign_timings_normalizes_invalid_visual_type_instead_of_raising()
    test_assign_timings_uses_real_whisper_durations_when_provided()
    test_assign_timings_floors_to_min_duration_when_whisper_words_run_out()
    test_assign_timings_generates_coverage_rows_when_narration_runs_past_the_table()
    test_force_contiguity_corrects_a_gap_between_consecutive_clips()
    test_enforce_max_clip_duration_splits_an_8s_row_into_multiple_sub_5s_rows()
    test_split_long_clip_gives_each_sub_row_its_own_distinct_keyword()
    test_split_long_clip_does_not_leak_part_suffix_into_keyword_when_generation_is_degenerate()
    test_assign_timings_forces_contiguity_when_a_real_narration_pause_creates_a_gap()
    test_assign_timings_floors_degenerate_short_real_span_and_cascades_forward()
    test_assign_timings_content_match_avoids_drift_from_punctuation_token_mismatch()
    test_assign_timings_falls_back_to_word_count_when_content_match_confidence_is_too_low()
    test_score_asset_weighting()
    test_keyword_overlap_ratio_full_match()
    test_keyword_overlap_ratio_partial_match()
    test_keyword_overlap_ratio_no_match()
    test_keyword_overlap_ratio_ignores_filler_words_in_denominator()
    test_keyword_overlap_ratio_empty_or_filler_only_keyword()
    test_keyword_overlap_ratio_strips_metadata_punctuation()
    test_word_rarity_weight_common_words_score_low()
    test_word_rarity_weight_specific_words_score_high()
    test_keyword_overlap_ratio_real_motivating_examples()
    test_score_asset_full_vs_single_word_keyword_match()
    test_download_asset_persists_selected_text_from_hit()
    test_render_keyword_match_report_includes_selected_text_and_keyword_match()
    test_conversion_prompt_targets_five_second_beats()
    test_niche_for_clip()
    test_provider_order_for_niche_prioritizes_archives_for_historical()
    test_satisfied_only_stops_early_on_a_genuinely_high_quality_hit()
    test_high_quality_hit_stops_search_early()
    test_mediocre_hits_no_longer_stop_the_provider_loop_early()
    test_search_clip_uses_table_canva_keyword_as_primary_query()
    test_search_clip_uses_fallback_keyword_when_canva_keyword_returns_nothing()
    test_known_and_not_horizontal_rejects_only_confirmed_non_16_9()
    test_search_clip_keeps_non_16_9_hit_with_known_metadata_for_post_download_normalization()
    test_search_clip_rejects_known_too_low_res_hit_before_download()
    test_resolve_clip_normalizes_downloaded_asset_that_turns_out_vertical()
    test_rank_acceptable_assets_orders_by_score_no_video_preference()
    test_media_type_bias_favors_underrepresented_image_and_flips_near_tie()
    test_media_type_bias_cannot_override_a_large_quality_gap()
    test_media_type_bias_never_excludes_the_only_available_candidate()
    test_resolve_clip_rejects_and_falls_through_when_top_candidate_is_too_low_res()
    test_pick_best_asset_returns_single_highest_scorer()
    test_pick_best_asset_ignores_below_min_score()
    test_pick_best_asset_returns_none_when_no_candidates()
    test_cross_clip_dedup_excludes_used_asset_unless_allowed()
    test_search_clip_advances_to_next_tier_when_all_hits_from_a_tier_are_already_used()
    test_search_clip_widens_to_fallback_when_canva_keyword_scores_below_threshold()
    test_search_clip_merged_pool_ranks_best_candidate_regardless_of_tier()
    test_resolve_clip_excludes_already_used_candidate_across_rows_and_falls_through()
    test_safe_search_swallows_errors_and_timeouts()
    test_generate_semantic_keywords_regenerates_once_when_first_attempt_is_abstract()
    test_generate_semantic_keywords_skips_regeneration_when_first_attempt_is_already_concrete()
    test_generate_semantic_keywords_replaces_niche_violating_keywords()
    test_convert_script_to_visual_table_replaces_niche_violating_keywords()
    test_keywords_are_concrete_defaults_to_true_when_self_check_call_fails()
    test_resolve_clip_placeholder_note_reflects_exhaustive_search()
    test_resolve_clip_uses_ai_generation_fallback_when_search_is_exhausted_and_enabled()
    test_resolve_clip_registers_ai_generated_asset_in_used_assets_after_pre_download_trigger()
    test_resolve_clip_registers_ai_generated_asset_in_used_assets_after_last_resort_trigger()
    test_two_ai_generated_clips_get_distinct_used_assets_keys()
    test_resolve_clip_falls_through_to_placeholder_when_ai_generation_fails()
    test_resolve_clip_falls_through_to_placeholder_when_normalize_to_16_9_fails()
    test_resolve_clip_skips_ai_generation_fallback_when_disabled_by_default()
    test_ai_generation_fallback_triggers_when_best_real_candidate_scores_below_threshold()
    test_ai_generation_fallback_does_not_trigger_when_best_real_candidate_meets_threshold()
    test_ai_generation_fallback_degrades_gracefully_to_real_candidate_when_generation_fails()
    test_resolve_clip_tries_ai_generation_as_last_resort_when_candidate_passes_pre_download_but_fails_clip()
    test_resolve_clip_falls_through_to_placeholder_when_last_resort_ai_generation_also_fails()
    test_resolve_clip_last_resort_ai_generation_skipped_when_disabled()
    test_render_keyword_match_report_labels_ai_generated_entries()
    test_render_keyword_match_report_distinguishes_real_after_failed_generation_from_direct_accept()
    test_run_never_fails_a_clip_with_no_assets()
    test_max_downloads_limit_stops_later_clips_without_provider_calls()
    test_run_threads_whisper_words_into_table_and_render_when_audio_provided()
    test_run_surfaces_alignment_warning_in_result()
    test_run_skips_audio_mux_entirely_when_no_audio_path_given()
    test_run_surfaces_audio_mux_error_without_crashing()
    test_run_propagates_transcription_error_without_falling_back_to_estimate()
    test_run_reads_script_from_script_path()
    print("OK")
