"""Offline/mocked tests for app.documentary_pipeline, app.documentary_table, and
app.scoring. No network/API keys required. Run with: pytest tests/test_documentary_pipeline.py
"""
import asyncio
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from app.audio_mux import AudioMuxError
from app.clip_ingest import probe
from app.config import settings
from app.documentary_pipeline import (
    ScoredAsset,
    _generate_semantic_keywords,
    _keywords_are_concrete,
    _known_and_not_horizontal,
    _resolve_clip,
    _safe_search,
    _satisfied,
    niche_for_clip,
    pexels_video,
    pick_best_asset,
    provider_order_for_niche,
    rank_acceptable_assets,
    run,
    search_clip,
)
from app.documentary_table import (
    MAX_ON_SCREEN_CLIP_SECONDS,
    MIN_CLIP_SECONDS,
    _enforce_max_clip_duration,
    _force_contiguity,
    _split_long_clip,
    assign_timings,
    generate_table,
)
from app.models import AssetInfo, TimelineClip
from app.scoring import score_asset
from app.stock.base import StockHit
from app.style_decision import StyleDecision
from app.subtitles import WordTiming
from app.timecode import parse_timestamp
from app.transcription import TranscriptionError

_PROVIDER_NAMES = [
    "pexels_video", "pixabay_video", "pexels_images", "pixabay_images",
    "internet_archive", "wikimedia", "nasa",
]


@contextmanager
def patch_providers(**overrides):
    """Patches all 7 provider search() functions. Providers not named in overrides return []."""
    with ExitStack() as stack:
        mocks = {}
        for provider_name in _PROVIDER_NAMES:
            mock = AsyncMock(return_value=overrides.get(provider_name, []))
            stack.enter_context(patch(f"app.documentary_pipeline.{provider_name}.search", new=mock))
            mocks[provider_name] = mock
        stack.enter_context(patch("app.documentary_pipeline.generate_json", return_value={"keywords": ["generic visual footage", "broad documentary scene", "archive style imagery"]}))
        yield mocks


def _clip(clip_number=1, visual_type="Cinematic", start="00:00:00", end="00:00:05"):
    return TimelineClip(
        clip_number=clip_number, section="INTRO", script_beat="beat", canva_keyword="k",
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
    with patch("app.documentary_table.generate_json",
               return_value={"clips": [{"canva_keyword": "split keyword", "fallback_keyword": "split fallback"}]}):
        clips = assign_timings(raw)
    assert (clips[0].start, clips[0].end) == ("00:00:00", "00:00:03")  # 2 words clamped up to 3s floor
    # 30 words clamped down to the 7s word-count ceiling, then that 7s row exceeds the 5s
    # on-screen cap and gets split into two sub-rows (3.5s each -> 3s + 4s after rounding).
    assert len(clips) == 3
    assert (clips[1].start, clips[1].end) == ("00:00:03", "00:00:06")
    assert (clips[2].start, clips[2].end) == ("00:00:06", "00:00:10")


def test_assign_timings_normalizes_invalid_visual_type_instead_of_raising():
    raw = [{"clip_number": 1, "section": "INTRO", "script_beat": "a", "canva_keyword": "k",
            "fallback_keyword": "f", "visual_type": "Metaphor", "edit_note": "n", "word_count": 5}]
    clips = assign_timings(raw)  # would raise a pydantic ValidationError without normalization
    assert clips[0].visual_type == "B-roll"


def test_assign_timings_uses_real_whisper_durations_when_provided():
    raw = [
        {"clip_number": 1, "section": "INTRO", "script_beat": "hello there world how are you today", "canva_keyword": "k1",
         "fallback_keyword": "f1", "visual_type": "Archive", "edit_note": "n1", "word_count": 7},
        {"clip_number": 2, "section": "INTRO", "script_beat": "goodbye now my good friend", "canva_keyword": "k2",
         "fallback_keyword": "f2", "visual_type": "Archive", "edit_note": "n2", "word_count": 5},
    ]
    # Realistic pace (0.6s/word) across enough words that both real spans
    # comfortably exceed MIN_CLIP_SECONDS — nothing here should get floored.
    words = ("hello", "there", "world", "how", "are", "you", "today",
             "goodbye", "now", "my", "good", "friend")
    whisper_words = [WordTiming(text=w, start=i * 0.6, end=(i + 1) * 0.6) for i, w in enumerate(words)]

    clips = assign_timings(raw, whisper_words=whisper_words)
    # real timestamps used instead of word_count/2.5 -> real durations, not the 3s floor
    assert (clips[0].start, clips[0].end) == ("00:00:00", "00:00:04")  # 7 words * 0.6s = 4.2s, real
    assert (clips[1].start, clips[1].end) == ("00:00:04", "00:00:07")  # continues to 12*0.6=7.2s, real


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
    # Transcript only covers the first clip's 6 words (real pace, no floor needed there)
    # — the second clip's words never arrive at all.
    whisper_words = [WordTiming(text=w, start=i * 0.6, end=(i + 1) * 0.6)
                      for i, w in enumerate(("one", "two", "three", "four", "five", "six"))]
    clips = assign_timings(raw, whisper_words=whisper_words)
    assert clips[0].start == "00:00:00" and clips[0].end == "00:00:04"  # 6*0.6=3.6s, real, not floored
    # second clip has no whisper words left at all -> floors to MIN_CLIP_SECONDS from where clip 1 ended
    assert clips[1].start == clips[0].end == "00:00:04"
    assert clips[1].end == "00:00:07"  # 3.6 + MIN_CLIP_SECONDS(3) = 6.6, rounds to 7


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
    assert end_s - start_s >= MIN_CLIP_SECONDS  # floored, never a zero/near-zero duration row
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


def test_split_long_clip_forces_distinct_keywords_even_when_generation_is_degenerate():
    """Requirement 2's edge case: when the original row's text is too short
    to meaningfully subdivide (or the LLM call fails/returns nothing usable
    for every sub-row), sub-row keywords may be closely related, but must
    never come out byte-identical."""
    clip = TimelineClip(
        clip_number=1, section="S", script_beat="wow", canva_keyword="original keyword",
        fallback_keyword="original fallback", visual_type="Archive", edit_note="n",
        start="00:00:00", end="00:00:08",
    )
    with patch("app.documentary_table.generate_json", side_effect=RuntimeError("llm unavailable")):
        sub_clips = _split_long_clip(clip)

    assert len(sub_clips) == 2
    assert sub_clips[0].canva_keyword != sub_clips[1].canva_keyword


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
    # what was "clip 4" (00:14-00:17) is now clips[4], not clips[3].
    assert clips[4].start == "00:00:14" and clips[4].end == "00:00:17"  # original clip 4 unchanged, as reported
    # without _force_contiguity, clip 5's natural content-matched span would
    # start at 00:00:18 (the real 1s pause) -> exactly the reported gap.
    # The final pass must force it back to clip 4's end instead.
    assert clips[5].start == clips[4].end == "00:00:17"
    assert clips[5].end == "00:00:22"  # end is untouched by the correction — only start is forced


def test_generate_table_uses_structured_conversion_prompt():
    raw = {"clips": [{"clip_number": 1, "section": "INTRO", "script_beat": "hello world",
                       "canva_keyword": "k", "fallback_keyword": "f", "visual_type": "Cinematic",
                       "edit_note": "n", "word_count": 2}]}
    with patch("app.documentary_table.generate_json", return_value=raw) as mock_generate_json:
        generate_table("some script")
    prompt = mock_generate_json.call_args.args[0]
    assert "Convert the following narration script" in prompt
    assert "Return valid JSON only" in prompt


def test_score_asset_weighting():
    clip = _clip(visual_type="Archive")
    hit = _hit(source="internet_archive", text="archival footage")
    with patch("app.scoring.embed", return_value=[1.0, 0.0]):
        score = score_asset(hit, clip, [1.0, 0.0])
    assert score == 100.0  # perfect semantic + historical + quality + cinematic + motion


def test_niche_for_clip():
    assert niche_for_clip(_clip(visual_type="Historical")) == "historical"
    assert niche_for_clip(_clip(visual_type="Technology")) == "modern"
    assert niche_for_clip(_clip(visual_type="Cinematic")) == "general"


def test_provider_order_for_niche_prioritizes_archives_for_historical():
    assert provider_order_for_niche("historical")[0] is not pexels_video
    assert provider_order_for_niche("modern")[0] is pexels_video
    assert provider_order_for_niche("general")[0] is pexels_video


def test_satisfied_uses_configurable_thresholds():
    low_score_asset = ScoredAsset(hit=_hit(), score=55)
    with patch.object(settings, "documentary_high_quality_score", 90), \
         patch.object(settings, "documentary_min_score", 50):
        assert _satisfied([low_score_asset], required=2) is False  # 1 acceptable hit, needs 2
        assert _satisfied([low_score_asset], required=1) is True   # 1 acceptable hit meets required=1
    with patch.object(settings, "documentary_high_quality_score", 50):
        assert _satisfied([low_score_asset], required=5) is True  # now counts as high-quality on its own


def test_high_quality_hit_stops_search_early():
    clip = _clip(visual_type="Technology")  # modern niche: required video=1, image=1
    video_hit = _hit(source="pexels", source_id="v1", media_type="video")
    image_hit = _hit(source="pexels_images", source_id="i1", media_type="image")

    with patch_providers(pexels_video=[video_hit], pexels_images=[image_hit]) as mocks, \
         patch("app.scoring.embed", return_value=[1.0, 0.0]):
        video_hits, image_hits = asyncio.run(search_clip(clip, used_assets=set()))

    assert video_hits and image_hits
    for later_provider in ["pixabay_images", "internet_archive", "wikimedia", "nasa"]:
        mocks[later_provider].assert_not_called()


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
        asyncio.run(search_clip(clip, used_assets=set()))

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
        video_hits, _image_hits = asyncio.run(search_clip(clip, used_assets=set()))

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
        video_hits, _image_hits = asyncio.run(search_clip(clip, used_assets=set()))

    source_ids = {a.hit.source_id for a in video_hits}
    assert source_ids == {"vert", "horiz"}  # both kept — orientation no longer gates sourcing


def test_search_clip_rejects_known_too_low_res_hit_before_download():
    clip = _clip(visual_type="Technology")
    tiny_hit = _hit(source="pexels", source_id="tiny", width=320, height=240)  # below the 480 floor
    good_hit = _hit(source="pixabay", source_id="good", width=1920, height=1080)

    with patch_providers(pexels_video=[tiny_hit, good_hit]), \
         patch("app.scoring.embed", return_value=[1.0, 0.0]):
        video_hits, _image_hits = asyncio.run(search_clip(clip, used_assets=set()))

    source_ids = {a.hit.source_id for a in video_hits}
    assert source_ids == {"good"}  # too-low-res rejected pre-download, nothing to normalize


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
             patch("app.documentary_pipeline._download_asset", new=AsyncMock(return_value=fake_asset)):
            entry, downloads_so_far = asyncio.run(
                _resolve_clip(clip, clips_root, used_assets=set(), allow_duplicates=False,
                              downloads_so_far=0, limit=None)
            )

        assert entry.placeholder is False
        assert entry.asset_metadata.width == 1920 and entry.asset_metadata.height == 1080
        assert downloads_so_far == 1
        assert vertical_path.exists()  # normalized in place, not discarded
        _duration, resolution = probe(str(vertical_path))
        assert resolution == "1920x1080"


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
             patch("app.documentary_pipeline._download_asset", side_effect=fake_download_asset):
            entry, downloads_so_far = asyncio.run(
                _resolve_clip(clip, clips_root, used_assets=set(), allow_duplicates=False,
                              downloads_so_far=0, limit=None)
            )

        assert entry.placeholder is False
        assert entry.asset_metadata.source_id == "second"  # top candidate (too low-res) skipped, next-best used
        assert downloads_so_far == 2  # both download attempts counted
        assert not tiny_path.exists()  # rejected candidate's file discarded
        assert horizontal_path.exists()  # winning candidate's file kept


def test_rank_acceptable_assets_orders_by_score_no_video_preference():
    video = ScoredAsset(hit=_hit(source_id="v", media_type="video"), score=70)
    image = ScoredAsset(hit=_hit(source_id="i", media_type="image"), score=95)  # scores higher than the video
    below_min = ScoredAsset(hit=_hit(source_id="low", media_type="video"), score=1)

    with patch.object(settings, "documentary_min_score", 50):
        ranked = rank_acceptable_assets([video, below_min], [image])

    assert [a.hit.source_id for a in ranked] == ["i", "v"]  # image wins purely on score, below_min excluded


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


def test_cross_clip_dedup_excludes_used_asset_unless_allowed():
    clip = _clip()
    hit = _hit(source="pexels", source_id="dup1")
    used = {("pexels", "dup1")}

    with patch_providers(pexels_video=[hit]), patch("app.scoring.embed", return_value=[1.0, 0.0]):
        video_hits, _ = asyncio.run(search_clip(clip, used_assets=used, allow_duplicates=False))
    assert video_hits == []

    with patch_providers(pexels_video=[hit]), patch("app.scoring.embed", return_value=[1.0, 0.0]):
        video_hits, _ = asyncio.run(search_clip(clip, used_assets=used, allow_duplicates=True))
    assert len(video_hits) == 1


def test_search_clip_advances_to_next_tier_when_all_hits_from_a_tier_are_already_used():
    """FIX2 gap: a keyword tier whose raw hits are non-empty but EVERY one
    is already in used_assets must be treated the same as an empty tier and
    escalate to the next keyword tier (canva -> fallback -> semantic),
    instead of stopping there (because raw_hits was technically non-empty)
    and silently contributing zero real candidates for this provider."""
    clip = _clip()
    used_hit = _hit(source="pexels", source_id="dup", text="")
    fresh_hit = _hit(source="pexels", source_id="fresh", text="")
    used = {("pexels", "dup")}

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


def test_resolve_clip_excludes_already_used_candidate_across_rows_and_falls_through():
    """FIX2 end-to-end: two rows whose keywords return overlapping stock
    results (the same top candidate for both) must not resolve to the same
    clip. Row 1 wins the higher-scored candidate; row 2's search must
    exclude that now-used candidate and pick the next-best alternative
    instead of repeating it, sharing one used_assets set across both calls
    exactly like the real run() loop does."""
    import subprocess

    clip1 = _clip(clip_number=1)
    clip2 = _clip(clip_number=2)
    hit_a = _hit(source="pexels", source_id="dupA", width=1920, height=1080)
    hit_b = _hit(source="pexels", source_id="dupB", width=1920, height=1080)

    def fake_score(hit, clip, query_embedding):
        return 90.0 if hit.source_id == "dupA" else 50.0  # dupA always the top-scored candidate

    with tempfile.TemporaryDirectory() as tmpdir:
        clips_root = Path(tmpdir)
        path_a = clips_root / "a.mp4"
        path_b = clips_root / "b.mp4"
        for path in (path_a, path_b):
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=1920x1080:r=25:d=1",
                 "-pix_fmt", "yuv420p", str(path)],
                capture_output=True, check=True,
            )

        asset_a = AssetInfo(path=str(path_a), source="pexels", source_id="dupA", media_type="video",
                             width=1920, height=1080, duration=5.0, score=90.0)
        asset_b = AssetInfo(path=str(path_b), source="pexels", source_id="dupB", media_type="video",
                             width=1920, height=1080, duration=5.0, score=50.0)

        async def fake_download(candidate, dest_dir, prefix, index, target_duration):
            return asset_a if candidate.hit.source_id == "dupA" else asset_b

        used_assets: set[tuple[str, str]] = set()

        with patch_providers(pexels_video=[hit_a, hit_b]), \
             patch("app.documentary_pipeline.score_asset", side_effect=fake_score), \
             patch("app.documentary_pipeline._download_asset", new=AsyncMock(side_effect=fake_download)):
            entry1, _ = asyncio.run(
                _resolve_clip(clip1, clips_root, used_assets, allow_duplicates=False, downloads_so_far=0, limit=None)
            )
            entry2, _ = asyncio.run(
                _resolve_clip(clip2, clips_root, used_assets, allow_duplicates=False, downloads_so_far=0, limit=None)
            )

    assert entry1.asset_metadata.source_id == "dupA"  # row 1: the higher-scored candidate wins
    assert entry2.asset_metadata.source_id == "dupB"  # row 2: dupA already used -> falls through to dupB
    assert entry1.asset_metadata.source_id != entry2.asset_metadata.source_id


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


def test_keywords_are_concrete_defaults_to_true_when_self_check_call_fails():
    """The self-check is a soft guard, not a hard gate — if the Groq call
    itself fails, assume the keywords are fine rather than blocking search."""
    with patch("app.documentary_pipeline.generate_json", side_effect=RuntimeError("network down")):
        assert _keywords_are_concrete(["some keyword"]) is True


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


def test_max_downloads_limit_stops_later_clips_without_provider_calls():
    clips = [_clip(clip_number=1, start="00:00:00", end="00:00:05"),
             _clip(clip_number=2, start="00:00:05", end="00:00:10")]
    hit = _hit(source="pexels", source_id="v1")

    def fake_assemble(project_dir, timeline):  # pass timeline through untouched
        return project_dir / "final_video.mp4", timeline

    with tempfile.TemporaryDirectory() as tmpdir, \
         patch("app.documentary_pipeline.generate_table", return_value=clips), \
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
    mock_generate_table.assert_called_once_with("some script", whisper_words)
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

    mock_generate_table.assert_called_once_with("A script read from a file on disk.", None)


if __name__ == "__main__":
    test_assign_timings_clamps_and_accumulates()
    test_assign_timings_normalizes_invalid_visual_type_instead_of_raising()
    test_assign_timings_uses_real_whisper_durations_when_provided()
    test_assign_timings_floors_to_min_duration_when_whisper_words_run_out()
    test_assign_timings_generates_coverage_rows_when_narration_runs_past_the_table()
    test_force_contiguity_corrects_a_gap_between_consecutive_clips()
    test_enforce_max_clip_duration_splits_an_8s_row_into_multiple_sub_5s_rows()
    test_split_long_clip_gives_each_sub_row_its_own_distinct_keyword()
    test_split_long_clip_forces_distinct_keywords_even_when_generation_is_degenerate()
    test_assign_timings_forces_contiguity_when_a_real_narration_pause_creates_a_gap()
    test_assign_timings_floors_degenerate_short_real_span_and_cascades_forward()
    test_assign_timings_content_match_avoids_drift_from_punctuation_token_mismatch()
    test_assign_timings_falls_back_to_word_count_when_content_match_confidence_is_too_low()
    test_score_asset_weighting()
    test_niche_for_clip()
    test_provider_order_for_niche_prioritizes_archives_for_historical()
    test_satisfied_uses_configurable_thresholds()
    test_high_quality_hit_stops_search_early()
    test_search_clip_uses_table_canva_keyword_as_primary_query()
    test_search_clip_uses_fallback_keyword_when_canva_keyword_returns_nothing()
    test_known_and_not_horizontal_rejects_only_confirmed_non_16_9()
    test_search_clip_keeps_non_16_9_hit_with_known_metadata_for_post_download_normalization()
    test_search_clip_rejects_known_too_low_res_hit_before_download()
    test_resolve_clip_normalizes_downloaded_asset_that_turns_out_vertical()
    test_rank_acceptable_assets_orders_by_score_no_video_preference()
    test_resolve_clip_rejects_and_falls_through_when_top_candidate_is_too_low_res()
    test_pick_best_asset_returns_single_highest_scorer()
    test_pick_best_asset_ignores_below_min_score()
    test_pick_best_asset_returns_none_when_no_candidates()
    test_cross_clip_dedup_excludes_used_asset_unless_allowed()
    test_search_clip_advances_to_next_tier_when_all_hits_from_a_tier_are_already_used()
    test_resolve_clip_excludes_already_used_candidate_across_rows_and_falls_through()
    test_safe_search_swallows_errors_and_timeouts()
    test_generate_semantic_keywords_regenerates_once_when_first_attempt_is_abstract()
    test_generate_semantic_keywords_skips_regeneration_when_first_attempt_is_already_concrete()
    test_keywords_are_concrete_defaults_to_true_when_self_check_call_fails()
    test_run_never_fails_a_clip_with_no_assets()
    test_max_downloads_limit_stops_later_clips_without_provider_calls()
    test_run_threads_whisper_words_into_table_and_render_when_audio_provided()
    test_run_surfaces_alignment_warning_in_result()
    test_run_skips_audio_mux_entirely_when_no_audio_path_given()
    test_run_surfaces_audio_mux_error_without_crashing()
    test_run_propagates_transcription_error_without_falling_back_to_estimate()
    test_run_reads_script_from_script_path()
    print("OK")
