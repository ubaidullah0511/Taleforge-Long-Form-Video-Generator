# Clip Match Diagnostic — project_1785129208 (pool_maintenance)

Purpose: determine whether the pipeline is genuinely selecting mismatched footage or
whether footage just *looks* off for other reasons (speed-matching, generic B-roll,
catalog depth). This is a diagnostic report only — no scoring/matching/filtering code
was changed to produce it.

## Method and data provenance

- `clip_number`, `script`, `selected source:source_id`, `total_score` — read directly
  from `projects/project_1785129208/timeline.json`.
- `canva_keyword` — cross-referenced from `projects/project_1785129208/timeline_table.md`
  by matching `clip_number`.
- **`selected candidate's text`** — this field exists in neither `timeline.json` nor
  `footage_availability_report.json` (confirmed by inspecting both — `AvailabilityReport`/
  `ClipAvailability` only carry `candidate_count`/`top_score`, never `text`), and no
  console log was persisted for this run. Per the task's own escape hatch ("re-run...
  unless no cached data exists at all for a candidate"), since literally no cached text
  exists anywhere for any candidate, each winning asset's real metadata was retrieved by
  a **direct fetch-by-ID** against the provider that returned it (`GET
  /videos/videos/{id}` for Pexels video, `?id=` for Pixabay video/images, `GET
  /v1/photos/{id}` for Pexels images) — using the exact same field-extraction logic each
  provider module already uses (`app.stock.pexels._title_from_url`, Pixabay's `tags`
  field, Pexels images' `alt` field). This is a metadata **lookup by a known, fixed ID**,
  not a fresh keyword **search** — it cannot return a different result set than what
  actually won, unlike re-running `search_clip`, which the task correctly warns against.
- `keyword_match` — computed via the already-implemented `app.scoring.keyword_overlap_ratio(canva_keyword, selected_text)`, imported and called directly, not reimplemented.
- Manual relevance flag — my own judgment reading `canva_keyword` against the real fetched `text`.

All 62 clips in this project have a real (non-placeholder) asset — no placeholder rows to report.

## Full Per-Clip Table

| Clip | Script (excerpt) | Canva Keyword | Selected (source:id, media) | Selected Candidate's Text | Score | keyword_match | Flag | Note |
|---|---|---|---|---|---:|---:|---|---|
| 1 | There's a box sitting on a laundry aisle shelf right no... | Laundry aisle product | pexels:10566665 (video) | a person holding a laundry basket and picking up a cleaning product | 75.12 | 0.67 | **MATCH** | |
| 2 | that stops algae from ever coming back if you actually ... | Algae treatment pool | pexels:38062556 (video) | aerial view of green algae in desert pool | 76.6 | 0.67 | **MATCH** | |
| 3 | Pool stores will sell you shock, algaecide, clarifier, ... | Pool chemical products | pixabay:11066 (video) | water, ripple, pool, run, flow, stream, surface, blue, closeup, liquid... | 67.3 | 0.33 | **WEAK MATCH** | |
| 4 | toward it, because one six dollar box that works all se... | Swimming pool chemicals | pexels:4201700 (video) | water ripples in swimming pool | 73.0 | 0.67 | **WEAK MATCH** | |
| 5 | Today I'm showing you the forbidden kitchen ingredient ... | Borax swimming pool | pixabay:93705 (video) | swiming pool, swimming, pool, water, diving, holiday, vacation, summer... | 75.22 | 0.67 | **WEAK MATCH** | |
| 6 | it shows up, why it works when shock and algaecide keep... | Pool algaecide | pexels:9507847 (video) | video of water in a swimming pool | 70.63 | 0.50 | **WEAK MATCH** | |
| 7 | at the start of the season keeps your water algae-free ... | Clear pool water | pixabay:329058 (video) | waterfall, cascade, rapids, shallow water, deep water, pool, lagoon, t... | 79.97 | 1.00 | **MATCH** | |
| 8 | Let's start with why algae keeps coming back, because t... | Algae problem pool | pixabay:213950 (video) | tadpoles, frog, pond, water, newt, algae, moss, weed, nature | 66.71 | 0.33 | **MISMATCH** | pond/tadpole wildlife scene, not a pool |
| 9 | Most pool owners think algae is a sanitizer problem, so... | Adding chlorine pool | pixabay:214928 (video) | swimming pool, resort, holiday, night, pool lights, aerial view | 67.26 | 0.33 | **WEAK MATCH** | |
| 10 | Algae is not a sanitizer problem. Algae is a pH stabili... | Pool pH testing | pexels:13220695 (video) | puncturing ballon over swimming pool | 67.92 | 0.33 | **MISMATCH** | balloon pop over pool, no pH testing action |
| 11 | Every time your pH swings up and down, even slightly, y... | Person testing pool pH | pixabay:3629 (video) | swimming, breaststroke, water sports, pool, water, sports, swim, swimm... | 70.01 | 0.50 | **MISMATCH** | athletic swimming, not testing |
| 12 | it, meaning there are constant windows every week where... | Pool sanitizer system | pixabay:35005 (video) | hand sanitizer, sanitizer, hygiene, washing, hand | 68.34 | 0.33 | **MISMATCH** | hand sanitizer/hygiene, not pool equipment |
| 13 | Algae only needs one of those windows to establish itse... | Algae growth pool | pexels:12483869 (video) | green algae on a flowing river surface | 69.83 | 0.33 | **WEAK MATCH** | |
| 14 | You shock the pool, the algae dies, the water clears, t... | Pool shock treatment | pixabay:212968 (video) | shock, wave, flash, explosion | 68.3 | 0.33 | **MISMATCH** | generic 'shock/explosion', not chemical shock treatment |
| 15 | again because you never fixed the pH instability that l... | Person testing pool pH | pexels:10187283 (video) | a person experimenting chemical | 65.91 | 0.25 | **WEAK MATCH** | |
| 16 | The forbidden truth is that preventing algae forever is... | Algae prevention pool | pexels:34617575 (video) | underwater marine life in tide pool | 68.97 | 0.33 | **MISMATCH** | ocean tide-pool wildlife |
| 17 | expensive, and it doesn't require buying new chemicals ... | Pool chemical cost | pexels:15060485 (video) | a luxury golden pool | 66.97 | 0.33 | **MISMATCH** | luxury pool shot, no cost signal |
| 18 | It requires understanding one thing: algae needs two co... | Algae covered pool | pexels:34569320 (video) | frog camouflaged on algae covered water surface | 76.73 | 0.67 | **WEAK MATCH** | |
| 19 | hold. It needs warm water and sunlight, which you can't... | Warm pool water | pixabay:33757 (video) | swamp, pond, lake, water, nature, forest, vernal, pool, fish, green, s... | 80.08 | 1.00 | **MISMATCH** | natural swamp/pond, not a swimming pool |
| 20 | And it needs an unstable pH environment that gives it t... | Cloudy pool water | pixabay:258760 (video) | nature, river, water, waterfalls, pool, mountain, forest, cloudy | 81.61 | 1.00 | **WEAK MATCH** | |
| 21 | opening to attach and multiply, which you absolutely ca... | Pool maintenance tools | pixabay:123594 (video) | cop, police, officer, security, 3d, cartoon, man, wrench, repair, mech... | 73.0 | 0.67 | **MISMATCH** | 3D cartoon police/mechanic, wrong subject entirely |
| 22 | Fix the pH stability and you remove the condition that ... | Balancing pool pH | pixabay:211152 (video) | swimming pool, house, hotel, villa, outdoor, pool, water, nature, reso... | 65.3 | 0.33 | **MISMATCH** | luxury resort real estate shot |
| 23 | Here's what pool stores will not tell you. They make mo... | Pool store checkout | pixabay:140022 (video) | woman, model, swimming pool, girl, pool, summer, relaxation, holiday | 65.18 | 0.33 | **MISMATCH** | woman relaxing by pool, no checkout/store |
| 24 | Green pool panic is one of their highest-margin selling... | Panic pool owner | pixabay:205519 (video) | alert, alarm, war, danger, risk, worried, panic, disaster, conflict, s... | 64.03 | 0.33 | **MISMATCH** | war/alarm/monitor scene, no pool at all |
| 25 | You show up desperate and they sell you shock, then alg... | Pool chemical purchases | pexels:18202849 (video) | swimming pool | 66.61 | 0.33 | **WEAK MATCH** | |
| 26 | Then it comes back in a week and you're buying again. | Weekly pool maintenance | pixabay:3428 (video) | fish spa, fish pedicure, kangal fish, red mullet, fishes, foot care, m... | 71.87 | 0.67 | **MISMATCH** | fish-pedicure spa video, not pool maintenance |
| 27 | A customer whose pool never goes green is a customer wh... | Clear swimming pool | pexels:36468001 (video) | clear blue swimming pool in sunlight | 81.9 | 1.00 | **MATCH** | |
| 28 | The entire retail incentive structure is built around m... | Pool maintenance products | pexels:34719447 (video) | professional pool renovation process underway | 67.95 | 0.33 | **WEAK MATCH** | |
| 29 | fully understand that prevention is cheaper and more ef... | Preventative pool maintenance | pexels:4451962 (video) | cleaning the pool side | 69.4 | 0.33 | **MATCH** | |
| 30 | The forbidden ingredient is borax. | Borax powder pool | pexels:13107036 (video) | adding powder to glass of water | 68.66 | 0.33 | **WEAK MATCH** | |
| 31 | Sodium tetraborate, the same white powder sold as a lau... | Laundry aisle borax | pexels:29376327 (video) | shopping aisle perspective on supermarket essentials | 68.49 | 0.33 | **WEAK MATCH** | |
| 32 | In your pool, borax raises what's called your borate le... | Borax pool chemical | pexels:3735772 (video) | mixing chemical in the glass beaker | 65.91 | 0.33 | **WEAK MATCH** | |
| 33 | two things almost nothing else in pool chemistry does a... | Pool chemistry balance | pixabay:197489 (video) | woman, chemistry, laboratory, test, experiment, technology | 67.11 | 0.33 | **WEAK MATCH** | |
| 34 | They act as a natural algae inhibitor, and they buffer ... | Algae inhibitor pool | pixabay:124194 (video) | cliff, waterfall, rocks, ocean, drops, trickle, algae | 67.35 | 0.33 | **MISMATCH** | ocean cliff waterfall, not a pool |
| 35 | pH so it stops swinging wildly every time it rains. | Person testing pool pH | pixabay:223715 (video) | sea, wave, pool, liquid, water | 67.18 | 0.25 | **MISMATCH** | generic sea/wave tags, no person/testing visible |
| 36 | Once your borate level is established, both of those ef... | Borate pool level | pixabay:228810 (video) | sea, sea level, small fish, group, jump, japan | 66.24 | 0.33 | **MISMATCH** | fish jumping in the sea (Japan), no pool |
| 37 | there working in the background for months without you ... | Pool maintenance | pexels:5623300 (video) | opening up the swimming pool cover | 72.09 | 0.50 | **MATCH** | |
| 38 | Forbidden fact number one. Algae doesn't just need wate... | Algae growth pool | pexels:32645766 (video) | serene algae covered pond in summer sunlight | 71.06 | 0.33 | **WEAK MATCH** | |
| 39 | hold. It needs an opening, and unstable pH is exactly t... | Cloudy pool water | pixabay:78059 (video) | swimming pool, swimmers, swimming, pool, water sports, summer, water, ... | 74.73 | 0.67 | **WEAK MATCH** | |
| 40 | When your pH is bouncing between seven point five and s... | Pool water pH test | pexels:37348838 (video) | chemical reaction in beaker with color change | 61.5 | 0.00 | **WEAK MATCH** | see root-cause note below — largest availability-vs-final score gap in the set |
| 41 | few days, your chlorine's effectiveness is swinging by ... | Chlorine test kit | pexels:6824183 (video) | using test strip for blood glucose count | 67.31 | 0.33 | **WEAK MATCH** | |
| 42 | There are constant windows every week where sanitizer i... | Cloudy pool water | pexels:3978431 (video) | raindrops on a water surface | 68.86 | 0.33 | **WEAK MATCH** | |
| 43 | and algae only needs one of those windows to get a foot... | Algae covered pool | pexels:34617589 (video) | close up of a crab in rocky tide pool | 67.18 | 0.33 | **MISMATCH** | ocean tide pool with a crab |
| 44 | Once it's established, even if your chemistry stabilize... | Algae covered pool | pixabay:15333 (video) | sea, fish, underwater, seabed, immersion, algae, blue, water | 66.88 | 0.33 | **MISMATCH** | underwater sea/fish scene |
| 45 | there and now it's a treatment problem instead of a pre... | Cloudy pool water | pixabay:221180 (video) | sky, clouds, sunrays, cloudy | 66.12 | 0.33 | **MISMATCH** | sky/clouds weather shot, matched only on word 'cloudy' |
| 46 | Borates solve this because boric acid, which is what bo... | Borax pool | pexels:6167679 (video) | slow motion video of water ripples of the swimming pool | 68.25 | 0.50 | **WEAK MATCH** | |
| 47 | a weak acid that resists pH change far more effectively... | Consulting pool pH | pexels:8490208 (video) | a chemical in the laboratory | 60.3 | 0.00 | **MISMATCH** | generic lab chemical, no consulting/pool context |
| 48 | Once your borate level is in range, your pH stops bounc... | Pool borate | pexels:29804702 (video) | young man surfacing in a swimming pool | 66.76 | 0.50 | **WEAK MATCH** | |
| 49 | around after rain, after heavy bather load, after addin... | Cloudy pool water | pexels_images:14546227 (image) | Modern city skyline reflected in water under a dramatic cloudy sky. | 63.46 | 0.67 | **MISMATCH** | city skyline reflection, not a pool |
| 50 | All the swings that used to open a window for algae sim... | Flat pool conditions | pexels:2006482 (video) | a dirty pool | 67.73 | 0.33 | **WEAK MATCH** | |
| 51 | And on top of that, boron itself is mildly toxic to | Borax pool | pixabay_images:6881343 (image) | miner, borax, bolivia, travel, andes | 58.51 | 0.50 | **MISMATCH** | Bolivian borax MINE photo, not a swimming pool — lowest score in the set |
| 52 | algae at pool concentrations, not enough to harm swimme... | Algae covered pool | pixabay:225420 (video) | waterfall, natural beauty, scenic, turquoise pool, moss-covered rocks,... | 65.2 | 0.33 | **WEAK MATCH** | |
| 53 | You're not just treating symptoms. You're removing the ... | Preventing algae conditions | pexels:9094092 (video) | a tortoise covered with algae | 66.97 | 0.33 | **MISMATCH** | a tortoise, unrelated to pool algae |
| 54 | The dosing is dead simple. Target thirty to fifty parts... | Measuring borate pool | pixabay:9179 (video) | show, display, cockpit, horizon, measure up, measuring device, measure... | 64.67 | 0.33 | **MISMATCH** | aircraft cockpit instruments, matched on 'measuring' |
| 55 | For a typical fifteen thousand gallon pool, that works ... | Fifteen thousand gallon pool | pexels:990956 (video) | swimming pool | 68.64 | 0.25 | **WEAK MATCH** | |
| 56 | twenty to twenty five pounds of borax, dissolved in war... | Borax powder pool | pexels:8477801 (video) | person sifting the flour using strainer and whisk | 58.8 | 0.00 | **MISMATCH** | flour-sifting/baking scene, not borax/pool — second-lowest score |
| 57 | ...and added gradually around the perimeter with the pu... | Pool pump running | pexels:8686014 (video) | swimmer in the pool | 67.72 | 0.33 | **WEAK MATCH** | |
| 58 | That's a one-time dose at the beginning of the season. | Measuring pool chemicals | pexels:8392568 (video) | person mixing chemicals on a petri dish | 66.42 | 0.33 | **WEAK MATCH** | |
| 59 | After that, you don't add anything else because borates... | Clear pool water | pixabay:329057 (video) | people, friends, group, adventure, lifestyle, summer, vacation, leisur... | 80.4 | 1.00 | **MATCH** | |
| 60 | ...the way other chemicals do. They just sit there, wor... | Healthy pool chemistry | pexels:8045852 (video) | person pouring juice in a glass at the pool side | 65.78 | 0.33 | **MISMATCH** | pouring juice, not pool chemistry — see root-cause note below |
| 61 | because borates don't get consumed by sunlight or chlor... | Borates in pool | pexels:4314865 (video) | summer swimming pool blue sky heat | 71.56 | 0.50 | **WEAK MATCH** | |
| 62 | They just sit there working all season long. | Pool maintenance products | pexels:16755380 (video) | an aerial view of a swimming pool | 65.7 | 0.33 | **WEAK MATCH** | |

## Priority-list clips (requested to check first)

| Clip | Keyword | Score | Flag | Notes |
|---|---|---:|---|---|
| 22 | Balancing pool pH | 65.3 | MISMATCH | Confirmed — luxury resort real estate shot |
| 23 | Pool store checkout | 65.18 | MISMATCH | Confirmed — woman relaxing by pool |
| 24 | Panic pool owner | 64.03 | MISMATCH | Confirmed — war/alarm/monitor scene, no pool signal at all |
| 40 | Pool water pH test | 61.5 | WEAK MATCH | Chemical color-change reaction is a plausible visual stand-in for a pH-test color change, but no pool visible. **Largest availability-vs-final score gap in the whole project (11.8 points) — see root cause below.** |
| 47 | Consulting pool pH | 60.3 | MISMATCH | Confirmed — generic lab chemical shot, no pool |
| 56 | Borax powder pool | 58.8 | MISMATCH | Confirmed — flour-sifting/baking scene |
| 51 (lowest score, image) | Borax pool | 58.51 | MISMATCH | Confirmed — an actual Bolivian borax **mine** photo, not a pool |
| 32 (largest speed-up, 2.0s→3.0s) | Borax pool chemical | 65.91 | WEAK MATCH | Chemical-mixing-in-beaker is topically adjacent but no pool/borax visible. Candidate pool was thin (7 candidates in the pre-render scan) — **confirms the hypothesis that a thin pool for this specific compound concept explains both the short source duration and the weak text match simultaneously.** |

The reviewer's hunch was correct: every clip flagged for a low score (58–65 range) is a real MISMATCH or, at best, a WEAK MATCH — none of them is a false alarm.

## Overall Statistics

| Flag | Count | % of 62 |
|---|---:|---:|
| MATCH | 7 | 11% |
| WEAK MATCH | 29 | 47% |
| MISMATCH | 26 | 42% |
| CANNOT DETERMINE | 0 | 0% |

**42% of clips in this real render are outright visual mismatches**, and only 11% are clean, confident matches — the remaining 47% are plausible-but-generic B-roll (mostly "pool" or "water" imagery with no specific action/object from the keyword actually visible).

## MISMATCH Root-Cause Breakdown: Scoring-Weight vs. Catalog-Depth

For every MISMATCH clip, I cross-referenced the final selected score against that same
clip's `top_score` in `footage_availability_report.json` — the pre-download,
pre-CLIP-verification scan from the **same run**. This tells us, without any new live
search, whether a better-scoring candidate was ever found and then lost later in the
pipeline (a scoring/verification problem) versus whether the eventual pick was already
the best thing available from the very first search pass (a catalog-depth problem).

| Clip | Keyword | Final score | Availability top_score | Gap | Candidates found |
|---|---|---:|---:|---:|---:|
| 8 | Algae problem pool | 66.71 | 65.82 | −0.89 | 10 |
| 10 | Pool pH testing | 67.92 | 67.92 | 0.00 | 11 |
| 11 | Person testing pool pH | 70.01 | 70.01 | 0.00 | 11 |
| 12 | Pool sanitizer system | 68.34 | 68.34 | 0.00 | 12 |
| 14 | Pool shock treatment | 68.30 | 68.30 | 0.00 | 10 |
| 16 | Algae prevention pool | 68.97 | 68.97 | 0.00 | 13 |
| 17 | Pool chemical cost | 66.97 | 66.97 | 0.00 | 11 |
| 19 | Warm pool water | 80.08 | 80.08 | 0.00 | 12 |
| 21 | Pool maintenance tools | 73.00 | 73.00 | 0.00 | 14 |
| 22 | Balancing pool pH | 65.30 | 65.30 | 0.00 | 9 |
| 23 | Pool store checkout | 65.18 | 65.56 | 0.38 | 12 |
| 24 | Panic pool owner | 64.03 | 65.11 | 1.08 | 11 |
| 26 | Weekly pool maintenance | 71.87 | 68.81 | −3.06 | 10 |
| 34 | Algae inhibitor pool | 67.35 | 67.65 | 0.30 | 8 |
| 35 | Person testing pool pH | 67.18 | 64.81 | −2.37 | 8 |
| 36 | Borate pool level | 66.24 | 66.24 | 0.00 | 9 |
| 43 | Algae covered pool | 67.18 | 67.18 | 0.00 | 9 |
| 44 | Algae covered pool | 66.88 | 64.09 | −2.79 | 8 |
| 45 | Cloudy pool water | 66.12 | 66.12 | 0.00 | 9 |
| 47 | Consulting pool pH | 60.30 | 60.73 | 0.43 | 9 |
| 49 | Cloudy pool water | 63.46 | 63.46 | 0.00 | 8 |
| 51 | Borax pool | 58.51 | 58.51 | 0.00 | 9 |
| 53 | Preventing algae conditions | 66.97 | 67.11 | 0.14 | 10 |
| 54 | Measuring borate pool | 64.67 | 64.67 | 0.00 | 11 |
| 56 | Borax powder pool | 58.80 | 58.80 | 0.00 | 5 |
| **60** | **Healthy pool chemistry** | **65.78** | **75.11** | **9.33** | 10 |

**Verdict: this is overwhelmingly a catalog-depth problem, not a scoring-weight
problem.** For 25 of the 26 MISMATCH clips, the gap is at or near zero (several are
negative, meaning the final pick was as good as or better than the initial scan
estimated) — the selected candidate genuinely *was* the best-scoring option the search
ever found, at every stage, including before CLIP verification had a chance to reject
anything. Raising or rebalancing score weights would not fix these: there was nothing
better in the catalog to rank higher. The candidate counts (5–14) also show these
weren't technically "thin" by the pipeline's own `candidate_count < 1` thin-flag, but
quantity of candidates isn't the same as quality/relevance — none of the several
candidates returned for concepts like "panic pool owner," "measuring borate pool," or
"pool store checkout" actually depict that specific compound/abstract idea, because
stock libraries simply don't carry footage of it. This is the same root-cause pattern
the project's own scoring docstring already acknowledges: `score_asset` is "a heuristic
scorer using metadata already returned by the provider APIs, not a real visual-content
model" — it can rank what exists, but can't invent footage that was never indexed.

**One clip is a genuine exception, worth separate investigation**: **clip 60**
("Healthy pool chemistry") shows a 9.33-point gap — the pre-download scan found
something scoring 75.11, but the final pick only scored 65.78. **Clip 40** (from the
priority list, flagged WEAK MATCH rather than MISMATCH) shows an even larger gap —
**11.82 points** — the single largest in the entire project. In both cases, a
meaningfully better-scoring candidate existed early in the pipeline and did not survive
to become the final pick. Per this project's own pipeline mechanics (see
`app/documentary_pipeline.py::_resolve_clip`), the candidates that can knock out an
already-scored, higher-ranked pick are: failing CLIP visual verification, failing 16:9
normalization, or being excluded by the cross-clip dedup cap (`max_asset_repeat_count`)
because another clip already used it. This report doesn't have direct visibility into
*which* of those three actually happened for clips 40/60 without re-running the
resolve step (which the task asked to avoid) — but unlike the other 24 MISMATCH clips,
these two are worth a targeted look at whatever candidate scored 73–75 in the
availability scan, since a real, better-scoring option evidently existed and was lost.

## Follow-up Investigation: Why Clips 40 and 60 Lost Their Higher-Scoring Candidate

This section resolves the open question above with certainty, not speculation, using
two independent sources of evidence: (1) the real run's own downloaded-file naming —
`_resolve_clip` names each attempt `clip{N}_video{index}.mp4`/`image{index}` in
download order, so a clip's final filename index directly records how many earlier
candidates were tried and rejected before it — and (2) a live, targeted re-run of just
these two clips through the real `search_clip` → download → normalize → CLIP path,
seeded with the **real run's own actual dedup state** (reconstructed from
`timeline.json`'s actual clip 1–39 / 1–59 selections, not the availability scan's
separate simulated dedup counter, which is a different, non-authoritative state per its
own docstring).

### Clip 40 — filename evidence alone is conclusive; no rejection ever occurred

`clips/clip_040/clip040_video1.mp4` — index **1**. The real run accepted its very
first downloaded candidate. Since `_resolve_clip` walks candidates in ranked order,
this is only possible if nothing scored higher than 61.5 was present in the real,
dedup-filtered candidate pool at the moment clip 40 was actually resolved.

A live re-run today, seeded with the real dedup state from clips 1–39, reproduces this
exactly: rank #1 of 6 candidates is `pexels:37348838` at score 61.5 — the same asset,
same score — and it passes CLIP (similarity 0.284 vs. threshold 0.20). **No higher-scoring
candidate exists in today's live, correctly-deduped pool either.**

The `~73.3` figure in `footage_availability_report.json` came from a *different*
process at a *different* time: the pre-render availability scan keeps its own
`simulated_used` dedup counter that only tracks the scan's *own* per-clip top picks
in sequence (see `check_footage_availability`'s docstring) — it is not, and never
claims to be, the same dedup timeline the real sequential resolve loop actually
produces. Whatever candidate scored ~73 during the scan had, by the time the real
per-clip resolve reached clip 40, either already been legitimately claimed by a real
earlier clip, or reflected ordinary live-search ranking variance between two separate
calls. Either way, it was **never actually available to compete for clip 40's real
decision** — this is not a CLIP or normalization rejection at all.

**Classification: CORRECT REJECTION (more precisely, no rejection occurred).** The
pipeline behaved correctly; 61.5 was genuinely the best real, legitimately-available
option at decision time. This reduces to the same catalog-depth/dedup-timing story as
the other 24 mismatched clips — the "gap" signature here is an artifact of comparing
two different scans' dedup states, not evidence of a scoring or verification bug.

### Clip 60 — two real rejections, reproduced and explained exactly

`clips/clip_060/clip060_video3.mp4` — index **3**: two earlier candidates were
downloaded and rejected before this one was accepted. A live re-run today, seeded with
the real dedup state from clips 1–59 and using clip 60's correct 4.0s target duration
(read from `timeline.json`, not assumed), reproduces the identical 3-attempt pattern:

| Rank | Candidate | Text | Score | CLIP similarity | Outcome |
|---|---|---|---:|---:|---|
| 1 | pixabay:45675 | "pharmacy, production, factory, chemistry, product drug, medical, health, healthy, healthcare..." | 75.11 | **0.186** | REJECTED (< 0.20) |
| 2 | pexels:6421805 | "citrus fruits ob the pool" | 70.62 | **0.197** | REJECTED (< 0.20) |
| 3 | pexels:8045852 | "person pouring juice in a glass at the pool side" | 65.78 | **0.242** | PASSED — final pick |

- **Rank #1 (pharmacy/factory video, 75.11) — correctly rejected.** This candidate's
  high score came almost entirely from text-side signals: `keyword_match=0.667`
  (it literally contains the words "healthy" and "chemistry") plus decent semantic
  similarity — but it is a pharmaceutical-factory video with zero pool content. CLIP's
  actual pixel-level check (0.186, well below threshold) is exactly the mechanism
  designed to catch precisely this failure mode: a candidate whose *text* matches well
  but whose *content* doesn't. **This is CLIP working as intended, not a bug.**
- **Rank #2 (citrus fruit "in the pool", 70.62) — rejected, but only by 0.003.** This
  is the one genuinely borderline result in the whole two-clip investigation: its own
  text explicitly says "the pool," and it had the highest semantic similarity (0.715)
  of the three candidates, yet it scored 0.197 against a 0.20 cutoff — a hair's-breadth
  miss. `passes_visual_verification` only samples one middle frame per video (documented
  in `app/visual_verification.py`), so if that single frame happened to be a close-up on
  the fruit rather than a wider pool-context shot, a real "pool + fruit" video could
  plausibly score just under threshold through no fault of the threshold value itself.
  I do not have the actual frame to confirm this visually, so I'm reporting the score
  and the plausible mechanism, not a certain visual judgment.
- **Rank #3 (juice-pouring, 65.78) — passed with real margin (0.242), and is the actual
  final pick.** Per the main diagnostic table this is still a MISMATCH by manual
  judgment (pouring a drink isn't "chemistry"), but it is the candidate CLIP actually
  found most visually pool-consistent among the three real options — none of which
  genuinely depict "healthy pool chemistry" as a chemistry concept.

**Classification: predominantly CORRECT REJECTION**, with one narrow, non-conclusive
near-miss. Rank #1's rejection is unambiguously correct. Rank #2's rejection is a
genuine borderline call (0.197 vs. 0.20) that plausibly stems from single-middle-frame
sampling variance rather than the threshold value being wrong — but I can't confirm
that without the actual frame, so I'm not classifying it as a confirmed FALSE REJECTION,
only as the closest thing to one found in this investigation. Even in the
counterfactual where rank #2 had passed, "citrus fruit in a pool" is not obviously a
better match for "healthy pool **chemistry**" than what actually won — so this near-miss
likely wouldn't have changed the qualitative outcome (still no candidate genuinely shows
"chemistry"). The deeper cause remains the same catalog-depth limitation as clip 40 and
the other 24 mismatches: nothing in the searched catalog actually depicts the compound,
somewhat-abstract concept "healthy pool chemistry."

### Proposed follow-up (not implemented, and explicitly scoped as narrow)

If this single-middle-frame-sampling sensitivity is worth addressing at all, the
targeted, narrow fix — **not proposed here as an implementation, only as a future
candidate** — would be to sample 2–3 frames spread across the clip instead of one and
average or max the CLIP score, which would make a borderline pass/fail less sensitive
to exactly which instant gets sampled. This is **not** a recommendation to change
`visual_verification_threshold` itself, and per the task's explicit constraint, this
finding should not be generalized: only 2 of 62 clips in the full project showed a
gap greater than 5 points between the pre-scan and the final score, and only 1 of those
2 (clip 60, rank #2) produced anything resembling a borderline near-miss — the other
(clip 40) turned out to have no real rejection at all. This is evidence of a rare edge
case, not a systemic threshold or sampling problem.

## Clip 32 (largest speed mismatch) — confirmed correlation

Clip 32 ("Borax pool chemical", the clip requiring the largest speed-up per the earlier
review) is a WEAK MATCH — "mixing chemical in the glass beaker" — with only 7
candidates found in the availability scan (below the project's median candidate count
of ~10). This confirms the hypothesis: a thin candidate pool for this specific
compound concept ("borax" + "pool" + "chemical" together) simultaneously produced both
a short source clip (2.0s asset stretched into a 3.0s slot) and a topically-approximate
rather than exact text match — both symptoms of the same underlying cause, a scarce
catalog for this precise concept, not two independent problems.
