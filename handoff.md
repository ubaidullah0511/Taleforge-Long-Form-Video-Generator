# Handoff

## Current Status

This is a B-roll retrieval / documentary-video assembly service (FastAPI, Python). It has **two independent pipelines** sharing the same embedding/vector-store/stock-provider infrastructure: a simple one-clip-per-scene retrieval endpoint (`/process-script`), and a full script-to-rendered-video documentary pipeline (`/generate-documentary-timeline`) that assembles a final `.mp4` with exact-duration clips, a Remotion-rendered caption/transition pass, and (when narration audio is supplied) real audio muxing. The documentary pipeline is the actively-developed half of this codebase, and now includes a browser-based **post-generation editor** (`/editor/{project_name}`) for fast per-clip swap/regenerate/upload without re-running the whole pipeline.

**Things that changed since the last handoff and are easy to assume are still true but aren't:**
- **There is now a real editor UI, not just a results page.** `templates/editor.html` (`GET /editor/{project_name}`) — timeline scrubbing, per-clip preview, and three edit actions per clip (swap to a recorded alternate, regenerate via AI, upload a replacement file), each a fast Tier-1 re-render (`documentary_pipeline.rerender_single_clip`) that only touches that one clip's segment + re-concats, not a full re-run. See "Pipeline Overview — C" below.
- **The documentary pipeline now has two run modes, not one.** "Generate & Edit" (Mode A, `POST /generate-documentary-timeline/edit(/upload)` → `documentary_pipeline.generate_and_edit`) resolves every clip's asset and renders per-clip segments, then stops — no final concat, no Remotion, no audio mux — and hands off straight to the editor. "Generate Full Video" (Mode B) is either the original one-shot `run()` (`POST /generate-documentary-timeline(/upload)`) or, from inside the editor, `POST /project/{project_name}/generate-full-video` (`generate_full_video()`), which re-runs the expensive tail (concat → Remotion → audio mux) over whatever the editor currently has, including any Tier-1 edits.
- **An AI-image generation fallback now exists** — `app/stock/openai_image.py` (OpenAI image generation, model configurable via `ai_generation_model`, default `gpt-image-2`). When no real candidate scores above `settings.ai_generation_trigger_threshold` (default 75.0), `documentary_pipeline._try_ai_generation_fallback` generates a photorealistic still instead of falling straight to a black placeholder. **This directly contradicts the previous handoff's "Known Gaps — Still Open" entry for this** — that gap is now closed, see below. Real black-frame placeholders still happen when generation itself fails or is disabled.
- **A curated local-footage library provider exists** — `app/stock/local_library.py`, backed by `clips/local/<niche>/index.json` (built/maintained by `app/stock/local_library_index.py`, a vision-LLM captioning + reconciliation CLI). Only niches with `NicheConfig.local_library_path` set use it (`bodybuilding`, `prison` today; `trucks`/`pool_maintenance` don't). It's checked first in provider order, and a strong-enough local match (`>= ai_generation_trigger_threshold`) short-circuits the rest of the stock-provider walk for that clip.
- **Four content niches now, not two** — `trucks`, `pool_maintenance`, `bodybuilding`, `prison` (`app/niches.py`). All four currently set `use_archive_org=False`.
- **Coverr is a stock video provider now** (`app/stock/coverr.py`, `COVERR_API_KEY`) — part of the default provider rotation alongside Pexels/Pixabay video+image, Internet Archive, and NASA. Wikimedia (`app/stock/wikimedia.py`) is implemented but **deliberately excluded** from both provider-order lists — it consistently failed to connect (ConnectTimeout) in this environment, burning ~10-15s/clip for nothing; the file is untouched in case it's worth revisiting elsewhere.
- **Per-request footage-source toggles exist** — `enable_ai_generation` / `enable_local_library` / `enable_stock_providers` (all default `True`), threaded from 3 checkboxes on the main page's UI through every entry point down to `search_clip`/`_should_try_ai_generation`. Unchecking all 3 is blocked client-side (would make every clip a placeholder).
- **LLM backend is OpenAI now, not Groq** (and not Gemini before that) — `app/llm_client.py` uses the `openai` SDK, `.env` uses `OPENAI_API_KEY`, models are `gpt-4o-mini` for both text and vision (`llm_text_model`/`llm_vision_model` in `app/config.py`).
- **Narration timing is real when audio is supplied, not always a word-count estimate.** If `audio_path` is given, local Whisper (`openai-whisper`, offline, no API key — `app/transcription.py::get_narration_timing`) produces real per-word timestamps that drive both per-clip duration (`app/documentary_table.py`) and captions. The word-count pacing estimate (`assign_timings`, `2.5 words/sec`) is now the **fallback only for when no audio is provided**, not the only path.
- **A real visual-content check exists now** (`app/visual_verification.py`) — CLIP (ViT-B/32) scores a sampled frame from each downloaded candidate against a short visual-query text, on top of the pre-existing metadata-only scoring heuristic. GPU is used automatically when available (`torch.cuda.is_available()`); see "GPU / CLIP / Whisper" below.
- **Content-niche keyword locking exists** (`app/niches.py`) — each niche has `banned_terms`/`positive_terms`/`system_context` used to keep LLM-generated keywords and stock-candidate metadata on-topic. `content_niche` is a required-ish param through the whole documentary call chain now.
- **Archive.org (`internet_archive.py`) now filters by license** — only items with a recognized public-domain/CC-BY `licenseurl` are returned; anything missing/ambiguous/restrictive is excluded (Archive.org hosts both genuinely PD material and restricted third-party uploads that look identical in search results).
- **A pre-render footage-availability scan exists** (`check_footage_availability` in `documentary_pipeline.py`) — runs before any download/CLIP-verify/render work, flags "thin" clips, writes `footage_availability_report.json` per project, and is also exposed standalone via `POST /generate-documentary-timeline/preview` (table + availability only, no render, near-instant).
- **Cross-clip asset reuse is now a bounded cap (default 2), not all-or-nothing** — `settings.max_asset_repeat_count`; previously an asset could either never repeat, or (via `allow_duplicates=True`) repeat with no limit at all.

## Pipeline Overview

### A. Simple pipeline — `POST /process-script`

Unchanged from before: scene segmentation → scene understanding → local ChromaDB match → stock fallback (4 providers) → candidate description/scoring → download/ingest → `list[SceneResult]`, one entry per scene. Still **no assembly stage** — never joins clips into a video. See `app/pipeline.py`.

### B. Documentary pipeline — `POST /generate-documentary-timeline`(`/edit`)

1. **Input** — script text or `script_path`; optional `audio_path` (manually-supplied narration recording, no TTS anywhere in this repo); optional `enable_ai_generation`/`enable_local_library`/`enable_stock_providers` toggles (all default `True`).
2. **Narration timing** (only if `audio_path` given) — `app/transcription.py::get_narration_timing` runs local Whisper (`whisper_model_size`, default `"base"`) for real per-word timestamps; `check_alignment` is a non-blocking sanity check comparing them against the supplied script text.
3. **Table generation** — `app/documentary_table.py::generate_table` — LLM splits the script into beats (`canva_keyword`, `fallback_keyword`, `visual_type`, `word_count`, niche-filtered), using real Whisper timing if available, else the word-count pacing estimate.
4. **Pre-render footage availability scan** — `app/documentary_pipeline.py::check_footage_availability` — walks the whole table with the real search/scoring/niche-filter code (semantic-keyword LLM fallback deliberately skipped for cost), simulating cross-clip dedup, **before any download or CLIP verification happens**. Writes `footage_availability_report.json`; logs a warning if too many clips look thin. Also reachable standalone via `POST /generate-documentary-timeline/preview`.
5. **Per-clip resolve** (`_resolve_clip`) — for each clip:
   - **Search** (`search_clip`) — walks up to 8 providers in niche-dependent order (`local_library` first when the niche has one configured and `enable_local_library`, then `pexels`/`pixabay` video+image, `coverr`, `internet_archive`, `nasa`; `internet_archive` skipped for niches with `NicheConfig.use_archive_org=False`, currently all four defined niches; the whole stock tier skipped if `enable_stock_providers=False`), trying `canva_keyword` → `fallback_keyword` → LLM semantic-keyword fallback, respecting the bounded repeat cap (`used_assets`, `settings.max_asset_repeat_count`). A strong local-library match (score `>= ai_generation_trigger_threshold`) short-circuits the rest of the walk for that clip.
   - **Scoring** — `app/scoring.py::score_asset`, metadata-only heuristic (semantic/historical/quality/cinematic/motion), plus content-niche filtering (`candidate_violates_niche`) against provider text metadata.
   - **Download + normalize** — best-scoring candidate downloaded in full; non-16:9 sources are cropped or blur-padded to 1920x1080 rather than rejected; too-low-res falls through to the next-best candidate.
   - **CLIP visual verification** — `passes_visual_verification(asset.path, clip.canva_keyword)` — scores the actual downloaded frame against `canva_keyword` **alone** (see "Known Gaps — Resolved" below for why not `canva_keyword + script_beat`). Below `settings.visual_verification_threshold` (0.20) falls through to the next-best candidate.
   - **AI-generation fallback** (`enable_ai_generation`, default `True`) — if no real candidate scores `>= ai_generation_trigger_threshold` (or none was found at all), `_try_ai_generation_fallback` generates a photorealistic image via OpenAI (`app/stock/openai_image.py`) instead. Only after real search AND generation both come up empty/rejected does a clip become a black-frame placeholder.
6. **Per-clip segment render** — `app/documentary_assembly.py::render_all_segments` — validates table contiguity, renders every row to an exact-duration silent 1080p/30fps segment (trim/loop/speed-match/blur-pad/black-for-placeholder). This alone is what Mode A ("Generate & Edit") runs before handing off to the editor — no concat yet.
7. **Final concat** — `assemble_video` (Mode B / `run()` / editor's "Generate Full Video" only) — calls `render_all_segments` above, then concats + re-encodes into `final_video.mp4` and validates the result's total duration against the **sum of each row's own duration** (not an absolute end-timestamp — see "Known Gaps — Resolved" below for why that distinction matters) and resolution.
8. **Style decision** — `app/style_decision.py::decide_style` — LLM picks pacing/transition-style/caption-emphasis words from the script.
9. **Remotion render** — `app/remotion_render.py::render_final_video` — composites the assembled segments with transitions and animated word-by-word captions (`remotion/`, `node render.mjs`) → `final_video_remotion.mp4`. This is silent; `final_video.mp4` (the ffmpeg-assembled, pre-Remotion artifact) is also kept.
10. **Audio mux** (only if `audio_path` was supplied) — `app/audio_mux.py::mux_narration_audio` — muxes the full, unmodified uploaded narration into the Remotion output → `final_video_with_narration.mp4`, the actual deliverable when real narration was provided. Freezes the last frame to cover any narration tail longer than the video.
11. **Output** — `projects/<project_name>/`: `final_video.mp4`, `final_video_remotion.mp4`, `final_video_with_narration.mp4` (if audio given), `captions.srt`, `footage_availability_report.json`, `timeline.json`, `timeline_table.md`, `project_meta.json` (script/content_niche/audio_path/whisper_words/table — lets a later editor-triggered re-render skip re-running table generation or transcription).

### C. Editor — `GET /editor/{project_name}`

Loads a project's persisted `timeline.json`/`project_meta.json` (`GET /project/{project_name}`) into a timeline-scrubbing UI (`templates/editor.html`). Every clip — including placeholders — is selectable and shows an explicit "no footage yet" state (not a stale/blank frame) when it has no rendered segment. Three per-clip edit actions, all going through `PATCH /project/{project_name}/clip/{clip_number}(/upload)` → `documentary_pipeline.rerender_single_clip`:
- **Swap to alternate** — materializes one of the clip's already-recorded (but not downloaded) search candidates (`TimelineEntry.alternates`, top 5 by score).
- **Regenerate via AI** — always attempts AI generation regardless of score/threshold (an explicit user override — bypasses the `enable_ai_generation` run-level toggle and `_should_try_ai_generation`'s gating by design).
- **Upload custom file** — copies a user-supplied file in directly.

Each only re-renders that one clip's segment + re-concats `final_video.mp4` from already-rendered segments (`rerender_single_clip_segment`) — never re-runs search/download/CLIP for any other clip. "Generate Full Video" (`POST /project/{project_name}/generate-full-video`) is the separate, expensive Tier-2 action that runs the real concat → Remotion → audio-mux tail over whatever the editor currently has.

The main page (`templates/index.html`) has a "Run mode" selector for Mode A vs Mode B, 3 footage-source checkboxes, and a niche dropdown, all feeding the same `/generate-documentary-timeline*` endpoints.

## GPU / CLIP / Whisper

- torch is CUDA-enabled (`2.13.0+cu126`) in this venv; `requirements.txt` now pins `torch==2.13.0+cu126` with `--extra-index-url https://download.pytorch.org/whl/cu126` at the top — **without this pin, a fresh `pip install -r requirements.txt` on Windows silently falls back to a CPU-only torch wheel** (this happened once already; root cause was that torch was never pinned, only pulled in transitively by whisper/CLIP).
- Both CLIP (`app/visual_verification.py`) and Whisper (`app/transcription.py`) auto-select `cuda` via `torch.cuda.is_available()` — no manual device wiring needed.
- Real benchmark on this machine (GTX 960 vs Ryzen 5 3600): CLIP inference is ~3.6x faster on GPU (~37ms vs ~135ms per candidate) — matters a lot since it runs once per downloaded candidate. Whisper `base` on short clips was roughly a wash or slightly *slower* on this particular (old, no-tensor-cores) GPU than on the CPU — GPU wins aren't guaranteed on older cards for small workloads.

## Known Gaps — Resolved

- ~~No real visual-content matching, metadata-only scoring~~ — CLIP visual verification now exists (`app/visual_verification.py`).
- ~~No real narration/TTS timing~~ — partially resolved: real Whisper STT timing exists when audio is supplied; there is still no TTS *generation* stage (user must supply their own narration recording).
- ~~CLIP visual verification wrong-text bug (2026-07-23)~~ — `passes_visual_verification` was being called with `f"{canva_keyword} {script_beat}"` — the **full narration sentence**, not a short visual description. Confirmed via a real pipeline run (DEBUG logging on `app.visual_verification`) that this diluted/truncated text (19-39+ words, well past CLIP's 77-token budget) was dragging genuinely correct candidates below threshold. **Fix:** pass `clip.canva_keyword` alone. **Also recalibrated** `visual_verification_threshold` from 0.26 → 0.20 based on a real controlled A/B. See `app/config.py`'s inline comment for the full data.
- ~~Archive.org licensing risk~~ — `internet_archive.py` now filters to a `licenseurl` allowlist (public-domain / CC-BY prefixes); missing/ambiguous/restrictive licenses are excluded before a file listing is even fetched.
- ~~Thin-candidate-pool only discoverable mid-run via logs~~ — `check_footage_availability` + `/generate-documentary-timeline/preview` now surface this before any expensive work.
- ~~All-or-nothing cross-clip asset dedup~~ — now a bounded repeat cap (`settings.max_asset_repeat_count`, default 2).
- ~~No generative fallback for exhausted candidate pools~~ — **this was listed as still-open in a previous handoff; it's since been built.** `app/stock/openai_image.py` + `documentary_pipeline._try_ai_generation_fallback` generate a photorealistic still when real search comes up short (see Pipeline Overview step 5 above). Black-frame placeholders now only happen when generation itself fails/is disabled and real search also came up empty.
- ~~Local library index/disk desync breaking the editor (recurring)~~ — `index.json` entries could reference files that had since been deleted (e.g. by the near-duplicate cleanup pass), which surfaced as an opaque "ffmpeg trim failed" 500 when the editor tried to materialize a stale alternate. **Root-cause fix:** `app/stock/local_library.py::search()` now skips any index entry whose file doesn't exist on disk, so no caller (primary search or editor alternates) can ever receive a dead reference going forward. **Defense-in-depth:** `_materialize_alternate_asset` explicitly checks for this and raises a clear `ValueError` ("this alternate is no longer available...") for alternates already persisted to old `timeline.json` files before the fix. **Reconciliation tool:** `python -m app.stock.local_library_index --niche <name> --verify [--dry-run]` audits/cleans an index against disk; `--niche` omitted scans every niche. Found and cleaned 181 stale entries in the real `prison` library during this investigation.
- ~~Mode A ("Generate & Edit") could leave every clip with no rendered segment, breaking the editor~~ — `generate_and_edit()` used to call the full `assemble_video()` (per-clip render + final concat/duration/resolution validation, all-or-nothing). Any failure in the concat-only portion discarded every row's already-rendered segment, not just the failing step's own output — the editor's very next single-clip edit then failed with "clip N has no existing rendered segment on disk." **Fix:** split `assemble_video` into `render_all_segments` (per-clip only, no concat) + the concat/validate tail; Mode A now calls only the former. Also removes a redundant `final_video.mp4` concat pass Mode A used to do and "Generate Full Video" immediately re-did from scratch anyway (~40% faster in a real timing test: 4.83s vs 8.10s for 8 real clips).
- ~~"Generate Full Video" silently producing a video with no narration audio~~ — the audio path was never actually lost (confirmed via real project inspection — `project_meta.json.audio_path` was always correctly persisted). The real bug: `assemble_video`'s final-duration check compared the concatenated video's real length against `parse_timestamp(last_row.end)`, an **absolute** clock timestamp — but real Whisper-timed narration commonly has the first clip start a moment after `00:00:00` (leading silence before the first detected word), so the two values differed by exactly that offset on essentially every real project, spuriously raising `AssemblyValidationError` *after* `final_video.mp4` was already written to disk. That exception discarded `final_video_path`, which gated off Remotion render and `mux_narration_audio` entirely — the user got the silent rough-cut with no mux error at all, because mux was never attempted. **Fix:** compare against the sum of each row's own `duration` instead (matches what the concat step already targets). Verified live end-to-end: real narration audio (AAC, correct duration, non-silent per `ffmpeg -af volumedetect`) now present in `final_video_with_narration.mp4`.
- ~~Editor download link pointed at the wrong/least-useful final-video variant~~ — the editor showed all three rendered variants as plain equal-weight text links with no clear primary action and no real `download` attribute. **Fix:** one styled primary download button (glassmorphism theme, `download` attribute) picks the best available real deliverable in priority order — `final_video_with_narration.mp4` → `final_video_remotion.mp4` → `final_video.mp4` (clearly labeled "Rough-Cut" if that's all that exists) — with the non-primary variants kept as secondary links.
- ~~Placeholder clips reportedly not fully editable~~ — investigated and found this was **not actually broken** at the data/backend level (`assemble_video`/`render_all_segments` already render a real black-frame segment for placeholders, and `rerender_single_clip` never special-cased `placeholder`) — confirmed with a live probe and a permanent regression test. What genuinely needed fixing was editor UX clarity: the big preview now explicitly shows a "no footage yet" state (instead of leaving a stale previous-clip frame on screen) when a selected clip has no rendered segment.

## Known Gaps — Still Open

- **No TTS generation** — narration audio must be supplied by the user; the pipeline only transcribes it (Whisper), never generates it.
- **No vertical/9:16 output** — 1920x1080 only, hardcoded in both `documentary_assembly.py` and `remotion_render.py`. Out of scope unless Shorts/vertical becomes an actual content-strategy decision.
- **No color-space (Rec. 709) normalization across clips from different providers** — `-pix_fmt yuv420p` is set but no explicit `-colorspace`/`-color_primaries` tagging or conversion filter exists. Low priority today; matters more as Archive.org (older film-scan color characteristics) gets used more.
- **No silence/pause (VAD) handling in narration** — `documentary_table.py::_force_contiguity` explicitly *removes* any real narration gap rather than detecting/using it. No evidence this is causing a visible problem in practice.
- **Documentary asset search is sequential, not concurrent, by design** — `used_assets` dedup counter and the download-limit counter are shared mutable state across beats (`app/documentary_pipeline.py::run`, ponytail comment).
- **`local_library_index.py --verify` is manual, not scheduled** — nothing currently re-runs it automatically after a cleanup pass deletes local library files; the root-cause fix in `local_library.py::search()` prevents dead references from ever being *served* again, but stale `index.json` entries themselves only get pruned when someone runs `--verify` by hand.
- **The editor's "Regenerate via AI" button intentionally bypasses the run-level `enable_ai_generation` toggle** — it's a manual, explicit per-clip override the user clicked, so it always attempts generation regardless of that checkbox's state. Worth a conscious re-decision if that ever surprises anyone.
- **In-memory progress state (`app/progress.py`) is lost on restart and not shared across workers** — fine for the current single-process local-tool usage; would need Redis/a DB before running with `uvicorn --workers > 1`.
- **Groq free-tier rate limit note is now stale** — the actual throttle (`app/llm_client.py::_throttle`) is still a 30 RPM sliding-window limiter, but it's throttling OpenAI calls now, not Groq's. The 30/min figure is a conservative starting default (`llm_max_requests_per_minute`), not an OpenAI-imposed hard limit — raise it if your account tier allows more.
- **`SceneResult.start`/`.end` (simple pipeline only) are source-clip in/out points, not timeline seconds** — different concept from `TimelineEntry.start`/`.end` in the documentary pipeline (genuine HH:MM:SS timeline positions). Don't confuse the two.
- **`select_best_asset(target_duration=...)` is dead/inert** in the simple pipeline (`app/asset_selection.py`) — `app/pipeline.py` never passes it.
- **`/clips/ingest` only supports video**, not images.

## How to Run It

```
pip install -r requirements.txt
```
This now installs a CUDA-enabled torch build (`--extra-index-url` at the top of `requirements.txt`) — on a machine with no NVIDIA GPU/driver, this install will fail; drop the pinned `torch` lines and let pip resolve a CPU wheel instead in that case.

Install [ffmpeg](https://ffmpeg.org/) and [Node.js](https://nodejs.org/) (for the Remotion render step, `remotion/`) and ensure `ffmpeg`/`ffprobe` are on `PATH`.

`.env`:
```
OPENAI_API_KEY=...
PEXELS_API_KEY=...
PIXABAY_API_KEY=...
COVERR_API_KEY=...
SIMILARITY_THRESHOLD=0.75
CHROMA_DIR=./data/chroma
CLIPS_DIR=./clips
```
No API key is needed for Archive.org or NASA search — free/unauthenticated public APIs. Wikimedia (`app/stock/wikimedia.py`) also needs no key but isn't in the active provider rotation (see "Current Status" above).

Start the server:
```
uvicorn app.main:app --reload
```

Run it:
- Simple pipeline: `POST /process-script` with `{"script": "..."}` (or `{"raw_data": "..."}`)
- Documentary pipeline, full one-shot render: `POST /generate-documentary-timeline` with `{"script": "...", "project_name": "optional", "content_niche": "trucks|pool_maintenance|bodybuilding|prison", "allow_duplicate_assets": false, "max_downloads": null, "audio_path": "optional path to a narration recording", "enable_ai_generation": true, "enable_local_library": true, "enable_stock_providers": true}`
- Documentary pipeline, "Generate & Edit" (Mode A — resolves assets + per-clip segments, then stops): `POST /generate-documentary-timeline/edit`, same body shape
- File-upload variants (multipart, script file + audio file): `POST /generate-documentary-timeline/upload`, `POST /generate-documentary-timeline/edit/upload`
- Cheap pre-render check: `POST /generate-documentary-timeline/preview` — same body, returns the planned table + footage-availability report only, no render, JSON-only (no file-upload variant yet)
- `GET /progress/{project_name}` — poll a running documentary job
- `GET /editor/{project_name}` — the post-generation editor UI
- `GET /project/{project_name}` — loads a project's timeline for the editor
- `PATCH /project/{project_name}/clip/{clip_number}` (JSON: `{"mode": "alternate"|"ai", "alternate_index": ...}`) / `PATCH /project/{project_name}/clip/{clip_number}/upload` (multipart file) — Tier-1 single-clip edit
- `POST /project/{project_name}/generate-full-video` — Tier-2 full render (concat → Remotion → audio mux) over the current, possibly-edited timeline
- Local library maintenance: `python -m app.stock.local_library_index --niche <name>` (index new files) or `--verify [--dry-run]` (reconcile against disk; omit `--niche` to scan every niche)
- `GET /health`, `GET /clips/search?query=...`, `GET /` (browser test UI, `templates/index.html`)

Tests (mocked/offline except `test_documentary_assembly.py`, which shells out to real ffmpeg with synthetic `lavfi` sources, and anything explicitly run live against real APIs during investigation — not part of the normal suite):
```
python -m pytest tests/ -q
```
296 tests as of this session, all passing.

## External Dependencies

| Dependency | Used for | Config |
|---|---|---|
| OpenAI API (`openai` SDK) | All LLM calls: scene segmentation, scene understanding, media-type decision, candidate description/scoring input, documentary table generation, semantic keyword fallback, style decision, local-library captioning, AI-image generation fallback | `OPENAI_API_KEY`; models `llm_text_model`/`llm_vision_model` (both `gpt-4o-mini`), `ai_generation_model` (`gpt-image-2`) in `app/config.py`; client-side throttled to `llm_max_requests_per_minute` (30, conservative default) |
| Local Whisper (`openai-whisper`, offline) | Real per-word narration timestamps when `audio_path` is supplied — drives per-clip duration and captions | `whisper_model_size` (default `"base"`) in `app/config.py`; no API key, runs on CPU or GPU automatically |
| CLIP (`ViT-B/32`, via `openai/CLIP` git dep, offline) | Post-download visual verification — scores a sampled frame against `canva_keyword` | `enable_visual_verification`, `visual_verification_threshold` (0.20) in `app/config.py`; runs on CPU or GPU automatically |
| Pexels (video + photo APIs) | Stock video/image search | `PEXELS_API_KEY` — `app/stock/pexels.py`, `app/stock/pexels_images.py` |
| Pixabay (video API) | Stock video search (query truncated to 100 chars) | `PIXABAY_API_KEY` — `app/stock/pixabay.py`, `app/stock/pixabay_images.py` |
| Coverr | Stock video search | `COVERR_API_KEY` — `app/stock/coverr.py` |
| Internet Archive (`archive.org`) | Historical/archival video+image search, license-filtered (PD/CC-BY only), no key needed | `app/stock/internet_archive.py` |
| Wikimedia Commons | Archival images only, no key needed — implemented but not in the active provider rotation (see "Current Status") | `app/stock/wikimedia.py` |
| NASA Images API | Space/science imagery, no key needed | `app/stock/nasa.py` |
| ffmpeg / ffprobe (OS binaries) | Frame sampling, trimming, looping, concat, subtitle burn-in, duration probing, CLIP frame extraction | Must be on system `PATH` |
| Node.js + Remotion (`remotion/`) | Final composited render with transitions + animated captions | Invoked via `node render.mjs` from `app/remotion_render.py` |
| ChromaDB (`PersistentClient`) | Local vector store for clip embeddings + metadata/dedup lookups | `CHROMA_DIR`; telemetry disabled in `app/vector_store.py` |
| sentence-transformers (`BAAI/bge-small-en-v1.5`) | Text embeddings for semantic search | `embedding_model` in `app/config.py`, lazy-loaded/cached in `app/embeddings.py` |

## File Reference

| File | Stage | Purpose |
|---|---|---|
| `app/main.py` | API | FastAPI routes: `/health`, `/`, `/process-script`, `/generate-documentary-timeline` (+ `/edit`, `/preview`, `/upload`, `/edit/upload`), `/progress/{project_name}`, `/editor/{project_name}`, `/project/{project_name}` (+ `/clip/{clip_number}(/upload)`, `/generate-full-video`), `/clips/ingest`, `/clips/search` |
| `app/config.py` | Config | `.env`-backed `Settings`; creates data dirs on import |
| `app/models.py` | Shared | `Scene`, `SceneAnalysis`, `ClipRecord`, `SceneResult`, `TimelineClip`, `AssetInfo`, `AlternateCandidate`, `TimelineEntry`, `DocumentaryResult`, `ClipAvailability`, `AvailabilityReport`, `DocumentaryPreview`, `ProjectMeta` |
| `app/llm_client.py` | Shared | OpenAI client, rate limiter, `generate_json`, `generate_json_from_images`, `generate_json_from_video` (local frame sampling) |
| `app/transcription.py` | Shared | Local Whisper narration transcription (`get_narration_timing`), script/transcript alignment sanity check |
| `app/visual_verification.py` | Shared | CLIP-based post-download visual verification (`passes_visual_verification`, `visual_match_score`) |
| `app/embeddings.py` | Shared | `embed()` — sentence-transformers wrapper, `lru_cache`d model load |
| `app/vector_store.py` | Shared | ChromaDB wrapper — upsert, semantic query, dedup lookups |
| `app/downloader.py` | Shared | `download()`, `download_video_trimmed()` |
| `app/niches.py` | Shared | Content-niche config (`trucks`, `pool_maintenance`, `bodybuilding`, `prison`) — banned/positive terms, `system_context`, `use_archive_org`, `local_library_path` |
| `app/progress.py` | Shared | In-memory per-project progress tracking (contextvars-based), polled via `/progress/{project_name}` |
| `app/stock/base.py` | Shared | `StockHit` model, `StockProvider` protocol |
| `app/stock/pexels.py`, `pixabay.py`, `pexels_images.py`, `pixabay_images.py`, `coverr.py`, `internet_archive.py`, `wikimedia.py`, `nasa.py` | Shared | One `search(query, per_page) -> list[StockHit]` per provider |
| `app/stock/local_library.py` | Shared | Curated local-footage provider — ranks `clips/local/<niche>/index.json` by caption overlap, skips entries whose file no longer exists on disk |
| `app/stock/local_library_index.py` | Shared | CLI: index new local clips (vision-LLM captioning + slug/hash dedup) or `--verify` an existing index against disk |
| `app/stock/openai_image.py` | Shared | AI-image generation fallback (`generate_fallback_image_openai`) |
| `app/scene_segmentation.py`, `scene_understanding.py`, `asset_selection.py`, `clip_ingest.py`, `pipeline.py` | Simple pipeline | Scene segmentation/understanding, candidate ranking, ffprobe/ingest, orchestration |
| `app/documentary_table.py` | Documentary pipeline | Script → `TimelineClip` table; Whisper-timed or word-count-paced |
| `app/scoring.py` | Documentary pipeline | `score_asset()` weighted heuristic scorer |
| `app/documentary_pipeline.py` | Documentary pipeline | Provider walk, scoring, download, CLIP verification, AI-generation fallback, bounded dedup, `check_footage_availability`, `plan_documentary`, `run()`, `generate_and_edit()` (Mode A), `rerender_single_clip()` (editor Tier-1 edit), `generate_full_video()` (editor Tier-2 render) |
| `app/documentary_assembly.py` | Documentary pipeline | Contiguity validation, per-row render (trim/loop/speed-match/blur-pad/black) — `render_all_segments()` (per-clip only, used by Mode A) and `assemble_video()` (adds concat + duration/resolution validation, used by Mode B/full render) |
| `app/style_decision.py` | Documentary pipeline | LLM-driven pacing/transition/caption-emphasis decision |
| `app/remotion_render.py` | Documentary pipeline | Invokes the Remotion project for the final composited render |
| `app/audio_mux.py` | Documentary pipeline | Muxes uploaded narration audio into the final render |
| `app/subtitles.py` | Documentary pipeline | Cue building, SRT render |
| `app/timecode.py` | Documentary pipeline | `"HH:MM:SS"` ⟷ seconds helpers |
| `templates/index.html` | Manual testing | Browser UI for kicking off a run — script/audio input, niche + run-mode + source-toggle selectors, Preview button |
| `templates/editor.html` | Manual testing | Post-generation editor UI — timeline scrubbing, per-clip swap/regenerate/upload, "Generate Full Video" |
| `templates/theme.css` | Manual testing | Shared glassmorphism theme for both pages |

## Next Steps / Recommended Priorities

1. **Monitor the recalibrated CLIP threshold (0.20) in real production runs** — it's based on a real but modest sample. Watch for either obviously-wrong candidates now passing, or still-correct candidates still failing, and adjust with more data rather than re-guessing.
2. **Consider scheduling `local_library_index.py --verify`** (e.g. a periodic task or a pre-run check) instead of relying on someone remembering to run it by hand after a cleanup pass — the root-cause fix in `search()` means a stale entry can no longer be *served*, but the index itself only gets cleaned on demand today.
3. **No TTS generation is in scope yet** — if/when it is, only `documentary_table.py` (where `start`/`end` come from) and `subtitles.py::build_cues` need to change on the input side; everything downstream already consumes whatever timing it's given.
4. **Reconsider `_resolve_clip`'s sequential loop** only if project sizes grow large enough that provider-search latency becomes the bottleneck.
5. **Clarify or rename `SceneResult.start`/`.end`** in the simple pipeline so it's not confused with `TimelineEntry.start`/`.end` in the documentary pipeline.
6. **`index.html`'s Mode B result page (`renderTimelineResults`) still shows the 3 final-video variants as flat unprioritized links**, same shape the editor's `renderFinalLinks()` had before this session's fix — worth applying the same priority-order + styled-download-button treatment there if that page's results view stays in active use.
