# Handoff

## Current Status

This is a B-roll retrieval / documentary-video assembly service (FastAPI, Python). It has **two independent pipelines** sharing the same embedding/vector-store/stock-provider infrastructure: a simple one-clip-per-scene retrieval endpoint (`/process-script`), and a full script-to-rendered-video documentary pipeline (`/generate-documentary-timeline`) that actually assembles a final `.mp4` with exact-duration clips and burned-in subtitles. The documentary pipeline is the more complete/working half of this codebase — it has real duration validation and its own test suite (`test_documentary_assembly.py` runs real ffmpeg). **The single most important thing to know:** there is no TTS/narration-audio stage anywhere in this repo — all clip timing (in both pipelines) is a word-count pacing estimate (`2.5 words/sec`, clamped 3-7s per beat), not real speech timing, and every rendered video segment is silent (`-an`). Also note: the LLM backend was switched from Gemini to **Groq** at some point — `app/gemini_client.py` no longer exists, it's `app/llm_client.py` now, and `.env` uses `GROQ_API_KEY` not `GEMINI_API_KEY`.

## Pipeline Overview

### A. Simple pipeline — `POST /process-script`

1. **Scene segmentation** — `app/scene_segmentation.py::split` — LLM (Groq) splits raw script into `{scene, text}` beats — ✅ working
2. **Scene understanding** — `app/scene_understanding.py::analyze` + `decide_media_type` — LLM turns scene text into a description/keywords query, and separately decides if the scene wants video or a still image — ✅ working
3. **Local semantic match** — `app/pipeline.py::_resolve_scene` → `vector_store.query_similar` — embeds the query, checks ChromaDB for a local clip above `similarity_threshold` (0.75) — ✅ working
4. **Local candidate ranking** — `app/asset_selection.py::select_best_asset` — if local matches exist, ranks by cosine similarity, prefers the decided media type, resolution as tiebreak — ✅ working
5. **Stock fallback search** — `app/pipeline.py::_search_stock` — queries 4 providers concurrently (`pexels`, `pixabay`, `pexels_images`, `pixabay_images`), pooled to `candidate_pool_size` (8) — ✅ working
6. **Candidate description + scoring** — `app/asset_selection.py::describe_candidates` — samples frames (video) or fetches bytes (image) per candidate, LLM-describes each, embeds, then `select_best_asset` picks a winner — ✅ working (note: `target_duration` filtering exists in `select_best_asset` but is **never called with one** — see Known Gaps)
7. **Dedup check** — `get_by_source_id` / `get_by_fingerprint` (`app/vector_store.py`) — skip re-downloading an asset already indexed — ✅ working
8. **Download** — `app/downloader.py::download` — streams to `clips/downloaded/`, SHA256 fingerprint — ✅ working
9. **Ingest/index** — `app/clip_ingest.py::analyze_and_store` — ffprobe metadata + (reused or fresh) LLM analysis → `ClipRecord` → `vector_store.upsert_clip` — ✅ working
10. **Response** — returns `list[SceneResult]`, one entry per scene, each with a single `selected_clip` path — ❌ **no assembly stage** — this endpoint never joins clips into a video or touches durations against a timeline

### B. Documentary pipeline — `POST /generate-documentary-timeline`

1. **Table generation** — `app/documentary_table.py::generate_table` — one LLM call splits the script into 3-7s beats (`canva_keyword`, `fallback_keyword`, `visual_type`, `word_count`); `assign_timings` then converts `word_count` into `start`/`end` HH:MM:SS strings via the pacing formula — ✅ working
2. **Asset search** — `app/documentary_pipeline.py::search_clip` — walks up to 7 providers in niche-dependent order (historical scripts try `internet_archive`/`wikimedia`/`nasa` first), trying `canva_keyword` → `fallback_keyword` → LLM-generated semantic keywords, stopping early once `_satisfied()` (score thresholds) is met — ✅ working
3. **Scoring** — `app/scoring.py::score_asset` — weighted heuristic (semantic 40% / historical 20% / quality 15% / cinematic 15% / motion 10%) from provider metadata only, no extra API calls, no real vision-similarity model — ✅ working (heuristic, see Known Gaps)
4. **Pick + download** — `app/documentary_pipeline.py::pick_best_asset` + `_download_asset` — single highest-scoring acceptable candidate per beat; video fetched via `download_video_trimmed` capped to that beat's own duration (not a flat cap); cross-clip dedup via `used_assets` unless `allow_duplicate_assets=True` — ✅ working
5. **Assembly** — `app/documentary_assembly.py::assemble_video` — validates the table is perfectly contiguous (`row[n].end == row[n+1].start`, raises `AssemblyValidationError` naming the exact rows if not), then renders **every** row to an exact-duration 1080p/30fps/silent segment: video trimmed or looped (`ffmpeg -stream_loop -1 ... -t {duration}`), image extended via `-loop 1 -t {duration}`, missing/placeholder rows rendered as black — then concats via `ffmpeg -f concat -c copy`. Final check: assembled duration vs. last row's `end`, tolerance 0.25s, raises if it doesn't match — ✅ working, this is the only place in the codebase with real duration validation
6. **Subtitles** — `app/subtitles.py::generate_subtitled_video` — builds cues from each row's own already-validated `start`/`end` window, split proportionally by word count, renders `captions.srt`, burns onto `final_video.mp4` in one ffmpeg pass → `final_video_subtitled.mp4` — ✅ working
7. **Output** — `projects/<project_name>/final_video.mp4`, `final_video_subtitled.mp4`, `captions.srt`, `timeline.json` (each entry has `status`: `ok`/`trimmed`/`looped-to-fill`/`missing`, `rendered_clip_path`), `timeline_table.md` — ✅ working

## Known Gaps

- **No real narration/TTS audio anywhere.** Both pipelines' timing is a `word_count / 2.5 wps` pacing estimate (`app/documentary_table.py::assign_timings`, `MIN_CLIP_SECONDS=3`, `MAX_CLIP_SECONDS=7`). Every assembled/rendered video segment is silent (`-an` in all three `_render_*_segment` functions in `documentary_assembly.py`). Subtitle cue timing (`app/subtitles.py::build_cues`) reuses this same pacing estimate — it is **not** synced to real spoken audio. Swapping in real timestamps (e.g. TTS + Whisper) only needs to change where `documentary_table.py` gets `start`/`end` and where `subtitles.py::build_cues` gets its timing — nothing else downstream needs to change (per the code's own comments).
- **`SceneResult.start`/`.end` (simple pipeline only) are source-clip in/out points, not timeline seconds.** In `app/pipeline.py::_resolve_scene`, `start`/`end` come straight from `ClipRecord.start`/`.end` — the trim window *within the stock/local asset* — not where the clip plays on any timeline. This is a different (and still-live) concept from `TimelineEntry.start`/`.end` in the documentary pipeline, which genuinely are HH:MM:SS timeline positions. Don't confuse the two when reading API responses.
- **`select_best_asset(target_duration=...)` is dead/inert.** `app/asset_selection.py` accepts a `target_duration` param to filter out video candidates shorter than a needed length, but its own docstring admits "this stays inert until a caller has one to pass" — grep confirms `app/pipeline.py` never passes it. So even a locally-matched clip in the simple pipeline has no duration guarantee.
- **Documentary asset search is sequential, not concurrent**, by design (`app/documentary_pipeline.py::run`, ponytail comment) — `used_assets` dedup set and the download-limit counter are shared mutable state across beats, so beats process one at a time. Fine for a single operator; will be the bottleneck if project size grows.
- **Scoring is metadata-only heuristics**, not real visual-content matching (`app/scoring.py`, explicit ponytail comment) — semantic score comes from embedding the provider's own text/alt/title field (`hit.text`), defaulting to a flat `0.5` if the provider gave no text at all (e.g. some `internet_archive`/`nasa` hits).
- **README's "Cold War example" manual-test script does not exist as a fixture file** — grepped the whole repo, no match outside the README prose itself. Anyone following the manual verification checklist needs to supply their own sample script.
- **Groq free-tier rate limit (30 RPM)** is shared across every LLM call in the process (`app/llm_client.py::_throttle`, sliding 60s window) — segmentation, scene understanding, media-type decision, per-candidate description, table generation, and semantic-keyword fallback all draw from the same 30/min budget. A documentary script with many beats and many stock candidates per beat can throttle noticeably (the code sleeps and retries rather than failing, so it's slow, not broken).
- **`/clips/ingest` only supports video**, not images — `IngestRequest`/`main.py::ingest_clip` calls `analyze_and_store(..., source="local")` with no `media_type` override, so it always defaults to `"video"`. Ingesting a local image into the library isn't wired up through this endpoint.
- **`clip_ingest.py::_extract_segment` (real ffmpeg trim-to-start/end) is dead on the simple pipeline's main path** — `pipeline.py` always calls `analyze_and_store` with `start=0, end=best.duration`, so the trim branch (`start != 0 or end != probed_duration`) never fires there. Only a manual `/clips/ingest` call with a real sub-range would exercise it.

## How to Run It

```
pip install -r requirements.txt
```
Install [ffmpeg](https://ffmpeg.org/) and ensure `ffmpeg`/`ffprobe` are on `PATH` (both `documentary_assembly.py`, `clip_ingest.py`, `subtitles.py`, and `asset_selection.py` shell out to them directly).

`.env` (see `.env.example`):
```
GROQ_API_KEY=...
PEXELS_API_KEY=...
PIXABAY_API_KEY=...
SIMILARITY_THRESHOLD=0.75
CHROMA_DIR=./data/chroma
CLIPS_DIR=./clips
```
Drop any existing B-roll into `clips/local/` (optional — only used by the simple pipeline's local-match step).

Start the server:
```
uvicorn app.main:app --reload
```

Run it:
- Simple pipeline: `POST /process-script` with `{"script": "..."}` (or `{"raw_data": "..."}`)
- Documentary pipeline: `POST /generate-documentary-timeline` with `{"script": "...", "project_name": "optional", "allow_duplicate_assets": false, "max_downloads": null}` — no sample script fixture ships in the repo, supply your own (a few paragraphs of narration works)
- `GET /health`, `GET /clips/search?query=...`, `GET /` (browser test UI at `templates/index.html`)

Tests (no API keys/network needed, fully mocked except `test_documentary_assembly.py` which uses real ffmpeg with synthetic `lavfi` sources):
```
.venv\Scripts\python.exe -m pytest tests/ -v
python tests/test_documentary_pipeline.py   # also runs standalone, prints OK
```

## External Dependencies

| Dependency | Used for | Config |
|---|---|---|
| Groq API (`groq` SDK) | All LLM calls: scene segmentation, scene understanding, media-type decision, candidate description/scoring input, documentary table generation, semantic keyword fallback | `GROQ_API_KEY` env var; models `llm_text_model` (`llama-3.3-70b-versatile`), `llm_vision_model` (`meta-llama/llama-4-scout-17b-16e-instruct`) in `app/config.py`; rate-limited client-side to `llm_max_requests_per_minute` (30, Groq free-tier RPM) |
| Pexels (video + photo APIs) | Stock video/image search | `PEXELS_API_KEY` — `app/stock/pexels.py`, `app/stock/pexels_images.py` |
| Pixabay (video API) | Stock video search (query truncated to 100 chars — API limit) | `PIXABAY_API_KEY` — `app/stock/pixabay.py`, `app/stock/pixabay_images.py` |
| Internet Archive (`archive.org`) | Historical/archival video+image search, no key needed | `app/stock/internet_archive.py` |
| Wikimedia Commons | Archival images only (no video), no key needed | `app/stock/wikimedia.py` |
| NASA Images API | Space/science imagery, no key needed | `app/stock/nasa.py` |
| ffmpeg / ffprobe (OS binaries) | Frame sampling, trimming, looping, concat, subtitle burn-in, duration probing | Must be on system `PATH` |
| ChromaDB (`PersistentClient`) | Local vector store for clip embeddings + metadata/dedup lookups | `CHROMA_DIR` env var; telemetry explicitly disabled in `app/vector_store.py` |
| sentence-transformers (`BAAI/bge-small-en-v1.5`) | Text embeddings for semantic search | `embedding_model` in `app/config.py`, lazy-loaded/cached in `app/embeddings.py` |

## File Reference

| File | Stage | Purpose |
|---|---|---|
| `app/main.py` | API | FastAPI routes: `/health`, `/`, `/process-script`, `/generate-documentary-timeline`, `/clips/ingest`, `/clips/search` |
| `app/config.py` | Config | `.env`-backed `Settings`; creates data dirs on import |
| `app/models.py` | Shared | `Scene`, `SceneAnalysis`, `ClipRecord`, `SceneResult`, `TimelineClip`, `AssetInfo`, `TimelineEntry`, `DocumentaryResult` |
| `app/llm_client.py` | Shared | Groq client, rate limiter, `generate_json`, `generate_json_from_images`, `generate_json_from_video` (local frame sampling) |
| `app/embeddings.py` | Shared | `embed()` — sentence-transformers wrapper, `lru_cache`d model load |
| `app/vector_store.py` | Shared | ChromaDB wrapper — upsert, semantic query, dedup lookups by source-id/fingerprint |
| `app/downloader.py` | Shared | `download()` (full fetch + SHA256), `download_video_trimmed()` (network-side ffmpeg trim, with full-download fallback) |
| `app/stock/base.py` | Shared | `StockHit` model, `StockProvider` protocol |
| `app/stock/pexels.py`, `pixabay.py`, `pexels_images.py`, `pixabay_images.py`, `internet_archive.py`, `wikimedia.py`, `nasa.py` | Shared | One `search(query, per_page) -> list[StockHit]` per provider |
| `app/scene_segmentation.py` | Simple pipeline | `split()` — script → `Scene` list |
| `app/scene_understanding.py` | Simple pipeline | `analyze()`, `decide_media_type()`, `describe_candidate()` |
| `app/asset_selection.py` | Simple pipeline | Candidate description + `select_best_asset()` ranking |
| `app/clip_ingest.py` | Simple pipeline + shared | `probe()` (ffprobe), `analyze_and_store()` → `ClipRecord` |
| `app/pipeline.py` | Simple pipeline | Orchestrates scene → single-clip resolution, `run()` entry point |
| `app/documentary_table.py` | Documentary pipeline | Script → `TimelineClip` table with word-count-paced `start`/`end` |
| `app/scoring.py` | Documentary pipeline | `score_asset()` weighted heuristic scorer |
| `app/documentary_pipeline.py` | Documentary pipeline | Provider walk, scoring, download, dedup, `run()` entry point |
| `app/documentary_assembly.py` | Documentary pipeline | Contiguity validation, per-row trim/loop/image/black render, concat, final duration check |
| `app/subtitles.py` | Documentary pipeline | Cue building from row timing, SRT render, ffmpeg subtitle burn-in |
| `app/timecode.py` | Documentary pipeline | `"HH:MM:SS"` ⟷ seconds helpers |
| `templates/index.html` | Manual testing | Browser UI for both endpoints |

## Next Steps / Recommended Priorities

1. **Decide if real narration timing is in scope.** If a TTS stage is coming, wire it in before anything else — per the code's own design, only `documentary_table.py` (where `start`/`end` come from) and `subtitles.py::build_cues` (where caption timing comes from) need to change; everything downstream (assembly's contiguity/duration checks, concat) already consumes whatever `start`/`end` it's given. Doing this late means re-validating the whole assembly/subtitle chain twice.
2. **Wire `target_duration` into `select_best_asset` calls in the simple pipeline** (`app/pipeline.py`) if scene-level duration ever matters there — the parameter and filtering logic already exist and are dead code waiting for a caller.
3. **Add a real sample-script fixture** (e.g. `fixtures/sample_script.txt`) so the README's manual verification checklist is actually runnable without someone writing their own test script first.
4. **Reconsider `_resolve_clip`'s sequential loop** in `documentary_pipeline.py::run` only if project sizes grow large enough that provider-search latency (currently paid one beat at a time) becomes the bottleneck — needs a lock-protected `used_assets` set to safely parallelize.
5. **Clarify or rename `SceneResult.start`/`.end`** in the simple pipeline so it's not confused with `TimelineEntry.start`/`.end` in the documentary pipeline — same field names, completely different meanings (source in/out vs. timeline position) across the two pipelines.
