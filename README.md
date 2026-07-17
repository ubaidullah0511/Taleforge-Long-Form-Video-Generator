# B-Roll Retrieval System

Given a script, splits it into scenes, finds the best matching B-roll clip from
a local semantic library, and falls back to Pexels/Pixabay when nothing local
matches well enough — downloading, analyzing, embedding, and caching the
result for next time.

## Setup

1. `pip install -r requirements.txt`
2. Install [ffmpeg](https://ffmpeg.org/) and make sure `ffmpeg`/`ffprobe` are on `PATH`.
3. `cp .env.example .env` and fill in `GROQ_API_KEY`, `PEXELS_API_KEY`, `PIXABAY_API_KEY`.
4. Drop any existing B-roll videos in `clips/local/`.

## Run

```
uvicorn app.main:app --reload
```

- `POST /process-script` `{ "script": "..." }` (or `{ "raw_data": "..." }`) → runs the full pipeline, returns one result per scene. Either field works; `script` wins if both are set.
- `POST /generate-documentary-timeline` `{ "script": "...", "project_name": "optional", "allow_duplicate_assets": false, "max_downloads": null }` → documentary research pipeline (see below).
- `POST /clips/ingest` `{ "video_path": "clips/local/forest.mp4", "start": 12, "end": 18 }` → index an existing local clip (ffprobe metadata + LLM vision analysis + embedding).
- `GET /clips/search?query=...` → debug endpoint for raw semantic search.
- `GET /health`.
- `GET /` → a small browser UI (`templates/index.html`) for manually testing `/process-script` and `/generate-documentary-timeline` without curl/Postman.

## Documentary Visual Research Pipeline

`POST /generate-documentary-timeline` turns a narration script into a scored,
downloaded, ready-to-import shot list:

1. **Table generation** (`app/documentary_table.py`) — one LLM call splits the
   script into ~3-7s beats with section/keywords/visual type/edit note. Row
   timing then comes from real Whisper word timestamps when narration audio is
   supplied (content-matched per beat, not just word-count order), or a
   word-count/pace estimate otherwise. If the real narration runs longer than
   the generated rows, additional row(s) are generated for the leftover audio
   (same LLM keyword-generation call, not a stretched/frozen last clip). A
   final pass then enforces a hard **5s-per-clip on-screen cap**: any row
   longer than 5s is split into multiple consecutive sub-rows (each with its
   own independently-generated keyword, not a copy of its sibling's), and
   contiguity is re-validated after every pass.
2. **Asset search** (`app/documentary_pipeline.py`) — walks up to 7 providers
   (Pexels/Pixabay video & images, Internet Archive, Wikimedia Commons, NASA
   Images) in a niche-aware priority order (historical scripts try Internet
   Archive/Wikimedia/NASA first; modern/tech scripts try Pexels/Pixabay first),
   stopping early once enough acceptable-quality assets are found — a single
   very high-scoring hit can satisfy a clip immediately. A clip already used
   anywhere earlier in the same run is excluded before scoring (zero repeats
   across the whole video, not just within one row), and a keyword tier that
   returns only already-used candidates escalates to the next tier instead of
   giving up.
3. **Scoring** (`app/scoring.py`) — weighted 0-100 score (semantic 40%,
   historical accuracy 20%, visual quality 15%, cinematic value 15%, motion
   10%) computed from provider metadata, no extra API calls.
4. **Download + normalize** — the single highest-scoring acceptable candidate
   per clip, fetched to the row's own duration (not a flat cap). Non-16:9
   candidates are no longer rejected: near-16:9 (4:3, square) gets cropped,
   portrait/vertical gets blur-padded to fill the frame — only unusably
   low-resolution candidates are skipped in favor of the next-best.
5. **Assembly** (`app/documentary_assembly.py`) — the table is the single
   source of truth for timing. Every row's source clip is normalized to
   *exactly* that row's duration: if the source is within 0.7x-1.4x of the
   target length it's speed-matched via ffmpeg's `setpts` (retimed, not cut),
   otherwise it falls back to trim (if longer) / loop (if shorter); a picked
   image is rendered into a video segment of that exact length, and a row
   with no asset gets a black placeholder. Rows are validated as perfectly
   contiguous (`row[n].end == row[n+1].start`) before any rendering starts; a
   mismatch raises `AssemblyValidationError`. All row segments are normalized
   to 1080p/30fps/no-audio so concatenation is a fast stream copy.
6. **Final render** (`app/remotion_render.py`, `remotion/`) — Remotion
   composites the exact-duration segments with transitions in one pass.
   Animated word-by-word captions exist in the Remotion composition but are
   **off by default** (`include_captions: bool = False` — pass `True` to
   `render_final_video` to turn them back on); caption *word timing* is still
   always computed (real Whisper timing when audio was supplied) so nothing
   needs to be regenerated later. `app/subtitles.py::build_cues` also still
   writes a plain `captions.srt` sidecar, independent of the Remotion render.
7. **Audio mux** (`app/audio_mux.py`) — only when narration audio was
   supplied: muxes it into the (silent) Remotion output, unmodified. If the
   video is shorter than the audio (narration ran past the last row by less
   than the coverage-row threshold), the last frame is held to cover the gap
   rather than cutting audio off.
8. **Output** — `projects/<project_name>/final_video.mp4` (legacy ffmpeg
   concat, silent), `final_video_remotion.mp4` (Remotion render, silent, no
   captions by default), `final_video_with_narration.mp4` (Remotion + real
   narration audio — the actual deliverable when audio was supplied),
   `captions.srt`, `timeline.json` (each row carries `status`: `ok` /
   `trimmed` / `looped-to-fill` / `speed-matched` / `missing`, and
   `rendered_clip_path`), `timeline_table.md`, `remotion_props.json`, and
   `clips/clip_001/…` with the downloaded + rendered files. A clip that finds
   nothing usable still gets a `placeholder: true` timeline row (with a
   `note` explaining why) instead of failing the run.

**Design note — row timing is real Whisper ASR when audio is supplied,
otherwise a word-count/pace estimate.** This codebase still has no TTS
generation stage, but `documentary_pipeline.run(..., audio_path=...)` accepts
a manually-supplied voiceover recording (you record/provide the audio
yourself — see `app/transcription.py`). When given, Groq's Whisper endpoint
transcribes it and its real per-word timestamps become the source of truth
for both `documentary_table.py`'s row `start`/`end` and the Remotion
captions burned into the final video (`app/subtitles.py::build_word_timings`).
`check_alignment` compares Whisper's transcription against your script as a
sanity check only (logs a warning on significant deviation, never blocks).
When no `audio_path` is given, row timing falls back to the original
estimate — `word_count / 2.5 words-per-second`, clamped to 3-7s — with no
audio track anywhere (assembly renders every segment with `-an`), exactly as
before. One known gap: `app/subtitles.py::build_cues` (the exported
`captions.srt`, separate from Remotion's burned-in captions) still always
uses the word-count estimate, even when real Whisper timing is available.

Tunable via `.env` / `Settings` (`app/config.py`):
`documentary_high_quality_score` (default 90), `documentary_min_score`
(default 50), `documentary_niche_min_assets`, `documentary_max_downloads_per_project`.

## Folder map

```
app/
  config.py              settings (.env)
  models.py              Scene / SceneAnalysis / ClipRecord / SceneResult / TimelineClip / TimelineEntry / AssetInfo / DocumentaryResult
  llm_client.py           Groq-backed generate_json / generate_json_from_images / generate_json_from_video (frame-sampled)
  scene_segmentation.py  LLM: script -> scenes
  scene_understanding.py LLM: scene text -> description/keywords/objects/mood/camera
  embeddings.py          sentence-transformers text -> vector (lru_cache'd)
  vector_store.py        ChromaDB wrapper (embedding + metadata storage, dedup lookups)
  clip_ingest.py          ffprobe + LLM vision analysis -> ClipRecord -> vector_store
  downloader.py           fetch + sha256 fingerprint
  stock/pexels.py, pixabay.py, pexels_images.py, pixabay_images.py, internet_archive.py, wikimedia.py, nasa.py
  pipeline.py             orchestrates the scene -> single-clip flow
  documentary_table.py    LLM: script -> timeline table (clips, keywords, visual type)
  scoring.py              weighted 0-100 asset scorer
  documentary_pipeline.py orchestrates the documentary timeline flow
  documentary_assembly.py contiguity validation + per-row speed-match/trim/loop/image render -> row segments
  remotion_render.py      builds Remotion props (segments/words/style/include_captions) + invokes render.mjs
  audio_mux.py            muxes supplied narration audio into the Remotion output
  subtitles.py            cue building (captions.srt) + word-timing for Remotion's captions
  timecode.py             "HH:MM:SS" <-> seconds helpers shared across documentary_pipeline.py / documentary_assembly.py / subtitles.py
  main.py                 FastAPI app (also serves GET / -> templates/index.html)
remotion/                 Remotion project (Assembly.tsx composites clips + transitions + optional Captions.tsx; render.mjs is invoked via node)
templates/index.html      manual test UI for /process-script and /generate-documentary-timeline
clips/local/              your existing B-roll library
clips/downloaded/         clips fetched from stock providers (scene pipeline)
projects/<name>/          documentary pipeline output (gitignored — final_video*.mp4, timeline.json, timeline_table.md, clips/)
data/chroma/              vector DB storage
```

## Running the tests

```
.venv\Scripts\python.exe -m pytest tests/ -v
```

No API keys or network needed for any suite. `test_documentary_assembly.py`
uses real ffmpeg (synthetic `lavfi` sources, no network) to prove exact-duration
trim/loop/image/concat behavior end to end — every other suite is fully
mocked. `test_documentary_pipeline.py` also runs standalone: `python tests/test_documentary_pipeline.py`.

### Manual verification checklist

- [ ] `pytest tests/ -v` — all pass
- [ ] `python tests/test_documentary_pipeline.py` — prints `OK`
- [ ] `uvicorn app.main:app --reload` starts without errors
- [ ] `POST /generate-documentary-timeline` with a real script (e.g. the Cold
      War example) returns a 200 with a non-empty `table` and `timeline`, and
      `final_video_with_narration_path` (or `final_video_path`/`subtitled_video_path`
      if no narration audio was supplied) pointing at a real file under
      `projects/<project_name>/` (or `assembly_error`/`subtitle_error` explaining why not)
- [ ] `projects/<project_name>/timeline_table.md` renders the expected pipe table,
      with no row longer than 5s
- [ ] `projects/<project_name>/timeline.json` has one entry per clip, each with
      `asset_path`/`asset_metadata` (or `placeholder: true` + `note`), plus a
      `status` (`ok`/`trimmed`/`looped-to-fill`/`speed-matched`/`missing`) and
      `rendered_clip_path`
- [ ] `projects/<project_name>/clips/clip_001/` (etc.) contains the downloaded
      source file (`clip001_video1.mp4`/`clip001_image1.jpg`) and the row-exact
      rendered segment (`clip001_rendered.mp4`)
- [ ] `ffprobe` each `clip{N}_rendered.mp4` and confirm its duration matches
      that row's `duration` in `timeline.json`
- [ ] `ffprobe projects/<project_name>/final_video_remotion.mp4` and confirm
      its total duration matches the last row's `end` timestamp
- [ ] `projects/<project_name>/captions.srt` exists and its cue text matches
      the `script` text in `timeline_table.md`, in order
- [ ] `remotion_props.json`'s `includeCaptions` matches what you passed (default
      `false` — extract a frame with ffmpeg and confirm no burned-in captions)
- [ ] If narration audio was supplied, `ffprobe final_video_with_narration.mp4`
      and confirm it has an audio stream matching the narration's duration
- [ ] Spot-check 2-3 downloaded assets actually match their clip's keywords
- [ ] Re-run with the same script/project_name and confirm no stock asset is
      reused across two different clips anywhere in the same video (unless
      `allow_duplicate_assets: true`)

## Design notes / deliberate simplifications

- **ChromaDB only, no SQL DB.** Metadata (source, source_id, fingerprint) lives
  alongside the embeddings; dedup is an exact-match `where` query on the same
  collection.
- **No OpenCV.** Clip analysis samples a few frames via ffmpeg (start/middle/
  end) and describes them through Groq's vision endpoint (`generate_json_from_video`
  in `app/llm_client.py`) — Groq has no video-file upload API like Gemini did,
  so full clips are never sent whole, only sampled frames.
- **No task queue.** Scenes are processed concurrently within one request via
  `asyncio.gather`. Fine for one operator running one script at a time.

## Upgrade paths (not built, add if you actually hit the wall)

- **Celery/RQ** — if you need to queue script-processing jobs across multiple
  workers/machines instead of one FastAPI process handling them inline.
- **Qdrant** — if the local clip library grows past what a single-machine
  ChromaDB instance handles comfortably, or you need the vector DB on a
  separate host.
- **Storyblocks / Shutterstock providers** — add another module under
  `app/stock/` matching the `search(query, per_page) -> list[StockHit]` shape
  once you have API credentials for them.
