# Clip Generation — B-Roll Retrieval & Documentary Assembly

FastAPI service that turns a script into a rendered documentary-style video: LLM
script segmentation → local/stock footage search & scoring → CLIP visual
verification → ffmpeg assembly → Remotion render (transitions + captions).

## 1. Setup

```powershell
cd D:\youtube_automation\clip_generation
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins a **CUDA-enabled torch build**
(`torch==2.13.0+cu126`, `--extra-index-url https://download.pytorch.org/whl/cu126`
at the top of the file). This is the confirmed working setup on this machine
(GTX 960, CUDA available, `torch.cuda.is_available() == True`) — CLIP
(`app/visual_verification.py`) and Whisper (`app/transcription.py`) both
auto-select `cuda` when available, no manual device wiring needed.

- **No NVIDIA GPU/driver?** Drop the pinned `torch` line and
  `--extra-index-url` and let `pip` resolve a CPU wheel instead — otherwise
  the install fails outright.
- **Without the pin**, a fresh install on Windows can silently fall back to a
  CPU-only torch wheel (pulled in transitively by whisper/CLIP) — this has
  happened before. If GPU inference isn't kicking in, `pip show torch` and
  confirm the version string ends in `+cu126`, not a bare version.

ffmpeg / ffprobe (frame sampling, trimming, concat, duration probing) ship
bundled as static binaries at `bin/ffmpeg.exe` / `bin/ffprobe.exe` — see
`bin/FFMPEG-README.txt` for the exact build and `bin/FFMPEG-LICENSE.txt` for
its GPL v3 terms. No separate install or PATH entry needed; `app/ffmpeg_utils.py`
resolves the bundled copy first and only falls back to PATH if it's missing
(e.g. a dev machine that already has ffmpeg installed globally).

Also required on `PATH`:
- **Node.js** — for the Remotion render step. Install its dependencies once:
  ```powershell
  cd remotion
  npm install
  cd ..
  ```

### `.env`

```
OPENAI_API_KEY=...
PEXELS_API_KEY=...
PIXABAY_API_KEY=...
COVERR_API_KEY=...
```

- `OPENAI_API_KEY` — all LLM calls (table generation, captioning, semantic
  keyword fallback, AI-image generation fallback) go through this. Required.
- `PEXELS_API_KEY` / `PIXABAY_API_KEY` / `COVERR_API_KEY` — stock video/image
  search. Required for those providers to return anything.
- No key is needed for Internet Archive or NASA — free/unauthenticated
  public APIs.

## 2. Running the Server

```powershell
cd D:\youtube_automation\clip_generation
.venv\Scripts\activate
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

Or just double-click `start_server.bat`, or run the installed `ClipGeneration.exe` (adds a system tray icon; see `installer.nsi`).

Serves on `http://localhost:8001` by default, bound to `0.0.0.0` so other
devices on the same LAN can reach it at `http://<this-machine's-LAN-IP>:8001`
(e.g. `http://192.168.1.23:8001`) — find the IP with `ipconfig`. This is
intended for use as an internal LAN tool only; there is no auth in front of
it, so don't expose port 8001 to the public internet (e.g. via router port
forwarding). `GET /` is a browser test UI (`templates/index.html`) with
Run/Preview buttons; `GET /health` is a bare liveness check.

**Before starting, verify the port is actually free.** `uvicorn --reload`
normally runs two `python.exe` processes for one server (a reloader parent +
a worker child) — that's expected. What's *not* expected is a stale server
from an earlier session still bound to the same port when you start a new
one:

```powershell
Get-NetTCPConnection -LocalPort 8001 -ErrorAction SilentlyContinue
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Select-Object ProcessId, CommandLine
```

If a `python.exe` running `app.main:app` shows up that you didn't just
start, stop it (`Stop-Process -Id <id>`) before starting a new one.

## 3. Generating a Video

Request body shared by all three endpoints below (`DocumentaryRequest` in
`app/main.py`):

```json
{
  "script": "your narration script text",
  "script_path": null,
  "project_name": null,
  "allow_duplicate_assets": false,
  "max_downloads": null,
  "audio_path": null,
  "content_niche": "trucks"
}
```

- `content_niche` — one of the keys in `app/niches.py` (`trucks`,
  `pool_maintenance`, `bodybuilding`); defaults to `trucks` if omitted.
- `audio_path` — path to a real narration recording already on disk. If
  given, Whisper transcribes it for real per-word timing instead of the
  word-count pacing estimate.
- Either `script` or `script_path` must resolve to non-empty text.

### Full pipeline run

```
POST /generate-documentary-timeline
```
Kicks off the whole pipeline (search → download → CLIP-verify → assemble →
Remotion render) as a background task and returns immediately:
```json
{"project_name": "project_1234567890"}
```
Poll `GET /progress/{project_name}` for status; output lands in
`projects/<project_name>/` (`final_video_with_narration.mp4` if audio was
supplied, else `final_video_remotion.mp4`, plus `timeline.json`,
`captions.srt`, `footage_availability_report.json`,
`keyword_match_report.md`).

### Cheap pre-render check

```
POST /generate-documentary-timeline/preview
```
Same request body. Runs table generation + a footage-availability scan only
— **no download, no CLIP verification, no render** — and returns the
planned table plus a thin-clip report synchronously. Use this to catch a
bad/thin script before paying for the full run.

### File upload variant

```
POST /generate-documentary-timeline/upload
```
`multipart/form-data` — same fields as form fields, plus optional
`script_file` / `audio_file` uploads. Used by the browser UI's file pickers.

### Progress polling

```
GET /progress/{project_name}
```
Returns 404 until the project has actually started (i.e. after one of the
`POST` calls above has returned its `project_name`).

## 4. Local Library (Niche-Specific Clips)

A curated, pre-indexed folder of your own real footage for a niche —
searched **before** any stock provider, with a scoring priority boost (see
`app/documentary_pipeline.py::search_clip`, `app/scoring.py::score_asset`).

**Folder structure:**
```
clips/local/<niche>/*.mp4
```
(`settings.local_clips_dir` = `clips_dir / "local"`, `clips_dir` defaults to
`./clips`.) e.g. `clips/local/bodybuilding/*.mp4`.

**Indexing command:**
```powershell
python -m app.stock.local_library_index --niche bodybuilding
```

What it does, per new (not-yet-indexed) file:
1. Computes a SHA256 content hash. If it exactly matches an already-indexed
   file's hash, **deletes the duplicate** and skips it — no captioning, no
   second entry. Only exact byte-identical duplicates are caught this way;
   near-duplicates are left alone (flagged as a possible future manual-review
   feature, not built).
2. Otherwise, captions it via a vision-LLM call (samples start/middle/end
   frames) and asks for both a detailed caption and a short, distinctive
   filename slug in the same call.
3. Renames the file on disk to that slug (collision-safe — appends `_2`,
   `_3`, etc. rather than ever overwriting).
4. Records `{caption, width, height, duration, hash}` under the new filename
   in `clips/local/<niche>/index.json`.

Already-indexed files are skipped entirely (neither re-captioned nor
renamed) — **safe to re-run anytime**, only new files cost API calls.
Progress is written incrementally (after each file), so an interrupted run
loses nothing already completed.

**To force a full re-index** of a niche, delete its `index.json` first, then
re-run the same command — every file will be treated as new (and re-costs
the vision API calls).

Real-world note: a large batch of new files can hit OpenAI's tokens-per-minute
limit — expect periodic `429 Too Many Requests` in the log. The indexer has
its own backoff/retry built in (`app/stock/local_library_index.py`), so this
is expected noise, not a failure, as long as you eventually see
`"N clip(s) indexed total"` printed at the end.

## 5. Adding a New Niche

Niches are defined in `app/niches.py`, in the `NICHES` dict (`NicheConfig`
`NamedTuple`). Fields:

| Field | Purpose |
|---|---|
| `key` | Dict key / the string passed as `content_niche` |
| `display_name` | Human-readable label |
| `system_context` | Injected into every keyword-generation LLM prompt — describes what's in/out of scope, with context-dependent nuance a denylist can't express |
| `banned_terms` | Hard denylist checked against generated keywords — only substrings that are **never** valid regardless of context |
| `positive_terms` | If present alongside a banned term in a stock candidate's own metadata, don't hard-reject it (real footage legitimately co-mentions both, e.g. "cars" and "truck" on ordinary highway footage) |
| `safe_fallback_keyword` | Substituted in when a generated keyword violates the denylist |
| `use_archive_org` | (optional, default `False`) whether to query Internet Archive for this niche — only worth it for historical/archival content |
| `local_library_path` | (optional, default `""`) subfolder name under `clips/local/` holding this niche's local library — see §4. Empty means no local library for this niche |

Current niches: `trucks`, `pool_maintenance`, `bodybuilding`.

python -m app.stock.local_library_index --niche bodybuilding

## 6. Key Settings (`app/config.py`)

| Setting | Default | Controls |
|---|---|---|
| `documentary_min_score` | `50.0` | Floor score (0-100) a candidate must clear to be considered acceptable at all |
| `ai_generation_trigger_threshold` | `75.0` | If the best real candidate for a clip scores below this, AI-image generation is attempted for it instead (a quality gate, not just an empty-results fallback) |
| `max_asset_repeat_count` | `1` | How many clips the same stock/local asset may win across one project before it's excluded from further search (true never-repeat at the default) |
| `enable_ai_generation_fallback` | `True` | Master switch for the AI-image generation fallback — since this can affect most of a video's visuals once the trigger threshold is in play, verify output quality with a real run before relying on it |
| `ai_generation_model` | `"gpt-image-2"` | OpenAI image model used for the fallback |
| `ai_generation_quality` | `"medium"` | Generation quality tier (`"low"` is cheapest) |
| `visual_verification_threshold` | `0.20` | CLIP cosine-similarity floor a downloaded candidate's actual frame must clear against `canva_keyword` — below this, falls through to the next-best candidate |

## 7. Troubleshooting

**Windows `--reload` stale-worker confusion** — see §2. Two `python.exe`
processes for one running server is normal; a *second, independent* server
left over from an earlier session on the same port is not. Run both
diagnostic commands from §2 before assuming a fresh `--reload` restart
actually replaced the old process.

**OpenAI `429 Too Many Requests`** — two different causes:
- *Request-rate limit* — `app/llm_client.py`'s own throttle
  (`llm_max_requests_per_minute`, default 30) paces this; shouldn't normally
  trigger real 429s on its own.
- *Tokens-per-minute limit* — multi-image vision calls (captioning,
  candidate description) can burn through a tier's TPM budget in a handful
  of requests, well before the request-count throttle would kick in. This
  is expected and handled via retry-with-backoff
  (`_call_with_rate_limit_retry` in `app/llm_client.py`; the local library
  indexer also has its own extra backoff layer on top).

If 429s are persistent rather than occasional, check your actual usage tier
and limits at
[platform.openai.com/settings/organization/limits](https://platform.openai.com/settings/organization/limits)
— a low tier's TPM budget can be the real bottleneck, not a bug.

## Tests

```powershell
python -m pytest tests/ -q
```
Offline/mocked except `test_documentary_assembly.py` (shells out to real
ffmpeg with synthetic `lavfi` sources, no network).
