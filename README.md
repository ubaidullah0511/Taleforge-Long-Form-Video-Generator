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
  keyword fallback, AI-image generation fallback, niche-config generation) go
  through this. Required.
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

Or just double-click `start_server.bat`, or run the installed `ClipGeneration.exe`
(adds a system tray icon). **For a non-technical end user, skip all of this
entirely and use the Windows installer instead — see §9.**

Serves on `http://localhost:8001` by default, bound to `0.0.0.0` so other
devices on the same LAN can reach it at `http://<this-machine's-LAN-IP>:8001`
(e.g. `http://192.168.1.23:8001`) — find the IP with `ipconfig`. This is
intended for use as an internal LAN tool only; there is no auth in front of
it, so don't expose port 8001 to the public internet (e.g. via router port
forwarding). `GET /` is the main browser UI (`templates/index.html`) with
Run/Preview buttons and niche management; `GET /editor/{project_name}` is the
post-generation timeline editor (`templates/editor.html`, see §6); `GET /health`
is a bare liveness check.

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

Request body shared by all the endpoints below (`DocumentaryRequest` in
`app/main.py`):

```json
{
  "script": "your narration script text",
  "script_path": null,
  "project_name": null,
  "allow_duplicate_assets": false,
  "max_downloads": null,
  "audio_path": null,
  "content_niche": "trucks",
  "enable_ai_generation": true,
  "enable_local_library": true,
  "enable_stock_providers": true
}
```

- `content_niche` — either a single niche key (string, e.g. `"trucks"`), or
  a list of 2+ sub-niche keys that share the same parent category (the
  website's multi-select sub-niche picker — see §5). Defaults to `trucks` if
  omitted. An unknown single key silently falls back to the default; an
  invalid multi-select combination (keys from different parent categories)
  is rejected with a `400` before any work is queued.
- `audio_path` — path to a real narration recording already on disk. If
  given, Whisper transcribes it for real per-word timing instead of the
  word-count pacing estimate. The `/upload` routes below accept an actual
  audio file instead of a path.
- Either `script` or `script_path` must resolve to non-empty text.

### Source toggles

Three independent switches over where footage/imagery is allowed to come
from — all default `true` (today's full-search behavior). Turning all three
off means every clip becomes a placeholder (nothing left to search). Mirrored
1:1 by the three checkboxes above the Run button in the website UI:

| Field | Website checkbox | Controls |
|---|---|---|
| `enable_ai_generation` | "AI-generated images" | Whether the AI-image fallback (`ai_generation_trigger_threshold`, see §7) is allowed to fire at all |
| `enable_local_library` | "Local library" | Whether the current niche's `clips/local/<niche>/` folder (§4) is searched |
| `enable_stock_providers` | "Stock providers" | Whether Pexels/Pixabay/Coverr/NASA/Archive.org are queried |

`POST /generate-documentary-timeline/preview` (below) only honors
`enable_local_library` and `enable_stock_providers` — it never reaches the
AI-generation stage regardless of `enable_ai_generation`, since it's a
pre-render availability scan, not a real resolve.

### Two run modes

The website's Run button has a mode dropdown (defaults to **Generate & Edit**):

| Mode | Endpoint | What it does |
|---|---|---|
| **Generate & Edit** | `POST /generate-documentary-timeline/edit` | Runs search → download → CLIP-verify → per-clip segment render, then **stops** — no Remotion pass, no audio mux. The website redirects straight to the timeline editor (§6) once this finishes. |
| **Generate Full Video** | `POST /generate-documentary-timeline` | The complete pipeline through Remotion render + audio mux, same as before. |

Both accept the same request body and return the same shape immediately
(the real work runs as a background task):
```json
{"project_name": "project_1234567890"}
```
Poll `GET /progress/{project_name}` for status either way.

### File upload variants

```
POST /generate-documentary-timeline/upload
POST /generate-documentary-timeline/edit/upload
```
`multipart/form-data` — same fields as form fields (repeat `content_niche_list`
for a multi-select sub-niche pick), plus optional `script_file` / `audio_file`
uploads. Used by the website's file pickers.

### Cheap pre-render check

```
POST /generate-documentary-timeline/preview
```
Same request body. Runs table generation + a footage-availability scan only
— **no download, no CLIP verification, no render** — and returns the
planned table plus a thin-clip report synchronously. Use this to catch a
bad/thin script before paying for the full run.

### Progress polling

```
GET /progress/{project_name}
```
Returns 404 until the project has actually started (i.e. after one of the
`POST` calls above has returned its `project_name`). Once a run finishes,
output for a full "Generate Full Video" run lands in `projects/<project_name>/`
(`final_video_with_narration.mp4` if audio was supplied, else
`final_video_remotion.mp4`, plus `timeline.json`, `captions.srt`,
`footage_availability_report.json`, `keyword_match_report.md`); a "Generate &
Edit" run stops after `timeline.json` and per-clip rendered segments, with no
final video yet — see §6 for how to produce one from the editor.

## 4. Local Library (Niche-Specific Clips)

A curated, pre-indexed folder of your own real footage for a niche —
searched **before** any stock provider, with a scoring priority boost (see
`app/documentary_pipeline.py::search_clip`, `app/scoring.py::score_asset`).

**Folder structure:**
```
clips/local/<niche>/*.mp4
```
(`settings.local_clips_dir` = `clips_dir / "local"`, `clips_dir` defaults to
`./clips`.) e.g. `clips/local/bodybuilding/*.mp4`. A niche's local library
folder can also be nested (e.g. `clips/local/tennis/alex_eala/`) — whatever
path is in that niche's `local_library_path` (see §5).

**Indexing:** two ways to trigger the same indexer
(`app/stock/local_library_index.py::index_niche`):
- **Website (recommended)** — open "Manage Niches" on the main page and click
  **Sync Library** next to a custom niche (only enabled for niches that have a
  local library folder linked). Runs as a background job with its own
  progress bar and a **Stop** button (`POST /niches/{key}/sync-library`,
  `POST /niches/{key}/sync-library/stop` — stops before the next file, never
  mid-file; already-indexed files from earlier in the run stay indexed).
- **CLI** (works for built-in niches too, not just custom ones):
  ```powershell
  python -m app.stock.local_library_index --niche bodybuilding
  ```

What it does, per new (not-yet-indexed) file:
1. Computes a SHA256 content hash. If it exactly matches an already-indexed
   file's hash, **deletes the duplicate** and skips it — no captioning, no
   second entry. Only exact byte-identical duplicates are caught this way;
   near-duplicates are handled separately (see `app/stock/find_near_duplicates.py`).
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

## 5. Niches

**Current built-in niches:** `trucks`, `pool_maintenance`, `bodybuilding`,
`prison` (`app/niches.py`'s `NICHES` dict). Beyond these, any number of
**custom niches** can be added at runtime (persisted to
`app/custom_niches.json`, merged into the live registry on load and on every
add — no restart needed) — see below.

### Managing niches from the website (recommended)

The category picker on the main page has two buttons next to it:

- **"+ Niche"** — type a name (e.g. "Tennis"), optionally pick an existing
  niche as its parent, and submit. An LLM (`POST /niches`) generates a
  starting `system_context` + `banned_terms` for it, the niche is
  immediately usable in the dropdown (no restart), and a matching
  `clips/local/<key>/` folder is created automatically if one doesn't
  already exist under that name — ready to drop footage into right away.
  Treat the LLM's output as a **starting point**: review/refine
  `system_context`/`banned_terms` by hand (in `app/custom_niches.json` or via
  `app/niches.py`-style editing) after running a real video through it once.
- **"Manage Niches"** — lists every *custom* niche (built-ins aren't listed
  here — they can't be renamed or deleted) with **Sync Library** (§4),
  **Rename** (display name only — see below), and **Delete** actions.

### Sub-niches (parent/child)

Any niche — built-in or custom — can be a parent; a niche becomes its child
by setting `parent_key` to the parent's key. Example already in this project:
**Tennis Documentary** (top-level) → **Tennis Players** (sub-niche) and
**Alex Eala** (sub-niche, narrower still, both parented to Tennis Documentary).
Creating a sub-niche via "+ Niche" with a parent selected:
- Generates a `system_context` narrowed to the sub-topic but still scoped
  within the parent's overall subject (the LLM prompt includes the parent's
  own `system_context` for reference).
- **Unions `banned_terms`** with the parent's — a sub-niche is never *less*
  restrictive than its parent, only more specific.
- Sorts directly under its parent in the category dropdown (`GET /niches`
  groups by `parent_key` for this).

Deleting a niche that still has sub-niches pointing at it is blocked (delete
the children first) — a parent is never removed out from under a niche that
still references it. Renaming a custom niche only changes its display name;
the key itself is immutable (changing it would break every already-generated
project whose `content_niche` points at the old key, plus every sub-niche's
`parent_key` cross-reference).

### Multi-select sub-niches (one generation run, several sub-niches at once)

Once a parent niche has 2+ sub-niches, the category picker grows a second,
multi-select dropdown ("Select all" / "Clear all" + checkboxes) for picking
several of them together for one generation run — e.g. searching both
**Tennis Players** and **Alex Eala** footage in the same video.
`content_niche` is then sent as a list of keys instead of one string
(`app/niches.py::resolve_sub_niches`):
- All selected keys must share the same `parent_key` (mixing sub-niches from
  different parent categories is rejected with a `400`).
- The keyword-generation config becomes a merge: the shared parent's
  `system_context`/`safe_fallback_keyword`, plus the **union** of every
  selected sub-niche's `banned_terms` and `positive_terms`.
- Crucially, **every** selected sub-niche's `local_library_path` gets
  searched in the same run (not just one) — a single `NicheConfig` only has
  room for one path, so this is handled as a separate list threaded through
  `documentary_pipeline.search_clip` alongside the merged config.

### Managing niches by hand (`app/niches.py`, built-in/advanced path)

Built-in niches (and any custom niche you'd rather hand-edit than manage
through the website) are plain `NicheConfig` `NamedTuple` entries in the
`NICHES` dict:

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
| `parent_key` | (optional, default `None`) key of the niche this one narrows — see Sub-niches above |

### Niche API endpoints

| Method & path | Purpose |
|---|---|
| `GET /niches` | Full list (built-in + custom), grouped with sub-niches sorted under their parent |
| `POST /niches` | `{"name": ..., "parent_key": null}` — LLM-generate + add a custom niche |
| `PATCH /niches/{key}` | `{"display_name": ...}` — rename a custom niche (display name only) |
| `DELETE /niches/{key}` | Delete a custom niche (blocked on built-ins and on niches with sub-niches) |
| `POST /niches/{key}/sync-library` | Start a local-library index/sync (background job, see §4) |
| `POST /niches/{key}/sync-library/stop` | Request the in-progress sync stop before its next file |

## 6. Editor & Two-Mode Generation

"Generate & Edit" (§3) stops right after per-clip footage resolution and
segment rendering, then the website redirects to `GET /editor/{project_name}`
— a fast, per-clip editing pass over `timeline.json` that avoids re-paying
for a full Remotion + audio-mux render on every small change.

**Editor UI (`templates/editor.html`):**
- **Timeline strip** — one block per clip, click to select and open its edit
  panel. A **zoom slider** controls how many clips fit on screen at once.
  Dragging a clip left/right previews a different order **for playback only**
  — see the "Generate Full Video" caveat below.
- **Preview** — clicking/dragging the timeline scrubs a `<video>` preview
  (per-clip segments are silent by design; narration plays back through a
  separately synced `<audio>` track), with a small hover thumbnail
  (`scrub-preview`) while dragging.
- **Caption preview strip** — shows the caption text active at the current
  playhead position, below the timeline.
- **Per-clip edit panel**, opened by clicking a clip, currently offers:
  - **Regenerate via AI** — `PATCH /project/{project_name}/clip/{clip_number}`
    with `{"mode": "ai"}`.
  - **Upload custom file** — `PATCH /project/{project_name}/clip/{clip_number}/upload`
    (multipart, replaces the clip with your own video/image).

  The backend also supports a third mode, swapping to an already-resolved
  **alternate** candidate (`{"mode": "alternate", "alternate_index": N}`) —
  implemented and tested, but not yet wired to a button in the editor UI.

**Producing a final video from the editor:** the **"Generate Full Video"**
button (`POST /project/{project_name}/generate-full-video`) runs the real
Remotion composite → concat → audio-mux pass over `timeline.json` **in its
original clip order** — any drag-reorder done in the preview above is
playback-preview-only and is never sent to this endpoint. Same
background-task/`GET /progress/{project_name}` polling shape as §3.

**Loading project state:** `GET /project/{project_name}` returns the current
timeline (each entry annotated with browser-fetchable `/project-files/...`
URLs for its asset/rendered-segment), the original planning table (for the
AI-regenerate path), and URLs for whichever final-video variants already
exist on disk.

## 7. Key Settings (`app/config.py`)

| Setting | Default | Controls |
|---|---|---|
| `documentary_min_score` | `50.0` | Floor score (0-100) a candidate must clear to be considered acceptable at all |
| `ai_generation_trigger_threshold` | `75.0` | If the best real candidate for a clip scores below this, AI-image generation is attempted for it instead (a quality gate, not just an empty-results fallback) |
| `max_asset_repeat_count` | `1` | How many clips the same stock/local asset may win across one project before it's excluded from further search (true never-repeat at the default) |
| `enable_ai_generation_fallback` | `True` | Master switch for the AI-image generation fallback — since this can affect most of a video's visuals once the trigger threshold is in play, verify output quality with a real run before relying on it |
| `ai_generation_model` | `"gpt-image-2"` | OpenAI image model used for the fallback |
| `ai_generation_quality` | `"medium"` | Generation quality tier (`"low"` is cheapest) |
| `visual_verification_threshold` | `0.20` | CLIP cosine-similarity floor a downloaded candidate's actual frame must clear against `canva_keyword` — below this, falls through to the next-best candidate |

## 8. Distribution — Windows Installer (Non-Technical Users)

For sharing this app with someone who shouldn't need to touch Python, `venv`,
or a terminal at all: a Windows installer built with
[NSIS](https://nsis.sourceforge.io/) packages the whole project (minus
`.venv`/`node_modules`/other generated directories, which are (re)built on
first run) plus a system-tray launcher, into one `.exe`.

- **`ClipGeneration-Setup.exe`** — the installer itself (`installer.nsi`,
  compiled with `makensis`). Installs to Program Files (or a chosen
  location), creates Desktop + Start Menu shortcuts, registers a normal
  Add/Remove Programs uninstaller, and bundles `bin/ffmpeg.exe` /
  `bin/ffprobe.exe` (§1) so no separate ffmpeg install is needed post-install
  either.
- **`ClipGeneration.exe`** — the system tray launcher (built from
  `app_tray.py` via PyInstaller). On first launch, it detects that `.venv`
  doesn't exist yet and runs `setup.bat` with a visible progress window
  (same GPU-aware CUDA-vs-CPU torch detection as §1); every launch after
  that skips straight to starting the server and opening the browser, with a
  tray icon whose "Exit" item cleanly stops the `uvicorn` child process.

**Rebuilding both**, on a real Windows machine, from the project root:
```powershell
# 1. Rebuild the tray launcher exe (only needed if app_tray.py changed)
.venv\Scripts\pip install -r requirements-launcher.txt
.venv\Scripts\pyinstaller --onefile --windowed --icon=icon.ico `
    --add-data "icon.ico;." --name=ClipGeneration app_tray.py
# produces dist\ClipGeneration.exe

# 2. Compile the installer (packages dist\ClipGeneration.exe + the project)
"C:\Program Files (x86)\NSIS\makensis.exe" installer.nsi
# produces ClipGeneration-Setup.exe in the project root
```
Copy the freshly compiled `ClipGeneration-Setup.exe` into `dist/` alongside
`ClipGeneration.exe`, `.env.example`, and `SETUP_INSTRUCTIONS.txt` — that
folder is the complete, ready-to-send package for a non-technical recipient
(`SETUP_INSTRUCTIONS.txt` walks them through install → first-launch wait →
adding their own API keys to `.env` → using the app).

This is the **recommended path for non-technical users**; the manual
`venv`/`uvicorn` setup in §1–§2 remains the path for development.

## 9. Troubleshooting

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
