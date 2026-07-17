"""Benchmarks the new Remotion final-render stage against the old ffmpeg
concat + subtitle burn-in, on a synthetic sample (solid-color 1920x1080
clips). Both paths need assemble_video's per-row rendered segments first —
that step is identical/shared cost either way, so it's timed once, separately,
and only the divergent final step (ffmpeg burn-in vs Remotion composite) is
compared head to head.

Run: python scripts/benchmark_render.py
"""
import subprocess
import tempfile
import time
from pathlib import Path

from app.documentary_assembly import _FPS, _HEIGHT, _WIDTH, assemble_video
from app.models import AssetInfo, TimelineEntry
from app.remotion_render import render_final_video
from app.style_decision import StyleDecision
from app.subtitles import generate_subtitled_video

NUM_CLIPS = 6
CLIP_SECONDS = 4.0
COLORS = ["red", "blue", "green", "yellow", "purple", "cyan"]


def _make_source(path: Path, color: str, duration: float) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c={color}:s={_WIDTH}x{_HEIGHT}:r={_FPS}:d={duration}",
         "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, check=True,
    )


def _build_timeline(sources_dir: Path) -> list[TimelineEntry]:
    entries = []
    for i in range(NUM_CLIPS):
        start_s, end_s = i * CLIP_SECONDS, (i + 1) * CLIP_SECONDS
        path = sources_dir / f"source_{i}.mp4"
        _make_source(path, COLORS[i % len(COLORS)], CLIP_SECONDS)
        entries.append(TimelineEntry(
            clip_number=i + 1,
            start=f"00:00:{int(start_s):02d}", end=f"00:00:{int(end_s):02d}", duration=CLIP_SECONDS,
            script=f"This is narration beat number {i + 1} describing the scene in enough words to caption.",
            asset_path=str(path),
            asset_metadata=AssetInfo(path=str(path), source="bench", source_id=str(i), media_type="video",
                                      width=_WIDTH, height=_HEIGHT, duration=CLIP_SECONDS, score=90.0),
            recommended_effect="slow zoom", transition="cross dissolve",
        ))
    return entries


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        sources_dir = project_dir / "sources"
        sources_dir.mkdir()
        timeline = _build_timeline(sources_dir)

        print(f"Sample: {NUM_CLIPS} clips x {CLIP_SECONDS}s = {NUM_CLIPS * CLIP_SECONDS:.0f}s total, "
              f"{_WIDTH}x{_HEIGHT}@{_FPS}fps")

        t0 = time.perf_counter()
        final_path, timeline = assemble_video(project_dir, timeline)
        assemble_time = time.perf_counter() - t0
        print(f"\n[shared]   assemble_video (per-row render + validation, same cost either way): "
              f"{assemble_time:.2f}s")

        # --- old path: ffmpeg concat (already done above) + ffmpeg subtitle burn-in ---
        t0 = time.perf_counter()
        generate_subtitled_video(project_dir, timeline, final_path)
        ffmpeg_subtitle_time = time.perf_counter() - t0
        print(f"[ffmpeg]   subtitle burn-in only:            {ffmpeg_subtitle_time:.2f}s")
        print(f"[ffmpeg]   TOTAL (assemble + burn-in):        {assemble_time + ffmpeg_subtitle_time:.2f}s")

        # --- new path: Remotion composite + captions + transitions in one pass ---
        style = StyleDecision(pacing="medium", transition_style="fade", caption_emphasis=["narration"])
        t0 = time.perf_counter()
        render_final_video(project_dir, timeline, style)
        remotion_time = time.perf_counter() - t0
        print(f"[remotion] composite + transitions + captions: {remotion_time:.2f}s")
        print(f"[remotion] TOTAL (assemble + remotion):        {assemble_time + remotion_time:.2f}s")

        slowdown = remotion_time / ffmpeg_subtitle_time if ffmpeg_subtitle_time > 0 else float("inf")
        print(f"\nRemotion's final-render step took {slowdown:.1f}x as long as ffmpeg's subtitle burn-in step.")
        if slowdown > 3:
            print("FLAG: Remotion is significantly slower at this stage — see summary for daily-volume implications.")


if __name__ == "__main__":
    main()
