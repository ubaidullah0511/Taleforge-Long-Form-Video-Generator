import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { bundle } from "@remotion/bundler";
import { renderMedia, selectComposition } from "@remotion/renderer";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function parseArgs() {
  const args = {};
  for (const arg of process.argv.slice(2)) {
    const stripped = arg.replace(/^--/, "");
    const eq = stripped.indexOf("=");
    args[stripped.slice(0, eq)] = stripped.slice(eq + 1);
  }
  if (!args.props || !args.output || !args["public-dir"]) {
    throw new Error("Usage: node render.mjs --props=<path.json> --output=<path.mp4> --public-dir=<dir>");
  }
  return args;
}

async function main() {
  const { props: propsPath, output, "public-dir": publicDir } = parseArgs();
  const inputProps = JSON.parse(fs.readFileSync(propsPath, "utf-8"));

  const entry = path.join(__dirname, "src", "index.ts");
  // publicDir is the project's own directory (segment paths in inputProps are
  // relative to it) — Remotion's headless-Chromium renderer can only fetch
  // assets served over http(s) by its own dev server, not via arbitrary
  // absolute paths or file:// URLs, hence serving this directory as the
  // bundle's static root instead.
  const bundleLocation = await bundle({ entryPoint: entry, publicDir });

  const composition = await selectComposition({
    serveUrl: bundleLocation,
    id: "Assembly",
    inputProps,
  });

  fs.mkdirSync(path.dirname(output), { recursive: true });

  await renderMedia({
    composition,
    serveUrl: bundleLocation,
    codec: "h264",
    outputLocation: output,
    inputProps,
    logLevel: "warn",
    // Remotion's own default is round(min(8, max(1, cpus/2))) — capped at 8
    // and only half the logical cores. This machine has os.cpus().length
    // logical cores available; use all of them.
    concurrency: os.cpus().length,
    // "angle" tested best on this machine's NVIDIA GPU/drivers (~40s vs
    // ~45-51s with Chrome's software-rendering default; "egl" also worked
    // but was marginally slower — see scripts/benchmark_render.py output).
    // Was briefly suspected as the cause of the "Timed out setting up
    // headless browser" failure, but that was actually app/remotion_render.py
    // not setting cwd (see its comment) — confirmed this option is unrelated.
    chromiumOptions: {
      gl: "angle",
    },
  });

  console.log(`RENDERED:${output}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
