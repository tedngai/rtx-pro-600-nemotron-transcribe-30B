#!/usr/bin/env python3
"""RTFx screening for the Nemotron Transcribe server.

Requires --execute --confirm-idle. Sends CONCURRENCY parallel transcription
requests of the same fixture, ROUNDS times, and reports:

  rtfx              = (audio_seconds * requests) / wall_seconds  (aggregate)
  per_request_rtfx  = audio_seconds / request_latency           (per stream)

Write one evidence report per invocation. Never restarts the server.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time

from common import Client, load_profile, report_path, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--label", default="rtfx-screen")
parser.add_argument("--execute", action="store_true")
parser.add_argument("--confirm-idle", action="store_true")
parser.add_argument("--concurrency", type=int, default=1)
parser.add_argument("--rounds", type=int, default=3)
parser.add_argument("--fixture", default=None,
                    help="audio fixture path relative to the workspace root "
                         "(default: tests/fixtures/en-Alice_woman.wav)")
args = parser.parse_args()

if not args.execute or not args.confirm_idle:
    raise SystemExit("Refusing to run: pass --execute --confirm-idle")

load_profile()  # fail fast on a missing profile
client = Client()
audio_path = args.fixture or "tests/fixtures/en-Alice_woman.wav"
audio_seconds = client.audio_duration_seconds(audio_path)

rounds = []
for index in range(args.rounds):
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(client.transcribe, audio_path=audio_path)
                   for _ in range(args.concurrency)]
        results = [future.result()["elapsed_seconds"] for future in futures]
    wall = time.perf_counter() - started
    round_summary = {
        "round": index,
        "requests": args.concurrency,
        "wall_seconds": round(wall, 3),
        "aggregate_rtfx": round(audio_seconds * args.concurrency / wall, 2),
        "per_request_rtfx": round(audio_seconds / max(results), 2),
        "latency_seconds_min": round(min(results), 3),
        "latency_seconds_median": round(sorted(results)[len(results) // 2], 3),
        "latency_seconds_max": round(max(results), 3),
    }
    rounds.append(round_summary)
    print(json.dumps(round_summary))

report = {
    "label": args.label,
    "mode": "rtfx",
    "concurrency": args.concurrency,
    "rounds": args.rounds,
    "audio_seconds_per_request": round(audio_seconds, 3),
    "results": rounds,
    "summary": {
        "median_aggregate_rtfx": sorted(r["aggregate_rtfx"] for r in rounds)[args.rounds // 2],
        "median_per_request_rtfx": sorted(r["per_request_rtfx"] for r in rounds)[args.rounds // 2],
    },
    "executed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "caveat": "Rounds reuse one fixture; the warm cache question does not apply "
              "because prefix caching is disabled for ReplaySSM. First requests "
              "after startup are slower (kernel warm-up).",
}
path = write_json(report_path(args.label), report)
print(json.dumps({"report": str(path), "summary": report["summary"]}, indent=2))
