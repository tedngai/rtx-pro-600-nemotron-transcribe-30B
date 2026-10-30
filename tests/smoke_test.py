#!/usr/bin/env python3
"""Smoke test for the Nemotron Transcribe serving workspace.

Requires --execute --confirm-idle (the workspace hard rule). Checks health,
the served model card, and two transcription requests (cold and repeated).
"""

import argparse
import json

from common import Client, load_profile, report_path, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--label", default="smoke-interface")
parser.add_argument("--execute", action="store_true")
parser.add_argument("--confirm-idle", action="store_true")
parser.add_argument("--base-url", default=None)
args = parser.parse_args()

if not args.execute or not args.confirm_idle:
    raise SystemExit("Refusing to run: pass --execute --confirm-idle")

profile = load_profile()
base_url = args.base_url or "http://127.0.0.1:8013"
client = Client(base_url=base_url)

report = {"label": args.label, "executed_at": None, "checks": {}, "error": None}
model_card = client.get_json("/v1/models")["data"][0]
expected = profile["expected_config"]
report["checks"]["model_card"] = {
    "passed": model_card["id"] == expected["served_model_name"]
              and model_card.get("max_model_len") == expected["context_length"],
    "served_model_name": model_card["id"],
    "max_model_len": model_card.get("max_model_len"),
}

duration = client.audio_duration_seconds()
cold = client.transcribe()
text_cold = (cold["response"]["choices"][0]["message"].get("content") or "").strip()
report["checks"]["transcribe_cold"] = {
    "passed": len(text_cold.split()) >= 3,
    "elapsed_seconds": round(cold["elapsed_seconds"], 3),
    "audio_seconds": round(duration, 2),
    "completion_tokens": cold["response"]["usage"]["completion_tokens"],
    "transcript_preview": text_cold[:400],
}
repeat = client.transcribe()
text_repeat = (repeat["response"]["choices"][0]["message"].get("content") or "").strip()
report["checks"]["transcribe_repeat"] = {
    "passed": text_repeat == text_cold,
    "elapsed_seconds": round(repeat["elapsed_seconds"], 3),
    "identical_to_cold": text_repeat == text_cold,
    "transcript_preview": text_repeat[:400],
}

report["passed"] = all(check.get("passed") for check in report["checks"].values())
import time
report["executed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
path = write_json(report_path(args.label), report)
print(json.dumps({"report": str(path), "passed": report["passed"],
                  "checks": {k: v["passed"] for k, v in report["checks"].items()}}, indent=2))
