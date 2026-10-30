#!/usr/bin/env python3
"""Shared client and helpers for the Nemotron Transcribe screening harness.

Standard library only. Follows the dgx-spark-nanbeige-4.3-33B workspace
conventions: reports under evidence/, profile-verified requests, no sudo.
"""

import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE = "production"
DEFAULT_URL = "http://127.0.0.1:8013"
DEFAULT_CONTAINER = "vllm-nemotron-transcribe-30b"
FIXTURE = ROOT / "tests" / "fixtures" / "en-Alice_woman.wav"

TRANSCRIBE_PROMPT = ("Audio transcription. Language: as spoken. Wording: verbatim, "
                     "including fillers and repetitions. Formatting: punctuation and "
                     "capitalization. Output: transcript only.")


def load_profile(name=DEFAULT_PROFILE):
    path = ROOT / "configs" / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"Unknown profile: {name}")
    return json.loads(path.read_text())


def stamp():
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime())


def report_path(label):
    directory = ROOT / "evidence"
    directory.mkdir(exist_ok=True)
    return directory / f"{label}-{stamp()}.json"


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


class Client:
    def __init__(self, base_url=DEFAULT_URL, timeout=600, profile=DEFAULT_PROFILE):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.served_name = load_profile(profile)["serving"]["served_model_name"]

    def get_json(self, path):
        request = urllib.request.Request(self.base_url + path)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def transcribe(self, prompt=TRANSCRIBE_PROMPT, audio_path=FIXTURE,
                   thinking=False, temperature=0.0, max_tokens=8192,
                   stream=False):
        audio_b64 = (ROOT / audio_path).read_bytes() if isinstance(audio_path, str) \
            else audio_path.read_bytes()
        audio_b64 = __import__("base64").b64encode(audio_b64).decode()
        payload = {
            "model": self.served_name,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "audio_url",
                 "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}},
            ]}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.load(response)
        elapsed = time.perf_counter() - started
        return {"response": body, "elapsed_seconds": elapsed}

    def audio_duration_seconds(self, path=FIXTURE):
        import wave
        with wave.open(str(ROOT / path)) as handle:
            return handle.getnframes() / handle.getframerate()
