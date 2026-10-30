# RTX Pro 6000 Blackwell — Nemotron 3.5 Transcribe 30B-A4B on vLLM

An evidence-first serving workspace for **NVIDIA-Nemotron-3.5-Transcribe-30B-A4B** (speech transcription/translation/audio-QA, 34B total / 4.3B active) on the GPU 0 MIG slice of one NVIDIA RTX Pro 6000 Blackwell DC-4-96Q. vLLM serves an OpenAI-compatible API; every measurement is written to `evidence/`.

Format adapted from [tedngai/dgx-spark-nanbeige-4.3-33B](https://github.com/tedngai/dgx-spark-nanbeige-4.3-33B) (same workspace conventions as the local `nanbeige4.3-33b` workspace).

This is a screening workspace, not a production deployment. `production_qualified` is false everywhere.

## Host

| | |
|---|---|
| GPU | NVIDIA RTX Pro 6000 Blackwell DC-4-96Q, MIG `4g.96gb` slice of GPU 0 (`MIG-b12c2fdf-…`) |
| Coexistence | GPU 1 runs `d3-systemone` (:8000); ComfyUI :8188; the `nanbeige4.3-33b` workspace also targets this slice (:8012, currently stopped) — only one of the two can hold the GPU |
| API | `http://<host>:8013/v1` — LAN-exposed without auth (same posture as `pratt-agent2`, deliberate choice) |

## Why BF16 (for now)

The publisher's BF16 checkpoint is the reference this card's recipe targets. At 62.88 GiB it fits the ~94 GiB MIG slice with `--gpu-memory-utilization 0.94` (~88 GiB plan). FP8 and NVFP4 variants exist and would free ~30 GiB for KV/audio batches — a candidate follow-up screen if transcription throughput becomes the bottleneck.

## Production recipe (frozen, from the model card)

| Setting | Value |
| --- | --- |
| Model | `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` @ `a99e33f…` (62.88 GiB BF16) |
| Architecture | Mamba-2–Transformer hybrid MoE + Parallel Expert Encoder (ASR + speaker) |
| Engine | vLLM 0.32.0 + NeMo Speech main (`nemo_speechlm` plugin) @ `548fd86…`, uv-locked py3.13 env per the card, torch 2.13.0+cu130, flashinfer 0.7.0.post1 |
| Image | `nemotron-transcribe:v0.32.0-nemo`, built from `docker/Dockerfile` (no upstream digest exists; the Dockerfile is the pinned contract) |
| Served context | 65,536 tokens; recordings up to 1 hour (`VLLM_MAX_AUDIO_DECODE_DURATION_S=3600`) |
| Speculative decoding | MTP8 (`{"method":"mtp","num_speculative_tokens":8}`) |
| Mamba | `--use-replayssm --mamba-backend flashinfer --mamba-cache-mode none`, prefix caching **off** (card requirement) |
| Concurrency | 192 sequences, 4,096-token prefill chunks |
| Word timestamps | `--middleware …ctc_timestamp_middleware --trust-request-mm-kwargs` |
| Auth | **None** — `REQUIRE_API_KEY=0`, LAN-exposed by explicit decision |

`configs/production.json` is the frozen contract; `scripts/start.sh` assembles the flags.

## Host prerequisites

- Docker with the NVIDIA runtime or CDI spec.
- ~130 GiB free disk for the weights and image.
- Python 3.12 for the harness (standard library only).

## Quick start

```
scripts/download-model.sh            # 62.9 GiB, resumes, verifies every SHA-256
docker build -t nemotron-transcribe:v0.32.0-nemo docker/
scripts/start.sh                     # serve and wait for /health
scripts/status.sh

export NANBEIGE_API_KEY=""           # no auth; the key is unused unless enabled
python3 -B tests/smoke_test.py --label smoke-interface --execute --confirm-idle
python3 -B tests/benchmark.py --label rtfx-c1 --execute --confirm-idle --concurrency 1 --rounds 3
```

`scripts/stop.sh` stops and preserves the container; `scripts/stop.sh --remove` deletes it.

## Sending a transcription request

```
curl http://<host>:8013/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "pratt-transcribe",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Audio transcription. Language: as spoken. Wording: verbatim. Output: transcript only."},
    {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,<base64>"}}
  ]}],
  "temperature": 0,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

## What the harness measures

- **Smoke** — model card contract, cold transcription, repeated-request stability.
- **RTFx screens** — aggregate and per-stream RTFx (audio-seconds processed per wall-second) at C1/C8, from `tests/benchmark.py`. The NIM-only serving optimizations are not in the open-source release; the card expects the open stack to be ~2× slower than the published NIM numbers.

## Results

Pending — see `evidence/` after the screen run.

## Contents

```
configs/     frozen profile (production.json)
docker/      Dockerfile replicating the card's env setup
scripts/     download, start, stop, status
tests/       smoke test, RTFx benchmark, fixtures
evidence/    append-only JSON reports
models/      downloaded weights (gitignored)
secrets/     API key placeholder (gitignored; unused while auth is off)
state/       model verification record (gitignored)
```

## Qualification boundary

Three-round screens are screening evidence, not stable tails. WER/cpWER quality is not reproduced here — only serving behavior. The first requests after startup are slower (kernel warm-up). The checkpoint is the publisher's BF16 release under OpenMDW-1.1.
