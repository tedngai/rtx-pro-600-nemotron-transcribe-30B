# Results

Screening run on 2026-10-30, GPU 0 MIG slice (`MIG-b12c2fdf-…`, 4g.96gb, ~96 GiB),
BF16 checkpoint @ `a99e33f…`, vLLM 0.32.0 + NeMo Speech @ `548fd86…` with the
card's flag set. `d3-systemone` (GPU 1) and ComfyUI were running throughout.

## Interface smoke test

`evidence/smoke-interface-20261030T040131.json`

| Check | Result |
| --- | --- |
| `/v1/models` contract | `pratt-transcribe`, 65,536 ctx — pass |
| Cold transcription (9.27 s clip) | 40.3 s (first request: kernel/CUDA-graph warm-up), transcript correct — pass |
| Repeat transcription | 0.203 s, byte-identical transcript — pass |

Transcript of `en-Alice_woman.wav`:

> So just to clarify, we've had nineteen to twenty year olds on our podcast so far, and if you don't mind me asking, how old are you?

## RTFx screens

RTFx = audio-seconds processed per wall-second. Three rounds each, median.
`x10` fixture = the same 9.27 s clip concatenated 10× (92.70 s).

| Screen | Fixture | Concurrency | Aggregate RTFx | Per-request RTFx | Latency |
| --- | --- | --- | --- | --- | --- |
| `rtfx-c1` | 9.27 s | 1 | **48.1×** | 48.6× | 0.19 s |
| `rtfx-c8` | 9.27 s | 8 | **199.0×** | 25.2× | 0.34 s |
| `rtfx-c1-long` | 92.70 s | 1 | **93.9×** | 95.2× | 0.98 s |
| `rtfx-c8-long` | 92.70 s | 8 | **371.1×** | 48.8× | 1.81 s |

Reports: `evidence/rtfx-c1-20261030T040137.json`,
`evidence/rtfx-c8-20261030T040141.json`,
`evidence/rtfx-c1-long-20261030T040157.json`,
`evidence/rtfx-c8-long-20261030T040206.json`.

## Reading the numbers

- Short clips are overhead-dominated: a 9.27 s file pays fixed per-request cost
  (audio decode, encoder pass, prefill), so per-stream RTFx is ~48–49×. The
  same server reaches ~94× per stream at 92.7 s of audio.
- Concurrency scales throughput ~4× at C8 while per-stream latency grows ~2×,
  i.e. batch efficiency is good up to 8 streams; `--max-num-seqs 192` leaves
  headroom not exercised here.
- The card's published throughput uses NVIDIA's optimized NIM kernels, which are
  not in the open-source release; the card itself expects the open stack to be
  materially slower. These numbers are screening evidence for this MIG slice,
  not a NIM comparison.
- GPU memory at steady state: ~91.7 GiB of 96 GiB (weights 62.9 GiB + KV pool at
  `--gpu-memory-utilization 0.94`).

## Deviations from the card

1. `--kernel-config '{"enable_flashinfer_autotune": false}'`: vLLM's FlashInfer
   warmup autotune runs after the KV cache fills the slice and OOMs on this MIG
   partition. FlashInfer kernels and the mamba flashinfer backend are unchanged;
   heuristic tactics are used instead of per-shape tuning.
2. Image is built locally (`docker/Dockerfile`): no upstream image exists for
   vLLM 0.32.0 + NeMo Speech main.

## What is not measured

- WER/cpWER quality; only serving behavior.
- Long recordings near the 1-hour decode limit and the 6-audio-items-per-prompt
  cap.
- Tail latency beyond three rounds and multi-hour stability.