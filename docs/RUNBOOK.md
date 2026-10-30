# Runbook

## Lifecycle

```bash
scripts/download-model.sh          # resume/verify 62.9 GiB of BF16 weights
docker build -t nemotron-transcribe:v0.32.0-nemo docker/
scripts/start.sh                   # create or start the container, wait for /health
scripts/status.sh                  # container state, model card
scripts/stop.sh                    # stop, preserve the container
scripts/stop.sh --remove           # stop and delete the container
```

`scripts/start.sh` is idempotent: an existing running container is reused; a
stopped one is started. If the container exists with stale flags, remove it and
start again.

## What start does

1. Resolves Docker access (direct, then `sg docker`, then passwordless sudo).
2. Checks the NVIDIA runtime and that the pinned BF16 checkpoint is verified
   (`state/model.json` written by `download-model.sh`).
3. Refuses to start if port 8013 is held by something else.
4. Runs `vllm serve` with the model card's full flag set (MTP8 speculative
   decoding, ReplaySSM with `--mamba-cache-mode none`, prefix caching off,
   CTC word-timestamp middleware) and waits up to 60 minutes for `/health`.

Expected startup: 63 GiB of BF16 weights from a warm page cache plus CUDA graph
capture; the first start also compiles NeMo/flashinfer kernels and can take
noticeably longer than later starts.

## Screens

```bash
python3 -B tests/smoke_test.py --label smoke-interface --execute --confirm-idle
python3 -B tests/benchmark.py --label rtfx-c1 --execute --confirm-idle --concurrency 1 --rounds 3
python3 -B tests/benchmark.py --label rtfx-c8 --execute --confirm-idle --concurrency 8 --rounds 3
```

Both scripts refuse to send inference without `--execute --confirm-idle`.
Screens never restart the server; prefix caching is disabled for ReplaySSM, so
rounds are independent by construction.

## Direct transcription calls

```bash
python3 - <<'PY'
import base64, json, urllib.request
audio = base64.b64encode(open("tests/fixtures/en-Alice_woman.wav", "rb").read()).decode()
body = {
    "model": "pratt-transcribe",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Audio transcription. Output: transcript only."},
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio}"}},
    ]}],
    "temperature": 0,
}
req = urllib.request.Request("http://127.0.0.1:8013/v1/chat/completions",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"])
PY
```

Word timestamps are available through the CTC middleware; pass timestamp
request options per the model card (`--trust-request-mm-kwargs` is enabled).

## Coexistence on this host

| Port | Tenant |
| --- | --- |
| 8000 | d3-systemone vLLM (GPU 1) — restart=no, restart manually if the daemon reboots |
| 8012 | nanbeige4.3-33b workspace (same MIG slice; only one of the two may run) |
| 8013 | this workspace |
| 8188 | ComfyUI (host process) |

This container and the nanbeige container share the GPU 0 MIG slice
(`MIG-b12c2fdf-…`). Starting one while the other runs will fail on GPU memory;
stop the other first.

## Known build notes

- ghcr.io is unreachable from this host: `uv` is installed from PyPI in the
  image, and no image stages pull from ghcr.
- The NeMo Speech repo pins `.python-version` to 3.12; the image sets
  `UV_PYTHON=3.13` to keep the model card's recipe, and
  `UV_PYTHON_INSTALL_DIR=/opt/uv/python` so the non-root runtime uid can reach
  the interpreter (`/root` is 0700 in ubuntu:22.04).
- The runtime image needs `build-essential` plus `CUDA_HOME` pointing at torch's
  bundled `nvidia/cu13` toolkit: Triton compiles its CUDA driver module at
  import and torch.compile/Inductor builds the encoder's flex_attention kernels.
  A slim base image crash-loops during `profile_run` without it.
- FlashInfer kernels come from the precompiled `flashinfer-jit-cache`
  `0.7.0.post1+cu130` wheel (`https://flashinfer.ai/whl/cu130`), while mamba SSU
  kernels JIT-compile with the matching CUDA 13.0 nvcc from NVIDIA's apt repo.
  JIT with torch's bundled nvcc 13.4 fails: it is paired with CUDA 13.0 headers,
  which FlashInfer's bundled CCCL rejects as incompatible.
- vLLM's FlashInfer warmup autotune profiles tactics after the KV cache has
  filled the slice to `gpu-memory-utilization 0.94` (~2 GiB free) and OOMs
  (surfacing as a `MemoryError: NVML_SUCCESS == r INTERNAL ASSERT` from torch's
  allocator on MIG). `start.sh` passes
  `--kernel-config '{"enable_flashinfer_autotune": false}'` so FlashInfer uses
  heuristic tactics instead; kernels and backend selection are unchanged.
- A daemon-side pull left running after a killed client can wedge the docker
  daemon silently (builds and `docker create` hang with no log output).
  Restarting dockerd clears it; d3-systemone has `restart=no` and must be
  started again afterwards.
