#!/usr/bin/env bash
# Start the vLLM serving container and wait for /health.
#
#   scripts/start.sh                 # create, or start the preserved container
#   BIND_ADDR=0.0.0.0 scripts/start.sh
#
# Flags follow the model card's "Start the Server" recipe (BF16, MTP8 spec
# decoding, ReplaySSM, word-timestamp middleware). Do not enable prefix
# caching; the card requires it off for ReplaySSM.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

resolve_docker
require_gpu_runtime
require_model

"${DOCKER[@]}" image inspect "$IMAGE" >/dev/null 2>&1 \
  || die "Image ${IMAGE} is missing. Build it: docker build -t ${IMAGE} docker/"

auth_args=()
if [[ "$REQUIRE_API_KEY" == "1" ]]; then
  auth_args=(--api-key "$(api_key_value)")
fi

if container_exists; then
  if container_running; then
    log "Container ${CONTAINER} is already running"
  else
    log "Starting the preserved container ${CONTAINER}"
    "${DOCKER[@]}" start "$CONTAINER" >/dev/null
  fi
  if wait_for_health "${START_TIMEOUT:-3600}"; then
    log "Ready at ${BASE_URL}"
    exit 0
  fi
  "${DOCKER[@]}" logs --tail 80 "$CONTAINER" || true
  die "Container started but /health never became ready"
fi

if curl -fsS --max-time 2 "${BASE_URL}/health" >/dev/null 2>&1; then
  die "Something already answers ${BASE_URL}/health; stop it before starting ${CONTAINER}"
fi
if ss -ltn "( sport = :${PORT} )" 2>/dev/null | grep -q ":${PORT}"; then
  die "Port ${PORT} is already in use"
fi

args=(
  serve
  "/models/$(basename "$MODEL_DIR")"
  --served-model-name "$SERVED_NAME"
  --host 0.0.0.0
  --port 8000
  --trust-remote-code
  --dtype bfloat16
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --enable-chunked-prefill
  --no-enable-prefix-caching
  --use-replayssm
  --mamba-backend flashinfer
  --mamba-cache-mode none
  --mm-processor-cache-gb 0
  --no-async-scheduling
  --limit-mm-per-prompt '{"audio":{"count":6,"length":120000000}}'
  --speculative-config '{"method":"mtp","num_speculative_tokens":8}'
  --reasoning-parser nemotron_v3
  --trust-request-mm-kwargs
  --middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware
  --no-enable-log-requests
  --uvicorn-log-level warning
  "${auth_args[@]}"
)

log "Starting ${CONTAINER} from ${IMAGE} on ${BIND_ADDR}:${PORT} -> :8000 (GPU: ${GPU_DEVICE})"
"${DOCKER[@]}" run -d --name "$CONTAINER" \
  --restart "$RESTART_POLICY" \
  --gpus "\"device=${GPU_DEVICE}\"" --ipc=host \
  --user "$(id -u):$(id -g)" \
  -p "${BIND_ADDR}:${PORT}:8000" \
  -v "$ROOT/models:/models:ro" \
  "$IMAGE" "${args[@]}" >/dev/null

log "Waiting for /health (loading 63 GiB of weights can take several minutes)"
if wait_for_health "${START_TIMEOUT:-3600}"; then
  log "Ready at ${BASE_URL}"
  python3 - "$BASE_URL" <<'PY'
import json, sys, urllib.request
base = sys.argv[1]
card = json.load(urllib.request.urlopen(base + "/v1/models", timeout=10))["data"][0]
print(json.dumps({"id": card["id"], "root": card.get("root"),
                  "max_model_len": card.get("max_model_len")}, indent=2))
PY
  exit 0
fi

"${DOCKER[@]}" logs --tail 120 "$CONTAINER" || true
die "Container did not become healthy. Inspect: ${DOCKER[*]} logs ${CONTAINER}"
