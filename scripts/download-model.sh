#!/usr/bin/env bash
# Download the frozen FP8 checkpoint into models/ and verify every shard hash.
#
#   scripts/download-model.sh            # resume or verify an existing copy
#   scripts/download-model.sh --verify   # hash only, no network
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

VERIFY_ONLY=0
[[ "${1:-}" == "--verify" ]] && VERIFY_ONLY=1

mkdir -p "$MODEL_PATH" state
manifest="$(python3 "${ROOT}/scripts/manifest.py" "$MODEL_REPO" "$MODEL_REV")"
[[ -n "$manifest" ]] || die "Empty manifest for ${MODEL_REPO}@${MODEL_REV}"

while IFS=$'\t' read -r path size oid; do
  [[ -n "$path" ]] || continue
  destination="${MODEL_PATH}/${path}"
  mkdir -p "$(dirname "$destination")"
  if [[ -f "$destination" && "$(stat -c %s "$destination")" == "$size" ]]; then
    log "present ${path}"
    continue
  fi
  if (( VERIFY_ONLY )); then
    die "${path} is missing or has the wrong size"
  fi
  log "downloading ${path} ($(numfmt --to=iec --suffix=B "$size"))"
  curl -fL --retry 5 --retry-delay 5 -C - \
    "https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REV}/${path}" \
    -o "$destination"
  actual="$(stat -c %s "$destination")"
  [[ "$actual" == "$size" ]] || die "${path}: expected ${size} bytes, got ${actual}"
done <<< "$manifest"

log "Hashing every file and comparing LFS object ids"
manifest="$manifest" python3 - "$MODEL_REPO" "$MODEL_REV" "$MODEL_PATH" "${ROOT}/state/model.json" <<'PY'
import hashlib, json, os, sys, time
repo, revision, directory, state_path = sys.argv[1:5]
files = []
for line in os.environ["manifest"].splitlines():
    if not line.strip():
        continue
    path, size, oid = line.split("\t")
    target = os.path.join(directory, path)
    digest = hashlib.sha256()
    with open(target, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if oid and actual != oid:
        raise SystemExit(f"{path}: SHA-256 {actual} != {oid}")
    files.append({"path": path, "size_bytes": int(size), "sha256": actual,
                  "lfs_verified": bool(oid)})
record = {
    "schema_version": 1,
    "kind": "model",
    "repo": repo,
    "revision": revision,
    "local_dir": os.path.relpath(directory),
    "file_count": len(files),
    "total_bytes": sum(item["size_bytes"] for item in files),
    "lfs_verified_count": sum(1 for item in files if item["lfs_verified"]),
    "files": files,
    "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(state_path, "w") as stream:
    json.dump(record, stream, indent=2)
    stream.write("\n")
print(json.dumps({"files": len(files), "total_gib": round(record["total_bytes"] / 2**30, 2),
                  "lfs_verified": record["lfs_verified_count"], "state": state_path}, indent=2))
PY
log "Model ready at ${MODEL_DIR}"
