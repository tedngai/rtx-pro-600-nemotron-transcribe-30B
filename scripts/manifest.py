#!/usr/bin/env python3
"""Emit a tab-separated download manifest for a pinned Hugging Face revision.

Columns: path, size in bytes, LFS SHA-256 (empty when the file is not LFS).
Standard library only; the caller owns retries and verification.
"""

import json
import sys
import urllib.request

KEEP = (".json", ".safetensors", ".jinja", ".model")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: manifest.py <repo> <revision>")
    repo, revision = sys.argv[1:3]
    url = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        rows = json.load(response)
    for row in sorted(rows, key=lambda item: item["path"]):
        if row.get("type") != "file" or not row["path"].endswith(KEEP):
            continue
        size = row.get("size")
        if not isinstance(size, int):
            raise SystemExit(f"{row['path']}: manifest has no size")
        oid = (row.get("lfs") or {}).get("oid", "")
        print(f"{row['path']}\t{size}\t{oid}")


if __name__ == "__main__":
    main()
