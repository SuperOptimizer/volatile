#!/bin/bash
# Test S3 streaming + local caching + rechunking pipeline
# Usage: ./scripts/test_s3_mirror.sh <s3_url> [cache_dir]
#
# Before running, set your AWS credentials:
#   export AWS_ACCESS_KEY_ID=<your key>
#   export AWS_SECRET_ACCESS_KEY=<your secret>
#   export AWS_SESSION_TOKEN=<optional session token>
#   export AWS_REGION=<region, default us-east-1>
#   export AWS_ENDPOINT_URL=<optional custom endpoint>
#
# Example:
#   export AWS_ACCESS_KEY_ID=AKIA...
#   export AWS_SECRET_ACCESS_KEY=...
#   ./scripts/test_s3_mirror.sh s3://vesuvius-data/volumes/20230205180739/
#
# Or with an HTTP URL (no auth needed for public data):
#   ./scripts/test_s3_mirror.sh https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes/20230205180739/

set -euo pipefail

URL="${1:?Usage: $0 <s3_or_http_url> [cache_dir]}"
CACHE_DIR="${2:-$HOME/.cache/volatile}"
CLI="./build/src/cli/volatile-cli"

echo "=== Volatile S3/HTTP Mirror Pipeline ==="
echo "URL: $URL"
echo "Cache: $CACHE_DIR"
echo ""

# check creds for s3
if [[ "$URL" == s3://* ]]; then
  if [ -z "${AWS_ACCESS_KEY_ID:-}" ]; then
    echo "ERROR: AWS_ACCESS_KEY_ID not set"
    echo "Set your AWS credentials:"
    echo "  export AWS_ACCESS_KEY_ID=<key>"
    echo "  export AWS_SECRET_ACCESS_KEY=<secret>"
    echo "  export AWS_SESSION_TOKEN=<token>  # optional"
    echo "  export AWS_REGION=<region>         # optional, default us-east-1"
    exit 1
  fi
  echo "AWS creds: key=${AWS_ACCESS_KEY_ID:0:8}... region=${AWS_REGION:-us-east-1}"
fi

# step 1: info
echo ""
echo "--- Step 1: Volume Info ---"
$CLI info "$URL" 2>&1 || echo "(info failed - may need auth or network)"

# step 2: sample a voxel
echo ""
echo "--- Step 2: Sample voxel at (500, 500, 500) ---"
$CLI sample "$URL" 500 500 500 2>&1 || echo "(sample failed)"

# step 3: mirror to local cache
echo ""
echo "--- Step 3: Mirror to local cache ---"
$CLI mirror "$URL" --cache-dir "$CACHE_DIR" --level 5 2>&1 || echo "(mirror failed)"

# step 4: stats on cached data
echo ""
echo "--- Step 4: Stats on cached data ---"
if [ -d "$CACHE_DIR" ]; then
  CACHED=$(find "$CACHE_DIR" -name "*.zarray" -o -name "zarr.json" | head -1)
  if [ -n "$CACHED" ]; then
    ZARR_DIR=$(dirname "$CACHED")
    $CLI stats "$ZARR_DIR" 2>&1 || echo "(stats failed)"
  else
    echo "No cached zarr found yet"
  fi
fi

# step 5: rechunk (if data is cached)
echo ""
echo "--- Step 5: Rechunk to 64,64,64 ---"
if [ -d "$CACHE_DIR" ]; then
  $CLI rechunk "$CACHE_DIR" --output "${CACHE_DIR}_rechunked" --chunk-size 64,64,64 2>&1 || echo "(rechunk failed or not enough cached data)"
fi

# step 6: compress4d
echo ""
echo "--- Step 6: Compress with compress4d ---"
if [ -d "$CACHE_DIR" ]; then
  $CLI compress4d "$CACHE_DIR" --output "${CACHE_DIR}.c4d" --stats 2>&1 || echo "(compress4d failed or not enough cached data)"
fi

echo ""
echo "=== Pipeline complete ==="
echo "Cached data: $CACHE_DIR"
echo "Rechunked: ${CACHE_DIR}_rechunked (if step 5 succeeded)"
echo "Compressed: ${CACHE_DIR}.c4d (if step 6 succeeded)"
du -sh "$CACHE_DIR" 2>/dev/null || true
