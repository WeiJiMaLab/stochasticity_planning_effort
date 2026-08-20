#!/usr/bin/env bash
# Download experimental data, CV splits, and fit outputs from the project's OSF storage
# (https://osf.io/vyh8u) and extract them into analysis/data, analysis/data_split, analysis/fit.
#
# Looks up each archive's current OSF file ID by name via the OSF API instead of a
# hardcoded guid: re-uploading a file (delete + upload, rather than a new version of the
# same file) gives it a new ID, which silently breaks any hardcoded link.
#
# Usage:
#   ./0_downloaddata.sh              # download + extract all three archives
#   ./0_downloaddata.sh data fit     # only download + extract selected archives
#
# The OSF project (DOI 10.17605/OSF.IO/VYH8U) is public, so no credentials are
# needed. If you are working against a private fork, export OSF_VIEW_ONLY_KEY
# with that project's view-only key.
set -euo pipefail

VIEW_ONLY_KEY="${OSF_VIEW_ONLY_KEY:-}"
NODE_ID="${OSF_NODE_ID:-vyh8u}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

declare -A DEST_DIR=(
  [data]="$ANALYSIS_DIR/data"
  [fit]="$ANALYSIS_DIR/fit"
  [data_split]="$ANALYSIS_DIR/data_split"
)

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
  TARGETS=(data fit data_split)
fi

for name in "${TARGETS[@]}"; do
  if [[ -z "${DEST_DIR[$name]:-}" ]]; then
    echo "Unknown target '$name' (expected: data, fit, data_split)" >&2
    exit 1
  fi
done

LISTING_URL="https://api.osf.io/v2/nodes/${NODE_ID}/files/osfstorage/"
if [[ -n "$VIEW_ONLY_KEY" ]]; then
  LISTING_URL="${LISTING_URL}?view_only=${VIEW_ONLY_KEY}"
fi

echo "==> Looking up current file IDs on OSF (project ${NODE_ID})"
LISTING_JSON="$(curl -sS --fail --max-time 30 "$LISTING_URL")"

for name in "${TARGETS[@]}"; do
  filename="${name}.tar.gz"

  url="$(python3 -c "
import json, sys
data = json.loads(sys.argv[1])
for item in data['data']:
    if item['attributes']['name'] == sys.argv[2]:
        print(item['links']['download'])
        break
" "$LISTING_JSON" "$filename")"

  if [[ -z "$url" ]]; then
    echo "Could not find '$filename' in the OSF project's storage listing." >&2
    echo "Check that it was uploaded to the root of https://osf.io/${NODE_ID}." >&2
    echo "If that project is private, export OSF_VIEW_ONLY_KEY with its view-only key." >&2
    exit 1
  fi

  archive="$SCRIPT_DIR/${filename}"

  echo "==> Downloading ${filename} from OSF"
  curl -L --fail --progress-bar -o "$archive" "$url"

  dest="${DEST_DIR[$name]}"
  mkdir -p "$dest"
  echo "==> Extracting ${filename} into $dest"
  tar -xzf "$archive" -C "$dest" --strip-components=1 --exclude='._*'
  rm "$archive"
done

echo "Done."
