#!/bin/bash
# Download the Arimo font used for the figures into analysis/workflows/fonts/.
#
# The font files are NOT stored in this repository -- they are Google's to
# distribute, not ours. This fetches them from the upstream project (SIL Open
# Font License 1.1) along with the license text. Figures render without them
# via the metric-compatible fallbacks in analysis/src/plots.py; running this
# just reproduces the published text metrics exactly.
set -euo pipefail

DEST="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)/workflows/fonts"
BASE="https://raw.githubusercontent.com/googlefonts/Arimo/main"

mkdir -p "$DEST"
for f in fonts/ttf/Arimo-Regular.ttf fonts/ttf/Arimo-Bold.ttf OFL.txt; do
  echo "Fetching $(basename "$f") ..."
  curl -fsSL "$BASE/$f" -o "$DEST/$(basename "$f")"
done

echo "Arimo installed in $DEST"
