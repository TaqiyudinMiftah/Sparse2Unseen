#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/external/MPCount"
if [[ -d "$DEST/.git" ]]; then
  echo "MPCount already exists at $DEST"
  exit 0
fi
git clone https://github.com/Shimmer93/MPCount.git "$DEST"
echo "Cloned MPCount into $DEST"
