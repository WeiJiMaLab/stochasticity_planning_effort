#!/usr/bin/env bash
# Deploy the public task demo to Firebase Hosting.
#
# This deploys Hosting only. The demo URLs (?type=demoR|demoV|demoT) run
# entirely in the browser and write no data, so the project needs no
# Realtime Database and no database credentials.
#
# The target project is read from .firebaserc, which is not tracked in this
# repository. Copy .firebaserc.sample to .firebaserc and set your own Firebase
# project ID, or pass --project <id> to deploy elsewhere.
#
# One-time setup:
#   1. npm install -g firebase-tools
#   2. firebase login
#
# Then, from the repository root:
#   bash experiment/deploy_demo.sh

set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v firebase >/dev/null 2>&1; then
  echo "error: firebase-tools not found. Install with: npm install -g firebase-tools" >&2
  exit 1
fi

if [ ! -f .firebaserc ]; then
  echo "error: .firebaserc not found. Copy .firebaserc.sample and set your project ID." >&2
  exit 1
fi

# Pass through any extra args (e.g. --project <id>).
firebase deploy --only hosting "$@"

echo
echo "Deployed. Demo URLs (append to your hosting domain):"
echo "  /?type=demoR   # Reliability"
echo "  /?type=demoV   # Volatility"
echo "  /?type=demoT   # Controllability"
