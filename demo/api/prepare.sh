#!/usr/bin/env bash
# Copies the only files the image needs from the repo root into this build context.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "$HERE/../../agent.py" "$HERE/../../LICENSE" "$HERE/"
