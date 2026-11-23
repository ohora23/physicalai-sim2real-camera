#!/usr/bin/env bash
# Fetch the WIRobotics ALLEX simulation model used by examples/genesis_allex.py into
# third_party/allex_model (gitignored), pinned to the commit the Genesis results were measured on.
# The meshes are (c) WIRobotics and not redistributable (MESHES-LICENSE): never commit them.
# Idempotent: re-running only re-checks-out the pinned commit.
set -euo pipefail
REV=5e120529b106902898382fbd3eb9cec3188deaa2  # v0.2.0 pre-release, 2026-07-03
DEST="$(cd "$(dirname "$0")/.." && pwd)/third_party/allex_model"
if [ ! -d "$DEST/.git" ]; then
  mkdir -p "$(dirname "$DEST")"
  git clone https://github.com/wirobotics-rih/allex_model.git "$DEST"
fi
git -C "$DEST" fetch -q origin "$REV" 2>/dev/null || true
git -C "$DEST" -c advice.detachedHead=false checkout -q "$REV"
ls "$DEST/mjcf/scene.xml" "$DEST/examples/motions/demo1.npz"
