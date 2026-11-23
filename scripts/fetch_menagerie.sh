#!/usr/bin/env bash
# Fetch the mujoco_menagerie models used by examples/g1_walk.py and examples/genesis_apollo.py into
# third_party/mujoco_menagerie (gitignored). Idempotent: re-running only
# re-applies the sparse-checkout set.
set -euo pipefail
DEST="$(cd "$(dirname "$0")/.." && pwd)/third_party/mujoco_menagerie"
if [ ! -d "$DEST/.git" ]; then
  mkdir -p "$(dirname "$DEST")"
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git "$DEST"
fi
git -C "$DEST" sparse-checkout set unitree_g1 realsense_d435i apptronik_apollo
ls "$DEST/unitree_g1/g1.xml" "$DEST/realsense_d435i/d435i.xml" "$DEST/apptronik_apollo/scene.xml"
