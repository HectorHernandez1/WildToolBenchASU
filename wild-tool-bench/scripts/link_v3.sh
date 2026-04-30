#!/usr/bin/env bash
# link_v3.sh — Mirror result/wtbmas:<model>/ → result_v3/<model>/ as symlinks
#              and similarly for score/. Idempotent. Safe to run mid-chain.
#
# Why this exists: the eval pipeline writes WTB-MAS outputs to
#   result/wtbmas:<model>/  and  score/wtbmas:<model>/
# Those colon-prefixed directory names are awkward and don't match the
# baseline (v1) / eval-improvements (v2) layout. This script creates
# clean v3 mirrors:
#   result_v3/<model>/  →  ../result/wtbmas:<model>/
#   score_v3/<model>/   →  ../score/wtbmas:<model>/
#
# So readers can browse v1/v2/v3 in parallel:
#   score/<model>/        — v1 baseline
#   score_v2/<model>/     — v2 eval-improvements
#   score_v3/<model>/     — v3 WTB-MAS  (this script's output)
#
# Usage:
#   bash scripts/link_v3.sh                        # link all 4 backbones
#   bash scripts/link_v3.sh qwen3:8b qwen3:14b     # subset

set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=(qwen3:8b qwen3:14b qwen3:32b gemma4:31b)
fi

mkdir -p result_v3 score_v3

linked=0
skipped=0
for model in "${MODELS[@]}"; do
    for kind in result score; do
        src="${kind}/wtbmas:${model}"
        dst="${kind}_v3/${model}"

        # Source must exist
        if [ ! -d "$src" ]; then
            echo "  [skip] $src does not exist yet"
            skipped=$((skipped + 1))
            continue
        fi

        # Skip if already linked (idempotent)
        if [ -L "$dst" ] && [ "$(readlink "$dst")" = "../$src" ]; then
            echo "  [ok]   $dst → ../$src (already linked)"
            continue
        fi

        # Don't clobber a regular directory
        if [ -d "$dst" ] && [ ! -L "$dst" ]; then
            echo "  [warn] $dst exists as a directory (not a symlink); leaving it alone"
            skipped=$((skipped + 1))
            continue
        fi

        # Create or refresh the symlink
        rm -f "$dst"
        # Make the relative-up symlink so it works regardless of where the
        # repo is checked out:  result_v3/qwen3:8b → ../result/wtbmas:qwen3:8b
        mkdir -p "$(dirname "$dst")"
        ln -s "../$src" "$dst"
        echo "  [link] $dst → ../$src"
        linked=$((linked + 1))
    done
done

echo ""
echo "Done. linked=$linked  skipped=$skipped"
