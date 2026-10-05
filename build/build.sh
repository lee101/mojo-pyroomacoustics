#!/usr/bin/env bash
# Build the single compilation unit into dist/libmojo-pyroomacoustics.so.
#
# src/ported.mojo holds every kernel, so build cost stays fixed (~6s) no matter
# how many functions are added: this is the "one compilation unit where
# practical" rule from the layout spec.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/dist"

mojo build --emit shared-lib \
    "$repo_dir/src/ported.mojo" \
    -o "$repo_dir/dist/libmojo-pyroomacoustics.so"

echo "built $repo_dir/dist/libmojo-pyroomacoustics.so"