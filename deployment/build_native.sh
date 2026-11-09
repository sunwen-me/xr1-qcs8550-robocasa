#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CXX:?Set CXX to an AArch64 cross compiler}"
: "${QNN_INCLUDE:?Set QNN_INCLUDE to the QAIRT/QNN include directory}"
OUTPUT="${1:-$ROOT/deployment/native/qnn_native_server_xr1}"
SOURCE="${SOURCE:-$ROOT/workbench/qnn_native_server.cpp}"
MAX_GLIBC="${MAX_GLIBC:-2.35}"
mkdir -p "$(dirname "$OUTPUT")"
args=(-std=c++17 -O2 -pipe -static-libstdc++ -static-libgcc -I"$QNN_INCLUDE")
if [[ -n "${SYSROOT:-}" ]]; then args+=(--sysroot="$SYSROOT"); fi
args+=("$SOURCE" -ldl -o "$OUTPUT")
"$CXX" "${args[@]}"
file "$OUTPUT"
sha256sum "$OUTPUT"
if command -v readelf >/dev/null 2>&1; then
  needed="$(readelf -V "$OUTPUT" | grep -oE 'GLIBC_[0-9]+\.[0-9]+' | sed 's/GLIBC_//' | sort -Vu | tail -1)"
  if [[ -n "$needed" && "$(printf '%s\n%s\n' "$needed" "$MAX_GLIBC" | sort -V | tail -1)" != "$MAX_GLIBC" ]]; then
    echo "ERROR: GLIBC_$needed exceeds board limit GLIBC_$MAX_GLIBC" >&2
    exit 3
  fi
fi
