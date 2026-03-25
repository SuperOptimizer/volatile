#!/usr/bin/env bash
# scripts/pgo.sh
# Automates the two-phase Profile-Guided Optimization (PGO) workflow:
#   Phase 1 — build with instrumentation, run tests to collect profiles.
#   Phase 2 — rebuild with the collected profiles to produce an optimised binary.
#
# Usage:
#   ./scripts/pgo.sh [build_dir] [cmake_extra_args...]
#
# Example:
#   ./scripts/pgo.sh build-pgo -DCMAKE_BUILD_TYPE=Release
#
# Requirements:
#   cmake, ninja (or make), and either gcc or clang in PATH.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(dirname "$SCRIPT_DIR")"

BUILD_DIR="${1:-${SOURCE_DIR}/build-pgo}"
shift 1 2>/dev/null || true          # remaining args forwarded to cmake
EXTRA_ARGS=("$@")

PROFILE_DIR="${BUILD_DIR}/pgo-profiles"

# Detect whether Ninja is available (preferred for speed).
if command -v ninja &>/dev/null; then
  GENERATOR="Ninja"
else
  GENERATOR="Unix Makefiles"
fi

# ---------------------------------------------------------------------------
# Phase 1: Instrumented build
# ---------------------------------------------------------------------------
echo "==> [PGO Phase 1] Configuring instrumented build in ${BUILD_DIR}/phase1"

cmake -G "${GENERATOR}" \
  -S "${SOURCE_DIR}" \
  -B "${BUILD_DIR}/phase1" \
  -DCMAKE_BUILD_TYPE=Release \
  -DVOLATILE_OPT_PGO_GENERATE=ON \
  -DVOLATILE_PGO_PROFILE_DIR="${PROFILE_DIR}" \
  -DVOLATILE_BUILD_TESTS=ON \
  "${EXTRA_ARGS[@]}"

echo "==> [PGO Phase 1] Building..."
cmake --build "${BUILD_DIR}/phase1" --parallel

echo "==> [PGO Phase 1] Running test suite to collect profiles..."
# Run CTest; failures are reported but don't abort the PGO flow because even
# partial profile coverage is better than none.
ctest --test-dir "${BUILD_DIR}/phase1" --output-on-failure || true

# Clang requires merging raw profile files into a single .profdata before use.
if command -v llvm-profdata &>/dev/null; then
  echo "==> [PGO Phase 1] Merging Clang profiles with llvm-profdata..."
  llvm-profdata merge \
    --output="${PROFILE_DIR}/default.profdata" \
    "${PROFILE_DIR}"/*.profraw 2>/dev/null || \
  llvm-profdata merge \
    --output="${PROFILE_DIR}/default.profdata" \
    "${PROFILE_DIR}" || true
  echo "    Merged profile: ${PROFILE_DIR}/default.profdata"
fi

# ---------------------------------------------------------------------------
# Phase 2: Optimised build
# ---------------------------------------------------------------------------
echo "==> [PGO Phase 2] Configuring optimised build in ${BUILD_DIR}/phase2"

cmake -G "${GENERATOR}" \
  -S "${SOURCE_DIR}" \
  -B "${BUILD_DIR}/phase2" \
  -DCMAKE_BUILD_TYPE=Release \
  -DVOLATILE_OPT_PGO_USE=ON \
  -DVOLATILE_PGO_PROFILE_DIR="${PROFILE_DIR}" \
  "${EXTRA_ARGS[@]}"

echo "==> [PGO Phase 2] Building..."
cmake --build "${BUILD_DIR}/phase2" --parallel

echo ""
echo "PGO build complete. Optimised binaries are in ${BUILD_DIR}/phase2."
