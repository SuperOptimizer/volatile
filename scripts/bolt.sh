#!/usr/bin/env bash
# scripts/bolt.sh
# Applies BOLT (Binary Optimization and Layout Tool) to a compiled binary.
#
# BOLT uses profile data recorded at runtime to reorder hot functions and
# basic blocks for improved instruction-cache utilisation.
#
# Usage:
#   ./scripts/bolt.sh <binary> [output_binary] [perf_data_or_fdata]
#
# Arguments:
#   binary            Path to the ELF binary to optimise.
#   output_binary     Destination path for the BOLT-optimised binary.
#                     Defaults to <binary>.bolt
#   perf_data_or_fdata
#                     Either a Linux perf.data file (converted internally via
#                     perf2bolt) or a pre-converted BOLT .fdata file.
#                     If omitted the binary is optimised without profile data
#                     using basic layout heuristics (still useful).
#
# Requirements:
#   llvm-bolt and optionally perf2bolt in PATH.
#   The binary must have been built without stripping (keep at least function
#   symbols; full debug info is not required).
#
# Example (with perf profile):
#   perf record -e cycles:u -j any,u -- ./volatile_server --benchmark
#   ./scripts/bolt.sh build/src/server/volatile_server \
#                     build/src/server/volatile_server.bolt \
#                     perf.data
#
# Example (layout-only, no profile):
#   ./scripts/bolt.sh build/src/server/volatile_server

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <binary> [output_binary] [perf_data_or_fdata]" >&2
  exit 1
fi

BINARY="${1}"
OUTPUT="${2:-${BINARY}.bolt}"
PROFILE_INPUT="${3:-}"

if [[ ! -f "${BINARY}" ]]; then
  echo "Error: binary not found: ${BINARY}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Locate required tools
# ---------------------------------------------------------------------------
LLVM_BOLT="$(command -v llvm-bolt 2>/dev/null || true)"
if [[ -z "${LLVM_BOLT}" ]]; then
  echo "Error: llvm-bolt not found in PATH. Install the LLVM toolchain." >&2
  exit 1
fi

PERF2BOLT="$(command -v perf2bolt 2>/dev/null || true)"

# ---------------------------------------------------------------------------
# Convert perf.data -> .fdata if necessary
# ---------------------------------------------------------------------------
FDATA=""
if [[ -n "${PROFILE_INPUT}" ]]; then
  if [[ "${PROFILE_INPUT}" == *.fdata ]]; then
    FDATA="${PROFILE_INPUT}"
    echo "==> Using pre-converted BOLT profile: ${FDATA}"
  else
    # Assume it is a perf.data file; convert with perf2bolt.
    if [[ -z "${PERF2BOLT}" ]]; then
      echo "Error: perf2bolt not found in PATH (needed to convert perf.data)." >&2
      exit 1
    fi
    FDATA="${PROFILE_INPUT%.data}.fdata"
    echo "==> Converting perf.data to BOLT profile: ${FDATA}"
    "${PERF2BOLT}" \
      -p "${PROFILE_INPUT}" \
      -o "${FDATA}" \
      "${BINARY}"
  fi
fi

# ---------------------------------------------------------------------------
# Run BOLT
# ---------------------------------------------------------------------------
BOLT_ARGS=(
  "${BINARY}"
  --dyno-stats                        # print before/after stats
  --eliminate-unreachable             # remove dead code
  --frame-opt=all                     # optimise stack frames
  --icf=1                             # identical-code folding
  --indirect-call-promotion=all       # devirtualise indirect calls
  --jump-tables=aggressive            # optimise jump table layout
  --plt=all                           # optimise PLT stubs
  --reorder-blocks=ext-tsp            # reorder basic blocks (TSP heuristic)
  --reorder-functions=hfsort+         # reorder functions by call frequency
  --split-all-cold                    # split cold code into a separate section
  --split-eh                          # split exception-handling code
  --use-gnu-stack                     # preserve GNU_STACK segment
  -o "${OUTPUT}"
)

if [[ -n "${FDATA}" ]]; then
  BOLT_ARGS+=(--data "${FDATA}")
  echo "==> Running BOLT with profile data..."
else
  echo "==> Running BOLT without profile data (layout-only mode)..."
fi

echo "    Input:  ${BINARY}"
echo "    Output: ${OUTPUT}"
echo ""

"${LLVM_BOLT}" "${BOLT_ARGS[@]}"

echo ""
echo "BOLT optimisation complete."
echo "Optimised binary: ${OUTPUT}"
echo ""
echo "To verify improvement, compare:"
echo "  perf stat ${BINARY} <args>"
echo "  perf stat ${OUTPUT} <args>"
