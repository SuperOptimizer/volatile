#!/bin/bash
# coverage.sh — build with gcov instrumentation, run tests, report line coverage.
# Usage: ./scripts/coverage.sh [build-dir]
#
# If lcov is installed, also generates HTML report in <build-dir>/coverage-html/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD="${1:-${ROOT}/build-coverage}"

# ---------------------------------------------------------------------------
# 1. Configure
# ---------------------------------------------------------------------------
echo "==> Configuring with coverage flags in ${BUILD}"
cmake -B "${BUILD}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DVOLATILE_ENABLE_COVERAGE=ON \
  -DVOLATILE_BUILD_TESTS=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  "${ROOT}"

# ---------------------------------------------------------------------------
# 2. Build
# ---------------------------------------------------------------------------
echo "==> Building"
cmake --build "${BUILD}" -j"$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"

# ---------------------------------------------------------------------------
# 3. Run tests
# ---------------------------------------------------------------------------
echo "==> Running tests"
ctest --test-dir "${BUILD}" --timeout 30 --output-on-failure || true

# ---------------------------------------------------------------------------
# 4. Collect .gcda / .gcno files and run gcov per source file.
#    Cover: src/core, src/render, src/gpu, src/server
# ---------------------------------------------------------------------------
echo ""
echo "==> Per-file line coverage"
echo "-----------------------------------------------------------"
printf "%-52s %8s %8s %6s\n" "File" "Lines" "Covered" "Pct"
echo "-----------------------------------------------------------"

TOTAL_LINES=0
TOTAL_COVERED=0

for dir in src/core src/render src/gpu src/server; do
  src_dir="${ROOT}/${dir}"
  [ -d "${src_dir}" ] || continue

  for src_file in "${src_dir}"/*.c; do
    [ -f "${src_file}" ] || continue
    base="$(basename "${src_file}" .c)"

    # gcov needs the .gcda alongside the .gcno; both live under the CMake
    # object directory tree.  Find the right .gcda file.
    gcda_file="$(find "${BUILD}" -name "${base}.c.gcda" 2>/dev/null | head -1)"
    if [ -z "${gcda_file}" ]; then
      printf "%-52s %8s\n" "${dir}/${base}.c" "(not instrumented)"
      continue
    fi

    # Run gcov, capturing output.  -b -c for branch info (ignored here).
    gcov_out="$(gcov -n -o "$(dirname "${gcda_file}")" "${src_file}" 2>/dev/null)" || continue

    # Parse "Lines executed:XX.XX% of NNN" from gcov stdout.
    pct="$(echo "${gcov_out}"    | grep -oP 'Lines executed:\K[0-9.]+(?=%)' | head -1)"
    total="$(echo "${gcov_out}"  | grep -oP 'of \K[0-9]+' | head -1)"

    if [ -z "${pct}" ] || [ -z "${total}" ]; then
      printf "%-52s %8s\n" "${dir}/${base}.c" "(no data)"
      continue
    fi

    covered="$(awk "BEGIN { printf \"%d\", ${total} * ${pct} / 100 + 0.5 }")"
    printf "%-52s %8s %8s %5.1f%%\n" "${dir}/${base}.c" "${total}" "${covered}" "${pct}"

    TOTAL_LINES=$((TOTAL_LINES + total))
    TOTAL_COVERED=$((TOTAL_COVERED + covered))
  done
done

echo "-----------------------------------------------------------"
if [ "${TOTAL_LINES}" -gt 0 ]; then
  overall="$(awk "BEGIN { printf \"%.1f\", 100.0 * ${TOTAL_COVERED} / ${TOTAL_LINES} }")"
  printf "%-52s %8d %8d %5s%%\n" "TOTAL" "${TOTAL_LINES}" "${TOTAL_COVERED}" "${overall}"
else
  echo "No coverage data found.  Did the tests run?"
fi
echo "-----------------------------------------------------------"

# ---------------------------------------------------------------------------
# 5. Optional: lcov + genhtml for HTML report
# ---------------------------------------------------------------------------
if command -v lcov &>/dev/null && command -v genhtml &>/dev/null; then
  HTML_DIR="${BUILD}/coverage-html"
  INFO_FILE="${BUILD}/coverage.info"

  echo ""
  echo "==> Generating HTML report with lcov → ${HTML_DIR}"

  # Capture coverage for our source tree only.
  lcov --capture \
       --directory "${BUILD}" \
       --base-directory "${ROOT}" \
       --output-file "${INFO_FILE}" \
       --no-external \
       --quiet 2>/dev/null || true

  # Filter to just the directories we care about.
  lcov --extract "${INFO_FILE}" \
       "${ROOT}/src/core/*" \
       "${ROOT}/src/render/*" \
       "${ROOT}/src/gpu/*" \
       "${ROOT}/src/server/*" \
       --output-file "${INFO_FILE}.filtered" \
       --quiet 2>/dev/null || true

  FILTERED="${INFO_FILE}.filtered"
  [ -f "${FILTERED}" ] && [ -s "${FILTERED}" ] || FILTERED="${INFO_FILE}"

  genhtml "${FILTERED}" \
          --output-directory "${HTML_DIR}" \
          --title "volatile coverage" \
          --legend \
          --quiet 2>/dev/null || true

  if [ -f "${HTML_DIR}/index.html" ]; then
    echo "    HTML report: ${HTML_DIR}/index.html"
  else
    echo "    genhtml did not produce output (no coverage data?)."
  fi
else
  echo ""
  echo "(lcov/genhtml not found — skipping HTML report.  Install with: apt install lcov)"
fi
