#!/usr/bin/env bash
# compile_shaders.sh — compile all GLSL compute shaders to SPIR-V .spv files
#
# Usage:
#   ./scripts/compile_shaders.sh [output_dir]
#
# Default output dir: src/gpu/shaders/spv/
# Requires: glslc (part of the Vulkan SDK or shaderc package)
#
# Install on Debian/Ubuntu:
#   sudo apt-get install glslc

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SHADER_DIR="${REPO_ROOT}/src/gpu/shaders"
OUT_DIR="${1:-${SHADER_DIR}/spv}"

if ! command -v glslc &>/dev/null; then
  echo "error: glslc not found. Install it via:" >&2
  echo "  sudo apt-get install glslc            (Debian/Ubuntu)" >&2
  echo "  or download the Vulkan SDK from https://vulkan.lunarg.com/" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

shaders=(
  composite.comp
  resample.comp
  window_level.comp
)

ok=0
fail=0

for shader in "${shaders[@]}"; do
  src="${SHADER_DIR}/${shader}"
  spv="${OUT_DIR}/${shader%.comp}.spv"

  if [[ ! -f "${src}" ]]; then
    echo "warning: shader not found: ${src}" >&2
    (( fail++ )) || true
    continue
  fi

  echo "  glslc  ${shader}  ->  spv/${shader%.comp}.spv"
  if glslc -o "${spv}" "${src}"; then
    (( ok++ )) || true
  else
    echo "error: compilation failed for ${shader}" >&2
    (( fail++ )) || true
  fi
done

echo ""
echo "compiled: ${ok}  failed: ${fail}"
[[ ${fail} -eq 0 ]]
