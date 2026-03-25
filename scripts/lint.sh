#!/bin/bash
set -euo pipefail
# clang-tidy on all source files
find src -name '*.c' | xargs clang-tidy -p build --quiet 2>&1
