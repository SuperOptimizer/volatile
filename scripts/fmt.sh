#!/bin/bash
set -euo pipefail
# Check formatting (--dry-run returns non-zero if changes needed)
find src test -name '*.c' -o -name '*.h' | xargs clang-format --dry-run --Werror 2>&1
