#!/bin/bash
set -euo pipefail
ctest --test-dir build --timeout 10 --output-on-failure 2>&1
