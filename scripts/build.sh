#!/bin/bash
set -euo pipefail
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER_LAUNCHER=ccache 2>&1
cmake --build build -j4 2>&1
