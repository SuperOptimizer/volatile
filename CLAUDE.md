# Volatile

Rewrite of ScrollPrize/villa in pure C23 + Python. Minimal dependencies, maximum performance.

## Build

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j4
ctest --test-dir build --timeout 10
```

## Build with sanitizers

```bash
cmake -B build-asan -G Ninja -DVOLATILE_ENABLE_ASAN=ON -DVOLATILE_ENABLE_UBSAN=ON
cmake --build build-asan -j4
ctest --test-dir build-asan --timeout 30
```

## Code Style

- **C**: 2-space indent, 120 char lines, snake_case everything, .c/.h split always
- **Python**: 2-space indent, 150 char lines, snake_case funcs/vars, PascalCase classes
- No classes in C. Structs + functions, explicit state passing.
- No global mutable state. Pass context explicitly.
- Liberal assertions (ASSERT for debug, REQUIRE for always-on).
- `// NOTE:` comments for non-obvious code. No redundant comments.
- Max ~500 lines per C file, ~1000 lines per Python file.
- Every module gets tests. Write tests FIRST (TDD).
- Errors are exceptional: assert+abort for bugs, return codes for expected failures.

## Agent Rules

- **Opus**: main process and team leads only
- **Sonnet**: code writing agents (use worktrees)
- **Haiku**: exploration, reading, research only
- Build ONLY your target: `cmake --build build --target <name> -j4`
- Test ONLY your module: `ctest --test-dir build -R test_<module> --timeout 10`
- Use `-j4` max. Leave cores for other agents.
- Build timeout: 30s. Test timeout: 10s. Lint: 15s.
- Don't recompile/retest unless you changed something.

## Dependencies

C: libcurl, blosc2, SQLite3, SDL3, Nuklear, Vulkan 1.3
Python: tinygrad (ML only), standard library
Test: greatest.h

## Directory Structure

- `src/core/` - C core library (volume, compression, cache, math, I/O)
- `src/gpu/` - Vulkan compute layer
- `src/render/` - Rendering engine
- `src/gui/` - SDL3 + Nuklear GUI
- `src/server/` - Multi-user server
- `src/cli/` - Unified CLI
- `py/volatile/` - Python package
- `test/c/` - C tests (greatest.h)
- `test/py/` - Python tests (pytest)
